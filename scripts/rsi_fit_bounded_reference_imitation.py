"""Fifty CPU gradient steps on TRAIN counterfactual labels plus real rehearsal."""

import argparse
import copy
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("collection-root", "learner-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in (
        "collection-unit",
        "collection-hash",
        "learner-unit",
        "learner-hash",
        "expected-head",
    ):
        p.add_argument("--" + name, required=True)
    p.add_argument("--rehearsal-root", type=Path)
    p.add_argument("--rehearsal-unit")
    p.add_argument("--rehearsal-hash")
    p.add_argument("--label-mode", choices=("all_frames", "kick_window"), default="all_frames")
    a = p.parse_args()
    from rosclaw_soccer.rsi.mjlab_motion_curriculum import (
        TRAIN_MOTION_NAMES,
        motion_checkpoint_filename,
        require_completed_stable_motion_smoke,
    )
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_collect_shadow_motor import wait_for_terminal_collection
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("unused output and exact frozen fitter source required")
    pins = {}
    receipts = []
    for root, unit, expected in (
        (a.collection_root, a.collection_unit, a.collection_hash),
        (a.learner_root, a.learner_unit, a.learner_hash),
    ):
        wait_for_terminal_collection(unit)
        path = root / "complete.json"
        if digest(path) != expected:
            raise ValueError("immutable actual completed prerequisites required")
        receipt = json.loads(path.read_text())
        pins.update(receipt["input_file_hashes"])
        pins[str(path)] = expected
        receipts.append(receipt)
    collection, learner = receipts
    if learner["status"] == "MJLAB_STABLE_PRIOR_REHEARSAL_PPO_SMOKE_NOT_SKILL_QUALIFIED":
        require_completed_stable_motion_smoke(learner)
        bank_root = a.learner_root
        bank_owner = learner
    elif learner["status"] == "MJLAB_BOUNDED_REFERENCE_IMITATION_NOT_SKILL_QUALIFIED":
        if not all((a.rehearsal_root, a.rehearsal_unit, a.rehearsal_hash)):
            raise ValueError("aggregated imitation requires the original native replay owner")
        wait_for_terminal_collection(a.rehearsal_unit)
        bank_path = a.rehearsal_root / "complete.json"
        if (
            digest(bank_path) != a.rehearsal_hash
            or learner["input_file_hashes"].get(str(bank_path)) != a.rehearsal_hash
        ):
            raise ValueError("same original rehearsal lineage required")
        bank_owner = json.loads(bank_path.read_text())
        require_completed_stable_motion_smoke(bank_owner)
        bank_root = a.rehearsal_root
        pins[str(bank_path)] = a.rehearsal_hash
    else:
        raise ValueError("explicit stable prior or bounded imitation lineage required")
    if (
        collection["status"] != "MJLAB_LEARNED_PRIOR_TRAIN_COLLECTION_NOT_TEACHER_QUALIFIED"
        or tuple(collection["training_sources"]) != TRAIN_MOTION_NAMES
        or collection["input_file_hashes"].get(str(a.learner_root / "complete.json"))
        != a.learner_hash
    ):
        raise ValueError("fixed TRAIN-only physical collection from this exact policy required")
    source_checkpoint = a.learner_root / motion_checkpoint_filename(learner)
    bank = bank_root / "native-rehearsal1200.npz"
    pins[str(source_checkpoint)] = learner["checkpoint_hash"]
    pins[str(bank)] = bank_owner["native_rehearsal"]["bank_hash"]
    pins[str(a.collection_root / "screen8.npz")] = collection["output_hash"]
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            gradient_steps=50,
            train_batch=256,
            rehearsal_batch=256,
            seed=2401,
            learning_rate=1e-4,
            alpha=0.25,
            error_cap_rad=0.6,
            maximum_label_delta_rad=0.15,
            label_mode=a.label_mode,
            kick_window_half_width_controls=15 if a.label_mode == "kick_window" else None,
            imitation_loss="joint_target_rad_squared_mean",
            rehearsal_loss="original_raw_action_squared_mean",
            rehearsal_weight=1.0,
            critic_updated=False,
            ppo_optimizer_continuation_valid=False,
            teacher_physically_proven=False,
            new_physics_steps=0,
            hardware_authorized=False,
            velocity_command_position_feedback_gain=collection.get(
                "velocity_command_position_feedback_gain", 0.0
            ),
            parent_checkpoint_hash=learner["checkpoint_hash"],
            velocity_command_anatomical_point=collection.get(
                "velocity_command_anatomical_point", "torso"
            ),
        ),
    )
    attempted = updates = neural_rows = forward_calls = 0
    try:
        import numpy as np
        import torch

        from rosclaw_soccer.providers.g1.velocity_policy import VelocityPolicyParameters
        from rosclaw_soccer.rsi.bounded_reference_imitation import (
            build_imitation_rows,
            localize_kick_window,
        )
        from rosclaw_soccer.rsi.velocity_prior_actor import VelocityPriorActor

        torch.set_num_threads(1)
        torch.manual_seed(2401)
        torch.use_deterministic_algorithms(True)
        source = torch.load(source_checkpoint, map_location="cpu", weights_only=True)
        configuration = source["prior_configuration"]["actor"]
        model_path = Path(configuration["prior_model"])
        config_path = next(
            Path(k) for k in pins if k.endswith("/policy/velocity/v0/params/deploy.yaml")
        )
        params = VelocityPolicyParameters.load(config_path, expected_hash=pins[str(config_path)])
        model = VelocityPriorActor.from_onnx(model_path, expected_hash=pins[str(model_path)])
        initial_rsl = {
            k: v.detach().cpu().clone() for k, v in source["algorithm"]["actor_state_dict"].items()
        }
        model.load_state_dict({k: initial_rsl["mlp." + k] for k in model.state_dict()}, strict=True)
        initial_model = {k: v.detach().clone() for k, v in model.state_dict().items()}
        with np.load(a.collection_root / "screen8.npz", allow_pickle=False) as z:
            raw = {k: z[k].copy() for k in z.files}
        dataset = build_imitation_rows(raw, params)
        if a.label_mode == "kick_window":
            dataset = localize_kick_window(dataset)
        np.savez_compressed(a.output / "counterfactual-labels.npz", **dataset)
        obs = torch.from_numpy(dataset["observation"])
        labels = torch.from_numpy(dataset["target_action"])
        with np.load(bank, allow_pickle=False) as z:
            old_obs = z["observation"].copy()
            old_actions = z["action"].copy()
        if (
            old_obs.shape != (1200, 98)
            or old_actions.shape != (1200, 29)
            or not np.isfinite(old_obs).all()
            or not np.isfinite(old_actions).all()
        ):
            raise ValueError("whole actual native rehearsal bank required")
        old_obs = torch.from_numpy(
            np.concatenate((old_obs, np.zeros((1200, 70), dtype=np.float32)), axis=1)
        )
        old_actions = torch.from_numpy(old_actions)
        scales = torch.tensor(params.action_scale, dtype=torch.float32)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        generator = torch.Generator().manual_seed(2401)

        def run(inputs):
            nonlocal neural_rows, forward_calls
            value = model(inputs)
            neural_rows += len(inputs)
            forward_calls += 1
            return value

        def metrics():
            with torch.no_grad():
                return dict(
                    imitation_joint_target_mse_rad2=float(
                        ((run(obs) - labels) * scales).square().mean()
                    ),
                    rehearsal_action_mse=float((run(old_obs) - old_actions).square().mean()),
                )

        before = metrics()
        losses = []
        for _ in range(50):
            ids = torch.randperm(len(obs), generator=generator)[:256]
            replay_ids = torch.randperm(1200, generator=generator)[:256]
            imitation = ((run(obs[ids]) - labels[ids]) * scales).square().mean()
            retention = (run(old_obs[replay_ids]) - old_actions[replay_ids]).square().mean()
            loss = imitation + retention
            if not torch.isfinite(loss):
                raise ValueError("finite imitation and rehearsal objectives required")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise ValueError("finite clipped actor gradients required")
            attempted += 1
            optimizer.step()
            updates += 1
            losses.append([float(imitation.detach()), float(retention.detach())])
        after = metrics()
        final_state = model.state_dict()
        if (
            not any(not torch.equal(initial_model[k], v) for k, v in final_state.items())
            or any(not torch.isfinite(v).all() for v in final_state.values())
            or not torch.equal(initial_model["prior_mean"], final_state["prior_mean"])
            or not torch.equal(initial_model["prior_denominator"], final_state["prior_denominator"])
        ):
            raise ValueError("actual finite actor updates with fixed prior normalization required")
        final_rsl = dict(initial_rsl)
        final_rsl.update({"mlp." + k: v.detach().cpu().clone() for k, v in final_state.items()})
        critic = {
            k: v.detach().cpu().clone() for k, v in source["algorithm"]["critic_state_dict"].items()
        }
        checkpoint = dict(
            algorithm=dict(actor_state_dict=final_rsl, critic_state_dict=critic),
            initial_actor_state_dict=initial_rsl,
            initial_critic_state_dict=critic,
            stable_prior_rehearsal=True,
            actor_observation_dim=168,
            prior_configuration=copy.deepcopy(source["prior_configuration"]),
            iteration=source["iteration"],
            imitation_iteration=source.get("imitation_iteration", 0) + updates,
            imitation_generation=source.get(
                "imitation_generation",
                1
                if learner["status"] == "MJLAB_BOUNDED_REFERENCE_IMITATION_NOT_SKILL_QUALIFIED"
                else 0,
            )
            + 1,
            parent_checkpoint_hash=learner["checkpoint_hash"],
            velocity_command_anatomical_point=collection.get(
                "velocity_command_anatomical_point", "torso"
            ),
            velocity_command_position_feedback_gain=collection.get(
                "velocity_command_position_feedback_gain", 0.0
            ),
            imitation_optimizer_state_dict=optimizer.state_dict(),
            imitation_label_mode=a.label_mode,
            ppo_optimizer_continuation_valid=False,
            critic_updated=False,
            torch_rng=torch.get_rng_state(),
            sampler_rng=generator.get_state(),
            source_head=a.expected_head,
        )
        path = a.output / "actor-imitation50.pt"
        with torch.no_grad():
            expected_output = run(obs).detach().clone()
        torch.save(checkpoint, path)
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        restored = VelocityPriorActor.from_onnx(model_path, expected_hash=pins[str(model_path)])
        restored.load_state_dict(
            {k: loaded["algorithm"]["actor_state_dict"]["mlp." + k] for k in restored.state_dict()},
            strict=True,
        )
        if any(
            not torch.equal(v, loaded["algorithm"]["critic_state_dict"][k])
            for k, v in critic.items()
        ):
            raise ValueError("critic must remain unchanged during actor-only imitation")
        with torch.no_grad():
            actual = restored(obs)
            neural_rows += len(obs)
            forward_calls += 1
        if not torch.equal(actual, expected_output):
            raise ValueError("fresh actor checkpoint reload differs")
        np.savez_compressed(a.output / "losses.npz", losses=np.array(losses))
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="MJLAB_BOUNDED_REFERENCE_IMITATION_NOT_SKILL_QUALIFIED",
                source_head=a.expected_head,
                input_file_hashes=pins,
                training_sources=list(TRAIN_MOTION_NAMES),
                training_rows=len(obs),
                imitation_label_mode=a.label_mode,
                counterfactual_labels_hash=digest(a.output / "counterfactual-labels.npz"),
                before=before,
                after=after,
                new_optimizer_steps=updates,
                attempted_optimizer_steps=attempted,
                training_sample_rows=updates * 256,
                rehearsal_sample_rows=updates * 256,
                actual_counted_forward_rows=neural_rows,
                actual_counted_forward_calls=forward_calls,
                critic_updated=False,
                new_ppo_steps=0,
                ppo_optimizer_continuation_valid=False,
                stable_prior_rehearsal=True,
                checkpoint_file=path.name,
                checkpoint_hash=digest(path),
                iterations=50,
                cumulative_imitation_updates=checkpoint["imitation_iteration"],
                imitation_generation=checkpoint["imitation_generation"],
                parent_checkpoint_hash=learner["checkpoint_hash"],
                velocity_command_anatomical_point=collection.get(
                    "velocity_command_anatomical_point", "torso"
                ),
                velocity_command_position_feedback_gain=collection.get(
                    "velocity_command_position_feedback_gain", 0.0
                ),
                weights_only_checkpoint_reload=True,
                new_physics_steps=0,
                teacher_physically_proven=False,
                football_skill_proven=False,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failure.json",
            dict(
                error=repr(error),
                attempted_optimizer_steps=attempted,
                completed_optimizer_steps=updates,
                actual_forward_rows=neural_rows,
                new_physics_steps=0,
                automatic_retry_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
