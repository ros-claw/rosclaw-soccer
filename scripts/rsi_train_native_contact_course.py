"""Fixed SIM native-contact24 PPO pilot; fresh177 critic, no automatic promotion."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("qualification-root", "reference-root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("qualification-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--actor-kind", choices=("shared", "bounded-residual"), default="shared")
    p.add_argument("--adapter-proof-root", type=Path)
    p.add_argument("--adapter-proof-hash")
    p.add_argument("--training-seed", type=int, choices=(7602, 7605, 7606))
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("frozen learner and unused output required")
    f = a.qualification_root / "complete.json"
    if digest(f) != a.qualification_hash:
        raise ValueError("exact native course qualification required")
    qual = json.loads(f.read_text())
    if (
        qual.get("status") != "NATIVE_CONTACT24_COURSE_QUALIFIED_NOT_SKILL"
        or any(
            qual.get(k) is not True
            for k in (
                "independent_repeat_exact",
                "active_reset_rejected",
                "credit_history_cleared",
                "failed_world_frozen",
            )
        )
        or qual.get("protocol") != "native-contact24-v1"
        or qual.get("critic_abi") != 177
        or len(qual.get("episodes", [])) != 2
    ):
        raise ValueError("normal episodes, source guards and actual fault/reset proof required")
    pins = {**qual["input_file_hashes"], str(f): a.qualification_hash}
    for i, h in enumerate(qual["raw_hashes"]):
        pins[str(a.qualification_root / f"episode{i}.npz")] = h
    pins[str(a.qualification_root / "fault-fixture.npz")] = qual["fixture_hash"]
    residual = a.actor_kind == "bounded-residual"
    if residual:
        if not a.adapter_proof_root or not a.adapter_proof_hash:
            raise ValueError("residual training requires its own actual zero-influence proof")
        proof_path = a.adapter_proof_root / "complete.json"
        if digest(proof_path) != a.adapter_proof_hash:
            raise ValueError("immutable residual proof required")
        adapter_proof = json.loads(proof_path.read_text())
        if (
            adapter_proof.get("status")
            != "NATIVE_BOUNDED_CONTACT_ADAPTER_ZERO_INFLUENCE_QUALIFIED_NOT_SKILL"
            or any(
                adapter_proof.get(k) is not True
                for k in (
                    "zero_influence_all_inputs",
                    "modified_finite_adapter_far_and_missing_perception_exact",
                    "full_paired_physics_exact",
                    "frozen_foundation_exact",
                )
            )
            or adapter_proof["input_file_hashes"].get(str(f)) != a.qualification_hash
        ):
            raise ValueError("same native course and actual protected adapter proof required")
        pins.update(adapter_proof["input_file_hashes"])
        pins[str(proof_path)] = a.adapter_proof_hash
        pins[str(a.adapter_proof_root / "function-test.npz")] = adapter_proof["function_test_hash"]
        for label, h in adapter_proof["paired_raw_hashes"].items():
            pins[str(a.adapter_proof_root / (label + ".npz"))] = h
        adapter_source = next(
            Path(k).parent.parent
            for k in adapter_proof["input_file_hashes"]
            if k.endswith("/scripts/rsi_qualify_native_contact_adapter.py")
        )
        relative = "src/rosclaw_soccer/rsi/native_contact_adapter.py"
        if digest(Path.cwd() / relative) != pins[str(adapter_source / relative)]:
            raise ValueError("same actually qualified adapter implementation required")
    elif a.adapter_proof_root or a.adapter_proof_hash:
        raise ValueError("adapter evidence cannot qualify a shared-actor experiment")
    if not residual and a.training_seed is not None:
        raise ValueError("shared pilot retains its original fixed seed")
    qualified_root = next(
        Path(k).parent.parent
        for k in pins
        if k.endswith("/scripts/rsi_qualify_native_contact_course.py")
    )
    for rel in (
        "src/rosclaw_soccer/rsi/native_contact_course.py",
        "src/rosclaw_soccer/rsi/native_course_tracking.py",
        "src/rosclaw_soccer/rsi/native_physics_batch.py",
        "src/rosclaw_soccer/rsi/native_reference_observation.py",
        "src/rosclaw_soccer/rsi/identity_kick_credit.py",
        "scripts/rsi_native_course_inputs.py",
    ):
        if digest(Path.cwd() / rel) != pins[str(qualified_root / rel)]:
            raise ValueError("learner must use exact actually qualified course/PD/contact source")
    for root in (Path.cwd() / "src", Path.cwd() / "scripts", a.reference_root / "src"):
        pins.update({str(f): digest(f) for f in root.rglob("*.py")})
    check_pins(pins)
    seed, actor_lr = ((a.training_seed or 7602), 3e-4) if residual else (7601, 3e-5)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            protocol="native-contact24-v1",
            seed=seed,
            actor_kind=a.actor_kind,
            frozen_foundation=residual,
            adapter_proof_hash=a.adapter_proof_hash,
            worlds=32,
            course_lead=10,
            controls=24,
            value_only_episodes=4,
            ppo_episodes=16,
            maximum_native_steps=153600,
            maximum_rollout_policy_rows=15360,
            maximum_optimizer_steps=320,
            epochs=4,
            minibatches=4,
            actor_abi=177,
            critic_abi=177,
            fresh_critic=True,
            fresh_optimizer=True,
            old_critic304_and_old_adam_loaded=False,
            actor_learning_rate=actor_lr,
            critic_learning_rate=3e-4,
            exploration_learning_rate=1e-5,
            gaussian_std_initial=0.05,
            gaussian_std_bounds=[0.01, 0.1],
            gamma=0.99,
            gae_lambda=0.95,
            ppo_clip=0.1,
            target_approximate_kl=0.02,
            baseline_anchor_kl_coefficient=0.01,
            rehearsal_mse_coefficient=1.0,
            rehearsal_batch=256,
            true_terminal_bootstrap=False,
            timeout_bootstrap_from_actual_final_observation=True,
            invalid_ended_world_samples_excluded_from_gradient_and_normalization=True,
            checkpoint_episodes=[0, 4, 8, 12, 16, 20],
            require_independent_full_native_exam_before_expansion=True,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    env = None
    counts = dict(
        optimizer_attempted=0,
        optimizer_completed=0,
        value_only_steps=0,
        ppo_steps=0,
        rollout_policy_rows=0,
        rollout_critic_rows=0,
        optimizer_actor_rows=0,
        optimizer_critic_rows=0,
        optimizer_baseline_rows=0,
        kl_skipped_minibatches=0,
    )
    history = []
    try:
        import math

        import mujoco
        import numpy as np
        import torch

        from rosclaw_soccer.providers.g1.velocity_policy import VelocityPolicyParameters
        from rosclaw_soccer.rsi.mjlab_motion_curriculum import TRAIN_MOTION_NAMES
        from rosclaw_soccer.rsi.native_contact_course import NativeContactCourse
        from rosclaw_soccer.rsi.native_contact_ppo import (
            course_gae,
            make_contact_actor_critic,
            native_ppo_loss,
        )
        from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
        from rosclaw_soccer.rsi.velocity_prior_actor import VelocityPriorActor
        from scripts.rsi_native_course_inputs import load_body_references

        torch.set_num_threads(1)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("one isolated GPU required for actual neural learning")

        def pinned(suffix):
            matches = [Path(k) for k in pins if k.endswith(suffix)]
            if len(matches) != 1:
                raise ValueError("one exact pinned input required: " + suffix)
            return matches[0]

        cfg = pinned("/policy/velocity/v0/params/deploy.yaml")
        onnx = pinned("/policy/velocity/v0/exported/policy.onnx")
        cp = pinned("/ball-aware-update200.pt")
        bp = pinned("/rsi-body-bound-precontact-curriculum-20261021/precontact-resets.npz")
        mp = pinned("/rsi-body-bound-foot-only-qualification-20261021/compiled_model.mjb")
        rp = pinned("/rsi-pelvis-online-ppo-smoke-20261021/native-rehearsal1200.npz")
        params = VelocityPolicyParameters.load(cfg, expected_hash=pins[str(cfg)])
        with np.load(bp, allow_pickle=False) as z:
            bank = {k: z[k].copy() for k in z.files}
        with np.load(rp, allow_pickle=False) as z:
            old_obs = z["observation"].copy()
        if (
            old_obs.shape != (1200, 98)
            or old_obs.dtype != np.float32
            or not np.isfinite(old_obs).all()
        ):
            raise ValueError("entire qualified old native rehearsal bank required")
        rehearsal = torch.from_numpy(
            np.concatenate((old_obs, np.zeros((1200, 79), dtype=np.float32)), axis=1)
        ).cuda()
        plans = []
        for name in TRAIN_MOTION_NAMES:
            f = pinned(
                "/rsi-train-starts-body-bound-index-check-20261021/"
                + name.removesuffix(".npz")
                + "-plan.npz"
            )
            with np.load(f, allow_pickle=False) as z:
                plans.append({k: z[k].copy() for k in z.files})
        model = mujoco.MjModel.from_binary_path(str(mp))
        layout = NativeVelocityLayout(model, robot_prefix="robot/", ball_body_name="ball/ball")
        refs = load_body_references(a.reference_root, model, layout, plans)
        env = NativeContactCourse(
            model=model,
            params=params,
            bank=bank,
            plans=plans,
            body_references=refs,
            worlds=32,
            lead=10,
        )
        actor = VelocityPriorActor.from_onnx(
            onnx, expected_hash=pins[str(onnx)], reference_features=79
        ).cuda()
        parent = torch.load(cp, map_location="cpu", weights_only=True)
        actor.load_state_dict(
            {k: parent["algorithm"]["actor_state_dict"]["mlp." + k] for k in actor.state_dict()},
            strict=True,
        )
        foundation = None
        adapter_config = None
        if residual:
            from rosclaw_soccer.rsi.native_contact_adapter import NativeResidualContactActor

            actor = NativeResidualContactActor(actor).cuda()
            foundation = {k: v.detach().cpu().clone() for k, v in actor.base.state_dict().items()}
            adapter_config = dict(
                near_radius_m=actor.near_radius_m,
                far_radius_m=actor.far_radius_m,
                maximum_normalized_residual=actor.maximum_normalized_residual,
                missing_ball9_disables_adapter=True,
                hidden_dims=[128, 64],
                input_dim=177,
                output_dim=29,
            )
        policy = make_contact_actor_critic(actor).cuda()
        optimizer = torch.optim.Adam(
            [
                dict(params=[p for p in policy.actor.parameters() if p.requires_grad], lr=actor_lr),
                dict(params=list(policy.critic.parameters()), lr=3e-4),
                dict(params=[policy.log_std], lr=1e-5),
            ]
        )
        if optimizer.state:
            raise ValueError("fresh Adam required")
        initial_actor = {k: v.detach().cpu().clone() for k, v in policy.actor.state_dict().items()}
        initial_critic = {
            k: v.detach().cpu().clone() for k, v in policy.critic.state_dict().items()
        }
        frozen_base = {k: v.detach().cpu().clone() for k, v in policy.baseline.state_dict().items()}
        initial_std = policy.log_std.detach().cpu().clone()
        checkpoints = []

        def checkpoint(episode):
            file = a.output / f"native-contact-episode{episode}.pt"
            value = dict(
                schema="soccer.rsi.native_contact24_residual_ppo.v1"
                if residual
                else "soccer.rsi.native_contact24_ppo.v1",
                actor_kind=a.actor_kind,
                adapter_config=adapter_config,
                adapter_proof_hash=a.adapter_proof_hash,
                protocol=env.protocol,
                episode_count=episode,
                value_warmup_episodes=4,
                ppo_episodes=max(0, episode - 4),
                actor_observation_dim=177,
                critic_observation_dim=177,
                native_pd_position_targets=29,
                parent_checkpoint_hash=pins[str(cp)],
                original_onnx_hash=pins[str(onnx)],
                parameter_hash=pins[str(cfg)],
                native_model_hash=pins[str(mp)],
                reset_bank_hash=pins[str(bp)],
                source_head=a.expected_head,
                training_seed=seed,
                qualification_hash=a.qualification_hash,
                actor_state_dict=policy.actor.state_dict(),
                critic_state_dict=policy.critic.state_dict(),
                policy_state_dict=policy.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                torch_rng=torch.get_rng_state(),
                cuda_rng=[v.cpu() for v in torch.cuda.get_rng_state_all()],
                counts=dict(counts),
                course_episode=env.episode,
                fresh_optimizer=True,
                activation_ceiling="SIM_ONLY",
                promotion_authorized=False,
                hardware_authorized=False,
            )
            torch.save(value, file)
            loaded = torch.load(file, map_location="cpu", weights_only=True)
            if any(
                not torch.equal(v.detach().cpu(), loaded["policy_state_dict"][k])
                for k, v in policy.state_dict().items()
            ):
                raise ValueError("checkpoint safe readback differs")
            checkpoints.append(dict(episode=episode, file=file.name, hash=digest(file)))

        checkpoint(0)
        results = []
        for episode in range(20):
            env.reset()
            history = []
            while not env.ended.all():
                obs = torch.from_numpy(env.observation).cuda()
                with torch.no_grad():
                    dist = policy.distribution(obs)
                    action = dist.sample()
                    logp = dist.log_prob(action).sum(-1)
                    value = policy.value(obs)
                counts["rollout_policy_rows"] += env.worlds
                counts["rollout_critic_rows"] += env.worlds
                result = env.step(action.cpu().numpy().copy())
                with torch.no_grad():
                    next_value = policy.value(torch.from_numpy(result["observation"]).cuda())
                counts["rollout_critic_rows"] += env.worlds
                history.append(
                    {
                        **result,
                        "action": action.cpu().numpy(),
                        "old_logp": logp.cpu().numpy(),
                        "old_value": value.cpu().numpy(),
                        "next_value": next_value.cpu().numpy(),
                    }
                )
            raw = {k: np.stack([r[k] for r in history]) for k in history[0]}
            folder = a.output / f"episode{episode:02d}"
            folder.mkdir()
            path = folder / "rollout.npz"
            np.savez_compressed(path, bank_index=env.index, **raw)
            adv, returns = course_gae(
                raw["reward"].astype(np.float32),
                raw["old_value"],
                raw["next_value"],
                raw["terminated"],
                raw["truncated"],
                raw["valid"],
            )
            mask = raw["valid"].reshape(-1)
            data = {
                k: torch.from_numpy(v.reshape((-1, *v.shape[2:]))[mask].copy()).cuda()
                for k, v in dict(
                    obs=raw["observation_before"],
                    action=raw["action"],
                    old_logp=raw["old_logp"],
                    old_value=raw["old_value"],
                    advantage=adv,
                    returns=returns,
                ).items()
            }
            data["advantage"] = (data["advantage"] - data["advantage"].mean()) / (
                data["advantage"].std(unbiased=False) + 1e-8
            )
            warmup = episode < 4
            update_rows = []
            stop = False
            for _epoch in range(4):
                if stop:
                    break
                permutation = torch.randperm(int(mask.sum()), device="cuda")
                for minibatch in torch.tensor_split(permutation, 4):
                    if not len(minibatch):
                        continue
                    old_ids = (
                        torch.arange(256, device="cuda") + counts["optimizer_completed"] * 256
                    ) % 1200
                    loss, metrics = native_ppo_loss(
                        policy,
                        data["obs"][minibatch],
                        data["action"][minibatch],
                        data["old_logp"][minibatch],
                        data["old_value"][minibatch],
                        data["advantage"][minibatch],
                        data["returns"][minibatch],
                        rehearsal[old_ids],
                        value_only=warmup,
                    )
                    counts["optimizer_critic_rows"] += len(minibatch)
                    if not warmup:
                        counts["optimizer_actor_rows"] += len(minibatch) + 256
                        counts["optimizer_baseline_rows"] += len(minibatch) + 256
                    if not warmup and float(metrics["approximate_kl"]) > 0.02:
                        counts["kl_skipped_minibatches"] += 1
                        stop = True
                        break
                    if not torch.isfinite(loss):
                        raise ValueError("nonfinite native critic/PPO loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(
                        [p for group in optimizer.param_groups for p in group["params"]], 1.0
                    )
                    if not torch.isfinite(norm):
                        raise ValueError("nonfinite clipped native PPO gradient")
                    counts["optimizer_attempted"] += 1
                    optimizer.step()
                    counts["optimizer_completed"] += 1
                    counts["value_only_steps" if warmup else "ppo_steps"] += 1
                    with torch.no_grad():
                        policy.log_std.clamp_(math.log(0.01), math.log(0.1))
                    if any(not torch.isfinite(v).all() for v in policy.state_dict().values()):
                        raise ValueError("nonfinite learned weights, no checkpoint promotion")
                    if residual and any(
                        not torch.equal(v.cpu(), foundation[k])
                        for k, v in policy.actor.base.state_dict().items()
                    ):
                        raise ValueError(
                            "immutable residual foundation changed during optimization"
                        )
                    update_rows.append({k: float(v) for k, v in metrics.items()})
            if any(
                not torch.equal(v.cpu(), frozen_base[k])
                for k, v in policy.baseline.state_dict().items()
            ):
                raise ValueError("frozen stability baseline changed")
            if warmup and (
                any(
                    not torch.equal(v.cpu(), initial_actor[k])
                    for k, v in policy.actor.state_dict().items()
                )
                or not torch.equal(policy.log_std.cpu(), initial_std)
                or any(p in optimizer.state for p in policy.actor.parameters())
                or policy.log_std in optimizer.state
            ):
                raise ValueError("critic-only warmup touched actor or exploration")
            row = dict(
                episode=episode,
                mode="value-only" if warmup else "PPO",
                valid_samples=int(mask.sum()),
                raw_hash=digest(path),
                reward_sum=float(raw["reward"].sum()),
                terminations=int(raw["terminated"].sum()),
                positive_contact_worlds=int((raw["credit_contact"] > 0).any(axis=0).sum()),
                forbidden_contact_worlds=int(
                    (raw["credit_forbidden_contact"] > 0).any(axis=0).sum()
                ),
                native_steps=env.native_steps,
                counts=dict(counts),
                minibatches=update_rows,
            )
            write_once(folder / "result.json", row)
            results.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "minibatches"}), flush=True)
            if episode + 1 in (4, 8, 12, 16, 20):
                checkpoint(episode + 1)
        actor_changed = [
            k
            for k, v in policy.actor.state_dict().items()
            if not torch.equal(v.cpu(), initial_actor[k])
        ]
        critic_changed = [
            k
            for k, v in policy.critic.state_dict().items()
            if not torch.equal(v.cpu(), initial_critic[k])
        ]
        if (
            not actor_changed
            or not critic_changed
            or counts["optimizer_attempted"] != counts["optimizer_completed"]
            or not counts["ppo_steps"]
            or not counts["value_only_steps"]
        ):
            raise ValueError("real completed actor/critic optimization required")
        if residual and any(not k.startswith("adapter.") for k in actor_changed):
            raise ValueError("residual learning changed non-adapter state")
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="NATIVE_CONTACT24_RESIDUAL_PPO_CANDIDATE_REQUIRES_FULL_EXAM"
                if residual
                else "NATIVE_CONTACT24_PPO_CANDIDATE_REQUIRES_FULL_EXAM",
                actor_kind=a.actor_kind,
                training_seed=seed,
                adapter_config=adapter_config,
                adapter_proof_hash=a.adapter_proof_hash,
                input_file_hashes=pins,
                results=results,
                checkpoints=checkpoints,
                actor_changed=actor_changed,
                critic_changed=critic_changed,
                native_steps=env.native_steps,
                counts=counts,
                actor_abi=177,
                critic_abi=177,
                protocol=env.protocol,
                full_native_exam_passed=False,
                goal_complete=False,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as exc:
        failure = dict(
            error=repr(exc),
            native_steps=env.native_steps if env else 0,
            counts=counts,
            hardware_authorized=False,
        )
        print(json.dumps(failure), flush=True)
        write_once(a.output / "failed.json", failure)
        if history:
            # Preserve completed transitions even if the current control/update
            # failed before its normal episode publication. Never overwrite.
            partial = {k: np.stack([r[k] for r in history]) for k in history[0]}
            np.savez_compressed(a.output / "failed-partial-episode.npz", **partial)
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
