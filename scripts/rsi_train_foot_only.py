"""Bounded new foot-only PPO objective with fresh Adam; SIM research only."""

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--value-warmup", action="store_true")
    for name in ("qualification-root", "learner-root", "reference-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("qualification-hash", "learner-hash", "expected-head"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("new output and exact frozen source required")
    pins = {}
    receipts = []
    for root, expected in (
        (a.qualification_root, a.qualification_hash),
        (a.learner_root, a.learner_hash),
    ):
        path = root / "complete.json"
        if digest(path) != expected:
            raise ValueError("immutable prerequisite bytes required")
        d = json.loads(path.read_text())
        pins.update(d["input_file_hashes"])
        pins[str(path)] = expected
        receipts.append(d)
    qual, learner = receipts
    if qual.get("ball_placement_protocol", "legacy-subset-full-v0") != "legacy-subset-full-v0":
        raise ValueError("legacy pilot cannot consume a body-bound placement qualification")
    if (
        qual.get("status") != "FOOT_ONLY_SENSOR_REWARD_QUALIFIED_NOT_SKILL"
        or qual.get("probe_passed") is not True
        or qual.get("optimizer_updates") != 0
        or qual.get("gpu_world_steps") != 560
        or qual.get("native_steps") != 480
        or qual.get("nn_rows") != 8
        or len(qual.get("results", [])) != 6
        or {(r["profile"], r["repeat"]) for r in qual["results"]}
        != {(s, i) for s in ("foot", "knee") for i in range(3)}
        or any(r.get("passed") is not True for r in qual["results"])
        or qual["input_file_hashes"].get(str(a.learner_root / "complete.json")) != a.learner_hash
    ):
        raise ValueError(
            "same candidate with six sensor fixtures and actual NN reward proof required"
        )
    for r in qual["results"]:
        pins[str(a.qualification_root / f"{r['profile']}-{r['repeat']}" / "fixture.npz")] = r[
            "raw_hash"
        ]
    for name, key in (("neural-probe.npz", "probe_hash"), ("compiled_model.mjb", "model_hash")):
        pins[str(a.qualification_root / name)] = qual[key]
    qualified = next(
        Path(k).parent.parent
        for k in qual["input_file_hashes"]
        if k.endswith("/scripts/rsi_qualify_foot_only.py")
    )
    for rel in (
        "scripts/rsi_mjlab_foot_only_backend.py",
        "scripts/rsi_mjlab_ball_aware_backend.py",
        "scripts/rsi_mjwarp_addressed_contact_force.py",
        "scripts/rsi_mjlab_prior_rl.py",
        "scripts/rsi_mjlab_ball_aware_rl.py",
        "src/rosclaw_soccer/rsi/identity_kick_credit.py",
    ):
        if digest(Path.cwd() / rel) != qual["input_file_hashes"].get(str(qualified / rel)):
            raise ValueError("new learning must use exact qualified contact/reward/NN source")
    checkpoint = a.learner_root / learner["checkpoint_file"]
    pins[str(checkpoint)] = learner["checkpoint_hash"]
    for root in (Path.cwd() / "src", Path.cwd() / "scripts", a.reference_root / "src"):
        pins.update({str(f): digest(f) for f in root.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    worlds, iterations, seed = 32, 32 if a.value_warmup else 16, 4601
    expected_updates = 656 if a.value_warmup else 336
    write_once(
        a.output / "declaration.json",
        dict(
            input_file_hashes=pins,
            source_head=a.expected_head,
            objective="foot-only-v1",
            worlds=worlds,
            iterations=iterations,
            seed=seed,
            control_steps_per_iteration=24,
            maximum_gpu_world_steps=iterations * 24 * 32 * 10,
            maximum_control_samples=iterations * 24 * 32,
            optimizer_updates=expected_updates,
            value_warmup_iterations=16 if a.value_warmup else 0,
            old_native_rehearsal_updates=16,
            fresh_optimizer=True,
            parent_optimizer_loaded=False,
            actor_abi=177,
            critic_abi=304,
            native_pd_position_targets=29,
            evaluation_required_before_budget_expansion=True,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    env = None
    events = []
    counts = dict(attempted=0, completed=0)
    before = 0
    try:
        import numpy as np
        import torch

        from scripts.rsi_mjwarp_addressed_contact_force import install

        install()
        sys.path.insert(0, str(a.reference_root))
        from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

        from rosclaw_soccer.providers.g1.velocity_policy import VelocityPolicyParameters
        from rosclaw_soccer.rsi.mjlab_motion_curriculum import TRAIN_MOTION_NAMES
        from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
        from scripts.rsi_foot_only_calibration import body_bindings
        from scripts.rsi_mjlab_foot_only_backend import (
            FootOnlyMotionEnv,
            make_foot_only_training_cfg,
        )
        from scripts.rsi_mjlab_motion_learning_smoke import changed_weight_names, cloned_tensors

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("one isolated GPU required")
        torch.set_num_threads(1)
        torch.manual_seed(seed)
        np.random.seed(seed)
        initial = torch.load(checkpoint, map_location="cpu", weights_only=True)
        config = next(Path(k) for k in pins if k.endswith("/policy/velocity/v0/params/deploy.yaml"))
        cfg = make_foot_only_training_cfg(
            parameters=VelocityPolicyParameters.load(config, expected_hash=pins[str(config)]),
            command_mode="reference_position_velocity",
            command_point="pelvis",
            motion_dir=a.reference_root / "src/assets/soccer/motions/shooter",
            num_envs=worlds,
            seed=seed,
            balanced_initial_sources=True,
        )
        cfg.commands["motion"].sampling_mode = "uniform"
        cfg.sim.nan_guard.enabled = True
        env = FootOnlyMotionEnv(cfg=cfg, device="cuda:0", render_mode=None)
        obs, _ = env.reset(seed=seed)
        if (
            tuple(n + ".npz" for n in env.command_manager.get_term("motion").motion.motion_names)
            != TRAIN_MOTION_NAMES
        ):
            raise ValueError("TRAIN8 only; DEV007/010 forbidden in gradients")
        if obs["actor"].shape != (worlds, 177) or obs["critic"].shape != (worlds, 304):
            raise ValueError("qualified ABI differs")
        layout = NativeVelocityLayout(
            env.sim.mj_model, robot_prefix="robot/", ball_body_name="ball/ball"
        )
        body_bindings(env.sim.mj_model, layout)
        import mujoco

        old = mujoco.MjModel.from_binary_path(str(a.qualification_root / "compiled_model.mjb"))
        if (
            layout.profile()
            != NativeVelocityLayout(
                old, robot_prefix="robot/", ball_body_name="ball/ball"
            ).profile()
        ):
            raise ValueError("body/motor profile differs")
        settings = copy.deepcopy(initial["prior_configuration"])
        settings.update(
            seed=seed,
            max_iterations=iterations,
            save_interval=iterations,
            num_steps_per_env=24,
            logger="tensorboard",
            upload_model=False,
        )
        settings["actor"]["class_name"] = "scripts.rsi_mjlab_ball_aware_rl:BallAwarePriorMLPModel"
        settings["critic"]["class_name"] = "scripts.rsi_mjlab_ball_aware_rl:BallAwareCriticMLPModel"
        settings["algorithm"]["class_name"] = "scripts.rsi_mjlab_prior_rl:RehearsalPPO"
        if a.value_warmup:
            settings["algorithm"].update(
                class_name="scripts.rsi_foot_only_value_warmup:FootOnlyValueWarmupPPO",
                value_warmup_iterations=16,
            )
        # RSL consumes class_name keys in-place during construction. Historical
        # configuration dictionaries were saved after that destructive pop.
        settings["actor"]["distribution_cfg"]["class_name"] = "GaussianDistribution"
        saved_settings = copy.deepcopy(settings)
        runner = MjlabOnPolicyRunner(
            RslRlVecEnvWrapper(env, clip_actions=settings.get("clip_actions")),
            settings,
            str(a.output / "rsl"),
            "cuda:0",
        )
        # Deliberately transfer only networks, never old reward-specific Adam or counters.
        for part in ("actor", "critic"):
            getattr(runner.alg, part).load_state_dict(
                initial["algorithm"][part + "_state_dict"], strict=True
            )
        if runner.alg.optimizer.state or runner.alg.rehearsal_updates:
            raise ValueError("fresh optimizer and rehearsal counters required")
        start = runner.alg.save()
        actor0 = cloned_tensors(start["actor_state_dict"])
        critic0 = cloned_tensors(start["critic_state_dict"])
        warmup_boundary = None
        if a.value_warmup:
            original_update = runner.alg.update

            def audited_update():
                nonlocal warmup_boundary
                loss = original_update()
                if runner.alg.value_warmup_completed == 16 and warmup_boundary is None:
                    boundary = runner.alg.save()
                    if any(
                        not torch.equal(v.cpu(), actor0[k])
                        for k, v in boundary["actor_state_dict"].items()
                    ):
                        raise ValueError("warmup changed actual actor weights or normalization")
                    if not changed_weight_names(
                        critic0, cloned_tensors(boundary["critic_state_dict"])
                    ):
                        raise ValueError("warmup must actually update critic")
                    if any(p in runner.alg.optimizer.state for p in runner.alg.actor.parameters()):
                        raise ValueError("frozen actor must have no Adam momentum")
                    warmup_path = a.output / "value-warmup-boundary16.pt"
                    torch.save(boundary, warmup_path)
                    warmup_boundary = dict(
                        checkpoint_hash=digest(warmup_path),
                        actor_bitexact=True,
                        critic_changed=True,
                        optimizer_updates=counts["completed"],
                        physical_substeps=env._sim_step_counter - before,
                    )
                    write_once(a.output / "value-warmup-boundary.json", warmup_boundary)
                return loss

            runner.alg.update = audited_update
        step = runner.alg.optimizer.step

        def counted(*args, **kwargs):
            counts["attempted"] += 1
            value = step(*args, **kwargs)
            counts["completed"] += 1
            return value

        runner.alg.optimizer.step = counted
        physical_step = env.step

        def observed(action):
            source = env.command_manager.get_term("motion").motion_idx.clone()
            result = physical_step(action)
            credit = env._identity_kick_credit
            if credit.last_step != env.common_step_counter:
                raise ValueError("actual current physical contact feedback required")
            snap = env.physical_step_data()
            events.append(
                dict(
                    source=source.cpu().numpy(),
                    **{k: v.detach().cpu().numpy() for k, v in credit.cached.items()},
                    nonfoot_peak=snap["nonfoot_force_history"].amax(dim=1).cpu().numpy(),
                    terminated=result[2].cpu().numpy(),
                    truncated=result[3].cpu().numpy(),
                )
            )
            return result

        env.step = observed
        before = env._sim_step_counter
        runner.learn(num_learning_iterations=iterations, init_at_random_ep_len=False)
        torch.cuda.synchronize()
        state = runner.alg.save()
        ac = changed_weight_names(actor0, cloned_tensors(state["actor_state_dict"]))
        cc = changed_weight_names(critic0, cloned_tensors(state["critic_state_dict"]))
        if (
            not ac
            or not cc
            or counts != dict(attempted=expected_updates, completed=expected_updates)
            or len(events) != iterations * 24
            or env._sim_step_counter - before != iterations * 240
            or runner.alg.rehearsal_updates != 16
            or runner.alg.rehearsal_rows != 4096
        ):
            raise ValueError("actual complete PPO, physics and rehearsal accounting required")
        if a.value_warmup and (
            warmup_boundary is None or warmup_boundary["optimizer_updates"] != 320
        ):
            raise ValueError("actual bit-exact sixteen-iteration value-only boundary required")
        path = a.output / (
            "foot-only-warmup-update32.pt" if a.value_warmup else "foot-only-update16.pt"
        )
        torch.save(
            dict(
                algorithm=state,
                initial_actor_state_dict=actor0,
                initial_critic_state_dict=critic0,
                prior_configuration=saved_settings,
                foot_only_intent="foot-only-v1",
                actor_observation_dim=177,
                critic_observation_dim=304,
                source_head=a.expected_head,
                iteration=iterations,
                value_warmup_boundary=warmup_boundary,
                parent_checkpoint_hash=learner["checkpoint_hash"],
                fresh_optimizer=True,
                force_decoder="addressed",
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all(),
            ),
            path,
        )
        reloaded = torch.load(path, map_location="cpu", weights_only=True)
        for part in ("actor_state_dict", "critic_state_dict"):
            if any(
                not torch.equal(v.cpu(), reloaded["algorithm"][part][k])
                for k, v in state[part].items()
            ):
                raise ValueError("weights-only roundtrip failed")
        event_path = a.output / "training-events.npz"
        np.savez_compressed(event_path, **{k: np.array([r[k] for r in events]) for k in events[0]})
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="FOOT_ONLY_VALUE_WARMUP_PPO_NOT_SKILL_QUALIFIED"
                if a.value_warmup
                else "FOOT_ONLY_PPO_PILOT_NOT_SKILL_QUALIFIED",
                input_file_hashes=pins,
                source_head=a.expected_head,
                worlds=32,
                iterations=iterations,
                seed=seed,
                gpu_world_steps=iterations * 24 * 32 * 10,
                value_warmup_boundary=warmup_boundary,
                new_native_steps=0,
                optimizer_counts=counts,
                changed_actor_weights=ac,
                changed_critic_weights=cc,
                event_hash=digest(event_path),
                checkpoint_file=path.name,
                checkpoint_hash=digest(path),
                forbidden_events=int(sum(r["forbidden_contact"].sum() for r in events)),
                correct_contact_events=int(sum(r["contact"].sum() for r in events)),
                native_rehearsal=state["native_rehearsal"],
                goal_complete=False,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as exc:
        write_once(
            a.output / "failed.json",
            dict(
                error=repr(exc),
                optimizer_counts=counts,
                physical_control_rows=len(events),
                gpu_substeps=getattr(env, "_sim_step_counter", 0) - before,
                hardware_authorized=False,
            ),
        )
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
