"""Actual bounded ball-aware PPO, only after independent perception/sensor qualification."""

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("qualification-root", "rehearsal-root", "reference-root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in (
        "qualification-unit",
        "qualification-hash",
        "rehearsal-unit",
        "rehearsal-hash",
        "expected-head",
    ):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_collect_shadow_motor import wait_for_terminal_collection
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            != a.expected_head
        )
    ):
        raise ValueError("unused output and frozen learner source required")
    pins = {}
    receipts = []
    for root, unit, expected in (
        (a.qualification_root, a.qualification_unit, a.qualification_hash),
        (a.rehearsal_root, a.rehearsal_unit, a.rehearsal_hash),
    ):
        wait_for_terminal_collection(unit)
        path = root / "complete.json"
        if digest(path) != expected:
            raise ValueError("actual terminal learning prerequisites required")
        d = json.loads(path.read_text())
        receipts.append(d)
        pins.update(d["input_file_hashes"])
        pins[str(path)] = expected
    qual, replay = receipts
    from rosclaw_soccer.rsi.ball_aware_inputs import (
        BALL_AWARE_ABI,
        require_ball_aware_qualification,
    )
    from rosclaw_soccer.rsi.mjlab_motion_curriculum import require_completed_stable_motion_smoke

    require_completed_stable_motion_smoke(replay)
    require_ball_aware_qualification(qual)
    if (
        qual["status"] != "BALL_AWARE_ABI_SENSOR_QUALIFICATION_NOT_LEARNED_SKILL"
        or qual["ball_aware_abi"] != BALL_AWARE_ABI
        or qual["new_optimizer_steps"] != 0
        or qual["actor_observation_dim"] != 177
        or qual["critic_observation_dim"] != 304
        or qual["optimizer_state_migrated"]
        or not qual["weights_only_checkpoint_reload"]
    ):
        raise ValueError("qualified new177/304 ABI, not an old motion certificate, required")
    initialization = a.qualification_root / qual["checkpoint_file"]
    qualified_runner = next(
        Path(k)
        for k in qual["input_file_hashes"]
        if k.endswith("/scripts/rsi_qualify_ball_aware_inputs.py")
    )
    qualified_source = qualified_runner.parent.parent
    for relative in (
        "scripts/rsi_mjlab_ball_aware_backend.py",
        "scripts/rsi_mjlab_ball_aware_rl.py",
        "scripts/rsi_mjlab_prior_rl.py",
        "src/rosclaw_soccer/rsi/ball_aware_migration.py",
        "src/rosclaw_soccer/rsi/identity_kick_credit.py",
    ):
        if digest(Path.cwd() / relative) != qual["input_file_hashes"].get(
            str(qualified_source / relative)
        ):
            raise ValueError(
                "training must use qualified backend, reward and migration source bytes"
            )
    bank = a.rehearsal_root / "native-rehearsal1200.npz"
    pins[str(initialization)] = qual["checkpoint_hash"]
    if qual["input_file_hashes"].get(str(bank)) != replay["native_rehearsal"]["bank_hash"]:
        raise ValueError("same original actually recorded native rehearsal bank required")
    pins[str(bank)] = replay["native_rehearsal"]["bank_hash"]
    for root in (Path.cwd() / "src", Path.cwd() / "scripts", a.reference_root / "src"):
        pins.update({str(f): digest(f) for f in root.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            ball_aware_abi=BALL_AWARE_ABI,
            num_envs=32,
            seed=2601,
            ppo_iterations=2,
            control_steps_per_iteration=24,
            maximum_gpu_world_substeps=15360,
            optimizer_updates=42,
            actor_observation_dim=177,
            critic_observation_dim=304,
            fresh_optimizer=True,
            parent_optimizer_state_loaded=False,
            ball_and_goal_are_current_sim_ground_truth_not_visual_perception=True,
            actor_position_targets_and_native_pd_not_direct_torque=True,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    env = None
    counts = []
    steps_before = 0
    try:
        import numpy as np
        import torch

        from rosclaw_soccer.providers.g1.velocity_policy import VelocityPolicyParameters
        from rosclaw_soccer.rsi.mjlab_motion_curriculum import TRAIN_MOTION_NAMES
        from scripts.rsi_mjlab_motion_learning_smoke import changed_weight_names, cloned_tensors

        sys.path.insert(0, str(a.reference_root))
        from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

        from scripts.rsi_mjlab_ball_aware_backend import (
            BallAwareMotionEnv,
            make_ball_aware_training_cfg,
        )

        torch.set_num_threads(1)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("one isolated qualified GPU required")
        torch.manual_seed(2601)
        np.random.seed(2601)
        initial = torch.load(initialization, map_location="cpu", weights_only=True)
        if initial["ball_aware_abi"] != BALL_AWARE_ABI or not initial["fresh_optimizer_required"]:
            raise ValueError("explicit new architecture migration required")
        policy_config = next(
            Path(k) for k in pins if k.endswith("/policy/velocity/v0/params/deploy.yaml")
        )
        params = VelocityPolicyParameters.load(
            policy_config, expected_hash=pins[str(policy_config)]
        )
        cfg = make_ball_aware_training_cfg(
            parameters=params,
            command_mode="reference_position_velocity",
            command_point="pelvis",
            motion_dir=a.reference_root / "src/assets/soccer/motions/shooter",
            num_envs=32,
            seed=2601,
            balanced_initial_sources=True,
            football_rewards=True,
        )
        cfg.commands["motion"].sampling_mode = "uniform"
        cfg.sim.nan_guard.enabled = True
        env = BallAwareMotionEnv(cfg=cfg, device="cuda:0", render_mode=None)
        obs, _ = env.reset(seed=2601)
        if (
            tuple(
                name + ".npz" for name in env.command_manager.get_term("motion").motion.motion_names
            )
            != TRAIN_MOTION_NAMES
        ):
            raise ValueError("exact TRAIN source split required")
        if obs["actor"].shape != (32, 177) or obs["critic"].shape != (32, 304):
            raise ValueError("qualified new ABI dimensions required")
        settings = copy.deepcopy(initial["prior_configuration"])
        settings.update(
            logger="tensorboard",
            upload_model=False,
            save_interval=2,
            max_iterations=2,
            num_steps_per_env=24,
            seed=2601,
        )
        settings["algorithm"].update(
            rehearsal_path=str(bank),
            rehearsal_hash=pins[str(bank)],
            rehearsal_reference_features=79,
        )
        runner = MjlabOnPolicyRunner(
            RslRlVecEnvWrapper(env, clip_actions=settings.get("clip_actions")),
            settings,
            str(a.output / "rsl"),
            "cuda:0",
        )
        start = runner.alg.save()
        for part in ("actor_state_dict", "critic_state_dict"):
            if any(
                not torch.equal(v.cpu(), initial["algorithm"][part][k])
                for k, v in start[part].items()
            ):
                raise ValueError("actual fresh RSL models differ from qualified migrated networks")
        actor0, critic0 = (
            cloned_tensors(start[k]) for k in ("actor_state_dict", "critic_state_dict")
        )
        optimizers = list(
            {
                id(o): o for o in vars(runner.alg).values() if isinstance(o, torch.optim.Optimizer)
            }.values()
        )
        if not optimizers or any(len(o.state) for o in optimizers):
            raise ValueError("fresh empty optimizer state required after architecture change")
        owned = {id(p) for o in optimizers for group in o.param_groups for p in group["params"]}
        if not all(
            id(p) in owned
            for model in (runner.alg.actor, runner.alg.critic)
            for p in model.parameters()
        ):
            raise ValueError("new optimizer must own actual migrated actor and critic parameters")
        for optimizer in optimizers:
            counter = dict(attempted=0, completed=0)
            old_step = optimizer.step

            def counted(*args, _step=old_step, _counter=counter, **kwargs):
                _counter["attempted"] += 1
                result = _step(*args, **kwargs)
                _counter["completed"] += 1
                return result

            optimizer.step = counted
            counts.append(counter)
        steps_before = env._sim_step_counter
        runner.learn(num_learning_iterations=2, init_at_random_ep_len=False)
        torch.cuda.synchronize()
        state = runner.alg.save()
        actor_changes = changed_weight_names(actor0, cloned_tensors(state["actor_state_dict"]))
        critic_changes = changed_weight_names(critic0, cloned_tensors(state["critic_state_dict"]))
        new_columns = state["actor_state_dict"]["mlp.mlp.0.weight"][:, 168:]
        if (
            not actor_changes
            or not critic_changes
            or torch.count_nonzero(new_columns) == 0
            or any(c["attempted"] != c["completed"] for c in counts)
            or sum(c["completed"] for c in counts) != 42
            or env._sim_step_counter - steps_before != 480
            or runner.alg.rehearsal_updates != 2
            or runner.alg.rehearsal_rows != 512
            or not torch.equal(
                actor0["mlp.prior_mean"], state["actor_state_dict"]["mlp.prior_mean"].cpu()
            )
            or not torch.equal(
                actor0["mlp.prior_denominator"],
                state["actor_state_dict"]["mlp.prior_denominator"].cpu(),
            )
        ):
            raise ValueError(
                "actual complete actor/critic/ball9 learning and rehearsal accounting required"
            )
        path = a.output / "ball-aware-update2.pt"
        result = dict(
            algorithm=state,
            initial_actor_state_dict=actor0,
            initial_critic_state_dict=critic0,
            ball_aware_abi=BALL_AWARE_ABI,
            actor_observation_dim=177,
            critic_observation_dim=304,
            prior_configuration=settings,
            iteration=2,
            source_head=a.expected_head,
            parent_checkpoint_hash=qual["checkpoint_hash"],
            optimizer_state_migrated=False,
            stable_prior_rehearsal=True,
            velocity_command_anatomical_point="pelvis",
            velocity_command_position_feedback_gain=1.0,
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(),
        )
        torch.save(result, path)
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        for part in ("actor_state_dict", "critic_state_dict"):
            if any(
                not torch.equal(v.cpu(), loaded["algorithm"][part][k])
                for k, v in state[part].items()
            ):
                raise ValueError("safe actor/critic learned checkpoint round-trip differs")
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="BALL_AWARE_GPU_PPO_SMOKE_NOT_SKILL_QUALIFIED",
                source_head=a.expected_head,
                input_file_hashes=pins,
                ball_aware_abi=BALL_AWARE_ABI,
                actor_observation_dim=177,
                critic_observation_dim=304,
                gpu_world_substeps=(env._sim_step_counter - steps_before) * 32,
                policy_control_samples=1536,
                new_optimizer_steps=sum(c["completed"] for c in counts),
                optimizer_counts=counts,
                changed_actor_weights=actor_changes,
                changed_critic_weights=critic_changes,
                ball9_weight_norm=float(new_columns.norm()),
                native_rehearsal=state["native_rehearsal"],
                checkpoint_file=path.name,
                checkpoint_hash=digest(path),
                weights_only_checkpoint_reload=True,
                parent_optimizer_state_loaded=False,
                parent_checkpoint_hash=qual["checkpoint_hash"],
                new_native_steps=0,
                goal_complete=False,
                football_skill_proven=False,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failure.json",
            dict(
                error=repr(error),
                optimizer_counts=counts,
                attempted_gpu_world_substeps=(getattr(env, "_sim_step_counter", 0) - steps_before)
                * 32,
                new_native_steps=0,
                automatic_retry_authorized=False,
            ),
        )
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
