"""177D NN migration, measured perception and actual contact-sensor SIM qualification."""

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("parent-root", "export-root", "reference-root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("parent-unit", "parent-hash", "export-unit", "export-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    from rosclaw_soccer.rsi.mjlab_motion_curriculum import require_completed_stable_motion_course
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
        raise ValueError("unused evidence root and exact frozen source required")
    pins = {}
    receipts = []
    for root, unit, expected in (
        (a.parent_root, a.parent_unit, a.parent_hash),
        (a.export_root, a.export_unit, a.export_hash),
    ):
        wait_for_terminal_collection(unit)
        path = root / "complete.json"
        if digest(path) != expected:
            raise ValueError("actual immutable terminal prerequisite required")
        d = json.loads(path.read_text())
        receipts.append(d)
        pins.update(d["input_file_hashes"])
        pins[str(path)] = expected
    parent, export = receipts
    require_completed_stable_motion_course(parent)
    checkpoint = a.parent_root / f"learner-update{parent['iterations']}.pt"
    pins[str(checkpoint)] = parent["checkpoint_hash"]
    pins[str(a.export_root / "compiled_model.mjb")] = export["model_hash"]
    for root in (Path.cwd() / "src", Path.cwd() / "scripts", a.reference_root / "src"):
        pins.update({str(f): digest(f) for f in root.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            actor_abi="98+70+9=177",
            critic_abi="295+9=304",
            zero_influence_new_columns=True,
            fresh_optimizer_required=True,
            optimizer_state_migrated=False,
            fixture_robot_raised_m=0.5,
            fixture_capsule_end_penetration_m=0.02,
            fixture_worlds=8,
            fixture_substeps=10,
            fixture_not_policy_success=True,
            sensor_force_atol_n=2.0,
            sensor_force_rtol=0.05,
            sensor_native_raw_atol_n=1e-5,
            native_qpos_atol_mixed_units=0.001,
            native_qvel_atol_mixed_units=0.02,
            maximum_episode_gpu_world_steps=24000,
            maximum_episode_nn_rows_per_actor=2400,
            maximum_perception_native_fk_calls=2400,
            perception_max_error=1e-5,
            maximum_actor_migration_error=1e-5,
            seed=1601,
            football_rewards_enabled=True,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    env = None
    steps = fk_calls = neural_rows = 0
    calibration = None
    try:
        import mujoco
        import numpy as np
        import torch

        from rosclaw_soccer.providers.g1.velocity_policy import VelocityPolicyParameters
        from rosclaw_soccer.rsi.ball_aware_inputs import BALL_AWARE_ABI, ball_goal9
        from rosclaw_soccer.rsi.ball_aware_migration import expand_ball_aware_state
        from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
        from rosclaw_soccer.rsi.velocity_prior_actor import VelocityPriorActor

        sys.path.insert(0, str(a.reference_root))
        from scripts.rsi_ball_aware_sensor_calibration import calibrate
        from scripts.rsi_mjlab_ball_aware_backend import (
            BallAwareMotionEnv,
            make_ball_aware_training_cfg,
        )

        torch.set_num_threads(1)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("one isolated GPU required")
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
        original = loaded["prior_configuration"]["actor"]
        onnx = Path(original["prior_model"])
        config = next(Path(k) for k in pins if k.endswith("/policy/velocity/v0/params/deploy.yaml"))
        params = VelocityPolicyParameters.load(config, expected_hash=pins[str(config)])
        actor = VelocityPriorActor.from_onnx(
            onnx, expected_hash=pins[str(onnx)], reference_features=79
        )
        state = expand_ball_aware_state(loaded["algorithm"]["actor_state_dict"], part="actor")
        actor.load_state_dict({k: state["mlp." + k] for k in actor.state_dict()}, strict=True)
        critic_state = expand_ball_aware_state(
            loaded["algorithm"]["critic_state_dict"], part="critic"
        )
        old = VelocityPriorActor.from_onnx(onnx, expected_hash=pins[str(onnx)])
        old.load_state_dict(
            {k: loaded["algorithm"]["actor_state_dict"]["mlp." + k] for k in old.state_dict()}
        )
        actor.cuda().eval()
        old.cuda().eval()
        cfg = make_ball_aware_training_cfg(
            parameters=params,
            command_mode="reference_position_velocity",
            command_point="pelvis",
            motion_dir=a.reference_root / "src/assets/soccer/motions/shooter",
            num_envs=8,
            seed=1601,
            balanced_initial_sources=True,
            football_rewards=True,
        )
        cfg.commands["motion"].sampling_mode = "uniform"
        cfg.sim.nan_guard.enabled = True
        env = BallAwareMotionEnv(cfg=cfg, device="cuda:0", render_mode=None)
        obs, _ = env.reset(seed=1601)
        if obs["actor"].shape != (8, 177) or obs["critic"].shape != (8, 304):
            raise ValueError("new explicit actor177/critic304 observation ABI differs")
        model = env.sim.mj_model
        layout = NativeVelocityLayout(model, robot_prefix="robot/", ball_body_name="ball/ball")
        old_model = mujoco.MjModel.from_binary_path(str(a.export_root / "compiled_model.mjb"))
        old_layout = NativeVelocityLayout(
            old_model, robot_prefix="robot/", ball_body_name="ball/ball"
        )
        if layout.profile() != old_layout.profile():
            raise ValueError("body/motor/collider profile changed during sensor-only addition")
        for side in ("left", "right"):
            sensor = model.sensor("ball_" + side + "_foot_ball_force")
            if (
                sensor.objtype[0] != mujoco.mjtObj.mjOBJ_BODY
                or sensor.objid[0] != layout.ball
                or sensor.reftype[0] != mujoco.mjtObj.mjOBJ_BODY
                or sensor.refid[0] != model.body("robot/" + side + "_ankle_roll_link").id
            ):
                raise ValueError("compiled sensor not bound to declared ball and exact foot body")
        mujoco.mj_saveModel(model, str(a.output / "compiled_model.mjb"), None)
        calibration, passed = calibrate(env, layout, a.output)
        write_once(a.output / "sensor-calibration.json", calibration)
        if not passed:
            raise ValueError("actual per-foot sensor/native collision calibration failed")
        obs, _ = env.reset(seed=1601)
        motion = env.command_manager.get_term("motion")
        active = np.ones(8, dtype=bool)
        completed = np.zeros(8, dtype=bool)
        failed = np.zeros(8, dtype=bool)
        rows = []
        perception_error = migration_error = 0.0
        scratch = mujoco.MjData(model)
        for _frame in range(300):
            q = env.sim.data.qpos.cpu().numpy().copy()
            v = env.sim.data.qvel.cpu().numpy().copy()
            full = obs["actor"].detach().cpu().numpy().copy()
            targets = (motion.target_destination_pos + env.scene.env_origins).cpu().numpy()
            for world in range(8):
                scratch.qpos[:] = q[world]
                scratch.qvel[:] = v[world]
                mujoco.mj_kinematics(model, scratch)
                fk_calls += 1
                expected = ball_goal9(
                    pelvis_rotation=scratch.xmat[layout.pelvis].reshape(3, 3),
                    pelvis_position=scratch.xpos[layout.pelvis],
                    pelvis_linear_velocity=v[world, layout.rv : layout.rv + 3],
                    ball_position=scratch.xpos[layout.ball],
                    ball_linear_velocity=v[world, layout.bv : layout.bv + 3],
                    target_position=targets[world],
                )
                perception_error = max(
                    perception_error, float(abs(full[world, 168:] - expected).max())
                )
            with torch.no_grad():
                action = actor(obs["actor"])
                old_action = old(obs["actor"][:, :168])
                neural_rows += 16
                migration_error = max(migration_error, float((action - old_action).abs().max()))
            if perception_error > 1e-5 or migration_error > 1e-5:
                raise ValueError("independent perception ABI or zero-influence migration differs")
            before_active = active.copy()
            phase = motion.time_steps.cpu().numpy().copy()
            source_ids = motion.motion_idx.cpu().numpy().copy()
            exhaustion = (motion.time_steps + 1 >= motion.motion_length - 1).cpu().numpy()
            obs, reward, terminated, truncated, _ = env.step(action)
            steps += 80
            term, trunc = terminated.cpu().numpy(), truncated.cpu().numpy()
            physical = env.physical_step_data()
            if not np.array_equal(physical["terminal_capture_mask"].cpu().numpy(), term | trunc):
                raise ValueError(
                    "all and only terminal physical worlds must be preserved before reset"
                )
            completed |= before_active & trunc & exhaustion & ~term
            failed |= before_active & term
            active &= ~(term | trunc)
            row = dict(
                qpos_before=q,
                qvel_before=v,
                observation=full,
                action=action.cpu().numpy(),
                active_before=before_active,
                source_id=source_ids,
                source_phase=phase,
                reward=reward.cpu().numpy(),
                terminated=term,
                truncated=trunc,
                **{key: value.cpu().numpy().copy() for key, value in physical.items()},
            )
            if hasattr(env, "_identity_kick_credit"):
                row.update(
                    {
                        "credit_" + k: value.cpu().numpy().copy()
                        for k, value in env._identity_kick_credit.cached.items()
                    }
                )
            rows.append(row)
            if not active.any():
                break
        np.savez_compressed(
            a.output / "neural-episodes.npz",
            **{k: np.asarray([r[k] for r in rows]) for k in rows[0]},
        )
        migrated = dict(
            algorithm=dict(actor_state_dict=state, critic_state_dict=critic_state),
            initial_actor_state_dict=state,
            initial_critic_state_dict=critic_state,
            ball_aware_abi=BALL_AWARE_ABI,
            actor_observation_dim=177,
            critic_observation_dim=304,
            stable_prior_rehearsal=True,
            parent_checkpoint_hash=parent["checkpoint_hash"],
            optimizer_state_migrated=False,
            optimizer_steps=0,
            fresh_optimizer_required=True,
            source_head=a.expected_head,
            velocity_command_anatomical_point="pelvis",
            velocity_command_position_feedback_gain=1.0,
            prior_configuration=loaded["prior_configuration"],
        )
        path = a.output / "ball-aware-initialization.pt"
        configuration = copy.deepcopy(loaded["prior_configuration"])
        configuration["actor"].pop("prior_checkpoint")
        configuration["actor"].pop("prior_checkpoint_hash")
        for part, cls in (
            ("actor", "BallAwarePriorMLPModel"),
            ("critic", "BallAwareCriticMLPModel"),
        ):
            configuration[part].update(
                class_name="scripts.rsi_mjlab_ball_aware_rl:" + cls,
                parent_checkpoint=str(checkpoint),
                parent_checkpoint_hash=parent["checkpoint_hash"],
            )
        configuration["algorithm"]["rehearsal_reference_features"] = 79
        migrated["prior_configuration"] = configuration
        torch.save(migrated, path)
        reloaded = torch.load(path, map_location="cpu", weights_only=True)
        if any(
            not torch.equal(v, reloaded["algorithm"]["actor_state_dict"][k])
            for k, v in state.items()
        ):
            raise ValueError("safe migration checkpoint round-trip differs")
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="BALL_AWARE_ABI_SENSOR_QUALIFICATION_NOT_LEARNED_SKILL",
                source_head=a.expected_head,
                input_file_hashes=pins,
                ball_aware_abi=BALL_AWARE_ABI,
                actor_observation_dim=177,
                critic_observation_dim=304,
                checkpoint_file=path.name,
                checkpoint_hash=digest(path),
                model_hash=digest(a.output / "compiled_model.mjb"),
                perception_max_error=perception_error,
                actor_migration_max_error=migration_error,
                full_reference_episodes_completed=completed.tolist(),
                failed=failed.tolist(),
                right_censored=active.tolist(),
                episode_gpu_world_steps=steps,
                fixture_gpu_world_steps=80,
                fixture_native_steps=80,
                actual_neural_rows=neural_rows,
                perception_native_fk_calls=fk_calls,
                sensor_calibration=calibration,
                new_optimizer_steps=0,
                parent_checkpoint_hash=parent["checkpoint_hash"],
                weights_only_checkpoint_reload=True,
                optimizer_state_migrated=False,
                promotion_authorized=False,
                goal_complete=False,
                football_skill_proven=False,
                hardware_authorized=False,
            ),
        )
    except Exception as error:
        if env is not None:
            import numpy as np

            np.savez_compressed(
                a.output / "failed-gpu-state.npz",
                qpos=env.sim.data.qpos.cpu().numpy(),
                qvel=env.sim.data.qvel.cpu().numpy(),
            )
            native = getattr(env, "_ball_fixture_native_data", [])
            if native:
                np.savez_compressed(
                    a.output / "failed-native-fixtures.npz",
                    qpos=np.asarray([d.qpos.copy() for d in native]),
                    qvel=np.asarray([d.qvel.copy() for d in native]),
                )
        write_once(
            a.output / "failure.json",
            dict(
                error=repr(error),
                episode_gpu_world_steps=steps,
                actual_neural_rows=neural_rows,
                perception_native_fk_calls=fk_calls,
                sensor_calibration=calibration,
                fixture_counts=getattr(env, "_ball_fixture_counts", {}),
                attempted_episode_gpu_world_steps=getattr(env, "_sim_step_counter", 0) * 8,
                new_optimizer_steps=0,
                automatic_retry_authorized=False,
            ),
        )
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
