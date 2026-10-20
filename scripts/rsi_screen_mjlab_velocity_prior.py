"""Eight worlds x eight GPU controls, qualified prior codec screen only."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("reference-root", "export-root", "prior-root", "policy", "config", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    for n in (
        "expected-head",
        "export-unit",
        "export-hash",
        "prior-unit",
        "prior-hash",
        "config-hash",
    ):
        p.add_argument("--" + n, required=True)
    p.add_argument("--reference-pilot", action="store_true")
    p.add_argument("--screen-root", type=Path)
    p.add_argument("--screen-unit")
    p.add_argument("--screen-hash")
    p.add_argument("--learner-root", type=Path)
    p.add_argument("--learner-unit")
    p.add_argument("--learner-hash")
    p.add_argument("--position-gain", type=float, choices=(0.0, 1.0), default=0.0)
    p.add_argument("--command-point", choices=("torso", "pelvis"), default="torso")
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_collect_shadow_motor import wait_for_terminal_collection
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("unused output and exact clean frozen source required")
    pins = {}
    receipts = []
    for root, unit, expected in (
        (a.export_root, a.export_unit, a.export_hash),
        (a.prior_root, a.prior_unit, a.prior_hash),
    ):
        wait_for_terminal_collection(unit)
        receipt = root / "complete.json"
        if digest(receipt) != expected:
            raise ValueError("actual immutable terminal prerequisite required")
        data = json.loads(receipt.read_text())
        pins.update(data["input_file_hashes"])
        pins[str(receipt)] = expected
        receipts.append(data)
    if (
        receipts[0]["status"] != "MJLAB_NATIVE_EXPORT_ONE_CONTROL_DIAGNOSTIC_NOT_LONG_RUN_QUALIFIED"
        or receipts[1]["status"]
        != "TORCH_VELOCITY_PRIOR_INITIALIZATION_QUALIFIED_NOT_LEARNED_SKILL"
    ):
        raise ValueError("original body and actual prior initialization qualification required")
    checkpoint = a.prior_root / "prior-actor-initialization.pt"
    if a.reference_pilot:
        if not all((a.screen_root, a.screen_unit, a.screen_hash)):
            raise ValueError("reference pilot requires actual completed codec screen")
        wait_for_terminal_collection(a.screen_unit)
        screen_path = a.screen_root / "complete.json"
        if digest(screen_path) != a.screen_hash:
            raise ValueError("actual immutable codec screen required")
        screen = json.loads(screen_path.read_text())
        if screen["status"] != "MJLAB_VELOCITY_PRIOR_CODEC_SCREEN_NOT_RETENTION_OR_FOOTBALL":
            raise ValueError("original codec screen must pass before a full reference pilot")
        pins[str(screen_path)] = a.screen_hash
    controls = 300 if a.reference_pilot else 8
    if a.position_gain and not a.reference_pilot:
        raise ValueError("position tracking requires the explicit reference-pilot protocol")
    command_mode = (
        ("reference_position_velocity" if a.position_gain else "reference_velocity")
        if a.reference_pilot
        else "zero"
    )
    pins[str(checkpoint)] = receipts[1]["checkpoint_hash"]
    pins[str(a.config)] = a.config_hash
    pins[str(a.export_root / "compiled_model.mjb")] = receipts[0]["model_hash"]
    learner = None
    if a.learner_root is not None:
        if not a.reference_pilot or not all((a.learner_unit, a.learner_hash)):
            raise ValueError(
                "learned TRAIN collector requires explicit reference pilot and receipt"
            )
        wait_for_terminal_collection(a.learner_unit)
        learner_path = a.learner_root / "complete.json"
        if digest(learner_path) != a.learner_hash:
            raise ValueError("actual learned collector policy commitment changed")
        learner = json.loads(learner_path.read_text())
        if learner["status"] not in (
            "MJLAB_STABLE_PRIOR_REHEARSAL_PPO_SMOKE_NOT_SKILL_QUALIFIED",
            "MJLAB_BOUNDED_REFERENCE_IMITATION_NOT_SKILL_QUALIFIED",
        ):
            raise ValueError("collector requires the declared stable prior or imitation lineage")
        pins[str(learner_path)] = a.learner_hash
        from rosclaw_soccer.rsi.mjlab_motion_curriculum import motion_checkpoint_filename

        checkpoint = a.learner_root / motion_checkpoint_filename(learner)
        pins[str(checkpoint)] = learner["checkpoint_hash"]
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts", a.reference_root / "src"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            worlds=8,
            controls=controls,
            maximum_gpu_world_substeps=controls * 80,
            maximum_native_codec_checks=controls * 8,
            new_optimizer_steps=0,
            command_mode=command_mode,
            not_whole_motion_or_retention_exam=True,
            learned_train_collection=learner is not None,
            teacher_action_labels_physically_proven=False,
            velocity_command_position_feedback_gain=a.position_gain,
            velocity_command_anatomical_point=a.command_point,
            hardware_authorized=False,
        ),
    )
    env = None
    neural_rows = 0
    rows = []
    try:
        import mujoco
        import numpy as np
        import torch

        from rosclaw_soccer.providers.g1.velocity_policy import (
            VelocityPolicyParameters,
            observation98,
        )
        from rosclaw_soccer.rsi.mjlab_motion_evaluation import MotionFirstEpisode
        from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
        from rosclaw_soccer.rsi.velocity_prior_actor import VelocityPriorActor, phase_table

        if torch.cuda.device_count() != 1:
            raise ValueError("one explicitly visible GPU required")
        sys.path.insert(0, str(a.reference_root))
        from scripts.rsi_mjlab_motion_backend import ExplicitBoundaryMotionEnv
        from scripts.rsi_mjlab_velocity_prior_backend import make_velocity_prior_training_cfg

        params = VelocityPolicyParameters.load(a.config, expected_hash=pins[str(a.config)])
        cfg = make_velocity_prior_training_cfg(
            parameters=params,
            command_mode=command_mode,
            command_point=a.command_point,
            motion_dir=a.reference_root / "src/assets/soccer/motions/shooter",
            num_envs=8,
            seed=1601,
            balanced_initial_sources=True,
        )
        cfg.commands["motion"].sampling_mode = "uniform"
        cfg.sim.nan_guard.enabled = True
        env = ExplicitBoundaryMotionEnv(cfg=cfg, device="cuda:0", render_mode=None)
        obs, _ = env.reset(seed=1601)
        model = env.sim.mj_model
        layout = NativeVelocityLayout(model, robot_prefix="robot/", ball_body_name="ball/ball")
        original_model = mujoco.MjModel.from_binary_path(str(a.export_root / "compiled_model.mjb"))
        original_layout = NativeVelocityLayout(
            original_model, robot_prefix="robot/", ball_body_name="ball/ball"
        )
        if layout.profile() != original_layout.profile():
            raise ValueError("new observation/action codec must not change native body dynamics")
        actor = VelocityPriorActor.from_onnx(a.policy, expected_hash=pins[str(a.policy)]).to(
            "cuda:0"
        )
        loaded = torch.load(checkpoint, map_location="cuda:0", weights_only=True)
        state = (
            loaded["actor_state_dict"]
            if learner is None
            else {
                k: loaded["algorithm"]["actor_state_dict"]["mlp." + k] for k in actor.state_dict()
            }
        )
        actor.load_state_dict(state, strict=True)
        actor.eval()
        immutable_state = {k: v.detach().clone() for k, v in actor.state_dict().items()}
        term = env.action_manager.get_term("joint_pos")
        motion = env.command_manager.get_term("motion")
        from rosclaw_soccer.rsi.mjlab_motion_curriculum import TRAIN_MOTION_NAMES

        if tuple(name + ".npz" for name in motion.motion.motion_names) != TRAIN_MOTION_NAMES:
            raise ValueError("full fixed TRAIN sources only; DEV cannot enter imitation data")
        rows = []
        max_error = 0.0
        phases = phase_table(controls)
        score = MotionFirstEpisode(8)
        previous_actions = np.zeros((8, 29), dtype=np.float32)
        for frame in range(controls):
            qpos = env.sim.data.qpos.cpu().numpy().copy()
            qvel = env.sim.data.qvel.cpu().numpy().copy()
            prefix = obs["actor"][:, :98].cpu().numpy().copy()
            full_observation = obs["actor"].cpu().numpy().copy()
            reference_joint = motion.joint_pos.cpu().numpy().copy()
            reference_joint_velocity = motion.joint_vel.cpu().numpy().copy()
            absolute_reference = dict(
                position=motion.anchor_pos_w.cpu().numpy().copy(),
                quaternion=motion.anchor_quat_w.cpu().numpy().copy(),
                linear=motion.anchor_lin_vel_w.cpu().numpy().copy(),
                angular=motion.anchor_ang_vel_w.cpu().numpy().copy(),
            )
            pelvis_index = motion.cfg.body_names.index("pelvis")
            root_reference = dict(
                position=motion.body_pos_w[:, pelvis_index].cpu().numpy().copy(),
                linear=motion.body_lin_vel_w[:, pelvis_index].cpu().numpy().copy(),
                angular=motion.body_ang_vel_w[:, pelvis_index].cpu().numpy().copy(),
            )
            source_id = motion.motion_idx.cpu().numpy().copy()
            source_phase = motion.time_steps.cpu().numpy().copy()
            active_before = score.active.copy()
            cached_gyro_before = (
                env.command_manager.get_term("motion")
                .robot.data.root_link_ang_vel_b.cpu()
                .numpy()
                .copy()
            )
            episode_frames = env.episode_length_buf.cpu().numpy().copy()
            if tuple(obs["actor"].shape) != (8, 168) or (
                not a.reference_pilot and np.any(episode_frames != frame)
            ):
                raise ValueError(
                    "actual eight-world semantic observation and no early reset required"
                )
            for world in range(8):
                data = mujoco.MjData(model)
                data.qpos[:] = qpos[world]
                data.qvel[:] = qvel[world]
                mujoco.mj_forward(model, data)
                q, v = layout.canonical(data)
                # observation98 advances one tick; use the primed phase for
                # frame0 and the preceding lookup phase for subsequent frames.
                from rosclaw_soccer.providers.g1.velocity_policy import phase_delta

                episode_frame = int(episode_frames[world])
                before_phase = phase_delta() if episode_frame == 0 else phases[episode_frame - 1]
                expected, _ = observation98(
                    q,
                    v,
                    layout.angular_velocity_body(data),
                    prefix[world, 6:9].copy(),
                    previous_actions[world],
                    before_phase,
                    params,
                )
                error = float(np.abs(prefix[world] - expected).max())
                max_error = max(max_error, error)
                if error > 1e-5:
                    np.savez_compressed(
                        a.output / "failed-codec-state.npz",
                        qpos=qpos,
                        qvel=qvel,
                        world=world,
                        frame=frame,
                        expected=expected,
                        actual=prefix[world],
                        cached_gyro=env.command_manager.get_term("motion")
                        .robot.data.root_link_ang_vel_b.cpu()
                        .numpy()
                        .copy(),
                    )
                    raise ValueError(
                        "actual GPU observation prefix differs from qualified native codec"
                    )
                if a.reference_pilot:
                    from rosclaw_soccer.rsi.native_motion_reference import (
                        point_velocity_command,
                        reference_inputs70,
                    )

                    torso = model.body("robot/torso_link").id
                    expected_command, expected_reference = reference_inputs70(
                        root_rotation=data.xmat[layout.pelvis].reshape(3, 3),
                        anchor_position=data.xpos[torso],
                        anchor_rotation=data.xmat[torso].reshape(3, 3),
                        joint_position=reference_joint[world],
                        joint_velocity=reference_joint_velocity[world],
                        reference_position=absolute_reference["position"][world],
                        reference_quaternion=absolute_reference["quaternion"][world],
                        reference_linear_velocity=absolute_reference["linear"][world],
                        reference_angular_velocity=absolute_reference["angular"][world],
                        position_gain=a.position_gain,
                    )
                    if a.command_point == "pelvis":
                        expected_command = point_velocity_command(
                            root_rotation=data.xmat[layout.pelvis].reshape(3, 3),
                            current_position=data.xpos[layout.pelvis],
                            reference_position=root_reference["position"][world],
                            reference_linear_velocity=root_reference["linear"][world],
                            reference_angular_velocity=root_reference["angular"][world],
                            position_gain=a.position_gain,
                        )
                    if not np.allclose(
                        expected_command, prefix[world, 6:9], rtol=0, atol=1e-5
                    ) or not np.allclose(
                        expected_reference, full_observation[world, 98:], rtol=0, atol=1e-5
                    ):
                        raise ValueError(
                            "actual full reference context differs from independent native geometry"
                        )
            with torch.no_grad():
                action = actor(obs["actor"])
            neural_rows += 8
            previous_actions = action.cpu().numpy().copy()
            before = env._sim_step_counter
            obs, reward, terminated, truncated, _ = env.step(action)
            torch.cuda.synchronize()
            target = term._processed_actions.cpu().numpy().copy()
            expected_target = previous_actions * np.asarray(
                params.action_scale, dtype=np.float32
            ) + np.asarray(params.action_offset, dtype=np.float32)
            if (
                not np.allclose(target, expected_target, rtol=0, atol=1e-7)
                or env._sim_step_counter - before != 10
            ):
                raise ValueError("actual GPU target decoder or physical step count differs")
            if (not a.reference_pilot and torch.any(terminated | truncated)) or not torch.isfinite(
                reward
            ).all():
                raise ValueError("early physical termination in eight-control codec screen")
            done = (terminated | truncated).cpu().numpy().copy()
            exhausted = env.termination_manager.get_term("reference_exhausted").cpu().numpy().copy()
            score.observe(
                reward.cpu().numpy().copy(),
                terminated.cpu().numpy().copy(),
                truncated.cpu().numpy().copy(),
                exhausted,
            )
            rows.append(
                dict(
                    qpos=qpos,
                    qvel=qvel,
                    observation=prefix,
                    action=previous_actions.copy(),
                    target=target,
                    terminated=terminated.cpu().numpy().copy(),
                    truncated=truncated.cpu().numpy().copy(),
                    reference_exhausted=exhausted,
                    episode_frame=episode_frames,
                    cached_gyro=cached_gyro_before,
                    full_observation=full_observation,
                    reference_joint=reference_joint,
                    reference_joint_velocity=reference_joint_velocity,
                    measured_joint=qpos[:, layout.qi].copy(),
                    source_id=source_id,
                    source_phase=source_phase,
                    active_before=active_before,
                    reward=reward.cpu().numpy().copy(),
                    actual_actuator_force=env.sim.data.actuator_force.cpu().numpy().copy(),
                    reference_anchor=absolute_reference["position"],
                    reference_anchor_quat=absolute_reference["quaternion"],
                    reference_anchor_linear_velocity=absolute_reference["linear"],
                    reference_anchor_angular_velocity=absolute_reference["angular"],
                    reference_pelvis_position=root_reference["position"],
                    reference_pelvis_linear_velocity=root_reference["linear"],
                    reference_pelvis_angular_velocity=root_reference["angular"],
                )
            )
            previous_actions[done] = 0
            if a.reference_pilot and not score.active.any():
                break
        np.savez_compressed(
            a.output / "screen8.npz", **{k: np.asarray([r[k] for r in rows]) for k in rows[0]}
        )
        check_pins(pins)
        if any(not torch.equal(v, actor.state_dict()[k]) for k, v in immutable_state.items()):
            raise ValueError("data collection cannot update its actor")
        write_once(
            a.output / "complete.json",
            dict(
                status=(
                    "MJLAB_LEARNED_PRIOR_TRAIN_COLLECTION_NOT_TEACHER_QUALIFIED"
                    if learner is not None
                    else "MJLAB_VELOCITY_PRIOR_REFERENCE_PILOT_NOT_LEARNING_OR_FOOTBALL"
                )
                if a.reference_pilot
                else "MJLAB_VELOCITY_PRIOR_CODEC_SCREEN_NOT_RETENTION_OR_FOOTBALL",
                source_head=a.expected_head,
                input_file_hashes=pins,
                gpu_world_substeps=env._sim_step_counter * 8,
                actual_neural_rows=neural_rows,
                max_native_gpu_observation_error=max_error,
                whole_motion_qualified=False,
                first_episode_summary=score.summary(),
                command_mode=command_mode,
                new_optimizer_steps=0,
                output_hash=digest(a.output / "screen8.npz"),
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
                training_sources=list(TRAIN_MOTION_NAMES),
                learned_train_collection=learner is not None,
                teacher_action_labels_physically_proven=False,
                velocity_command_position_feedback_gain=a.position_gain,
                velocity_command_anatomical_point=a.command_point,
            ),
        )
    except Exception as error:
        if rows:
            import numpy as np

            np.savez_compressed(
                a.output / "failed-partial.npz",
                **{k: np.asarray([r[k] for r in rows]) for k in rows[0]},
            )
        write_once(
            a.output / "failure.json",
            dict(
                error=repr(error),
                actual_gpu_step_calls=env._sim_step_counter if env is not None else 0,
                actual_neural_rows=neural_rows,
                automatic_retry_authorized=False,
            ),
        )
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
