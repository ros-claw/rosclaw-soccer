"""Measure zero-command settling after an independently confirmed left movement."""

import argparse
import atexit
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--source-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    complete = a.source / "complete.json"
    if a.output.exists() or complete.is_symlink() or digest(complete) != a.source_hash:
        raise ValueError("unused output and exact prior live report required")
    source = json.loads(complete.read_text())
    if (
        source["status"] != "WARM_LEFT_LIVE_NN_AND_MANUAL_NATIVE_MATCH"
        or source["controls"] != 57
        or source["native_steps"] != 570
        or source["predicted_32_controls_exact"] is not True
    ):
        raise ValueError("confirmed complete live prefix required")
    pins = dict(source["input_file_hashes"])
    pins[str(complete)] = a.source_hash
    prior_path = a.source / source["artifact"]["path"]
    if prior_path.is_symlink() or prior_path.parent != a.source:
        raise ValueError("confined ordinary prior trace required")
    pins[str(prior_path)] = source["artifact"]["hash"]
    source_pins(pins)
    check_pins(pins)
    scenes = [Path(x) for x in pins if x.endswith("/cached-standard/declared-scene.xml")]
    models = [Path(x) for x in pins if x.endswith("/velocity/v0/exported/policy.onnx")]
    configs = [Path(x) for x in pins if x.endswith("/velocity/v0/params/deploy.yaml")]
    if not (len(scenes) == len(models) == len(configs) == 1):
        raise ValueError("exact scene, actor and codec required")
    import mujoco
    import numpy as np
    from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

    from rosclaw_soccer.providers.g1.continuous_velocity_stream import ContinuousVelocityStream
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.relay_audit_clock import RelayAuditClock
    from rosclaw_soccer.rsi.shared_native_motor_world import (
        NativePDParameters,
        SharedNativeMotorWorld,
    )
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash

    model = mujoco.MjModel.from_xml_string(
        shared_native_scene(scenes[0].read_text(), prefixes=("a/",))
    )
    if compiled_model_hash(model) != (
        "sha256:115bfd4a136b66903a54bc2807ec728cc2fa251d3e782f1906f352b45199568d"
    ):
        raise ValueError("exact canonical SIM model required")
    layout = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball")
    with np.load(prior_path, allow_pickle=False) as saved:
        prior = {k: saved[k].copy() for k in saved.files}
    stream = ContinuousVelocityStream(
        models[0],
        configs[0],
        expected_model_hash=pins[str(models[0])],
        expected_config_hash=pins[str(configs[0])],
        horizon_controls=301,
    )
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:] = prior["initial_qpos"], prior["initial_qvel"]
    mujoco.mj_forward(model, data)
    pd = NativePDParameters(
        np.array(stream.params.stiffness),
        np.array(stream.params.damping),
        0.85 * np.array(G1_HARD_TORQUE_LIMITS),
    )
    clock = RelayAuditClock(
        SharedNativeMotorWorld(model, data, players={"a/": pd}, ball_body_name="soccer_ball")
    )
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_hash=a.source_hash,
            source_head=a.expected_head,
            input_file_hashes=pins,
            warm_controls=25,
            left_controls=32,
            zero_controls=100,
            max_native_steps=1570,
            original_57_controls_must_match=True,
            no_reset_or_kick=True,
            optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    raw = {key: [] for key in ("qpos", "qvel", "torque", "ball_forces")}
    targets, observations, actions, diagnostics = [], [], [], []

    def retain_partial():
        if (a.output / "complete.json").exists():
            return
        partial = {key: np.concatenate(value) for key, value in raw.items() if value}
        partial.update(initial_qpos=prior["initial_qpos"], initial_qvel=prior["initial_qvel"])
        if targets:
            partial.update(
                joint_targets=np.stack(targets),
                neural_obs=np.stack(observations),
                neural_actions=np.stack(actions),
            )
        path = a.output / "partial.npz"
        np.savez_compressed(path, **partial)
        write_once(
            a.output / "failed.json",
            dict(
                status="INCOMPLETE_NO_AUTOMATIC_RETRY",
                controls_recorded=len(targets),
                native_steps_recorded=len(partial.get("qpos", [])),
                partial_hash=digest(path),
                hardware_authorized=False,
                goal_complete=False,
            ),
        )

    atexit.register(retain_partial)
    for frame in range(157):
        q, v = layout.canonical(data)
        command = np.array([0.0, 0.25 if 25 <= frame < 57 else 0.0, 0.0], np.float32)
        target = stream.propose(q, v, layout.angular_velocity_body(data), command, frame=frame)
        result = clock.advance({"a/": target})
        for key in raw:
            raw[key].append(result[key])
        targets.append(target.copy())
        observations.append(stream.last_observation.copy())
        actions.append(stream.last_action.copy())
        if frame < 57 and (
            any(not np.array_equal(result[k], prior[k][frame * 10 : (frame + 1) * 10]) for k in raw)
            or not np.array_equal(target, prior["joint_targets"][frame])
            or not np.array_equal(stream.last_observation, prior["neural_obs"][frame])
            or not np.array_equal(stream.last_action, prior["neural_actions"][frame])
        ):
            raise ValueError(f"confirmed source prefix diverged at control {frame}")
        q_after, v_after = layout.canonical(data)
        diagnostics.append(
            dict(
                control=frame + 1,
                max_joint_default_error_rad=float(
                    np.max(np.abs(q_after[7:36] - stream.params.defaults))
                ),
                max_joint_speed_rad_s=float(np.max(np.abs(v_after[6:35]))),
                root_xy=q_after[:2].tolist(),
                root_z_m=float(q_after[2]),
                ball_force_max_n=float(np.max(np.abs(result["ball_forces"]))),
            )
        )
        if clock.failed:
            break
    arrays = {key: np.concatenate(value) for key, value in raw.items()}
    arrays.update(
        initial_qpos=prior["initial_qpos"],
        initial_qvel=prior["initial_qvel"],
        joint_targets=np.stack(targets),
        neural_obs=np.stack(observations),
        neural_actions=np.stack(actions),
    )
    artifact = a.output / "live.npz"
    np.savez_compressed(artifact, **arrays)
    check_pins(pins)
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="WARM_LEFT_SETTLING_MEASURED_NOT_KICK",
            source_head=a.expected_head,
            source_hash=a.source_hash,
            input_file_hashes=pins,
            controls=len(targets),
            native_steps=len(arrays["qpos"]),
            neural_calls=stream.neural_calls,
            body_failed=clock.failed,
            original_prefix_exact=len(targets) >= 57,
            diagnostics=diagnostics,
            artifact=dict(path=artifact.name, hash=digest(artifact)),
            optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )
    atexit.unregister(retain_partial)


if __name__ == "__main__":
    main()
