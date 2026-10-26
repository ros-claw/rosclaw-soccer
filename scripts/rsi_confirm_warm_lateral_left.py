"""Re-execute warm plus selected left prediction on one live SIM clock."""

import argparse
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
        raise ValueError("exact closed counterfactual report and unused output required")
    source = json.loads(complete.read_text())
    if (
        source["status"] != "WARM_LATERAL_COUNTERFACTUALS_RECORDED"
        or [r["case"] for r in source["records"]] != ["left", "hold", "right"]
        or not source["branches_not_actual_execution"]
    ):
        raise ValueError("complete declared warm fork experiment required")
    pins = dict(source["input_file_hashes"])
    pins[str(complete)] = a.source_hash
    prediction = source["records"][0]
    path = a.source / prediction["artifact"]["path"]
    if (
        path.is_symlink()
        or path.parent != a.source
        or digest(path) != prediction["artifact"]["hash"]
    ):
        raise ValueError("exact confined left prediction required")
    pins[str(path)] = prediction["artifact"]["hash"]
    source_pins(pins)
    check_pins(pins)
    scenes = [Path(x) for x in pins if x.endswith("/cached-standard/declared-scene.xml")]
    models = [Path(x) for x in pins if x.endswith("/velocity/v0/exported/policy.onnx")]
    configs = [Path(x) for x in pins if x.endswith("/velocity/v0/params/deploy.yaml")]
    initials = [
        Path(x) for x in pins if x.endswith("/rsi-lateral-velocity-baseline-20261027/new-left.npz")
    ]
    if not (len(scenes) == len(models) == len(configs) == len(initials) == 1):
        raise ValueError("unique pinned scene, actor, codec and source initial state required")
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
    if (
        compiled_model_hash(model)
        != "sha256:115bfd4a136b66903a54bc2807ec728cc2fa251d3e782f1906f352b45199568d"
    ):
        raise ValueError("exact source native model required")
    layout = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball")
    with np.load(initials[0], allow_pickle=False) as saved:
        q0, v0 = saved["initial_qpos"].copy(), saved["initial_qvel"].copy()
    with np.load(path, allow_pickle=False) as saved:
        predicted = {name: saved[name].copy() for name in saved.files}
    if prediction["controls"] != 32 or predicted["qpos"].shape != (320, model.nq):
        raise ValueError("complete 32-control left prediction required")
    stream = ContinuousVelocityStream(
        models[0],
        configs[0],
        expected_model_hash=pins[str(models[0])],
        expected_config_hash=pins[str(configs[0])],
        horizon_controls=301,
    )
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:] = q0, v0
    mujoco.mj_forward(model, data)
    pd = NativePDParameters(
        np.array(stream.params.stiffness),
        np.array(stream.params.damping),
        0.85 * np.array(G1_HARD_TORQUE_LIMITS),
    )
    initialized = SharedNativeMotorWorld(
        model, data, players={"a/": pd}, ball_body_name="soccer_ball"
    )
    clock = RelayAuditClock(initialized)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_hash=a.source_hash,
            source_head=a.expected_head,
            input_file_hashes=pins,
            one_live_manual_clock=True,
            warm_controls=25,
            left_controls=32,
            max_native_steps=570,
            shared_original_velocity_codec=True,
            command_change_from_zero_is_abrupt=True,
            no_skill_handoff_or_kick=True,
            optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    raw = {key: [] for key in ("qpos", "qvel", "torque", "ball_forces")}
    targets, obs, actions = [], [], []
    for frame in range(57):
        q, v = layout.canonical(data)
        command = np.array([0.0, 0.0 if frame < 25 else 0.25, 0.0], np.float32)
        target = stream.propose(q, v, layout.angular_velocity_body(data), command, frame=frame)
        result = clock.advance({"a/": target})
        for key in raw:
            raw[key].append(result[key])
        targets.append(target.copy())
        obs.append(stream.last_observation.copy())
        actions.append(stream.last_action.copy())
        if frame >= 25:
            i = frame - 25
            if (
                any(not np.array_equal(result[k], predicted[k][i * 10 : (i + 1) * 10]) for k in raw)
                or not np.array_equal(target, predicted["joint_targets"][i])
                or not np.array_equal(stream.last_observation, predicted["neural_obs"][i])
                or not np.array_equal(stream.last_action, predicted["neural_actions"][i])
            ):
                raise ValueError(f"live left execution diverges from predicted NN/native at {i}")
        if clock.failed:
            raise ValueError("actual warmed body failed; do not continue")
    arrays = {key: np.concatenate(value) for key, value in raw.items()}
    arrays.update(
        initial_qpos=q0,
        initial_qvel=v0,
        joint_targets=np.stack(targets),
        neural_obs=np.stack(obs),
        neural_actions=np.stack(actions),
    )
    artifact = a.output / "live.npz"
    np.savez_compressed(artifact, **arrays)
    check_pins(pins)
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="WARM_LEFT_LIVE_NN_AND_MANUAL_NATIVE_MATCH",
            source_hash=a.source_hash,
            source_head=a.expected_head,
            input_file_hashes=pins,
            controls=57,
            native_steps=570,
            neural_calls=57,
            predicted_32_controls_exact=True,
            body_failed=clock.failed,
            artifact=dict(path=artifact.name, hash=digest(artifact)),
            no_kick_or_handoff=True,
            optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
