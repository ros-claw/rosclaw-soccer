"""Re-execute settled SIM trajectory and separately calculate CFgen pose gap."""

import argparse
import json
import sys
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
        raise ValueError("unused output and exact source report required")
    old = json.loads(complete.read_text())
    if (
        old["status"] != "WARM_LEFT_SETTLING_MEASURED_NOT_KICK"
        or old["controls"] != 157
        or old["native_steps"] != 1570
        or old["original_prefix_exact"] is not True
        or old["body_failed"] is not False
    ):
        raise ValueError("complete healthy source prefix and full settling trial required")
    trace = a.source / old["artifact"]["path"]
    if trace.is_symlink() or trace.parent != a.source or digest(trace) != old["artifact"]["hash"]:
        raise ValueError("exact confined raw trajectory required")
    pins = dict(old["input_file_hashes"])
    pins[str(complete)] = a.source_hash
    pins[str(trace)] = old["artifact"]["hash"]
    source_pins(pins)
    check_pins(pins)
    scenes = [Path(x) for x in pins if x.endswith("/cached-standard/declared-scene.xml")]
    models = [Path(x) for x in pins if x.endswith("/velocity/v0/exported/policy.onnx")]
    configs = [Path(x) for x in pins if x.endswith("/velocity/v0/params/deploy.yaml")]
    cfgen = [Path(x) for x in pins if x.endswith("/policy/omnicontact/CFgen_meta1_loco.py")]
    if not (len(scenes) == len(models) == len(configs) == len(cfgen) == 1):
        raise ValueError("unique pinned scene, velocity actor, config and CFgen required")
    sys.path.insert(0, str(cfgen[0].parents[2]))
    import mujoco
    import numpy as np
    from policy.omnicontact.CFgen_meta1_loco import DEFAULT_JOINT_POS_MJ
    from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

    from rosclaw_soccer.providers.g1.continuous_velocity_stream import ContinuousVelocityStream
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
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
        raise ValueError("pinned canonical physical model required")
    layout = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball")
    with np.load(trace, allow_pickle=False) as f:
        raw = {k: f[k].copy() for k in f.files}
    if raw["qpos"].shape != (1570, model.nq) or raw["joint_targets"].shape != (157, 29):
        raise ValueError("full bounded native trajectory required")
    stream = ContinuousVelocityStream(
        models[0],
        configs[0],
        expected_model_hash=pins[str(models[0])],
        expected_config_hash=pins[str(configs[0])],
        horizon_controls=301,
    )
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:] = raw["initial_qpos"], raw["initial_qvel"]
    mujoco.mj_forward(model, data)
    pd = NativePDParameters(
        np.array(stream.params.stiffness),
        np.array(stream.params.damping),
        0.85 * np.array(G1_HARD_TORQUE_LIMITS),
    )
    world = SharedNativeMotorWorld(model, data, players={"a/": pd}, ball_body_name="soccer_ball")
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_hash=a.source_hash,
            source_head=a.expected_head,
            input_file_hashes=pins,
            max_controls=157,
            max_native_steps=1570,
            manual_source_vs_shared_world_replay=True,
            codec_scene_and_policy_shared=True,
            no_kick_or_training=True,
            hardware_authorized=False,
        ),
    )
    samples = []
    for frame in range(157):
        q, v = layout.canonical(data)
        command = np.array([0.0, 0.25 if 25 <= frame < 57 else 0.0, 0.0], np.float32)
        target = stream.propose(q, v, layout.angular_velocity_body(data), command, frame=frame)
        if (
            not np.array_equal(target, raw["joint_targets"][frame])
            or not np.array_equal(stream.last_observation, raw["neural_obs"][frame])
            or not np.array_equal(stream.last_action, raw["neural_actions"][frame])
        ):
            raise ValueError(f"neural output mismatch at control {frame}")
        result = world.advance({"a/": target})
        if any(
            not np.array_equal(result[k], raw[k][frame * 10 : (frame + 1) * 10])
            for k in ("qpos", "qvel", "torque", "ball_forces")
        ):
            raise ValueError(f"native dynamics mismatch at control {frame}")
        q, v = layout.canonical(data)
        saved = old["diagnostics"][frame]
        if (
            saved["control"] != frame + 1
            or saved["max_joint_default_error_rad"]
            != float(np.max(np.abs(q[7:36] - stream.params.defaults)))
            or saved["max_joint_speed_rad_s"] != float(np.max(np.abs(v[6:35])))
            or saved["root_xy"] != q[:2].tolist()
            or saved["root_z_m"] != float(q[2])
            or saved["ball_force_max_n"] != float(np.max(np.abs(result["ball_forces"])))
        ):
            raise ValueError(f"saved diagnostic mismatch at control {frame}")
        if frame in (56, 81, 106, 131, 156):
            samples.append(
                dict(
                    control=frame + 1,
                    cfgen_default_max_error_rad=float(
                        np.max(np.abs(q[7:36] - DEFAULT_JOINT_POS_MJ))
                    ),
                    velocity_default_max_error_rad=saved["max_joint_default_error_rad"],
                    max_joint_speed_rad_s=saved["max_joint_speed_rad_s"],
                )
            )
        if world.failed:
            raise ValueError(f"replay failed at control {frame}")
    check_pins(pins)
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="WARM_LEFT_SETTLING_NN_AND_NATIVE_EXACT_NOT_KICK",
            source_head=a.expected_head,
            source_hash=a.source_hash,
            input_file_hashes=pins,
            controls=157,
            native_steps=1570,
            neural_calls=stream.neural_calls,
            saved_diagnostics_exact=True,
            samples=samples,
            codec_scene_and_policy_shared=True,
            no_kick_or_training=True,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
