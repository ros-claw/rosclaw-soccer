"""Rerun velocity NN and independent manual native clock on matched SIM cases."""

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
        raise ValueError("unused output and exact producer required")
    source = json.loads(complete.read_text())
    if (
        source["status"] != "BILATERAL_VELOCITY_BASELINE_RECORDED"
        or [x["case"] for x in source["records"]] != ["new-left", "new-right"]
        or source["optimizer_updates"] != 0
    ):
        raise ValueError("complete declared bilateral velocity producer required")
    pins = dict(source["input_file_hashes"])
    pins[str(complete)] = a.source_hash
    for row in source["records"]:
        path = a.source / row["artifact"]["path"]
        if path.is_symlink() or path.parent != a.source or digest(path) != row["artifact"]["hash"]:
            raise ValueError("exact confined producer trace required")
        pins[str(path)] = row["artifact"]["hash"]
    source_pins(pins)
    check_pins(pins)
    scenes = [Path(x) for x in pins if x.endswith("/cached-standard/declared-scene.xml")]
    models = [Path(x) for x in pins if x.endswith("/velocity/v0/exported/policy.onnx")]
    configs = [Path(x) for x in pins if x.endswith("/velocity/v0/params/deploy.yaml")]
    if not (len(scenes) == len(models) == len(configs) == 1):
        raise ValueError("one pinned scene and velocity model/config required")
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
        raise ValueError("exact native physical model required")
    layout = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball")
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_hash=a.source_hash,
            input_file_hashes=pins,
            source_head=a.expected_head,
            manual_audit_clock=True,
            saved_actions_executed=False,
            nn_codec_and_scene_validation_shared=True,
            optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    results = []
    for old in source["records"]:
        path = a.source / old["artifact"]["path"]
        with np.load(path, allow_pickle=False) as f:
            raw = {name: f[name].copy() for name in f.files}
        staged = [
            Path(x)
            for x in pins
            if x.endswith("/rsi-cftrack-staged-lateral-20261027/" + old["case"] + ".npz")
        ]
        expected_command = {"new-left": 0.25, "new-right": -0.25}[old["case"]]
        if len(staged) != 1 or old["lateral_command_body_m_s"] != expected_command:
            raise ValueError(
                "one source-bound matching initial state and declared command required"
            )
        with np.load(staged[0], allow_pickle=False) as prior:
            if any(
                not np.array_equal(raw[key], prior[key]) for key in ("initial_qpos", "initial_qvel")
            ):
                raise ValueError("audit initial state differs from original stage")
        controls = old["controls"]
        if (
            type(controls) is not int
            or not 1 <= controls <= 300
            or raw["qpos"].shape != (controls * 10, model.nq)
            or raw["qvel"].shape != (controls * 10, model.nv)
            or raw["joint_targets"].shape != (controls, 29)
            or raw["neural_obs"].shape != (controls, 98)
            or raw["neural_actions"].shape != (controls, 29)
        ):
            raise ValueError("bounded complete native and NN trace required")
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
        params = NativePDParameters(
            np.array(stream.params.stiffness),
            np.array(stream.params.damping),
            0.85 * np.array(G1_HARD_TORQUE_LIMITS),
        )
        donor = SharedNativeMotorWorld(
            model, data, players={"a/": params}, ball_body_name="soccer_ball"
        )
        clock = RelayAuditClock(donor)
        command = np.array([0.0, old["lateral_command_body_m_s"], 0.0], dtype=np.float32)
        for frame in range(controls):
            q, v = layout.canonical(data)
            target = stream.propose(q, v, layout.angular_velocity_body(data), command, frame=frame)
            if (
                not np.array_equal(target, raw["joint_targets"][frame])
                or not np.array_equal(stream.last_observation, raw["neural_obs"][frame])
                or not np.array_equal(stream.last_action, raw["neural_actions"][frame])
            ):
                raise ValueError(f"independent NN input/output differs at {old['case']}:{frame}")
            actual = clock.advance({"a/": target})
            for key in ("qpos", "qvel", "torque", "ball_forces"):
                if not np.array_equal(actual[key], raw[key][frame * 10 : (frame + 1) * 10]):
                    raise ValueError(f"independent native {key} differs at {old['case']}:{frame}")
            if clock.failed and frame != controls - 1:
                raise ValueError("producer continued past audit body failure")
        if clock.failed != old["body_failed"] or stream.neural_calls != old["neural_calls"]:
            raise ValueError("matched final body state and NN call count required")
        row = dict(
            case=old["case"],
            controls=controls,
            native_steps=clock.native_steps,
            neural_calls=stream.neural_calls,
            values_exact=True,
            body_failed=clock.failed,
        )
        results.append(row)
        write_once(a.output / (old["case"] + ".json"), row)
        print(json.dumps(row), flush=True)
    check_pins(pins)
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="BILATERAL_VELOCITY_NN_AND_MANUAL_NATIVE_EXACT",
            source_hash=a.source_hash,
            source_head=a.expected_head,
            records=results,
            input_file_hashes=pins,
            neural_codec_and_scene_validation_shared=True,
            optimizer_updates=0,
            new_independent_scenarios=0,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
