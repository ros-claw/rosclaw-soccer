"""Strict native verification replay after independent CFtrack NN readback.

Saved targets are replayed ONLY as an evidence check, never as a runtime actor.
The producer and readback must separately prove actual neural observations.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qualification", type=Path, required=True)
    p.add_argument("--readback", type=Path, required=True)
    p.add_argument("--readback-hash", required=True)
    p.add_argument("--external-source", type=Path, required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    if a.output.exists() or a.readback.is_symlink() or digest(a.readback) != a.readback_hash:
        raise ValueError("unused output and exact closed independent NN readback required")
    audit = json.loads(a.readback.read_text())
    complete = a.qualification / "complete.json"
    if (
        audit["status"] != "CFTRACK_ACTUAL_NN1244_HISTORY_AND_TARGETS_VERIFIED"
        or audit["actual_neural_readbacks"] != 600
        or complete.is_symlink()
        or digest(complete) != audit["qualification_hash"]
    ):
        raise ValueError("full independent actual neural history proof required")
    report = json.loads(complete.read_text())
    artifact = a.qualification / report["artifact"]["path"]
    if (
        report["controls"] != 600
        or report["runtime_failure"] is not None
        or artifact.parent != a.qualification
        or artifact.is_symlink()
        or digest(artifact) != report["artifact"]["hash"]
    ):
        raise ValueError("exact full native raw required")
    pins = dict(audit["input_file_hashes"])
    pins[str(a.readback)] = a.readback_hash
    source_pins(pins)
    check_pins(pins)
    reference_source = report.get("reference_source", "CFgen")
    motion = None
    if audit.get("reference_source", "CFgen") != reference_source:
        raise ValueError("producer and neural readback reference protocols disagree")
    external = a.external_source.resolve()
    if reference_source == "NPZmotion":
        from rosclaw_soccer.rsi.cftrack_motion_reference import load_cftrack_motion

        motion_path = Path(report["motion_source"])
        if (
            str(motion_path) not in pins
            or not motion_path.is_relative_to(external)
            or report.get("reference_conditioning") is not None
            or audit.get("motion_source") != str(motion_path)
        ):
            raise ValueError("same producer-pinned complete motion required")
        motion = load_cftrack_motion(motion_path)
    elif reference_source != "CFgen" or report.get("motion_source") is not None:
        raise ValueError("explicit qualified CFtrack reference protocol required")
    config = external / "policy/omnicontact/config/OmniContact.yaml"
    pose = external / "policy/defaultpose/config/DefaultPose.yaml"
    if any(str(path) not in pins for path in (config, pose)):
        raise ValueError("producer-pinned PD/initial pose configuration required")
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            maximum_native_steps=6000,
            new_neural_calls=0,
            neural_optimizer_updates=0,
            hardware_authorized=False,
            recorded_targets_are_verification_only_not_runtime_controller=True,
            reference_source=reference_source,
        ),
    )
    import mujoco
    import numpy as np
    import yaml
    from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.shared_native_motor_world import (
        NativePDParameters,
        SharedNativeMotorWorld,
    )
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash

    config_data, pose_data = yaml.safe_load(config.read_text()), yaml.safe_load(pose.read_text())
    scenes = [Path(x) for x in pins if x.endswith("/cached-standard/declared-scene.xml")]
    if len(scenes) != 1:
        raise ValueError("one pinned standard native scene required")
    m = mujoco.MjModel.from_xml_string(shared_native_scene(scenes[0].read_text(), prefixes=("a/",)))
    if (
        compiled_model_hash(m)
        != "sha256:115bfd4a136b66903a54bc2807ec728cc2fa251d3e782f1906f352b45199568d"
    ):
        raise ValueError("exact compiled body required")
    lo = NativeVelocityLayout(m, robot_prefix="a/", ball_body_name="soccer_ball")
    d = mujoco.MjData(m)
    d.qpos[lo.rq : lo.rq + 7] = [0, 0, 0.77, 1, 0, 0, 0]
    d.qpos[lo.qi] = np.array(pose_data["default_angles"], np.float32)
    d.qpos[lo.bq : lo.bq + 7] = [1, 0, 0.11, 1, 0, 0, 0]
    if motion is not None:
        order = np.array(config_data["lab2mj"], int)
        d.qpos[lo.rq : lo.rq + 3] = motion["body_pos_w"][0, 0]
        d.qpos[lo.rq + 3 : lo.rq + 7] = motion["body_quat_w"][0, 0]
        d.qpos[lo.qi] = motion["joint_pos"][0, order]
        d.qpos[lo.bq : lo.bq + 3] = motion["object_pos_w"][0]
        d.qpos[lo.bq + 2] = max(0.11, float(d.qpos[lo.bq + 2]))
        d.qpos[lo.bq + 3 : lo.bq + 7] = motion["object_quat_w"][0]
    mujoco.mj_forward(m, d)
    order = np.array(config_data["lab2mj"], int)
    params = NativePDParameters(
        np.array(config_data["kp_lab"], np.float32)[order],
        np.array(config_data["kd_lab"], np.float32)[order],
        0.85 * np.array(G1_HARD_TORQUE_LIMITS),
    )
    world = SharedNativeMotorWorld(m, d, players={"a/": params}, ball_body_name="soccer_ball")
    with np.load(artifact, allow_pickle=False) as saved:
        raw = {key: saved[key].copy() for key in saved.files}
    failure = None
    matched = 0
    try:
        for tick in range(600):
            row = world.advance({"a/": raw["joint_targets"][tick]})
            for key in ("qpos", "qvel", "torque", "ball_forces"):
                if not np.array_equal(row[key], raw[key][tick * 10 : (tick + 1) * 10]):
                    raise ValueError(f"strict native replay mismatch at control {tick}: {key}")
            matched += 1
            if row["body_failed"]["a/"]:
                raise ValueError("native replay body failed")
    except Exception as error:
        failure = dict(type=type(error).__name__, message=str(error))
    check_pins(pins)
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="CFTRACK_FULL_NATIVE_REPLAY_EXACT"
            if matched == 600 and failure is None
            else "CFTRACK_NATIVE_REPLAY_NOT_QUALIFIED",
            source_head=a.expected_head,
            input_file_hashes=pins,
            readback_hash=a.readback_hash,
            compared_controls=matched,
            new_native_steps=world.native_steps,
            new_neural_calls=0,
            reference_source=reference_source,
            runtime_failure=failure,
            neural_optimizer_updates=0,
            recorded_targets_are_verification_only=True,
            task_pass_is_not_implied_by_replay=True,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
