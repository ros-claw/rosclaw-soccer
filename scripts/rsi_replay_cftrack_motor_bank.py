"""Independent native bank replay after complete NN readback, verification only."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--readback", type=Path, required=True)
    parser.add_argument("--readback-hash", required=True)
    parser.add_argument("--external-source", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    complete = a.bank / "complete.json"
    if (
        a.output.exists()
        or a.readback.is_symlink()
        or complete.is_symlink()
        or digest(a.readback) != a.readback_hash
    ):
        raise ValueError("unused replay and exact independent readback required")
    audit = json.loads(a.readback.read_text())
    bank = json.loads(complete.read_text())
    if (
        audit["status"] != "CFTRACK_STATIC_BANK_NN_AND_LABELS_VERIFIED"
        or audit["runtime_failure"] is not None
        or audit["completed_neural_readbacks"] != 3000
        or digest(complete) != audit["source_bank_hash"]
        or len(audit["records"]) != 10
        or len(bank["records"]) != 10
    ):
        raise ValueError("all ten H300 NN/label proofs required before replay")
    pins = dict(audit["input_file_hashes"])
    pins[str(a.readback)] = a.readback_hash
    external = a.external_source.resolve()
    config = external / "policy/omnicontact/config/OmniContact.yaml"
    pose = external / "policy/defaultpose/config/DefaultPose.yaml"
    if any(str(p) not in pins for p in (config, pose)):
        raise ValueError("source-pinned original PD and initial pose required")
    source_pins(pins)
    check_pins(pins)
    scenes = [Path(x) for x in pins if x.endswith("/cached-standard/declared-scene.xml")]
    if len(scenes) != 1:
        raise ValueError("one qualified native body required")
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            maximum_native_steps=30000,
            new_neural_calls=0,
            recorded_targets_are_verification_only=True,
            neural_optimizer_updates=0,
            hardware_authorized=False,
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

    cfg = yaml.safe_load(config.read_text())
    default = yaml.safe_load(pose.read_text())
    model = mujoco.MjModel.from_xml_string(
        shared_native_scene(scenes[0].read_text(), prefixes=("a/",))
    )
    if (
        compiled_model_hash(model)
        != "sha256:115bfd4a136b66903a54bc2807ec728cc2fa251d3e782f1906f352b45199568d"
    ):
        raise ValueError("exact compiled body required")
    lo = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball")
    order = np.array(cfg["lab2mj"], int)
    params = NativePDParameters(
        np.array(cfg["kp_lab"], np.float32)[order],
        np.array(cfg["kd_lab"], np.float32)[order],
        0.85 * np.array(G1_HARD_TORQUE_LIMITS),
    )
    records = []
    total = 0
    for source, verified in zip(bank["records"], audit["records"], strict=True):
        if (
            source["case"] != verified["case"]
            or source["arm"] != verified["arm"]
            or source["artifact"]["hash"] != verified["artifact_hash"]
            or source["motor_plan"] != verified["motor_plan"]
        ):
            raise ValueError("exact source/NN row identity required")
        path = a.bank / source["artifact"]["path"]
        if (
            path.parent != a.bank
            or path.is_symlink()
            or pins.get(str(path)) != digest(path)
            or source["artifact"]["hash"] != digest(path)
        ):
            raise ValueError("bound native raw required")
        with np.load(path, allow_pickle=False) as saved:
            raw = {
                key: saved[key].copy()
                for key in [
                    "qpos",
                    "qvel",
                    "torque",
                    "ball_forces",
                    "joint_targets",
                    "initial_qpos",
                    "initial_qvel",
                ]
            }
        data = mujoco.MjData(model)
        data.qpos[lo.rq : lo.rq + 7] = [0, 0, 0.77, 1, 0, 0, 0]
        data.qpos[lo.qi] = np.array(default["default_angles"], np.float32)
        data.qpos[lo.bq : lo.bq + 7] = [*source["ball_xy"], 0.11, 1, 0, 0, 0]
        if not np.array_equal(data.qpos, raw["initial_qpos"]) or not np.array_equal(
            data.qvel, raw["initial_qvel"]
        ):
            raise ValueError("declared initial recipe does not match raw")
        mujoco.mj_forward(model, data)
        world = SharedNativeMotorWorld(
            model, data, players={"a/": params}, ball_body_name="soccer_ball"
        )
        matched = 0
        failure = None
        try:
            for tick in range(300):
                row = world.advance({"a/": raw["joint_targets"][tick]})
                for key in ["qpos", "qvel", "torque", "ball_forces"]:
                    if not np.array_equal(row[key], raw[key][tick * 10 : (tick + 1) * 10]):
                        raise ValueError(f"strict native mismatch at {tick}: {key}")
                matched += 1
                if world.failed:
                    raise ValueError("replay body failed")
        except Exception as error:
            failure = dict(type=type(error).__name__, message=str(error))
        total += world.native_steps
        result = dict(
            case=source["case"],
            arm=source["arm"],
            matched_controls=matched,
            native_steps=world.native_steps,
            runtime_failure=failure,
            exact=matched == 300 and failure is None,
            verified_outcome=verified["outcome"],
        )
        write_once(a.output / (source["case"] + "-" + source["arm"] + ".json"), result)
        records.append(result)
    pairs = [records[i : i + 2] for i in range(0, 10, 2)]
    check_pins(pins)
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="CFTRACK_STATIC_BANK_NATIVE_REPLAY_EXACT"
            if all(r["exact"] for r in records)
            else "CFTRACK_STATIC_BANK_NATIVE_REPLAY_NOT_QUALIFIED",
            source_head=a.expected_head,
            input_file_hashes=pins,
            readback_hash=a.readback_hash,
            source_bank_hash=audit["source_bank_hash"],
            records=records,
            new_native_steps=total,
            new_neural_calls=0,
            verified_parent_task_passes=sum(r[0]["verified_outcome"]["task_pass"] for r in pairs),
            verified_candidate_task_passes=sum(
                r[1]["verified_outcome"]["task_pass"] for r in pairs
            ),
            verification_replays_are_not_extra_independent_cases=True,
            scope_static_centimeter_development_not_generalized_teacher=True,
            neural_optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
            commercial_publicity_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
