"""Offline endpoint IK from saved actual poses; no NN or dynamics execution."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("composed-complete", "source-audit", "source-root", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    for key in ("composed-hash", "source-audit-hash", "expected-head"):
        parser.add_argument("--" + key, required=True)
    a = parser.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    if a.output.exists():
        raise ValueError("unused offline diagnostic output required")
    for path, commitment in (
        (a.composed_complete, a.composed_hash),
        (a.source_audit, a.source_audit_hash),
    ):
        if path.is_symlink() or digest(path) != commitment:
            raise ValueError("exact closed composed and old physical source required")
    proof = json.loads(a.composed_complete.read_text())
    audit = json.loads(a.source_audit.read_text())
    if (
        proof["status"] != "CURRENT_WALK_TO_KICK_NN_AND_NATIVE_FUTURE_EXACT"
        or audit["status"] != "MOTION_SYNERGY_TEACHER_RAW_CEM_FUTURES_AND_FULL_EXAM_VERIFIED"
    ):
        raise ValueError("independently measured source evidence required")
    pins = dict(proof["input_file_hashes"])
    pins[str(a.composed_complete)] = a.composed_hash
    pins[str(a.source_audit)] = a.source_audit_hash
    records = []
    for index in (2, 6, 8, 9):
        path = a.source_root / f"course-{index:02d}" / "B.json"
        if path.is_symlink() or audit["input_file_hashes"].get(str(path)) != digest(path):
            raise ValueError("closed original source record required")
        pins[str(path)] = digest(path)
        row = json.loads(path.read_text())
        raw = path.parent / row["artifact"]["path"]
        if (
            raw.parent != path.parent
            or raw.is_symlink()
            or audit["input_file_hashes"].get(str(raw)) != digest(raw)
        ):
            raise ValueError("exact original raw source required")
        pins[str(raw)] = digest(raw)
        records.append((index, row, raw))
    source_pins(pins)
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            protocol="soccer.contact_reference_endpoint_ik_geometry.v1",
            source_head=a.expected_head,
            input_file_hashes=pins,
            source_indices=[2, 6, 8, 9],
            maximum_ik_solver_iterations_per_pose=16,
            actual_root_or_ball_variables=False,
            reference_joint_trust_rad=0.35,
            future_first_contact_used_for_OFFLINE_seed_selection=True,
            runtime_teacher_or_complete_motion_reference=False,
            new_physical_steps=0,
            new_neural_calls=0,
            optimizer_updates=0,
            hardware_authorized=False,
        ),
    )
    import mujoco
    import numpy as np

    from rosclaw_soccer.rsi.contact_reference_ik import solve_contact_reference_pose
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash

    scenes = [Path(p) for p in pins if p.endswith("/cached-standard/declared-scene.xml")]
    if len(scenes) != 1:
        raise ValueError("one pinned original body source required")
    model = mujoco.MjModel.from_xml_string(
        shared_native_scene(scenes[0].read_text(), prefixes=("a/",))
    )
    if (
        compiled_model_hash(model)
        != "sha256:115bfd4a136b66903a54bc2807ec728cc2fa251d3e782f1906f352b45199568d"
    ):
        raise ValueError("original compiled physical body required")
    layout = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball")
    results = []
    for index, row, raw in records:
        tick = row["outcome"]["first_foot_contact_frame"] - 1
        if tick < 1:
            raise ValueError("observed original precontact FK seed required")
        data = mujoco.MjData(model)
        with np.load(raw, allow_pickle=False) as saved:
            data.qpos[:], data.qvel[:] = saved["qpos"][tick * 10 - 1], saved["qvel"][tick * 10 - 1]
        data.time = 0.02 * tick  # Read-only reconstructed timestamp, not integrated steps.
        result = solve_contact_reference_pose(model, data, layout, goal=np.array([8.0, 0.0, 0.11]))
        results.append(
            dict(
                index=index,
                seed_control=tick,
                original_task_pass=row["outcome"]["task_pass"],
                result=result,
            )
        )
        write_once(a.output / f"pose-{index:02d}.json", results[-1])
    check_pins(pins)
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="CONTACT_REFERENCE_ENDPOINT_GEOMETRY_RECORDED_NOT_PHYSICS",
            source_head=a.expected_head,
            input_file_hashes=pins,
            results=results,
            new_physical_steps=0,
            new_neural_calls=0,
            optimizer_updates=0,
            future_first_contact_used_for_OFFLINE_seed_selection=True,
            runtime_teacher_or_complete_motion_reference=False,
            task_effectiveness_proven=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
