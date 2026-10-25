"""Read independently audited teacher failures and inspect pre-integration geometry."""

import argparse
import json
from pathlib import Path


def diagnostic_records(audit, producer, *, paired_aim_pilot=False):
    """Select bounded audited records, preserving actual course identities."""
    if audit.get("new_physical_steps") != 0:
        raise ValueError("offline independently audited evidence required")
    expected = (
        ("FULL_NN_AIM_COMMAND_RAW_AND_PHYSICAL_EXAM_VERIFIED", 12, [0, 2, 8, 22, 27, 30])
        if paired_aim_pilot
        else (
            "MOTION_SYNERGY_TEACHER_RAW_CEM_FUTURES_AND_FULL_EXAM_VERIFIED",
            80,
            list(range(40)),
        )
    )
    status, worlds, indices = expected
    results = producer.get("results", [])
    if (
        audit.get("status") != status
        or audit.get("actual_worlds") != worlds
        or [r.get("course_index") for r in results] != indices
        or any(set(r.get("arms", {})) != {"B", "C"} for r in results)
    ):
        raise ValueError("exact independently audited paired course bank required")
    return [
        (item["course_index"], arm, item["arms"][arm])
        for item in results
        for arm in (("B", "C") if paired_aim_pilot else ("C",))
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("audit", "producer-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--audit-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--paired-aim-pilot", action="store_true")
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    if a.output.exists() or digest(a.audit) != a.audit_hash:
        raise ValueError("unused diagnostic output and exact independent audit required")
    audit = json.loads(a.audit.read_text())
    pins = dict(audit["input_file_hashes"])
    pins[str(a.audit)] = a.audit_hash
    cp = a.producer_root / "complete.json"
    if pins.get(str(cp)) != digest(cp):
        raise ValueError("exact bound physical producer required")
    producer = json.loads(cp.read_text())
    selected = diagnostic_records(audit, producer, paired_aim_pilot=a.paired_aim_pilot)
    source_pins(pins)
    check_pins(pins)
    import mujoco
    import numpy as np

    from rosclaw_soccer.rsi.contact_transition_diagnostics import diagnose_contact_transition
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene

    scenes = [Path(k) for k in pins if k.endswith("/cached-standard/declared-scene.xml")]
    if len(scenes) != 1:
        raise ValueError("one exact qualified collision model required")
    model = mujoco.MjModel.from_xml_string(
        shared_native_scene(scenes[0].read_text(), prefixes=("a/",))
    )
    layout = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball")
    results = []
    for i, arm, ref in selected:
        path = a.producer_root / ref["record"]
        if (
            not path.resolve().is_relative_to(a.producer_root.resolve())
            or pins.get(str(path)) != ref["hash"]
        ):
            raise ValueError("bound actual teacher record required")
        row = json.loads(path.read_text())
        if row["outcome"]["task_pass"] and not a.paired_aim_pilot:
            continue
        artifact = path.parent / row["artifact"]["path"]
        if (
            not artifact.resolve().is_relative_to(a.producer_root.resolve())
            or pins.get(str(artifact)) != row["artifact"]["hash"]
        ):
            raise ValueError("exact bound actual failure arrays required")
        with np.load(artifact, allow_pickle=False) as saved:
            raw = {k: saved[k].copy() for k in ("qpos", "qvel", "ball_forces")}
        diagnosis = diagnose_contact_transition(model, layout, raw)
        first = diagnosis["first_foot_native_index"]
        results.append(
            dict(
                course_index=i,
                arm=arm,
                actual_outcome=row["outcome"],
                entry_control=row["entry_control"],
                entry_to_first_foot_ms=None
                if first is None
                else 2 * (first - 10 * row["entry_control"]),
                desired_command_world_point=row.get("desired_command_world_point"),
                task_scoring_goal_world_point=row.get("task_scoring_goal_world_point"),
                **diagnosis,
            )
        )
    check_pins(pins)
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "complete.json",
        dict(
            status="AUDITED_AIM_PILOT_PAIRED_CONTACT_GEOMETRY_DIAGNOSED"
            if a.paired_aim_pilot
            else "AUDITED_ACTUAL_FAILURE_CONTACT_GEOMETRY_DIAGNOSED",
            source_head=a.expected_head,
            input_file_hashes=pins,
            results=results,
            new_physical_steps=0,
            neural_optimizer_updates=0,
            candidate_generated=False,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
