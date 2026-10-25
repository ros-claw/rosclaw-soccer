"""Read independently audited teacher failures and inspect pre-integration geometry."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("audit", "producer-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--audit-hash", required=True)
    p.add_argument("--expected-head", required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    if a.output.exists() or digest(a.audit) != a.audit_hash:
        raise ValueError("unused diagnostic output and exact independent audit required")
    audit = json.loads(a.audit.read_text())
    if (
        audit["status"] != "MOTION_SYNERGY_TEACHER_RAW_CEM_FUTURES_AND_FULL_EXAM_VERIFIED"
        or audit["actual_worlds"] != 80
        or audit["new_physical_steps"] != 0
    ):
        raise ValueError("complete independently audited40 paired physical worlds required")
    pins = dict(audit["input_file_hashes"])
    pins[str(a.audit)] = a.audit_hash
    cp = a.producer_root / "complete.json"
    if pins.get(str(cp)) != digest(cp):
        raise ValueError("exact bound physical producer required")
    producer = json.loads(cp.read_text())
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
    for i, item in enumerate(producer["results"]):
        ref = item["arms"]["C"]
        path = a.producer_root / ref["record"]
        if (
            not path.resolve().is_relative_to(a.producer_root.resolve())
            or pins.get(str(path)) != ref["hash"]
        ):
            raise ValueError("bound actual teacher record required")
        row = json.loads(path.read_text())
        if row["outcome"]["task_pass"]:
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
        results.append(dict(course_index=i, actual_outcome=row["outcome"], **diagnosis))
    check_pins(pins)
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "complete.json",
        dict(
            status="AUDITED_ACTUAL_FAILURE_CONTACT_GEOMETRY_DIAGNOSED",
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
