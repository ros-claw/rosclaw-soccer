"""Bind a post-hoc contact contamination diagnostic to all original pilot data."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("source-hash", "expected-head"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    from rosclaw_soccer.rsi.native_contact_authority_metrics import contact_attribution_diagnostics
    from rosclaw_soccer.training.native_team_courses import native_team_course_bank
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    source = a.source / "complete.json"
    if a.output.exists() or source.is_symlink() or digest(source) != a.source_hash:
        raise ValueError("exact completed source and new diagnostic output required")
    report = json.loads(source.read_text())
    if (
        report["status"] != "NATIVE8_B_AND_B_PLUS_C_CONTACT_PILOT_MEASURED_NOT_LEARNED"
        or [r["course"] for r in report["results"]]
        != [c.course_id for c in native_team_course_bank()]
        or report["promotion_authorized"] is not False
    ):
        raise ValueError("complete unchanged five-course unpromoted source required")
    pins = dict(report["input_file_hashes"])
    pins[str(source)] = a.source_hash
    source_pins(pins)
    check_pins(pins)
    import numpy as np

    rows = []
    for case in report["results"]:
        root = a.source / case["course"]
        path = root / "complete.json"
        if path.is_symlink() or digest(path) != case["source_hash"]:
            raise ValueError("exact original physical case required")
        physical = json.loads(path.read_text())
        if len(physical["records"]) != 6 or not physical["fresh_reexecution_exact"]:
            raise ValueError("all paired original physical arms required")
        check_pins(physical["input_file_hashes"])
        for record in physical["records"]:
            path = root / record["artifact"]["path"]
            if (
                path.parent.resolve() != root.resolve()
                or path.is_symlink()
                or digest(path) != record["artifact"]["hash"]
            ):
                raise ValueError("unchanged original physical arrays required")
            pins[str(path)] = record["artifact"]["hash"]
            if record["repeat"]:
                continue
            with np.load(path) as saved:
                arrays = dict(saved)
            review = contact_attribution_diagnostics(arrays)
            review.update(
                interplayer_peak_force_n=float(arrays["interplayer_contact_forces"].max()),
                interplayer_collision_native_samples=int(
                    (arrays["interplayer_contact_forces"].max(axis=(1, 2)) > 1).sum()
                ),
            )
            rows.append(
                dict(
                    course=case["course"],
                    arm=record["arm"],
                    artifact_hash=record["artifact"]["hash"],
                    review=review,
                )
            )
    check_pins(pins)
    clean(a.expected_head)
    write_once(
        a.output,
        dict(
            schema="soccer.native8_approach_contact_attribution_review.v1",
            source_head=a.expected_head,
            source_hash=a.source_hash,
            input_file_hashes=pins,
            results=rows,
            all_cases_and_arms_retained=True,
            defined_after_pilot=True,
            original_scores_unchanged=True,
            physical_steps_added=0,
            optimizer_updates=0,
            physical_teacher_qualified=False,
            promotion_authorized=False,
            goal_complete=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
