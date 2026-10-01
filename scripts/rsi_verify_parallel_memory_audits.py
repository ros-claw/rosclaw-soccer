"""Reconstruct the same declared real courses serially and in worker processes.

This is audit equivalence, not additional physics executions or learned gain.
All 300 motor frames of every sample are audited in both passes.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_audit_memory_learning_rollouts import audit_course, ordered_audits
from scripts.rsi_train_bilateral_contact_motor_v303 import write_once


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--behavior-kind", choices=("output-memory", "smooth-memory"), required=True
    )
    for name in ("exploration-root", "parent-bank-root", "model", "output-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.behavior_kind == "smooth-memory":
        from rosclaw_soccer.rsi.smooth_memory_motor import validate_model
        from scripts.rsi_fit_smooth_memory_motor import checked_curriculum
    else:
        from rosclaw_soccer.rsi.output_memory_step_motor import validate_model
        from scripts.rsi_fit_output_memory_motor import checked_curriculum
    model = json.loads(args.model.read_text())
    validate_model(model)
    summary = _sealed(args.exploration_root / "training_summary.json")
    bank = _sealed(args.parent_bank_root / "validation_summary.json")
    review = _sealed(args.parent_bank_root / "independent_review.json")
    commitment = summary["commitment"]
    courses = checked_curriculum(
        model,
        summary,
        bank,
        review,
        json.loads((args.exploration_root / "commitment.json").read_text()),
    )
    if len(courses) < 2:
        raise ValueError("two declared distinct courses required")
    jobs = [
        dict(
            kind=args.behavior_kind,
            model_path=str(args.model.resolve()),
            model_hash=model["model_hash"],
            index=i,
            course=course,
            row=summary["rows"][i],
            commitment=commitment,
            exploration_root=str(args.exploration_root.resolve()),
        )
        for i, course in enumerate(courses[:2])
    ]
    args.output_root.mkdir(parents=True, exist_ok=False)
    results, timings, data_hashes = [], [], []
    for workers, name in ((1, "serial"), (2, "parallel")):
        started = time.monotonic()
        measured = list(ordered_audits(audit_course, jobs, workers))
        records = [record for rows, _ in measured for record in rows]
        arrays = {k: np.concatenate([a[k] for _, a in measured]) for k in measured[0][1]}
        path = args.output_root / f"{name}.npz"
        np.savez_compressed(path, **arrays)
        timings.append(time.monotonic() - started)
        data_hashes.append(hash_bytes(path.read_bytes()))
        results.append((records, arrays))
    serial_records, serial = results[0]
    parallel_records, parallel = results[1]
    if serial_records != parallel_records or serial.keys() != parallel.keys():
        raise ValueError("serial/parallel audit identity differs")
    if any(
        serial[k].dtype != parallel[k].dtype or not np.array_equal(serial[k], parallel[k])
        for k in serial
    ):
        raise ValueError("parallel reconstruction changed measured training arrays")
    report = dict(
        schema="soccer.rsi.parallel_memory_audit_equivalence.v1",
        source_summary_hash=summary["report_hash"],
        model_hash=model["model_hash"],
        course_selection="FIRST_TWO_IN_SEALED_ORDER",
        courses=courses[:2],
        trajectories_per_pass=len(serial_records),
        full_motor_frames_per_pass=300 * len(serial_records),
        learning_frames_per_pass=len(serial["observation"]),
        serial_parallel_arrays_bitwise_equal=True,
        serial_parallel_records_equal=True,
        measured_seconds=timings,
        data_hashes=data_hashes,
        source_hash=hash_bytes(Path(__file__).read_bytes()),
        helper_hash=hash_bytes(
            Path(__file__).with_name("rsi_audit_memory_learning_rollouts.py").read_bytes()
        ),
        physical_executions_added=0,
        qualification="AUDIT_EQUIVALENCE_ONLY_NOT_LEARNING_GAIN",
        promotion_authorized=False,
        hardware_authorized=False,
    )
    report["report_hash"] = hash_json(report)
    write_once(args.output_root / "report.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
