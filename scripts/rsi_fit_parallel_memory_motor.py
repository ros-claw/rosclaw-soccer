"""Fully audit a sealed on-policy bank in bounded CPU workers, then learn.

The model/optimizer mathematics and current-parent curriculum checks are reused
unchanged. Parallelism does not skip reconstruction or accept partial banks.
"""

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

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
    parser.add_argument("--audit-workers", type=int, default=4, choices=range(1, 5))
    for name in ("exploration-root", "parent-bank-root", "model", "output-root"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    learner: Callable[..., Any]
    if args.behavior_kind == "smooth-memory":
        from rosclaw_soccer.rsi.smooth_memory_learning import fit_update as fit_smooth
        from rosclaw_soccer.rsi.smooth_memory_motor import validate_model
        from scripts.rsi_fit_smooth_memory_motor import checked_curriculum

        learner = fit_smooth
    else:
        from rosclaw_soccer.rsi.output_memory_motor_learning import fit_update as fit_output
        from rosclaw_soccer.rsi.output_memory_step_motor import validate_model
        from scripts.rsi_fit_output_memory_motor import checked_curriculum

        learner = fit_output
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
    jobs = [
        dict(
            kind=args.behavior_kind,
            model_path=str(args.model.resolve()),
            model_hash=model["model_hash"],
            index=i,
            course=course,
            row=row,
            commitment=commitment,
            exploration_root=str(args.exploration_root.resolve()),
        )
        for i, (course, row) in enumerate(zip(courses, summary["rows"], strict=True))
    ]
    records: list[dict[str, Any]] = []
    chunks: dict[str, list[Any]] = {}
    for measured, arrays in ordered_audits(audit_course, jobs, args.audit_workers):
        records.extend(measured)
        for key, array in arrays.items():
            chunks.setdefault(key, []).append(array)
    combined = {k: np.concatenate(v) for k, v in chunks.items()}
    expected = len(courses) * commitment["samples_per_course"]
    if (
        len(records) != expected
        or [r["group"] for r in records] != list(range(expected))
        or any(len(v) != expected * 270 for v in combined.values())
        or not np.array_equal(combined["trajectory_index"], np.repeat(np.arange(expected), 270))
        or model != json.loads(args.model.read_text())
    ):
        raise ValueError("complete ordered immutable current-parent bank required")
    args.output_root.mkdir(parents=True, exist_ok=False)
    path = args.output_root / "rollouts.npz"
    np.savez_compressed(path, **combined)
    smooth = args.behavior_kind == "smooth-memory"
    manifest = dict(
        schema="soccer.rsi.smooth_memory_on_policy_bank.v1"
        if smooth
        else "soccer.rsi.output_memory_on_policy_bank.v1",
        partition="TRAIN_CONSUMED",
        source_summary_hash=summary["report_hash"],
        parent_model_hash=model["model_hash"],
        records=records,
        physical_rollout_count=len(records),
        frame_sample_count=len(combined["observation"]),
        independent_contexts=len(courses),
        data_hash=hash_bytes(path.read_bytes()),
        source_hash=hash_bytes(Path(__file__).read_bytes()),
        audit_helper_hash=hash_bytes(
            Path(__file__).with_name("rsi_audit_memory_learning_rollouts.py").read_bytes()
        ),
        audit_workers=args.audit_workers,
        audit_order="DECLARED_COURSE_SAMPLE_NOT_COMPLETION_ORDER",
        promotion_authorized=False,
        hardware_authorized=False,
    )
    if smooth:
        manifest.update(sampling_rho=0.9, candidate_previous_mean_required=True)
    manifest["report_hash"] = hash_json(manifest)
    write_once(args.output_root / "rollout_manifest.json", manifest)
    # Both existing fitters default to rho=.9 for the smooth policy. Explicit
    # conditioning still occurs in its unchanged optimizer, not this launcher.
    learned = learner(model, combined, batch_hash=manifest["report_hash"])
    write_once(args.output_root / "model.json", learned)
    print(json.dumps(dict(model_hash=learned["model_hash"], receipt=learned["learning_receipt"])))


if __name__ == "__main__":
    main()
