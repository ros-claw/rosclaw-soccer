"""Fail-closed consumed SIM comparison gate, never promotion or hardware permit."""

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
from rosclaw_soccer.rsi.output_memory_step_motor import validate_model
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_collect_step_motor_pilot import COURSES
from scripts.rsi_train_bilateral_contact_motor_v303 import write_once


def check_pilot(summary: dict[str, Any], review: dict[str, Any], model_hash: str) -> None:
    if (
        summary["schema"] != "soccer.rsi.online_step_motor_physical_validation.v1"
        or summary["commitment"]["model_hash"] != model_hash
        or review["schema"] != "soccer.rsi.step_motor_physics_independent_review.v1"
        or review["source_summary_hash"] != summary["report_hash"]
        or review["actual_reports_reviewed"] != 12
        or review["actual_motor_actions_reconstructed"] != 2400
        or summary["independent_contexts"] != 4
        or review["independent_contexts"] != 4
        or summary["physical_executions"] != 12
        or summary["commitment"]["partition"] != "CONSUMED_PILOT"
        or [tuple(v) for v in summary["commitment"]["courses"]] != list(COURSES)
        or review["safe_pelvis"] is not True
        or any(
            review[k] != 0 or summary[k] != 0
            for k in ("old_high_quality_loss", "old_clean_foot_loss", "new_out_of_play")
        )
        or review["candidate_high_quality"] != summary["online_high_quality"]
        or review["baseline_high_quality"] != summary["baseline_high_quality"]
        or any(
            obj.get(k) is not False
            for obj in (summary, summary["commitment"], review)
            for k in ("promotion_authorized", "hardware_authorized")
        )
    ):
        raise ValueError("complete safe retained independently reviewed consumed pilot required")


def check_cpu(
    report: dict[str, Any],
    review: dict[str, Any],
    baseline: dict[str, Any],
    baseline_review: dict[str, Any],
    *,
    model_hash: str,
    baseline_hash: str,
    course: tuple[int, int],
) -> None:
    for meta, proof, expected in (
        (report, review, model_hash),
        (baseline, baseline_review, baseline_hash),
    ):
        if (
            meta["schema"] != "soccer.rsi.cpu_motor_transfer.v1"
            or meta["partition"] != "CONSUMED_TRANSFER_DIAGNOSTIC"
            or (meta["seed"], meta["lane"]) != course
            or meta["step_model_hash"] != expected
            or proof["schema"] != "soccer.rsi.cpu_motor_transfer_review.v1"
            or proof["reviewed_report_hash"] != meta["report_hash"]
            or proof["physical_substeps"] != 3000
            or any(
                proof[k] is not True
                for k in (
                    "actual_mujoco_dynamics_replayed",
                    "actual_pd_torque_reconstructed",
                    "neural_target_reconstructed",
                    "safety_passed",
                )
            )
            or any(
                type(proof[k]) not in (float, int) or not math.isfinite(proof[k])
                for k in ("minimum_pelvis_z_m", "maximum_lateral_excursion_m")
            )
            or proof["minimum_pelvis_z_m"] < 0.65
            or any(type(proof[k]) is not bool for k in ("high_quality", "clean_foot_only"))
            or any(
                obj.get(k) is not False
                for obj in (meta, proof)
                for k in ("promotion_authorized", "hardware_authorized")
            )
        ):
            raise ValueError("fully replayed bound safe CPU comparison required")
    if (
        baseline_review["high_quality"]
        and not review["high_quality"]
        or baseline_review["clean_foot_only"]
        and not review["clean_foot_only"]
        or baseline_review["maximum_lateral_excursion_m"]
        <= 4
        < review["maximum_lateral_excursion_m"]
    ):
        raise ValueError("CPU comparison lost old success/clean contact or added out-of-play")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate", "baseline", "pilot-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--cpu-root", type=Path, action="append", required=True)
    parser.add_argument("--baseline-cpu-root", type=Path, action="append", required=True)
    parser.add_argument("--storage-root", type=Path, required=True)
    args = parser.parse_args()
    if len(args.cpu_root) != 4 or len(args.baseline_cpu_root) != 4:
        parser.error("all four paired CPU courses required")
    model = json.loads(args.candidate.read_text())
    baseline = json.loads(args.baseline.read_text())
    validate_model(model)
    validate_model(baseline)
    pilot = _sealed(args.pilot_root / "validation_summary.json")
    review = _sealed(args.pilot_root / "independent_review.json")
    check_pilot(pilot, review, model["model_hash"])
    cpu_reviews = []
    for course, root, old_root in zip(COURSES, args.cpu_root, args.baseline_cpu_root, strict=True):
        current_report = _sealed(root / "report.json")
        current_review = _sealed(root / "independent_review.json")
        old_report = _sealed(old_root / "report.json")
        old_review = _sealed(old_root / "review.json")
        check_cpu(
            current_report,
            current_review,
            old_report,
            old_review,
            model_hash=model["model_hash"],
            baseline_hash=baseline["model_hash"],
            course=course,
        )
        cpu_reviews.append(current_review["report_hash"])
    free = shutil.disk_usage(args.storage_root).free
    if free < 120 * 1024**3:
        raise ValueError("20 GiB full-bank allowance plus 100 GiB system reserve required")
    report = dict(
        schema="soccer.rsi.consumed_memory_bank_preflight.v1",
        candidate_model_hash=model["model_hash"],
        cpu_baseline_model_hash=baseline["model_hash"],
        source_pilot_summary_hash=pilot["report_hash"],
        source_pilot_review_hash=review["report_hash"],
        cpu_review_hashes=cpu_reviews,
        all_four_cpu_courses_retained=True,
        storage_available_bytes=free,
        system_reserve_bytes=100 * 1024**3,
        full_bank_allowance_bytes=20 * 1024**3,
        source_hash=hash_bytes(Path(__file__).read_bytes()),
        qualification="CONSUMED_SIM_COMPARISON_ONLY_NOT_PROMOTION_NOT_FRESH",
        promotion_authorized=False,
        hardware_authorized=False,
    )
    report["report_hash"] = hash_json(report)
    write_once(args.output, report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
