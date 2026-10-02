"""One new actual execution tests the known out-of-play counterexample.

Reuses and independently reconstructs existing parent/qualified-NN controls.
This early rejection diagnostic never replaces a full bank or a fresh exam.
"""

import argparse
import math
from pathlib import Path
from typing import Any

from rosclaw_soccer.rsi.consolidated_smooth_motor import validate_model
from rosclaw_soccer.rsi.failure_curriculum_evidence import _outcome, _sealed
from rosclaw_soccer.rsi.json_artifact_io import load_json_artifact
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_collect_approach_lateral_tracking_v286 import _run
from scripts.rsi_collect_failed_step_courses import qualified_memory_failure_rows
from scripts.rsi_compressed_bank_storage import capacity_check
from scripts.rsi_train_protected_online_motor_v308 import _head

MODEL = "sha256:f4d964d0be368f8fbf4be555f00519212296514de1b8e664cf8d58967923e1d1"
LEARNING = "sha256:cbe81f457bbf7c8729b086ecb85565b1d4b5d91d8ea6da1b79014e6908e027ae"
COURSE = (20262104, 2)


def risk_gate(warm: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    for row in (warm, candidate):
        if (
            any(type(row[k]) is not bool for k in ("high_quality", "clean_foot_only"))
            or any(
                type(row[k]) not in (float, int) or not math.isfinite(row[k])
                for k in ("maximum_lateral_excursion_m", "minimum_pelvis_z_m")
            )
            or row["maximum_lateral_excursion_m"] < 0
        ):
            raise ValueError("finite measured booleans and physical risk values required")
    losses = dict(
        old_high_quality_loss=warm["high_quality"] and not candidate["high_quality"],
        old_clean_foot_loss=warm["clean_foot_only"] and not candidate["clean_foot_only"],
        new_out_of_play=warm["maximum_lateral_excursion_m"]
        <= 4
        < candidate["maximum_lateral_excursion_m"],
        safe_pelvis=candidate["minimum_pelvis_z_m"] >= 0.65,
    )
    return dict(
        **losses,
        known_counterexample_resolved=(
            losses["safe_pelvis"]
            and not any(
                losses[k]
                for k in ("old_high_quality_loss", "old_clean_foot_loss", "new_out_of_play")
            )
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "reference-root",
        "qualified-parent-bank",
        "candidate",
        "learning-commitment",
        "execution-source",
        "output-root",
        "isaac-python",
        "g1-usd",
        "model-root",
        "late-swing-policy",
        "core-root",
        "system-reserve-path",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    model = load_json_artifact(args.candidate)
    validate_model(model)
    learning = _sealed(args.learning_commitment)
    if (
        model["model_hash"] != MODEL
        or learning["report_hash"] != LEARNING
        or model["base_model"]["learning_receipt"]["physical_batch_hash"] != LEARNING
        or learning["existing_physical_rollouts_reused"] != 104
        or learning["new_physical_executions"] != 0
    ):
        raise ValueError("exact preregistered completed risk-margin learner required")
    reference = load_json_artifact(args.reference_root / "commitment.json")
    bank, review = (
        _sealed(args.qualified_parent_bank / name)
        for name in ("validation_summary.json", "independent_review.json")
    )
    qualified_memory_failure_rows(bank, review)
    runner = args.execution_source / "scripts/rsi_isaac_vector_first_touch.py"
    if (
        bank["report_hash"] != reference["parent_bank_hash"]
        or bank["commitment"]["model_hash"] != model["base_model"]["frozen_parent"]["model_hash"]
        or reference["source_commit"] != _head(args.execution_source)
        or reference["runner_hash"] != hash_bytes(runner.read_bytes())
        or reference["core_commit"] != _head(args.core_root)
        or reference["asset_hash"] != hash_bytes(args.g1_usd.read_bytes())
    ):
        raise ValueError("qualified frozen physical controls must retain exact source and body")
    seed, lane = COURSE
    stem = f"seed{seed}-lane{lane}"
    parent_path = args.reference_root / f"{stem}-reproduction-parent/report.json"
    parent = _sealed(parent_path)
    warm_folder = args.reference_root / f"{stem}-warm-actor"
    warm = _sealed(warm_folder / "report.json")
    if (
        (warm["training_course_seed"], warm["single_course_lane"]) != COURSE
        or warm["parent_report_hash"] != parent["report_hash"]
        or warm["contact_motor_policy"]["step_motor_proof"]["model"]
        != model["base_model"]["frozen_parent"]
    ):
        raise ValueError("exact fixed known out-of-play control required")
    baseline = _outcome(warm_folder, warm["contact_motor_policy_hash"], reference)["outcome"]
    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    capacity = capacity_check(args.output_root.parent, args.system_reserve_path, 512 * 1024**2)
    inputs = [
        args.candidate,
        args.learning_commitment,
        args.reference_root / "commitment.json",
        args.qualified_parent_bank / "validation_summary.json",
        args.qualified_parent_bank / "independent_review.json",
        runner,
        args.g1_usd,
        args.late_swing_policy,
    ]
    pins = {str(p.resolve()): hash_bytes(p.read_bytes()) for p in inputs}
    commitment = dict(
        schema="soccer.rsi.risk_margin_out_retest_commitment.v1",
        course=list(COURSE),
        runner_hash=reference["runner_hash"],
        asset_hash=reference["asset_hash"],
        execution_source_commit=_head(args.execution_source),
        core_commit=_head(args.core_root),
        model_hash=MODEL,
        input_hashes=pins,
        capacity=capacity,
        parent_report_hash=parent["report_hash"],
        qualified_nn_report_hash=warm["report_hash"],
        new_physical_executions_planned=1,
        existing_controls_consumed=2,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    args.output_root.mkdir(exist_ok=False)
    (args.output_root / "logs").mkdir()
    write_once(args.output_root / "commitment.json", commitment)
    raw, _ = _run(
        root=args.output_root,
        runner=runner,
        isaac_python=args.isaac_python,
        g1_usd=args.g1_usd,
        model_root=args.model_root,
        actor=args.late_swing_policy,
        seed=seed,
        lane=lane,
        gpu=3,
        gain=1.2,
        negative_only=True,
        core_root=args.core_root,
        arm="risk-margin",
        kind="actor",
        motor_step=args.candidate,
        parent_report_override=parent_path,
        compressed_report=True,
        execution_timeout_s=600,
    )
    folder = args.output_root / f"{stem}-risk-margin-actor"
    outcome = _outcome(folder, raw["contact_motor_policy_hash"], commitment)["outcome"]
    if (
        any(hash_bytes(p.read_bytes()) != pins[str(p.resolve())] for p in inputs)
        or _sealed(parent_path) != parent
        or _sealed(warm_folder / "report.json") != warm
        or raw["parent_report_hash"] != parent["report_hash"]
        or raw["contact_motor_policy"]["step_motor_proof"]["model"]["model_hash"] != MODEL
    ):
        raise ValueError("retest model/control/source drift")
    result = dict(
        schema="soccer.rsi.risk_margin_out_retest_review.v1",
        commitment_hash=hash_json(commitment),
        model_hash=MODEL,
        course=list(COURSE),
        new_physical_executions=1,
        existing_controls_consumed=2,
        motor_frames_reconstructed=600,
        reference_outcome=baseline,
        candidate_outcome=outcome,
        **risk_gate(baseline, outcome),
        qualification="FIXED_COUNTEREXAMPLE_ONLY_NOT_FULL_BANK_OR_FRESH",
        promotion_authorized=False,
        hardware_authorized=False,
    )
    result["report_hash"] = hash_json(result)
    write_once(args.output_root / "independent_review.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
