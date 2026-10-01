"""Independent full-trace review of consumed per-frame motor comparisons."""

import argparse
import json
from pathlib import Path
from typing import Any

from rosclaw_soccer.rsi.approach_lateral_tracking_evidence import audit_lateral_approach
from rosclaw_soccer.rsi.failure_curriculum_evidence import _outcome, _sealed
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_collect_step_motor_pilot import COURSES
from scripts.rsi_train_bilateral_contact_motor_v303 import write_once


def review(root: Path) -> dict[str, Any]:
    online = (root / "validation_summary.json").is_file()
    path = root / ("validation_summary.json" if online else "pilot_summary.json")
    summary = _sealed(path)
    expected_schema = (
        "soccer.rsi.online_step_motor_physical_validation.v1"
        if online
        else "soccer.rsi.step_motor_physical_pilot.v1"
    )
    commitment = summary["commitment"]
    if (
        summary["schema"] != expected_schema
        or summary["physical_executions"] != 12
        or summary["independent_contexts"] != 4
        or commitment["courses"] != [list(c) for c in COURSES]
        or summary["promotion_authorized"] is not False
        or summary["hardware_authorized"] is not False
        or commitment != json.loads((root / "commitment.json").read_text())
        or [(r["seed"], r["lane"]) for r in summary["rows"]] != list(COURSES)
    ):
        raise ValueError("complete sealed four-course consumed comparison required")
    baseline_arm = "warm" if online else "champion"
    candidate_arm = "online" if online else "step-neural"
    candidate_key = "online" if online else "neural"
    rows = []
    report_hashes = []
    for row in summary["rows"]:
        seed, lane = row["seed"], row["lane"]
        parent_folder = root / f"seed{seed}-lane{lane}-reproduction-parent"
        parent = _sealed(parent_folder / "report.json")
        audit_lateral_approach(parent_folder)
        if (
            parent["report_hash"] != row["parent_report_hash"]
            or parent["source_hash"] != commitment["runner_hash"]
            or parent["asset_hash"] != commitment["asset_hash"]
        ):
            raise ValueError("comparison parent identity or source changed")
        observed = {}
        for arm, key in ((baseline_arm, "baseline"), (candidate_arm, candidate_key)):
            folder = root / f"seed{seed}-lane{lane}-{arm}-actor"
            raw = _sealed(folder / "report.json")
            outcome = _outcome(folder, raw["contact_motor_policy_hash"], commitment)["outcome"]
            if (
                raw["report_hash"] != row[key]["report_hash"]
                or raw["parent_report_hash"] != parent["report_hash"]
                or raw["training_course_seed"] != seed
                or raw["single_course_lane"] != lane
                or raw["sonic_qualification_hash"] != parent["sonic_qualification_hash"]
                or raw["environments"][0]["course"] != parent["environments"][0]["course"]
                or any(outcome[k] != row[key][k] for k in outcome)
            ):
                raise ValueError("reported comparison differs from reconstructed physics")
            if (
                key == candidate_key
                and raw["contact_motor_policy"]["step_motor_proof"]["model"]["model_hash"]
                != commitment["model_hash"]
            ):
                raise ValueError("candidate motor model identity changed")
            observed[key] = outcome
            report_hashes.append(raw["report_hash"])
        rows.append(observed)
        report_hashes.append(parent["report_hash"])
    baseline_hq = sum(r["baseline"]["high_quality"] for r in rows)
    candidate_hq = sum(r[candidate_key]["high_quality"] for r in rows)
    counts = dict(
        baseline_high_quality=baseline_hq,
        candidate_high_quality=candidate_hq,
        old_high_quality_loss=sum(
            r["baseline"]["high_quality"] and not r[candidate_key]["high_quality"] for r in rows
        ),
        old_clean_foot_loss=sum(
            r["baseline"]["clean_foot_only"] and not r[candidate_key]["clean_foot_only"]
            for r in rows
        ),
        safe_pelvis=all(r[candidate_key]["minimum_pelvis_z_m"] >= 0.65 for r in rows),
        new_out_of_play=sum(
            r["baseline"]["maximum_lateral_excursion_m"]
            <= 4
            < r[candidate_key]["maximum_lateral_excursion_m"]
            for r in rows
        ),
    )
    if (
        summary["baseline_high_quality"] != baseline_hq
        or summary[candidate_key + "_high_quality"] != candidate_hq
    ):
        raise ValueError("reported high-quality count differs from actual execution")
    if online and any(
        summary[k] != counts[k]
        for k in counts
        if k not in ("candidate_high_quality", "baseline_high_quality")
    ):
        raise ValueError("reported retention or safety result differs from physics")
    result = dict(
        schema="soccer.rsi.step_motor_physics_independent_review.v1",
        source_summary_hash=summary["report_hash"],
        actual_reports_reviewed=12,
        actual_motor_actions_reconstructed=8 * 300,
        independent_contexts=4,
        reviewed_report_hashes=report_hashes,
        **counts,
        qualification="CONSUMED_COMPARISON_ONLY_NOT_FRESH_NOT_PROMOTION",
        promotion_authorized=False,
        hardware_authorized=False,
        source_hash=hash_bytes(Path(__file__).read_bytes()),
    )
    result["report_hash"] = hash_json(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = review(args.root)
    write_once(args.output, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
