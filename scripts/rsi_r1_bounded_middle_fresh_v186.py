"""SIM_ONLY frozen middle-course fresh exam of measured-lateral receiving."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from rsi_r1_compliant_piecewise_v174 import run
from rsi_r1_side_navigation_fresh_v160 import FRESH_COURSES, clean
from rsi_r1_taskspace_feedback_v125 import COURSES

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.role_receiving_courses import ReceivingCourse

SCHEMA = "rosclaw_soccer.rsi.r1_bounded_middle_fresh_v186.result.v1"
# New seeds, unseen lateral launches and two velocity shifts, frozen before running.
EXAM_COURSES = (
    ReceivingCourse("red.finisher", 186001, 1.25, 0.065),
    ReceivingCourse("red.finisher", 186002, 1.25, 0.075),
    ReceivingCourse("red.finisher", 186003, 1.20, 0.070),
    ReceivingCourse("red.finisher", 186004, 1.30, 0.070),
)


def exam(
    asset_root: Path,
    policy: Path,
    parent_report: Path,
    right_report: Path,
    refine_report: Path,
    lateral_report: Path,
    far_report: Path,
    output: Path,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY frozen fresh evidence required")
    parent, right_parent, refine, lateral, far = (
        json.loads(path.read_text())
        for path in (
            parent_report,
            right_report,
            refine_report,
            lateral_report,
            far_report,
        )
    )
    if any(
        item["report_hash"] != hash_json({k: v for k, v in item.items() if k != "report_hash"})
        for item in (parent, right_parent, refine, lateral, far)
    ) or (
        lateral["status"] != "DEVELOPMENT_TWO_COURSE_CONTROLLED_UNVALIDATED"
        or far["status"] != "REJECTED_FAR_BODY_SYNERGY_GATE"
        or parent["policy_hash"] != hash_bytes(policy.read_bytes())
    ):
        raise ValueError("sealed candidate and failed-far lineage required")
    coordination = tuple(parent["selected"]["weights"])
    left = tuple(refine["best"]["left_weights"])
    right = tuple(
        float(v) for v in np.asarray(right_parent["best_right"]["weights"], dtype=np.float64) * 0.7
    )
    slope = tuple(lateral["best"]["slope"])
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_bounded_middle_fresh_v186.py",
            "scripts/rsi_r1_compliant_piecewise_v174.py",
            "src/rosclaw_soccer/rsi/receiving_lateral_piecewise_expert.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    output.mkdir(parents=True)
    retention = []
    for course in (FRESH_COURSES[0], COURSES[0]):
        row = run(asset_root, policy, course, coordination, left, right, slope, (0.0,) * 12)
        retention.append(row)
    parent_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for course in EXAM_COURSES:
        old = run(asset_root, policy, course, coordination, left, right, (0.0,) * 12, (0.0,) * 12)
        new = run(asset_root, policy, course, coordination, left, right, slope, (0.0,) * 12)
        parent_rows.append(old)
        candidate_rows.append(new)
        (output / "progress.json").write_text(
            json.dumps({"parent": parent_rows, "candidate": candidate_rows}, indent=2) + "\n"
        )
        print(
            json.dumps(
                {
                    "course": vars(course),
                    "parent": clean(old) and old["controlled_reception"],
                    "candidate": clean(new) and new["controlled_reception"],
                    "candidate_nonfoot": new["own_nonfoot_frames"],
                    "candidate_distance": new["tail_maximum_foot_distance_m"],
                    "candidate_speed": new["tail_maximum_ball_speed_mps"],
                }
            ),
            flush=True,
        )
    retention_pass = all(clean(row) and row["controlled_reception"] for row in retention)
    parent_pass = sum(clean(row) and row["controlled_reception"] for row in parent_rows)
    candidate_pass = sum(clean(row) and row["controlled_reception"] for row in candidate_rows)
    qualified = (
        retention_pass and candidate_pass == len(EXAM_COURSES) and candidate_pass > parent_pass
    )
    report = {
        "schema": SCHEMA,
        "far_report_hash": far["report_hash"],
        "lateral_report_hash": lateral["report_hash"],
        "source_hashes": sources,
        "partition": "SEALED_NEW_MIDDLE_RECEIVING_HOLDOUT",
        "exam_courses": [vars(course) for course in EXAM_COURSES],
        "retention": retention,
        "retention_pass": retention_pass,
        "parent": parent_rows,
        "candidate": candidate_rows,
        "parent_pass": parent_pass,
        "candidate_pass": candidate_pass,
        "status": "QUALIFIED_BOUNDED_MIDDLE_LOCAL_ONLY"
        if qualified
        else "REJECTED_BOUNDED_MIDDLE_FRESH_GATE",
        "promotion_authorized": False,
        "team_video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during frozen fresh exam")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--right-report", type=Path, required=True)
    parser.add_argument("--refine-report", type=Path, required=True)
    parser.add_argument("--lateral-report", type=Path, required=True)
    parser.add_argument("--far-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = exam(
        args.asset_root,
        args.policy,
        args.parent_report,
        args.right_report,
        args.refine_report,
        args.lateral_report,
        args.far_report,
        args.output,
    )
    print(json.dumps({"status": report["status"], "report_hash": report["report_hash"]}))


if __name__ == "__main__":
    main()
