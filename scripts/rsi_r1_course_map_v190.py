"""SIM_ONLY multi-expert receiving map over predeclared position-speed grid."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from rsi_r1_middle_basis_cem_v187 import run
from rsi_r1_side_navigation_fresh_v160 import clean

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses
from rosclaw_soccer.training.role_receiving_courses import ReceivingCourse

SCHEMA = "rosclaw_soccer.rsi.r1_course_map_v190.result.v1"
LATERALS_M = (0.062, 0.067, 0.073, 0.078)
SPEEDS_MPS = (1.15, 1.22, 1.28, 1.35)
COURSES = tuple(
    ReceivingCourse("red.finisher", 190001 + i * 4 + j, speed, lateral)
    for i, lateral in enumerate(LATERALS_M)
    for j, speed in enumerate(SPEEDS_MPS)
)
EXPERTS = ("parent", "low", "center", "high")


def _find_candidate(report: dict[str, Any], generation: int, index: int) -> dict[str, Any]:
    matches = [
        row
        for group in report["generations"]
        for row in group["rows"]
        if row["generation"] == generation and row["candidate"] == index
    ]
    if len(matches) != 1:
        raise ValueError("sealed specialist missing")
    return matches[0]


def _run_case(args: tuple[Any, ...]) -> dict[str, Any]:
    asset_root, policy, course, coordination, left, right, slope, weights, expert = args
    result = run(asset_root, policy, course, coordination, left, right, slope, weights)
    return {"expert": expert, "summary": result}


def map_courses(
    asset_root: Path,
    policy: Path,
    parent_report: Path,
    right_report: Path,
    refine_report: Path,
    lateral_report: Path,
    v187_report: Path,
    v188_report: Path,
    v189_report: Path,
    output: Path,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY course map required")
    bank_hash = preflight_receiving_courses(COURSES)
    parent, right_parent, refine, lateral, v187, v188, v189 = (
        json.loads(path.read_text())
        for path in (
            parent_report,
            right_report,
            refine_report,
            lateral_report,
            v187_report,
            v188_report,
            v189_report,
        )
    )
    if any(
        item["report_hash"] != hash_json({k: v for k, v in item.items() if k != "report_hash"})
        for item in (parent, right_parent, refine, lateral, v187, v188, v189)
    ) or (
        v189["status"] != "REJECTED_THREE_ZONE_FRESH_GATE"
        or v189["v187_report_hash"] != v187["report_hash"]
        or v189["v188_report_hash"] != v188["report_hash"]
        or parent["policy_hash"] != hash_bytes(policy.read_bytes())
    ):
        raise ValueError("sealed three-zone failure required")
    coordination = tuple(parent["selected"]["weights"])
    left = tuple(refine["best"]["left_weights"])
    right = tuple(
        float(v) for v in np.asarray(right_parent["best_right"]["weights"], dtype=np.float64) * 0.7
    )
    slope = tuple(lateral["best"]["slope"])
    weights = {
        "parent": (0.0,) * 12,
        "low": tuple(_find_candidate(v188, 1, 6)["middle_weights"]),
        "center": tuple(_find_candidate(v188, 2, 7)["middle_weights"]),
        "high": tuple(_find_candidate(v187, 2, 2)["middle_weights"]),
    }
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_course_map_v190.py",
            "scripts/rsi_r1_middle_basis_cem_v187.py",
            "src/rosclaw_soccer/rsi/receiving_middle_basis_expert.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    output.mkdir(parents=True)
    jobs = [
        (asset_root, policy, course, coordination, left, right, slope, weights[expert], expert)
        for course in COURSES
        for expert in EXPERTS
    ]
    rows: list[dict[str, Any]] = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=4, mp_context=context) as pool:
        for index, item in enumerate(pool.map(_run_case, jobs)):
            rows.append(item)
            (output / "progress.json").write_text(json.dumps(rows, indent=2) + "\n")
            if index % len(EXPERTS) == len(EXPERTS) - 1:
                block = rows[-len(EXPERTS) :]
                print(
                    json.dumps(
                        {
                            "course": vars(COURSES[index // len(EXPERTS)]),
                            "passes": {
                                row["expert"]: bool(
                                    clean(row["summary"]) and row["summary"]["controlled_reception"]
                                )
                                for row in block
                            },
                        }
                    ),
                    flush=True,
                )
    course_rows = [rows[i : i + len(EXPERTS)] for i in range(0, len(rows), len(EXPERTS))]
    coverage = {
        expert: sum(
            clean(row["summary"]) and row["summary"]["controlled_reception"]
            for row in rows
            if row["expert"] == expert
        )
        for expert in EXPERTS
    }
    oracle_coverage = sum(
        any(clean(row["summary"]) and row["summary"]["controlled_reception"] for row in block)
        for block in course_rows
    )
    report = {
        "schema": SCHEMA,
        "v189_report_hash": v189["report_hash"],
        "source_hashes": sources,
        "partition": "PREDECLARED_CONSUMED_DEVELOPMENT_GRID_NOT_FRESH_EXAM",
        "course_bank_hash": bank_hash,
        "courses": [vars(course) for course in COURSES],
        "expert_ids": EXPERTS,
        "rows": rows,
        "coverage": coverage,
        "oracle_coverage": oracle_coverage,
        "status": "INSUFFICIENT_SPECIALIST_COVERAGE"
        if oracle_coverage < 12
        else "DEVELOPMENT_SPECIALIST_COVERAGE_ONLY",
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during course mapping")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--right-report", type=Path, required=True)
    parser.add_argument("--refine-report", type=Path, required=True)
    parser.add_argument("--lateral-report", type=Path, required=True)
    parser.add_argument("--v187-report", type=Path, required=True)
    parser.add_argument("--v188-report", type=Path, required=True)
    parser.add_argument("--v189-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = map_courses(
        args.asset_root,
        args.policy,
        args.parent_report,
        args.right_report,
        args.refine_report,
        args.lateral_report,
        args.v187_report,
        args.v188_report,
        args.v189_report,
        args.output,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "coverage": report["coverage"],
                "oracle_coverage": report["oracle_coverage"],
                "report_hash": report["report_hash"],
            }
        )
    )


if __name__ == "__main__":
    main()
