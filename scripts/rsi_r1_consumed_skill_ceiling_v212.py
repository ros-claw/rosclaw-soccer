"""SIM_ONLY specialist coverage ceiling on the already-consumed v207 holdout."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from rsi_r1_coherent_exploration_v198 import _checked
from rsi_r1_course_map_v190 import EXPERTS, _find_candidate, _run_case
from rsi_r1_measured_skill_router_v207 import FRESH_COURSES
from rsi_r1_side_navigation_fresh_v160 import clean

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

SCHEMA = "rosclaw_soccer.rsi.r1_consumed_skill_ceiling_v212.result.v1"


def assess(
    asset_root: Path,
    policy_path: Path,
    parent_report: Path,
    right_report: Path,
    refine_report: Path,
    lateral_report: Path,
    v187_report: Path,
    v188_report: Path,
    router_report_path: Path,
    output: Path,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY consumed skill ceiling required")
    parent, right_parent, refine, lateral, v187, v188, router = (
        _checked(path)
        for path in (
            parent_report,
            right_report,
            refine_report,
            lateral_report,
            v187_report,
            v188_report,
            router_report_path,
        )
    )
    if (
        router["status"] != "REJECTED_MEASURED_SKILL_ROUTER_GATE"
        or len(router["fresh_candidate"]) != 8
        or parent["policy_hash"] != hash_bytes(policy_path.read_bytes())
    ):
        raise ValueError("sealed failed v207 eight-course bank required")
    coordination = tuple(parent["selected"]["weights"])
    left = tuple(refine["best"]["left_weights"])
    right = tuple(float(value) for value in np.asarray(right_parent["best_right"]["weights"]) * 0.7)
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
            "scripts/rsi_r1_consumed_skill_ceiling_v212.py",
            "scripts/rsi_r1_course_map_v190.py",
            "scripts/rsi_r1_middle_basis_cem_v187.py",
            "src/rosclaw_soccer/rsi/receiving_middle_basis_expert.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    output.mkdir(parents=True)
    jobs = [
        (asset_root, policy_path, course, coordination, left, right, slope, weights[expert], expert)
        for course in FRESH_COURSES
        for expert in EXPERTS
    ]
    with ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        rows = list(pool.map(_run_case, jobs))
    blocks = [rows[index : index + 4] for index in range(0, len(rows), 4)]
    selected_equal = all(
        next(item for item in block if item["expert"] == old["selected_expert"])["summary"][
            "physical_trace_hash"
        ]
        == old["physical_trace_hash"]
        for block, old in zip(blocks, router["fresh_candidate"], strict=True)
    )
    if not selected_equal:
        raise ValueError("specialist ceiling must replay each originally selected skill")
    coverage = {
        expert: sum(
            clean(item["summary"]) and item["summary"]["controlled_reception"]
            for item in rows
            if item["expert"] == expert
        )
        for expert in EXPERTS
    }
    oracle = sum(
        any(clean(item["summary"]) and item["summary"]["controlled_reception"] for item in block)
        for block in blocks
    )
    report = {
        "schema": SCHEMA,
        "router_report_hash": router["report_hash"],
        "source_hashes": sources,
        "partition": "CONSUMED_V207_FRESH_SPECIALIST_ORACLE_UPPER_BOUND_NOT_FRESH",
        "courses": [vars(course) for course in FRESH_COURSES],
        "expert_ids": EXPERTS,
        "rows": rows,
        "selected_physical_equal": selected_equal,
        "coverage": coverage,
        "oracle_coverage": oracle,
        "status": "INSUFFICIENT_MOTOR_SKILL_COVERAGE"
        if oracle < 6
        else "CONSUMED_ORACLE_COVERAGE_ONLY",
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during specialist ceiling assessment")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in (
        "asset-root",
        "policy",
        "parent-report",
        "right-report",
        "refine-report",
        "lateral-report",
        "v187-report",
        "v188-report",
        "router-report",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    report = assess(
        args.asset_root,
        args.policy,
        args.parent_report,
        args.right_report,
        args.refine_report,
        args.lateral_report,
        args.v187_report,
        args.v188_report,
        args.router_report,
        args.output,
    )
    print(
        json.dumps(
            {key: report[key] for key in ("status", "coverage", "oracle_coverage", "report_hash")}
        )
    )


if __name__ == "__main__":
    main()
