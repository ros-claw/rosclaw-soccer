"""SIM_ONLY broad development coverage map for state-conditioned skill training."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from rsi_r1_coherent_exploration_v198 import _checked
from rsi_r1_contact_manifold_v222 import context, tasks
from rsi_r1_measured_skill_router_v207 import _score, clean
from rsi_r1_protected_online_ppo_v218 import _run_task

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses
from rosclaw_soccer.training.role_receiving_courses import ReceivingCourse

SCHEMA = "rosclaw_soccer.rsi.r1_broad_contact_manifold_v223.result.v1"
COURSES = tuple(
    ReceivingCourse("red.finisher", 223001 + i * 7 + j, speed, lateral)
    for i, lateral in enumerate((0.063, 0.067, 0.071, 0.075, 0.079))
    for j, speed in enumerate((1.15, 1.20, 1.24, 1.28, 1.32, 1.36, 1.40))
)


def run(
    asset_root: Path,
    policy_path: Path,
    parent_report: Path,
    right_report: Path,
    refine_report: Path,
    lateral_report: Path,
    v205_dir: Path,
    map_report_path: Path,
    v214_report_path: Path,
    v215_dir: Path,
    v216_report_path: Path,
    v218_dir: Path,
    v220_report_path: Path,
    v222_report_path: Path,
    output: Path,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY broad development evidence required")
    previous = _checked(v222_report_path)
    if previous["status"] != "DEVELOPMENT_CONTACT_MANIFOLD_ONLY":
        raise ValueError("sealed contact coverage audit required")
    bank_hash = preflight_receiving_courses(COURSES)
    common, policies, lineage = context(
        asset_root,
        policy_path,
        parent_report,
        right_report,
        refine_report,
        lateral_report,
        v205_dir,
        map_report_path,
        v214_report_path,
        v215_dir,
        v216_report_path,
        v218_dir,
        v220_report_path,
    )
    if previous["v220_report_hash"] != lineage["v220_report_hash"]:
        raise ValueError("sealed motor skill lineage required")
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_broad_contact_manifold_v223.py",
            "scripts/rsi_r1_contact_manifold_v222.py",
            "scripts/rsi_r1_protected_online_ppo_v218.py",
            "src/rosclaw_soccer/rsi/receiving_protected_composed_neural.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    output.mkdir(parents=True)
    labels = ("baseline", "online", "high_contact")
    with ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        runs = {
            label: list(pool.map(_run_task, tasks(common, COURSES, policy)))
            for label, policy in zip(labels, policies, strict=True)
        }
    result_grid: list[dict[str, Any]] = []
    for index, course in enumerate(COURSES):
        success = [
            label
            for label in labels
            if clean(runs[label][index]) and runs[label][index]["controlled_reception"]
        ]
        result_grid.append(
            {
                "course": vars(course),
                "successful_policies": success,
                "clean_policies": [label for label in labels if clean(runs[label][index])],
                "first_foot_frames": {
                    label: runs[label][index]["first_foot_frame"] for label in labels
                },
            }
        )
    union_count = sum(bool(row["successful_policies"]) for row in result_grid)
    report = {
        "schema": SCHEMA,
        "v222_report_hash": previous["report_hash"],
        **lineage,
        "source_hashes": sources,
        "partition": "BROAD_DEVELOPMENT_MANIFOLD_NOT_FRESH_EXAM",
        "bank_hash": bank_hash,
        "courses": [vars(course) for course in COURSES],
        "rows": runs,
        "scores": {label: _score(rows) for label, rows in runs.items()},
        "outcome_grid": result_grid,
        "oracle_union_count": union_count,
        "all_failed_count": len(COURSES) - union_count,
        "status": "DEVELOPMENT_BROAD_CONTACT_MANIFOLD_ONLY",
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during broad manifold mapping")
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
        "v205-dir",
        "map-report",
        "v214-report",
        "v215-dir",
        "v216-report",
        "v218-dir",
        "v220-report",
        "v222-report",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    report = run(
        args.asset_root,
        args.policy,
        args.parent_report,
        args.right_report,
        args.refine_report,
        args.lateral_report,
        args.v205_dir,
        args.map_report,
        args.v214_report,
        args.v215_dir,
        args.v216_report,
        args.v218_dir,
        args.v220_report,
        args.v222_report,
        args.output,
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("scores", "oracle_union_count", "all_failed_count", "report_hash")
            }
        )
    )


if __name__ == "__main__":
    main()
