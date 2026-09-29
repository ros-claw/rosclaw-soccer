"""SIM_ONLY measured shin-clearance guard ablation after failed phase split."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from rsi_r1_coherent_exploration_v198 import _checked
from rsi_r1_foot_phase_development_v209 import _run_task
from rsi_r1_measured_skill_router_v207 import FRESH_COURSES, _score

from rosclaw_soccer.rsi.receiving_foot_phase_router import ReceivingFootPhaseReference
from rosclaw_soccer.rsi.receiving_measured_skill_router import ReceivingSkillKnot
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

SCHEMA = "rosclaw_soccer.rsi.r1_foot_phase_guard_v211.result.v1"
GUARDS_M = (0.0, 0.02, 0.04, 0.06)


def ablate(
    asset_root: Path,
    policy_path: Path,
    parent_report: Path,
    right_report: Path,
    refine_report: Path,
    lateral_report: Path,
    router_report_path: Path,
    teacher_dir: Path,
    map_report_path: Path,
    v210_report_path: Path,
    output: Path,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY shin guard evidence required")
    parent, right_parent, refine, lateral, router, teachers, mapping, v210 = (
        _checked(path)
        for path in (
            parent_report,
            right_report,
            refine_report,
            lateral_report,
            router_report_path,
            teacher_dir / "report.json",
            map_report_path,
            v210_report_path,
        )
    )
    if (
        v210["status"] != "REJECTED_POST_PHASE_DEVELOPMENT_GATE"
        or v210["router_report_hash"] != router["report_hash"]
        or v210["scores"][1][:2] != [2, 6]
        or teachers["map_report_hash"] != mapping["report_hash"]
        or parent["policy_hash"] != hash_bytes(policy_path.read_bytes())
    ):
        raise ValueError("sealed phase failure and genuine teachers required")
    knots = tuple(
        ReceivingSkillKnot(
            float(row["lateral_m"]),
            float(row["closing_speed_mps"]),
            row["expert"],
            tuple(float(value) for value in row["weights"]),
        )
        for row in router["skill_knots"]
    )
    references = []
    for index, row in enumerate(teachers["rows"]):
        teacher_path = teacher_dir / f"teacher-{index}.npz"
        if hash_bytes(teacher_path.read_bytes()) != row["feature_hash"]:
            raise ValueError("sealed teacher feature hash required")
        with np.load(teacher_path, allow_pickle=False) as arrays:
            features = np.asarray(arrays["features"], dtype=np.float64)
        historical = next(
            item["summary"]
            for item in mapping["rows"]
            if item["expert"] == row["expert"]
            and item["summary"]["course"]["seed"] == row["course"]["seed"]
        )
        references.append(
            ReceivingFootPhaseReference(
                row["expert"],
                float(features[0, 1] * 0.2),
                float(features[0, 3] * 2.0),
                int(historical["first_foot_frame"]),
                tuple(tuple(float(value) for value in frame) for frame in features),
            )
        )
    coordination = tuple(parent["selected"]["weights"])
    left = tuple(refine["best"]["left_weights"])
    right = tuple(float(value) for value in np.asarray(right_parent["best_right"]["weights"]) * 0.7)
    slope = tuple(lateral["best"]["slope"])
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_foot_phase_guard_v211.py",
            "scripts/rsi_r1_foot_phase_development_v209.py",
            "src/rosclaw_soccer/rsi/receiving_foot_phase_router.py",
            "src/rosclaw_soccer/rsi/receiving_measured_skill_router.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    output.mkdir(parents=True)
    tasks = [
        (
            asset_root,
            policy_path,
            course,
            coordination,
            left,
            right,
            slope,
            knots,
            tuple(references),
            0.3,
            0.5,
            guard,
        )
        for guard in GUARDS_M
        for course in FRESH_COURSES
    ]
    with ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        rows = list(pool.map(_run_task, tasks))
    blocks = [rows[i : i + 8] for i in range(0, len(rows), 8)]
    equal = all(
        row["physical_trace_hash"] == old["physical_trace_hash"]
        for row, old in zip(blocks[0], v210["rows"][1], strict=True)
    )
    if not equal:
        raise ValueError("zero shin guard must physically replay v210")
    scores = [_score(block) for block in blocks]
    qualified = [
        guard
        for guard, block, score in zip(GUARDS_M, blocks, scores, strict=True)
        if score[0] >= 3
        and score[1] >= 7
        and all(row["safe"] and row["peak_correction_rad"] <= 0.0400001 for row in block)
    ]
    report = {
        "schema": SCHEMA,
        "v210_report_hash": v210["report_hash"],
        "router_report_hash": router["report_hash"],
        "source_hashes": sources,
        "partition": "CONSUMED_V207_FRESH_SHIN_GUARD_ABLATION_NOT_NEW_EXAM",
        "guards_m": GUARDS_M,
        "rows": blocks,
        "scores": scores,
        "zero_guard_physical_equal": equal,
        "qualified_guards_m": qualified,
        "status": "DEVELOPMENT_SHIN_GUARD_CANDIDATE_ONLY"
        if qualified
        else "REJECTED_SHIN_GUARD_DEVELOPMENT_GATE",
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during shin guard ablation")
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
        "router-report",
        "teacher-dir",
        "map-report",
        "v210-report",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    report = ablate(
        args.asset_root,
        args.policy,
        args.parent_report,
        args.right_report,
        args.refine_report,
        args.lateral_report,
        args.router_report,
        args.teacher_dir,
        args.map_report,
        args.v210_report,
        args.output,
    )
    print(json.dumps({key: report[key] for key in ("status", "scores", "report_hash")}))


if __name__ == "__main__":
    main()
