"""SIM_ONLY pre/post-contact foot feedback ablation on consumed cases."""

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

SCHEMA = "rosclaw_soccer.rsi.r1_foot_phase_post_v210.result.v1"
MULTIPLIERS = (0.0, 0.5, 1.0, 2.0)


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
    v209_report_path: Path,
    output: Path,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY phase ablation evidence required")
    parent, right_parent, refine, lateral, router, teachers, mapping, v209 = (
        _checked(path)
        for path in (
            parent_report,
            right_report,
            refine_report,
            lateral_report,
            router_report_path,
            teacher_dir / "report.json",
            map_report_path,
            v209_report_path,
        )
    )
    if (
        v209["status"] != "REJECTED_FOOT_PHASE_DEVELOPMENT_GATE"
        or v209["router_report_hash"] != router["report_hash"]
        or v209["scores"][1][0] != 1
        or teachers["map_report_hash"] != mapping["report_hash"]
        or parent["policy_hash"] != hash_bytes(policy_path.read_bytes())
    ):
        raise ValueError("sealed failed whole-phase study required")
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
            "scripts/rsi_r1_foot_phase_post_v210.py",
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
            multiplier,
        )
        for multiplier in MULTIPLIERS
        for course in FRESH_COURSES
    ]
    with ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        rows = list(pool.map(_run_task, tasks))
    blocks = [rows[i : i + 8] for i in range(0, len(rows), 8)]
    equal = all(
        row["physical_trace_hash"] == old["physical_trace_hash"]
        for row, old in zip(blocks[2], v209["rows"][1], strict=True)
    )
    if not equal:
        raise ValueError("post multiplier one must physically replay v209-r2")
    scores = [_score(block) for block in blocks]
    qualified = [
        multiplier
        for multiplier, block, score in zip(MULTIPLIERS, blocks, scores, strict=True)
        if score[0] >= 3
        and score[1] >= 7
        and all(row["safe"] and row["peak_correction_rad"] <= 0.0400001 for row in block)
    ]
    report = {
        "schema": SCHEMA,
        "v209_report_hash": v209["report_hash"],
        "router_report_hash": router["report_hash"],
        "source_hashes": sources,
        "partition": "CONSUMED_V207_FRESH_PHASE_ABLATION_NOT_NEW_EXAM",
        "multipliers": MULTIPLIERS,
        "rows": blocks,
        "scores": scores,
        "post_one_physical_equal": equal,
        "qualified_multipliers": qualified,
        "status": "DEVELOPMENT_POST_PHASE_CANDIDATE_ONLY"
        if qualified
        else "REJECTED_POST_PHASE_DEVELOPMENT_GATE",
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during phase ablation")
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
        "v209-report",
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
        args.v209_report,
        args.output,
    )
    print(json.dumps({key: report[key] for key in ("status", "scores", "report_hash")}))


if __name__ == "__main__":
    main()
