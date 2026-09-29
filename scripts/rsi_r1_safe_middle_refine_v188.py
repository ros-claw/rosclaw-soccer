"""SIM_ONLY safety-first refinement of a consumed middle-course anchor."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import Any

import numpy as np
from rsi_r1_bounded_middle_fresh_v186 import EXAM_COURSES
from rsi_r1_middle_basis_cem_v187 import run
from rsi_r1_side_navigation_fresh_v160 import FRESH_COURSES, clean
from rsi_r1_taskspace_feedback_v125 import COURSES

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

SCHEMA = "rosclaw_soccer.rsi.r1_safe_middle_refine_v188.result.v1"
POPULATION = 8
GENERATIONS = 2


def rank(rows: list[dict[str, Any]]) -> tuple[float, ...]:
    clean_count = sum(clean(row) for row in rows)
    controlled = sum(clean(row) and row["controlled_reception"] for row in rows)
    excess = sum(
        max(0.0, row["tail_maximum_foot_distance_m"] - 0.35) / 0.35
        + max(0.0, row["tail_maximum_ball_speed_mps"] - 0.35) / 0.35
        for row in rows
    )
    return (
        float(clean_count == len(rows)),
        float(controlled),
        float(clean_count),
        -excess,
        -sum(row["tail_maximum_foot_distance_m"] for row in rows),
        -sum(row["tail_maximum_ball_speed_mps"] for row in rows),
    )


def evaluate(
    pool: ProcessPoolExecutor,
    asset_root: Path,
    policy: Path,
    coordination: tuple[float, ...],
    left: tuple[float, ...],
    right: tuple[float, ...],
    slope: tuple[float, ...],
    weights: tuple[float, ...],
) -> list[dict[str, Any]]:
    return list(
        pool.map(
            run,
            repeat(asset_root),
            repeat(policy),
            EXAM_COURSES,
            repeat(coordination),
            repeat(left),
            repeat(right),
            repeat(slope),
            repeat(weights),
        )
    )


def train(
    asset_root: Path,
    policy: Path,
    parent_report: Path,
    right_report: Path,
    refine_report: Path,
    lateral_report: Path,
    previous_report: Path,
    output: Path,
    *,
    seed: int,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root) or not 0 <= seed < 2**31:
        raise ValueError("new external SIM_ONLY safe-middle evidence required")
    parent, right_parent, refine, lateral, previous = (
        json.loads(path.read_text())
        for path in (
            parent_report,
            right_report,
            refine_report,
            lateral_report,
            previous_report,
        )
    )
    if any(
        item["report_hash"] != hash_json({k: v for k, v in item.items() if k != "report_hash"})
        for item in (parent, right_parent, refine, lateral, previous)
    ) or (
        previous["status"] != "REJECTED_MIDDLE_TRAINING_GATE"
        or not previous["zero_equal"]
        or previous["training_pass"] != 2
        or parent["policy_hash"] != hash_bytes(policy.read_bytes())
    ):
        raise ValueError("sealed failed middle training required")
    anchors = [
        row
        for generation in previous["generations"]
        for row in generation["rows"]
        if all(clean(summary) for summary in row["summaries"])
    ]
    if not anchors:
        raise ValueError("all-clean training anchor required")
    anchor = max(anchors, key=lambda item: rank(item["summaries"]))
    coordination = tuple(parent["selected"]["weights"])
    left = tuple(refine["best"]["left_weights"])
    right = tuple(
        float(v) for v in np.asarray(right_parent["best_right"]["weights"], dtype=np.float64) * 0.7
    )
    slope = tuple(lateral["best"]["slope"])
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_safe_middle_refine_v188.py",
            "scripts/rsi_r1_middle_basis_cem_v187.py",
            "src/rosclaw_soccer/rsi/receiving_middle_basis_expert.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    output.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    mean = np.asarray(anchor["middle_weights"], dtype=np.float64)
    std = np.full(12, 0.06)
    generations: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=4, mp_context=context) as pool:
        baseline = evaluate(
            pool,
            asset_root,
            policy,
            coordination,
            left,
            right,
            slope,
            tuple(float(value) for value in mean),
        )
        if any(
            row["physical_trace_hash"] != old["physical_trace_hash"]
            for row, old in zip(baseline, anchor["summaries"], strict=True)
        ):
            raise ValueError("all-clean anchor failed exact physical replay")
        for generation in range(GENERATIONS):
            candidates = np.clip(rng.normal(mean, std, size=(POPULATION, 12)), -1.0, 1.0)
            candidates[0] = mean
            rows: list[dict[str, Any]] = []
            for index, candidate in enumerate(candidates):
                weights = tuple(float(value) for value in candidate)
                summaries = evaluate(
                    pool, asset_root, policy, coordination, left, right, slope, weights
                )
                if any(row["active_substeps"] > 32 for row in summaries):
                    raise ValueError("contact compliance budget exceeded")
                item = {
                    "generation": generation + 1,
                    "candidate": index,
                    "middle_weights": weights,
                    "summaries": summaries,
                }
                rows.append(item)
                if best is None or rank(summaries) > rank(best["summaries"]):
                    best = item
                (output / "progress.json").write_text(
                    json.dumps(
                        [*generations, {"generation": generation + 1, "rows": rows}], indent=2
                    )
                    + "\n"
                )
                print(
                    json.dumps(
                        {
                            "generation": generation + 1,
                            "candidate": index,
                            "clean": sum(clean(row) for row in summaries),
                            "controlled": sum(
                                clean(row) and row["controlled_reception"] for row in summaries
                            ),
                            "distance": [row["tail_maximum_foot_distance_m"] for row in summaries],
                            "speed": [row["tail_maximum_ball_speed_mps"] for row in summaries],
                        }
                    ),
                    flush=True,
                )
            rows.sort(key=lambda item: rank(item["summaries"]), reverse=True)
            elite = np.stack([np.asarray(item["middle_weights"]) for item in rows[:3]])
            mean = elite.mean(axis=0)
            std = np.maximum(0.025, elite.std(axis=0))
            generations.append({"generation": generation + 1, "rows": rows})
        assert best is not None
        retention = [
            run(
                asset_root,
                policy,
                course,
                coordination,
                left,
                right,
                slope,
                tuple(best["middle_weights"]),
            )
            for course in (FRESH_COURSES[0], COURSES[0])
        ]
    retained = all(clean(row) and row["controlled_reception"] for row in retention)
    passed = sum(clean(row) and row["controlled_reception"] for row in best["summaries"])
    report = {
        "schema": SCHEMA,
        "previous_report_hash": previous["report_hash"],
        "source_hashes": sources,
        "partition": "CONSUMED_V186_SAFETY_FIRST_REFINEMENT",
        "seed": seed,
        "anchor": anchor,
        "anchor_replay_equal": True,
        "generations": generations,
        "best": best,
        "retention": retention,
        "retention_pass": retained,
        "training_pass": passed,
        "status": "DEVELOPMENT_MIDDLE_ALL_CONTROLLED_UNVALIDATED"
        if retained and passed == len(EXAM_COURSES)
        else "REJECTED_SAFE_MIDDLE_REFINEMENT_GATE",
        "fresh_exam_required": True,
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during safe-middle refinement")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--right-report", type=Path, required=True)
    parser.add_argument("--refine-report", type=Path, required=True)
    parser.add_argument("--lateral-report", type=Path, required=True)
    parser.add_argument("--previous-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=188930)
    args = parser.parse_args()
    report = train(
        args.asset_root,
        args.policy,
        args.parent_report,
        args.right_report,
        args.refine_report,
        args.lateral_report,
        args.previous_report,
        args.output,
        seed=args.seed,
    )
    print(json.dumps({"status": report["status"], "report_hash": report["report_hash"]}))


if __name__ == "__main__":
    main()
