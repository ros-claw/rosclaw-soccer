"""SIM_ONLY train a shared bounded motor residual over neighboring hard contacts."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from rsi_r1_coherent_exploration_v198 import _checked
from rsi_r1_contact_manifold_v222 import context, tasks
from rsi_r1_measured_skill_router_v207 import _score, clean
from rsi_r1_protected_online_ppo_v218 import _run_task

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.role_receiving_courses import ReceivingCourse

SCHEMA = "rosclaw_soccer.rsi.r1_neighbor_contact_cem_v224.result.v1"
SEED = 224929
COURSE_SEEDS = (223008, 223010, 223016, 223031)
GENERATIONS = 3
POPULATION = 16


def rank(rows: list[dict[str, Any]]) -> tuple[int, int, float]:
    successes = sum(clean(row) and row["controlled_reception"] for row in rows)
    clean_count = sum(clean(row) for row in rows)
    deficit = sum(
        max(0.0, row["tail_maximum_foot_distance_m"] - 0.35)
        + max(0.0, row["tail_maximum_ball_speed_mps"] - 0.35)
        + 0.1 * len(row["own_nonfoot_frames"])
        for row in rows
    )
    return successes, clean_count, -float(deficit)


def train(
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
    v223_report_path: Path,
    output: Path,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY neighboring-contact evidence required")
    previous = _checked(v223_report_path)
    if previous["status"] != "DEVELOPMENT_BROAD_CONTACT_MANIFOLD_ONLY":
        raise ValueError("sealed broad contact coverage required")
    common, (baseline_weights, _, _), lineage = context(
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
        raise ValueError("sealed baseline motor lineage required")
    courses = tuple(
        ReceivingCourse(**row["course"])
        for row in previous["outcome_grid"]
        if row["course"]["seed"] in COURSE_SEEDS
    )
    if tuple(course.seed for course in courses) != COURSE_SEEDS:
        raise ValueError("fixed neighboring contact cases required")
    old_rows = [
        row for row in previous["rows"]["baseline"] if row["course"]["seed"] in COURSE_SEEDS
    ]
    if [clean(row) and row["controlled_reception"] for row in old_rows] != [
        True,
        False,
        True,
        False,
    ]:
        raise ValueError("two old successes and two hard negatives required")
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_neighbor_contact_cem_v224.py",
            "scripts/rsi_r1_contact_manifold_v222.py",
            "scripts/rsi_r1_protected_online_ppo_v218.py",
            "src/rosclaw_soccer/rsi/receiving_protected_composed_neural.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    output.mkdir(parents=True)
    rng = np.random.default_rng(SEED)
    base_bias = np.asarray(baseline_weights.output_bias)
    center = np.zeros(12, dtype=np.float64)
    sigma = 0.35
    history = []
    with ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        baseline = list(pool.map(_run_task, tasks(common, courses, baseline_weights)))
        if any(
            row["physical_trace_hash"] != old["physical_trace_hash"]
            for row, old in zip(baseline, old_rows, strict=True)
        ):
            raise ValueError("four neighboring anchors must physically reproduce")
        for generation in range(GENERATIONS):
            offsets = [center.copy()]
            offsets.extend(
                np.clip(center + rng.normal(0.0, sigma, 12), -1.2, 1.2)
                for _ in range(POPULATION - 1)
            )
            policies = [
                replace(
                    baseline_weights,
                    output_bias=tuple(float(value) for value in base_bias + offset),
                )
                for offset in offsets
            ]
            flat = list(
                pool.map(
                    _run_task,
                    [task for policy in policies for task in tasks(common, courses, policy)],
                )
            )
            population = [
                {
                    "offset": [float(value) for value in offsets[index]],
                    "rows": flat[index * 4 : index * 4 + 4],
                    "rank": rank(flat[index * 4 : index * 4 + 4]),
                }
                for index in range(POPULATION)
            ]
            population.sort(key=lambda row: tuple(row["rank"]), reverse=True)
            center = np.mean([row["offset"] for row in population[:4]], axis=0)
            sigma = max(0.08, sigma * 0.65)
            history.append({"generation": generation + 1, "population": population})
            print(
                json.dumps({"generation": generation + 1, "best": population[0]["rank"]}),
                flush=True,
            )
    best = max(
        (row for generation in history for row in generation["population"]),
        key=lambda row: tuple(row["rank"]),
    )
    qualified = best["rank"][0] >= 3 and best["rank"][1] == 4
    report = {
        "schema": SCHEMA,
        "v223_report_hash": previous["report_hash"],
        **lineage,
        "source_hashes": sources,
        "partition": "CONSUMED_FOUR_NEIGHBOR_CONTACT_COURSES_CEM_ONLY",
        "seed": SEED,
        "course_seeds": COURSE_SEEDS,
        "baseline": baseline,
        "baseline_score": _score(baseline),
        "history": history,
        "best_offset": best["offset"],
        "best_rank": best["rank"],
        "best_rows": best["rows"],
        "status": "DEVELOPMENT_NEIGHBOR_CONTACT_CANDIDATE_ONLY"
        if qualified
        else "REJECTED_NEIGHBOR_CONTACT_CEM_GATE",
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during neighboring contact training")
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
        "v223-report",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    report = train(
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
        args.v223_report,
        args.output,
    )
    print(
        json.dumps(
            {key: report[key] for key in ("status", "baseline_score", "best_rank", "report_hash")}
        )
    )


if __name__ == "__main__":
    main()
