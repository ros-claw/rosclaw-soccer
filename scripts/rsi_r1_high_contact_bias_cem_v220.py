"""SIM_ONLY contact-rich search for one reusable high-zone neural motor skill."""

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
from rsi_r1_composed_contact_fresh_v214 import FRESH_COURSES as V214_COURSES
from rsi_r1_measured_skill_router_v207 import FRESH_COURSES as V207_COURSES
from rsi_r1_measured_skill_router_v207 import _score, clean
from rsi_r1_protected_kinematic_fresh_v202 import _load_policy
from rsi_r1_protected_neural_fresh_v217 import FRESH_COURSES as V217_COURSES
from rsi_r1_protected_online_ppo_v218 import _run_task

from rosclaw_soccer.rsi.receiving_foot_phase_router import ReceivingFootPhaseReference
from rosclaw_soccer.rsi.receiving_measured_skill_router import ReceivingSkillKnot
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

SCHEMA = "rosclaw_soccer.rsi.r1_high_contact_bias_cem_v220.result.v1"
SEED = 220929
GENERATIONS = 3
POPULATION = 16
TRAIN_SEEDS = (214002, 217002, 219003)


def _rank(rows: list[dict[str, Any]]) -> tuple[int, int, float]:
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
    v217_report_path: Path,
    v219_report_path: Path,
    output: Path,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY contact training evidence required")
    parent, right_parent, refine, lateral, v205, mapping, v214, v215, v216, v217, v219 = (
        _checked(path)
        for path in (
            parent_report,
            right_report,
            refine_report,
            lateral_report,
            v205_dir / "report.json",
            map_report_path,
            v214_report_path,
            v215_dir / "report.json",
            v216_report_path,
            v217_report_path,
            v219_report_path,
        )
    )
    if (
        v219["status"] != "REJECTED_GROWING_SKILL_BANK_FRESH_GATE"
        or v219["fresh_candidate_score"][:2] != [5, 7]
        or v217["v216_report_hash"] != v216["report_hash"]
        or v216["v215_report_hash"] != v215["report_hash"]
        or v205["map_report_hash"] != mapping["report_hash"]
        or parent["policy_hash"] != hash_bytes(policy_path.read_bytes())
    ):
        raise ValueError("sealed high-contact failure lineage required")
    weights = _load_policy(
        v215_dir / "update-3.npz",
        next(row for row in v215["history"] if row["update"] == 3)["checkpoint_hash"],
    )
    knots = tuple(
        ReceivingSkillKnot(
            float(row["lateral_m"]),
            float(row["closing_speed_mps"]),
            row["expert"],
            tuple(float(value) for value in row["weights"]),
        )
        for row in v214["knots"]
    )
    references = []
    for index, row in enumerate(v205["rows"]):
        if row["expert"] != "high":
            continue
        path = v205_dir / f"teacher-{index}.npz"
        if hash_bytes(path.read_bytes()) != row["feature_hash"]:
            raise ValueError("sealed physical high teacher required")
        with np.load(path, allow_pickle=False) as arrays:
            features = np.asarray(arrays["features"], dtype=np.float64)
        historical = next(
            item["summary"]
            for item in mapping["rows"]
            if item["expert"] == "high"
            and item["summary"]["course"]["seed"] == row["course"]["seed"]
        )
        references.append(
            ReceivingFootPhaseReference(
                "high",
                float(features[0, 1] * 0.2),
                float(features[0, 3] * 2.0),
                int(historical["first_foot_frame"]),
                tuple(tuple(float(value) for value in frame) for frame in features),
            )
        )
    protected = tuple(
        tuple(float(value) for value in row) for row in v216["protected_initial_features"]
    )
    common = (
        asset_root,
        policy_path,
        tuple(parent["selected"]["weights"]),
        tuple(refine["best"]["left_weights"]),
        tuple(float(value) for value in np.asarray(right_parent["best_right"]["weights"]) * 0.7),
        tuple(lateral["best"]["slope"]),
        knots,
        tuple(references),
        tuple(float(value) for value in v214["low_weights"]),
    )
    courses = (*V207_COURSES, *V214_COURSES, *V217_COURSES)
    courses += tuple(type(V217_COURSES[0])(**row["course"]) for row in v219["fresh_candidate"])
    selected = tuple(course for course in courses if course.seed in TRAIN_SEEDS)
    if len(selected) != 3 or len({course.seed for course in selected}) != 3:
        raise ValueError("three fixed consumed high-contact courses required")

    def tasks(course_bank: tuple[Any, ...], candidate: Any) -> list[tuple[Any, ...]]:
        return [
            (
                common[0],
                common[1],
                course,
                *common[2:],
                candidate,
                protected,
                0,
                0.0,
            )
            for course in course_bank
        ]

    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_high_contact_bias_cem_v220.py",
            "scripts/rsi_r1_protected_online_ppo_v218.py",
            "src/rosclaw_soccer/rsi/receiving_protected_composed_neural.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    output.mkdir(parents=True)
    rng = np.random.default_rng(SEED)
    center = np.zeros(12, dtype=np.float64)
    sigma = 0.35
    base_bias = np.asarray(weights.output_bias)
    history = []
    with ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        baseline = list(pool.map(_run_task, tasks(selected, weights)))
        if (
            not clean(baseline[0])
            or not baseline[0]["controlled_reception"]
            or clean(baseline[1])
            or clean(baseline[2])
        ):
            raise ValueError("expected one retained success and two nonfoot failures")
        for generation in range(GENERATIONS):
            offsets = [center.copy()]
            offsets.extend(
                np.clip(center + rng.normal(0.0, sigma, 12), -1.2, 1.2)
                for _ in range(POPULATION - 1)
            )
            candidate_rows = []
            for offset in offsets:
                candidate = replace(
                    weights,
                    output_bias=tuple(float(value) for value in base_bias + offset),
                )
                candidate_rows.append((offset, candidate))
            flat_tasks = [
                task for _, candidate in candidate_rows for task in tasks(selected, candidate)
            ]
            flat = list(pool.map(_run_task, flat_tasks))
            ranked = []
            for index, (offset, candidate) in enumerate(candidate_rows):
                rows = flat[index * 3 : index * 3 + 3]
                ranked.append(
                    {
                        "offset": offset,
                        "weights": candidate,
                        "rows": rows,
                        "rank": _rank(rows),
                    }
                )
            ranked.sort(key=lambda row: row["rank"], reverse=True)
            elites = ranked[:4]
            center = np.mean([row["offset"] for row in elites], axis=0)
            sigma = max(0.08, sigma * 0.65)
            history.append(
                {
                    "generation": generation + 1,
                    "best_offset": [float(value) for value in ranked[0]["offset"]],
                    "best_rank": ranked[0]["rank"],
                    "best_rows": ranked[0]["rows"],
                    "population_ranks": [row["rank"] for row in ranked],
                }
            )
            print(
                json.dumps({"generation": generation + 1, "best": ranked[0]["rank"]}),
                flush=True,
            )
    best = max(history, key=lambda row: tuple(row["best_rank"]))
    qualified = best["best_rank"][0] >= 2 and best["best_rank"][1] == 3
    report = {
        "schema": SCHEMA,
        "v219_report_hash": v219["report_hash"],
        "source_hashes": sources,
        "partition": "CONSUMED_THREE_HIGH_CONTACT_COURSES_CEM_ONLY",
        "seed": SEED,
        "train_seeds": TRAIN_SEEDS,
        "baseline": baseline,
        "baseline_score": _score(baseline),
        "history": history,
        "best_generation": best["generation"],
        "status": "DEVELOPMENT_HIGH_CONTACT_CANDIDATE_ONLY"
        if qualified
        else "REJECTED_HIGH_CONTACT_CEM_GATE",
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during high contact training")
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
        "v217-report",
        "v219-report",
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
        args.v217_report,
        args.v219_report,
        args.output,
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("status", "baseline_score", "best_generation", "report_hash")
            }
        )
    )


if __name__ == "__main__":
    main()
