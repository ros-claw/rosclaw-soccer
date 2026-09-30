"""SIM_ONLY precontact motor curriculum for three hard near-side receiving courses."""

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
from rsi_r1_contact_manifold_v222 import context
from rsi_r1_foot_servo_contact_v235 import _run
from rsi_r1_measured_skill_router_v207 import _score, clean
from rsi_r1_protected_kinematic_fresh_v202 import _load_policy

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.role_receiving_courses import ReceivingCourse

SCHEMA = "rosclaw_soccer.rsi.r1_nearside_precontact_v236.result.v1"
SEED = 236929
TRAIN_SEEDS = (223001, 223002, 223003)
POPULATION = 16
SIGMAS = (0.32, 0.20, 0.12)
ANCHOR = 158


def rank(rows: list[dict[str, Any]]) -> tuple[int, int, float]:
    deficit = sum(
        max(0.0, row["tail_maximum_foot_distance_m"] - 0.35)
        + max(0.0, row["tail_maximum_ball_speed_mps"] - 0.35)
        + 0.1 * len(row["own_nonfoot_frames"])
        for row in rows
    )
    return (
        sum(clean(row) and row["controlled_reception"] for row in rows),
        sum(clean(row) for row in rows),
        -float(deficit),
    )


def train(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    output = args.output
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY near-side curriculum evidence required")
    v224, v229, v232, v233, v235 = (
        _checked(path)
        for path in (
            args.v224_report,
            args.v229_dir / "report.json",
            args.v232_dir / "report.json",
            args.v233_dir / "report.json",
            args.v235_dir / "report.json",
        )
    )
    if (
        v235["status"] != "REJECTED_FOOT_SERVO_CONTACT_GATE"
        or v235["v234_report_hash"] != _checked(args.v234_dir / "report.json")["report_hash"]
        or v233["v232_report_hash"] != v232["report_hash"]
    ):
        raise ValueError("sealed contact failure lineage required")
    anchor = next(row for row in v233["candidates"] if row["source_episode"] == ANCHOR)
    common, (base_weights, _, _), lineage = context(
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
    )
    if lineage["v220_report_hash"] != v224["v220_report_hash"]:
        raise ValueError("sealed precontact policy lineage required")
    base_bias = np.asarray(base_weights.output_bias) + np.asarray(v224["best_offset"])
    neural = _load_policy(args.v229_dir / "update-1.npz", v229["history"][0]["checkpoint_hash"])
    post_weights = replace(
        neural,
        output_bias=tuple(
            float(x) for x in np.asarray(neural.output_bias) + np.asarray(anchor["latent_offset"])
        ),
    )
    courses = tuple(ReceivingCourse(**row["course"]) for row in v232["course_summary"])
    train_courses = tuple(course for course in courses if course.seed in TRAIN_SEEDS)
    if tuple(course.seed for course in train_courses) != TRAIN_SEEDS:
        raise ValueError("three consumed near-side failure courses required")

    def tasks(
        offset: np.ndarray, course_bank: tuple[ReceivingCourse, ...]
    ) -> list[tuple[Any, ...]]:
        pre_weights = replace(base_weights, output_bias=tuple(float(x) for x in base_bias + offset))
        return [
            (common[0], common[1], course, *common[2:], pre_weights, post_weights, 0.0, 0.0)
            for course in course_bank
        ]

    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_nearside_precontact_v236.py",
            "scripts/rsi_r1_foot_servo_contact_v235.py",
            "src/rosclaw_soccer/rsi/receiving_foot_servo_phase_motor.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
        )
    }
    output.mkdir(parents=True)
    rng = np.random.default_rng(SEED)
    center = np.zeros(12, dtype=np.float64)
    records: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=8, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        baseline = list(pool.map(_run, tasks(center, courses)))
        if any(
            row["physical_trace_hash"] != old["physical_trace_hash"]
            for row, old in zip(baseline, v235["baseline"], strict=True)
        ):
            raise ValueError("zero precontact residual must replay sealed foot-servo parent")
        for generation, sigma in enumerate(SIGMAS):
            offsets = [center.copy()]
            offsets.extend(
                np.clip(center + rng.normal(0.0, sigma, 12), -1.2, 1.2)
                for _ in range(POPULATION - 1)
            )
            generation_rows = []
            for index, offset in enumerate(offsets):
                rows = list(pool.map(_run, tasks(offset, train_courses)))
                candidate = {
                    "generation": generation,
                    "index": index,
                    "sigma": sigma,
                    "offset": offset.tolist(),
                    "rank": rank(rows),
                    "rows": rows,
                }
                generation_rows.append(candidate)
                records.append(candidate)
                print(
                    json.dumps(
                        {
                            "generation": generation,
                            "index": index,
                            "rank": candidate["rank"],
                        }
                    ),
                    flush=True,
                )
            elite = sorted(generation_rows, key=lambda row: tuple(row["rank"]), reverse=True)[:4]
            center = np.mean([row["offset"] for row in elite], axis=0)
        best = max(records, key=lambda row: tuple(row["rank"]))
        broad = list(pool.map(_run, tasks(np.asarray(best["offset"]), courses)))
    base_wins = {
        row["course"]["seed"] for row in baseline if clean(row) and row["controlled_reception"]
    }
    broad_wins = {
        row["course"]["seed"] for row in broad if clean(row) and row["controlled_reception"]
    }
    qualified = (
        best["rank"][0] >= 1
        and base_wins <= broad_wins
        and _score(broad)[0] >= _score(baseline)[0] + 1
        and _score(broad)[1] >= _score(baseline)[1]
    )
    report = {
        "schema": SCHEMA,
        "v235_report_hash": v235["report_hash"],
        "source_hashes": sources,
        "partition": "CONSUMED_NEARSIDE_THREE_TRAIN_AND_ELEVEN_RETENTION_ONLY",
        "seed": SEED,
        "population": POPULATION,
        "sigmas": SIGMAS,
        "baseline_physical_equal": True,
        "baseline_score": _score(baseline),
        "train_baseline_rank": rank(baseline[:3]),
        "records": records,
        "best_offset": best["offset"],
        "best_train_rank": best["rank"],
        "broad_score": _score(broad),
        "broad_old_success_retained": base_wins <= broad_wins,
        "broad_rows": broad,
        "status": "DEVELOPMENT_NEARSIDE_PRECONTACT_CANDIDATE_ONLY"
        if qualified
        else "REJECTED_NEARSIDE_PRECONTACT_GATE",
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during near-side precontact training")
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
        "v224-report",
        "v229-dir",
        "v232-dir",
        "v233-dir",
        "v234-dir",
        "v235-dir",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    report = train(parser.parse_args())
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("status", "best_train_rank", "broad_score", "report_hash")
            }
        )
    )


if __name__ == "__main__":
    main()
