"""Train a bounded near-contact motor residual on already consumed failures."""

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
from rsi_r1_gated_nearside_fresh_v237 import TRAIN_SEEDS
from rsi_r1_measured_skill_router_v207 import _score, clean
from rsi_r1_protected_kinematic_fresh_v202 import _load_policy
from rsi_r1_protection_radius_development_v254 import _run_protected
from rsi_r1_velocity_gated_fresh_v238 import ANCHOR

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses
from rosclaw_soccer.training.role_receiving_courses import ReceivingCourse

SCHEMA = "rosclaw_soccer.rsi.r1_near_contact_residual_train_v256.v1"
SEED = 2560929
POPULATION = 12
SIGMAS = (0.16, 0.10, 0.06)
TRAIN_SEED_SET = {255002, 255004, 255007, 255008}
PROTECTION_RADIUS = 0.002


def rank(rows: list[dict[str, Any]]) -> tuple[int, int, int, float]:
    """Prioritize physical safety and clean foot contact before terminal control."""
    deficit = sum(
        max(0.0, row["tail_maximum_foot_distance_m"] - 0.35)
        + max(0.0, row["tail_maximum_ball_speed_mps"] - 0.35)
        + 0.1 * len(row["own_nonfoot_frames"])
        for row in rows
    )
    return (
        sum(row["safe"] for row in rows),
        sum(clean(row) for row in rows),
        sum(clean(row) and row["controlled_reception"] for row in rows),
        -float(deficit),
    )


def train(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    output = args.output
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external consumed-course training evidence required")
    v233 = _checked(args.v233_dir / "report.json")
    v236 = _checked(args.v236_dir / "report.json")
    v237 = _checked(args.v237_dir / "report.json")
    v238 = _checked(args.v238_dir / "report.json")
    v252 = _checked(args.v252_dir / "report.json")
    v254 = _checked(args.v254_dir / "report.json")
    v255 = _checked(args.v255_dir / "report.json")
    v224 = _checked(args.v224_report)
    v229 = _checked(args.v229_dir / "report.json")
    if (
        v255["status"] != "REJECTED_PROTECTED_SKILL_FRESH_GATE"
        or v255["v254_report_hash"] != v254["report_hash"]
        or v254["v253_report_hash"] != _checked(args.v253_dir / "report.json")["report_hash"]
        or v252["v251_report_hash"] != _checked(args.v251_dir / "report.json")["report_hash"]
        or v237["v236_report_hash"] != v236["report_hash"]
        or v255["fresh_candidate_score"][:2] != [4, 7]
    ):
        raise ValueError("sealed Fresh failure required for consumed-course training")
    old = tuple(ReceivingCourse(**row["course"]) for row in v254["rows"][2]["old"])
    consumed = tuple(ReceivingCourse(**row) for row in v238["fresh_courses"])
    v252_consumed = tuple(ReceivingCourse(**row) for row in v252["fresh_courses"])
    failed = tuple(ReceivingCourse(**row) for row in v255["fresh_courses"])
    train_courses = tuple(course for course in failed if course.seed in TRAIN_SEED_SET)
    if (len(old), len(consumed), len(v252_consumed), len(failed), len(train_courses)) != (
        11,
        8,
        8,
        8,
        4,
    ):
        raise ValueError("complete consumed development and failed courses required")
    bank_hash = preflight_receiving_courses(old + consumed + v252_consumed + failed)
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
        raise ValueError("sealed neural contact base differs")
    base_bias = np.asarray(base_weights.output_bias) + np.asarray(v224["best_offset"])
    pre_weights = replace(base_weights, output_bias=tuple(float(x) for x in base_bias))
    trained_bias = base_bias + np.asarray(v236["best_offset"])
    neural = _load_policy(args.v229_dir / "update-1.npz", v229["history"][0]["checkpoint_hash"])
    anchor = next(row for row in v233["candidates"] if row["source_episode"] == ANCHOR)
    post_weights = replace(
        neural,
        output_bias=tuple(
            float(x) for x in np.asarray(neural.output_bias) + np.asarray(anchor["latent_offset"])
        ),
    )
    states = tuple(tuple(float(x) for x in row) for row in v237["activation_states"])
    threshold = float(v254["learned_lateral_threshold"])

    def jobs(offset: np.ndarray, courses: tuple[ReceivingCourse, ...]) -> list[tuple[Any, ...]]:
        candidate_weights = replace(
            base_weights, output_bias=tuple(float(x) for x in trained_bias + offset)
        )
        return [
            (
                PROTECTION_RADIUS,
                threshold,
                (
                    common[0],
                    common[1],
                    course,
                    *common[2:],
                    pre_weights,
                    post_weights,
                    candidate_weights,
                    states,
                ),
            )
            for course in courses
        ]

    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_near_contact_residual_train_v256.py",
            "scripts/rsi_r1_protection_radius_development_v254.py",
            "src/rosclaw_soccer/rsi/receiving_lateral_velocity_gated_precontact_motor.py",
            "src/rosclaw_soccer/rsi/receiving_protected_composed_neural.py",
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
        baseline = list(pool.map(_run_protected, jobs(center, failed)))
        if any(
            row["physical_trace_hash"] != sealed["physical_trace_hash"]
            for row, sealed in zip(baseline, v255["fresh_candidate"], strict=True)
        ):
            raise ValueError("zero residual must replay sealed v255 failed candidate")
        for generation, sigma in enumerate(SIGMAS):
            offsets = [center.copy()]
            offsets.extend(
                np.clip(center + rng.normal(0.0, sigma, 12), -0.35, 0.35)
                for _ in range(POPULATION - 1)
            )
            generation_rows = []
            for index, offset in enumerate(offsets):
                rows = list(pool.map(_run_protected, jobs(offset, train_courses)))
                candidate = {
                    "generation": generation,
                    "index": index,
                    "sigma": sigma,
                    "offset": offset.tolist(),
                    "rank": rank(rows),
                    "rows": rows,
                }
                records.append(candidate)
                generation_rows.append(candidate)
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
        finalist_rows = []
        for record in sorted(records, key=lambda row: tuple(row["rank"]), reverse=True)[:3]:
            broad = list(
                pool.map(
                    _run_protected,
                    jobs(np.asarray(record["offset"]), old + consumed + v252_consumed + failed),
                )
            )
            old_rows, v238_rows, v252_rows, v255_rows = (
                broad[:11],
                broad[11:19],
                broad[19:27],
                broad[27:],
            )
            old_retained = all(
                not (clean(parent) and parent["controlled_reception"])
                or (clean(row) and row["controlled_reception"])
                for parent, row in zip(v254["rows"][2]["old"], old_rows, strict=True)
            )
            non_target_exact = all(
                parent["physical_trace_hash"] == row["physical_trace_hash"]
                for parent, row in zip(v254["rows"][2]["old"], old_rows, strict=True)
                if row["course"]["seed"] not in TRAIN_SEEDS
            )
            scores = [_score(rows) for rows in (old_rows, v238_rows, v252_rows, v255_rows)]
            passed = bool(
                old_retained
                and non_target_exact
                and scores[0][0] >= 7
                and scores[0][1] >= 9
                and scores[1][0] >= 5
                and scores[1][1] == 8
                and scores[2][0] >= 5
                and scores[2][1] == 8
                and scores[3][0] >= 5
                and scores[3][1] >= 7
            )
            finalist_rows.append(
                {
                    "generation": record["generation"],
                    "index": record["index"],
                    "offset": record["offset"],
                    "training_rank": record["rank"],
                    "scores": scores,
                    "old_retained": old_retained,
                    "non_target_exact": non_target_exact,
                    "development_gate": passed,
                    "rows": broad,
                }
            )
            print(
                json.dumps({"finalist": [record["generation"], record["index"]], "scores": scores}),
                flush=True,
            )
    qualified = next((row for row in finalist_rows if row["development_gate"]), None)
    report = {
        "schema": SCHEMA,
        "v255_report_hash": v255["report_hash"],
        "source_hashes": sources,
        "partition": "CONSUMED_FAILED_V255_TRAIN_AND_PRIOR_35_RETENTION_ONLY",
        "seed": SEED,
        "population": POPULATION,
        "sigmas": SIGMAS,
        "train_seed_set": sorted(TRAIN_SEED_SET),
        "development_bank_hash": bank_hash,
        "zero_replay_exact": True,
        "records": records,
        "finalists": finalist_rows,
        "selected_offset": qualified["offset"] if qualified else None,
        "status": "DEVELOPMENT_NEAR_CONTACT_RESIDUAL_FOR_FRESH_EXAM"
        if qualified
        else "REJECTED_NEAR_CONTACT_RESIDUAL_DEVELOPMENT_GATE",
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during near-contact residual training")
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
        "v233-dir",
        "v236-dir",
        "v237-dir",
        "v238-dir",
        "v251-dir",
        "v252-dir",
        "v253-dir",
        "v254-dir",
        "v255-dir",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    report = train(parser.parse_args())
    print(json.dumps({"status": report["status"], "report_hash": report["report_hash"]}))


if __name__ == "__main__":
    main()
