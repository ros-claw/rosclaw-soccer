"""SIM_ONLY local refinement of a cross-course, retention-safe contact latent."""

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
from rsi_r1_latent_contact_exploration_v232 import _run_task
from rsi_r1_measured_skill_router_v207 import _score, clean
from rsi_r1_protected_kinematic_fresh_v202 import _load_policy

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.role_receiving_courses import ReceivingCourse

SCHEMA = "rosclaw_soccer.rsi.r1_coherent_latent_refine_v234.result.v1"
SEED = 234091
SAMPLES = 8
STDS = (0.08, 0.04)
ANCHOR = 158


def refine(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    output = args.output
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY latent refinement evidence required")
    v232 = _checked(args.v232_dir / "report.json")
    v233 = _checked(args.v233_dir / "report.json")
    v229 = _checked(args.v229_dir / "report.json")
    v224 = _checked(args.v224_report)
    if (
        v233["status"] != "REJECTED_COHERENT_TRANSFER_GATE"
        or v233["v232_report_hash"] != v232["report_hash"]
        or v232["v229_report_hash"] != v229["report_hash"]
        or v233["baseline_score"][:2] != [4, 9]
    ):
        raise ValueError("sealed coherent cross-course lineage required")
    anchor = next(row for row in v233["candidates"] if row["source_episode"] == ANCHOR)
    if not anchor["old_success_retained"] or anchor["score"][:2] != [5, 9]:
        raise ValueError("retention-safe local refinement anchor required")
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
        raise ValueError("sealed precontact motor lineage required")
    pre_weights = replace(
        base_weights,
        output_bias=tuple(
            float(x) for x in np.asarray(base_weights.output_bias) + np.asarray(v224["best_offset"])
        ),
    )
    post_weights = _load_policy(
        args.v229_dir / "update-1.npz", v229["history"][0]["checkpoint_hash"]
    )
    courses = tuple(ReceivingCourse(**row["course"]) for row in v232["course_summary"])
    old = v233["baseline"]
    base_success = {
        row["course"]["seed"] for row in old if clean(row) and row["controlled_reception"]
    }
    if len(courses) != 11 or len(base_success) != 4:
        raise ValueError("eleven-course sealed baseline required")

    def tasks(offset: np.ndarray) -> list[tuple[Any, ...]]:
        weights = replace(
            post_weights,
            output_bias=tuple(float(x) for x in np.asarray(post_weights.output_bias) + offset),
        )
        return [
            (common[0], common[1], course, *common[2:], pre_weights, weights, 0, 0.0)
            for course in courses
        ]

    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_coherent_latent_refine_v234.py",
            "scripts/rsi_r1_latent_contact_exploration_v232.py",
            "src/rosclaw_soccer/rsi/receiving_latent_phase_motor.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
        )
    }
    output.mkdir(parents=True)
    rng = np.random.default_rng(SEED)
    center = np.asarray(anchor["latent_offset"], dtype=np.float64)
    records: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=8, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        replay = list(pool.map(_run_task, tasks(center)))
        if any(
            new["physical_trace_hash"] != sealed["physical_trace_hash"]
            for new, sealed in zip(replay, anchor["rows"], strict=True)
        ):
            raise ValueError("anchor must physically replay sealed cross-course evidence")
        for generation, std in enumerate(STDS):
            offsets = [center.copy()]
            offsets.extend(
                np.clip(center + rng.normal(0.0, std, 12), -1.2, 1.2) for _ in range(SAMPLES)
            )
            generation_records = []
            for index, offset in enumerate(offsets):
                rows = list(pool.map(_run_task, tasks(offset)))
                wins = {
                    row["course"]["seed"]
                    for row in rows
                    if clean(row) and row["controlled_reception"]
                }
                retained = base_success <= wins
                score = _score(rows)
                record = {
                    "generation": generation,
                    "index": index,
                    "std": std,
                    "offset": offset.tolist(),
                    "score": score,
                    "old_success_retained": retained,
                    "wins": sorted(wins),
                    "rows": rows,
                }
                generation_records.append(record)
                records.append(record)
                print(
                    json.dumps(
                        {
                            "generation": generation,
                            "index": index,
                            "score": score[:2],
                            "retained": retained,
                        }
                    ),
                    flush=True,
                )
            eligible = [
                row
                for row in generation_records
                if row["old_success_retained"] and row["score"][1] >= 9
            ]
            center = np.asarray(
                max(eligible, key=lambda row: tuple(row["score"]))["offset"],
                dtype=np.float64,
            )
    qualified = [
        row
        for row in records
        if row["old_success_retained"] and row["score"][0] >= 6 and row["score"][1] >= 9
    ]
    best = max(qualified or records, key=lambda row: tuple(row["score"]))
    report = {
        "schema": SCHEMA,
        "v233_report_hash": v233["report_hash"],
        "source_hashes": sources,
        "partition": "CONSUMED_ELEVEN_HIGH_ZONE_COURSES_ONLY",
        "seed": SEED,
        "samples_per_generation": SAMPLES,
        "stds": STDS,
        "anchor_source_episode": ANCHOR,
        "anchor_replay_equal": True,
        "baseline_score": v233["baseline_score"],
        "anchor_score": anchor["score"],
        "records": records,
        "best_offset": best["offset"],
        "best_score": best["score"],
        "status": "DEVELOPMENT_RETENTION_SAFE_LATENT_ONLY"
        if qualified
        else "REJECTED_COHERENT_LATENT_REFINEMENT_GATE",
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during coherent latent refinement")
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
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    report = refine(parser.parse_args())
    print(json.dumps({key: report[key] for key in ("status", "best_score", "report_hash")}))


if __name__ == "__main__":
    main()
