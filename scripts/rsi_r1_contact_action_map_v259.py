"""Map retained contact-residual actions across a continuous consumed course grid."""

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
from rsi_r1_measured_skill_router_v207 import _score, clean
from rsi_r1_protected_kinematic_fresh_v202 import _load_policy
from rsi_r1_protection_radius_development_v254 import _run_protected
from rsi_r1_velocity_gated_fresh_v238 import ANCHOR

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses
from rosclaw_soccer.training.role_receiving_courses import ReceivingCourse

GRID = tuple(
    ReceivingCourse("red.finisher", 259001 + i * 4 + j, speed, lateral)
    for i, lateral in enumerate((0.0649, 0.0653, 0.0659))
    for j, speed in enumerate((1.16, 1.175, 1.19, 1.205))
)
ACTION_IDS = (
    (0, 0),
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (0, 5),
    (0, 6),
    (0, 7),
    (0, 8),
    (0, 9),
    (0, 10),
    (0, 11),
    (1, 3),
    (1, 9),
    (1, 11),
)


def map_actions(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    output = args.output
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external consumed action-map evidence required")
    v233 = _checked(args.v233_dir / "report.json")
    v236 = _checked(args.v236_dir / "report.json")
    v237 = _checked(args.v237_dir / "report.json")
    v254 = _checked(args.v254_dir / "report.json")
    v255 = _checked(args.v255_dir / "report.json")
    v256 = _checked(args.v256_dir / "report.json")
    v257 = _checked(args.v257_dir / "report.json")
    v258 = _checked(args.v258_dir / "report.json")
    v224 = _checked(args.v224_report)
    v229 = _checked(args.v229_dir / "report.json")
    if (
        v258["status"] != "REJECTED_BAND_REFINEMENT_FRESH_GATE"
        or v258["v257_report_hash"] != v257["report_hash"]
        or v257["v256_report_hash"] != v256["report_hash"]
        or v256["v255_report_hash"] != v255["report_hash"]
        or v255["v254_report_hash"] != v254["report_hash"]
        or v237["v236_report_hash"] != v236["report_hash"]
    ):
        raise ValueError("sealed local-only action failure lineage required")
    grid_hash = preflight_receiving_courses(GRID)
    records = {(row["generation"], row["index"]): row for row in v256["records"]}
    if len(records) != 36 or any(action_id not in records for action_id in ACTION_IDS):
        raise ValueError("complete sealed bounded action population required")
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
    learned_bias = base_bias + np.asarray(v236["best_offset"])
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

    def jobs(offset: tuple[float, ...]) -> list[tuple[Any, ...]]:
        weights = replace(
            base_weights,
            output_bias=tuple(float(x) for x in learned_bias + np.asarray(offset)),
        )
        return [
            (
                0.002,
                threshold,
                (
                    common[0],
                    common[1],
                    course,
                    *common[2:],
                    pre_weights,
                    post_weights,
                    weights,
                    states,
                ),
            )
            for course in GRID
        ]

    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_contact_action_map_v259.py",
            "scripts/rsi_r1_near_contact_residual_train_v256.py",
            "src/rosclaw_soccer/rsi/receiving_lateral_velocity_gated_precontact_motor.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
        )
    }
    output.mkdir(parents=True)
    action_rows = []
    with ProcessPoolExecutor(
        max_workers=8, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        parent = list(pool.map(_run_protected, jobs((0.0,) * 12)))
        for action_id in ACTION_IDS:
            offset = tuple(float(value) for value in records[action_id]["offset"])
            rows = list(pool.map(_run_protected, jobs(offset)))
            new_wins = [
                row["course"]["seed"]
                for old, row in zip(parent, rows, strict=True)
                if not (clean(old) and old["controlled_reception"])
                and clean(row)
                and row["controlled_reception"]
            ]
            lost_wins = [
                row["course"]["seed"]
                for old, row in zip(parent, rows, strict=True)
                if clean(old)
                and old["controlled_reception"]
                and not (clean(row) and row["controlled_reception"])
            ]
            action_rows.append(
                {
                    "action_id": list(action_id),
                    "offset": list(offset),
                    "score": _score(rows),
                    "new_wins": new_wins,
                    "lost_wins": lost_wins,
                    "rows": rows,
                }
            )
            print(
                json.dumps({"action_id": action_id, "new_wins": new_wins, "lost_wins": lost_wins}),
                flush=True,
            )
    positive_courses = sorted({seed for action in action_rows for seed in action["new_wins"]})
    report = {
        "schema": "rosclaw_soccer.rsi.r1_contact_action_map_v259.v1",
        "v258_report_hash": v258["report_hash"],
        "source_hashes": sources,
        "partition": "NEW_CONSUMED_GRID_ACTION_MAPPING_NOT_FRESH",
        "grid_hash": grid_hash,
        "grid": [vars(course) for course in GRID],
        "action_ids": [list(action_id) for action_id in ACTION_IDS],
        "parent_score": _score(parent),
        "parent": parent,
        "actions": action_rows,
        "positive_course_seeds": positive_courses,
        "status": "ACTION_MAP_HAS_MULTIPLE_POSITIVE_COURSES"
        if len(positive_courses) >= 3
        else "ACTION_MAP_INSUFFICIENT_POSITIVE_COURSES",
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during contextual action mapping")
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
        "v254-dir",
        "v255-dir",
        "v256-dir",
        "v257-dir",
        "v258-dir",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    report = map_actions(parser.parse_args())
    print(
        json.dumps(
            {
                "status": report["status"],
                "parent_score": report["parent_score"],
                "positive_course_seeds": report["positive_course_seeds"],
                "report_hash": report["report_hash"],
            }
        )
    )


if __name__ == "__main__":
    main()
