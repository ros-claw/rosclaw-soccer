"""Untouched paired receiving exam after protected-muscle-memory development."""

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

PARENT_PROTECTION = 0.003
CANDIDATE_PROTECTION = 0.002
FRESH = tuple(
    ReceivingCourse("red.finisher", 255001 + i * 4 + j, speed, lateral)
    for i, lateral in enumerate((0.0654, 0.0658))
    for j, speed in enumerate((1.16, 1.18, 1.20, 1.22))
)


def exam(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    output = args.output
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external fresh receiving evidence required")
    v233 = _checked(args.v233_dir / "report.json")
    v236 = _checked(args.v236_dir / "report.json")
    v237 = _checked(args.v237_dir / "report.json")
    v238 = _checked(args.v238_dir / "report.json")
    v251 = _checked(args.v251_dir / "report.json")
    v252 = _checked(args.v252_dir / "report.json")
    v253 = _checked(args.v253_dir / "report.json")
    v254 = _checked(args.v254_dir / "report.json")
    v224 = _checked(args.v224_report)
    v229 = _checked(args.v229_dir / "report.json")
    selected = next(row for row in v254["rows"] if row["protection_radius"] == 0.002)
    if (
        v254["status"] != "DEVELOPMENT_PROTECTION_CANDIDATE_FOR_FRESH_EXAM"
        or v254["v253_report_hash"] != v253["report_hash"]
        or v253["v252_report_hash"] != v252["report_hash"]
        or v252["v251_report_hash"] != v251["report_hash"]
        or v251["v238_report_hash"] != v238["report_hash"]
        or v237["v236_report_hash"] != v236["report_hash"]
        or not selected["development_gate"]
        or selected["old_score"][:2] != [7, 9]
        or selected["consumed_score"][:2] != [5, 8]
        or selected["v252_consumed_score"][:2] != [5, 8]
    ):
        raise ValueError("sealed protected development success required before Fresh")
    fresh_hash = preflight_receiving_courses(FRESH)
    consumed_seeds = {
        row["course"]["seed"]
        for bank in ("old", "consumed", "v252_consumed")
        for row in selected[bank]
    }
    if consumed_seeds & {course.seed for course in FRESH}:
        raise ValueError("new Fresh course seeds must be disjoint")
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
        raise ValueError("sealed precontact base differs")
    base_bias = np.asarray(base_weights.output_bias) + np.asarray(v224["best_offset"])
    pre_weights = replace(base_weights, output_bias=tuple(float(x) for x in base_bias))
    candidate_weights = replace(
        base_weights,
        output_bias=tuple(float(x) for x in base_bias + np.asarray(v236["best_offset"])),
    )
    neural = _load_policy(args.v229_dir / "update-1.npz", v229["history"][0]["checkpoint_hash"])
    anchor = next(row for row in v233["candidates"] if row["source_episode"] == ANCHOR)
    post_weights = replace(
        neural,
        output_bias=tuple(
            float(x) for x in np.asarray(neural.output_bias) + np.asarray(anchor["latent_offset"])
        ),
    )
    states = tuple(tuple(float(x) for x in row) for row in v237["activation_states"])

    def task(course: ReceivingCourse) -> tuple[Any, ...]:
        return (
            common[0],
            common[1],
            course,
            *common[2:],
            pre_weights,
            post_weights,
            candidate_weights,
            states,
        )

    learned_threshold = float(v254["learned_lateral_threshold"])
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_protection_fresh_v255.py",
            "scripts/rsi_r1_protection_radius_development_v254.py",
            "src/rosclaw_soccer/rsi/receiving_lateral_velocity_gated_precontact_motor.py",
            "src/rosclaw_soccer/rsi/receiving_protected_composed_neural.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
        )
    }
    output.mkdir(parents=True)
    with ProcessPoolExecutor(
        max_workers=8, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        parent = list(
            pool.map(
                _run_protected,
                [(PARENT_PROTECTION, learned_threshold, task(c)) for c in FRESH],
            )
        )
        candidate = list(
            pool.map(
                _run_protected,
                [(CANDIDATE_PROTECTION, learned_threshold, task(c)) for c in FRESH],
            )
        )
    parent_score, candidate_score = _score(parent), _score(candidate)
    parent_success_retained = all(
        not (clean(old) and old["controlled_reception"])
        or (clean(row) and row["controlled_reception"])
        for old, row in zip(parent, candidate, strict=True)
    )
    qualified = bool(
        parent_success_retained
        and candidate_score[0] >= 5
        and candidate_score[1] == 8
        and candidate_score[0] >= parent_score[0] + 1
    )
    report = {
        "schema": "rosclaw_soccer.rsi.r1_protection_fresh_v255.v1",
        "v254_report_hash": v254["report_hash"],
        "source_hashes": sources,
        "partition": "QUALIFIED_CONSUMED_PROTECTION_THEN_UNTOUCHED_FRESH8",
        "parent_protection_radius": PARENT_PROTECTION,
        "candidate_protection_radius": CANDIDATE_PROTECTION,
        "learned_lateral_threshold": learned_threshold,
        "fresh_bank_hash": fresh_hash,
        "fresh_courses": [vars(course) for course in FRESH],
        "fresh_parent_score": parent_score,
        "fresh_candidate_score": candidate_score,
        "fresh_parent_success_retained": parent_success_retained,
        "fresh_parent": parent,
        "fresh_candidate": candidate,
        "status": "FRESH_PROTECTED_SKILL_QUALIFIED_FOR_CHAIN_GATE"
        if qualified
        else "REJECTED_PROTECTED_SKILL_FRESH_GATE",
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during protected Fresh examination")
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
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    report = exam(parser.parse_args())
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "fresh_parent_score",
                    "fresh_candidate_score",
                    "report_hash",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
