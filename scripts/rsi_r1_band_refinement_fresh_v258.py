"""Untouched paired exam of measured-band receiving residual versus its parent."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from rsi_r1_band_refinement_development_v257 import _run_band
from rsi_r1_coherent_exploration_v198 import _checked
from rsi_r1_contact_manifold_v222 import context
from rsi_r1_measured_skill_router_v207 import _score, clean
from rsi_r1_protected_kinematic_fresh_v202 import _load_policy
from rsi_r1_protection_radius_development_v254 import _run_protected
from rsi_r1_velocity_gated_fresh_v238 import ANCHOR

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses
from rosclaw_soccer.training.role_receiving_courses import ReceivingCourse

FRESH = tuple(
    ReceivingCourse("red.finisher", 258001 + i * 4 + j, speed, lateral)
    for i, lateral in enumerate((0.0656, 0.0662))
    for j, speed in enumerate((1.17, 1.19, 1.21, 1.23))
)


def exam(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    output = args.output
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external untouched Fresh evidence required")
    v233 = _checked(args.v233_dir / "report.json")
    v236 = _checked(args.v236_dir / "report.json")
    v237 = _checked(args.v237_dir / "report.json")
    v254 = _checked(args.v254_dir / "report.json")
    v255 = _checked(args.v255_dir / "report.json")
    v256 = _checked(args.v256_dir / "report.json")
    v257 = _checked(args.v257_dir / "report.json")
    v224 = _checked(args.v224_report)
    v229 = _checked(args.v229_dir / "report.json")
    if (
        v257["status"] != "DEVELOPMENT_BAND_REFINEMENT_FOR_FRESH_EXAM"
        or v257["v256_report_hash"] != v256["report_hash"]
        or v256["v255_report_hash"] != v255["report_hash"]
        or v255["v254_report_hash"] != v254["report_hash"]
        or v237["v236_report_hash"] != v236["report_hash"]
        or v257["selected_offset"] is None
    ):
        raise ValueError("sealed retained development policy required before Fresh")
    fresh_hash = preflight_receiving_courses(FRESH)
    consumed = {row["course"]["seed"] for result in v257["rows"] for row in result["result"]}
    if consumed & {course.seed for course in FRESH}:
        raise ValueError("Fresh seeds must not occur in consumed development")
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
    threshold = float(v254["learned_lateral_threshold"])
    offset = tuple(float(value) for value in v257["selected_offset"])

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

    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_band_refinement_fresh_v258.py",
            "scripts/rsi_r1_band_refinement_development_v257.py",
            "src/rosclaw_soccer/rsi/receiving_band_refined_precontact_motor.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
        )
    }
    output.mkdir(parents=True)
    with ProcessPoolExecutor(
        max_workers=8, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        parent = list(pool.map(_run_protected, [(0.002, threshold, task(c)) for c in FRESH]))
        candidate = list(pool.map(_run_band, [(offset, threshold, task(c)) for c in FRESH]))
    parent_score, candidate_score = _score(parent), _score(candidate)
    retained = all(
        not (clean(old) and old["controlled_reception"])
        or (clean(new) and new["controlled_reception"])
        for old, new in zip(parent, candidate, strict=True)
    )
    qualified = bool(
        retained
        and candidate_score[0] >= 5
        and candidate_score[1] >= 7
        and candidate_score[0] >= parent_score[0] + 1
        and candidate_score[1] >= parent_score[1]
    )
    report = {
        "schema": "rosclaw_soccer.rsi.r1_band_refinement_fresh_v258.v1",
        "v257_report_hash": v257["report_hash"],
        "source_hashes": sources,
        "partition": "QUALIFIED_CONSUMED_BAND_THEN_UNTOUCHED_FRESH8",
        "fresh_bank_hash": fresh_hash,
        "fresh_courses": [vars(course) for course in FRESH],
        "fresh_parent_score": parent_score,
        "fresh_candidate_score": candidate_score,
        "fresh_parent_success_retained": retained,
        "fresh_parent": parent,
        "fresh_candidate": candidate,
        "status": "FRESH_BAND_REFINEMENT_FOR_CHAIN_GATE"
        if qualified
        else "REJECTED_BAND_REFINEMENT_FRESH_GATE",
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during measured-band Fresh exam")
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
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    report = exam(parser.parse_args())
    print(
        json.dumps(
            {
                "status": report["status"],
                "fresh_parent_score": report["fresh_parent_score"],
                "fresh_candidate_score": report["fresh_candidate_score"],
                "report_hash": report["report_hash"],
            }
        )
    )


if __name__ == "__main__":
    main()
