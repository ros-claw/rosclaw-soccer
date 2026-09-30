"""Test a broader measured-speed band on consumed physical courses only."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import rsi_r1_band_refinement_development_v257 as band
from rsi_r1_coherent_exploration_v198 import _checked
from rsi_r1_contact_manifold_v222 import context
from rsi_r1_measured_skill_router_v207 import _score, clean
from rsi_r1_protected_kinematic_fresh_v202 import _load_policy
from rsi_r1_velocity_gated_fresh_v238 import ANCHOR

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses
from rosclaw_soccer.training.role_receiving_courses import ReceivingCourse

WIDE_VX_MAXIMUM = -0.40


def _run_wide(job: tuple[tuple[float, ...], float, tuple[Any, ...]]) -> dict[str, Any]:
    band.VX_MAXIMUM = WIDE_VX_MAXIMUM
    return band._run_band(job)


def _wide_band(row: dict[str, Any]) -> bool:
    feature = row["activation_features"]
    return bool(
        row["activation_enabled"]
        and not row["protected_episode"]
        and band.LATERAL_MINIMUM <= feature[1] <= band.LATERAL_MAXIMUM
        and band.VX_MINIMUM <= feature[3] <= WIDE_VX_MAXIMUM
    )


def develop(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    output = args.output
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external consumed wide-band evidence required")
    v233 = _checked(args.v233_dir / "report.json")
    v236 = _checked(args.v236_dir / "report.json")
    v237 = _checked(args.v237_dir / "report.json")
    v254 = _checked(args.v254_dir / "report.json")
    v255 = _checked(args.v255_dir / "report.json")
    v256 = _checked(args.v256_dir / "report.json")
    v257 = _checked(args.v257_dir / "report.json")
    v258 = _checked(args.v258_dir / "report.json")
    v259 = _checked(args.v259_dir / "report.json")
    v224 = _checked(args.v224_report)
    v229 = _checked(args.v229_dir / "report.json")
    if (
        v259["status"] != "ACTION_MAP_INSUFFICIENT_POSITIVE_COURSES"
        or v259["v258_report_hash"] != v258["report_hash"]
        or v258["v257_report_hash"] != v257["report_hash"]
        or v257["v256_report_hash"] != v256["report_hash"]
        or v237["v236_report_hash"] != v236["report_hash"]
        or v257["selected_offset"]
        != next(row["offset"] for row in v259["actions"] if row["action_id"] == [1, 3])
    ):
        raise ValueError("sealed local action-map evidence and selected actor required")
    baseline = (
        v254["rows"][2]["old"]
        + v254["rows"][2]["consumed"]
        + v254["rows"][2]["v252_consumed"]
        + v255["fresh_candidate"]
        + v258["fresh_parent"]
        + v259["parent"]
    )
    courses = tuple(ReceivingCourse(**row["course"]) for row in baseline)
    if len(courses) != 55:
        raise ValueError("complete consumed old and new course bank required")
    bank_hash = preflight_receiving_courses(courses)
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
            "scripts/rsi_r1_continuous_band_development_v260.py",
            "scripts/rsi_r1_band_refinement_development_v257.py",
            "src/rosclaw_soccer/rsi/receiving_band_refined_precontact_motor.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
        )
    }
    output.mkdir(parents=True)
    with ProcessPoolExecutor(
        max_workers=8, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        rows = list(pool.map(_run_wide, [(offset, threshold, task(c)) for c in courses]))
    segments = (rows[:11], rows[11:19], rows[19:27], rows[27:35], rows[35:43], rows[43:])
    scores = [_score(segment) for segment in segments]
    old_success_retained = all(
        not (clean(parent) and parent["controlled_reception"])
        or (clean(row) and row["controlled_reception"])
        for parent, row in zip(baseline, rows, strict=True)
    )
    unrelated_exact = all(
        parent["physical_trace_hash"] == row["physical_trace_hash"]
        for parent, row in zip(baseline, rows, strict=True)
        if not _wide_band(parent)
    )
    qualified = bool(
        old_success_retained
        and unrelated_exact
        and scores[0][0] >= 7
        and scores[0][1] >= 9
        and scores[1][0] >= 5
        and scores[1][1] == 8
        and scores[2][0] >= 5
        and scores[2][1] == 8
        and scores[3][0] >= 5
        and scores[3][1] >= 7
        and scores[4][0] >= 3
        and scores[4][1] >= 7
        and scores[5][0] >= 11
        and scores[5][1] >= 10
    )
    report = {
        "schema": "rosclaw_soccer.rsi.r1_continuous_band_development_v260.v1",
        "v259_report_hash": v259["report_hash"],
        "source_hashes": sources,
        "partition": "CONSUMED_OLD11_V2388_V2528_V2558_V2588_V25912_ONLY",
        "course_bank_hash": bank_hash,
        "wide_relative_vx_maximum": WIDE_VX_MAXIMUM,
        "selected_offset": list(offset),
        "parent_scores": [
            _score(baseline[:11]),
            _score(baseline[11:19]),
            _score(baseline[19:27]),
            _score(baseline[27:35]),
            _score(baseline[35:43]),
            _score(baseline[43:]),
        ],
        "candidate_scores": scores,
        "old_success_retained": old_success_retained,
        "outside_refinement_physical_exact": unrelated_exact,
        "refined_course_seeds": [row["course"]["seed"] for row in baseline if _wide_band(row)],
        "rows": rows,
        "status": "DEVELOPMENT_WIDE_BAND_FOR_FRESH_EXAM"
        if qualified
        else "REJECTED_WIDE_BAND_DEVELOPMENT_GATE",
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during wide-band development")
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
        "v259-dir",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    report = develop(parser.parse_args())
    print(
        json.dumps(
            {
                "status": report["status"],
                "parent_scores": report["parent_scores"],
                "candidate_scores": report["candidate_scores"],
                "report_hash": report["report_hash"],
            }
        )
    )


if __name__ == "__main__":
    main()
