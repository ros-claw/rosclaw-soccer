"""Evaluate continuous measured-state residual ramps on consumed courses."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import rsi_r1_velocity_gated_fresh_v238 as prior
from rsi_r1_coherent_exploration_v198 import _checked
from rsi_r1_contact_manifold_v222 import context
from rsi_r1_measured_skill_router_v207 import _score, clean
from rsi_r1_protected_kinematic_fresh_v202 import _load_policy
from rsi_r1_velocity_gated_fresh_v238 import ANCHOR

from rosclaw_soccer.rsi.receiving_smooth_residual_motor import ReceivingSmoothResidualMotor
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses
from rosclaw_soccer.training.role_receiving_courses import ReceivingCourse

RAMPS = ((0.7645, 0.7660), (0.7650, 0.7664), (0.7650, 0.7670))


def _run_smooth(
    job: tuple[tuple[float, float], tuple[float, ...], float, tuple[Any, ...]],
) -> dict[str, Any]:
    ramp, offset, threshold, task = job
    if ramp not in RAMPS or len(offset) != 12:
        raise ValueError("predeclared finite SIM_ONLY residual ramp required")
    parent_weights = task[-2]
    refined_weights = replace(
        parent_weights,
        output_bias=tuple(
            float(x) for x in np.asarray(parent_weights.output_bias) + np.asarray(offset)
        ),
    )
    prior.RADIUS = 0.019
    prior.ReceivingVelocityGatedPrecontactMotor = partial(
        ReceivingSmoothResidualMotor,
        maximum_relative_lateral_feature=threshold,
        protection_radius=0.002,
        refined_neural_weights=refined_weights,
        gain_lateral_start=ramp[0],
        gain_lateral_full=ramp[1],
    )
    return prior._run_velocity(task)


def develop(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    output = args.output
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external consumed smooth-residual evidence required")
    v233 = _checked(args.v233_dir / "report.json")
    v236 = _checked(args.v236_dir / "report.json")
    v237 = _checked(args.v237_dir / "report.json")
    v254 = _checked(args.v254_dir / "report.json")
    v255 = _checked(args.v255_dir / "report.json")
    v256 = _checked(args.v256_dir / "report.json")
    v257 = _checked(args.v257_dir / "report.json")
    v258 = _checked(args.v258_dir / "report.json")
    v259 = _checked(args.v259_dir / "report.json")
    v260 = _checked(args.v260_dir / "report.json")
    v224 = _checked(args.v224_report)
    v229 = _checked(args.v229_dir / "report.json")
    if (
        v260["status"] != "REJECTED_WIDE_BAND_DEVELOPMENT_GATE"
        or v260["v259_report_hash"] != v259["report_hash"]
        or v259["v258_report_hash"] != v258["report_hash"]
        or v258["v257_report_hash"] != v257["report_hash"]
        or v257["v256_report_hash"] != v256["report_hash"]
        or v237["v236_report_hash"] != v236["report_hash"]
        or v260["selected_offset"] != v257["selected_offset"]
    ):
        raise ValueError("sealed wide-band failure and retained actor required")
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
        raise ValueError("complete consumed 55-course bank required")
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
            "scripts/rsi_r1_smooth_residual_development_v261.py",
            "src/rosclaw_soccer/rsi/receiving_smooth_residual_motor.py",
            "scripts/rsi_r1_velocity_gated_fresh_v238.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
        )
    }
    output.mkdir(parents=True)
    results = []
    with ProcessPoolExecutor(
        max_workers=8, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for ramp in RAMPS:
            rows = list(
                pool.map(_run_smooth, [(ramp, offset, threshold, task(c)) for c in courses])
            )
            segments = (rows[:11], rows[11:19], rows[19:27], rows[27:35], rows[35:43], rows[43:])
            scores = [_score(segment) for segment in segments]
            retained = all(
                not (clean(parent) and parent["controlled_reception"])
                or (clean(row) and row["controlled_reception"])
                for parent, row in zip(baseline, rows, strict=True)
            )
            outside_exact = all(
                parent["physical_trace_hash"] == row["physical_trace_hash"]
                for parent, row in zip(baseline, rows, strict=True)
                if not parent["activation_enabled"]
                or parent["protected_episode"]
                or parent["activation_features"][1] <= ramp[0]
            )
            qualified = bool(
                retained
                and outside_exact
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
            results.append(
                {
                    "ramp": list(ramp),
                    "scores": scores,
                    "old_success_retained": retained,
                    "outside_gain_physical_exact": outside_exact,
                    "development_gate": qualified,
                    "rows": rows,
                }
            )
            print(json.dumps({"ramp": ramp, "scores": scores, "gate": qualified}), flush=True)
    selected = next((row for row in results if row["development_gate"]), None)
    report = {
        "schema": "rosclaw_soccer.rsi.r1_smooth_residual_development_v261.v1",
        "v260_report_hash": v260["report_hash"],
        "source_hashes": sources,
        "partition": "CONSUMED_55_COURSE_CONTINUOUS_GAIN_ONLY",
        "course_bank_hash": bank_hash,
        "ramps": [list(ramp) for ramp in RAMPS],
        "selected_offset": list(offset),
        "results": results,
        "selected_ramp": selected["ramp"] if selected else None,
        "status": "DEVELOPMENT_SMOOTH_RESIDUAL_FOR_FRESH_EXAM"
        if selected
        else "REJECTED_SMOOTH_RESIDUAL_DEVELOPMENT_GATE",
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during smooth-residual development")
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
        "v260-dir",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    report = develop(parser.parse_args())
    print(json.dumps({"status": report["status"], "report_hash": report["report_hash"]}))


if __name__ == "__main__":
    main()
