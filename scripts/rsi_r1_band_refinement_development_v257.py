"""Audit whether a measured sub-skill band retains old receiving abilities."""

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

from rosclaw_soccer.rsi.receiving_band_refined_precontact_motor import (
    ReceivingBandRefinedPrecontactMotor,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses
from rosclaw_soccer.training.role_receiving_courses import ReceivingCourse

RADIUS = 0.019
PROTECTION = 0.002
LATERAL_MINIMUM = 0.765
LATERAL_MAXIMUM = 0.769
VX_MINIMUM = -0.480
VX_MAXIMUM = -0.465


def _run_band(job: tuple[tuple[float, ...], float, tuple[Any, ...]]) -> dict[str, Any]:
    offset, threshold, task = job
    if len(offset) != 12 or any(not np.isfinite(value) or abs(value) > 0.35 for value in offset):
        raise ValueError("bounded trained residual required")
    parent_weights = task[-2]
    refined_weights = replace(
        parent_weights,
        output_bias=tuple(
            float(x) for x in np.asarray(parent_weights.output_bias) + np.asarray(offset)
        ),
    )
    prior.RADIUS = RADIUS
    prior.ReceivingVelocityGatedPrecontactMotor = partial(
        ReceivingBandRefinedPrecontactMotor,
        maximum_relative_lateral_feature=threshold,
        protection_radius=PROTECTION,
        refined_neural_weights=refined_weights,
        refined_lateral_minimum=LATERAL_MINIMUM,
        refined_lateral_maximum=LATERAL_MAXIMUM,
        refined_relative_vx_minimum=VX_MINIMUM,
        refined_relative_vx_maximum=VX_MAXIMUM,
    )
    return prior._run_velocity(task)


def in_band(row: dict[str, Any]) -> bool:
    feature = row["activation_features"]
    return bool(
        row["activation_enabled"]
        and not row["protected_episode"]
        and LATERAL_MINIMUM <= feature[1] <= LATERAL_MAXIMUM
        and VX_MINIMUM <= feature[3] <= VX_MAXIMUM
    )


def develop(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    output = args.output
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external measured-band development evidence required")
    v233 = _checked(args.v233_dir / "report.json")
    v236 = _checked(args.v236_dir / "report.json")
    v237 = _checked(args.v237_dir / "report.json")
    v254 = _checked(args.v254_dir / "report.json")
    v255 = _checked(args.v255_dir / "report.json")
    v256 = _checked(args.v256_dir / "report.json")
    v224 = _checked(args.v224_report)
    v229 = _checked(args.v229_dir / "report.json")
    if (
        v256["status"] != "REJECTED_NEAR_CONTACT_RESIDUAL_DEVELOPMENT_GATE"
        or v256["v255_report_hash"] != v255["report_hash"]
        or v255["v254_report_hash"] != v254["report_hash"]
        or v237["v236_report_hash"] != v236["report_hash"]
        or len(v256["finalists"]) != 3
    ):
        raise ValueError("sealed trained-but-forgetting candidates required")
    baseline = (
        v254["rows"][2]["old"]
        + v254["rows"][2]["consumed"]
        + v254["rows"][2]["v252_consumed"]
        + v255["fresh_candidate"]
    )
    courses = tuple(ReceivingCourse(**row["course"]) for row in baseline)
    if len(courses) != 35:
        raise ValueError("complete consumed development bank required")
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
        raise ValueError("sealed contact base differs")
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
            "scripts/rsi_r1_band_refinement_development_v257.py",
            "src/rosclaw_soccer/rsi/receiving_band_refined_precontact_motor.py",
            "scripts/rsi_r1_velocity_gated_fresh_v238.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
        )
    }
    output.mkdir(parents=True)
    rows = []
    with ProcessPoolExecutor(
        max_workers=8, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for finalist in v256["finalists"]:
            offset = tuple(float(value) for value in finalist["offset"])
            result = list(pool.map(_run_band, [(offset, threshold, task(c)) for c in courses]))
            segments = (result[:11], result[11:19], result[19:27], result[27:])
            scores = [_score(segment) for segment in segments]
            old_retained = all(
                not (clean(parent) and parent["controlled_reception"])
                or (clean(row) and row["controlled_reception"])
                for parent, row in zip(baseline[:11], result[:11], strict=True)
            )
            unrelated_exact = all(
                parent["physical_trace_hash"] == row["physical_trace_hash"]
                for parent, row in zip(baseline, result, strict=True)
                if not in_band(parent)
            )
            qualified = bool(
                old_retained
                and unrelated_exact
                and scores[0][0] >= 7
                and scores[0][1] >= 9
                and scores[1][0] >= 5
                and scores[1][1] == 8
                and scores[2][0] >= 5
                and scores[2][1] == 8
                and scores[3][0] >= 5
                and scores[3][1] >= 7
            )
            rows.append(
                {
                    "source_generation": finalist["generation"],
                    "source_index": finalist["index"],
                    "offset": finalist["offset"],
                    "scores": scores,
                    "old_success_retained": old_retained,
                    "outside_refinement_physical_exact": unrelated_exact,
                    "development_gate": qualified,
                    "refined_course_seeds": [
                        row["course"]["seed"] for row in baseline if in_band(row)
                    ],
                    "result": result,
                }
            )
            print(
                json.dumps(
                    {
                        "source": [finalist["generation"], finalist["index"]],
                        "scores": scores,
                        "gate": qualified,
                    }
                ),
                flush=True,
            )
    selected = next((row for row in rows if row["development_gate"]), None)
    report = {
        "schema": "rosclaw_soccer.rsi.r1_band_refinement_development_v257.v1",
        "v256_report_hash": v256["report_hash"],
        "source_hashes": sources,
        "partition": "CONSUMED_OLD11_V2388_V2528_V2558_ONLY",
        "course_bank_hash": bank_hash,
        "measured_refinement_band": {
            "lateral_minimum": LATERAL_MINIMUM,
            "lateral_maximum": LATERAL_MAXIMUM,
            "relative_vx_minimum": VX_MINIMUM,
            "relative_vx_maximum": VX_MAXIMUM,
        },
        "rows": rows,
        "selected_offset": selected["offset"] if selected else None,
        "status": "DEVELOPMENT_BAND_REFINEMENT_FOR_FRESH_EXAM"
        if selected
        else "REJECTED_BAND_REFINEMENT_DEVELOPMENT_GATE",
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during measured-band development")
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
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    report = develop(parser.parse_args())
    print(json.dumps({"status": report["status"], "report_hash": report["report_hash"]}))


if __name__ == "__main__":
    main()
