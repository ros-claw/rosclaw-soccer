"""Test stability/plasticity protection radii on consumed physical courses."""

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
from rsi_r1_gated_nearside_fresh_v237 import TRAIN_SEEDS
from rsi_r1_learned_lateral_split_fresh_v253 import fit_from_consumed_radius
from rsi_r1_measured_skill_router_v207 import _score, clean
from rsi_r1_protected_kinematic_fresh_v202 import _load_policy

from rosclaw_soccer.rsi.receiving_lateral_velocity_gated_precontact_motor import (
    ReceivingLateralVelocityGatedPrecontactMotor,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses
from rosclaw_soccer.training.role_receiving_courses import ReceivingCourse

RADII = (0.003, 0.0025, 0.002, 0.0015)
ACTIVATION_RADIUS = 0.019


def _run_protected(job: tuple[float, float, tuple[Any, ...]]) -> dict[str, Any]:
    protection_radius, lateral_threshold, task = job
    if protection_radius not in RADII or not 0.760 <= lateral_threshold <= 0.775:
        raise ValueError("predeclared SIM_ONLY protection and lateral bounds required")
    prior.RADIUS = ACTIVATION_RADIUS
    prior.ReceivingVelocityGatedPrecontactMotor = partial(
        ReceivingLateralVelocityGatedPrecontactMotor,
        maximum_relative_lateral_feature=lateral_threshold,
        protection_radius=protection_radius,
    )
    return prior._run_velocity(task)


def develop(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    output = args.output
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external protection-radius development evidence required")
    v232 = _checked(args.v232_dir / "report.json")
    v233 = _checked(args.v233_dir / "report.json")
    v236 = _checked(args.v236_dir / "report.json")
    v237 = _checked(args.v237_dir / "report.json")
    v238 = _checked(args.v238_dir / "report.json")
    v251 = _checked(args.v251_dir / "report.json")
    v252 = _checked(args.v252_dir / "report.json")
    v253 = _checked(args.v253_dir / "report.json")
    v224 = _checked(args.v224_report)
    v229 = _checked(args.v229_dir / "report.json")
    if (
        v253["status"] != "REJECTED_LEARNED_LATERAL_DEVELOPMENT_GATE"
        or v253["v252_report_hash"] != v252["report_hash"]
        or v253["v251_report_hash"] != v251["report_hash"]
        or v252["v251_report_hash"] != v251["report_hash"]
        or v251["v238_report_hash"] != v238["report_hash"]
        or v237["v236_report_hash"] != v236["report_hash"]
        or v233["v232_report_hash"] != v232["report_hash"]
    ):
        raise ValueError("sealed protected-state failure lineage required")
    lateral_threshold, labels = fit_from_consumed_radius(v251)
    if lateral_threshold != v253["learned_lateral_threshold"]:
        raise ValueError("measured lateral threshold differs from sealed learner")
    old = tuple(ReceivingCourse(**row["course"]) for row in v232["course_summary"])
    consumed = tuple(ReceivingCourse(**row) for row in v238["fresh_courses"])
    v252_courses = tuple(ReceivingCourse(**row) for row in v252["fresh_courses"])
    if (len(old), len(consumed), len(v252_courses)) != (11, 8, 8):
        raise ValueError("complete consumed physical development banks required")
    bank_hash = preflight_receiving_courses(old + consumed + v252_courses)
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
        raise ValueError("sealed neural contact base required")
    base_bias = np.asarray(base_weights.output_bias) + np.asarray(v224["best_offset"])
    pre_weights = replace(base_weights, output_bias=tuple(float(x) for x in base_bias))
    candidate_weights = replace(
        base_weights,
        output_bias=tuple(float(x) for x in base_bias + np.asarray(v236["best_offset"])),
    )
    neural = _load_policy(args.v229_dir / "update-1.npz", v229["history"][0]["checkpoint_hash"])
    anchor = next(row for row in v233["candidates"] if row["source_episode"] == prior.ANCHOR)
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

    bank = old + consumed + v252_courses
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_protection_radius_development_v254.py",
            "scripts/rsi_r1_velocity_gated_fresh_v238.py",
            "scripts/rsi_r1_learned_lateral_split_fresh_v253.py",
            "src/rosclaw_soccer/rsi/receiving_lateral_velocity_gated_precontact_motor.py",
            "src/rosclaw_soccer/rsi/receiving_protected_composed_neural.py",
        )
    }
    output.mkdir(parents=True)
    jobs = [(radius, lateral_threshold, task(course)) for radius in RADII for course in bank]
    with ProcessPoolExecutor(
        max_workers=8, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        results = list(pool.map(_run_protected, jobs))
    rows = []
    for index, radius in enumerate(RADII):
        batch = results[index * 27 : (index + 1) * 27]
        old_rows, consumed_rows, v252_rows = batch[:11], batch[11:19], batch[19:]
        if radius == 0.003 and any(
            row["physical_trace_hash"] != sealed["physical_trace_hash"]
            for row, sealed in zip(
                batch,
                v253["old_candidate"]
                + v253["consumed_candidate"]
                + v253["v252_consumed_candidate"],
                strict=True,
            )
        ):
            raise ValueError("frozen protection radius failed sealed physical replay")
        old_retained = all(
            not (clean(parent) and parent["controlled_reception"])
            or (clean(row) and row["controlled_reception"])
            for parent, row in zip(v253["old_candidate"], old_rows, strict=True)
        )
        non_target_exact = all(
            parent["physical_trace_hash"] == row["physical_trace_hash"]
            for parent, row in zip(v253["old_candidate"], old_rows, strict=True)
            if row["course"]["seed"] not in TRAIN_SEEDS
        )
        old_score, consumed_score, v252_score = (
            _score(old_rows),
            _score(consumed_rows),
            _score(v252_rows),
        )
        rows.append(
            {
                "protection_radius": radius,
                "old_score": old_score,
                "consumed_score": consumed_score,
                "v252_consumed_score": v252_score,
                "old_retained": old_retained,
                "non_target_exact": non_target_exact,
                "development_gate": bool(
                    old_retained
                    and non_target_exact
                    and old_score[0] >= 7
                    and old_score[1] >= 9
                    and consumed_score[0] >= 5
                    and consumed_score[1] == 8
                    and v252_score[0] >= 5
                    and v252_score[1] == 8
                ),
                "old": old_rows,
                "consumed": consumed_rows,
                "v252_consumed": v252_rows,
            }
        )
    result = {
        "schema": "rosclaw_soccer.rsi.r1_protection_radius_development_v254.v1",
        "v253_report_hash": v253["report_hash"],
        "source_hashes": sources,
        "partition": "CONSUMED_OLD11_V238_FRESH8_V252_FRESH8_ONLY",
        "course_bank_hash": bank_hash,
        "lateral_training_labels": labels,
        "learned_lateral_threshold": lateral_threshold,
        "protection_radii": list(RADII),
        "rows": rows,
        "status": "DEVELOPMENT_PROTECTION_CANDIDATE_FOR_FRESH_EXAM"
        if any(row["development_gate"] for row in rows[1:])
        else "REJECTED_PROTECTION_RADIUS_DEVELOPMENT_GATE",
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    result["report_hash"] = hash_json(result)
    (output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during protected-state development")
    return result


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
        "v236-dir",
        "v237-dir",
        "v238-dir",
        "v251-dir",
        "v252-dir",
        "v253-dir",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    report = develop(parser.parse_args())
    print(
        json.dumps(
            {
                "status": report["status"],
                "scores": [
                    (
                        row["protection_radius"],
                        row["old_score"],
                        row["consumed_score"],
                        row["v252_consumed_score"],
                        row["development_gate"],
                    )
                    for row in report["rows"]
                ],
                "report_hash": report["report_hash"],
            }
        )
    )


if __name__ == "__main__":
    main()
