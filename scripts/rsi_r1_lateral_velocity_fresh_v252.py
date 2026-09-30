"""SIM_ONLY lateral-speed receiving selector: retention before untouched exam."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import rsi_r1_velocity_gated_fresh_v238 as prior
from rsi_r1_coherent_exploration_v198 import _checked
from rsi_r1_contact_manifold_v222 import context
from rsi_r1_gated_nearside_fresh_v237 import TRAIN_SEEDS
from rsi_r1_measured_skill_router_v207 import _score, clean
from rsi_r1_protected_kinematic_fresh_v202 import _load_policy

from rosclaw_soccer.rsi.receiving_lateral_velocity_gated_precontact_motor import (
    ReceivingLateralVelocityGatedPrecontactMotor,
)
from rosclaw_soccer.rsi.receiving_velocity_gated_precontact_motor import (
    ReceivingVelocityGatedPrecontactMotor,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses
from rosclaw_soccer.training.role_receiving_courses import ReceivingCourse

RADIUS = 0.019
LATERAL_MAX = 0.767
FRESH = tuple(
    ReceivingCourse("red.finisher", 252001 + i * 4 + j, speed, lateral)
    for i, lateral in enumerate((0.0645, 0.0655))
    for j, speed in enumerate((1.17, 1.19, 1.21, 1.23))
)


def _run_mode(job: tuple[str, tuple[Any, ...]]) -> dict[str, Any]:
    mode, task = job
    if mode == "frozen":
        prior.RADIUS = 0.017
        prior.ReceivingVelocityGatedPrecontactMotor = ReceivingVelocityGatedPrecontactMotor
    elif mode == "lateral":
        prior.RADIUS = RADIUS
        prior.ReceivingVelocityGatedPrecontactMotor = ReceivingLateralVelocityGatedPrecontactMotor
    else:
        raise ValueError("known frozen or lateral-only simulation mode required")
    return prior._run_velocity(task)


def exam(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    output = args.output
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY lateral-speed evidence required")
    v232 = _checked(args.v232_dir / "report.json")
    v233 = _checked(args.v233_dir / "report.json")
    v236 = _checked(args.v236_dir / "report.json")
    v237 = _checked(args.v237_dir / "report.json")
    v238 = _checked(args.v238_dir / "report.json")
    v251 = _checked(args.v251_dir / "report.json")
    v224 = _checked(args.v224_report)
    v229 = _checked(args.v229_dir / "report.json")
    if (
        v251["status"] != "REJECTED_MEASURED_RADIUS_DEVELOPMENT_GATE"
        or v251["v238_report_hash"] != v238["report_hash"]
        or v238["v237_report_hash"] != v237["report_hash"]
        or v237["v236_report_hash"] != v236["report_hash"]
        or v233["v232_report_hash"] != v232["report_hash"]
    ):
        raise ValueError("sealed old-skill radius conflict required")
    old = tuple(ReceivingCourse(**row["course"]) for row in v232["course_summary"])
    consumed = tuple(ReceivingCourse(**row) for row in v238["fresh_courses"])
    if len(old) != 11 or len(consumed) != 8:
        raise ValueError("complete consumed development partition required")
    fresh_hash = preflight_receiving_courses(FRESH)
    if {course.seed for course in FRESH} & {course.seed for course in old + consumed}:
        raise ValueError("untouched examination seeds must be disjoint")
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

    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_lateral_velocity_fresh_v252.py",
            "scripts/rsi_r1_velocity_gated_fresh_v238.py",
            "src/rosclaw_soccer/rsi/receiving_gated_precontact_motor.py",
            "src/rosclaw_soccer/rsi/receiving_lateral_velocity_gated_precontact_motor.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
        )
    }
    output.mkdir(parents=True)
    with ProcessPoolExecutor(
        max_workers=8, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        old_candidate = list(pool.map(_run_mode, [("lateral", task(c)) for c in old]))
        consumed_candidate = list(pool.map(_run_mode, [("lateral", task(c)) for c in consumed]))
        old_retained = all(
            not (clean(parent) and parent["controlled_reception"])
            or (clean(row) and row["controlled_reception"])
            for parent, row in zip(v238["old_candidate"], old_candidate, strict=True)
        )
        non_target_exact = all(
            parent["physical_trace_hash"] == row["physical_trace_hash"]
            for parent, row in zip(v238["old_candidate"], old_candidate, strict=True)
            if row["course"]["seed"] not in TRAIN_SEEDS
        )
        old_score = _score(old_candidate)
        consumed_score = _score(consumed_candidate)
        development_gate = bool(
            old_retained
            and non_target_exact
            and old_score[0] >= 7
            and old_score[1] >= 9
            and consumed_score[0] >= 5
            and consumed_score[1] >= 8
        )
        if development_gate:
            fresh_parent = list(pool.map(_run_mode, [("frozen", task(c)) for c in FRESH]))
            fresh_candidate = list(pool.map(_run_mode, [("lateral", task(c)) for c in FRESH]))
        else:
            fresh_parent = []
            fresh_candidate = []
    fresh_parent_score = _score(fresh_parent) if development_gate else None
    fresh_candidate_score = _score(fresh_candidate) if development_gate else None
    qualified = bool(
        development_gate
        and fresh_parent_score is not None
        and fresh_candidate_score is not None
        and fresh_candidate_score[0] >= 5
        and fresh_candidate_score[1] >= 7
        and fresh_candidate_score[0] >= fresh_parent_score[0] + 1
    )
    report = {
        "schema": "rosclaw_soccer.rsi.r1_lateral_velocity_fresh_v252.v1",
        "v251_report_hash": v251["report_hash"],
        "source_hashes": sources,
        "partition": "CONSUMED_OLD11_FRESH8_THEN_UNTOUCHED_NEW_FRESH8",
        "activation_radius": RADIUS,
        "maximum_relative_lateral_feature": LATERAL_MAX,
        "fresh_bank_hash": fresh_hash,
        "fresh_courses": [vars(course) for course in FRESH],
        "old_score": old_score,
        "consumed_score": consumed_score,
        "old_success_retained": old_retained,
        "non_target_exact": non_target_exact,
        "development_gate": development_gate,
        "old_candidate": old_candidate,
        "consumed_candidate": consumed_candidate,
        "fresh_parent_score": fresh_parent_score,
        "fresh_candidate_score": fresh_candidate_score,
        "fresh_parent": fresh_parent,
        "fresh_candidate": fresh_candidate,
        "status": "FRESH_LATERAL_VELOCITY_QUALIFIED_FOR_CHAIN_GATE"
        if qualified
        else "REJECTED_LATERAL_VELOCITY_FRESH_GATE"
        if development_gate
        else "REJECTED_LATERAL_VELOCITY_DEVELOPMENT_GATE",
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during lateral-speed examination")
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
        "v236-dir",
        "v237-dir",
        "v238-dir",
        "v251-dir",
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
                    "old_score",
                    "consumed_score",
                    "fresh_parent_score",
                    "fresh_candidate_score",
                    "report_hash",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
