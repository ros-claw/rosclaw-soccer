"""Learn skill-selection boundary from paired failures, then test fresh physics."""

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
from rsi_r1_measured_skill_router_v207 import _score, clean
from rsi_r1_protected_kinematic_fresh_v202 import _load_policy

from rosclaw_soccer.rsi.measured_lateral_split import fit_measured_lateral_split
from rosclaw_soccer.rsi.receiving_lateral_velocity_gated_precontact_motor import (
    ReceivingLateralVelocityGatedPrecontactMotor,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses
from rosclaw_soccer.training.role_receiving_courses import ReceivingCourse

FROZEN_THRESHOLD = 0.767
RADIUS = 0.019
FRESH = tuple(
    ReceivingCourse("red.finisher", 253001 + i * 4 + j, speed, lateral)
    for i, lateral in enumerate((0.0652, 0.0657))
    for j, speed in enumerate((1.16, 1.18, 1.20, 1.22))
)


def _run_boundary(job: tuple[float, tuple[Any, ...]]) -> dict[str, Any]:
    threshold, task = job
    if type(threshold) is not float or not 0.760 <= threshold <= 0.775:
        raise ValueError("bounded learned SIM_ONLY lateral threshold required")
    prior.RADIUS = RADIUS
    prior.ReceivingVelocityGatedPrecontactMotor = partial(
        ReceivingLateralVelocityGatedPrecontactMotor,
        maximum_relative_lateral_feature=threshold,
    )
    return prior._run_velocity(task)


def fit_from_consumed_radius(v251: dict[str, Any]) -> tuple[float, list[dict[str, Any]]]:
    baseline = next(row for row in v251["rows"] if row["radius"] == 0.017)
    expanded = next(row for row in v251["rows"] if row["radius"] == 0.019)
    labels = []
    benefit, harm = [], []
    for partition, before, after in (
        ("consumed", baseline["consumed"], expanded["consumed"]),
        ("old", baseline["old"], expanded["old"]),
    ):
        for parent, candidate in zip(before, after, strict=True):
            if candidate["physical_trace_hash"] == parent["physical_trace_hash"]:
                continue
            if (
                clean(candidate)
                and candidate["controlled_reception"]
                and not (clean(parent) and parent["controlled_reception"])
            ):
                label = "BENEFIT"
                benefit.append(float(candidate["activation_features"][1]))
            elif (
                clean(parent)
                and parent["controlled_reception"]
                and not (clean(candidate) and candidate["controlled_reception"])
            ):
                label = "HARM"
                harm.append(float(candidate["activation_features"][1]))
            else:
                continue
            labels.append(
                {
                    "partition": partition,
                    "source_course_seed": candidate["course"]["seed"],
                    "measured_lateral_feature": candidate["activation_features"][1],
                    "label": label,
                    "parent_trace_hash": parent["physical_trace_hash"],
                    "candidate_trace_hash": candidate["physical_trace_hash"],
                }
            )
    return fit_measured_lateral_split(tuple(benefit), tuple(harm)), labels


def exam(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    output = args.output
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external learned-boundary evidence required")
    v232 = _checked(args.v232_dir / "report.json")
    v233 = _checked(args.v233_dir / "report.json")
    v236 = _checked(args.v236_dir / "report.json")
    v237 = _checked(args.v237_dir / "report.json")
    v238 = _checked(args.v238_dir / "report.json")
    v251 = _checked(args.v251_dir / "report.json")
    v252 = _checked(args.v252_dir / "report.json")
    v224 = _checked(args.v224_report)
    v229 = _checked(args.v229_dir / "report.json")
    if (
        v252["status"] != "REJECTED_LATERAL_VELOCITY_FRESH_GATE"
        or not v252["development_gate"]
        or v252["v251_report_hash"] != v251["report_hash"]
        or v251["v238_report_hash"] != v238["report_hash"]
        or v238["v237_report_hash"] != v237["report_hash"]
        or v237["v236_report_hash"] != v236["report_hash"]
        or v233["v232_report_hash"] != v232["report_hash"]
    ):
        raise ValueError("sealed consumed benefit/risk lineage required")
    learned_threshold, labels = fit_from_consumed_radius(v251)
    old = tuple(ReceivingCourse(**row["course"]) for row in v232["course_summary"])
    consumed = tuple(ReceivingCourse(**row) for row in v238["fresh_courses"])
    consumed_v252 = tuple(ReceivingCourse(**row) for row in v252["fresh_courses"])
    if (len(old), len(consumed), len(consumed_v252)) != (11, 8, 8):
        raise ValueError("complete old and consumed development banks required")
    fresh_hash = preflight_receiving_courses(FRESH)
    if {course.seed for course in FRESH} & {
        course.seed for course in old + consumed + consumed_v252
    }:
        raise ValueError("fresh examination seeds must be disjoint")
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
        raise ValueError("sealed precontact policy differs")
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
            "scripts/rsi_r1_learned_lateral_split_fresh_v253.py",
            "scripts/rsi_r1_velocity_gated_fresh_v238.py",
            "src/rosclaw_soccer/rsi/measured_lateral_split.py",
            "src/rosclaw_soccer/rsi/receiving_lateral_velocity_gated_precontact_motor.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
        )
    }
    output.mkdir(parents=True)
    with ProcessPoolExecutor(
        max_workers=8, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        old_candidate = list(pool.map(_run_boundary, [(learned_threshold, task(c)) for c in old]))
        consumed_candidate = list(
            pool.map(_run_boundary, [(learned_threshold, task(c)) for c in consumed])
        )
        v252_candidate = list(
            pool.map(_run_boundary, [(learned_threshold, task(c)) for c in consumed_v252])
        )
        old_retained = all(
            not (clean(parent) and parent["controlled_reception"])
            or (clean(row) and row["controlled_reception"])
            for parent, row in zip(v252["old_candidate"], old_candidate, strict=True)
        )
        non_target_exact = all(
            parent["physical_trace_hash"] == row["physical_trace_hash"]
            for parent, row in zip(v252["old_candidate"], old_candidate, strict=True)
            if row["course"]["seed"] not in TRAIN_SEEDS
        )
        old_score = _score(old_candidate)
        consumed_score = _score(consumed_candidate)
        v252_score = _score(v252_candidate)
        development_gate = bool(
            old_retained
            and non_target_exact
            and old_score[0] >= 7
            and old_score[1] >= 9
            and consumed_score[0] >= 5
            and consumed_score[1] == 8
            and v252_score[0] >= 5
            and v252_score[1] == 8
        )
        if development_gate:
            fresh_parent = list(
                pool.map(_run_boundary, [(FROZEN_THRESHOLD, task(c)) for c in FRESH])
            )
            fresh_candidate = list(
                pool.map(_run_boundary, [(learned_threshold, task(c)) for c in FRESH])
            )
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
        and fresh_candidate_score[1] == 8
        and fresh_candidate_score[0] >= fresh_parent_score[0] + 1
    )
    result = {
        "schema": "rosclaw_soccer.rsi.r1_learned_lateral_split_fresh_v253.v1",
        "v252_report_hash": v252["report_hash"],
        "v251_report_hash": v251["report_hash"],
        "source_hashes": sources,
        "partition": "FIT_CONSUMED_V251_PAIRS_THEN_V252_DEVELOPMENT_THEN_NEW_FRESH8",
        "training_labels": labels,
        "frozen_lateral_threshold": FROZEN_THRESHOLD,
        "learned_lateral_threshold": learned_threshold,
        "activation_radius": RADIUS,
        "fresh_bank_hash": fresh_hash,
        "fresh_courses": [vars(course) for course in FRESH],
        "old_score": old_score,
        "consumed_score": consumed_score,
        "v252_consumed_score": v252_score,
        "old_success_retained": old_retained,
        "non_target_exact": non_target_exact,
        "development_gate": development_gate,
        "old_candidate": old_candidate,
        "consumed_candidate": consumed_candidate,
        "v252_consumed_candidate": v252_candidate,
        "fresh_parent_score": fresh_parent_score,
        "fresh_candidate_score": fresh_candidate_score,
        "fresh_parent": fresh_parent,
        "fresh_candidate": fresh_candidate,
        "status": "FRESH_LEARNED_LATERAL_QUALIFIED_FOR_CHAIN_GATE"
        if qualified
        else "REJECTED_LEARNED_LATERAL_FRESH_GATE"
        if development_gate
        else "REJECTED_LEARNED_LATERAL_DEVELOPMENT_GATE",
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    result["report_hash"] = hash_json(result)
    (output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during learned-boundary examination")
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
                    "learned_lateral_threshold",
                    "old_score",
                    "consumed_score",
                    "v252_consumed_score",
                    "fresh_parent_score",
                    "fresh_candidate_score",
                    "report_hash",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
