"""Consumed-course, bounded measured-state skill-radius development only."""

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

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses
from rosclaw_soccer.training.role_receiving_courses import ReceivingCourse

RADII = (0.017, 0.018, 0.019, 0.020, 0.022)


def _run_radius(job: tuple[float, tuple[Any, ...]]) -> dict[str, Any]:
    radius, task = job
    if radius not in RADII:
        raise ValueError("predeclared measured activation radius required")
    # This worker-owned historical research runner has no global motor or
    # hardware authority. Override only its SIM_ONLY constructor argument.
    prior.RADIUS = radius
    return prior._run_velocity(task)


def develop(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external development evidence required")
    v232 = _checked(args.v232_dir / "report.json")
    v233 = _checked(args.v233_dir / "report.json")
    v236 = _checked(args.v236_dir / "report.json")
    v237 = _checked(args.v237_dir / "report.json")
    v238 = _checked(args.v238_dir / "report.json")
    v224 = _checked(args.v224_report)
    v229 = _checked(args.v229_dir / "report.json")
    if (
        v238["status"] != "FRESH_VELOCITY_SAFE_NEARSIDE_QUALIFIED_FOR_NEXT_CHAIN_GATE"
        or v238["v237_report_hash"] != v237["report_hash"]
        or v237["v236_report_hash"] != v236["report_hash"]
        or v233["v232_report_hash"] != v232["report_hash"]
    ):
        raise ValueError("sealed qualified receiving lineage required")
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
    old = tuple(ReceivingCourse(**row["course"]) for row in v232["course_summary"])
    consumed = tuple(ReceivingCourse(**row) for row in v238["fresh_courses"])
    if len(old) != 11 or len(consumed) != 8:
        raise ValueError("frozen old and consumed examination counts required")
    bank_hash = preflight_receiving_courses(old + consumed)
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

    jobs = [(radius, task(course)) for radius in RADII for course in old + consumed]
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_measured_radius_development_v251.py",
            "scripts/rsi_r1_velocity_gated_fresh_v238.py",
            "src/rosclaw_soccer/rsi/receiving_gated_precontact_motor.py",
            "src/rosclaw_soccer/rsi/receiving_velocity_gated_precontact_motor.py",
        )
    }
    output.mkdir(parents=True)
    with ProcessPoolExecutor(
        max_workers=8, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        results = list(pool.map(_run_radius, jobs))
    rows = []
    for index, radius in enumerate(RADII):
        row = results[index * 19 : (index + 1) * 19]
        old_rows, consumed_rows = row[:11], row[11:]
        if radius == 0.017 and (
            any(
                value["physical_trace_hash"] != seal["physical_trace_hash"]
                for value, seal in zip(old_rows, v238["old_candidate"], strict=True)
            )
            or any(
                value["physical_trace_hash"] != seal["physical_trace_hash"]
                for value, seal in zip(consumed_rows, v238["fresh_candidate"], strict=True)
            )
        ):
            raise ValueError("historical radius did not replay sealed physical evidence")
        old_success_retained = all(
            not (clean(previous) and previous["controlled_reception"])
            or (clean(value) and value["controlled_reception"])
            for previous, value in zip(v238["old_candidate"], old_rows, strict=True)
        )
        non_target_exact = all(
            previous["physical_trace_hash"] == value["physical_trace_hash"]
            for previous, value in zip(v238["old_candidate"], old_rows, strict=True)
            if value["course"]["seed"] not in TRAIN_SEEDS
        )
        rows.append(
            {
                "radius": radius,
                "old_score": _score(old_rows),
                "consumed_score": _score(consumed_rows),
                "old_success_retained": old_success_retained,
                "non_target_exact": non_target_exact,
                "old": old_rows,
                "consumed": consumed_rows,
                "development_gate": bool(
                    old_success_retained
                    and non_target_exact
                    and _score(old_rows)[0] >= 7
                    and _score(old_rows)[1] >= 9
                    and _score(consumed_rows)[0] >= 5
                    and _score(consumed_rows)[1] == 8
                ),
            }
        )
    result = {
        "schema": "rosclaw_soccer.rsi.measured_radius_development_v251.v1",
        "v238_report_hash": v238["report_hash"],
        "source_hashes": sources,
        "partition": "CONSUMED_V238_FRESH8_AND_OLD11_DEVELOPMENT_ONLY",
        "course_bank_hash": bank_hash,
        "radii": list(RADII),
        "rows": rows,
        "status": "DEVELOPMENT_RADIUS_CANDIDATE_FOR_UNTOUCHED_EXAM"
        if any(row["development_gate"] for row in rows[1:])
        else "REJECTED_MEASURED_RADIUS_DEVELOPMENT_GATE",
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    result["report_hash"] = hash_json(result)
    (output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during radius development")
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
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    report = develop(parser.parse_args())
    print(
        json.dumps(
            {
                "status": report["status"],
                "scores": [
                    (r["radius"], r["old_score"], r["consumed_score"], r["development_gate"])
                    for r in report["rows"]
                ],
                "report_hash": report["report_hash"],
            }
        )
    )


if __name__ == "__main__":
    main()
