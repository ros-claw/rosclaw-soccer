"""SIM_ONLY full-retention audit and predeclared fresh phase-contact exam."""

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
from rsi_r1_phase_contact_cem_v226 import _run_task

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses
from rosclaw_soccer.training.role_receiving_courses import ReceivingCourse

SCHEMA = "rosclaw_soccer.rsi.r1_phase_contact_fresh_v227.result.v1"
FRESH_COURSES = tuple(
    ReceivingCourse("red.finisher", 227001 + i * 4 + j, speed, lateral)
    for i, lateral in enumerate((0.068, 0.074))
    for j, speed in enumerate((1.17, 1.22, 1.26, 1.30))
)


def exam(
    asset_root: Path,
    policy_path: Path,
    parent_report: Path,
    right_report: Path,
    refine_report: Path,
    lateral_report: Path,
    v205_dir: Path,
    map_report_path: Path,
    v214_report_path: Path,
    v215_dir: Path,
    v216_report_path: Path,
    v218_dir: Path,
    v220_report_path: Path,
    v223_report_path: Path,
    v224_report_path: Path,
    v226_report_path: Path,
    output: Path,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY phase-contact exam evidence required")
    fresh_hash = preflight_receiving_courses(FRESH_COURSES)
    v223, v224, v226 = (
        _checked(path) for path in (v223_report_path, v224_report_path, v226_report_path)
    )
    if (
        v226["status"] != "DEVELOPMENT_PHASE_CONTACT_CANDIDATE_ONLY"
        or v226["v224_report_hash"] != v224["report_hash"]
        or v224["v223_report_hash"] != v223["report_hash"]
        or v226["best_rank"][:2] != [4, 4]
    ):
        raise ValueError("sealed clean four-course phase contact training required")
    common, (base_weights, _, _), lineage = context(
        asset_root,
        policy_path,
        parent_report,
        right_report,
        refine_report,
        lateral_report,
        v205_dir,
        map_report_path,
        v214_report_path,
        v215_dir,
        v216_report_path,
        v218_dir,
        v220_report_path,
    )
    if lineage["v220_report_hash"] != v224["v220_report_hash"]:
        raise ValueError("sealed precontact motor lineage required")
    weights = replace(
        base_weights,
        output_bias=tuple(
            float(value)
            for value in np.asarray(base_weights.output_bias) + np.asarray(v224["best_offset"])
        ),
    )
    post_logits = tuple(float(value) for value in v226["best_post_logits"])
    old_courses = tuple(ReceivingCourse(**row["course"]) for row in v223["rows"]["baseline"])

    def tasks(
        courses: tuple[ReceivingCourse, ...], post: tuple[float, ...]
    ) -> list[tuple[Any, ...]]:
        return [(common[0], common[1], course, *common[2:], weights, post) for course in courses]

    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_phase_contact_fresh_v227.py",
            "scripts/rsi_r1_phase_contact_cem_v226.py",
            "scripts/rsi_r1_contact_manifold_v222.py",
            "src/rosclaw_soccer/rsi/receiving_phase_contact_motor.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    output.mkdir(parents=True)
    with ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        old_parent = list(pool.map(_run_task, tasks(old_courses, (0.0,) * 12)))
        old_candidate = list(pool.map(_run_task, tasks(old_courses, post_logits)))
        old_score = _score(old_parent)
        candidate_score = _score(old_candidate)
        retained = all(
            not (clean(old) and old["controlled_reception"])
            or (clean(new) and new["controlled_reception"])
            for old, new in zip(old_parent, old_candidate, strict=True)
        )
        sealed_training_rows = {row["course"]["seed"]: row for row in v226["zero_post_replay"]}
        zero_equal = all(
            old["physical_trace_hash"]
            == sealed_training_rows[old["course"]["seed"]]["physical_trace_hash"]
            for old in old_parent
            if old["course"]["seed"] in sealed_training_rows
        )
        qualified_development = (
            retained
            and zero_equal
            and candidate_score[0] >= old_score[0] + 2
            and candidate_score[1] >= old_score[1]
        )
        print(
            json.dumps(
                {
                    "old_parent": old_score,
                    "old_candidate": candidate_score,
                    "retained": retained,
                    "development_qualified": qualified_development,
                }
            ),
            flush=True,
        )
        if qualified_development:
            fresh_parent = list(pool.map(_run_task, tasks(FRESH_COURSES, (0.0,) * 12)))
            fresh_candidate = list(pool.map(_run_task, tasks(FRESH_COURSES, post_logits)))
        else:
            fresh_parent = []
            fresh_candidate = []
    fresh_parent_score = _score(fresh_parent) if qualified_development else None
    fresh_candidate_score = _score(fresh_candidate) if qualified_development else None
    fresh_qualified = bool(
        qualified_development
        and fresh_parent_score is not None
        and fresh_candidate_score is not None
        and fresh_candidate_score[0] >= 6
        and fresh_candidate_score[1] >= 7
        and fresh_candidate_score[0] >= fresh_parent_score[0] + 2
    )
    report = {
        "schema": SCHEMA,
        "v226_report_hash": v226["report_hash"],
        "source_hashes": sources,
        "partition": "CONSUMED_BROAD_RETENTION_THEN_PREDECLARED_FRESH8",
        "fresh_bank_hash": fresh_hash,
        "fresh_courses": [vars(course) for course in FRESH_COURSES],
        "old_parent": old_parent,
        "old_candidate": old_candidate,
        "old_parent_score": old_score,
        "old_candidate_score": candidate_score,
        "old_success_retained": retained,
        "old_zero_parent_equal": zero_equal,
        "fresh_parent": fresh_parent,
        "fresh_candidate": fresh_candidate,
        "fresh_parent_score": fresh_parent_score,
        "fresh_candidate_score": fresh_candidate_score,
        "status": (
            "FRESH_PHASE_CONTACT_QUALIFIED_FOR_NEXT_CHAIN_GATE"
            if fresh_qualified
            else "REJECTED_PHASE_CONTACT_FRESH_GATE"
            if qualified_development
            else "REJECTED_PHASE_CONTACT_RETENTION_GATE"
        ),
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during phase-contact fresh exam")
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
        "v223-report",
        "v224-report",
        "v226-report",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    report = exam(
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
        args.v223_report,
        args.v224_report,
        args.v226_report,
        args.output,
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "old_parent_score",
                    "old_candidate_score",
                    "fresh_parent_score",
                    "fresh_candidate_score",
                    "report_hash",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
