"""SIM_ONLY predeclared fresh paired exam of hard-protected neural receiving."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from rsi_r1_coherent_exploration_v198 import _checked
from rsi_r1_composed_contact_fresh_v214 import _run_task as _run_parent
from rsi_r1_measured_skill_router_v207 import _score
from rsi_r1_protected_composed_neural_v216 import _run_task as _run_candidate
from rsi_r1_protected_kinematic_fresh_v202 import _load_policy

from rosclaw_soccer.rsi.receiving_foot_phase_router import ReceivingFootPhaseReference
from rosclaw_soccer.rsi.receiving_measured_skill_router import ReceivingSkillKnot
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses
from rosclaw_soccer.training.role_receiving_courses import ReceivingCourse

SCHEMA = "rosclaw_soccer.rsi.r1_protected_neural_fresh_v217.result.v1"
FRESH_COURSES = tuple(
    ReceivingCourse("red.finisher", 217001 + i * 4 + j, speed, lateral)
    for i, lateral in enumerate((0.064, 0.070))
    for j, speed in enumerate((1.16, 1.23, 1.29, 1.32))
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
    output: Path,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY fresh protected neural exam required")
    fresh_hash = preflight_receiving_courses(FRESH_COURSES)
    parent, right_parent, refine, lateral, v205, mapping, v214, v215, v216 = (
        _checked(path)
        for path in (
            parent_report,
            right_report,
            refine_report,
            lateral_report,
            v205_dir / "report.json",
            map_report_path,
            v214_report_path,
            v215_dir / "report.json",
            v216_report_path,
        )
    )
    if (
        v216["status"] != "DEVELOPMENT_PROTECTED_NEURAL_CANDIDATE_ONLY"
        or v216["score"][:2] != [7, 16]
        or not v216["protected_physical_equal"]
        or not v216["anchor_physical_equal"]
        or v216["v215_report_hash"] != v215["report_hash"]
        or v216["v214_report_hash"] != v214["report_hash"]
        or v205["map_report_hash"] != mapping["report_hash"]
        or parent["policy_hash"] != hash_bytes(policy_path.read_bytes())
    ):
        raise ValueError("sealed hard-retained development breakthrough required")
    best = next(row for row in v215["history"] if row["update"] == v215["best_update"])
    weights = _load_policy(v215_dir / "update-3.npz", best["checkpoint_hash"])
    knots = tuple(
        ReceivingSkillKnot(
            float(row["lateral_m"]),
            float(row["closing_speed_mps"]),
            row["expert"],
            tuple(float(value) for value in row["weights"]),
        )
        for row in v214["knots"]
    )
    references = []
    for index, row in enumerate(v205["rows"]):
        if row["expert"] != "high":
            continue
        path = v205_dir / f"teacher-{index}.npz"
        if hash_bytes(path.read_bytes()) != row["feature_hash"]:
            raise ValueError("sealed physical high teacher required")
        with np.load(path, allow_pickle=False) as arrays:
            features = np.asarray(arrays["features"], dtype=np.float64)
        historical = next(
            item["summary"]
            for item in mapping["rows"]
            if item["expert"] == "high"
            and item["summary"]["course"]["seed"] == row["course"]["seed"]
        )
        references.append(
            ReceivingFootPhaseReference(
                "high",
                float(features[0, 1] * 0.2),
                float(features[0, 3] * 2.0),
                int(historical["first_foot_frame"]),
                tuple(tuple(float(value) for value in frame) for frame in features),
            )
        )
    coordination = tuple(parent["selected"]["weights"])
    left = tuple(refine["best"]["left_weights"])
    right = tuple(float(value) for value in np.asarray(right_parent["best_right"]["weights"]) * 0.7)
    slope = tuple(lateral["best"]["slope"])
    low_weights = tuple(float(value) for value in v214["low_weights"])
    protected = tuple(
        tuple(float(value) for value in row) for row in v216["protected_initial_features"]
    )
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_protected_neural_fresh_v217.py",
            "scripts/rsi_r1_protected_composed_neural_v216.py",
            "scripts/rsi_r1_composed_contact_fresh_v214.py",
            "src/rosclaw_soccer/rsi/receiving_protected_composed_neural.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    output.mkdir(parents=True)
    parent_tasks = [
        (
            asset_root,
            policy_path,
            course,
            coordination,
            left,
            right,
            slope,
            knots,
            tuple(references),
            low_weights,
        )
        for course in FRESH_COURSES
    ]
    candidate_tasks = [(*task, weights, protected) for task in parent_tasks]
    with ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        baseline = list(pool.map(_run_parent, parent_tasks))
        candidate = list(pool.map(_run_candidate, candidate_tasks))
    baseline_score = _score(baseline)
    candidate_score = _score(candidate)
    qualified = (
        candidate_score[0] >= 6
        and candidate_score[1] >= 7
        and candidate_score[0] >= baseline_score[0] + 2
    )
    report = {
        "schema": SCHEMA,
        "v216_report_hash": v216["report_hash"],
        "source_hashes": sources,
        "partition": "PREDECLARED_NEW_FRESH_PAIRED_PARENT_AND_PROTECTED_NEURAL",
        "fresh_bank_hash": fresh_hash,
        "fresh_courses": [vars(course) for course in FRESH_COURSES],
        "parent": baseline,
        "parent_score": baseline_score,
        "candidate": candidate,
        "candidate_score": candidate_score,
        "status": "FRESH_PROTECTED_NEURAL_QUALIFIED_FOR_NEXT_CHAIN_GATE"
        if qualified
        else "REJECTED_PROTECTED_NEURAL_FRESH_GATE",
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during fresh protected neural exam")
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
        args.output,
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("status", "parent_score", "candidate_score", "report_hash")
            }
        )
    )


if __name__ == "__main__":
    main()
