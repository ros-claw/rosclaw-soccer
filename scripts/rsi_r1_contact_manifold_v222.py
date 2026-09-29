"""SIM_ONLY dense receiving state-domain mapping, explicitly development-only."""

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
from rsi_r1_measured_skill_router_v207 import _score, clean
from rsi_r1_protected_kinematic_fresh_v202 import _load_policy
from rsi_r1_protected_online_ppo_v218 import _run_task

from rosclaw_soccer.rsi.receiving_foot_phase_router import ReceivingFootPhaseReference
from rosclaw_soccer.rsi.receiving_measured_skill_router import ReceivingSkillKnot
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses
from rosclaw_soccer.training.role_receiving_courses import ReceivingCourse

SCHEMA = "rosclaw_soccer.rsi.r1_contact_manifold_v222.result.v1"
COURSES = tuple(
    ReceivingCourse("red.finisher", 222001 + i * 4 + j, speed, lateral)
    for i, lateral in enumerate((0.065, 0.069, 0.073))
    for j, speed in enumerate((1.18, 1.25, 1.30, 1.35))
)


def context(
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
) -> tuple[tuple[Any, ...], tuple[Any, ...], dict[str, Any]]:
    parent, right_parent, refine, lateral, v205, mapping, v214, v215, v216, v218, v220 = (
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
            v218_dir / "report.json",
            v220_report_path,
        )
    )
    if (
        v220["status"] != "REJECTED_HIGH_CONTACT_CEM_GATE"
        or v216["v215_report_hash"] != v215["report_hash"]
        or v218["v216_report_hash"] != v216["report_hash"]
        or v205["map_report_hash"] != mapping["report_hash"]
        or parent["policy_hash"] != hash_bytes(policy_path.read_bytes())
    ):
        raise ValueError("sealed baseline and learned contact lineage required")
    base = _load_policy(
        v215_dir / "update-3.npz",
        next(row for row in v215["history"] if row["update"] == 3)["checkpoint_hash"],
    )
    online = _load_policy(
        v218_dir / "update-3.npz",
        next(row for row in v218["history"] if row["update"] == 3)["checkpoint_hash"],
    )
    best = next(row for row in v220["history"] if row["generation"] == v220["best_generation"])
    high = replace(
        base,
        output_bias=tuple(
            float(value) for value in np.asarray(base.output_bias) + np.asarray(best["best_offset"])
        ),
    )
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
    protected = tuple(
        tuple(float(value) for value in row) for row in v216["protected_initial_features"]
    )
    common = (
        asset_root,
        policy_path,
        tuple(parent["selected"]["weights"]),
        tuple(refine["best"]["left_weights"]),
        tuple(float(value) for value in np.asarray(right_parent["best_right"]["weights"]) * 0.7),
        tuple(lateral["best"]["slope"]),
        knots,
        tuple(references),
        tuple(float(value) for value in v214["low_weights"]),
        protected,
    )
    return (
        common,
        (base, online, high),
        {
            "v220_report_hash": v220["report_hash"],
            "v218_report_hash": v218["report_hash"],
        },
    )


def tasks(
    common: tuple[Any, ...], courses: tuple[ReceivingCourse, ...], weights: Any
) -> list[tuple[Any, ...]]:
    return [
        (
            common[0],
            common[1],
            course,
            *common[2:-1],
            weights,
            common[-1],
            0,
            0.0,
        )
        for course in courses
    ]


def run(
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
    output: Path,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY manifold evidence required")
    bank_hash = preflight_receiving_courses(COURSES)
    common, policies, lineage = context(
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
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_contact_manifold_v222.py",
            "scripts/rsi_r1_protected_online_ppo_v218.py",
            "src/rosclaw_soccer/rsi/receiving_protected_composed_neural.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    output.mkdir(parents=True)
    labels = ("baseline", "online", "high_contact")
    with ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        runs = {
            label: list(pool.map(_run_task, tasks(common, COURSES, policy)))
            for label, policy in zip(labels, policies, strict=True)
        }
    union = [
        course.seed
        for index, course in enumerate(COURSES)
        if any(
            clean(runs[label][index]) and runs[label][index]["controlled_reception"]
            for label in labels
        )
    ]
    report = {
        "schema": SCHEMA,
        **lineage,
        "source_hashes": sources,
        "partition": "DEVELOPMENT_CONTACT_MANIFOLD_NOT_FRESH_EXAM",
        "bank_hash": bank_hash,
        "courses": [vars(course) for course in COURSES],
        "rows": runs,
        "scores": {label: _score(rows) for label, rows in runs.items()},
        "oracle_union_seeds": union,
        "status": "DEVELOPMENT_CONTACT_MANIFOLD_ONLY",
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during manifold mapping")
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
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    report = run(
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
        args.output,
    )
    print(json.dumps({key: report[key] for key in ("scores", "oracle_union_seeds", "report_hash")}))


if __name__ == "__main__":
    main()
