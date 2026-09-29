"""SIM_ONLY old-skill retention and untouched-course exam for a learned 48D actor."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from rsi_r1_coherent_exploration_v198 import _checked
from rsi_r1_temporal_actor_critic_v193 import clean, run_episode, score
from rsi_r1_temporal_self_imitation_v196 import FRESH_COURSES
from rsi_r1_temporal_zero_v192 import TRAIN_COURSES

from rosclaw_soccer.rsi.receiving_kinematic_temporal_expert import KinematicMotorWeights
from rosclaw_soccer.rsi.receiving_temporal_motor_expert import TemporalMotorWeights
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses

SCHEMA = "rosclaw_soccer.rsi.r1_protected_kinematic_fresh_v202.result.v1"


def _load_policy(path: Path, expected_hash: str) -> KinematicMotorWeights:
    if hash_bytes(path.read_bytes()) != expected_hash:
        raise ValueError("sealed learned 48D policy required")
    with np.load(path, allow_pickle=False) as arrays:
        if set(arrays.files) != {"input_matrix", "input_bias", "output_matrix", "output_bias"}:
            raise ValueError("safe four-array kinematic policy required")
        return KinematicMotorWeights(
            tuple(float(value) for value in arrays["input_matrix"].reshape(-1)),
            tuple(float(value) for value in arrays["input_bias"].reshape(-1)),
            tuple(float(value) for value in arrays["output_matrix"].reshape(-1)),
            tuple(float(value) for value in arrays["output_bias"].reshape(-1)),
        )


def _run_task(task: tuple[Any, ...]) -> dict[str, Any]:
    return run_episode(*task)


def _tasks(
    asset_root: Path,
    policy_path: Path,
    courses: tuple[Any, ...],
    coordination: tuple[float, ...],
    left: tuple[float, ...],
    right: tuple[float, ...],
    slope: tuple[float, ...],
    weights: KinematicMotorWeights | None,
    protected: tuple[tuple[float, ...], ...] | None,
) -> list[tuple[Any, ...]]:
    return [
        (
            asset_root,
            policy_path,
            course,
            coordination,
            left,
            right,
            slope,
            TemporalMotorWeights(),
            0.0,
            0,
            0.0,
            weights,
            protected,
        )
        for course in courses
    ]


def exam(
    asset_root: Path,
    policy_path: Path,
    parent_report: Path,
    right_report: Path,
    refine_report: Path,
    lateral_report: Path,
    zero_report_path: Path,
    v201_report_path: Path,
    learned_policy: Path,
    output: Path,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY untouched-course exam required")
    development_hash = preflight_receiving_courses(TRAIN_COURSES)
    fresh_hash = preflight_receiving_courses(FRESH_COURSES)
    parent, right_parent, refine, lateral, zero, learned = (
        _checked(path)
        for path in (
            parent_report,
            right_report,
            refine_report,
            lateral_report,
            zero_report_path,
            v201_report_path,
        )
    )
    candidate = next((row for row in learned["candidates"] if row["alpha"] == 0.15), None)
    if (
        candidate is None
        or learned["status"] != "REJECTED_KINEMATIC_ADVANTAGE_GATE"
        or tuple(candidate["score"][:3]) != (0, 4, 10)
        or zero["course_bank_hash"] != development_hash
        or parent["policy_hash"] != hash_bytes(policy_path.read_bytes())
    ):
        raise ValueError("sealed 4/16 non-retaining learned candidate required")
    weights = _load_policy(learned_policy, candidate["policy_hash"])
    protected = tuple(
        tuple(float(value) for value in row["observed_features"][0]) for row in zero["rows"][-2:]
    )
    coordination = tuple(parent["selected"]["weights"])
    left = tuple(refine["best"]["left_weights"])
    right = tuple(float(value) for value in np.asarray(right_parent["best_right"]["weights"]) * 0.7)
    slope = tuple(lateral["best"]["slope"])
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_protected_kinematic_fresh_v202.py",
            "scripts/rsi_r1_temporal_actor_critic_v193.py",
            "src/rosclaw_soccer/rsi/receiving_kinematic_temporal_expert.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    output.mkdir(parents=True)
    with ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        development = list(
            pool.map(
                _run_task,
                _tasks(
                    asset_root,
                    policy_path,
                    TRAIN_COURSES,
                    coordination,
                    left,
                    right,
                    slope,
                    weights,
                    protected,
                ),
            )
        )
        retention_equal = all(
            row["protected_episode"] is True
            and row["physical_trace_hash"] == old["physical_trace_hash"]
            for row, old in zip(development[-2:], zero["rows"][-2:], strict=True)
        )
        development_score = score(development)
        candidate_fresh: list[dict[str, Any]] = []
        parent_fresh: list[dict[str, Any]] = []
        if retention_equal and development_score[1] >= 4:
            candidate_fresh = list(
                pool.map(
                    _run_task,
                    _tasks(
                        asset_root,
                        policy_path,
                        FRESH_COURSES,
                        coordination,
                        left,
                        right,
                        slope,
                        weights,
                        protected,
                    ),
                )
            )
            parent_fresh = list(
                pool.map(
                    _run_task,
                    _tasks(
                        asset_root,
                        policy_path,
                        FRESH_COURSES,
                        coordination,
                        left,
                        right,
                        slope,
                        None,
                        None,
                    ),
                )
            )
    candidate_pass = sum(clean(row) and row["controlled_reception"] for row in candidate_fresh)
    parent_pass = sum(clean(row) and row["controlled_reception"] for row in parent_fresh)
    qualified = (
        retention_equal
        and development_score[1] >= 4
        and candidate_pass == len(FRESH_COURSES)
        and candidate_pass > parent_pass
    )
    report = {
        "schema": SCHEMA,
        "learned_report_hash": learned["report_hash"],
        "source_hashes": sources,
        "partition": "PROTECTED_CONSUMED_DEVELOPMENT_THEN_PREDECLARED_UNTOUCHED_FRESH",
        "development_bank_hash": development_hash,
        "fresh_bank_hash": fresh_hash,
        "fresh_courses": [vars(course) for course in FRESH_COURSES],
        "learned_policy_hash": candidate["policy_hash"],
        "development_score": development_score,
        "development": development,
        "retention_physical_equal": retention_equal,
        "fresh_candidate": candidate_fresh,
        "fresh_parent": parent_fresh,
        "fresh_candidate_pass": candidate_pass,
        "fresh_parent_pass": parent_pass,
        "status": "QUALIFIED_LOCAL_KINEMATIC_CANDIDATE_ONLY"
        if qualified
        else "REJECTED_PROTECTED_KINEMATIC_FRESH_GATE",
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during protected kinematic exam")
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
        "zero-report",
        "v201-report",
        "learned-policy",
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
        args.zero_report,
        args.v201_report,
        args.learned_policy,
        args.output,
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "development_score",
                    "retention_physical_equal",
                    "fresh_candidate_pass",
                    "fresh_parent_pass",
                    "report_hash",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
