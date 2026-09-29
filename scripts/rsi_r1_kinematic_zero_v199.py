"""SIM_ONLY native zero-action equivalence for 48D measured-limb neural policy."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from rsi_r1_temporal_actor_critic_v193 import run_episode
from rsi_r1_temporal_zero_v192 import TRAIN_COURSES

from rosclaw_soccer.rsi.receiving_kinematic_temporal_expert import KinematicMotorWeights
from rosclaw_soccer.rsi.receiving_temporal_motor_expert import TemporalMotorWeights
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses

SCHEMA = "rosclaw_soccer.rsi.r1_kinematic_zero_v199.result.v1"


def _checked(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value["report_hash"] != hash_json(
        {key: item for key, item in value.items() if key != "report_hash"}
    ):
        raise ValueError(f"unsealed prerequisite: {path}")
    return value


def _run_task(task: tuple[Any, ...]) -> dict[str, Any]:
    return run_episode(*task)


def validate(
    asset_root: Path,
    policy_path: Path,
    parent_report: Path,
    right_report: Path,
    refine_report: Path,
    lateral_report: Path,
    zero_report_path: Path,
    output: Path,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY proprioceptive zero evidence required")
    bank_hash = preflight_receiving_courses(TRAIN_COURSES)
    parent, right_parent, refine, lateral, old = (
        _checked(path)
        for path in (
            parent_report,
            right_report,
            refine_report,
            lateral_report,
            zero_report_path,
        )
    )
    if (
        old["course_bank_hash"] != bank_hash
        or len(old["rows"]) != len(TRAIN_COURSES)
        or parent["policy_hash"] != hash_bytes(policy_path.read_bytes())
    ):
        raise ValueError("sealed original zero policy and native course bank required")
    coordination = tuple(parent["selected"]["weights"])
    left = tuple(refine["best"]["left_weights"])
    right = tuple(
        float(v) for v in np.asarray(right_parent["best_right"]["weights"], dtype=np.float64) * 0.7
    )
    slope = tuple(lateral["best"]["slope"])
    kinematic = KinematicMotorWeights()
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_kinematic_zero_v199.py",
            "scripts/rsi_r1_temporal_actor_critic_v193.py",
            "src/rosclaw_soccer/rsi/receiving_kinematic_temporal_expert.py",
            "src/rosclaw_soccer/rsi/receiving_temporal_motor_expert.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    tasks = [
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
            kinematic,
        )
        for course in TRAIN_COURSES
    ]
    output.mkdir(parents=True)
    rows = []
    with ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for index, episode in enumerate(pool.map(_run_task, tasks)):
            old_row = old["rows"][index]
            features = np.asarray(episode["observed_features"], dtype=np.float64)
            logits = np.asarray(episode["sampled_logits"], dtype=np.float64)
            if (
                episode["course"] != old_row["course"]
                or features.shape != (50, 48)
                or logits.shape != (50, 12)
                or not np.isfinite(features).all()
                or not np.isfinite(logits).all()
                or not np.array_equal(logits, np.zeros_like(logits))
            ):
                raise ValueError("complete zero-output measured-limb trace required")
            feature_path = output / f"features-{TRAIN_COURSES[index].seed}.npz"
            np.savez_compressed(feature_path, features=features)
            equal = episode["physical_trace_hash"] == old_row["physical_trace_hash"]
            rows.append(
                {
                    "course": episode["course"],
                    "physical_trace_hash": episode["physical_trace_hash"],
                    "old_physical_trace_hash": old_row["physical_trace_hash"],
                    "physical_equal": equal,
                    "feature_hash": hash_bytes(feature_path.read_bytes()),
                    "feature_shape": list(features.shape),
                    "safe": episode["safe"],
                    "controlled_reception": episode["controlled_reception"],
                }
            )
            (output / "progress.json").write_text(json.dumps(rows, indent=2) + "\n")
            print(json.dumps({"course": index, "physical_equal": equal}), flush=True)
    report = {
        "schema": SCHEMA,
        "old_zero_report_hash": old["report_hash"],
        "source_hashes": sources,
        "partition": "CONSUMED_EIGHT_G1_KINEMATIC_ZERO_NOT_FRESH",
        "course_bank_hash": bank_hash,
        "kinematic_policy_hash": kinematic.contract_hash,
        "rows": rows,
        "physical_equal_count": sum(row["physical_equal"] for row in rows),
        "status": "KINEMATIC_ZERO_PHYSICS_EQUIVALENT"
        if all(row["physical_equal"] for row in rows)
        else "REJECTED_KINEMATIC_ZERO_DRIFT",
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during kinematic zero validation")
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
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    report = validate(
        args.asset_root,
        args.policy,
        args.parent_report,
        args.right_report,
        args.refine_report,
        args.lateral_report,
        args.zero_report,
        args.output,
    )
    print(
        json.dumps({key: report[key] for key in ("status", "physical_equal_count", "report_hash")})
    )


if __name__ == "__main__":
    main()
