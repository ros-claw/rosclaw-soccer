"""SIM_ONLY full legacy-policy replay while collecting measured 48D limb states."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from rsi_r1_coherent_exploration_v198 import _checked
from rsi_r1_course_map_v190 import COURSES
from rsi_r1_protected_temporal_fresh_v194 import load_weights
from rsi_r1_temporal_actor_critic_v193 import clean, run_episode

from rosclaw_soccer.rsi.receiving_kinematic_temporal_expert import KinematicMotorWeights
from rosclaw_soccer.rsi.receiving_temporal_motor_expert import TemporalMotorWeights
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses

SCHEMA = "rosclaw_soccer.rsi.r1_kinematic_replay_v200.result.v1"
REPETITIONS = 8
EXPLORATION_STD = 0.12


def _run_task(task: tuple[Any, ...]) -> dict[str, Any]:
    return run_episode(*task)


def collect(
    asset_root: Path,
    policy_path: Path,
    parent_report: Path,
    right_report: Path,
    refine_report: Path,
    lateral_report: Path,
    v193_report_path: Path,
    experience_dir: Path,
    zero_report_path: Path,
    checkpoint: Path,
    output: Path,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY kinematic replay evidence required")
    bank_hash = preflight_receiving_courses(COURSES)
    parent, right_parent, refine, lateral, v193, experience, zero = (
        _checked(path)
        for path in (
            parent_report,
            right_report,
            refine_report,
            lateral_report,
            v193_report_path,
            experience_dir / "report.json",
            zero_report_path,
        )
    )
    checkpoint_hash = v193["history"][1]["checkpoint_hash"]
    if (
        zero["status"] != "KINEMATIC_ZERO_PHYSICS_EQUIVALENT"
        or zero["physical_equal_count"] != 18
        or experience["status"] != "DEVELOPMENT_TEMPORAL_EXPERIENCE_ONLY"
        or experience["course_bank_hash"] != bank_hash
        or experience["checkpoint_hash"] != checkpoint_hash
        or parent["policy_hash"] != hash_bytes(policy_path.read_bytes())
    ):
        raise ValueError("sealed legacy-policy and zero-kinematic lineage required")
    legacy = load_weights(checkpoint, checkpoint_hash)
    kinematic = KinematicMotorWeights.from_legacy(legacy)
    coordination = tuple(parent["selected"]["weights"])
    left = tuple(refine["best"]["left_weights"])
    right = tuple(
        float(value)
        for value in np.asarray(right_parent["best_right"]["weights"], dtype=np.float64) * 0.7
    )
    slope = tuple(lateral["best"]["slope"])
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
            EXPLORATION_STD,
            195000 + course_index * REPETITIONS + repetition,
            0.0,
            kinematic,
        )
        for course_index, course in enumerate(COURSES)
        for repetition in range(REPETITIONS)
    ]
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_kinematic_replay_v200.py",
            "scripts/rsi_r1_temporal_actor_critic_v193.py",
            "src/rosclaw_soccer/rsi/receiving_kinematic_temporal_expert.py",
            "src/rosclaw_soccer/rsi/receiving_temporal_motor_expert.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    output.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for index, episode in enumerate(pool.map(_run_task, tasks)):
            course_index, repetition = divmod(index, REPETITIONS)
            features = np.asarray(episode["observed_features"], dtype=np.float64)
            logits = np.asarray(episode["sampled_logits"], dtype=np.float64)
            frames = np.asarray(episode["observed_frames"], dtype=np.int64)
            if (
                features.shape != (50, 48)
                or logits.shape != (50, 12)
                or not np.array_equal(frames, np.arange(15, 65))
                or not np.isfinite(features).all()
                or not np.isfinite(logits).all()
            ):
                raise ValueError("complete finite measured-limb exploration required")
            trajectory_path = output / f"course-{COURSES[course_index].seed}-rep-{repetition}.npz"
            np.savez_compressed(trajectory_path, frames=frames, features=features, logits=logits)
            old = experience["rows"][index]
            if old["course"] != episode["course"] or old["exploration_seed"] != tasks[index][9]:
                raise ValueError("unpaired legacy experience row")
            physical_equal = episode["physical_trace_hash"] == old["physical_trace_hash"]
            if not physical_equal:
                raise ValueError(
                    f"legacy kinematic physics drift at course {course_index}, "
                    f"repetition {repetition}"
                )
            rows.append(
                {
                    "course": episode["course"],
                    "course_index": course_index,
                    "repetition": repetition,
                    "exploration_seed": tasks[index][9],
                    "trajectory_hash": hash_bytes(trajectory_path.read_bytes()),
                    "physical_trace_hash": episode["physical_trace_hash"],
                    "legacy_physical_trace_hash": old["physical_trace_hash"],
                    "physical_equal": physical_equal,
                    "clean": clean(episode),
                    "strict_controlled": bool(clean(episode) and episode["controlled_reception"]),
                }
            )
            (output / "progress.json").write_text(json.dumps(rows, indent=2) + "\n")
            if repetition == REPETITIONS - 1:
                print(
                    json.dumps(
                        {
                            "course": course_index,
                            "physical_equal": sum(
                                row["physical_equal"] for row in rows[-REPETITIONS:]
                            ),
                        }
                    ),
                    flush=True,
                )
    equal_count = sum(row["physical_equal"] for row in rows)
    report = {
        "schema": SCHEMA,
        "experience_report_hash": experience["report_hash"],
        "kinematic_zero_report_hash": zero["report_hash"],
        "source_hashes": sources,
        "partition": "CONSUMED_EIGHT_G1_PAIRED_LEGACY_REPLAY_NOT_FRESH",
        "course_bank_hash": bank_hash,
        "checkpoint_hash": checkpoint_hash,
        "kinematic_policy_hash": kinematic.contract_hash,
        "rows": rows,
        "physical_equal_count": equal_count,
        "strict_controlled": sum(row["strict_controlled"] for row in rows),
        "status": "KINEMATIC_LEGACY_PHYSICS_EQUIVALENT"
        if equal_count == 128
        else "REJECTED_KINEMATIC_LEGACY_REPLAY_DRIFT",
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during kinematic legacy replay")
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
        "v193-report",
        "experience-dir",
        "zero-report",
        "checkpoint",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    report = collect(
        args.asset_root,
        args.policy,
        args.parent_report,
        args.right_report,
        args.refine_report,
        args.lateral_report,
        args.v193_report,
        args.experience_dir,
        args.zero_report,
        args.checkpoint,
        args.output,
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("status", "physical_equal_count", "strict_controlled", "report_hash")
            }
        )
    )


if __name__ == "__main__":
    main()
