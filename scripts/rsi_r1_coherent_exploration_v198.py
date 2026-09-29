"""SIM_ONLY paired native test of temporally correlated neural motor exploration."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from rsi_r1_course_map_v190 import COURSES
from rsi_r1_protected_temporal_fresh_v194 import load_weights
from rsi_r1_temporal_actor_critic_v193 import clean, reward, run_episode

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses

SCHEMA = "rosclaw_soccer.rsi.r1_coherent_exploration_v198.result.v1"
CORRELATION = 0.85
REPETITIONS = 8
EXPLORATION_STD = 0.12


def _checked(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text())
    if report["report_hash"] != hash_json(
        {key: value for key, value in report.items() if key != "report_hash"}
    ):
        raise ValueError(f"unsealed prerequisite report: {path}")
    return report


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
    checkpoint: Path,
    output: Path,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY coherent exploration evidence required")
    bank_hash = preflight_receiving_courses(COURSES)
    parent, right_parent, refine, lateral, v193, experience = (
        _checked(path)
        for path in (
            parent_report,
            right_report,
            refine_report,
            lateral_report,
            v193_report_path,
            experience_dir / "report.json",
        )
    )
    checkpoint_hash = v193["history"][1]["checkpoint_hash"]
    if (
        experience["status"] != "DEVELOPMENT_TEMPORAL_EXPERIENCE_ONLY"
        or experience["checkpoint_hash"] != checkpoint_hash
        or parent["policy_hash"] != hash_bytes(policy_path.read_bytes())
        or experience["course_bank_hash"] != bank_hash
    ):
        raise ValueError("sealed prior eight-G1 experience lineage required")
    weights = load_weights(checkpoint, checkpoint_hash)
    coordination = tuple(parent["selected"]["weights"])
    left = tuple(refine["best"]["left_weights"])
    right = tuple(
        float(v) for v in np.asarray(right_parent["best_right"]["weights"], dtype=np.float64) * 0.7
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
            weights,
            EXPLORATION_STD,
            195000 + course_index * REPETITIONS + repetition,
            CORRELATION,
        )
        for course_index, course in enumerate(COURSES)
        for repetition in range(REPETITIONS)
    ]
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_coherent_exploration_v198.py",
            "scripts/rsi_r1_temporal_actor_critic_v193.py",
            "src/rosclaw_soccer/rsi/receiving_temporal_motor_expert.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    output.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=4, mp_context=context) as pool:
        for index, episode in enumerate(pool.map(_run_task, tasks)):
            course_index, repetition = divmod(index, REPETITIONS)
            trajectory_path = output / f"course-{COURSES[course_index].seed}-rep-{repetition}.npz"
            np.savez_compressed(
                trajectory_path,
                frames=np.asarray(episode["observed_frames"], dtype=np.int64),
                features=np.asarray(episode["observed_features"], dtype=np.float64).reshape(-1, 10),
                logits=np.asarray(episode["sampled_logits"], dtype=np.float64).reshape(-1, 12),
                shaped_reward=np.asarray(episode["shaped_reward"], dtype=np.float64),
            )
            summary = {
                key: value
                for key, value in episode.items()
                if key
                not in ("observed_frames", "observed_features", "sampled_logits", "shaped_reward")
            }
            summary.update(
                {
                    "course_index": course_index,
                    "repetition": repetition,
                    "exploration_seed": tasks[index][9],
                    "trajectory_hash": hash_bytes(trajectory_path.read_bytes()),
                    "clean": clean(episode),
                    "strict_controlled": bool(clean(episode) and episode["controlled_reception"]),
                    "training_reward": reward(episode),
                }
            )
            rows.append(summary)
            (output / "progress.json").write_text(json.dumps(rows, indent=2) + "\n")
            if repetition == REPETITIONS - 1:
                block = rows[-REPETITIONS:]
                print(
                    json.dumps(
                        {
                            "course": course_index,
                            "iid_controlled": sum(
                                old["strict_controlled"]
                                for old in experience["rows"][index + 1 - REPETITIONS : index + 1]
                            ),
                            "coherent_controlled": sum(row["strict_controlled"] for row in block),
                            "coherent_clean": sum(row["clean"] for row in block),
                        }
                    ),
                    flush=True,
                )
    old_rows = experience["rows"]
    paired = [
        {
            "course_index": index,
            "iid_controlled": sum(
                row["strict_controlled"] for row in old_rows[index * 8 : index * 8 + 8]
            ),
            "coherent_controlled": sum(
                row["strict_controlled"] for row in rows[index * 8 : index * 8 + 8]
            ),
        }
        for index in range(len(COURSES))
    ]
    total = sum(row["strict_controlled"] for row in rows)
    old_total = sum(row["strict_controlled"] for row in old_rows)
    report = {
        "schema": SCHEMA,
        "experience_report_hash": experience["report_hash"],
        "source_hashes": sources,
        "partition": "PAIRED_CONSUMED_EIGHT_G1_EXPLORATION_NOT_FRESH",
        "course_bank_hash": bank_hash,
        "repetitions": REPETITIONS,
        "exploration_std": EXPLORATION_STD,
        "exploration_correlation": CORRELATION,
        "checkpoint_hash": checkpoint_hash,
        "rows": rows,
        "paired_course_counts": paired,
        "iid_controlled": old_total,
        "coherent_controlled": total,
        "coherent_clean": sum(row["clean"] for row in rows),
        "status": "EXPLORATION_STUDY_ONLY",
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during coherent exploration")
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
        args.checkpoint,
        args.output,
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("status", "iid_controlled", "coherent_controlled", "report_hash")
            }
        )
    )


if __name__ == "__main__":
    main()
