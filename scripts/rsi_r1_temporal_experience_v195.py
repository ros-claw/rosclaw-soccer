"""SIM_ONLY repeated native temporal-policy experience, not a fresh exam."""

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

SCHEMA = "rosclaw_soccer.rsi.r1_temporal_experience_v195.result.v1"
REPETITIONS = 8
EXPLORATION_STD = 0.12


def _run_task(args: tuple[Any, ...]) -> dict[str, Any]:
    return run_episode(*args)


def collect(
    asset_root: Path,
    policy_path: Path,
    parent_report: Path,
    right_report: Path,
    refine_report: Path,
    lateral_report: Path,
    v193_report_path: Path,
    v194_report_path: Path,
    checkpoint: Path,
    output: Path,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY temporal experience required")
    bank_hash = preflight_receiving_courses(COURSES)
    parent, right_parent, refine, lateral, v193, v194 = (
        json.loads(path.read_text())
        for path in (
            parent_report,
            right_report,
            refine_report,
            lateral_report,
            v193_report_path,
            v194_report_path,
        )
    )
    if any(
        item["report_hash"] != hash_json({k: v for k, v in item.items() if k != "report_hash"})
        for item in (parent, right_parent, refine, lateral, v193, v194)
    ) or (
        v194["status"] != "REJECTED_PROTECTED_TEMPORAL_FRESH_GATE"
        or v194["v193_report_hash"] != v193["report_hash"]
        or parent["policy_hash"] != hash_bytes(policy_path.read_bytes())
    ):
        raise ValueError("sealed rejected temporal exam lineage required")
    checkpoint_hash = v193["history"][1]["checkpoint_hash"]
    weights = load_weights(checkpoint, checkpoint_hash)
    coordination = tuple(parent["selected"]["weights"])
    left = tuple(refine["best"]["left_weights"])
    right = tuple(
        float(v) for v in np.asarray(right_parent["best_right"]["weights"], dtype=np.float64) * 0.7
    )
    slope = tuple(lateral["best"]["slope"])
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_temporal_experience_v195.py",
            "scripts/rsi_r1_temporal_actor_critic_v193.py",
            "src/rosclaw_soccer/rsi/receiving_temporal_motor_expert.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    output.mkdir(parents=True)
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
        )
        for course_index, course in enumerate(COURSES)
        for repetition in range(REPETITIONS)
    ]
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
                not in (
                    "observed_frames",
                    "observed_features",
                    "sampled_logits",
                    "shaped_reward",
                )
            }
            summary.update(
                {
                    "course_index": course_index,
                    "repetition": repetition,
                    "exploration_seed": tasks[index][-1],
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
                            "course": vars(COURSES[course_index]),
                            "clean": sum(row["clean"] for row in block),
                            "controlled": sum(row["strict_controlled"] for row in block),
                            "early": sum(row["early_termination"] for row in block),
                        }
                    ),
                    flush=True,
                )
    coverage = sum(
        any(row["strict_controlled"] for row in rows[i : i + REPETITIONS])
        for i in range(0, len(rows), REPETITIONS)
    )
    repeated_coverage = sum(
        sum(row["strict_controlled"] for row in rows[i : i + REPETITIONS]) >= 2
        for i in range(0, len(rows), REPETITIONS)
    )
    report = {
        "schema": SCHEMA,
        "v194_report_hash": v194["report_hash"],
        "source_hashes": sources,
        "partition": "REPEATED_CONSUMED_EIGHT_G1_ONLINE_EXPERIENCE_NOT_FRESH",
        "course_bank_hash": bank_hash,
        "course_count": len(COURSES),
        "repetitions": REPETITIONS,
        "exploration_std": EXPLORATION_STD,
        "checkpoint_hash": checkpoint_hash,
        "rows": rows,
        "total_clean": sum(row["clean"] for row in rows),
        "total_controlled": sum(row["strict_controlled"] for row in rows),
        "total_early_terminations": sum(row["early_termination"] for row in rows),
        "any_success_coverage": coverage,
        "repeated_success_coverage": repeated_coverage,
        "status": "DEVELOPMENT_TEMPORAL_EXPERIENCE_ONLY",
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during temporal experience collection")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--right-report", type=Path, required=True)
    parser.add_argument("--refine-report", type=Path, required=True)
    parser.add_argument("--lateral-report", type=Path, required=True)
    parser.add_argument("--v193-report", type=Path, required=True)
    parser.add_argument("--v194-report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = collect(
        args.asset_root,
        args.policy,
        args.parent_report,
        args.right_report,
        args.refine_report,
        args.lateral_report,
        args.v193_report,
        args.v194_report,
        args.checkpoint,
        args.output,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "total_controlled": report["total_controlled"],
                "any_success_coverage": report["any_success_coverage"],
                "repeated_success_coverage": report["repeated_success_coverage"],
                "report_hash": report["report_hash"],
            }
        )
    )


if __name__ == "__main__":
    main()
