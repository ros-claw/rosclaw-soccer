"""SIM_ONLY multi-course left receiving curriculum after failed fresh exam."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from rsi_r1_side_navigation_fresh_v160 import FRESH_COURSES, clean, run_candidate
from rsi_r1_taskspace_feedback_v125 import COURSES

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.role_receiving_courses import ReceivingCourse

SCHEMA = "rosclaw_soccer.rsi.r1_multicourse_left_v161.result.v1"
POPULATION = 8
GENERATIONS = 2
TRAIN_COURSES: tuple[ReceivingCourse, ...] = (
    COURSES[0],
    FRESH_COURSES[0],
    FRESH_COURSES[1],
)


def rank(summaries: list[dict[str, Any]]) -> tuple[float, ...]:
    return (
        float(sum(clean(row) and row["controlled_reception"] for row in summaries)),
        float(sum(clean(row) for row in summaries)),
        float(sum(row["safe"] and not row["fault_agents"] for row in summaries)),
        -float(sum(len(row["own_nonfoot_frames"]) for row in summaries)),
        -float(max(row["tail_maximum_foot_distance_m"] for row in summaries)),
        -float(max(row["tail_maximum_ball_speed_mps"] for row in summaries)),
    )


def train(
    asset_root: Path,
    policy: Path,
    parent_report: Path,
    right_report: Path,
    joint_report: Path,
    fresh_report: Path,
    output: Path,
    *,
    seed: int,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root) or not 0 <= seed < 2**31:
        raise ValueError("new external SIM_ONLY multi-course evidence required")
    parent, right_parent, joint, fresh = (
        json.loads(path.read_text())
        for path in (parent_report, right_report, joint_report, fresh_report)
    )
    if any(
        item["report_hash"] != hash_json({k: v for k, v in item.items() if k != "report_hash"})
        for item in (parent, right_parent, joint, fresh)
    ) or (
        joint["status"] != "DEVELOPMENT_NATIVE_BILATERAL_CONTROLLED_UNVALIDATED"
        or fresh["status"] != "REJECTED_FRESH_RECEIVING_GATE"
        or not fresh["local_equivalent"]
        or not fresh["side_aligned"]
        or parent["policy_hash"] != hash_bytes(policy.read_bytes())
    ):
        raise ValueError("sealed failed-fresh curriculum lineage required")
    coordination = tuple(parent["selected"]["weights"])
    base = np.asarray(joint["best"]["left_weights"], dtype=np.float64)
    right = tuple(
        float(v) for v in np.asarray(right_parent["best_right"]["weights"], dtype=np.float64) * 0.7
    )
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_multicourse_left_v161.py",
            "scripts/rsi_r1_side_navigation_fresh_v160.py",
            "src/rosclaw_soccer/rsi/team_receive_side_navigation.py",
        )
    }
    output.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    mean = base.copy()
    std = np.full(12, 0.10)
    generations: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for generation in range(GENERATIONS):
        candidates = np.clip(rng.normal(mean, std, size=(POPULATION, 12)), -1.0, 1.0)
        candidates[0] = base if generation == 0 else mean
        rows: list[dict[str, Any]] = []
        for index, candidate in enumerate(candidates):
            left = tuple(float(value) for value in candidate)
            summaries = []
            for course in TRAIN_COURSES:
                summaries.append(
                    run_candidate(asset_root, policy, course, coordination, left, right)
                )
                (output / "progress.json").write_text(
                    json.dumps(
                        [
                            *generations,
                            {"generation": generation + 1, "rows": rows},
                            {
                                "candidate": index,
                                "left_weights": left,
                                "partial_summaries": summaries,
                            },
                        ],
                        indent=2,
                        allow_nan=False,
                    )
                    + "\n"
                )
            row = {
                "generation": generation + 1,
                "candidate": index,
                "left_weights": left,
                "summaries": summaries,
            }
            rows.append(row)
            if best is None or rank(summaries) > rank(best["summaries"]):
                best = row
            print(
                json.dumps(
                    {
                        "generation": generation + 1,
                        "candidate": index,
                        "clean": [clean(item) for item in summaries],
                        "controlled": [item["controlled_reception"] for item in summaries],
                        "nonfoot": [item["own_nonfoot_frames"] for item in summaries],
                        "distance": [item["tail_maximum_foot_distance_m"] for item in summaries],
                        "speed": [item["tail_maximum_ball_speed_mps"] for item in summaries],
                    }
                ),
                flush=True,
            )
        rows.sort(key=lambda item: rank(item["summaries"]), reverse=True)
        elite = np.stack([np.asarray(row["left_weights"]) for row in rows[:3]])
        mean = elite.mean(axis=0)
        std = np.maximum(0.035, elite.std(axis=0))
        generations.append({"generation": generation + 1, "rows": rows})
    assert best is not None
    selected_left = tuple(best["left_weights"])
    left_checks = [
        run_candidate(asset_root, policy, course, coordination, selected_left, right)
        for course in TRAIN_COURSES
    ]
    right_checks = [
        run_candidate(asset_root, policy, course, coordination, selected_left, right)
        for course in (COURSES[1], FRESH_COURSES[2], FRESH_COURSES[3])
    ]
    all_controlled = all(
        clean(row) and row["controlled_reception"] for row in left_checks + right_checks
    )
    report = {
        "schema": SCHEMA,
        "fresh_report_hash": fresh["report_hash"],
        "source_hashes": sources,
        "partition": "CONSUMED_NATIVE_EIGHT_G1_MULTICOURSE_CURRICULUM",
        "seed": seed,
        "train_courses": [vars(course) for course in TRAIN_COURSES],
        "generations": generations,
        "best": best,
        "left_checks": left_checks,
        "right_checks": right_checks,
        "status": "DEVELOPMENT_MULTICOURSE_CONTROLLED_UNVALIDATED"
        if all_controlled
        else "REJECTED_MULTICOURSE_CONTROLLED_GATE",
        "fresh_exam_required": True,
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during multi-course learning")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--right-report", type=Path, required=True)
    parser.add_argument("--joint-report", type=Path, required=True)
    parser.add_argument("--fresh-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=161928)
    args = parser.parse_args()
    report = train(
        args.asset_root,
        args.policy,
        args.parent_report,
        args.right_report,
        args.joint_report,
        args.fresh_report,
        args.output,
        seed=args.seed,
    )
    print(json.dumps({"status": report["status"], "report_hash": report["report_hash"]}))


if __name__ == "__main__":
    main()
