"""SIM_ONLY hard-gated native positive-sample mining at one consumed ball position."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from rsi_r1_side_navigation_fresh_v160 import FRESH_COURSES, clean, run_candidate
from rsi_r1_taskspace_feedback_v125 import COURSES

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

SCHEMA = "rosclaw_soccer.rsi.r1_left_positive_mining_v163.result.v1"
POPULATION = 12
GENERATIONS = 2


def rank(row: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(clean(row) and row["controlled_reception"]),
        float(clean(row)),
        float(row["safe"] and not row["fault_agents"]),
        -float(len(row["own_nonfoot_frames"])),
        -float(row["tail_maximum_foot_distance_m"]),
        -float(row["tail_maximum_ball_speed_mps"]),
    )


def train(
    asset_root: Path,
    policy: Path,
    parent_report: Path,
    right_report: Path,
    joint_report: Path,
    conditional_report: Path,
    output: Path,
    *,
    course_index: int,
    seed: int,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if (
        output.exists()
        or output.resolve().is_relative_to(root)
        or course_index not in (0, 1)
        or not 0 <= seed < 2**31
    ):
        raise ValueError("new external SIM_ONLY single-course mining evidence required")
    parent, right_parent, joint, conditional = (
        json.loads(path.read_text())
        for path in (parent_report, right_report, joint_report, conditional_report)
    )
    if any(
        item["report_hash"] != hash_json({k: v for k, v in item.items() if k != "report_hash"})
        for item in (parent, right_parent, joint, conditional)
    ) or (
        conditional["status"] != "REJECTED_CONDITIONAL_MULTICOURSE_GATE"
        or joint["status"] != "DEVELOPMENT_NATIVE_BILATERAL_CONTROLLED_UNVALIDATED"
        or parent["policy_hash"] != hash_bytes(policy.read_bytes())
    ):
        raise ValueError("sealed failed-conditional curriculum required")
    course = FRESH_COURSES[course_index]
    coordination = tuple(parent["selected"]["weights"])
    base = np.asarray(joint["best"]["left_weights"], dtype=np.float64)
    right = tuple(
        float(v) for v in np.asarray(right_parent["best_right"]["weights"], dtype=np.float64) * 0.7
    )
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_left_positive_mining_v163.py",
            "scripts/rsi_r1_side_navigation_fresh_v160.py",
            "src/rosclaw_soccer/rsi/team_receive_side_navigation.py",
        )
    }
    output.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    mean = base.copy()
    std = np.full(12, 0.18)
    generations: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for generation in range(GENERATIONS):
        candidates = np.clip(rng.normal(mean, std, size=(POPULATION, 12)), -1.0, 1.0)
        candidates[0] = base if generation == 0 else mean
        rows: list[dict[str, Any]] = []
        for index, candidate in enumerate(candidates):
            left = tuple(float(value) for value in candidate)
            summary = run_candidate(asset_root, policy, course, coordination, left, right)
            row = {
                "generation": generation + 1,
                "candidate": index,
                "left_weights": left,
                "summary": summary,
            }
            rows.append(row)
            if best is None or rank(summary) > rank(best["summary"]):
                best = row
            (output / "progress.json").write_text(
                json.dumps(
                    [*generations, {"generation": generation + 1, "rows": rows}],
                    indent=2,
                    allow_nan=False,
                )
                + "\n"
            )
            print(
                json.dumps(
                    {
                        "course": vars(course),
                        "generation": generation + 1,
                        "candidate": index,
                        "clean": clean(summary),
                        "controlled": summary["controlled_reception"],
                        "nonfoot": summary["own_nonfoot_frames"],
                        "distance": summary["tail_maximum_foot_distance_m"],
                        "speed": summary["tail_maximum_ball_speed_mps"],
                    }
                ),
                flush=True,
            )
        rows.sort(key=lambda item: rank(item["summary"]), reverse=True)
        elite = np.stack([np.asarray(row["left_weights"]) for row in rows[:4]])
        mean = elite.mean(axis=0)
        std = np.maximum(0.05, elite.std(axis=0))
        generations.append({"generation": generation + 1, "rows": rows})
    assert best is not None
    selected = tuple(best["left_weights"])
    selected_check = run_candidate(asset_root, policy, course, coordination, selected, right)
    original_check = run_candidate(asset_root, policy, COURSES[0], coordination, selected, right)
    positive = clean(selected_check) and selected_check["controlled_reception"]
    report = {
        "schema": SCHEMA,
        "conditional_report_hash": conditional["report_hash"],
        "source_hashes": sources,
        "partition": "CONSUMED_NATIVE_EIGHT_G1_SINGLE_COURSE_POSITIVE_MINING",
        "course_index": course_index,
        "course": vars(course),
        "seed": seed,
        "generations": generations,
        "best": best,
        "selected_check": selected_check,
        "other_consumed_check": original_check,
        "status": "DEVELOPMENT_SINGLE_COURSE_POSITIVE_UNVALIDATED"
        if positive
        else "REJECTED_SINGLE_COURSE_POSITIVE_GATE",
        "fresh_exam_required": True,
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during positive mining")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--right-report", type=Path, required=True)
    parser.add_argument("--joint-report", type=Path, required=True)
    parser.add_argument("--conditional-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--course-index", type=int, required=True)
    parser.add_argument("--seed", type=int, default=163928)
    args = parser.parse_args()
    report = train(
        args.asset_root,
        args.policy,
        args.parent_report,
        args.right_report,
        args.joint_report,
        args.conditional_report,
        args.output,
        course_index=args.course_index,
        seed=args.seed,
    )
    print(json.dumps({"status": report["status"], "report_hash": report["report_hash"]}))


if __name__ == "__main__":
    main()
