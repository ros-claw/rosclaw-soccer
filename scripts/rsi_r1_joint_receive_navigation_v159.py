"""SIM_ONLY eight-G1 left contact and post-touch navigation co-adaptation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from rsi_r1_posttouch_navigation_v158 import POST_WEIGHTS, run
from rsi_r1_taskspace_feedback_v125 import COURSES

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

SCHEMA = "rosclaw_soccer.rsi.r1_joint_receive_navigation_v159.result.v1"
POPULATION = 12
GENERATIONS = 2


def rank(summary: dict[str, Any]) -> tuple[float, ...]:
    clean = (
        summary["safe"]
        and not summary["fault_agents"]
        and summary["first_foot_frame"] is not None
        and not summary["own_nonfoot_frames"]
    )
    return (
        float(clean and summary["controlled_reception"]),
        float(clean),
        float(summary["safe"] and not summary["fault_agents"]),
        float(summary["first_foot_frame"] is not None),
        -float(len(summary["own_nonfoot_frames"])),
        -float(summary["tail_maximum_foot_distance_m"]),
        -float(summary["tail_maximum_ball_speed_mps"]),
    )


def train(
    asset_root: Path,
    policy: Path,
    parent_report: Path,
    right_report: Path,
    native_report: Path,
    navigation_report: Path,
    output: Path,
    *,
    seed: int,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root) or not 0 <= seed < 2**31:
        raise ValueError("new bounded external SIM_ONLY joint-training evidence required")
    parent, right_parent, native, navigation = (
        json.loads(path.read_text())
        for path in (parent_report, right_report, native_report, navigation_report)
    )
    if any(
        item["report_hash"] != hash_json({k: v for k, v in item.items() if k != "report_hash"})
        for item in (parent, right_parent, native, navigation)
    ) or (
        not navigation["zero_physics_equal"]
        or not navigation["right_retained"]
        or navigation["status"] != "REJECTED_LEFT_CONTROLLED_GATE"
        or parent["policy_hash"] != hash_bytes(policy.read_bytes())
    ):
        raise ValueError("sealed left-course and retained-right lineage required")
    coordination = tuple(parent["selected"]["weights"])
    left_parent = np.asarray(native["best"]["weights"], dtype=np.float64)
    right = tuple(
        float(v) for v in np.asarray(right_parent["best_right"]["weights"], dtype=np.float64) * 0.7
    )
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_joint_receive_navigation_v159.py",
            "scripts/rsi_r1_posttouch_navigation_v158.py",
            "src/rosclaw_soccer/rsi/receiving_side_conditioned_expert.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
        )
    }
    output.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    mean = left_parent.copy()
    std = np.full(12, 0.12, dtype=np.float64)
    weights = POST_WEIGHTS[6]
    generations: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for generation in range(GENERATIONS):
        candidates = np.clip(rng.normal(mean, std, size=(POPULATION, 12)), -1.0, 1.0)
        candidates[0] = left_parent if generation == 0 else mean
        rows: list[dict[str, Any]] = []
        for index, candidate in enumerate(candidates):
            left = tuple(float(value) for value in candidate)
            summary, _ = run(asset_root, policy, COURSES[0], coordination, left, right, weights)
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
                        "generation": generation + 1,
                        "candidate": index,
                        "clean": rank(summary)[1] == 1.0,
                        "controlled": summary["controlled_reception"],
                        "nonfoot": summary["own_nonfoot_frames"],
                        "distance": summary["tail_maximum_foot_distance_m"],
                        "speed": summary["tail_maximum_ball_speed_mps"],
                    }
                ),
                flush=True,
            )
        rows.sort(key=lambda row: rank(row["summary"]), reverse=True)
        elite = np.stack([np.asarray(row["left_weights"]) for row in rows[:4]])
        mean = elite.mean(axis=0)
        std = np.maximum(0.04, elite.std(axis=0))
        generations.append({"generation": generation + 1, "rows": rows})
    assert best is not None
    selected_left = tuple(best["left_weights"])
    left_check, _ = run(asset_root, policy, COURSES[0], coordination, selected_left, right, weights)
    right_check, _ = run(asset_root, policy, COURSES[1], coordination, selected_left, right, None)
    right_retained = all(
        right_check[key] == navigation["right_parent"][key]
        for key in (
            "safe",
            "fault_agents",
            "first_foot_frame",
            "own_nonfoot_frames",
            "controlled_reception",
            "tail_maximum_foot_distance_m",
            "tail_maximum_ball_speed_mps",
        )
    )
    controlled = (
        left_check["safe"]
        and not left_check["fault_agents"]
        and left_check["first_foot_frame"] is not None
        and not left_check["own_nonfoot_frames"]
        and left_check["controlled_reception"]
    )
    report = {
        "schema": SCHEMA,
        "navigation_report_hash": navigation["report_hash"],
        "source_hashes": sources,
        "partition": "CONSUMED_NATIVE_EIGHT_G1_JOINT_LEFT_CONTACT_NAVIGATION",
        "seed": seed,
        "post_weights": weights,
        "generations": generations,
        "best": best,
        "left_check": left_check,
        "right_check": right_check,
        "right_retained": right_retained,
        "status": "DEVELOPMENT_NATIVE_BILATERAL_CONTROLLED_UNVALIDATED"
        if controlled and right_retained
        else "REJECTED_LEFT_CONTROLLED_GATE"
        if right_retained
        else "REJECTED_RIGHT_RETENTION_GATE",
        "fresh_exam_required": True,
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during joint left training")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--right-report", type=Path, required=True)
    parser.add_argument("--native-report", type=Path, required=True)
    parser.add_argument("--navigation-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=159928)
    args = parser.parse_args()
    report = train(
        args.asset_root,
        args.policy,
        args.parent_report,
        args.right_report,
        args.native_report,
        args.navigation_report,
        args.output,
        seed=args.seed,
    )
    print(json.dumps({"status": report["status"], "report_hash": report["report_hash"]}))


if __name__ == "__main__":
    main()
