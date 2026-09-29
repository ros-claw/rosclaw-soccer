"""SIM_ONLY local CEM near a clean eight-G1 compliant contact sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from rsi_r1_impedance_follow_v170 import run
from rsi_r1_left_positive_mining_v163 import rank
from rsi_r1_side_navigation_fresh_v160 import FRESH_COURSES, clean
from rsi_r1_taskspace_feedback_v125 import COURSES

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

SCHEMA = "rosclaw_soccer.rsi.r1_compliant_contact_refine_v172.result.v1"
POPULATION = 12
GENERATIONS = 2


def train(
    asset_root: Path,
    policy: Path,
    parent_report: Path,
    right_report: Path,
    cem_report: Path,
    output: Path,
    *,
    seed: int,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root) or not 0 <= seed < 2**31:
        raise ValueError("new external SIM_ONLY compliant refinement evidence required")
    parent, right_parent, cem = (
        json.loads(path.read_text()) for path in (parent_report, right_report, cem_report)
    )
    if any(
        item["report_hash"] != hash_json({k: v for k, v in item.items() if k != "report_hash"})
        for item in (parent, right_parent, cem)
    ) or (
        cem["status"] != "REJECTED_LOCAL_COMPLIANT_CONTROLLED_GATE"
        or cem["best"]["generation"] != 2
        or cem["local_check"]["own_nonfoot_frames"]
        or parent["policy_hash"] != hash_bytes(policy.read_bytes())
    ):
        raise ValueError("sealed clean near-threshold native anchor required")
    course = FRESH_COURSES[0]
    coordination = tuple(parent["selected"]["weights"])
    base = np.asarray(cem["best"]["left_weights"], dtype=np.float64)
    right = tuple(
        float(v) for v in np.asarray(right_parent["best_right"]["weights"], dtype=np.float64) * 0.7
    )
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_compliant_contact_refine_v172.py",
            "scripts/rsi_r1_impedance_follow_v170.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    output.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    mean = base.copy()
    std = np.full(12, 0.025)
    generations: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for generation in range(GENERATIONS):
        candidates = np.clip(rng.normal(mean, std, size=(POPULATION, 12)), -1.0, 1.0)
        candidates[0] = base if generation == 0 else mean
        rows: list[dict[str, Any]] = []
        for index, candidate in enumerate(candidates):
            left = tuple(float(value) for value in candidate)
            summary = run(asset_root, policy, course, coordination, left, right, 0.0, 0.5)
            if summary["active_substeps"] > 32:
                raise ValueError("compliance budget exceeded")
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
        std = np.maximum(0.008, elite.std(axis=0))
        generations.append({"generation": generation + 1, "rows": rows})
    assert best is not None
    selected_left = tuple(best["left_weights"])
    local_check = run(asset_root, policy, course, coordination, selected_left, right, 0.0, 0.5)
    old_left_check = run(
        asset_root, policy, COURSES[0], coordination, selected_left, right, 0.0, 0.5
    )
    far_left_check = run(
        asset_root, policy, FRESH_COURSES[1], coordination, selected_left, right, 0.0, 0.5
    )
    positive = clean(local_check) and local_check["controlled_reception"]
    report = {
        "schema": SCHEMA,
        "cem_report_hash": cem["report_hash"],
        "source_hashes": sources,
        "partition": "CONSUMED_NATIVE_EIGHT_G1_COMPLIANT_REFINEMENT",
        "seed": seed,
        "generations": generations,
        "best": best,
        "local_check": local_check,
        "old_left_check": old_left_check,
        "far_left_check": far_left_check,
        "status": "DEVELOPMENT_LOCAL_COMPLIANT_CONTROLLED_UNVALIDATED"
        if positive
        else "REJECTED_LOCAL_COMPLIANT_REFINEMENT_GATE",
        "fresh_exam_required": True,
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during compliant refinement")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--right-report", type=Path, required=True)
    parser.add_argument("--cem-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=172929)
    args = parser.parse_args()
    report = train(
        args.asset_root,
        args.policy,
        args.parent_report,
        args.right_report,
        args.cem_report,
        args.output,
        seed=args.seed,
    )
    print(json.dumps({"status": report["status"], "report_hash": report["report_hash"]}))


if __name__ == "__main__":
    main()
