"""SIM_ONLY 500 Hz clearance training of the far-course pre-contact policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from rsi_r1_compliant_piecewise_v174 import run as run_retention
from rsi_r1_predictive_shin_sweep_v182 import run
from rsi_r1_side_navigation_fresh_v160 import FRESH_COURSES, clean
from rsi_r1_taskspace_feedback_v125 import COURSES

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

SCHEMA = "rosclaw_soccer.rsi.r1_precontact_substep_cem_v183.result.v1"
POPULATION = 12
GENERATIONS = 2
ZERO_REFLEX = (0.03, 0.06, 0.0)


def rank(row: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(clean(row) and row["controlled_reception"]),
        float(clean(row)),
        -float(len(row["own_nonfoot_frames"])),
        float(row["minimum_shin_clearance_substep_m"]),
        -float(row["tail_maximum_foot_distance_m"]),
        -float(row["tail_maximum_ball_speed_mps"]),
    )


def train(
    asset_root: Path,
    policy: Path,
    parent_report: Path,
    right_report: Path,
    refine_report: Path,
    lateral_report: Path,
    clearance_report: Path,
    predictive_report: Path,
    output: Path,
    *,
    seed: int,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root) or not 0 <= seed < 2**31:
        raise ValueError("new external SIM_ONLY pre-contact substep CEM evidence required")
    parent, right_parent, refine, lateral, clearance, predictive = (
        json.loads(path.read_text())
        for path in (
            parent_report,
            right_report,
            refine_report,
            lateral_report,
            clearance_report,
            predictive_report,
        )
    )
    if any(
        item["report_hash"] != hash_json({k: v for k, v in item.items() if k != "report_hash"})
        for item in (parent, right_parent, refine, lateral, clearance, predictive)
    ) or (
        predictive["status"] != "REJECTED_PREDICTIVE_SHIN_CONTROLLED_GATE"
        or parent["policy_hash"] != hash_bytes(policy.read_bytes())
    ):
        raise ValueError("sealed zero-qualified predictive fallback required")
    coordination = tuple(parent["selected"]["weights"])
    left = tuple(refine["best"]["left_weights"])
    right = tuple(
        float(v) for v in np.asarray(right_parent["best_right"]["weights"], dtype=np.float64) * 0.7
    )
    old_slope = tuple(lateral["best"]["slope"])
    base = np.asarray(clearance["best"]["far_slope"], dtype=np.float64)
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_precontact_substep_cem_v183.py",
            "scripts/rsi_r1_predictive_shin_sweep_v182.py",
            "src/rosclaw_soccer/rsi/receiving_lateral_piecewise_expert.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    output.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    mean = base.copy()
    std = np.full(12, 0.04)
    generations: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for generation in range(GENERATIONS):
        candidates = np.clip(rng.normal(mean, std, size=(POPULATION, 12)), -1.0, 1.0)
        candidates[0] = base if generation == 0 else mean
        rows: list[dict[str, Any]] = []
        for index, candidate in enumerate(candidates):
            far_slope = tuple(float(value) for value in candidate)
            summary = run(
                asset_root,
                policy,
                coordination,
                left,
                right,
                old_slope,
                far_slope,
                ZERO_REFLEX,
            )
            if summary["active_substeps"] > 32:
                raise ValueError("compliance budget exceeded")
            row = {
                "generation": generation + 1,
                "candidate": index,
                "far_slope": far_slope,
                "summary": summary,
            }
            rows.append(row)
            if best is None or rank(summary) > rank(best["summary"]):
                best = row
            (output / "progress.json").write_text(
                json.dumps([*generations, {"generation": generation + 1, "rows": rows}], indent=2)
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
                        "min_shin_m": summary["minimum_shin_clearance_substep_m"],
                        "distance": summary["tail_maximum_foot_distance_m"],
                        "speed": summary["tail_maximum_ball_speed_mps"],
                    }
                ),
                flush=True,
            )
        rows.sort(key=lambda row: rank(row["summary"]), reverse=True)
        elite = np.stack([np.asarray(row["far_slope"]) for row in rows[:4]])
        mean = elite.mean(axis=0)
        std = np.maximum(0.015, elite.std(axis=0))
        generations.append({"generation": generation + 1, "rows": rows})
    assert best is not None
    selected = tuple(best["far_slope"])
    far_check = run(asset_root, policy, coordination, left, right, old_slope, selected, ZERO_REFLEX)
    retained = [
        run_retention(asset_root, policy, course, coordination, left, right, old_slope, selected)
        for course in (FRESH_COURSES[0], COURSES[0])
    ]
    positive = all(clean(row) and row["controlled_reception"] for row in (far_check, *retained))
    report = {
        "schema": SCHEMA,
        "predictive_report_hash": predictive["report_hash"],
        "source_hashes": sources,
        "partition": "CONSUMED_NATIVE_EIGHT_G1_PRECONTACT_500HZ_CEM",
        "seed": seed,
        "generations": generations,
        "best": best,
        "far_check": far_check,
        "retained": retained,
        "status": "DEVELOPMENT_THREE_COURSE_SUBSTEP_CONTROLLED_UNVALIDATED"
        if positive
        else "REJECTED_PRECONTACT_SUBSTEP_CONTROLLED_GATE",
        "fresh_exam_required": True,
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during precontact substep training")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--right-report", type=Path, required=True)
    parser.add_argument("--refine-report", type=Path, required=True)
    parser.add_argument("--lateral-report", type=Path, required=True)
    parser.add_argument("--clearance-report", type=Path, required=True)
    parser.add_argument("--predictive-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=183930)
    args = parser.parse_args()
    report = train(
        args.asset_root,
        args.policy,
        args.parent_report,
        args.right_report,
        args.refine_report,
        args.lateral_report,
        args.clearance_report,
        args.predictive_report,
        args.output,
        seed=args.seed,
    )
    print(json.dumps({"status": report["status"], "report_hash": report["report_hash"]}))


if __name__ == "__main__":
    main()
