"""SIM_ONLY CEM of post-foot response against the worst 500 Hz shin gap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from rsi_r1_postfoot_clearance_cem_v178 import run as run_full
from rsi_r1_postfoot_substep_audit_v180 import run as run_substep
from rsi_r1_side_navigation_fresh_v160 import clean

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.continuous_match_residual_ppo import collection_fixture
from rosclaw_soccer.training.receiving_classroom import r1_contact_tap_receiving_configuration
from rosclaw_soccer.world.multi_player import build_g1_multi_player_stadium_model

SCHEMA = "rosclaw_soccer.rsi.r1_substep_clearance_cem_v181.result.v1"
POPULATION = 12
GENERATIONS = 2


def substep_rank(summary: dict[str, Any]) -> tuple[float, ...]:
    window = [row for row in summary["timeline"] if 32 <= row["frame"] <= 34]
    collision_count = sum(row["nonfoot_force_n"] > 1e-6 for row in window)
    minimum = min(float(row["minimum_substep_clearance_m"]) for row in window)
    return (-float(collision_count), minimum)


def train(
    asset_root: Path,
    policy: Path,
    parent_report: Path,
    right_report: Path,
    refine_report: Path,
    lateral_report: Path,
    clearance_report: Path,
    audit_report: Path,
    output: Path,
    *,
    seed: int,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root) or not 0 <= seed < 2**31:
        raise ValueError("new external SIM_ONLY substep-CEM evidence required")
    parent, right_parent, refine, lateral, clearance, audit = (
        json.loads(path.read_text())
        for path in (
            parent_report,
            right_report,
            refine_report,
            lateral_report,
            clearance_report,
            audit_report,
        )
    )
    if any(
        item["report_hash"] != hash_json({k: v for k, v in item.items() if k != "report_hash"})
        for item in (parent, right_parent, refine, lateral, clearance, audit)
    ) or (
        audit["status"] != "DIAGNOSTIC_500HZ_ONLY_NO_PROMOTION"
        or not audit["anchor_equal"]
        or parent["policy_hash"] != hash_bytes(policy.read_bytes())
    ):
        raise ValueError("sealed non-intervening 500 Hz audit required")
    coordination = tuple(parent["selected"]["weights"])
    left = tuple(refine["best"]["left_weights"])
    right = tuple(
        float(v) for v in np.asarray(right_parent["best_right"]["weights"], dtype=np.float64) * 0.7
    )
    old_slope = tuple(lateral["best"]["slope"])
    far_slope = tuple(clearance["best"]["far_slope"])
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_substep_clearance_cem_v181.py",
            "scripts/rsi_r1_postfoot_substep_audit_v180.py",
            "scripts/rsi_r1_postfoot_clearance_cem_v178.py",
            "src/rosclaw_soccer/rsi/receiving_postfoot_clearance_expert.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    output.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    mean = np.zeros(3)
    std = np.full(3, 0.045)
    generations: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for generation in range(GENERATIONS):
        candidates = np.clip(rng.normal(mean, std, size=(POPULATION, 3)), -0.12, 0.12)
        if generation == 0:
            candidates[0] = 0.0
            candidates[1] = np.asarray((-0.05, 0.05, 0.0))
            candidates[2] = np.asarray((0.05, -0.05, 0.0))
        else:
            candidates[0] = mean
        rows: list[dict[str, Any]] = []
        for index, candidate in enumerate(candidates):
            postfoot = tuple(float(value) for value in candidate)
            summary = run_substep(
                asset_root, policy, coordination, left, right, old_slope, far_slope, postfoot
            )
            row = {
                "generation": generation + 1,
                "candidate": index,
                "postfoot": postfoot,
                "summary": summary,
            }
            rows.append(row)
            if best is None or substep_rank(summary) > substep_rank(best["summary"]):
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
                        "postfoot": postfoot,
                        "rank": substep_rank(summary),
                    }
                ),
                flush=True,
            )
        rows.sort(key=lambda row: substep_rank(row["summary"]), reverse=True)
        elite = np.stack([np.asarray(row["postfoot"]) for row in rows[:4]])
        mean = elite.mean(axis=0)
        std = np.maximum(0.015, elite.std(axis=0))
        generations.append({"generation": generation + 1, "rows": rows})
    assert best is not None
    selected = tuple(best["postfoot"])
    paired = run_substep(
        asset_root, policy, coordination, left, right, old_slope, far_slope, selected
    )
    fixture = collection_fixture(asset_root, keeper_preview=True)
    world, _ = r1_contact_tap_receiving_configuration()
    model = build_g1_multi_player_stadium_model(
        asset_root,
        players=fixture.players,
        spec=fixture.goal,
        left_goal_plane_x_m=world.left_goal_plane_x_m if world.bilateral_goals else None,
    )
    full = run_full(
        asset_root,
        policy,
        coordination,
        left,
        right,
        old_slope,
        far_slope,
        selected,
        model,
        mujoco.MjData(model),
    )
    same_physics = paired["result_hash_with_diagnostic_trace"] == full["result_hash"]
    positive = same_physics and clean(full) and full["controlled_reception"]
    report = {
        "schema": SCHEMA,
        "audit_report_hash": audit["report_hash"],
        "source_hashes": sources,
        "partition": "CONSUMED_NATIVE_EIGHT_G1_500HZ_CLEARANCE_CEM",
        "seed": seed,
        "generations": generations,
        "best": best,
        "paired": paired,
        "full": full,
        "same_physics": same_physics,
        "status": "DEVELOPMENT_SUBSTEP_CLEAR_CONTROLLED_UNVALIDATED"
        if positive
        else "REJECTED_SUBSTEP_CLEARANCE_GATE",
        "fresh_exam_required": True,
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during substep CEM")
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
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=181930)
    args = parser.parse_args()
    report = train(
        args.asset_root,
        args.policy,
        args.parent_report,
        args.right_report,
        args.refine_report,
        args.lateral_report,
        args.clearance_report,
        args.audit_report,
        args.output,
        seed=args.seed,
    )
    print(json.dumps({"status": report["status"], "report_hash": report["report_hash"]}))


if __name__ == "__main__":
    main()
