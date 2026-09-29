"""SIM_ONLY bounded knee-direction follow-up on measured shin clearance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from rsi_r1_postfoot_clearance_cem_v178 import rank, run
from rsi_r1_side_navigation_fresh_v160 import clean

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.continuous_match_residual_ppo import collection_fixture
from rosclaw_soccer.training.receiving_classroom import r1_contact_tap_receiving_configuration
from rosclaw_soccer.world.multi_player import build_g1_multi_player_stadium_model

SCHEMA = "rosclaw_soccer.rsi.r1_postfoot_knee_sweep_v179.result.v1"
SETTINGS = tuple(
    (float(hip), float(knee), 0.0) for knee in (0.08, 0.10, 0.12) for hip in (-0.05, 0.0, 0.05)
)


def train(
    asset_root: Path,
    policy: Path,
    parent_report: Path,
    right_report: Path,
    refine_report: Path,
    lateral_report: Path,
    clearance_report: Path,
    postfoot_report: Path,
    output: Path,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY knee sweep evidence required")
    parent, right_parent, refine, lateral, clearance, postfoot = (
        json.loads(path.read_text())
        for path in (
            parent_report,
            right_report,
            refine_report,
            lateral_report,
            clearance_report,
            postfoot_report,
        )
    )
    if any(
        item["report_hash"] != hash_json({k: v for k, v in item.items() if k != "report_hash"})
        for item in (parent, right_parent, refine, lateral, clearance, postfoot)
    ) or (
        postfoot["status"] != "REJECTED_FAR_POSTFOOT_CLEARANCE_GATE"
        or parent["policy_hash"] != hash_bytes(policy.read_bytes())
    ):
        raise ValueError("sealed event-conditioned failure required")
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
            "scripts/rsi_r1_postfoot_knee_sweep_v179.py",
            "scripts/rsi_r1_postfoot_clearance_cem_v178.py",
            "src/rosclaw_soccer/rsi/receiving_postfoot_clearance_expert.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    fixture = collection_fixture(asset_root, keeper_preview=True)
    world, _ = r1_contact_tap_receiving_configuration()
    model = build_g1_multi_player_stadium_model(
        asset_root,
        players=fixture.players,
        spec=fixture.goal,
        left_goal_plane_x_m=world.left_goal_plane_x_m if world.bilateral_goals else None,
    )
    data = mujoco.MjData(model)
    output.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    for post in SETTINGS:
        summary = run(
            asset_root,
            policy,
            coordination,
            left,
            right,
            old_slope,
            far_slope,
            post,
            model,
            data,
        )
        if summary["active_substeps"] > 32:
            raise ValueError("compliance budget exceeded")
        row = {"postfoot": post, "summary": summary}
        rows.append(row)
        (output / "progress.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "postfoot": post,
                    "clean": clean(summary),
                    "controlled": summary["controlled_reception"],
                    "nonfoot": summary["own_nonfoot_frames"],
                    "shin_clearance_m": summary["minimum_shin_clearance_m"],
                    "distance": summary["tail_maximum_foot_distance_m"],
                    "speed": summary["tail_maximum_ball_speed_mps"],
                }
            ),
            flush=True,
        )
    best = max(rows, key=lambda row: rank(row["summary"]))
    check = run(
        asset_root,
        policy,
        coordination,
        left,
        right,
        old_slope,
        far_slope,
        tuple(best["postfoot"]),
        model,
        data,
    )
    positive = clean(check) and check["controlled_reception"]
    report = {
        "schema": SCHEMA,
        "postfoot_report_hash": postfoot["report_hash"],
        "source_hashes": sources,
        "partition": "CONSUMED_NATIVE_EIGHT_G1_POSTFOOT_KNEE_SWEEP",
        "rows": rows,
        "best": best,
        "check": check,
        "status": "DEVELOPMENT_KNEE_SWEEP_CONTROLLED_UNVALIDATED"
        if positive
        else "REJECTED_KNEE_SWEEP_CONTROLLED_GATE",
        "fresh_exam_required": True,
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during knee sweep")
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
    parser.add_argument("--postfoot-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = train(
        args.asset_root,
        args.policy,
        args.parent_report,
        args.right_report,
        args.refine_report,
        args.lateral_report,
        args.clearance_report,
        args.postfoot_report,
        args.output,
    )
    print(json.dumps({"status": report["status"], "report_hash": report["report_hash"]}))


if __name__ == "__main__":
    main()
