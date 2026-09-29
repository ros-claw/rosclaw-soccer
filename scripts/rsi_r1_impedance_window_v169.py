"""SIM_ONLY fine-window search between clean and body-contaminated contact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from rsi_r1_left_positive_mining_v163 import rank
from rsi_r1_side_navigation_fresh_v160 import FRESH_COURSES, clean
from rsi_r1_timed_stiffness_v168 import run

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

SCHEMA = "rosclaw_soccer.rsi.r1_impedance_window_v169.result.v1"
SETTINGS = tuple(
    (int(cap), float(scale)) for cap in (24, 28, 32, 36) for scale in (0.85, 0.7, 0.55, 0.4)
)


def train(
    asset_root: Path,
    policy: Path,
    parent_report: Path,
    right_report: Path,
    mining_report: Path,
    timed_report: Path,
    output: Path,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY fine-window evidence required")
    parent, right_parent, mining, timed = (
        json.loads(path.read_text())
        for path in (parent_report, right_report, mining_report, timed_report)
    )
    if any(
        item["report_hash"] != hash_json({k: v for k, v in item.items() if k != "report_hash"})
        for item in (parent, right_parent, mining, timed)
    ) or (
        timed["status"] != "REJECTED_TIMED_IMPEDANCE_CONTROLLED_GATE"
        or not timed["zero_physics_equal"]
        or parent["policy_hash"] != hash_bytes(policy.read_bytes())
    ):
        raise ValueError("sealed zero-qualified timed stiffness lineage required")
    course = FRESH_COURSES[mining["course_index"]]
    coordination = tuple(parent["selected"]["weights"])
    left = tuple(mining["best"]["left_weights"])
    right = tuple(
        float(v) for v in np.asarray(right_parent["best_right"]["weights"], dtype=np.float64) * 0.7
    )
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_impedance_window_v169.py",
            "scripts/rsi_r1_timed_stiffness_v168.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    output.mkdir(parents=True)
    anchor, _ = run(asset_root, policy, course, coordination, left, right, 0.4, 20)
    old_anchor = next(
        row
        for row in timed["rows"]
        if row["max_active_substeps"] == 20 and row["stiffness_scale"] == 0.4
    )
    keys = (
        "active_substeps",
        "active_frames",
        "safe",
        "fault_agents",
        "first_foot_frame",
        "own_nonfoot_frames",
        "controlled_reception",
        "tail_maximum_foot_distance_m",
        "tail_maximum_ball_speed_mps",
    )
    anchor_equal = all(anchor[key] == old_anchor[key] for key in keys)
    rows = []
    if anchor_equal:
        for cap, scale in SETTINGS:
            row, _ = run(asset_root, policy, course, coordination, left, right, scale, cap)
            if row["active_substeps"] > cap:
                raise ValueError("contact-impedance substep cap exceeded")
            rows.append(row)
            (output / "progress.json").write_text(
                json.dumps(rows, indent=2, allow_nan=False) + "\n"
            )
            print(
                json.dumps(
                    {
                        "cap": cap,
                        "scale": scale,
                        "active_substeps": row["active_substeps"],
                        "clean": clean(row),
                        "controlled": row["controlled_reception"],
                        "nonfoot": row["own_nonfoot_frames"],
                        "distance": row["tail_maximum_foot_distance_m"],
                        "speed": row["tail_maximum_ball_speed_mps"],
                    }
                ),
                flush=True,
            )
    best = max([anchor, *rows], key=rank)
    check, _ = run(
        asset_root,
        policy,
        course,
        coordination,
        left,
        right,
        best["stiffness_scale"],
        best["max_active_substeps"],
    )
    positive = clean(check) and check["controlled_reception"]
    report = {
        "schema": SCHEMA,
        "timed_report_hash": timed["report_hash"],
        "source_hashes": sources,
        "partition": "CONSUMED_NATIVE_EIGHT_G1_FINE_IMPEDANCE_WINDOW",
        "course": vars(course),
        "anchor_equal": anchor_equal,
        "anchor": anchor,
        "rows": rows,
        "best": best,
        "check": check,
        "status": "DEVELOPMENT_IMPEDANCE_WINDOW_FEASIBLE_UNVALIDATED"
        if positive and anchor_equal
        else "REJECTED_IMPEDANCE_WINDOW_CONTROLLED_GATE"
        if anchor_equal
        else "REJECTED_IMPEDANCE_ANCHOR_GATE",
        "fresh_exam_required": True,
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during fine stiffness search")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--right-report", type=Path, required=True)
    parser.add_argument("--mining-report", type=Path, required=True)
    parser.add_argument("--timed-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = train(
        args.asset_root,
        args.policy,
        args.parent_report,
        args.right_report,
        args.mining_report,
        args.timed_report,
        args.output,
    )
    print(json.dumps({"status": report["status"], "report_hash": report["report_hash"]}))


if __name__ == "__main__":
    main()
