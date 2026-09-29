"""SIM_ONLY native eight-G1 side-specific amplitude training after A2 transfer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from rsi_r1_precontact_expert_world_v150 import run
from rsi_r1_taskspace_feedback_v125 import COURSES

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

SCHEMA = "rosclaw_soccer.rsi.r1_native_expert_scale_v152.result.v1"
SCALES = (0.25, 0.4, 0.55, 0.7, 0.85, 1.0)


def rank(row: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(row["safe"] and not row["fault_agents"]),
        float(row["first_foot_frame"] is not None and not row["own_nonfoot_frames"]),
        float(row["controlled_reception"]),
        -float(row["tail_maximum_foot_distance_m"]),
        -float(row["tail_maximum_ball_speed_mps"]),
    )


def train(
    asset_root: Path,
    policy: Path,
    parent_report: Path,
    experts_report: Path,
    right_report: Path,
    authority_report: Path,
    output: Path,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY native expert evidence required")
    parent = json.loads(parent_report.read_text())
    experts = json.loads(experts_report.read_text())
    right = json.loads(right_report.read_text())
    authority = json.loads(authority_report.read_text())
    if any(
        row["report_hash"] != hash_json({k: v for k, v in row.items() if k != "report_hash"})
        for row in (parent, experts, right, authority)
    ) or (
        authority["status"] != "REJECTED_EIGHT_G1_CONTROLLED_GATE"
        or not all(row["zero_physics_equal"] for row in authority["rows"])
        or experts["parent_report_hash"] != right["report_hash"]
        or parent["policy_hash"] != hash_bytes(policy.read_bytes())
    ):
        raise ValueError("qualified zero-action authority and sealed experts required")
    left_weights = np.asarray(experts["best_left"]["weights"], dtype=np.float64)
    right_weights = np.asarray(right["best_right"]["weights"], dtype=np.float64)
    coordination = tuple(parent["selected"]["weights"])
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_native_expert_scale_v152.py",
            "scripts/rsi_r1_precontact_expert_world_v150.py",
            "src/rosclaw_soccer/rsi/receiving_precontact_expert.py",
        )
    }
    output.mkdir(parents=True)
    search_rows: list[dict[str, Any]] = []
    best: list[dict[str, Any]] = []
    for side, course in enumerate(COURSES):
        side_rows = []
        for scale in SCALES:
            candidate = tuple(
                float(v) for v in (left_weights if side == 0 else right_weights) * scale
            )
            left = candidate if side == 0 else (0.0,) * 12
            right_action = candidate if side == 1 else (0.0,) * 12
            summary, _ = run(asset_root, policy, course, coordination, left, right_action)
            row = {"side": side, "scale": scale, "summary": summary}
            side_rows.append(row)
            search_rows.append(row)
            (output / "progress.json").write_text(
                json.dumps(search_rows, indent=2, allow_nan=False) + "\n"
            )
            print(json.dumps({"side": side, "scale": scale, "summary": summary}), flush=True)
        best.append(max(side_rows, key=lambda item: rank(item["summary"])))
    chosen_left = tuple(float(v) for v in left_weights * best[0]["scale"])
    chosen_right = tuple(float(v) for v in right_weights * best[1]["scale"])
    paired = [
        run(asset_root, policy, course, coordination, chosen_left, chosen_right)[0]
        for course in COURSES
    ]
    success = all(
        row["safe"]
        and not row["fault_agents"]
        and row["first_foot_frame"] is not None
        and not row["own_nonfoot_frames"]
        and row["controlled_reception"]
        for row in paired
    )
    report = {
        "schema": SCHEMA,
        "authority_report_hash": authority["report_hash"],
        "source_hashes": sources,
        "partition": "CONSUMED_NATIVE_EIGHT_G1_SIDE_SCALE_TRAINING",
        "rollout_count": len(search_rows) + len(paired),
        "search_rows": search_rows,
        "selected": best,
        "paired": paired,
        "status": "DEVELOPMENT_NATIVE_BILATERAL_CONTROLLED_UNVALIDATED"
        if success
        else "REJECTED_NATIVE_BILATERAL_CONTROLLED_GATE",
        "fresh_exam_required": True,
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during native scale training")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--experts-report", type=Path, required=True)
    parser.add_argument("--right-report", type=Path, required=True)
    parser.add_argument("--authority-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = train(
        args.asset_root,
        args.policy,
        args.parent_report,
        args.experts_report,
        args.right_report,
        args.authority_report,
        args.output,
    )
    print(json.dumps({"status": report["status"], "report_hash": report["report_hash"]}))


if __name__ == "__main__":
    main()
