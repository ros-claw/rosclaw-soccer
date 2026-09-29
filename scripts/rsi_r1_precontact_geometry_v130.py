"""Search precontact foot geometry using the qualified SIM_ONLY feedback slot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rsi_r1_taskspace_feedback_v125 import COURSES, SCHEDULE, evaluate

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

# pre_gain, post_gain, velocity horizon, hip, ankle, foot depth, foot lateral.
PARAMETERS = (
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.18, 0.03),
    (0.25, 0.0, 0.0, 0.0, 0.0, 0.18, 0.03),
    (0.25, 0.0, 0.0, 0.0, 0.0, 0.12, 0.03),
    (0.25, 0.0, 0.0, 0.0, 0.0, 0.25, 0.03),
    (0.25, 0.0, 0.0, 0.0, 0.0, 0.18, -0.03),
    (0.25, 0.0, 0.0, 0.0, 0.0, 0.18, 0.08),
    (0.25, 0.0, 0.0, 0.0, 0.0, 0.18, 0.15),
    (0.25, 0.0, 0.0, 0.0, 0.0, 0.12, 0.12),
    (0.25, 0.0, 0.0, 0.0, 0.0, 0.25, 0.12),
    (0.5, 0.0, 0.0, 0.0, 0.0, 0.18, 0.08),
)


def train(asset_root: Path, policy: Path, parent_report: Path, output: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY precontact evidence required")
    parent = json.loads(parent_report.read_text())
    if (
        parent["schema"] != "rosclaw_soccer.rsi.r1_taskspace_feedback_v125.result.v1"
        or parent["report_hash"]
        != hash_json({k: v for k, v in parent.items() if k != "report_hash"})
        or parent["policy_hash"] != hash_bytes(policy.read_bytes())
        or parent["schedule_hash"] != SCHEDULE.contract_hash
    ):
        raise ValueError("integrity-checked taskspace parent required")
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_precontact_geometry_v130.py",
            "scripts/rsi_r1_taskspace_feedback_v125.py",
            "src/rosclaw_soccer/rsi/receiving_taskspace_feedback.py",
            "src/rosclaw_soccer/training/receiving_feedback.py",
            "src/rosclaw_soccer/training/receiving_rollout.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    output.mkdir(parents=True)
    rows = []
    for index, parameters in enumerate(PARAMETERS):
        trials = [evaluate(asset_root, policy, course, parameters) for course in COURSES]
        row = {
            "candidate": index,
            "parameters": parameters,
            "safe": all(t["safe"] and not t["physics_evidence_fault_agents"] for t in trials),
            "all_clean_first_foot": all(
                t["first_own_foot_time_sec"] is not None and t["prefoot_nonfoot_count"] == 0
                for t in trials
            ),
            "controlled_count": sum(t["outcome"]["controlled_reception"] for t in trials),
            "own_shin_frame_count": sum(len(t["own_shin_frames"]) for t in trials),
            "tail_distance_sum_m": sum(
                t["explanation"]["tail_maximum_foot_distance_m"] for t in trials
            ),
            "tail_speed_sum_mps": sum(
                t["explanation"]["tail_maximum_ball_speed_mps"] for t in trials
            ),
            "trials": trials,
        }
        rows.append(row)
        (output / "progress.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(json.dumps({k: v for k, v in row.items() if k != "trials"}), flush=True)
    eligible = [row for row in rows if row["safe"] and row["all_clean_first_foot"]]
    if not eligible:
        raise ValueError("no clean first-foot candidate; progress evidence retained")
    selected = max(
        eligible,
        key=lambda row: (
            row["controlled_count"],
            -row["own_shin_frame_count"],
            -row["tail_distance_sum_m"],
            -row["tail_speed_sum_mps"],
        ),
    )
    report = {
        "schema": "rosclaw_soccer.rsi.r1_precontact_geometry_v130.result.v1",
        "partition": "CONSUMED_EIGHT_G1_PRECONTACT_DEVELOPMENT",
        "source_hashes": sources,
        "parent_report_hash": parent["report_hash"],
        "policy_hash": hash_bytes(policy.read_bytes()),
        "schedule_hash": SCHEDULE.contract_hash,
        "candidate_count": len(rows),
        "rollout_count": len(rows) * len(COURSES),
        "parent": rows[0],
        "selected": selected,
        "all_candidates": rows,
        "status": "DEVELOPMENT_CONTROLLED_GAIN_UNVALIDATED"
        if selected["controlled_count"] > rows[0]["controlled_count"]
        else "REJECTED_NO_CONTROLLED_GAIN",
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "selection.json").write_text(json.dumps(report, indent=2) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during precontact development")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = train(args.asset_root, args.policy, args.parent_report, args.output)
    print(json.dumps({"status": report["status"], "report_hash": report["report_hash"]}))


if __name__ == "__main__":
    main()
