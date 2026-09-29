"""Paired bilateral physical search of pre-impact hip roll/yaw geometry."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
from rsi_r1_coordinated_receiving_v136 import rank, summarize
from rsi_r1_taskspace_feedback_v125 import COURSES

from rosclaw_soccer.rsi.receiving_impact_orientation_feedback import (
    ReceivingImpactOrientationFeedback,
)
from rosclaw_soccer.rsi.team_receive_contact_evidence import (
    ReceiveContactEvidence,
    ReceiveContactMailbox,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.receiving_experiment import simulate_r0_receiving_course
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule
from rosclaw_soccer.training.receiving_rollout import explain_receiving_window, receiving_window
from rosclaw_soccer.training.role_receiving_courses import ReceivingCourse

SCHEMA = "rosclaw_soccer.rsi.r1_impact_orientation_v137.result.v1"
ORIENTATIONS: tuple[tuple[float, float], ...] = tuple(
    (float(a), float(b)) for a, b in itertools.product((-0.75, 0.0, 0.75), repeat=2)
)


def evaluate(
    asset_root: Path,
    policy: Path,
    course: ReceivingCourse,
    weights: tuple[float, ...],
    orientation: tuple[float, float],
) -> dict[str, Any]:
    schedule = ReceivingOracleSchedule(course.agent_id, "A1_body29", 15, 10, ((0.0,) * 29,))
    mailbox = ReceiveContactMailbox(course.agent_id)
    actor = ReceivingImpactOrientationFeedback(
        course.agent_id,
        schedule.contract_hash,
        mailbox,
        0.35,
        0.0,
        0.0,
        target_depth_m=0.25,
        target_lateral_m=0.12,
        coordination=weights,
        impact_roll=orientation[0],
        impact_yaw=orientation[1],
    )
    result, trace = simulate_r0_receiving_course(
        asset_root=asset_root,
        reference_policy_path=policy,
        course=course,
        scenario_id=f"s199.rsi.r1.impact-orientation.{course.seed}",
        configuration_profile="R1_CONTACT_TAP",
        oracle=schedule,
        feedback_provider=actor,
        physics_evidence_consumers={course.agent_id: ReceiveContactEvidence(mailbox)},
    )
    info = result.to_dict()
    ids = tuple(sorted(row["agent_id"] for row in info["qualities"]))
    _, outcome = receiving_window(
        trace, agent_ids=ids, agent_id=course.agent_id, start=20, frames=100
    )
    detail = explain_receiving_window(
        trace, agent_ids=ids, agent_id=course.agent_id, start=20, frames=100
    )
    code = ids.index(course.agent_id) + 1
    foot = np.asarray(trace["ball_contact_agent_code"])
    effector = np.asarray(trace["ball_contact_effector_code"])
    force = np.asarray(trace["ball_contact_force_n"])
    nonfoot = np.asarray(trace["ball_nonfoot_contact_agent_code"])
    nonfoot_force = np.asarray(trace["ball_nonfoot_contact_force_n"])
    first = next(
        (i for i in range(20, 120) if foot[i] == code and effector[i] in (1, 2) and force[i] > 0),
        None,
    )
    shin = (
        []
        if first is None
        else [i for i in range(first, 120) if nonfoot[i] == code and nonfoot_force[i] > 0]
    )
    return {
        "course": vars(course),
        "safe": info["safe"],
        "physics_evidence_fault_agents": info["physics_evidence_fault_agents"],
        "first_own_foot_time_sec": mailbox.snapshot.first_own_foot_time_sec,
        "prefoot_nonfoot_count": mailbox.snapshot.prefoot_nonfoot_count,
        "own_shin_frames": shin,
        "outcome": outcome,
        "tail_maximum_foot_distance_m": detail["tail_maximum_foot_distance_m"],
        "tail_maximum_ball_speed_mps": detail["tail_maximum_ball_speed_mps"],
        "peak_actual_residual_rad": float(
            np.max(np.abs(np.asarray(trace["receiving_oracle_delta_rad"])))
        ),
        "result_hash": hash_json(info),
    }


def train(asset_root: Path, policy: Path, parent_report: Path, output: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY impact evidence required")
    parent = json.loads(parent_report.read_text())
    if (
        parent["schema"] != "rosclaw_soccer.rsi.r1_coordinated_receiving_v136.result.v1"
        or parent["report_hash"]
        != hash_json({k: v for k, v in parent.items() if k != "report_hash"})
        or parent["status"] != "REJECTED_NO_CONTROLLED_GAIN"
        or parent["policy_hash"] != hash_bytes(policy.read_bytes())
        or parent["selected"] is None
    ):
        raise ValueError("integrity-checked rejected development parent required")
    weights = tuple(parent["selected"]["weights"])
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_impact_orientation_v137.py",
            "scripts/rsi_r1_coordinated_receiving_v136.py",
            "src/rosclaw_soccer/rsi/receiving_impact_orientation_feedback.py",
            "src/rosclaw_soccer/rsi/receiving_coordinated_feedback.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    output.mkdir(parents=True)
    protocol: dict[str, Any] = {
        "schema": SCHEMA,
        "partition": "CONSUMED_BILATERAL_EIGHT_G1_DEVELOPMENT",
        "source_hashes": sources,
        "parent_report_hash": parent["report_hash"],
        "policy_hash": parent["policy_hash"],
        "weights": weights,
        "courses": [vars(c) for c in COURSES],
        "orientations": ORIENTATIONS,
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    protocol["protocol_hash"] = hash_json(protocol)
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    rows: list[dict[str, Any]] = []
    for index, orientation in enumerate(ORIENTATIONS):
        trials = [evaluate(asset_root, policy, course, weights, orientation) for course in COURSES]
        row = summarize(index, 0, weights, trials)
        row["orientation"] = orientation
        rows.append(row)
        (output / "progress.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
        print(json.dumps({k: v for k, v in row.items() if k != "trials"}), flush=True)
    zero = next(row for row in rows if row["orientation"] == (0.0, 0.0))
    eligible = [row for row in rows if row["safe"] and row["all_clean_first_foot"]]
    selected = max(eligible, key=rank) if eligible else None
    report = {
        "schema": SCHEMA,
        "protocol_hash": protocol["protocol_hash"],
        "source_hashes": sources,
        "parent_report_hash": parent["report_hash"],
        "policy_hash": protocol["policy_hash"],
        "rollout_count": len(rows) * len(COURSES),
        "parent": zero,
        "selected": selected,
        "all_candidates": rows,
        "status": "DEVELOPMENT_CONTROLLED_GAIN_UNVALIDATED"
        if selected is not None and selected["controlled_count"] > zero["controlled_count"]
        else "REJECTED_NO_CONTROLLED_GAIN",
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "selection.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during impact-orientation development")
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
