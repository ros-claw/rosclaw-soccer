"""Prove idle unique motor preserves the frozen 8-G1 first-touch parent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.receiving_taskspace_motor import ReceivingTaskspaceMotor
from rosclaw_soccer.rsi.team_receive_contact_evidence import (
    ReceiveContactEvidence,
    ReceiveContactMailbox,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.receiving_experiment import simulate_r0_receiving_course
from rosclaw_soccer.training.receiving_rollout import receiving_window
from rosclaw_soccer.training.role_receiving_courses import ReceivingCourse

COURSES = (
    ReceivingCourse("red.finisher", 92801, 1.25, 0.08),
    ReceivingCourse("red.finisher", 92803, 1.5, -0.08),
)
PHYSICAL_KEYS = (
    "ball_pose",
    "ball_velocity",
    "ball_contact_agent_code",
    "ball_contact_foot_code",
    "ball_nonfoot_contact_agent_code",
    "red_finisher_joint_position",
    "red_finisher_joint_velocity",
)


def audit(asset_root: Path, policy: Path, rejected: Path, output: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY handoff evidence required")
    previous = json.loads(rejected.read_text())
    if (
        previous["schema"] != "rosclaw_soccer.rsi.r1_motor_ownership_v126.result.v1"
        or previous["status"] != "REJECTED_ZERO_MOTOR_OWNERSHIP"
        or previous["report_hash"]
        != hash_json({key: value for key, value in previous.items() if key != "report_hash"})
        or previous["policy_hash"] != hash_bytes(policy.read_bytes())
    ):
        raise ValueError("frozen rejected motor-ownership report required")
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_idle_motor_handoff_v127.py",
            "src/rosclaw_soccer/rsi/receiving_taskspace_motor.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
            "src/rosclaw_soccer/skills/team/motor_option.py",
        )
    }
    rows = []
    for course in COURSES:
        parent, parent_trace = simulate_r0_receiving_course(
            asset_root=asset_root,
            reference_policy_path=policy,
            course=course,
            scenario_id=f"s199.rsi.r1.idle-handoff.{course.seed}",
            configuration_profile="R1_CONTACT_TAP",
        )
        mailbox = ReceiveContactMailbox(course.agent_id)
        motor = ReceivingTaskspaceMotor(
            course.agent_id, mailbox, 0.0, 0.0, 0.0, idle_before_first_touch=True
        )
        child, child_trace = simulate_r0_receiving_course(
            asset_root=asset_root,
            reference_policy_path=policy,
            course=course,
            scenario_id=f"s199.rsi.r1.idle-handoff.{course.seed}",
            configuration_profile="R1_CONTACT_TAP",
            research_motor_option=motor,
            physics_evidence_consumers={course.agent_id: ReceiveContactEvidence(mailbox)},
        )
        parent_info, child_info = parent.to_dict(), child.to_dict()
        ids = tuple(sorted(row["agent_id"] for row in child_info["qualities"]))
        _, reception = receiving_window(
            child_trace, agent_ids=ids, agent_id=course.agent_id, start=20, frames=100
        )
        comparisons = {
            key: bool(np.array_equal(parent_trace[key], child_trace[key])) for key in PHYSICAL_KEYS
        }
        rows.append(
            {
                "course": vars(course),
                "parent_result_hash": hash_json(parent_info),
                "child_result_hash": hash_json(child_info),
                "parent_safe": parent_info["safe"],
                "child_safe": child_info["safe"],
                "motor_fault_agents": child_info["motor_fault_agents"],
                "physics_evidence_fault_agents": child_info["physics_evidence_fault_agents"],
                "control_frames": len(child_trace["time"]),
                "physical_array_equal": comparisons,
                "zero_motor_nonzero_target_frames": motor.nonzero_target_frames,
                "first_own_foot_time_sec": mailbox.snapshot.first_own_foot_time_sec,
                "prefoot_nonfoot_count": mailbox.snapshot.prefoot_nonfoot_count,
                "reception": reception,
            }
        )
    qualified = all(
        row["parent_safe"]
        and row["child_safe"]
        and not row["motor_fault_agents"]
        and not row["physics_evidence_fault_agents"]
        and row["zero_motor_nonzero_target_frames"] == 0
        and row["first_own_foot_time_sec"] is not None
        and row["prefoot_nonfoot_count"] == 0
        and all(row["physical_array_equal"].values())
        for row in rows
    )
    report = {
        "schema": "rosclaw_soccer.rsi.r1_idle_motor_handoff_v127.result.v1",
        "partition": "CONSUMED_EIGHT_G1_MOTOR_AUTHORITY",
        "source_hashes": sources,
        "v126_rejection_hash": previous["report_hash"],
        "policy_hash": hash_bytes(policy.read_bytes()),
        "rollout_count": 2 * len(COURSES),
        "courses": rows,
        "status": "IDLE_MOTOR_NONINTERFERENCE_QUALIFIED"
        if qualified
        else "REJECTED_IDLE_MOTOR_INTERFERENCE",
        "training_authorized": False,
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    output.mkdir(parents=True)
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during idle motor handoff audit")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--rejected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.asset_root, args.policy, args.rejected, args.output)
    print(json.dumps({"status": report["status"], "report_hash": report["report_hash"]}))


if __name__ == "__main__":
    main()
