"""Qualify a zero-correction unique receiver motor before motor-space learning."""

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
from rosclaw_soccer.training.receiving_rollout import explain_receiving_window, receiving_window
from rosclaw_soccer.training.role_receiving_courses import ReceivingCourse

COURSES = (
    ReceivingCourse("red.finisher", 92801, 1.25, 0.08),
    ReceivingCourse("red.finisher", 92803, 1.5, -0.08),
)


def audit(asset_root: Path, policy: Path, output: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY motor ownership evidence required")
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_motor_ownership_v126.py",
            "src/rosclaw_soccer/rsi/receiving_taskspace_motor.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
            "src/rosclaw_soccer/skills/team/motor_option.py",
        )
    }
    rows = []
    for course in COURSES:
        no_motor, no_trace = simulate_r0_receiving_course(
            asset_root=asset_root,
            reference_policy_path=policy,
            course=course,
            scenario_id=f"s199.rsi.r1.motor-ownership.{course.seed}",
            configuration_profile="R1_CONTACT_TAP",
        )
        mailbox = ReceiveContactMailbox(course.agent_id)
        motor = ReceivingTaskspaceMotor(course.agent_id, mailbox, 0.0, 0.0, 0.0)
        zero_motor, motor_trace = simulate_r0_receiving_course(
            asset_root=asset_root,
            reference_policy_path=policy,
            course=course,
            scenario_id=f"s199.rsi.r1.motor-ownership.{course.seed}",
            configuration_profile="R1_CONTACT_TAP",
            research_motor_option=motor,
            physics_evidence_consumers={course.agent_id: ReceiveContactEvidence(mailbox)},
        )
        no_info, motor_info = no_motor.to_dict(), zero_motor.to_dict()
        ids = tuple(sorted(row["agent_id"] for row in motor_info["qualities"]))
        _, reception = receiving_window(
            motor_trace, agent_ids=ids, agent_id=course.agent_id, start=20, frames=100
        )
        explanation = explain_receiving_window(
            motor_trace, agent_ids=ids, agent_id=course.agent_id, start=20, frames=100
        )
        rows.append(
            {
                "course": vars(course),
                "no_motor_result_hash": hash_json(no_info),
                "zero_motor_result_hash": hash_json(motor_info),
                "no_motor_safe": no_info["safe"],
                "zero_motor_safe": motor_info["safe"],
                "motor_fault_agents": motor_info["motor_fault_agents"],
                "physics_evidence_fault_agents": motor_info["physics_evidence_fault_agents"],
                "zero_motor_contract_hash": motor.contract_hash,
                "zero_motor_nonzero_target_frames": motor.nonzero_target_frames,
                "zero_motor_peak_correction_rad": motor.peak_correction_rad,
                "no_motor_control_frames": len(no_trace["time"]),
                "zero_motor_control_frames": len(motor_trace["time"]),
                "no_motor_vs_zero_motor_ball_equal": bool(
                    np.array_equal(no_trace["ball_pose"], motor_trace["ball_pose"])
                ),
                "first_own_foot_time_sec": mailbox.snapshot.first_own_foot_time_sec,
                "prefoot_nonfoot_count": mailbox.snapshot.prefoot_nonfoot_count,
                "reception": reception,
                "explanation": explanation,
            }
        )
    qualified = all(
        row["zero_motor_safe"]
        and not row["motor_fault_agents"]
        and not row["physics_evidence_fault_agents"]
        and row["zero_motor_nonzero_target_frames"] == 0
        and row["zero_motor_peak_correction_rad"] == 0
        and row["first_own_foot_time_sec"] is not None
        and row["prefoot_nonfoot_count"] == 0
        for row in rows
    )
    report = {
        "schema": "rosclaw_soccer.rsi.r1_motor_ownership_v126.result.v1",
        "partition": "CONSUMED_EIGHT_G1_MOTOR_AUTHORITY",
        "source_hashes": sources,
        "policy_hash": hash_bytes(policy.read_bytes()),
        "rollout_count": 2 * len(COURSES),
        "courses": rows,
        "status": "ZERO_MOTOR_OWNERSHIP_QUALIFIED"
        if qualified
        else "REJECTED_ZERO_MOTOR_OWNERSHIP",
        "training_authorized": False,
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    output.mkdir(parents=True)
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during motor ownership audit")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.asset_root, args.policy, args.output)
    print(json.dumps({"status": report["status"], "report_hash": report["report_hash"]}))


if __name__ == "__main__":
    main()
