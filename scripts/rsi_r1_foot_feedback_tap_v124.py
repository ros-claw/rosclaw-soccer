"""Prove read-only same-frame foot kinematics do not alter 8-G1 physics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.receiving_whole_body_foot_tap import ReceivingWholeBodyFootTap
from rosclaw_soccer.rsi.team_receive_contact_evidence import (
    ReceiveContactEvidence,
    ReceiveContactMailbox,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.receiving_experiment import simulate_r0_receiving_course
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule
from rosclaw_soccer.training.role_receiving_courses import ReceivingCourse

COURSE = ReceivingCourse("red.finisher", 92801, 1.25, 0.08)
SCHEDULE = ReceivingOracleSchedule("red.finisher", "A1_body29", 20, 10, ((0.0,) * 29,))


def audit(asset_root: Path, policy: Path, output: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY foot evidence directory required")
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_foot_feedback_tap_v124.py",
            "src/rosclaw_soccer/rsi/receiving_whole_body_foot_tap.py",
            "src/rosclaw_soccer/rsi/receiving_whole_body_contact_tap.py",
            "src/rosclaw_soccer/training/receiving_feedback.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
            "src/rosclaw_soccer/skills/team/foot_kinematics.py",
        )
    }
    parent, parent_trace = simulate_r0_receiving_course(
        asset_root=asset_root,
        reference_policy_path=policy,
        course=COURSE,
        scenario_id="s199.rsi.r1.foot-feedback-tap.92801",
        configuration_profile="R1_CONTACT_TAP",
    )
    mailbox = ReceiveContactMailbox(COURSE.agent_id)
    tap = ReceivingWholeBodyFootTap(COURSE.agent_id, SCHEDULE.contract_hash, mailbox)
    child, child_trace = simulate_r0_receiving_course(
        asset_root=asset_root,
        reference_policy_path=policy,
        course=COURSE,
        scenario_id="s199.rsi.r1.foot-feedback-tap.92801",
        configuration_profile="R1_CONTACT_TAP",
        oracle=SCHEDULE,
        feedback_provider=tap,
        physics_evidence_consumers={COURSE.agent_id: ReceiveContactEvidence(mailbox)},
    )
    arrays = tap.arrays()
    keys = (
        "ball_pose",
        "ball_velocity",
        "ball_contact_agent_code",
        "ball_contact_foot_code",
        "ball_nonfoot_contact_agent_code",
        "red_finisher_joint_position",
        "red_finisher_joint_velocity",
    )
    comparisons = {key: bool(np.array_equal(parent_trace[key], child_trace[key])) for key in keys}
    parent_info, child_info = parent.to_dict(), child.to_dict()
    matched = bool(
        len(parent_trace["time"]) == len(child_trace["time"])
        and len(arrays["frame"]) == len(child_trace["time"]) - 20
        and np.array_equal(arrays["frame"], np.arange(20, len(child_trace["time"])))
        and arrays["foot_position_world_m"].shape == (len(arrays["frame"]), 2, 3)
        and arrays["foot_velocity_world_mps"].shape == (len(arrays["frame"]), 2, 3)
        and arrays["foot_jacobian_world"].shape == (len(arrays["frame"]), 2, 3, 6)
        and np.any(np.abs(arrays["foot_jacobian_world"]) > 1e-6)
        and not child_info["physics_evidence_fault_agents"]
        and parent_info["safe"] == child_info["safe"]
        and all(comparisons.values())
    )
    output.mkdir(parents=True)
    np.savez_compressed(output / "foot-body-contact-tape.npz", **arrays)  # type: ignore[arg-type]
    report = {
        "schema": "rosclaw_soccer.rsi.r1_foot_feedback_tap_v124.result.v1",
        "partition": "CONSUMED_EIGHT_G1_OBSERVATION",
        "source_hashes": sources,
        "policy_hash": hash_bytes(policy.read_bytes()),
        "schedule_hash": SCHEDULE.contract_hash,
        "tap_contract_hash": tap.contract_hash,
        "parent_result_hash": hash_json(parent_info),
        "child_result_hash": hash_json(child_info),
        "tape_hash": hash_bytes((output / "foot-body-contact-tape.npz").read_bytes()),
        "control_frames": len(child_trace["time"]),
        "tape_frames": len(arrays["frame"]),
        "first_own_foot_time_sec": mailbox.snapshot.first_own_foot_time_sec,
        "physical_array_equal": comparisons,
        "parent_safe": parent_info["safe"],
        "child_safe": child_info["safe"],
        "physics_evidence_fault_agents": child_info["physics_evidence_fault_agents"],
        "status": "FOOT_BODY_CONTACT_OBSERVATION_QUALIFIED"
        if matched
        else "REJECTED_OBSERVATION_INTERFERENCE",
        "training_authorized": False,
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during foot observation audit")
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
