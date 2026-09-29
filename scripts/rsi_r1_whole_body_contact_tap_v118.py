"""Prove 500Hz contact + 50Hz A1 whole-body tape is noninvasive in eight-G1 world."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from rsi_receiving_student_shared_world_exam import COURSE

from rosclaw_soccer.rsi.receiving_whole_body_contact_tap import ReceivingWholeBodyContactTap
from rosclaw_soccer.rsi.team_receive_contact_evidence import (
    ReceiveContactEvidence,
    ReceiveContactMailbox,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.receiving_experiment import simulate_r0_receiving_course
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule


def audit(asset_root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text())
    capture_path = Path(str(protocol["source_capture"]))
    capture = json.loads(capture_path.read_text())
    if (
        protocol["schema"] != "rosclaw_soccer.rsi.r1_whole_body_contact_tap_v118.protocol.v1"
        or protocol["partition"] != "CONSUMED_EIGHT_G1_CONTACT_BODY_OBSERVATION"
        or protocol["oracle_substrate"] != "A1_body29"
        or protocol["configuration_profile"] != "R1_CONTACT_TAP"
        or protocol["world_config_hash"]
        != "sha256:7d213ea5bb1f94720c962f3a2b673ab7b71be636f426372846bd20efe396a1f9"
        or protocol["teacher_config_hash"]
        != "sha256:b4645b702103e59dc11be405bdbde9fb1e5fa8a5d4f88a7b10648694855c2363"
        or protocol["oracle_start_frame"] != 20
        or protocol["rollout_count"] != 2
        or capture["report_hash"] != protocol["source_capture_hash"]
        or capture["report_hash"]
        != hash_json({key: value for key, value in capture.items() if key != "report_hash"})
        or not (capture_path.parent / "zero-near-ball-parent.npz").is_file()
        or output.exists()
        or protocol["course"]
        != {
            "agent_id": "red.finisher",
            "seed": 92801,
            "ball_speed_mps": 1.25,
            "lateral_offset_m": 0.08,
        }
    ):
        raise ValueError("frozen eight-G1 contact/body tap course required")
    root = Path(__file__).resolve().parents[1]
    if output.resolve().is_relative_to(root):
        raise ValueError("external immutable tape output required")
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_whole_body_contact_tap_v118.py",
            "src/rosclaw_soccer/rsi/receiving_whole_body_contact_tap.py",
            "src/rosclaw_soccer/rsi/team_receive_contact_evidence.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
            "src/rosclaw_soccer/training/receiving_classroom.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
            "src/rosclaw_soccer/training/receiving_feedback.py",
        )
    }
    policy_path = capture_path.parent / "zero-near-ball-parent.npz"
    parent_result, parent_trace = simulate_r0_receiving_course(
        asset_root=asset_root,
        reference_policy_path=policy_path,
        course=COURSE,
        scenario_id="s199.rsi.r1.whole-body-tap-parent.92801",
        configuration_profile="R1_CONTACT_TAP",
    )
    schedule = ReceivingOracleSchedule("red.finisher", "A1_body29", 20, 10, ((0.0,) * 29,))
    mailbox = ReceiveContactMailbox("red.finisher")
    consumer = ReceiveContactEvidence(mailbox)
    tap = ReceivingWholeBodyContactTap("red.finisher", schedule.contract_hash, mailbox)
    child_result, child_trace = simulate_r0_receiving_course(
        asset_root=asset_root,
        reference_policy_path=policy_path,
        course=COURSE,
        scenario_id="s199.rsi.r1.whole-body-tap-parent.92801",
        configuration_profile="R1_CONTACT_TAP",
        oracle=schedule,
        feedback_provider=tap,
        physics_evidence_consumers={"red.finisher": consumer},
    )
    arrays = tap.arrays()
    required = (
        "ball_pose",
        "ball_velocity",
        "ball_contact_agent_code",
        "ball_contact_foot_code",
        "ball_nonfoot_contact_agent_code",
        "red_finisher_joint_position",
        "red_finisher_joint_velocity",
    )
    comparisons = {
        key: bool(np.array_equal(parent_trace[key], child_trace[key])) for key in required
    }
    parent_info = parent_result.to_dict()
    child_info = child_result.to_dict()
    clean = bool(
        len(parent_trace["time"]) == len(child_trace["time"])
        and len(arrays["frame"]) == len(child_trace["time"]) - 20
        and np.array_equal(arrays["frame"], np.arange(20, len(child_trace["time"])))
        and not child_info["physics_evidence_fault_agents"]
        and all(comparisons.values())
        and parent_info["safe"] == child_info["safe"]
    )
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("whole-body tape source drift")
    output.mkdir(parents=True)
    np.savez_compressed(output / "body-contact-tape.npz", **arrays)  # type: ignore[arg-type]
    result = {
        "schema": "rosclaw_soccer.rsi.r1_whole_body_contact_tap_v118.result.v1",
        "protocol_hash": hash_bytes(protocol_path.read_bytes()),
        "source_hashes": sources,
        "capture_report_hash": capture["report_hash"],
        "parent_result_hash": hash_json(parent_info),
        "child_result_hash": hash_json(child_info),
        "tape_hash": hash_bytes((output / "body-contact-tape.npz").read_bytes()),
        "tape_contract_hash": tap.contract_hash,
        "evidence_contract_hash": consumer.contract_hash,
        "actual_rollouts": 2,
        "tape_frames": len(arrays["frame"]),
        "actual_control_frames": len(child_trace["time"]),
        "actual_decision_frames": child_info["decision_frame_count"],
        "first_own_foot_time_sec": mailbox.snapshot.first_own_foot_time_sec,
        "first_own_foot": mailbox.snapshot.first_own_foot,
        "prefoot_nonfoot_count": mailbox.snapshot.prefoot_nonfoot_count,
        "physical_array_equal": comparisons,
        "parent_safe": parent_info["safe"],
        "child_safe": child_info["safe"],
        "physics_evidence_fault_agents": child_info["physics_evidence_fault_agents"],
        "status": "A1_CONTACT_BODY_OBSERVATION_QUALIFIED"
        if clean
        else "REJECTED_OBSERVATION_INTERFERENCE",
        "training_authorized": False,
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    result["report_hash"] = hash_json(result)
    (output / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.asset_root, args.protocol, args.output)
    print(json.dumps({"status": report["status"], "report_hash": report["report_hash"]}))


if __name__ == "__main__":
    main()
