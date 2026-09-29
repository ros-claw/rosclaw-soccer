"""SIM_ONLY eight-G1 A2 zero-authority and side-conditioned receiving audition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from rsi_r1_taskspace_feedback_v125 import COURSES

from rosclaw_soccer.rsi.receiving_coordinated_feedback import ReceivingCoordinatedFeedback
from rosclaw_soccer.rsi.receiving_precontact_expert import ReceivingPrecontactExpert
from rosclaw_soccer.rsi.team_receive_contact_evidence import (
    ReceiveContactEvidence,
    ReceiveContactMailbox,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.receiving_experiment import simulate_r0_receiving_course
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule
from rosclaw_soccer.training.receiving_rollout import explain_receiving_window, receiving_window
from rosclaw_soccer.training.role_receiving_courses import ReceivingCourse

SCHEMA = "rosclaw_soccer.rsi.r1_precontact_expert_world_v150.result.v1"
PHYSICAL_KEYS = (
    "ball_pose",
    "ball_velocity",
    "ball_contact_agent_code",
    "ball_contact_effector_code",
    "ball_contact_force_n",
    "ball_nonfoot_contact_agent_code",
    "ball_nonfoot_contact_force_n",
    "receiving_feedback_qpos",
    "receiving_feedback_qvel",
)


def run(
    asset_root: Path,
    policy: Path,
    course: ReceivingCourse,
    coordination: tuple[float, ...],
    left: tuple[float, ...] | None,
    right: tuple[float, ...] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    substrate = "A1_body29" if left is None else "A2_body29_precontact"
    schedule = ReceivingOracleSchedule(course.agent_id, substrate, 15, 10, ((0.0,) * 29,))
    mailbox = ReceiveContactMailbox(course.agent_id)
    actor = (
        ReceivingCoordinatedFeedback(
            course.agent_id,
            schedule.contract_hash,
            mailbox,
            0.35,
            0.0,
            0.0,
            target_depth_m=0.25,
            target_lateral_m=0.12,
            coordination=coordination,
        )
        if left is None
        else ReceivingPrecontactExpert(
            course.agent_id,
            schedule.contract_hash,
            mailbox,
            0.35,
            0.0,
            0.0,
            target_depth_m=0.25,
            target_lateral_m=0.12,
            coordination=coordination,
            left_weights=left,
            right_weights=right if right is not None else (0.0,) * 12,
        )
    )
    result, trace = simulate_r0_receiving_course(
        asset_root=asset_root,
        reference_policy_path=policy,
        course=course,
        scenario_id=f"s199.rsi.r1.coordinated-receiving.{course.seed}",
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
    own_foot = np.asarray(trace["ball_contact_agent_code"])
    nonfoot = np.asarray(trace["ball_nonfoot_contact_agent_code"])
    first = next(
        (
            frame
            for frame in range(20, 120)
            if own_foot[frame] == code and trace["ball_contact_force_n"][frame] > 0
        ),
        None,
    )
    nonfoot_frames = [
        frame
        for frame in range(20, 120)
        if nonfoot[frame] == code and trace["ball_nonfoot_contact_force_n"][frame] > 0
    ]
    summary = {
        "course": vars(course),
        "safe": info["safe"],
        "fault_agents": info["physics_evidence_fault_agents"],
        "first_foot_frame": first,
        "own_nonfoot_frames": nonfoot_frames,
        "controlled_reception": outcome["controlled_reception"],
        "tail_maximum_foot_distance_m": detail["tail_maximum_foot_distance_m"],
        "tail_maximum_ball_speed_mps": detail["tail_maximum_ball_speed_mps"],
        "peak_actual_residual_rad": float(
            np.max(np.abs(np.asarray(trace["receiving_oracle_delta_rad"])))
        ),
        "result_hash": hash_json(info),
    }
    return summary, trace


def audition(
    asset_root: Path,
    policy: Path,
    parent_report: Path,
    expert_report: Path,
    right_report: Path,
    output: Path,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY eight-G1 evidence required")
    parent = json.loads(parent_report.read_text())
    experts = json.loads(expert_report.read_text())
    right_parent = json.loads(right_report.read_text())
    if (
        parent["report_hash"] != hash_json({k: v for k, v in parent.items() if k != "report_hash"})
        or experts["report_hash"]
        != hash_json({k: v for k, v in experts.items() if k != "report_hash"})
        or experts["status"] != "DEVELOPMENT_BILATERAL_CONTROLLED_UNVALIDATED"
        or experts["parent_report_hash"] != right_parent["report_hash"]
        or right_parent["report_hash"]
        != hash_json({k: v for k, v in right_parent.items() if k != "report_hash"})
        or parent["selected"] is None
        or parent["policy_hash"] != hash_bytes(policy.read_bytes())
    ):
        raise ValueError("sealed parent and conditional proxy experts required")
    coordination = tuple(parent["selected"]["weights"])
    left = tuple(experts["best_left"]["weights"])
    right = tuple(right_parent["best_right"]["weights"])
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_precontact_expert_world_v150.py",
            "src/rosclaw_soccer/rsi/receiving_precontact_expert.py",
            "src/rosclaw_soccer/training/receiving_oracle_schedule.py",
            "src/rosclaw_soccer/training/receiving_feedback.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    output.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    for course, expected in zip(COURSES, parent["selected"]["trials"], strict=True):
        baseline, baseline_trace = run(asset_root, policy, course, coordination, None, None)
        if baseline["result_hash"] != expected["result_hash"]:
            raise ValueError("old A1 baseline failed sealed parent reproduction")
        zero, zero_trace = run(asset_root, policy, course, coordination, (0.0,) * 12, (0.0,) * 12)
        physical_equal = all(
            np.array_equal(np.asarray(baseline_trace[key]), np.asarray(zero_trace[key]))
            for key in PHYSICAL_KEYS
        )
        rows.append(
            {
                "course": vars(course),
                "baseline": baseline,
                "a2_zero": zero,
                "zero_physics_equal": physical_equal,
            }
        )
        if not physical_equal:
            break
    if all(row["zero_physics_equal"] for row in rows) and len(rows) == 2:
        for row, course in zip(rows, COURSES, strict=True):
            candidate, _ = run(asset_root, policy, course, coordination, left, right)
            row["candidate"] = candidate
    qualified = len(rows) == 2 and all(row["zero_physics_equal"] for row in rows)
    controlled = qualified and all(
        row["candidate"]["safe"]
        and not row["candidate"]["fault_agents"]
        and row["candidate"]["first_foot_frame"] is not None
        and not row["candidate"]["own_nonfoot_frames"]
        and row["candidate"]["controlled_reception"]
        for row in rows
    )
    report = {
        "schema": SCHEMA,
        "parent_report_hash": parent["report_hash"],
        "expert_report_hash": experts["report_hash"],
        "source_hashes": sources,
        "rows": rows,
        "status": "DEVELOPMENT_EIGHT_G1_BILATERAL_CONTROLLED_UNVALIDATED"
        if controlled
        else "REJECTED_EIGHT_G1_CONTROLLED_GATE"
        if qualified
        else "REJECTED_ZERO_AUTHORITY_GATE",
        "fresh_exam_required": True,
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during eight-G1 authority audition")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--expert-report", type=Path, required=True)
    parser.add_argument("--right-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audition(
        args.asset_root,
        args.policy,
        args.parent_report,
        args.expert_report,
        args.right_report,
        args.output,
    )
    print(json.dumps({"status": report["status"], "report_hash": report["report_hash"]}))


if __name__ == "__main__":
    main()
