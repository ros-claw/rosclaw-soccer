"""SIM_ONLY left post-foot velocity matching with right skill retained."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from rsi_r1_taskspace_feedback_v125 import COURSES

from rosclaw_soccer.rsi.receiving_side_conditioned_expert import ReceivingSideConditionedExpert
from rosclaw_soccer.rsi.team_receive_contact_evidence import (
    ReceiveContactEvidence,
    ReceiveContactMailbox,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.receiving_experiment import simulate_r0_receiving_course
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule
from rosclaw_soccer.training.receiving_rollout import explain_receiving_window, receiving_window
from rosclaw_soccer.training.role_receiving_courses import ReceivingCourse

SCHEMA = "rosclaw_soccer.rsi.r1_left_velocity_cushion_v157.result.v1"
SETTINGS = (
    (0.2, 0.05, 0.25, 0.12),
    (0.2, 0.10, 0.25, 0.12),
    (0.2, 0.15, 0.25, 0.12),
    (0.4, 0.05, 0.25, 0.12),
    (0.4, 0.10, 0.25, 0.12),
    (0.4, 0.15, 0.25, 0.12),
    (0.4, 0.10, 0.18, -0.08),
    (0.6, 0.05, 0.25, 0.12),
    (0.6, 0.10, 0.25, 0.12),
    (0.6, 0.15, 0.25, 0.12),
)


def run(
    asset_root: Path,
    policy: Path,
    course: ReceivingCourse,
    coordination: tuple[float, ...],
    left: tuple[float, ...],
    right: tuple[float, ...],
    setting: tuple[float, float, float, float],
) -> dict[str, Any]:
    gain, horizon, depth, lateral = setting
    schedule = ReceivingOracleSchedule(
        course.agent_id, "A2_body29_precontact", 15, 10, ((0.0,) * 29,)
    )
    mailbox = ReceiveContactMailbox(course.agent_id)
    actor = ReceivingSideConditionedExpert(
        course.agent_id,
        schedule.contract_hash,
        mailbox,
        coordination,
        left,
        right,
        left_post_gain=gain,
        right_post_gain=0.4,
        left_velocity_horizon_sec=horizon,
        left_target_depth_m=depth,
        left_target_lateral_m=lateral,
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
    foot = np.asarray(trace["ball_contact_agent_code"])
    foot_force = np.asarray(trace["ball_contact_force_n"])
    nonfoot = np.asarray(trace["ball_nonfoot_contact_agent_code"])
    nonfoot_force = np.asarray(trace["ball_nonfoot_contact_force_n"])
    return {
        "course": vars(course),
        "setting": setting,
        "safe": info["safe"],
        "fault_agents": info["physics_evidence_fault_agents"],
        "first_foot_frame": next(
            (frame for frame in range(20, 120) if foot[frame] == code and foot_force[frame] > 0),
            None,
        ),
        "own_nonfoot_frames": [
            frame for frame in range(20, 120) if nonfoot[frame] == code and nonfoot_force[frame] > 0
        ],
        "controlled_reception": outcome["controlled_reception"],
        "tail_maximum_foot_distance_m": detail["tail_maximum_foot_distance_m"],
        "tail_maximum_ball_speed_mps": detail["tail_maximum_ball_speed_mps"],
        "result_hash": hash_json(info),
    }


def train(
    asset_root: Path,
    policy: Path,
    parent_report: Path,
    right_report: Path,
    native_report: Path,
    side_report: Path,
    output: Path,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY left velocity evidence required")
    parent = json.loads(parent_report.read_text())
    right_parent = json.loads(right_report.read_text())
    native = json.loads(native_report.read_text())
    side = json.loads(side_report.read_text())
    if any(
        item["report_hash"] != hash_json({k: v for k, v in item.items() if k != "report_hash"})
        for item in (parent, right_parent, native, side)
    ) or (
        not side["right_retained"]
        or side["status"] != "REJECTED_LEFT_CONTROLLED_GATE"
        or parent["policy_hash"] != hash_bytes(policy.read_bytes())
    ):
        raise ValueError("sealed right skill and left course required")
    coordination = tuple(parent["selected"]["weights"])
    left = tuple(native["best"]["weights"])
    right = tuple(
        float(v) for v in np.asarray(right_parent["best_right"]["weights"], dtype=np.float64) * 0.7
    )
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_left_velocity_cushion_v157.py",
            "src/rosclaw_soccer/rsi/receiving_side_conditioned_expert.py",
            "src/rosclaw_soccer/training/receiving_oracle_schedule.py",
        )
    }
    output.mkdir(parents=True)
    right_check = run(
        asset_root,
        policy,
        COURSES[1],
        coordination,
        left,
        right,
        (0.0, 0.0, 0.25, 0.12),
    )
    right_retained = all(
        right_check[key] == side["right_check"][key]
        for key in (
            "safe",
            "fault_agents",
            "first_foot_frame",
            "own_nonfoot_frames",
            "controlled_reception",
            "tail_maximum_foot_distance_m",
            "tail_maximum_ball_speed_mps",
        )
    )
    rows = []
    if right_retained:
        for setting in SETTINGS:
            row = run(asset_root, policy, COURSES[0], coordination, left, right, setting)
            rows.append(row)
            (output / "progress.json").write_text(
                json.dumps(rows, indent=2, allow_nan=False) + "\n"
            )
            print(json.dumps(row), flush=True)
    controlled = any(
        row["safe"]
        and not row["fault_agents"]
        and row["first_foot_frame"] is not None
        and not row["own_nonfoot_frames"]
        and row["controlled_reception"]
        for row in rows
    )
    report = {
        "schema": SCHEMA,
        "side_report_hash": side["report_hash"],
        "source_hashes": sources,
        "partition": "CONSUMED_NATIVE_EIGHT_G1_LEFT_VELOCITY_CUSHION",
        "right_retained": right_retained,
        "right_check": right_check,
        "left_rows": rows,
        "status": "DEVELOPMENT_NATIVE_BILATERAL_CONTROLLED_UNVALIDATED"
        if right_retained and controlled
        else "REJECTED_LEFT_CONTROLLED_GATE"
        if right_retained
        else "REJECTED_RIGHT_RETENTION_GATE",
        "fresh_exam_required": True,
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during left velocity learning")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--right-report", type=Path, required=True)
    parser.add_argument("--native-report", type=Path, required=True)
    parser.add_argument("--side-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = train(
        args.asset_root,
        args.policy,
        args.parent_report,
        args.right_report,
        args.native_report,
        args.side_report,
        args.output,
    )
    print(json.dumps({"status": report["status"], "report_hash": report["report_hash"]}))


if __name__ == "__main__":
    main()
