"""SIM_ONLY post-contact body relocation on the consumed left receiving course."""

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
from rosclaw_soccer.rsi.team_receive_contact_phase_actor import TeamReceiveContactPhaseActor
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.receiving_experiment import simulate_r0_receiving_course
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule
from rosclaw_soccer.training.receiving_rollout import explain_receiving_window, receiving_window
from rosclaw_soccer.training.role_receiving_courses import ReceivingCourse

SCHEMA = "rosclaw_soccer.rsi.r1_posttouch_navigation_v158.result.v1"
POST_WEIGHTS = (
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (0.5, 0.0, 0.0, 0.0, 0.0),
    (-0.5, 0.0, 0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.5, 0.0, 0.0, 0.0),
    (0.0, -0.5, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.5, 0.0, 0.0),
    (0.0, 0.0, -0.5, 0.0, 0.0),
)
PHYSICS_KEYS = (
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
    left: tuple[float, ...],
    right: tuple[float, ...],
    weights: tuple[float, float, float, float, float] | None,
) -> tuple[dict[str, Any], dict[str, np.ndarray[Any, Any]]]:
    schedule = ReceivingOracleSchedule(
        course.agent_id, "A2_body29_precontact", 15, 10, ((0.0,) * 29,)
    )
    mailbox = ReceiveContactMailbox(course.agent_id)
    feedback = ReceivingSideConditionedExpert(
        course.agent_id,
        schedule.contract_hash,
        mailbox,
        coordination,
        left,
        right,
        left_post_gain=0.0,
        right_post_gain=0.4,
    )
    navigation = (
        TeamReceiveContactPhaseActor(
            course.agent_id,
            hash_bytes((asset_root / "policy/loco_mode/model/policy_29dof.pt").read_bytes()),
            hash_bytes((asset_root / "policy/loco_mode/config/LocoMode.yaml").read_bytes()),
            weights=(0.0,) * 5,
            post_weights=weights,
            mailbox=mailbox,
        )
        if weights is not None
        else None
    )
    result, trace = simulate_r0_receiving_course(
        asset_root=asset_root,
        reference_policy_path=policy,
        course=course,
        scenario_id=f"s199.rsi.r1.coordinated-receiving.{course.seed}",
        configuration_profile="R1_CONTACT_TAP",
        oracle=schedule,
        feedback_provider=feedback,
        research_navigation_policy=navigation,
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
    summary = {
        "course": vars(course),
        "weights": weights,
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
        "navigation_active_frames": 0 if navigation is None else navigation.post_active_frames,
        "navigation_peak_delta_mps": 0.0 if navigation is None else navigation.peak_delta_mps,
        "result_hash": hash_json(info),
    }
    return summary, trace


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
        raise ValueError("new external SIM_ONLY navigation evidence required")
    parent, right_parent, native, side = (
        json.loads(path.read_text())
        for path in (parent_report, right_report, native_report, side_report)
    )
    if any(
        item["report_hash"] != hash_json({k: v for k, v in item.items() if k != "report_hash"})
        for item in (parent, right_parent, native, side)
    ) or (
        side["status"] != "REJECTED_LEFT_CONTROLLED_GATE"
        or not side["right_retained"]
        or parent["policy_hash"] != hash_bytes(policy.read_bytes())
    ):
        raise ValueError("sealed right-skill lineage required")
    coordination = tuple(parent["selected"]["weights"])
    left = tuple(native["best"]["weights"])
    right = tuple(
        float(v) for v in np.asarray(right_parent["best_right"]["weights"], dtype=np.float64) * 0.7
    )
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_posttouch_navigation_v158.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
            "src/rosclaw_soccer/rsi/team_receive_contact_phase_actor.py",
            "src/rosclaw_soccer/rsi/receiving_side_conditioned_expert.py",
        )
    }
    output.mkdir(parents=True)
    right_parent_row, right_trace = run(
        asset_root, policy, COURSES[1], coordination, left, right, None
    )
    right_zero, right_zero_trace = run(
        asset_root, policy, COURSES[1], coordination, left, right, POST_WEIGHTS[0]
    )
    left_parent_row, left_trace = run(
        asset_root, policy, COURSES[0], coordination, left, right, None
    )
    left_zero, left_zero_trace = run(
        asset_root, policy, COURSES[0], coordination, left, right, POST_WEIGHTS[0]
    )
    if any(
        key not in trace
        for trace in (left_trace, left_zero_trace, right_trace, right_zero_trace)
        for key in PHYSICS_KEYS
    ):
        raise ValueError("complete physical zero-action arrays required")
    keys = PHYSICS_KEYS
    zero_equal = all(
        np.array_equal(a[key], b[key])
        for a, b in ((right_trace, right_zero_trace), (left_trace, left_zero_trace))
        for key in keys
    )
    right_retained = all(
        right_parent_row[key] == side["right_check"][key]
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
    rows: list[dict[str, Any]] = []
    if zero_equal and right_retained:
        for weights in POST_WEIGHTS[1:]:
            row, _ = run(asset_root, policy, COURSES[0], coordination, left, right, weights)
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
        "partition": "CONSUMED_NATIVE_EIGHT_G1_POSTTOUCH_NAVIGATION",
        "zero_physics_equal": zero_equal,
        "zero_physics_keys": keys,
        "right_retained": right_retained,
        "right_parent": right_parent_row,
        "right_zero": right_zero,
        "left_parent": left_parent_row,
        "left_zero": left_zero,
        "left_rows": rows,
        "status": "DEVELOPMENT_NATIVE_BILATERAL_CONTROLLED_UNVALIDATED"
        if controlled and right_retained and zero_equal
        else "REJECTED_LEFT_CONTROLLED_GATE"
        if right_retained and zero_equal
        else "REJECTED_ZERO_OR_RIGHT_RETENTION_GATE",
        "fresh_exam_required": True,
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during navigation learning")
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
