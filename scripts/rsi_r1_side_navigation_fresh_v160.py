"""SIM_ONLY autonomous side selection and untouched receiving-course exam."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from rsi_r1_posttouch_navigation_v158 import POST_WEIGHTS
from rsi_r1_posttouch_navigation_v158 import run as run_parent
from rsi_r1_taskspace_feedback_v125 import COURSES

from rosclaw_soccer.rsi.receiving_side_conditioned_expert import ReceivingSideConditionedExpert
from rosclaw_soccer.rsi.team_receive_contact_evidence import (
    ReceiveContactEvidence,
    ReceiveContactMailbox,
)
from rosclaw_soccer.rsi.team_receive_side_navigation import TeamReceiveSideNavigation
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.receiving_experiment import simulate_r0_receiving_course
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule
from rosclaw_soccer.training.receiving_rollout import explain_receiving_window, receiving_window
from rosclaw_soccer.training.role_receiving_courses import ReceivingCourse

SCHEMA = "rosclaw_soccer.rsi.r1_side_navigation_fresh_v160.result.v1"
FRESH_COURSES = (
    ReceivingCourse("red.finisher", 160001, 1.25, 0.06),
    ReceivingCourse("red.finisher", 160002, 1.35, 0.10),
    ReceivingCourse("red.finisher", 160003, 1.50, -0.06),
    ReceivingCourse("red.finisher", 160004, 1.40, -0.10),
)


def run_candidate(
    asset_root: Path,
    policy: Path,
    course: ReceivingCourse,
    coordination: tuple[float, ...],
    left: tuple[float, ...],
    right: tuple[float, ...],
) -> dict[str, Any]:
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
    navigation = TeamReceiveSideNavigation(
        course.agent_id,
        hash_bytes((asset_root / "policy/loco_mode/model/policy_29dof.pt").read_bytes()),
        hash_bytes((asset_root / "policy/loco_mode/config/LocoMode.yaml").read_bytes()),
        mailbox,
        POST_WEIGHTS[6],
        POST_WEIGHTS[0],
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
    return {
        "course": vars(course),
        "selected_side": navigation.selected_side,
        "feedback_selected_side": feedback._selected_side,
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
        "navigation_active_frames": navigation.post_active_frames,
        "navigation_peak_delta_mps": navigation.peak_delta_mps,
        "initial_ball_pose": np.asarray(trace["ball_pose"])[0].tolist(),
        "result_hash": hash_json(info),
    }


def clean(row: dict[str, Any]) -> bool:
    return bool(
        row["safe"]
        and not row["fault_agents"]
        and row["first_foot_frame"] is not None
        and not row["own_nonfoot_frames"]
    )


def exam(
    asset_root: Path,
    policy: Path,
    parent_report: Path,
    right_report: Path,
    native_report: Path,
    joint_report: Path,
    output: Path,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY fresh exam evidence required")
    parent, right_parent, native, joint = (
        json.loads(path.read_text())
        for path in (parent_report, right_report, native_report, joint_report)
    )
    if any(
        item["report_hash"] != hash_json({k: v for k, v in item.items() if k != "report_hash"})
        for item in (parent, right_parent, native, joint)
    ) or (
        joint["status"] != "DEVELOPMENT_NATIVE_BILATERAL_CONTROLLED_UNVALIDATED"
        or not joint["right_retained"]
        or parent["policy_hash"] != hash_bytes(policy.read_bytes())
    ):
        raise ValueError("sealed local bilateral candidate required")
    coordination = tuple(parent["selected"]["weights"])
    left_parent = tuple(native["best"]["weights"])
    left_candidate = tuple(joint["best"]["left_weights"])
    right = tuple(
        float(v) for v in np.asarray(right_parent["best_right"]["weights"], dtype=np.float64) * 0.7
    )
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_side_navigation_fresh_v160.py",
            "src/rosclaw_soccer/rsi/team_receive_side_navigation.py",
            "src/rosclaw_soccer/rsi/receiving_side_conditioned_expert.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
        )
    }
    output.mkdir(parents=True)
    development = []
    for course in COURSES:
        candidate = run_candidate(asset_root, policy, course, coordination, left_candidate, right)
        development.append(candidate)
        print(json.dumps({"partition": "development", "row": candidate}), flush=True)
    expected = (joint["left_check"], joint["right_check"])
    local_equivalent = all(
        row[key] == old[key]
        for row, old in zip(development, expected, strict=True)
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
    side_aligned = all(
        row["selected_side"] == row["feedback_selected_side"] == side
        for row, side in zip(development, (0, 1), strict=True)
    )
    fresh = []
    if local_equivalent and side_aligned:
        for course in FRESH_COURSES:
            parent_row, _ = run_parent(
                asset_root, policy, course, coordination, left_parent, right, None
            )
            candidate = run_candidate(
                asset_root, policy, course, coordination, left_candidate, right
            )
            row = {"course": vars(course), "parent": parent_row, "candidate": candidate}
            fresh.append(row)
            (output / "progress.json").write_text(
                json.dumps(fresh, indent=2, allow_nan=False) + "\n"
            )
            print(json.dumps({"partition": "fresh", "row": row}), flush=True)
    fresh_distinct = len(fresh) == len(FRESH_COURSES) and len(
        {tuple(row["candidate"]["initial_ball_pose"][:3]) for row in fresh}
    ) == len(FRESH_COURSES)
    fresh_success = (
        fresh_distinct
        and all(
            clean(row["candidate"]) and row["candidate"]["controlled_reception"] for row in fresh
        )
        and all(
            row["candidate"]["selected_side"]
            == row["candidate"]["feedback_selected_side"]
            == (0 if row["course"]["lateral_m"] > 0 else 1)
            for row in fresh
        )
    )
    report = {
        "schema": SCHEMA,
        "joint_report_hash": joint["report_hash"],
        "source_hashes": sources,
        "partition": "UNTOUCHED_NATIVE_EIGHT_G1_SIDE_SELECTION_EXAM",
        "local_equivalent": local_equivalent,
        "side_aligned": side_aligned,
        "development_rows": development,
        "fresh_distinct": fresh_distinct,
        "fresh_rows": fresh,
        "fresh_success": fresh_success,
        "status": "FRESH_LOCAL_RECEIVING_QUALIFIED_NOT_MATCH"
        if fresh_success
        else "REJECTED_FRESH_RECEIVING_GATE"
        if local_equivalent and side_aligned
        else "REJECTED_SIDE_SELECTION_OR_RETENTION_GATE",
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during fresh exam")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--right-report", type=Path, required=True)
    parser.add_argument("--native-report", type=Path, required=True)
    parser.add_argument("--joint-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = exam(
        args.asset_root,
        args.policy,
        args.parent_report,
        args.right_report,
        args.native_report,
        args.joint_report,
        args.output,
    )
    print(json.dumps({"status": report["status"], "report_hash": report["report_hash"]}))


if __name__ == "__main__":
    main()
