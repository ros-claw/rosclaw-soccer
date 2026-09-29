"""SIM_ONLY joint post-touch follow after clean short-window compliance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from rsi_r1_left_positive_mining_v163 import rank
from rsi_r1_side_navigation_fresh_v160 import FRESH_COURSES, clean

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

SCHEMA = "rosclaw_soccer.rsi.r1_impedance_follow_v170.result.v1"
SETTINGS = (
    *((0.0, float(nav)) for nav in (-0.5, -0.2, 0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)),
    *((0.2, float(nav)) for nav in (0.2, 0.4, 0.6, 0.8)),
    *((0.4, float(nav)) for nav in (0.2, 0.4, 0.6, 0.8)),
)


def run(
    asset_root: Path,
    policy: Path,
    course: ReceivingCourse,
    coordination: tuple[float, ...],
    left: tuple[float, ...],
    right: tuple[float, ...],
    foot_gain: float,
    navigation_gain: float,
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
        left_post_gain=foot_gain,
        right_post_gain=0.4,
    )
    navigation = TeamReceiveSideNavigation(
        course.agent_id,
        hash_bytes((asset_root / "policy/loco_mode/model/policy_29dof.pt").read_bytes()),
        hash_bytes((asset_root / "policy/loco_mode/config/LocoMode.yaml").read_bytes()),
        mailbox,
        (0.0, 0.0, navigation_gain, 0.0, 0.0),
        (0.0,) * 5,
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
        research_contact_leg_stiffness_scale=0.4,
        research_contact_distance_threshold_m=0.24,
        research_contact_max_active_substeps=32,
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
    active = np.asarray(trace["receiving_impedance_active_substeps"], dtype=np.int64)
    return {
        "course": vars(course),
        "foot_gain": foot_gain,
        "navigation_gain": navigation_gain,
        "active_substeps": int(active.sum()),
        "active_frames": np.flatnonzero(active > 0).tolist(),
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
    mining_report: Path,
    window_report: Path,
    output: Path,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY impedance-follow evidence required")
    parent, right_parent, mining, window = (
        json.loads(path.read_text())
        for path in (parent_report, right_report, mining_report, window_report)
    )
    if any(
        item["report_hash"] != hash_json({k: v for k, v in item.items() if k != "report_hash"})
        for item in (parent, right_parent, mining, window)
    ) or (
        window["status"] != "REJECTED_IMPEDANCE_WINDOW_CONTROLLED_GATE"
        or not window["anchor_equal"]
        or parent["policy_hash"] != hash_bytes(policy.read_bytes())
    ):
        raise ValueError("sealed clean short-window impedance lineage required")
    course = FRESH_COURSES[mining["course_index"]]
    coordination = tuple(parent["selected"]["weights"])
    left = tuple(mining["best"]["left_weights"])
    right = tuple(
        float(v) for v in np.asarray(right_parent["best_right"]["weights"], dtype=np.float64) * 0.7
    )
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_impedance_follow_v170.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
            "src/rosclaw_soccer/rsi/team_receive_side_navigation.py",
        )
    }
    output.mkdir(parents=True)
    rows = []
    anchor_equal = False
    anchor_old = next(
        row
        for row in window["rows"]
        if row["max_active_substeps"] == 32 and row["stiffness_scale"] == 0.4
    )
    for foot_gain, navigation_gain in SETTINGS:
        row = run(asset_root, policy, course, coordination, left, right, foot_gain, navigation_gain)
        if row["active_substeps"] > 32:
            raise ValueError("focal impedance budget exceeded")
        rows.append(row)
        if foot_gain == 0.0 and navigation_gain == 0.5:
            anchor_equal = all(
                row[key] == anchor_old[key]
                for key in (
                    "active_substeps",
                    "active_frames",
                    "safe",
                    "fault_agents",
                    "first_foot_frame",
                    "own_nonfoot_frames",
                    "controlled_reception",
                    "tail_maximum_foot_distance_m",
                    "tail_maximum_ball_speed_mps",
                )
            )
        (output / "progress.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
        print(
            json.dumps(
                {
                    "foot_gain": foot_gain,
                    "navigation_gain": navigation_gain,
                    "clean": clean(row),
                    "controlled": row["controlled_reception"],
                    "nonfoot": row["own_nonfoot_frames"],
                    "distance": row["tail_maximum_foot_distance_m"],
                    "speed": row["tail_maximum_ball_speed_mps"],
                }
            ),
            flush=True,
        )
    best = max(rows, key=rank)
    check = run(
        asset_root,
        policy,
        course,
        coordination,
        left,
        right,
        best["foot_gain"],
        best["navigation_gain"],
    )
    positive = clean(check) and check["controlled_reception"]
    report = {
        "schema": SCHEMA,
        "window_report_hash": window["report_hash"],
        "source_hashes": sources,
        "partition": "CONSUMED_NATIVE_EIGHT_G1_IMPEDANCE_POSTTOUCH_FOLLOW",
        "course": vars(course),
        "anchor_equal": anchor_equal,
        "rows": rows,
        "best": best,
        "check": check,
        "status": "DEVELOPMENT_IMPEDANCE_FOLLOW_CONTROLLED_UNVALIDATED"
        if positive and anchor_equal
        else "REJECTED_IMPEDANCE_FOLLOW_GATE"
        if anchor_equal
        else "REJECTED_IMPEDANCE_FOLLOW_ANCHOR_GATE",
        "fresh_exam_required": True,
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during impedance-follow training")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--right-report", type=Path, required=True)
    parser.add_argument("--mining-report", type=Path, required=True)
    parser.add_argument("--window-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = train(
        args.asset_root,
        args.policy,
        args.parent_report,
        args.right_report,
        args.mining_report,
        args.window_report,
        args.output,
    )
    print(json.dumps({"status": report["status"], "report_hash": report["report_hash"]}))


if __name__ == "__main__":
    main()
