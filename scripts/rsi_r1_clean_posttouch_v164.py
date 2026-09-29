"""SIM_ONLY post-touch foot/body coordination on clean native contact starts."""

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

SCHEMA = "rosclaw_soccer.rsi.r1_clean_posttouch_v164.result.v1"
SETTINGS = tuple(
    (float(foot), float(navigation))
    for foot in (0.0, 0.2, 0.4, 0.6)
    for navigation in (0.3, 0.5, 0.7)
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
        "foot_gain": foot_gain,
        "navigation_gain": navigation_gain,
        "selected_side": feedback._selected_side,
        "navigation_selected_side": navigation.selected_side,
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
    output: Path,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY clean-posttouch evidence required")
    parent, right_parent, mining = (
        json.loads(path.read_text()) for path in (parent_report, right_report, mining_report)
    )
    if any(
        item["report_hash"] != hash_json({k: v for k, v in item.items() if k != "report_hash"})
        for item in (parent, right_parent, mining)
    ) or (
        mining["status"] != "REJECTED_SINGLE_COURSE_POSITIVE_GATE"
        or not clean(mining["selected_check"])
        or parent["policy_hash"] != hash_bytes(policy.read_bytes())
    ):
        raise ValueError("sealed clean native contact seed required")
    course = FRESH_COURSES[mining["course_index"]]
    coordination = tuple(parent["selected"]["weights"])
    left = tuple(mining["best"]["left_weights"])
    right = tuple(
        float(v) for v in np.asarray(right_parent["best_right"]["weights"], dtype=np.float64) * 0.7
    )
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_clean_posttouch_v164.py",
            "src/rosclaw_soccer/rsi/team_receive_side_navigation.py",
            "src/rosclaw_soccer/rsi/receiving_side_conditioned_expert.py",
        )
    }
    output.mkdir(parents=True)
    rows = []
    for foot_gain, navigation_gain in SETTINGS:
        summary = run(
            asset_root,
            policy,
            course,
            coordination,
            left,
            right,
            foot_gain,
            navigation_gain,
        )
        rows.append(summary)
        (output / "progress.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
        print(
            json.dumps(
                {
                    "course": vars(course),
                    "foot_gain": foot_gain,
                    "navigation_gain": navigation_gain,
                    "clean": clean(summary),
                    "controlled": summary["controlled_reception"],
                    "nonfoot": summary["own_nonfoot_frames"],
                    "distance": summary["tail_maximum_foot_distance_m"],
                    "speed": summary["tail_maximum_ball_speed_mps"],
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
        "mining_report_hash": mining["report_hash"],
        "source_hashes": sources,
        "partition": "CONSUMED_NATIVE_EIGHT_G1_CLEAN_POSTTOUCH_CURRICULUM",
        "course": vars(course),
        "rows": rows,
        "best": best,
        "check": check,
        "status": "DEVELOPMENT_SINGLE_COURSE_POSTTOUCH_UNVALIDATED"
        if positive
        else "REJECTED_POSTTOUCH_CONTROLLED_GATE",
        "fresh_exam_required": True,
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during clean posttouch learning")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--right-report", type=Path, required=True)
    parser.add_argument("--mining-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = train(
        args.asset_root,
        args.policy,
        args.parent_report,
        args.right_report,
        args.mining_report,
        args.output,
    )
    print(json.dumps({"status": report["status"], "report_hash": report["report_hash"]}))


if __name__ == "__main__":
    main()
