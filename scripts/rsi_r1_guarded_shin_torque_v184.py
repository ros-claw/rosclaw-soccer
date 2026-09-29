"""SIM_ONLY guarded 500 Hz focal knee-torque feasibility probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.asset_qualification import trajectory_digest
from rosclaw_soccer.rsi.receiving_predictive_shin_expert import ReceivingPredictiveShinExpert
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

SCHEMA = "rosclaw_soccer.rsi.r1_guarded_shin_torque_v184.result.v1"
TORQUES = (0.0, -2.0, 2.0, -4.0, 4.0, -8.0, 8.0)


def run(
    asset_root: Path,
    policy: Path,
    course: ReceivingCourse,
    coordination: tuple[float, ...],
    left: tuple[float, ...],
    right: tuple[float, ...],
    old_slope: tuple[float, ...],
    far_slope: tuple[float, ...],
    torque_nm: float,
) -> dict[str, Any]:
    schedule = ReceivingOracleSchedule(
        course.agent_id, "A2_body29_precontact", 15, 10, ((0.0,) * 29,)
    )
    mailbox = ReceiveContactMailbox(course.agent_id)
    feedback = ReceivingPredictiveShinExpert(
        course.agent_id,
        schedule.contract_hash,
        mailbox,
        coordination,
        left,
        right,
        left_post_gain=0.0,
        right_post_gain=0.4,
        old_lateral_slope=old_slope,
        far_lateral_slope=far_slope,
        shin_target_gap_m=0.03,
        shin_prediction_horizon_sec=0.06,
        shin_knee_gain_rad=0.0,
    )
    navigation = TeamReceiveSideNavigation(
        course.agent_id,
        hash_bytes((asset_root / "policy/loco_mode/model/policy_29dof.pt").read_bytes()),
        hash_bytes((asset_root / "policy/loco_mode/config/LocoMode.yaml").read_bytes()),
        mailbox,
        (0.0, 0.0, 0.5, 0.0, 0.0),
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
        research_focal_shin_reflex_torque_nm=torque_nm,
        physics_evidence_consumers={course.agent_id: ReceiveContactEvidence(mailbox)},
    )
    info = result.to_dict()
    legacy = dict(info)
    legacy["trajectory_hash"] = trajectory_digest(
        {
            name: np.asarray(values)
            for name, values in trace.items()
            if name != "research_focal_shin_reflex_active_substeps"
        }
    )
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
    reflex = np.asarray(trace["research_focal_shin_reflex_active_substeps"], dtype=np.int64)
    gaps = np.asarray(trace["research_focal_left_shin_clearance_substeps_m"])
    return {
        "course": vars(course),
        "torque_nm": torque_nm,
        "active_substeps": int(active.sum()),
        "reflex_active_substeps": int(reflex.sum()),
        "reflex_active_frames": np.flatnonzero(reflex > 0).tolist(),
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
        "minimum_shin_clearance_substep_m": float(gaps[28:36].min()),
        "legacy_projected_result_hash": hash_json(legacy),
        "result_hash": hash_json(info),
    }


def train(
    asset_root: Path,
    policy: Path,
    parent_report: Path,
    right_report: Path,
    refine_report: Path,
    lateral_report: Path,
    precontact_report: Path,
    output: Path,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY guarded-torque evidence required")
    parent, right_parent, refine, lateral, precontact = (
        json.loads(path.read_text())
        for path in (
            parent_report,
            right_report,
            refine_report,
            lateral_report,
            precontact_report,
        )
    )
    if any(
        item["report_hash"] != hash_json({k: v for k, v in item.items() if k != "report_hash"})
        for item in (parent, right_parent, refine, lateral, precontact)
    ) or (
        precontact["status"] != "REJECTED_PRECONTACT_SUBSTEP_CONTROLLED_GATE"
        or parent["policy_hash"] != hash_bytes(policy.read_bytes())
    ):
        raise ValueError("sealed failed pre-contact substep anchor required")
    course = ReceivingCourse("red.finisher", 160002, 1.35, 0.10)
    coordination = tuple(parent["selected"]["weights"])
    left = tuple(refine["best"]["left_weights"])
    right = tuple(
        float(v) for v in np.asarray(right_parent["best_right"]["weights"], dtype=np.float64) * 0.7
    )
    old_slope = tuple(lateral["best"]["slope"])
    far_slope = tuple(precontact["best"]["far_slope"])
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_guarded_shin_torque_v184.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    output.mkdir(parents=True)
    rows = []
    for torque in TORQUES:
        row = run(
            asset_root, policy, course, coordination, left, right, old_slope, far_slope, torque
        )
        if row["active_substeps"] > 32:
            raise ValueError("compliance budget exceeded")
        rows.append(row)
        (output / "progress.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "torque": torque,
                    "safe": row["safe"],
                    "controlled": row["controlled_reception"],
                    "nonfoot": row["own_nonfoot_frames"],
                    "min_shin_m": row["minimum_shin_clearance_substep_m"],
                    "distance": row["tail_maximum_foot_distance_m"],
                    "speed": row["tail_maximum_ball_speed_mps"],
                }
            ),
            flush=True,
        )
    zero_equal = rows[0]["legacy_projected_result_hash"] == precontact["far_check"]["result_hash"]
    controlled = [
        row
        for row in rows
        if row["safe"]
        and not row["fault_agents"]
        and row["first_foot_frame"] is not None
        and not row["own_nonfoot_frames"]
        and row["controlled_reception"]
    ]
    report = {
        "schema": SCHEMA,
        "precontact_report_hash": precontact["report_hash"],
        "source_hashes": sources,
        "partition": "CONSUMED_NATIVE_EIGHT_G1_GUARDED_500HZ_TORQUE_PROBE",
        "zero_equal": zero_equal,
        "rows": rows,
        "status": "DEVELOPMENT_FAR_GUARDED_TORQUE_CONTROLLED_UNVALIDATED"
        if zero_equal and controlled
        else "REJECTED_FAR_GUARDED_TORQUE_GATE"
        if zero_equal
        else "REJECTED_FAR_GUARDED_TORQUE_ANCHOR_GATE",
        "fresh_exam_required": True,
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during guarded torque probe")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--right-report", type=Path, required=True)
    parser.add_argument("--refine-report", type=Path, required=True)
    parser.add_argument("--lateral-report", type=Path, required=True)
    parser.add_argument("--precontact-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = train(
        args.asset_root,
        args.policy,
        args.parent_report,
        args.right_report,
        args.refine_report,
        args.lateral_report,
        args.precontact_report,
        args.output,
    )
    print(json.dumps({"status": report["status"], "report_hash": report["report_hash"]}))


if __name__ == "__main__":
    main()
