"""SIM_ONLY 500 Hz clearance audit of the first shin collision after a foot touch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from rsi_r1_side_navigation_fresh_v160 import FRESH_COURSES

from rosclaw_soccer.providers.g1.asset_qualification import trajectory_digest
from rosclaw_soccer.rsi.receiving_postfoot_clearance_expert import ReceivingPostfootClearanceExpert
from rosclaw_soccer.rsi.team_receive_contact_evidence import (
    ReceiveContactEvidence,
    ReceiveContactMailbox,
)
from rosclaw_soccer.rsi.team_receive_side_navigation import TeamReceiveSideNavigation
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.receiving_experiment import simulate_r0_receiving_course
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule

SCHEMA = "rosclaw_soccer.rsi.r1_postfoot_substep_audit_v180.result.v1"


def run(
    asset_root: Path,
    policy: Path,
    coordination: tuple[float, ...],
    left: tuple[float, ...],
    right: tuple[float, ...],
    old_slope: tuple[float, ...],
    far_slope: tuple[float, ...],
    postfoot: tuple[float, float, float],
) -> dict[str, Any]:
    course = FRESH_COURSES[1]
    schedule = ReceivingOracleSchedule(
        course.agent_id, "A2_body29_precontact", 15, 10, ((0.0,) * 29,)
    )
    mailbox = ReceiveContactMailbox(course.agent_id)
    feedback = ReceivingPostfootClearanceExpert(
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
        postfoot_left_hip_knee_ankle=postfoot,
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
        physics_evidence_consumers={course.agent_id: ReceiveContactEvidence(mailbox)},
    )
    clearances = np.asarray(trace["research_focal_left_shin_clearance_substeps_m"])
    if clearances.ndim != 2 or clearances.shape[1] != 10 or not np.isfinite(clearances).all():
        raise ValueError("finite complete 500 Hz shin clearance trace required")
    timeline = []
    for frame in range(28, 36):
        values = clearances[frame]
        timeline.append(
            {
                "frame": frame,
                "substep_clearance_m": values.tolist(),
                "minimum_substep_clearance_m": float(values.min()),
                "nonfoot_geom_id": int(np.asarray(trace["ball_nonfoot_contact_geom_id"])[frame]),
                "nonfoot_force_n": float(np.asarray(trace["ball_nonfoot_contact_force_n"])[frame]),
                "oracle_left_hip_knee_ankle_rad": np.asarray(trace["receiving_oracle_delta_rad"])[
                    frame, [0, 3, 4]
                ].tolist(),
                "feedback_left_hip_knee_ankle_rad": np.asarray(
                    trace["receiving_feedback_desired_rad"]
                )[frame, [0, 3, 4]].tolist(),
            }
        )
    info = result.to_dict()
    info["trajectory_hash"] = trajectory_digest(
        {
            name: np.asarray(values)
            for name, values in trace.items()
            if name != "research_focal_left_shin_clearance_substeps_m"
        }
    )
    return {
        "postfoot": postfoot,
        "result_hash_with_diagnostic_trace": hash_json(result.to_dict()),
        "legacy_projected_result_hash": hash_json(info),
        "world_safe": result.to_dict()["safe"],
        "timeline": timeline,
    }


def audit(
    asset_root: Path,
    policy: Path,
    parent_report: Path,
    right_report: Path,
    refine_report: Path,
    lateral_report: Path,
    clearance_report: Path,
    knee_report: Path,
    output: Path,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY 500 Hz audit required")
    parent, right_parent, refine, lateral, clearance, knee = (
        json.loads(path.read_text())
        for path in (
            parent_report,
            right_report,
            refine_report,
            lateral_report,
            clearance_report,
            knee_report,
        )
    )
    if any(
        item["report_hash"] != hash_json({k: v for k, v in item.items() if k != "report_hash"})
        for item in (parent, right_parent, refine, lateral, clearance, knee)
    ) or (
        knee["status"] != "REJECTED_KNEE_SWEEP_CONTROLLED_GATE"
        or parent["policy_hash"] != hash_bytes(policy.read_bytes())
    ):
        raise ValueError("sealed failed knee-sweep evidence required")
    coordination = tuple(parent["selected"]["weights"])
    left = tuple(refine["best"]["left_weights"])
    right = tuple(
        float(v) for v in np.asarray(right_parent["best_right"]["weights"], dtype=np.float64) * 0.7
    )
    old_slope = tuple(lateral["best"]["slope"])
    far_slope = tuple(clearance["best"]["far_slope"])
    selected = tuple(knee["best"]["postfoot"])
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_postfoot_substep_audit_v180.py",
            "src/rosclaw_soccer/rsi/receiving_postfoot_clearance_expert.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    output.mkdir(parents=True)
    zero = run(asset_root, policy, coordination, left, right, old_slope, far_slope, (0.0, 0.0, 0.0))
    active = run(asset_root, policy, coordination, left, right, old_slope, far_slope, selected)
    anchor_equal = active["legacy_projected_result_hash"] == knee["check"]["result_hash"]
    report = {
        "schema": SCHEMA,
        "knee_report_hash": knee["report_hash"],
        "source_hashes": sources,
        "anchor_equal": anchor_equal,
        "zero": zero,
        "active": active,
        "status": "DIAGNOSTIC_500HZ_ONLY_NO_PROMOTION"
        if anchor_equal
        else "REJECTED_500HZ_ANCHOR_GATE",
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during 500 Hz diagnostic")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--right-report", type=Path, required=True)
    parser.add_argument("--refine-report", type=Path, required=True)
    parser.add_argument("--lateral-report", type=Path, required=True)
    parser.add_argument("--clearance-report", type=Path, required=True)
    parser.add_argument("--knee-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(
        args.asset_root,
        args.policy,
        args.parent_report,
        args.right_report,
        args.refine_report,
        args.lateral_report,
        args.clearance_report,
        args.knee_report,
        args.output,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "anchor_equal": report["anchor_equal"],
                "report_hash": report["report_hash"],
            }
        )
    )


if __name__ == "__main__":
    main()
