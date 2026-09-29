"""SIM_ONLY native eight-G1 reconstruction of far-course post-foot contact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from rsi_r1_shin_clearance_v121 import _reconstruct
from rsi_r1_side_navigation_fresh_v160 import FRESH_COURSES

from rosclaw_soccer.rsi.receiving_lateral_piecewise_expert import ReceivingLateralPiecewiseExpert
from rosclaw_soccer.rsi.team_receive_contact_evidence import (
    ReceiveContactEvidence,
    ReceiveContactMailbox,
)
from rosclaw_soccer.rsi.team_receive_side_navigation import TeamReceiveSideNavigation
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.continuous_match_residual_ppo import collection_fixture
from rosclaw_soccer.training.receiving_classroom import r1_contact_tap_receiving_configuration
from rosclaw_soccer.training.receiving_experiment import simulate_r0_receiving_course
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule
from rosclaw_soccer.world.multi_player import build_g1_multi_player_stadium_model

SCHEMA = "rosclaw_soccer.rsi.r1_far_contact_geometry_v176.result.v1"


def diagnose(
    asset_root: Path,
    policy: Path,
    parent_report: Path,
    right_report: Path,
    refine_report: Path,
    lateral_report: Path,
    piecewise_report: Path,
    window_report: Path,
    output: Path,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY geometry evidence required")
    parent, right_parent, refine, lateral, piecewise, window = (
        json.loads(path.read_text())
        for path in (
            parent_report,
            right_report,
            refine_report,
            lateral_report,
            piecewise_report,
            window_report,
        )
    )
    if any(
        item["report_hash"] != hash_json({k: v for k, v in item.items() if k != "report_hash"})
        for item in (parent, right_parent, refine, lateral, piecewise, window)
    ) or (
        window["status"] != "REJECTED_FAR_WINDOW_CONTROLLED_GATE"
        or not window["anchor_equal"]
        or parent["policy_hash"] != hash_bytes(policy.read_bytes())
    ):
        raise ValueError("sealed native far-contact failure required")
    course = FRESH_COURSES[1]
    coordination = tuple(parent["selected"]["weights"])
    left = tuple(refine["best"]["left_weights"])
    right = tuple(
        float(v) for v in np.asarray(right_parent["best_right"]["weights"], dtype=np.float64) * 0.7
    )
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_far_contact_geometry_v176.py",
            "src/rosclaw_soccer/rsi/receiving_lateral_piecewise_expert.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
            "src/rosclaw_soccer/world/multi_player.py",
        )
    }
    schedule = ReceivingOracleSchedule(
        course.agent_id, "A2_body29_precontact", 15, 10, ((0.0,) * 29,)
    )
    mailbox = ReceiveContactMailbox(course.agent_id)
    feedback = ReceivingLateralPiecewiseExpert(
        course.agent_id,
        schedule.contract_hash,
        mailbox,
        coordination,
        left,
        right,
        left_post_gain=0.0,
        right_post_gain=0.4,
        old_lateral_slope=tuple(lateral["best"]["slope"]),
        far_lateral_slope=tuple(piecewise["best"]["far_slope"]),
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
    fixture = collection_fixture(asset_root, keeper_preview=True)
    world, _ = r1_contact_tap_receiving_configuration()
    model = build_g1_multi_player_stadium_model(
        asset_root,
        players=fixture.players,
        spec=fixture.goal,
        left_goal_plane_x_m=world.left_goal_plane_x_m if world.bilateral_goals else None,
    )
    data = mujoco.MjData(model)
    ball_geom = model.geom("ball_geom").id
    geom_ids = {name: model.geom(f"red_finisher_{name}").id for name in ("left_shin", "right_shin")}
    for side in ("left", "right"):
        for index in range(1, 8):
            name = f"{side}_foot{index}_collision"
            geom_ids[name] = model.geom(f"red_finisher_{name}").id
    nonfoot_geom = np.asarray(trace["ball_nonfoot_contact_geom_id"], dtype=np.int64)
    nonfoot_force = np.asarray(trace["ball_nonfoot_contact_force_n"], dtype=np.float64)
    foot_force = np.asarray(trace["ball_contact_force_n"], dtype=np.float64)
    timeline = []
    for frame in range(25, 38):
        _reconstruct(model, data, trace, frame)
        distances = {
            name: float(mujoco.mj_geomDistance(model, data, ball_geom, geom, 10.0, None))
            for name, geom in geom_ids.items()
        }
        geom_id = int(nonfoot_geom[frame])
        timeline.append(
            {
                "frame": frame,
                "time_sec": float(np.asarray(trace["time"])[frame]),
                "nonfoot_geom_id": geom_id,
                "nonfoot_geom_name": model.geom(geom_id).name if geom_id >= 0 else None,
                "nonfoot_force_n": float(nonfoot_force[frame]),
                "foot_force_n": float(foot_force[frame]),
                "clearance_m": distances,
                "ball_pose": np.asarray(trace["ball_pose"])[frame].tolist(),
                "pelvis_pose": np.asarray(trace["red_finisher_pelvis_pose"])[frame].tolist(),
            }
        )
    output.mkdir(parents=True)
    report = {
        "schema": SCHEMA,
        "window_report_hash": window["report_hash"],
        "source_hashes": sources,
        "course": vars(course),
        "world_safe": result.to_dict()["safe"],
        "same_anchor_result_hash": hash_json(result.to_dict())
        == piecewise["checks"][2]["result_hash"],
        "timeline": timeline,
        "status": "DIAGNOSTIC_ONLY_NO_PROMOTION",
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during geometry diagnosis")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--right-report", type=Path, required=True)
    parser.add_argument("--refine-report", type=Path, required=True)
    parser.add_argument("--lateral-report", type=Path, required=True)
    parser.add_argument("--piecewise-report", type=Path, required=True)
    parser.add_argument("--window-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = diagnose(
        args.asset_root,
        args.policy,
        args.parent_report,
        args.right_report,
        args.refine_report,
        args.lateral_report,
        args.piecewise_report,
        args.window_report,
        args.output,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "anchor_equal": report["same_anchor_result_hash"],
                "report_hash": report["report_hash"],
            }
        )
    )


if __name__ == "__main__":
    main()
