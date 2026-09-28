"""Measure actual foot and ball velocities around shared-world G1 contacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.rsi.team_intercept_navigation import TeamInterceptNavigation
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.independent_team_world import IndependentTeamWorldScenario
from rosclaw_soccer.training.independent_team_growth import build_independent_three_vs_three_fixture
from scripts.rsi_team_taskspace_first_touch import _run_one


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--prior-library", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("output directory already exists")
    protocol: dict[str, Any] = json.loads(args.protocol.read_text(encoding="utf-8"))
    courses = protocol.get("courses", [])
    prior = json.loads(args.prior_library.read_text(encoding="utf-8"))
    if (
        protocol.get("schema") != "rsi_team_contact_velocity_diagnostic_protocol_v20"
        or protocol.get("development_only") is not True
        or protocol.get("promotion_authorized") is not False
        or protocol.get("frames") != 250
        or protocol.get("diagnostic_arm")
        != {
            "name": "fixed_left_negative",
            "foot_selection": "left",
            "forward_gain": 0.0,
            "lateral_gain": -0.8,
        }
        or [course.get("name") for course in courses] != [f"h{index:02d}" for index in range(1, 9)]
        or prior.get("report_hash") != protocol.get("parent_action_library_report_hash")
        or prior.get("schema") != "rsi_team_action_library_feasibility_report_v19"
    ):
        raise ValueError("uncommitted SIM_ONLY contact-velocity diagnosis")
    prior_arm = next(row for row in prior["rows"] if row["arm"]["name"] == "fixed_left_negative")
    qualification = qualify_g1_assets(args.asset_root)
    qualification.require_eligible()
    fixture = build_independent_three_vs_three_fixture(args.asset_root)
    policy_hash = hash_bytes(
        (args.asset_root / "policy/loco_mode/model/policy_29dof.pt").read_bytes()
    )
    config_hash = hash_bytes(
        (args.asset_root / "policy/loco_mode/config/LocoMode.yaml").read_bytes()
    )
    root = Path(__file__).parents[1]
    sources = (
        Path(__file__),
        Path(__file__).with_name("rsi_team_taskspace_first_touch.py"),
        root / "src/rosclaw_soccer/skills/team/foot_kinematics.py",
        root / "src/rosclaw_soccer/skills/team/independent_team_world.py",
    )
    source_hashes = {str(p.relative_to(root)): hash_bytes(p.read_bytes()) for p in sources}
    args.output_dir.mkdir(parents=True)
    rows = []
    for course in courses:
        scenario = IndependentTeamWorldScenario(
            course["scenario_id"],
            tuple(course["ball_initial_position_m"]),
            tuple(course["ball_initial_velocity_mps"]),
            course["seed"],
        )
        nav = TeamInterceptNavigation(
            protocol["focal_agent_id"], policy_hash, config_hash, 0.0, -0.8
        )
        folder = args.output_dir / course["name"]
        report = _run_one(
            mode="candidate",
            asset_root=args.asset_root,
            output_dir=folder,
            fixture=fixture,
            scenario=scenario,
            protocol=protocol,
            navigation_policy=nav,
        )
        old_path = (
            args.prior_library.parent
            / "fixed_left_negative"
            / course["name"]
            / "candidate/report.json"
        )
        old = json.loads(old_path.read_text(encoding="utf-8"))
        if (
            old["report_hash"]
            != next(row for row in prior_arm["outcomes"] if row["course"] == course["name"])[
                "report_hash"
            ]
            or report["trajectory_digest"] != old["trajectory_digest"]
        ):
            raise ValueError("read-only velocity enrichment changed frozen world physics")
        with np.load(folder / "candidate/taskspace_trace.npz", allow_pickle=False) as action:
            foot_velocity = np.asarray(action["pre_step_foot_linear_velocity_w"])
            sides = np.asarray(action["taskspace_selected_side"])
            feet = np.asarray(action["pre_step_foot_link_position_w"])
            ball = np.asarray(action["pre_step_ball_position_local_m"])
            contact_position = np.asarray(action["observed_own_foot_contact_position_w"])
            contact_normal = np.asarray(action["observed_own_foot_contact_normal_ball_to_foot_w"])
            contact_relative_velocity = np.asarray(
                action["observed_own_foot_counterpart_minus_ball_velocity_w"]
            )
        with np.load(folder / "candidate/trajectory.npz", allow_pickle=False) as physics:
            ball_velocity = np.asarray(physics["ball_velocity"])
            force = np.asarray(physics["ball_contact_force_n"])
            effector = np.asarray(physics["ball_contact_effector_code"])
        if (
            foot_velocity.shape != (250, 1, 2, 3)
            or not np.isfinite(foot_velocity).all()
            or sides.shape != (250, 1)
            or contact_position.shape != (250, 1, 3)
            or contact_normal.shape != (250, 1, 3)
            or contact_relative_velocity.shape != (250, 1, 3)
        ):
            raise ValueError("invalid measured full-body foot velocity trace")
        frames = report["focal_foot_contact_frames"]
        first = frames[0] if frames else None
        selected_side = int(sides[first, 0]) if first is not None else None
        rows.append(
            {
                "course": course["name"],
                "report_hash": report["report_hash"],
                "trajectory_digest_matches_v19": True,
                "focal_foot_contact_frames": frames,
                "focal_nonfoot_contact_frames": report["focal_nonfoot_contact_frames"],
                "first_contact_selected_side": selected_side,
                "first_contact_actual_foot_code": (
                    int(effector[first]) if first is not None else None
                ),
                "selected_side_matches_contact_foot": (
                    selected_side == int(effector[first]) - 1 if first is not None else None
                ),
                "precontact_foot_velocity_world_mps": (
                    foot_velocity[first, 0, selected_side].tolist()
                    if first is not None and selected_side is not None and selected_side >= 0
                    else None
                ),
                "precontact_ball_velocity_world_mps": (
                    ball_velocity[first - 1, :3].tolist() if first is not None else None
                ),
                "postcontact_ball_velocity_world_mps": (
                    ball_velocity[first, :3].tolist() if first is not None else None
                ),
                "precontact_foot_ball_center_gap_m": (
                    float(np.linalg.norm(feet[first, 0, selected_side] - ball[first, 0]))
                    if first is not None and selected_side is not None and selected_side >= 0
                    else None
                ),
                "first_contact_peak_force_n": float(force[first]) if first is not None else None,
                "first_contact_position_world_m": (
                    contact_position[first, 0].tolist() if first is not None else None
                ),
                "first_contact_normal_ball_to_foot_world": (
                    contact_normal[first, 0].tolist() if first is not None else None
                ),
                "first_contact_foot_minus_ball_velocity_world_mps": (
                    contact_relative_velocity[first, 0].tolist() if first is not None else None
                ),
                "first_contact_signed_closing_speed_mps": (
                    float(-np.dot(contact_relative_velocity[first, 0], contact_normal[first, 0]))
                    if first is not None
                    else None
                ),
            }
        )
    if source_hashes != {str(p.relative_to(root)): hash_bytes(p.read_bytes()) for p in sources}:
        raise ValueError("source changed during physical velocity diagnosis")
    report = {
        "schema": "rsi_team_contact_velocity_diagnostic_report_v20",
        "activation_ceiling": "SIM_ONLY",
        "protocol_hash": hash_bytes(args.protocol.read_bytes()),
        "source_hashes": source_hashes,
        "asset_body_hash": qualification.body_hash,
        "rows": rows,
        "all_world_trajectories_match_v19": all(r["trajectory_digest_matches_v19"] for r in rows),
        "promotion_authorized": False,
    }
    report["report_hash"] = hash_json(report)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("RSI_TEAM_CONTACT_VELOCITY_DIAGNOSTIC=" + json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
