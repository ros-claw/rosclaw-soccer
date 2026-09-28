"""Development-only physical action-library feasibility on consumed 3v3 scenes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.rsi.team_intercept_navigation import TeamInterceptNavigation
from rosclaw_soccer.rsi.team_phase_intercept_navigation import TeamPhaseInterceptNavigation
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.independent_team_world import IndependentTeamWorldScenario
from rosclaw_soccer.training.independent_team_growth import build_independent_three_vs_three_fixture
from scripts.rsi_team_intercept_navigation_search import _score
from scripts.rsi_team_taskspace_first_touch import _run_one


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("output directory already exists")
    protocol: dict[str, Any] = json.loads(args.protocol.read_text(encoding="utf-8"))
    courses = protocol.get("courses", [])
    arms = protocol.get("arms", [])
    if (
        protocol.get("schema") != "rsi_team_action_library_feasibility_protocol_v19"
        or protocol.get("development_only") is not True
        or protocol.get("promotion_authorized") is not False
        or protocol.get("consumed_failed_holdout_hash")
        != "sha256:b074859685b89dc7d1a4883c3f417c50d30136a58dfbc2909c78e7a02f88c332"
        or protocol.get("frames") != 250
        or protocol.get("candidate_action")
        != {
            "entry_frame": 30,
            "forward_cap_m": 0.16,
            "lateral_cap_m": 0.05,
            "vertical_offset_m": 0.04,
            "swing_foot_acquisition_gap_m": 0.55,
        }
        or [course.get("name") for course in courses] != [f"h{index:02d}" for index in range(1, 9)]
        or [
            (
                arm.get("name"),
                arm.get("foot_selection"),
                arm.get("forward_gain"),
                arm.get("lateral_gain"),
            )
            for arm in arms
        ]
        != [
            ("phase_front04", "phase", 0.4, 0.0),
            ("phase_negative_lat08", "phase", 0.0, -0.8),
            ("phase_front04_negative", "phase", 0.4, -0.8),
            ("phase_front04_positive", "phase", 0.4, 0.8),
            ("fixed_left_negative", "left", 0.0, -0.8),
        ]
    ):
        raise ValueError("uncommitted SIM_ONLY action-library feasibility protocol")
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
        root / "src/rosclaw_soccer/rsi/team_intercept_navigation.py",
        root / "src/rosclaw_soccer/rsi/team_phase_intercept_navigation.py",
        root / "src/rosclaw_soccer/skills/team/foot_kinematics.py",
        root / "src/rosclaw_soccer/skills/team/independent_team_world.py",
    )
    source_hashes = {str(p.relative_to(root)): hash_bytes(p.read_bytes()) for p in sources}
    args.output_dir.mkdir(parents=True)
    rows = []
    for arm in arms:
        outcomes = []
        for course in courses:
            scenario = IndependentTeamWorldScenario(
                course["scenario_id"],
                tuple(course["ball_initial_position_m"]),
                tuple(course["ball_initial_velocity_mps"]),
                course["seed"],
            )
            nav_type = (
                TeamPhaseInterceptNavigation
                if arm["foot_selection"] == "phase"
                else TeamInterceptNavigation
            )
            nav_policy = nav_type(
                agent_id=protocol["focal_agent_id"],
                foundation_hash=policy_hash,
                foundation_config_hash=config_hash,
                forward_gain=float(arm["forward_gain"]),
                lateral_gain=float(arm["lateral_gain"]),
            )
            folder = args.output_dir / arm["name"] / course["name"]
            candidate = _run_one(
                mode="candidate",
                asset_root=args.asset_root,
                output_dir=folder,
                fixture=fixture,
                scenario=scenario,
                protocol=protocol,
                navigation_policy=nav_policy,
            )
            outcome = _score(candidate, folder / "candidate/trajectory.npz")
            outcome["course"] = course["name"]
            outcome["navigation_contract_hash"] = nav_policy.contract_hash
            outcomes.append(outcome)
        rows.append(
            {
                "arm": arm,
                "outcomes": outcomes,
                "safe_contact_count": sum(
                    bool(r["safe"] and r["foot_contact_frames"]) for r in outcomes
                ),
                "useful_pass_count": sum(bool(r["useful_pass"]) for r in outcomes),
                "all_safe": all(bool(r["safe"]) for r in outcomes),
            }
        )
    if source_hashes != {str(p.relative_to(root)): hash_bytes(p.read_bytes()) for p in sources}:
        raise ValueError("source changed during action-library physics")
    per_course = []
    for course in courses:
        outcomes = [
            next(r for r in row["outcomes"] if r["course"] == course["name"]) for row in rows
        ]
        per_course.append(
            {
                "course": course["name"],
                "any_safe_contact": any(r["safe"] and r["foot_contact_frames"] for r in outcomes),
                "any_useful_pass": any(r["useful_pass"] for r in outcomes),
                "safe_arm_count": sum(bool(r["safe"]) for r in outcomes),
            }
        )
    report = {
        "schema": "rsi_team_action_library_feasibility_report_v19",
        "activation_ceiling": "SIM_ONLY",
        "protocol_hash": hash_bytes(args.protocol.read_bytes()),
        "source_hashes": source_hashes,
        "asset_body_hash": qualification.body_hash,
        "rows": rows,
        "per_course_oracle_diagnostic_only": per_course,
        "oracle_safe_contact_upper_bound": sum(bool(r["any_safe_contact"]) for r in per_course),
        "oracle_useful_pass_upper_bound": sum(bool(r["any_useful_pass"]) for r in per_course),
        "train_chooser_authorized": False,
        "promotion_authorized": False,
    }
    report["report_hash"] = hash_json(report)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("RSI_TEAM_ACTION_LIBRARY_FEASIBILITY=" + json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
