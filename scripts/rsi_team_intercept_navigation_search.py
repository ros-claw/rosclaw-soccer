"""Train a bounded receiving-stance adapter on consumed 3v3 MuJoCo courses."""

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
from scripts.rsi_team_taskspace_first_touch import _run_one, world_contact_code


def _score(report: dict[str, Any], trajectory_path: Path) -> dict[str, Any]:
    contacts = report["focal_foot_contact_frames"]
    with np.load(trajectory_path, allow_pickle=False) as physics:
        velocity = np.asarray(physics["ball_velocity"])
        nav = np.asarray(physics["navigation_proposed_delta"])
        nav_fault = np.asarray(physics["navigation_faulted"])
    code = world_contact_code(
        "red.playmaker",
        tuple(
            sorted(
                (
                    "red.goalkeeper",
                    "red.playmaker",
                    "red.finisher",
                    "blue.goalkeeper",
                    "blue.playmaker",
                    "blue.finisher",
                )
            )
        ),
    )
    first = contacts[0] if contacts else None
    outgoing_vx = (
        float(np.max(velocity[first : min(first + 5, len(velocity)), 0]))
        if first is not None
        else None
    )
    safe = bool(
        report["world_result"]["safe"]
        and not report["world_result"]["motor_fault_agents"]
        and not report["world_result"].get("navigation_fault_agents", [])
        and not report["focal_nonfoot_contact_frames"]
        and report["action_audit"]["taskspace_action_audited"]
        and report["motor_first_contact_frame"] == first
        and bool(report["motor_observed_own_foot_contact_frames"]) == bool(contacts)
        and not bool(np.any(nav_fault))
    )
    return {
        "report_hash": report["report_hash"],
        "foot_contact_frames": contacts,
        "nonfoot_contact_frames": report["focal_nonfoot_contact_frames"],
        "outgoing_ball_vx_mps": outgoing_vx,
        "navigation_proposed_active_frames": int(
            np.sum(np.linalg.norm(nav[:, code - 1, :2], axis=1) > 1e-6)
        ),
        "safe": safe,
        "useful_pass": bool(safe and outgoing_vx is not None and outgoing_vx >= 0.5),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("output directory already exists")
    protocol: dict[str, Any] = json.loads(args.protocol.read_text(encoding="utf-8"))
    params = protocol.get("parameter_family", [])
    courses = protocol.get("training_courses", [])
    if (
        protocol.get("schema") != "rsi_team_intercept_navigation_search_protocol_v15"
        or protocol.get("development_only") is not True
        or protocol.get("promotion_authorized") is not False
        or protocol.get("frames") != 250
        or [(p.get("name"), p.get("forward_gain"), p.get("lateral_gain")) for p in params]
        != [
            ("lat04", 0.0, 0.4),
            ("lat08", 0.0, 0.8),
            ("lat12", 0.0, 1.2),
            ("front04_lat08", 0.4, 0.8),
            ("back04_lat08", -0.4, 0.8),
            ("negative_lat08", 0.0, -0.8),
            ("front04", 0.4, 0.0),
            ("back04", -0.4, 0.0),
        ]
        or [p.get("name") for p in courses] != ["near", "far", "left", "right"]
    ):
        raise ValueError("uncommitted SIM_ONLY navigation search protocol")
    qualification = qualify_g1_assets(args.asset_root)
    qualification.require_eligible()
    fixture = build_independent_three_vs_three_fixture(args.asset_root)
    policy_path = args.asset_root / "policy/loco_mode/model/policy_29dof.pt"
    config_path = args.asset_root / "policy/loco_mode/config/LocoMode.yaml"
    policy_hash = hash_bytes(policy_path.read_bytes())
    config_hash = hash_bytes(config_path.read_bytes())
    root = Path(__file__).parents[1]
    sources = (
        Path(__file__),
        Path(__file__).with_name("rsi_team_taskspace_first_touch.py"),
        root / "src/rosclaw_soccer/rsi/team_intercept_navigation.py",
        root / "src/rosclaw_soccer/skills/team/independent_team_world.py",
    )
    source_hashes = {str(p.relative_to(root)): hash_bytes(p.read_bytes()) for p in sources}
    args.output_dir.mkdir(parents=True)
    rows = []
    for param in params:
        outcomes = []
        for course in courses:
            scenario = IndependentTeamWorldScenario(
                course["scenario_id"],
                tuple(course["ball_initial_position_m"]),
                tuple(course["ball_initial_velocity_mps"]),
                course["seed"],
            )
            nav_policy = TeamInterceptNavigation(
                agent_id=protocol["focal_agent_id"],
                foundation_hash=policy_hash,
                foundation_config_hash=config_hash,
                forward_gain=float(param["forward_gain"]),
                lateral_gain=float(param["lateral_gain"]),
            )
            folder = args.output_dir / param["name"] / course["name"]
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
                "name": param["name"],
                "parameters": param,
                "outcomes": outcomes,
                "safe_foot_contact_count": sum(
                    bool(r["safe"] and r["foot_contact_frames"]) for r in outcomes
                ),
                "useful_pass_count": sum(bool(r["useful_pass"]) for r in outcomes),
                "all_safe": all(bool(r["safe"]) for r in outcomes),
            }
        )
    if source_hashes != {str(p.relative_to(root)): hash_bytes(p.read_bytes()) for p in sources}:
        raise ValueError("source changed during navigation search")
    eligible = [r for r in rows if r["all_safe"]]
    winner = max(
        eligible,
        key=lambda r: (
            r["safe_foot_contact_count"],
            r["useful_pass_count"],
            -abs(r["parameters"]["forward_gain"]) - abs(r["parameters"]["lateral_gain"]),
        ),
        default=None,
    )
    report = {
        "schema": "rsi_team_intercept_navigation_search_report_v15",
        "activation_ceiling": "SIM_ONLY",
        "protocol_hash": hash_bytes(args.protocol.read_bytes()),
        "source_hashes": source_hashes,
        "asset_body_hash": qualification.body_hash,
        "rows": rows,
        "selected_training_candidate": winner["name"]
        if winner and winner["safe_foot_contact_count"] > 0
        else None,
        "fresh_holdout_open_authorized": bool(
            winner and winner["safe_foot_contact_count"] >= 3 and winner["useful_pass_count"] >= 2
        ),
        "promotion_authorized": False,
    }
    report["report_hash"] = hash_json(report)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("RSI_TEAM_INTERCEPT_NAVIGATION_SEARCH=" + json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
