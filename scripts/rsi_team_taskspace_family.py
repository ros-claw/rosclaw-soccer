"""Predeclared bounded action-family diagnosis on a consumed shared-world course."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.independent_team_world import IndependentTeamWorldScenario
from rosclaw_soccer.training.independent_team_growth import build_independent_three_vs_three_fixture
from scripts.rsi_team_taskspace_first_touch import _run_one


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("output directory already exists")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    actions: list[dict[str, Any]] = protocol.get("candidate_actions", [])
    if (
        protocol.get("schema") != "rsi_team_taskspace_family_protocol_v13b"
        or protocol.get("promotion_authorized") is not False
        or [item.get("name") for item in actions] != ["lateral10", "forward16", "combined"]
        or [
            (item["forward_cap_m"], item["lateral_cap_m"], item["vertical_offset_m"])
            for item in actions
        ]
        != [(0.08, 0.1, 0.04), (0.16, 0.05, 0.04), (0.16, 0.1, 0.04)]
        or protocol.get("frames") != 250
    ):
        raise ValueError("uncommitted task-space family protocol")
    qualification = qualify_g1_assets(args.asset_root)
    qualification.require_eligible()
    fixture = build_independent_three_vs_three_fixture(args.asset_root)
    course = protocol["scenario"]
    scenario = IndependentTeamWorldScenario(
        course["scenario_id"],
        tuple(course["ball_initial_position_m"]),
        tuple(course["ball_initial_velocity_mps"]),
        course["seed"],
    )
    source_paths = (
        Path(__file__),
        Path(__file__).with_name("rsi_team_taskspace_first_touch.py"),
        Path(__file__).parents[1] / "src/rosclaw_soccer/rsi/taskspace_swing_probe.py",
        Path(__file__).parents[1] / "src/rosclaw_soccer/skills/team/independent_team_world.py",
    )
    root = Path(__file__).parents[1]
    source_hashes = {
        str(path.relative_to(root)): hash_bytes(path.read_bytes()) for path in source_paths
    }
    args.output_dir.mkdir(parents=True)
    parent = _run_one(
        mode="parent",
        asset_root=args.asset_root,
        output_dir=args.output_dir / "baseline",
        fixture=fixture,
        scenario=scenario,
        protocol={**protocol, "candidate_action": actions[0]},
    )
    results = []
    for action in actions:
        candidate = _run_one(
            mode="candidate",
            asset_root=args.asset_root,
            output_dir=args.output_dir / action["name"],
            fixture=fixture,
            scenario=scenario,
            protocol={**protocol, "candidate_action": action},
        )
        contact_agrees = candidate["motor_first_contact_frame"] == (
            candidate["focal_foot_contact_frames"][0]
            if candidate["focal_foot_contact_frames"]
            else None
        ) and bool(candidate["motor_observed_own_foot_contact_frames"]) == bool(
            candidate["focal_foot_contact_frames"]
        )
        safe = bool(
            candidate["world_result"]["safe"]
            and not candidate["world_result"]["motor_fault_agents"]
            and not candidate["focal_nonfoot_contact_frames"]
            and candidate["action_audit"]["taskspace_action_audited"]
            and contact_agrees
        )
        results.append(
            {
                "name": action["name"],
                "candidate_report_hash": candidate["report_hash"],
                "focal_foot_contact_frames": candidate["focal_foot_contact_frames"],
                "focal_nonfoot_contact_frames": candidate["focal_nonfoot_contact_frames"],
                "safe": safe,
                "contact_observers_agree": contact_agrees,
                "mechanism_contact_found": bool(safe and candidate["focal_foot_contact_frames"]),
                "action_audit": candidate["action_audit"],
            }
        )
    if source_hashes != {
        str(path.relative_to(root)): hash_bytes(path.read_bytes()) for path in source_paths
    }:
        raise ValueError("source changed during family physics execution")
    report = {
        "schema": "rsi_team_taskspace_family_report_v13b",
        "activation_ceiling": "SIM_ONLY",
        "protocol_hash": hash_bytes(args.protocol.read_bytes()),
        "source_hashes": source_hashes,
        "asset_body_hash": qualification.body_hash,
        "parent_report_hash": parent["report_hash"],
        "parent_focal_foot_contact_frames": parent["focal_foot_contact_frames"],
        "candidates": results,
        "any_safe_physical_foot_contact": any(row["mechanism_contact_found"] for row in results),
        "fresh_holdout_open_authorized": False,
        "promotion_authorized": False,
    }
    report["report_hash"] = hash_json(report)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("RSI_TEAM_TASKSPACE_FAMILY=" + json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
