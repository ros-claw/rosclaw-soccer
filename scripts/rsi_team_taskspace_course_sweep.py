"""SIM_ONLY paired six-course robustness check of a frozen shared-world foot action."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

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
    protocol: dict[str, Any] = json.loads(args.protocol.read_text(encoding="utf-8"))
    courses = protocol.get("courses", [])
    action = protocol.get("candidate_action", {})
    if (
        protocol.get("schema") != "rsi_team_taskspace_course_sweep_protocol_v14"
        or protocol.get("development_only") is not True
        or protocol.get("promotion_authorized") is not False
        or protocol.get("frames") != 250
        or [course.get("name") for course in courses]
        != ["near", "far", "left", "right", "slow", "fast"]
        or action
        != {
            "entry_frame": 30,
            "forward_cap_m": 0.16,
            "lateral_cap_m": 0.05,
            "vertical_offset_m": 0.04,
            "swing_foot_acquisition_gap_m": 0.55,
        }
    ):
        raise ValueError("uncommitted SIM_ONLY six-course protocol")
    qualification = qualify_g1_assets(args.asset_root)
    qualification.require_eligible()
    fixture = build_independent_three_vs_three_fixture(args.asset_root)
    root = Path(__file__).parents[1]
    sources = (
        Path(__file__),
        Path(__file__).with_name("rsi_team_taskspace_first_touch.py"),
        root / "src/rosclaw_soccer/rsi/taskspace_swing_probe.py",
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
        folder = args.output_dir / course["name"]
        parent = _run_one(
            mode="parent",
            asset_root=args.asset_root,
            output_dir=folder,
            fixture=fixture,
            scenario=scenario,
            protocol=protocol,
        )
        candidate = _run_one(
            mode="candidate",
            asset_root=args.asset_root,
            output_dir=folder,
            fixture=fixture,
            scenario=scenario,
            protocol=protocol,
        )
        with np.load(folder / "candidate/trajectory.npz", allow_pickle=False) as trace:
            ball_velocity = np.asarray(trace["ball_velocity"])
        contact = candidate["focal_foot_contact_frames"]
        first = contact[0] if contact else None
        outgoing_vx = (
            float(np.max(ball_velocity[first : min(first + 5, len(ball_velocity)), 0]))
            if first is not None
            else None
        )
        contact_agrees = candidate["motor_first_contact_frame"] == first and bool(
            candidate["motor_observed_own_foot_contact_frames"]
        ) == bool(contact)
        safe = bool(
            parent["world_result"]["safe"]
            and candidate["world_result"]["safe"]
            and not parent["world_result"]["motor_fault_agents"]
            and not candidate["world_result"]["motor_fault_agents"]
            and not candidate["focal_nonfoot_contact_frames"]
            and candidate["action_audit"]["taskspace_action_audited"]
            and contact_agrees
        )
        rows.append(
            {
                "course": course["name"],
                "scenario_hash": scenario.scenario_hash,
                "parent_report_hash": parent["report_hash"],
                "candidate_report_hash": candidate["report_hash"],
                "parent_foot_contact_frames": parent["focal_foot_contact_frames"],
                "candidate_foot_contact_frames": contact,
                "candidate_nonfoot_contact_frames": candidate["focal_nonfoot_contact_frames"],
                "outgoing_ball_vx_mps": outgoing_vx,
                "contact_observers_agree": contact_agrees,
                "safe": safe,
                "useful_pass": bool(safe and outgoing_vx is not None and outgoing_vx >= 0.5),
            }
        )
    if source_hashes != {str(p.relative_to(root)): hash_bytes(p.read_bytes()) for p in sources}:
        raise ValueError("source changed during six-course physics execution")
    report = {
        "schema": "rsi_team_taskspace_course_sweep_report_v14",
        "activation_ceiling": "SIM_ONLY",
        "protocol_hash": hash_bytes(args.protocol.read_bytes()),
        "source_hashes": source_hashes,
        "asset_body_hash": qualification.body_hash,
        "courses": rows,
        "safe_contact_count": sum(
            bool(row["safe"] and row["candidate_foot_contact_frames"]) for row in rows
        ),
        "useful_pass_count": sum(bool(row["useful_pass"]) for row in rows),
        "promotion_authorized": False,
    }
    report["report_hash"] = hash_json(report)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("RSI_TEAM_TASKSPACE_COURSE_SWEEP=" + json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
