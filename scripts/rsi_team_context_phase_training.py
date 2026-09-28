"""Test one state-conditioned receiving actor on consumed 3v3 courses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.rsi.team_context_phase_navigation import TeamContextPhaseNavigation
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.independent_team_world import IndependentTeamWorldScenario
from rosclaw_soccer.training.independent_team_growth import build_independent_three_vs_three_fixture
from scripts.rsi_team_intercept_navigation_search import _score
from scripts.rsi_team_taskspace_first_touch import _run_one, world_contact_code


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("output directory already exists")
    protocol: dict[str, Any] = json.loads(args.protocol.read_text(encoding="utf-8"))
    courses = protocol.get("training_courses", [])
    if (
        protocol.get("schema") != "rsi_team_context_phase_training_protocol_v17"
        or protocol.get("development_only") is not True
        or protocol.get("promotion_authorized") is not False
        or protocol.get("frames") != 250
        or [course.get("name") for course in courses] != ["center", "near", "far", "left", "right"]
        or protocol.get("candidate_action")
        != {
            "entry_frame": 30,
            "forward_cap_m": 0.16,
            "lateral_cap_m": 0.05,
            "vertical_offset_m": 0.04,
            "swing_foot_acquisition_gap_m": 0.55,
        }
    ):
        raise ValueError("uncommitted SIM_ONLY contextual training protocol")
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
        root / "src/rosclaw_soccer/rsi/team_context_phase_navigation.py",
        root / "src/rosclaw_soccer/skills/team/independent_team_world.py",
    )
    source_hashes = {str(p.relative_to(root)): hash_bytes(p.read_bytes()) for p in sources}
    args.output_dir.mkdir(parents=True)
    rows = []
    focal_code = world_contact_code(
        protocol["focal_agent_id"], tuple(player.agent_id for player in fixture.players)
    )
    for course in courses:
        scenario = IndependentTeamWorldScenario(
            course["scenario_id"],
            tuple(course["ball_initial_position_m"]),
            tuple(course["ball_initial_velocity_mps"]),
            course["seed"],
        )
        nav_policy = TeamContextPhaseNavigation(
            protocol["focal_agent_id"], policy_hash, config_hash
        )
        folder = args.output_dir / course["name"]
        candidate = _run_one(
            mode="candidate",
            asset_root=args.asset_root,
            output_dir=folder,
            fixture=fixture,
            scenario=scenario,
            protocol=protocol,
            navigation_policy=nav_policy,
        )
        physics_path = folder / "candidate/trajectory.npz"
        outcome = _score(candidate, physics_path)
        with np.load(physics_path, allow_pickle=False) as physics:
            proposed = np.asarray(physics["navigation_proposed_delta"])
        expected = np.asarray([(dx, dy) for _, _, _, dx, dy in nav_policy.history])
        audit = bool(
            proposed.shape == (250, 6, 3)
            and expected.shape == (250, 2)
            and np.allclose(proposed[:, focal_code - 1, :2], expected, atol=1e-6, rtol=0)
            and np.allclose(proposed[:, focal_code - 1, 2], 0.0, atol=1e-6, rtol=0)
        )
        outcome.update(
            {
                "course": course["name"],
                "navigation_contract_hash": nav_policy.contract_hash,
                "entry_foot_gap_m": nav_policy.entry_foot_gap_m,
                "selected_lateral_gain": nav_policy.selected_lateral_gain,
                "navigation_trace_audited": audit,
                "safe": bool(outcome["safe"] and audit),
            }
        )
        outcome["useful_pass"] = bool(outcome["useful_pass"] and audit)
        rows.append(outcome)
    if source_hashes != {str(p.relative_to(root)): hash_bytes(p.read_bytes()) for p in sources}:
        raise ValueError("source changed during contextual training")
    safe_contacts = sum(bool(r["safe"] and r["foot_contact_frames"]) for r in rows)
    useful = sum(bool(r["useful_pass"]) for r in rows)
    report = {
        "schema": "rsi_team_context_phase_training_report_v17",
        "activation_ceiling": "SIM_ONLY",
        "protocol_hash": hash_bytes(args.protocol.read_bytes()),
        "source_hashes": source_hashes,
        "asset_body_hash": qualification.body_hash,
        "courses": rows,
        "safe_foot_contact_count": safe_contacts,
        "useful_pass_count": useful,
        "fresh_holdout_open_authorized": bool(
            len(rows) == 5 and all(r["safe"] for r in rows) and safe_contacts >= 4 and useful >= 2
        ),
        "promotion_authorized": False,
    }
    report["report_hash"] = hash_json(report)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("RSI_TEAM_CONTEXT_PHASE_TRAINING=" + json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
