"""One-shot fresh paired 3v3 MuJoCo exam for a frozen contextual navigation memory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.rsi.team_contextual_nav_policy import TeamContextualNavigationMemory
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.independent_team_world import IndependentTeamWorldScenario
from rosclaw_soccer.training.independent_team_growth import build_independent_three_vs_three_fixture
from scripts.rsi_team_contextual_action_dataset import entry_features
from scripts.rsi_team_intercept_navigation_search import _score
from scripts.rsi_team_taskspace_first_touch import TeamSwingMotor, _run_one


def fresh_scenarios(protocol: dict[str, Any]) -> list[IndependentTeamWorldScenario]:
    grid = protocol["fresh_grid"]
    scenarios = []
    for x in grid["x_m"]:
        for y in grid["y_m"]:
            for vx in grid["incoming_vx_mps"]:
                index = len(scenarios)
                scenarios.append(
                    IndependentTeamWorldScenario(
                        f"s199.rsi-context-nav-fresh-f{index:02d}",
                        (x, y, 0.115),
                        (vx, 0.0, 0.0),
                        grid["seed_base"] + index,
                    )
                )
    return scenarios


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("fresh exam output already exists")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    model_path = Path(protocol["model_path"])
    if (
        protocol.get("schema") != "rsi_team_contextual_nav_fresh_exam_protocol_v25"
        or protocol.get("activation_ceiling") != "SIM_ONLY"
        or protocol.get("promotion_authorized") is not False
        or protocol.get("frames") != 250
        or len(fresh_scenarios(protocol)) != 12
        or protocol["candidate_action"]
        != {
            "entry_frame": 30,
            "forward_cap_m": 0.16,
            "lateral_cap_m": 0.05,
            "vertical_offset_m": 0.04,
            "swing_foot_acquisition_gap_m": 0.55,
        }
    ):
        raise ValueError("invalid frozen SIM_ONLY fresh protocol")
    model = json.loads(model_path.read_text(encoding="utf-8"))
    if (
        model.get("model_hash") != protocol["model_hash"]
        or hash_json({key: value for key, value in model.items() if key != "model_hash"})
        != protocol["model_hash"]
    ):
        raise ValueError("frozen model commitment mismatch")
    qualification = qualify_g1_assets(args.asset_root)
    qualification.require_eligible()
    if model["physical_body_hash"] != qualification.body_hash:
        raise ValueError("model trained on different qualified body")
    fixture = build_independent_three_vs_three_fixture(args.asset_root)
    policy_hash = hash_bytes(
        (args.asset_root / "policy/loco_mode/model/policy_29dof.pt").read_bytes()
    )
    config_hash = hash_bytes(
        (args.asset_root / "policy/loco_mode/config/LocoMode.yaml").read_bytes()
    )
    root = Path(__file__).parents[1]
    source_paths = (
        Path(__file__),
        Path(__file__).with_name("rsi_team_taskspace_first_touch.py"),
        Path(__file__).with_name("rsi_team_intercept_navigation_search.py"),
        root / "src/rosclaw_soccer/rsi/team_contextual_nav_policy.py",
        root / "src/rosclaw_soccer/rsi/team_intercept_navigation.py",
        root / "src/rosclaw_soccer/rsi/team_phase_intercept_navigation.py",
        root / "src/rosclaw_soccer/skills/team/independent_team_world.py",
        root / "src/rosclaw_soccer/skills/team/foot_kinematics.py",
    )
    source_hashes = {
        str(path.relative_to(root)): hash_bytes(path.read_bytes()) for path in source_paths
    }
    for name, expected_hash in model["model_source_hashes"].items():
        if hash_bytes((root / name).read_bytes()) != expected_hash:
            raise ValueError("model implementation changed since training")
    args.output_dir.mkdir(parents=True)
    episodes = []
    for index, scenario in enumerate(fresh_scenarios(protocol)):
        folder = args.output_dir / f"f{index:02d}"
        parent = _run_one(
            mode="parent",
            asset_root=args.asset_root,
            output_dir=folder,
            fixture=fixture,
            scenario=scenario,
            protocol=protocol,
        )
        memory = TeamContextualNavigationMemory(
            agent_id=protocol["focal_agent_id"],
            foundation_hash=policy_hash,
            foundation_config_hash=config_hash,
            model_path=model_path,
        )
        motor = TeamSwingMotor(
            protocol["focal_agent_id"],
            True,
            protocol["candidate_action"],
            activation_selector=memory,
        )
        candidate = _run_one(
            mode="candidate",
            asset_root=args.asset_root,
            output_dir=folder,
            fixture=fixture,
            scenario=scenario,
            protocol=protocol,
            navigation_policy=memory,
            motor_option=motor,
        )
        parent_entry = entry_features(folder / "parent/taskspace_trace.npz", 30)
        candidate_entry = entry_features(folder / "candidate/taskspace_trace.npz", 30)
        if parent_entry["hash"] != candidate_entry["hash"]:
            raise ValueError("fresh pair diverged before decision")
        outcome = _score(candidate, folder / "candidate/trajectory.npz")
        if (
            memory.selected_arm is None
            and parent["trajectory_digest"] != candidate["trajectory_digest"]
        ):
            raise ValueError("abstention changed physical parent world")
        with np.load(folder / "parent/trajectory.npz", allow_pickle=False) as physics:
            velocity = np.asarray(physics["ball_velocity"])
        parent_contacts = parent["focal_foot_contact_frames"]
        parent_first = parent_contacts[0] if parent_contacts else None
        parent_outgoing = (
            float(np.max(velocity[parent_first : min(parent_first + 5, len(velocity)), 0]))
            if parent_first is not None
            else None
        )
        parent_safe = bool(
            parent["world_result"]["safe"] and not parent["focal_nonfoot_contact_frames"]
        )
        entry = {
            "scenario_hash": scenario.scenario_hash,
            "entry_hash": parent_entry["hash"],
            "selected_arm": memory.selected_arm,
            "neighbor_indices": memory.neighbor_indices,
            "parent_report_hash": parent["report_hash"],
            "candidate_report_hash": candidate["report_hash"],
            "parent_safe": parent_safe,
            "parent_useful_pass": bool(
                parent_safe and parent_outgoing is not None and parent_outgoing >= 0.5
            ),
            "candidate_safe": outcome["safe"],
            "candidate_foot_contact": bool(outcome["foot_contact_frames"]),
            "candidate_useful_pass": outcome["useful_pass"],
            "candidate_outgoing_ball_vx_mps": outcome["outgoing_ball_vx_mps"],
        }
        episodes.append(entry)
        print(
            f"f{index:02d} selected={memory.selected_arm} "
            f"safe={outcome['safe']} useful={outcome['useful_pass']}",
            flush=True,
        )
    if source_hashes != {
        str(path.relative_to(root)): hash_bytes(path.read_bytes()) for path in source_paths
    }:
        raise ValueError("physics source changed during fresh exam")
    safe_count = sum(row["candidate_safe"] for row in episodes)
    useful_count = sum(row["candidate_useful_pass"] for row in episodes)
    parent_useful = sum(row["parent_useful_pass"] for row in episodes)
    gate_passed = bool(
        safe_count >= protocol["gate"]["minimum_safe"]
        and useful_count >= protocol["gate"]["minimum_useful_passes"]
        and useful_count - parent_useful >= protocol["gate"]["minimum_useful_gain"]
    )
    report = {
        "schema": "rsi_team_contextual_nav_fresh_exam_report_v25",
        "activation_ceiling": "SIM_ONLY",
        "protocol_hash": hash_bytes(args.protocol.read_bytes()),
        "model_hash": protocol["model_hash"],
        "asset_body_hash": qualification.body_hash,
        "source_hashes": source_hashes,
        "episodes": episodes,
        "parent_useful_pass_count": parent_useful,
        "candidate_safe_count": safe_count,
        "candidate_foot_contact_count": sum(
            row["candidate_safe"] and row["candidate_foot_contact"] for row in episodes
        ),
        "candidate_useful_pass_count": useful_count,
        "abstention_count": sum(row["selected_arm"] is None for row in episodes),
        "gate_passed": gate_passed,
        "promotion_authorized": False,
    }
    report["report_hash"] = hash_json(report)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "RSI_TEAM_CONTEXTUAL_FRESH="
        + json.dumps(
            {
                key: report[key]
                for key in (
                    "report_hash",
                    "candidate_safe_count",
                    "candidate_foot_contact_count",
                    "candidate_useful_pass_count",
                    "parent_useful_pass_count",
                    "gate_passed",
                )
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
