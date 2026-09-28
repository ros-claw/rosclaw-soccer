"""Run a frozen measured-foot chooser against paired Parent in new 3v3 physics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.rsi.team_foot_velocity_chooser import TeamFootVelocityChooser
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.independent_team_world import IndependentTeamWorldScenario
from rosclaw_soccer.training.independent_team_growth import build_independent_three_vs_three_fixture
from scripts.rsi_team_contextual_action_dataset import entry_features
from scripts.rsi_team_intercept_navigation_search import _score
from scripts.rsi_team_taskspace_first_touch import TeamSwingMotor, _run_one


def scenarios(protocol: dict[str, Any]) -> list[IndependentTeamWorldScenario]:
    rng = np.random.default_rng(protocol["coordinate_seed"])
    result = []
    for index in range(protocol["scene_count"]):
        x = round(float(rng.uniform(*protocol["ball_x_m"])), 4)
        y = round(float(rng.uniform(*protocol["ball_y_m"])), 4)
        vx = round(float(rng.uniform(*protocol["ball_vx_mps"])), 4)
        result.append(
            IndependentTeamWorldScenario(
                f"s199.rsi-foot-velocity-fresh-f{index:03d}",
                (x, y, 0.115),
                (vx, 0.0, 0.0),
                protocol["identity_seed_base"] + index,
            )
        )
    return result


def score_parent(report: dict[str, Any], trajectory_path: Path) -> dict[str, Any]:
    """Parent has no navigation trace field; score only observed body and ball."""
    contacts = report["focal_foot_contact_frames"]
    first = contacts[0] if contacts else None
    with np.load(trajectory_path, allow_pickle=False) as physics:
        ball_velocity = np.asarray(physics["ball_velocity"])
    outgoing = (
        float(np.max(ball_velocity[first : min(first + 5, len(ball_velocity)), 0]))
        if first is not None
        else None
    )
    safe = bool(
        report["world_result"]["safe"]
        and not report["world_result"]["motor_fault_agents"]
        and not report["focal_nonfoot_contact_frames"]
        and report["action_audit"]["taskspace_action_audited"]
    )
    return {
        "report_hash": report["report_hash"],
        "safe": safe,
        "foot_contact_frames": contacts,
        "useful_pass": bool(safe and outgoing is not None and outgoing >= 0.5),
        "outgoing_ball_vx_mps": outgoing,
        "navigation_proposed_active_frames": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("fresh exam output already exists")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if (
        protocol.get("schema") != "rsi_team_foot_velocity_fresh_exam_protocol_v33"
        or protocol.get("development_only") is not True
        or protocol.get("promotion_authorized") is not False
        or protocol.get("frames") != 250
        or protocol.get("scene_count") != 24
        or protocol.get("gate")
        != {
            "minimum_scenes": 24,
            "maximum_unsafe_excess_vs_paired_parent": 0,
            "minimum_useful_gain_vs_paired_parent": 4,
            "minimum_useful_total": 5,
        }
    ):
        raise ValueError("invalid frozen SIM_ONLY fresh-exam protocol")
    model_path = Path(protocol["model_path"])
    model = json.loads(model_path.read_text(encoding="utf-8"))
    if (
        model.get("model_hash") != protocol["model_hash"]
        or hash_json({key: value for key, value in model.items() if key != "model_hash"})
        != protocol["model_hash"]
    ):
        raise ValueError("frozen foot-velocity model hash mismatch")
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
    source_paths = (
        Path(__file__),
        Path(__file__).with_name("rsi_team_taskspace_first_touch.py"),
        root / "src/rosclaw_soccer/rsi/team_foot_velocity_chooser.py",
        root / "src/rosclaw_soccer/skills/team/navigation_option.py",
        root / "src/rosclaw_soccer/skills/team/independent_team_world.py",
        root / "src/rosclaw_soccer/skills/team/foot_kinematics.py",
    )
    source_hashes = {
        str(path.relative_to(root)): hash_bytes(path.read_bytes()) for path in source_paths
    }
    course = scenarios(protocol)
    physical_states = {
        (item.ball_initial_position_m, item.ball_initial_velocity_mps) for item in course
    }
    if len(physical_states) != 24:
        raise ValueError("new physical exam states are not distinct")
    args.output_dir.mkdir(parents=True)
    rows = []
    for index, scenario in enumerate(course):
        folder = args.output_dir / f"f{index:03d}"
        parent = _run_one(
            mode="parent",
            asset_root=args.asset_root,
            output_dir=folder,
            fixture=fixture,
            scenario=scenario,
            protocol=protocol,
        )
        parent_score = score_parent(parent, folder / "parent/trajectory.npz")
        parent_entry = entry_features(folder / "parent/taskspace_trace.npz", 30)
        navigation = TeamFootVelocityChooser(
            agent_id=protocol["focal_agent_id"],
            foundation_hash=policy_hash,
            foundation_config_hash=config_hash,
            model_path=model_path,
        )
        motor = TeamSwingMotor(
            protocol["focal_agent_id"],
            True,
            protocol["candidate_action"],
            activation_selector=navigation,
        )
        try:
            candidate = _run_one(
                mode="candidate",
                asset_root=args.asset_root,
                output_dir=folder,
                fixture=fixture,
                scenario=scenario,
                protocol=protocol,
                navigation_policy=navigation,
                motor_option=motor,
            )
            score = _score(candidate, folder / "candidate/trajectory.npz")
            candidate_entry = entry_features(folder / "candidate/taskspace_trace.npz", 30)
            if parent_entry["hash"] != candidate_entry["hash"]:
                raise ValueError("fresh paired intervention changed pre-decision physical state")
            status = "COMPLETE"
        except ValueError as error:
            if str(error) != "incomplete paired motor episode":
                raise
            score = {
                "safe": False,
                "foot_contact_frames": [],
                "useful_pass": False,
                "report_hash": None,
            }
            status = "INCOMPLETE"
        row = {
            "scene": f"f{index:03d}",
            "scenario_hash": scenario.scenario_hash,
            "parent_entry_hash": parent_entry["hash"],
            "candidate_status": status,
            "selected_arm": navigation.selected_arm,
            "parent": parent_score,
            "candidate": score,
        }
        rows.append(row)
        print(
            f"f{index:03d} parent_safe={parent_score['safe']} "
            f"candidate_safe={score['safe']} useful={score['useful_pass']} "
            f"arm={navigation.selected_arm} status={status}",
            flush=True,
        )
    if source_hashes != {
        str(path.relative_to(root)): hash_bytes(path.read_bytes()) for path in source_paths
    }:
        raise ValueError("physical source changed during fresh exam")
    parent_unsafe = sum(not row["parent"]["safe"] for row in rows)
    candidate_unsafe = sum(not row["candidate"]["safe"] for row in rows)
    parent_useful = sum(row["parent"]["useful_pass"] for row in rows)
    candidate_useful = sum(row["candidate"]["useful_pass"] for row in rows)
    passed = bool(
        len(rows) >= protocol["gate"]["minimum_scenes"]
        and candidate_unsafe - parent_unsafe
        <= protocol["gate"]["maximum_unsafe_excess_vs_paired_parent"]
        and candidate_useful - parent_useful
        >= protocol["gate"]["minimum_useful_gain_vs_paired_parent"]
        and candidate_useful >= protocol["gate"]["minimum_useful_total"]
    )
    result = {
        "schema": "rsi_team_foot_velocity_fresh_exam_report_v33",
        "activation_ceiling": "SIM_ONLY",
        "fresh_physical_coordinates": True,
        "protocol_hash": hash_bytes(args.protocol.read_bytes()),
        "model_hash": protocol["model_hash"],
        "body_hash": qualification.body_hash,
        "physical_source_hashes": source_hashes,
        "rows": rows,
        "parent_unsafe": parent_unsafe,
        "candidate_unsafe": candidate_unsafe,
        "parent_useful": parent_useful,
        "candidate_useful": candidate_useful,
        "candidate_safe_contact": sum(
            bool(row["candidate"]["safe"] and row["candidate"]["foot_contact_frames"])
            for row in rows
        ),
        "gate_passed": passed,
        "promotion_authorized": False,
    }
    result["report_hash"] = hash_json(result)
    (args.output_dir / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("RSI_TEAM_FOOT_VELOCITY_FRESH=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
