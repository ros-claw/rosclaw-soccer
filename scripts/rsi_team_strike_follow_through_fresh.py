"""Three-way new-scene physical exam for a bounded learned first-touch strike."""

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
from scripts.rsi_team_foot_velocity_fresh_exam import score_parent
from scripts.rsi_team_intercept_navigation_search import _score
from scripts.rsi_team_taskspace_first_touch import TeamSwingMotor, _run_one


def _courses(protocol: dict[str, Any]) -> list[IndependentTeamWorldScenario]:
    rng = np.random.default_rng(protocol["coordinate_seed"])
    return [
        IndependentTeamWorldScenario(
            f"s199.rsi-strike-through-fresh-f{index:03d}",
            (
                round(float(rng.uniform(*protocol["ball_x_m"])), 4),
                round(float(rng.uniform(*protocol["ball_y_m"])), 4),
                0.115,
            ),
            (round(float(rng.uniform(*protocol["ball_vx_mps"])), 4), 0.0, 0.0),
            protocol["identity_seed_base"] + index,
        )
        for index in range(protocol["scene_count"])
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("fresh three-way exam output already exists")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if (
        protocol.get("schema") != "rsi_team_strike_follow_through_fresh_protocol_v36"
        or protocol.get("development_only") is not True
        or protocol.get("promotion_authorized") is not False
        or protocol.get("frames") != 250
        or protocol.get("scene_count") != 24
        or protocol.get("baseline_strike_through_m") != 0.0
        or protocol.get("candidate_strike_through_m") != 0.16
        or protocol.get("gate")
        != {
            "minimum_scenes": 24,
            "maximum_unsafe_excess_vs_zero": 0,
            "minimum_useful_gain_vs_zero": 3,
            "minimum_candidate_useful": 5,
        }
    ):
        raise ValueError("invalid frozen three-way fresh-exam protocol")
    search = json.loads(Path(protocol["search_report_path"]).read_text(encoding="utf-8"))
    if (
        search.get("report_hash") != protocol["search_report_hash"]
        or hash_json({key: value for key, value in search.items() if key != "report_hash"})
        != protocol["search_report_hash"]
        or search.get("selected_variant") != "through_16cm"
    ):
        raise ValueError("unsealed physical strike selection")
    model_path = Path(protocol["navigation_model_path"])
    model = json.loads(model_path.read_text(encoding="utf-8"))
    if (
        model.get("model_hash") != protocol["navigation_model_hash"]
        or hash_json({key: value for key, value in model.items() if key != "model_hash"})
        != protocol["navigation_model_hash"]
    ):
        raise ValueError("unsealed navigation model")
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
        root / "src/rosclaw_soccer/rsi/taskspace_swing_probe.py",
        root / "src/rosclaw_soccer/skills/team/independent_team_world.py",
    )
    source_hashes = {
        str(path.relative_to(root)): hash_bytes(path.read_bytes()) for path in source_paths
    }
    courses = _courses(protocol)
    if len({(s.ball_initial_position_m, s.ball_initial_velocity_mps) for s in courses}) != 24:
        raise ValueError("new physical states not distinct")
    args.output_dir.mkdir(parents=True)
    rows = []
    for index, course in enumerate(courses):
        folder = args.output_dir / f"f{index:03d}"
        try:
            parent_report = _run_one(
                mode="parent",
                asset_root=args.asset_root,
                output_dir=folder,
                fixture=fixture,
                scenario=course,
                protocol=protocol,
            )
        except ValueError as error:
            if str(error) != "incomplete paired motor episode":
                raise
            # No authenticated pre-decision Parent state exists for this pair.
            # Never infer a candidate win from a scene without causal pairing.
            failed = {
                "status": "PAIR_INCOMPLETE",
                "safe": False,
                "foot_contact_frames": [],
                "useful_pass": False,
                "outgoing_ball_vx_mps": None,
                "report_hash": None,
            }
            rows.append(
                {
                    "scene": f"f{index:03d}",
                    "scenario_hash": course.scenario_hash,
                    "entry_hash": None,
                    "selected_arm": None,
                    "parent": failed,
                    "zero": failed,
                    "through": failed,
                }
            )
            print(f"f{index:03d} PAIR_INCOMPLETE all variants fail-closed", flush=True)
            continue
        parent = score_parent(parent_report, folder / "parent/trajectory.npz")
        parent_entry = entry_features(folder / "parent/taskspace_trace.npz", 30)
        variants: dict[str, dict[str, Any]] = {}
        selections: dict[str, str | None] = {}
        for name, through in (
            ("zero", protocol["baseline_strike_through_m"]),
            ("through", protocol["candidate_strike_through_m"]),
        ):
            navigation = TeamFootVelocityChooser(
                agent_id=protocol["focal_agent_id"],
                foundation_hash=policy_hash,
                foundation_config_hash=config_hash,
                model_path=model_path,
            )
            action = dict(protocol["candidate_action"])
            action["strike_through_m"] = through
            motor = TeamSwingMotor(
                protocol["focal_agent_id"], True, action, activation_selector=navigation
            )
            episode_protocol = dict(protocol)
            episode_protocol["candidate_action"] = action
            out = folder / name
            try:
                episode = _run_one(
                    mode="candidate",
                    asset_root=args.asset_root,
                    output_dir=out,
                    fixture=fixture,
                    scenario=course,
                    protocol=episode_protocol,
                    navigation_policy=navigation,
                    motor_option=motor,
                )
                score = _score(episode, out / "candidate/trajectory.npz")
                entry = entry_features(out / "candidate/taskspace_trace.npz", 30)
                if entry["hash"] != parent_entry["hash"]:
                    raise ValueError("changed pre-decision physical state")
                status = "COMPLETE"
            except ValueError as error:
                if str(error) not in {
                    "incomplete paired motor episode",
                    "task-space side used future contact or altered support leg",
                }:
                    raise
                score = {
                    "safe": False,
                    "foot_contact_frames": [],
                    "useful_pass": False,
                    "outgoing_ball_vx_mps": None,
                    "report_hash": None,
                }
                status = (
                    "INCOMPLETE"
                    if str(error) == "incomplete paired motor episode"
                    else "AUDIT_REJECTED"
                )
            variants[name] = {"status": status, **score}
            selections[name] = navigation.selected_arm
        if selections["zero"] != selections["through"]:
            raise ValueError("motor variant changed frozen pre-decision arm choice")
        rows.append(
            {
                "scene": f"f{index:03d}",
                "scenario_hash": course.scenario_hash,
                "entry_hash": parent_entry["hash"],
                "selected_arm": selections["zero"],
                "parent": parent,
                **variants,
            }
        )
        print(
            f"f{index:03d} arm={selections['zero']} "
            f"zero={variants['zero']['safe']}/{variants['zero']['useful_pass']} "
            f"through={variants['through']['safe']}/{variants['through']['useful_pass']}",
            flush=True,
        )
    if source_hashes != {
        str(path.relative_to(root)): hash_bytes(path.read_bytes()) for path in source_paths
    }:
        raise ValueError("physical source changed during fresh strike exam")
    totals = {
        name: {
            "unsafe": sum(not row[name]["safe"] for row in rows),
            "safe_contact": sum(
                bool(row[name]["safe"] and row[name]["foot_contact_frames"]) for row in rows
            ),
            "useful": sum(row[name]["useful_pass"] for row in rows),
        }
        for name in ("parent", "zero", "through")
    }
    gate = bool(
        len(rows) >= protocol["gate"]["minimum_scenes"]
        and totals["through"]["unsafe"] - totals["zero"]["unsafe"]
        <= protocol["gate"]["maximum_unsafe_excess_vs_zero"]
        and totals["through"]["useful"] - totals["zero"]["useful"]
        >= protocol["gate"]["minimum_useful_gain_vs_zero"]
        and totals["through"]["useful"] >= protocol["gate"]["minimum_candidate_useful"]
    )
    result = {
        "schema": "rsi_team_strike_follow_through_fresh_report_v36",
        "activation_ceiling": "SIM_ONLY",
        "fresh_physical_coordinates": True,
        "protocol_hash": hash_bytes(args.protocol.read_bytes()),
        "navigation_model_hash": protocol["navigation_model_hash"],
        "search_report_hash": protocol["search_report_hash"],
        "body_hash": qualification.body_hash,
        "physical_source_hashes": source_hashes,
        "rows": rows,
        "totals": totals,
        "gate_passed": gate,
        "promotion_authorized": False,
    }
    result["report_hash"] = hash_json(result)
    (args.output_dir / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "RSI_TEAM_STRIKE_FRESH="
        + json.dumps(
            {key: result[key] for key in ("report_hash", "totals", "gate_passed")},
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
