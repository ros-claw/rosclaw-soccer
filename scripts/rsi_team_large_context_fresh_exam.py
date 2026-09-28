"""Triangulate a frozen learned chooser against Parent and fixed G1 navigation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.rsi.team_foot_velocity_chooser import TeamFootVelocityChooser
from rosclaw_soccer.rsi.team_phase_intercept_navigation import TeamPhaseInterceptNavigation
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.independent_team_world import IndependentTeamWorldScenario
from rosclaw_soccer.training.independent_team_growth import build_independent_three_vs_three_fixture
from scripts.rsi_team_contextual_action_dataset import entry_features
from scripts.rsi_team_foot_velocity_fresh_exam import score_parent
from scripts.rsi_team_intercept_navigation_search import _score
from scripts.rsi_team_taskspace_first_touch import TeamSwingMotor, _run_one


def _scenes(protocol: dict[str, Any]) -> list[IndependentTeamWorldScenario]:
    rng = np.random.default_rng(protocol["coordinate_seed"])
    scenes = []
    for index in range(protocol["scene_count"]):
        x = round(float(rng.uniform(*protocol["ball_x_m"])), 4)
        y = round(float(rng.uniform(*protocol["ball_y_m"])), 4)
        vx = round(float(rng.uniform(*protocol["ball_vx_mps"])), 4)
        scenes.append(
            IndependentTeamWorldScenario(
                f"s199.rsi-large-chooser-fresh-f{index:03d}",
                (x, y, 0.115),
                (vx, 0.0, 0.0),
                protocol["identity_seed_base"] + index,
            )
        )
    return scenes


def _failed(status: str) -> dict[str, Any]:
    return {
        "status": status,
        "safe": False,
        "foot_contact_frames": [],
        "useful_pass": False,
        "outgoing_ball_vx_mps": None,
        "report_hash": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("fresh chooser exam output exists")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if (
        protocol.get("schema") != "rsi_team_large_context_fresh_exam_protocol_v40"
        or protocol.get("development_only") is not True
        or protocol.get("promotion_authorized") is not False
        or protocol.get("fixed_arm") != "phase_front04_lat12"
        or protocol.get("frames") != 250
        or protocol.get("scene_count") != 48
        or protocol.get("gate")
        != {
            "minimum_scenes": 48,
            "minimum_candidate_useful": 10,
            "minimum_useful_gain_vs_fixed": 3,
            "maximum_unsafe_excess_vs_fixed": 0,
            "maximum_unsafe_excess_vs_parent": 0,
        }
    ):
        raise ValueError("invalid frozen 48-scene physical exam protocol")
    model_path = Path(protocol["model_path"])
    model = json.loads(model_path.read_text(encoding="utf-8"))
    report = json.loads(Path(protocol["train_report_path"]).read_text(encoding="utf-8"))
    if (
        model.get("model_hash") != protocol["model_hash"]
        or hash_json({key: value for key, value in model.items() if key != "model_hash"})
        != protocol["model_hash"]
        or model.get("schema") != "rsi_team_large_context_chooser_model_v39"
        or report.get("report_hash") != protocol["train_report_hash"]
        or hash_json({key: value for key, value in report.items() if key != "report_hash"})
        != protocol["train_report_hash"]
        or report.get("fresh_exam_authorized") is not True
    ):
        raise ValueError("unsealed development model or failed prerequisite")
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
        root / "src/rosclaw_soccer/rsi/team_contextual_nav_policy.py",
        root / "src/rosclaw_soccer/rsi/team_phase_intercept_navigation.py",
        root / "src/rosclaw_soccer/skills/team/navigation_option.py",
        root / "src/rosclaw_soccer/skills/team/independent_team_world.py",
    )
    sources = {str(path.relative_to(root)): hash_bytes(path.read_bytes()) for path in source_paths}
    scenes = _scenes(protocol)
    if len({(s.ball_initial_position_m, s.ball_initial_velocity_mps) for s in scenes}) != 48:
        raise ValueError("fresh exam physical states not distinct")
    args.output_dir.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    for index, scene in enumerate(scenes):
        folder = args.output_dir / f"f{index:03d}"
        try:
            parent_report = _run_one(
                mode="parent",
                asset_root=args.asset_root,
                output_dir=folder,
                fixture=fixture,
                scenario=scene,
                protocol=protocol,
            )
        except ValueError as error:
            if str(error) != "incomplete paired motor episode":
                raise
            rows.append(
                {
                    "scene": f"f{index:03d}",
                    "scenario_hash": scene.scenario_hash,
                    "entry_hash": None,
                    "selected_arm": None,
                    "parent": _failed("PAIR_INCOMPLETE"),
                    "fixed": _failed("PAIR_INCOMPLETE"),
                    "learned": _failed("PAIR_INCOMPLETE"),
                }
            )
            print(f"f{index:03d} PAIR_INCOMPLETE all fail-closed", flush=True)
            continue
        parent = score_parent(parent_report, folder / "parent/trajectory.npz")
        parent_entry = entry_features(folder / "parent/taskspace_trace.npz", 30)
        fixed_nav = TeamPhaseInterceptNavigation(
            agent_id=protocol["focal_agent_id"],
            foundation_hash=policy_hash,
            foundation_config_hash=config_hash,
            forward_gain=0.4,
            lateral_gain=1.2,
        )
        learned_nav = TeamFootVelocityChooser(
            agent_id=protocol["focal_agent_id"],
            foundation_hash=policy_hash,
            foundation_config_hash=config_hash,
            model_path=model_path,
        )
        variants = {
            "fixed": (fixed_nav, None),
            "learned": (
                learned_nav,
                TeamSwingMotor(
                    protocol["focal_agent_id"],
                    True,
                    protocol["candidate_action"],
                    activation_selector=learned_nav,
                ),
            ),
        }
        scored: dict[str, dict[str, Any]] = {}
        for name, (navigation, motor) in variants.items():
            output = folder / name
            try:
                episode = _run_one(
                    mode="candidate",
                    asset_root=args.asset_root,
                    output_dir=output,
                    fixture=fixture,
                    scenario=scene,
                    protocol=protocol,
                    navigation_policy=navigation,
                    motor_option=motor,
                )
                score = _score(episode, output / "candidate/trajectory.npz")
                entry = entry_features(output / "candidate/taskspace_trace.npz", 30)
                if entry["hash"] != parent_entry["hash"]:
                    raise ValueError("fresh intervention changed pre-decision physical state")
                scored[name] = {"status": "COMPLETE", **score}
            except ValueError as error:
                if str(error) not in {
                    "incomplete paired motor episode",
                    "task-space side used future contact or altered support leg",
                }:
                    raise
                status = (
                    "INCOMPLETE"
                    if str(error) == "incomplete paired motor episode"
                    else "AUDIT_REJECTED"
                )
                scored[name] = _failed(status)
        rows.append(
            {
                "scene": f"f{index:03d}",
                "scenario_hash": scene.scenario_hash,
                "entry_hash": parent_entry["hash"],
                "selected_arm": learned_nav.selected_arm,
                "parent": {"status": "COMPLETE", **parent},
                **scored,
            }
        )
        print(
            f"f{index:03d} arm={learned_nav.selected_arm} "
            f"fixed={scored['fixed']['safe']}/{scored['fixed']['useful_pass']} "
            f"learned={scored['learned']['safe']}/{scored['learned']['useful_pass']}",
            flush=True,
        )
    if sources != {
        str(path.relative_to(root)): hash_bytes(path.read_bytes()) for path in source_paths
    }:
        raise ValueError("physical source changed during fresh exam")
    totals = {
        name: {
            "unsafe": sum(not row[name]["safe"] for row in rows),
            "safe_contact": sum(
                bool(row[name]["safe"] and row[name]["foot_contact_frames"]) for row in rows
            ),
            "useful": sum(row[name]["useful_pass"] for row in rows),
            "incomplete": sum(row[name]["status"] != "COMPLETE" for row in rows),
        }
        for name in ("parent", "fixed", "learned")
    }
    gate = bool(
        len(rows) >= protocol["gate"]["minimum_scenes"]
        and totals["learned"]["useful"] >= protocol["gate"]["minimum_candidate_useful"]
        and totals["learned"]["useful"] - totals["fixed"]["useful"]
        >= protocol["gate"]["minimum_useful_gain_vs_fixed"]
        and totals["learned"]["unsafe"] - totals["fixed"]["unsafe"]
        <= protocol["gate"]["maximum_unsafe_excess_vs_fixed"]
        and totals["learned"]["unsafe"] - totals["parent"]["unsafe"]
        <= protocol["gate"]["maximum_unsafe_excess_vs_parent"]
    )
    result = {
        "schema": "rsi_team_large_context_fresh_exam_report_v40",
        "activation_ceiling": "SIM_ONLY",
        "fresh_physical_coordinates": True,
        "protocol_hash": hash_bytes(args.protocol.read_bytes()),
        "model_hash": protocol["model_hash"],
        "body_hash": qualification.body_hash,
        "physical_source_hashes": sources,
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
        "RSI_TEAM_LARGE_CHOOSER_FRESH="
        + json.dumps(
            {key: result[key] for key in ("report_hash", "totals", "gate_passed")},
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
