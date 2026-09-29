"""Independent G1 physical triads for the frozen full-proprio tree selector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.rsi.team_phase_intercept_navigation import TeamPhaseInterceptNavigation
from rosclaw_soccer.rsi.team_proprio_tree_chooser import TeamProprioTreeChooser
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.independent_team_growth import build_independent_three_vs_three_fixture
from scripts.rsi_team_contextual_action_dataset import entry_features
from scripts.rsi_team_foot_velocity_fresh_exam import score_parent
from scripts.rsi_team_intercept_navigation_search import _score
from scripts.rsi_team_large_context_fresh_exam import _failed, _scenes
from scripts.rsi_team_lateral_foot_interception import _totals
from scripts.rsi_team_taskspace_first_touch import TeamSwingMotor, _run_one


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("fresh tree chooser exam output exists")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if (
        protocol.get("schema") != "rsi_team_proprio_tree_fresh_exam_protocol_v48"
        or protocol.get("activation_ceiling") != "SIM_ONLY"
        or protocol.get("development_only") is not True
        or protocol.get("promotion_authorized") is not False
        or protocol.get("focal_agent_id") != "red.playmaker"
        or protocol.get("frames") != 250
        or protocol.get("scene_count") != 48
        or protocol.get("coordinate_seed") != 20261023
        or protocol.get("identity_seed_base") != 1400000
        or protocol.get("ball_x_m") != [3.4, 4.15]
        or protocol.get("ball_y_m") != [-0.95, -0.45]
        or protocol.get("ball_vx_mps") != [-0.75, -0.25]
        or protocol.get("fixed_arm") != "phase_front04_lat12"
        or protocol.get("candidate_action")
        != {
            "entry_frame": 30,
            "forward_cap_m": 0.16,
            "lateral_cap_m": 0.05,
            "vertical_offset_m": 0.04,
            "swing_foot_acquisition_gap_m": 0.55,
        }
        or protocol.get("gate")
        != {
            "minimum_complete_triads": 48,
            "minimum_learned_useful": 10,
            "minimum_useful_gain_vs_fixed": 4,
            "minimum_safe_contact_gain_vs_fixed": 3,
            "maximum_unsafe_excess_vs_fixed": 0,
            "maximum_unsafe_excess_vs_parent": 0,
        }
    ):
        raise ValueError("invalid frozen full-proprio fresh protocol")
    export = json.loads(Path(protocol["export_report_path"]).read_text())
    if (
        export.get("report_hash") != protocol["export_report_hash"]
        or hash_json({key: value for key, value in export.items() if key != "report_hash"})
        != protocol["export_report_hash"]
        or export.get("model_hash") != protocol["model_hash"]
        or export.get("validation_decisions_matched") != 128
    ):
        raise ValueError("unsealed learned model export")
    root = Path(__file__).parents[1]
    qualification = qualify_g1_assets(args.asset_root)
    qualification.require_eligible()
    model_path = Path(protocol["model_path"])
    model = json.loads(model_path.read_text())
    if (
        model.get("model_hash") != protocol["model_hash"]
        or hash_json({key: value for key, value in model.items() if key != "model_hash"})
        != protocol["model_hash"]
        or model.get("body_hash") != qualification.body_hash
    ):
        raise ValueError("tree model not bound to eligible G1 physical body")
    fixture = build_independent_three_vs_three_fixture(args.asset_root)
    policy_hash = hash_bytes(
        (args.asset_root / "policy/loco_mode/model/policy_29dof.pt").read_bytes()
    )
    config_hash = hash_bytes(
        (args.asset_root / "policy/loco_mode/config/LocoMode.yaml").read_bytes()
    )
    source_paths = (
        Path(__file__),
        Path(__file__).with_name("rsi_team_taskspace_first_touch.py"),
        root / "src/rosclaw_soccer/rsi/team_proprio_tree_chooser.py",
        root / "src/rosclaw_soccer/rsi/team_adaptive_intercept_navigation.py",
        root / "src/rosclaw_soccer/rsi/team_intercept_navigation.py",
        root / "src/rosclaw_soccer/rsi/team_phase_intercept_navigation.py",
        root / "src/rosclaw_soccer/rsi/taskspace_swing_probe.py",
        root / "src/rosclaw_soccer/rsi/taskspace_swing_evidence.py",
        root / "src/rosclaw_soccer/skills/team/navigation_option.py",
        root / "src/rosclaw_soccer/skills/team/independent_team_world.py",
    )
    sources = {str(path.relative_to(root)): hash_bytes(path.read_bytes()) for path in source_paths}
    scenes = _scenes(protocol)
    if len({(s.ball_initial_position_m, s.ball_initial_velocity_mps) for s in scenes}) != 48:
        raise ValueError("new physical states not distinct")
    old_coordinates = set()
    for prior_protocol in (
        "docs/rsi/protocols/team-large-context-fresh-exam-v40.json",
        "docs/rsi/protocols/team-lateral-foot-fresh-exam-v42.json",
        "docs/rsi/protocols/team-local-joint-risk-fresh-exam-v45.json",
    ):
        prior = json.loads((root / prior_protocol).read_text())
        old_coordinates.update(
            (s.ball_initial_position_m, s.ball_initial_velocity_mps) for s in _scenes(prior)
        )
    if old_coordinates.intersection(
        (s.ball_initial_position_m, s.ball_initial_velocity_mps) for s in scenes
    ):
        raise ValueError("fresh chooser reused consumed physical coordinate")
    args.output_dir.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    for index, scene in enumerate(scenes):
        folder = args.output_dir / f"f{index:03d}"
        row: dict[str, Any] = {
            "scene": f"f{index:03d}",
            "scenario_hash": scene.scenario_hash,
            "entry_hash": None,
            "selected_arm": None,
        }
        try:
            parent_report = _run_one(
                mode="parent",
                asset_root=args.asset_root,
                output_dir=folder / "parent",
                fixture=fixture,
                scenario=scene,
                protocol=protocol,
            )
        except ValueError as error:
            if str(error) != "incomplete paired motor episode":
                raise
            row.update(
                {name: _failed("PAIR_INCOMPLETE") for name in ("parent", "fixed", "learned")}
            )
            rows.append(row)
            print(f"f{index:03d} PAIR_INCOMPLETE all fail-closed", flush=True)
            continue
        row["parent"] = {
            "status": "COMPLETE",
            **score_parent(parent_report, folder / "parent/parent/trajectory.npz"),
        }
        parent_entry = entry_features(folder / "parent/parent/taskspace_trace.npz", 30)
        row["entry_hash"] = parent_entry["hash"]
        fixed_nav = TeamPhaseInterceptNavigation(
            agent_id=protocol["focal_agent_id"],
            foundation_hash=policy_hash,
            foundation_config_hash=config_hash,
            forward_gain=0.4,
            lateral_gain=1.2,
        )
        learned_nav = TeamProprioTreeChooser(
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
                entry = entry_features(output / "candidate/taskspace_trace.npz", 30)
                if entry["hash"] != parent_entry["hash"]:
                    raise ValueError("tree chooser changed pre-decision physical state")
                row[name] = {
                    "status": "COMPLETE",
                    **_score(episode, output / "candidate/trajectory.npz"),
                }
            except ValueError as error:
                if str(error) not in {
                    "incomplete paired motor episode",
                    "task-space side used future contact or altered support leg",
                }:
                    raise
                row[name] = _failed(
                    "INCOMPLETE"
                    if str(error) == "incomplete paired motor episode"
                    else "AUDIT_REJECTED"
                )
        row["selected_arm"] = learned_nav.selected_arm
        rows.append(row)
        print(
            f"f{index:03d} arm={learned_nav.selected_arm} "
            f"fixed={row['fixed']['safe']}/{row['fixed']['useful_pass']} "
            f"learned={row['learned']['safe']}/{row['learned']['useful_pass']}",
            flush=True,
        )
    if sources != {
        str(path.relative_to(root)): hash_bytes(path.read_bytes()) for path in source_paths
    }:
        raise ValueError("physical source changed during tree chooser exam")
    totals = {name: _totals(rows, name) for name in ("parent", "fixed", "learned")}
    gate = protocol["gate"]
    passed = bool(
        len(rows) == gate["minimum_complete_triads"]
        and all(totals[name]["incomplete"] == 0 for name in totals)
        and totals["learned"]["useful"] >= gate["minimum_learned_useful"]
        and totals["learned"]["useful"] - totals["fixed"]["useful"]
        >= gate["minimum_useful_gain_vs_fixed"]
        and totals["learned"]["safe_contact"] - totals["fixed"]["safe_contact"]
        >= gate["minimum_safe_contact_gain_vs_fixed"]
        and totals["learned"]["unsafe"] - totals["fixed"]["unsafe"]
        <= gate["maximum_unsafe_excess_vs_fixed"]
        and totals["learned"]["unsafe"] - totals["parent"]["unsafe"]
        <= gate["maximum_unsafe_excess_vs_parent"]
    )
    result = {
        "schema": "rsi_team_proprio_tree_fresh_exam_report_v48",
        "activation_ceiling": "SIM_ONLY",
        "fresh_physical_coordinates": True,
        "promotion_authorized": False,
        "protocol_hash": hash_bytes(args.protocol.read_bytes()),
        "model_hash": protocol["model_hash"],
        "body_hash": qualification.body_hash,
        "physical_source_hashes": sources,
        "rows": rows,
        "totals": totals,
        "gate_passed": passed,
    }
    result["report_hash"] = hash_json(result)
    (args.output_dir / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "RSI_TEAM_PROPRIO_TREE_FRESH="
        + json.dumps(
            {key: result[key] for key in ("report_hash", "totals", "gate_passed")},
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
