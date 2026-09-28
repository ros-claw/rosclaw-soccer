"""Audit a causal swing-leg risk taper on consumed physical team scenes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.rsi.team_phase_intercept_navigation import TeamPhaseInterceptNavigation
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.independent_team_growth import build_independent_three_vs_three_fixture
from scripts.rsi_team_contextual_action_dataset import entry_features
from scripts.rsi_team_intercept_navigation_search import _score
from scripts.rsi_team_large_context_fresh_exam import _failed, _scenes
from scripts.rsi_team_lateral_foot_interception import _totals
from scripts.rsi_team_taskspace_first_touch import _run_one


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("local-risk development output already exists")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if (
        protocol.get("schema") != "rsi_team_local_joint_risk_swing_protocol_v44"
        or protocol.get("activation_ceiling") != "SIM_ONLY"
        or protocol.get("development_only") is not True
        or protocol.get("promotion_authorized") is not False
        or protocol.get("focal_agent_id") != "red.playmaker"
        or protocol.get("frames") != 250
        or protocol.get("candidate_action")
        != {
            "entry_frame": 30,
            "forward_cap_m": 0.16,
            "lateral_cap_m": 0.10,
            "vertical_offset_m": 0.04,
            "swing_foot_acquisition_gap_m": 0.55,
            "joint_risk_guard_margin_rad": 0.12,
        }
        or protocol.get("development_gate")
        != {
            "minimum_scenes": 48,
            "maximum_candidate_incomplete": 0,
            "minimum_candidate_useful": 12,
            "minimum_useful_gain_vs_fixed": 4,
            "minimum_safe_contact_gain_vs_fixed": 4,
            "maximum_unsafe_excess_vs_fixed": 0,
        }
    ):
        raise ValueError("invalid frozen local joint-risk swing protocol")
    root = Path(__file__).parents[1]
    consumed_protocol = json.loads((root / protocol["consumed_scenes_protocol"]).read_text())
    consumed = json.loads(Path(protocol["consumed_scenes_report"]).read_text())
    if (
        consumed.get("report_hash") != protocol["consumed_scenes_report_hash"]
        or hash_json({key: value for key, value in consumed.items() if key != "report_hash"})
        != protocol["consumed_scenes_report_hash"]
        or consumed.get("gate_passed") is not False
        or len(consumed.get("rows", [])) != 48
    ):
        raise ValueError("consumed v42 evidence missing or changed")
    scenes = _scenes(consumed_protocol)
    qualification = qualify_g1_assets(args.asset_root)
    qualification.require_eligible()
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
        root / "src/rosclaw_soccer/rsi/taskspace_swing_probe.py",
        root / "src/rosclaw_soccer/rsi/taskspace_swing_evidence.py",
        root / "src/rosclaw_soccer/rsi/team_phase_intercept_navigation.py",
        root / "src/rosclaw_soccer/skills/team/independent_team_world.py",
    )
    sources = {str(path.relative_to(root)): hash_bytes(path.read_bytes()) for path in source_paths}
    args.output_dir.mkdir(parents=True)
    # A fixed-arm sentinel confirms the new optional guard did not alter old physics.
    sentinel_scene = scenes[1]
    sentinel_protocol = {
        "focal_agent_id": protocol["focal_agent_id"],
        "frames": protocol["frames"],
        "candidate_action": consumed_protocol["fixed_action"],
    }
    sentinel_nav = TeamPhaseInterceptNavigation(
        agent_id=protocol["focal_agent_id"],
        foundation_hash=policy_hash,
        foundation_config_hash=config_hash,
        forward_gain=0.4,
        lateral_gain=1.2,
    )
    sentinel = _run_one(
        mode="candidate",
        asset_root=args.asset_root,
        output_dir=args.output_dir / "sentinel_fixed_f001",
        fixture=fixture,
        scenario=sentinel_scene,
        protocol=sentinel_protocol,
        navigation_policy=sentinel_nav,
    )
    old_sentinel = json.loads(
        (
            Path(protocol["consumed_scenes_report"]).parent / "f001/fixed/candidate/report.json"
        ).read_text()
    )
    if (
        sentinel["trace_hash"] != old_sentinel["trace_hash"]
        or sentinel["action_trace_hash"] != old_sentinel["action_trace_hash"]
        or sentinel["world_result"] != old_sentinel["world_result"]
    ):
        raise ValueError("optional local guard changed fixed baseline physics")
    rows: list[dict[str, Any]] = []
    for index, (scene, previous) in enumerate(zip(scenes, consumed["rows"], strict=True)):
        if scene.scenario_hash != previous["scenario_hash"] or previous["entry_hash"] is None:
            raise ValueError("consumed physical scene or entry changed")
        folder = args.output_dir / f"f{index:03d}"
        run_protocol = {
            "focal_agent_id": protocol["focal_agent_id"],
            "frames": protocol["frames"],
            "candidate_action": protocol["candidate_action"],
        }
        navigation = TeamPhaseInterceptNavigation(
            agent_id=protocol["focal_agent_id"],
            foundation_hash=policy_hash,
            foundation_config_hash=config_hash,
            forward_gain=0.4,
            lateral_gain=1.2,
        )
        try:
            episode = _run_one(
                mode="candidate",
                asset_root=args.asset_root,
                output_dir=folder,
                fixture=fixture,
                scenario=scene,
                protocol=run_protocol,
                navigation_policy=navigation,
            )
            entry = entry_features(folder / "candidate/taskspace_trace.npz", 30)
            if entry["hash"] != previous["entry_hash"]:
                raise ValueError("local guard changed pre-decision physical state")
            candidate = {
                "status": "COMPLETE",
                **_score(episode, folder / "candidate/trajectory.npz"),
            }
        except ValueError as error:
            if str(error) not in {
                "incomplete paired motor episode",
                "task-space side used future contact or altered support leg",
            }:
                raise
            candidate = _failed(
                "INCOMPLETE"
                if str(error) == "incomplete paired motor episode"
                else "AUDIT_REJECTED"
            )
        rows.append(
            {
                "scene": f"f{index:03d}",
                "scenario_hash": scene.scenario_hash,
                "entry_hash": previous["entry_hash"],
                "fixed": previous["fixed"],
                "candidate": candidate,
            }
        )
        print(
            f"f{index:03d} fixed={previous['fixed']['safe']}/"
            f"{previous['fixed']['useful_pass']} "
            f"candidate={candidate['safe']}/{candidate['useful_pass']}",
            flush=True,
        )
    if sources != {
        str(path.relative_to(root)): hash_bytes(path.read_bytes()) for path in source_paths
    }:
        raise ValueError("physical source changed during local guard study")
    totals = {name: _totals(rows, name) for name in ("fixed", "candidate")}
    gate = protocol["development_gate"]
    passed = bool(
        len(rows) == gate["minimum_scenes"]
        and totals["candidate"]["incomplete"] <= gate["maximum_candidate_incomplete"]
        and totals["candidate"]["useful"] >= gate["minimum_candidate_useful"]
        and totals["candidate"]["useful"] - totals["fixed"]["useful"]
        >= gate["minimum_useful_gain_vs_fixed"]
        and totals["candidate"]["safe_contact"] - totals["fixed"]["safe_contact"]
        >= gate["minimum_safe_contact_gain_vs_fixed"]
        and totals["candidate"]["unsafe"] - totals["fixed"]["unsafe"]
        <= gate["maximum_unsafe_excess_vs_fixed"]
    )
    result = {
        "schema": "rsi_team_local_joint_risk_swing_report_v44",
        "activation_ceiling": "SIM_ONLY",
        "development_only": True,
        "promotion_authorized": False,
        "fresh_physical_coordinates": False,
        "protocol_hash": hash_bytes(args.protocol.read_bytes()),
        "consumed_report_hash": consumed["report_hash"],
        "sentinel_trace_hash": sentinel["trace_hash"],
        "body_hash": qualification.body_hash,
        "physical_source_hashes": sources,
        "rows": rows,
        "totals": totals,
        "development_gate_passed": passed,
    }
    result["report_hash"] = hash_json(result)
    (args.output_dir / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "RSI_TEAM_LOCAL_JOINT_RISK="
        + json.dumps(
            {key: result[key] for key in ("report_hash", "totals", "development_gate_passed")},
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
