"""Paired SIM_ONLY lateral-foot interventions on consumed v40 physical states."""

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
from scripts.rsi_team_taskspace_first_touch import _run_one


def _totals(rows: list[dict[str, Any]], arm: str) -> dict[str, int]:
    return {
        "unsafe": sum(not row[arm]["safe"] for row in rows),
        "safe_contact": sum(
            bool(row[arm]["safe"] and row[arm]["foot_contact_frames"]) for row in rows
        ),
        "useful": sum(bool(row[arm]["useful_pass"]) for row in rows),
        "incomplete": sum(row[arm]["status"] != "COMPLETE" for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("v41 development output already exists")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if (
        protocol.get("schema") != "rsi_team_lateral_foot_interception_protocol_v41"
        or protocol.get("activation_ceiling") != "SIM_ONLY"
        or protocol.get("development_only") is not True
        or protocol.get("promotion_authorized") is not False
        or protocol.get("candidate_variants")
        != [
            {
                "name": "lateral10_gap55",
                "lateral_cap_m": 0.10,
                "swing_foot_acquisition_gap_m": 0.55,
            },
            {
                "name": "lateral10_gap95",
                "lateral_cap_m": 0.10,
                "swing_foot_acquisition_gap_m": 0.95,
            },
        ]
        or protocol.get("development_gate")
        != {
            "minimum_scenes": 48,
            "minimum_safe_contact_gain_vs_fixed": 5,
            "minimum_useful_gain_vs_fixed": 3,
            "maximum_unsafe_excess_vs_fixed": 0,
        }
    ):
        raise ValueError("invalid frozen lateral-foot intervention protocol")
    root = Path(__file__).parents[1]
    baseline_protocol = json.loads((root / protocol["baseline_protocol"]).read_text())
    baseline = json.loads(Path(protocol["baseline_report"]).read_text())
    if (
        baseline.get("report_hash") != protocol["baseline_report_hash"]
        or hash_json({key: value for key, value in baseline.items() if key != "report_hash"})
        != protocol["baseline_report_hash"]
        or baseline.get("gate_passed") is not False
        or len(baseline.get("rows", [])) != 48
    ):
        raise ValueError("v40 consumed baseline missing or changed")
    scenes = _scenes(baseline_protocol)
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
        root / "src/rosclaw_soccer/rsi/team_phase_intercept_navigation.py",
        root / "src/rosclaw_soccer/skills/team/independent_team_world.py",
    )
    sources = {str(path.relative_to(root)): hash_bytes(path.read_bytes()) for path in source_paths}
    args.output_dir.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    for scene, prior in zip(scenes, baseline["rows"], strict=True):
        if scene.scenario_hash != prior["scenario_hash"]:
            raise ValueError("physical scenario no longer matches consumed baseline")
        row = {
            "scene": prior["scene"],
            "scenario_hash": scene.scenario_hash,
            "entry_hash": prior["entry_hash"],
            "fixed": prior["fixed"],
        }
        for variant in protocol["candidate_variants"]:
            name = variant["name"]
            run_protocol = dict(baseline_protocol)
            run_protocol["candidate_action"] = {
                **baseline_protocol["candidate_action"],
                "lateral_cap_m": variant["lateral_cap_m"],
                "swing_foot_acquisition_gap_m": variant["swing_foot_acquisition_gap_m"],
            }
            folder = args.output_dir / prior["scene"] / name
            navigation = TeamPhaseInterceptNavigation(
                agent_id=run_protocol["focal_agent_id"],
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
                if entry["hash"] != prior["entry_hash"]:
                    raise ValueError("v41 intervention changed pre-decision physical state")
                row[name] = {
                    "status": "COMPLETE",
                    **_score(episode, folder / "candidate/trajectory.npz"),
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
        rows.append(row)
        print(
            f"{row['scene']} "
            + " ".join(
                f"{variant['name']}={row[variant['name']]['safe']}/"
                f"{row[variant['name']]['useful_pass']}"
                for variant in protocol["candidate_variants"]
            ),
            flush=True,
        )
    if sources != {
        str(path.relative_to(root)): hash_bytes(path.read_bytes()) for path in source_paths
    }:
        raise ValueError("physical source changed during v41")
    totals = {name: _totals(rows, name) for name in ("fixed", "lateral10_gap55", "lateral10_gap95")}
    gate = protocol["development_gate"]
    passing = [
        name
        for name in ("lateral10_gap55", "lateral10_gap95")
        if len(rows) >= gate["minimum_scenes"]
        and totals[name]["safe_contact"] - totals["fixed"]["safe_contact"]
        >= gate["minimum_safe_contact_gain_vs_fixed"]
        and totals[name]["useful"] - totals["fixed"]["useful"]
        >= gate["minimum_useful_gain_vs_fixed"]
        and totals[name]["unsafe"] - totals["fixed"]["unsafe"]
        <= gate["maximum_unsafe_excess_vs_fixed"]
    ]
    result = {
        "schema": "rsi_team_lateral_foot_interception_report_v41",
        "activation_ceiling": "SIM_ONLY",
        "development_only": True,
        "promotion_authorized": False,
        "fresh_physical_coordinates": False,
        "protocol_hash": hash_bytes(args.protocol.read_bytes()),
        "baseline_report_hash": baseline["report_hash"],
        "body_hash": qualification.body_hash,
        "physical_source_hashes": sources,
        "rows": rows,
        "totals": totals,
        "development_gate_passed": bool(passing),
        "passing_variants": passing,
    }
    result["report_hash"] = hash_json(result)
    (args.output_dir / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "RSI_TEAM_LATERAL_FOOT="
        + json.dumps(
            {key: result[key] for key in ("report_hash", "totals", "passing_variants")},
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
