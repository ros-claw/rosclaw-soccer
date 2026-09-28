"""Paired consumed-state G1 joint-boundary training under a stronger guard."""

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
        parser.error("joint-guard development output exists")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if (
        protocol.get("schema") != "rsi_team_joint_guard_transfer_protocol_v43"
        or protocol.get("activation_ceiling") != "SIM_ONLY"
        or protocol.get("development_only") is not True
        or protocol.get("promotion_authorized") is not False
        or protocol.get("focal_agent_id") != "red.playmaker"
        or protocol.get("frames") != 250
        or protocol.get("world_joint_guard_margin_rad") != 0.08
        or protocol.get("fixed_action")
        != {
            "entry_frame": 30,
            "forward_cap_m": 0.16,
            "lateral_cap_m": 0.05,
            "vertical_offset_m": 0.04,
            "swing_foot_acquisition_gap_m": 0.55,
        }
        or protocol.get("candidate_action")
        != {
            "entry_frame": 30,
            "forward_cap_m": 0.16,
            "lateral_cap_m": 0.10,
            "vertical_offset_m": 0.04,
            "swing_foot_acquisition_gap_m": 0.55,
        }
        or protocol.get("development_gate")
        != {
            "minimum_complete_pairs": 48,
            "minimum_candidate_useful": 12,
            "minimum_useful_gain_vs_fixed": 4,
            "minimum_safe_contact_gain_vs_fixed": 4,
            "maximum_unsafe_excess_vs_fixed": 0,
        }
    ):
        raise ValueError("invalid frozen joint-guard transfer protocol")
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
        root / "src/rosclaw_soccer/rsi/team_phase_intercept_navigation.py",
        root / "src/rosclaw_soccer/skills/team/independent_team_world.py",
    )
    sources = {str(path.relative_to(root)): hash_bytes(path.read_bytes()) for path in source_paths}
    args.output_dir.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    for index, (scene, previous) in enumerate(zip(scenes, consumed["rows"], strict=True)):
        if scene.scenario_hash != previous["scenario_hash"]:
            raise ValueError("consumed physical scenario changed")
        folder = args.output_dir / f"f{index:03d}"
        row: dict[str, Any] = {
            "scene": f"f{index:03d}",
            "scenario_hash": scene.scenario_hash,
            "entry_hash": None,
        }
        for name, action in (
            ("fixed", protocol["fixed_action"]),
            ("candidate", protocol["candidate_action"]),
        ):
            run_protocol = {
                "focal_agent_id": protocol["focal_agent_id"],
                "frames": protocol["frames"],
                "candidate_action": action,
                "world_joint_guard_margin_rad": protocol["world_joint_guard_margin_rad"],
            }
            navigation = TeamPhaseInterceptNavigation(
                agent_id=protocol["focal_agent_id"],
                foundation_hash=policy_hash,
                foundation_config_hash=config_hash,
                forward_gain=0.4,
                lateral_gain=1.2,
            )
            output = folder / name
            try:
                episode = _run_one(
                    mode="candidate",
                    asset_root=args.asset_root,
                    output_dir=output,
                    fixture=fixture,
                    scenario=scene,
                    protocol=run_protocol,
                    navigation_policy=navigation,
                )
                entry = entry_features(output / "candidate/taskspace_trace.npz", 30)
                if row["entry_hash"] is None:
                    row["entry_hash"] = entry["hash"]
                elif row["entry_hash"] != entry["hash"]:
                    raise ValueError("joint-guard pair changed pre-decision physical state")
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
        rows.append(row)
        print(
            f"f{index:03d} fixed={row['fixed']['safe']}/{row['fixed']['useful_pass']} "
            f"candidate={row['candidate']['safe']}/{row['candidate']['useful_pass']}",
            flush=True,
        )
    if sources != {
        str(path.relative_to(root)): hash_bytes(path.read_bytes()) for path in source_paths
    }:
        raise ValueError("physical source changed during joint-guard development")
    totals = {name: _totals(rows, name) for name in ("fixed", "candidate")}
    gate = protocol["development_gate"]
    passed = bool(
        len(rows) == gate["minimum_complete_pairs"]
        and totals["fixed"]["incomplete"] == 0
        and totals["candidate"]["incomplete"] == 0
        and totals["candidate"]["useful"] >= gate["minimum_candidate_useful"]
        and totals["candidate"]["useful"] - totals["fixed"]["useful"]
        >= gate["minimum_useful_gain_vs_fixed"]
        and totals["candidate"]["safe_contact"] - totals["fixed"]["safe_contact"]
        >= gate["minimum_safe_contact_gain_vs_fixed"]
        and totals["candidate"]["unsafe"] - totals["fixed"]["unsafe"]
        <= gate["maximum_unsafe_excess_vs_fixed"]
    )
    result = {
        "schema": "rsi_team_joint_guard_transfer_report_v43",
        "activation_ceiling": "SIM_ONLY",
        "development_only": True,
        "promotion_authorized": False,
        "fresh_physical_coordinates": False,
        "protocol_hash": hash_bytes(args.protocol.read_bytes()),
        "consumed_report_hash": consumed["report_hash"],
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
        "RSI_TEAM_JOINT_GUARD="
        + json.dumps(
            {key: result[key] for key in ("report_hash", "totals", "development_gate_passed")},
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
