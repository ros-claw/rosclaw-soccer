"""Compare bounded physical strike-through motor variants on consumed 3v3 scenes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.rsi.team_intercept_navigation import TeamInterceptNavigation
from rosclaw_soccer.rsi.team_phase_intercept_navigation import TeamPhaseInterceptNavigation
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.independent_team_growth import build_independent_three_vs_three_fixture
from scripts.rsi_team_contextual_action_dataset import entry_features
from scripts.rsi_team_contextual_nav_fresh_exam import fresh_scenarios
from scripts.rsi_team_intercept_navigation_search import _score
from scripts.rsi_team_taskspace_first_touch import _run_one


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--arm", required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("strike-through output already exists")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    arm = next((item for item in protocol["arms"] if item["name"] == args.arm), None)
    prior = json.loads(Path(protocol["failed_exam_report_path"]).read_text())
    if (
        protocol.get("schema") != "rsi_team_strike_through_probe_protocol_v27"
        or protocol.get("development_only") is not True
        or protocol.get("promotion_authorized") is not False
        or prior.get("report_hash") != protocol["failed_exam_report_hash"]
        or hash_json({key: value for key, value in prior.items() if key != "report_hash"})
        != prior["report_hash"]
        or arm is None
        or len({item["name"] for item in protocol["arms"]}) != len(protocol["arms"])
    ):
        raise ValueError("uncommitted consumed SIM_ONLY motor protocol")
    fresh_protocol = json.loads(Path(protocol["failed_exam_protocol_path"]).read_text())
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
        root / "src/rosclaw_soccer/rsi/taskspace_swing_probe.py",
        root / "src/rosclaw_soccer/rsi/taskspace_swing_evidence.py",
        root / "src/rosclaw_soccer/rsi/team_intercept_navigation.py",
        root / "src/rosclaw_soccer/rsi/team_phase_intercept_navigation.py",
        root / "src/rosclaw_soccer/skills/team/independent_team_world.py",
    )
    source_hashes = {
        str(path.relative_to(root)): hash_bytes(path.read_bytes()) for path in source_paths
    }
    args.output_dir.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    for index, scenario in enumerate(fresh_scenarios(fresh_protocol)):
        candidate_action = dict(fresh_protocol["candidate_action"])
        candidate_action["strike_through_m"] = arm["strike_through_m"]
        episode_protocol = dict(fresh_protocol)
        episode_protocol["candidate_action"] = candidate_action
        mode = "parent" if arm["name"] == "parent" else "candidate"
        navigation = None
        if mode == "candidate":
            nav_type = (
                TeamPhaseInterceptNavigation
                if arm["foot_selection"] == "phase"
                else TeamInterceptNavigation
            )
            navigation = nav_type(
                agent_id=fresh_protocol["focal_agent_id"],
                foundation_hash=policy_hash,
                foundation_config_hash=config_hash,
                forward_gain=float(arm["forward_gain"]),
                lateral_gain=float(arm["lateral_gain"]),
            )
        folder = args.output_dir / f"f{index:02d}"
        try:
            report = _run_one(
                mode=mode,
                asset_root=args.asset_root,
                output_dir=folder,
                fixture=fixture,
                scenario=scenario,
                protocol=episode_protocol,
                navigation_policy=navigation,
            )
        except ValueError as error:
            if mode == "parent" or str(error) != "incomplete paired motor episode":
                raise
            rows.append(
                {
                    "scene": f"f{index:02d}",
                    "status": "INCOMPLETE",
                    "safe": False,
                    "foot_contact_frames": [],
                    "useful_pass": False,
                }
            )
            print(f"{args.arm} f{index:02d} INCOMPLETE", flush=True)
            continue
        entry = entry_features(folder / mode / "taskspace_trace.npz", 30)
        if entry["hash"] != prior["episodes"][index]["entry_hash"]:
            raise ValueError("motor experiment changed pre-intervention entry observation")
        if mode == "candidate":
            outcome = _score(report, folder / mode / "trajectory.npz")
        else:
            outcome = {
                "safe": report["world_result"]["safe"],
                "foot_contact_frames": report["focal_foot_contact_frames"],
                "useful_pass": False,
            }
        outcome.update(
            {"scene": f"f{index:02d}", "status": "COMPLETE", "entry_hash": entry["hash"]}
        )
        if mode == "parent":
            old_parent = json.loads(
                (
                    Path(protocol["failed_exam_report_path"]).parent
                    / f"f{index:02d}/parent/report.json"
                ).read_text()
            )
            if report["trajectory_digest"] != old_parent["trajectory_digest"]:
                raise ValueError("zero-through frozen parent physics changed")
        rows.append(outcome)
        print(
            f"{args.arm} f{index:02d} safe={outcome['safe']} useful={outcome['useful_pass']}",
            flush=True,
        )
    if source_hashes != {
        str(path.relative_to(root)): hash_bytes(path.read_bytes()) for path in source_paths
    }:
        raise ValueError("source changed during physical strike-through probe")
    summary = {
        "schema": "rsi_team_strike_through_probe_arm_v27",
        "activation_ceiling": "SIM_ONLY",
        "protocol_hash": hash_bytes(args.protocol.read_bytes()),
        "source_hashes": source_hashes,
        "asset_body_hash": qualification.body_hash,
        "arm": arm,
        "rows": rows,
        "safe_contact_count": sum(bool(row["safe"] and row["foot_contact_frames"]) for row in rows),
        "useful_pass_count": sum(bool(row["useful_pass"]) for row in rows),
        "incomplete_count": sum(row["status"] != "COMPLETE" for row in rows),
        "fresh_holdout": False,
        "promotion_authorized": False,
    }
    summary["report_hash"] = hash_json(summary)
    (args.output_dir / "report.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "RSI_TEAM_STRIKE_THROUGH="
        + json.dumps(
            {
                key: summary[key]
                for key in (
                    "arm",
                    "safe_contact_count",
                    "useful_pass_count",
                    "incomplete_count",
                    "report_hash",
                )
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
