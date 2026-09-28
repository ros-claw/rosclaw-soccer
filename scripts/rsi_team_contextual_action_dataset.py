"""Collect paired, development-only 3v3 physical action outcomes on a frozen grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.rsi.team_intercept_navigation import TeamInterceptNavigation
from rosclaw_soccer.rsi.team_phase_intercept_navigation import TeamPhaseInterceptNavigation
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.independent_team_world import IndependentTeamWorldScenario
from rosclaw_soccer.training.independent_team_growth import build_independent_three_vs_three_fixture
from scripts.rsi_team_intercept_navigation_search import _score
from scripts.rsi_team_taskspace_first_touch import _run_one


def courses_from_protocol(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    grid = protocol["scenario_grid"]
    courses = []
    for x in grid["x_m"]:
        for y in grid["y_m"]:
            for vx in grid["incoming_vx_mps"]:
                index = len(courses)
                courses.append(
                    {
                        "name": f"d{index:03d}",
                        "scenario_id": f"s199.rsi-context-action-development-d{index:03d}",
                        "ball_initial_position_m": [x, y, 0.115],
                        "ball_initial_velocity_mps": [vx, 0.0, 0.0],
                        "seed": grid["seed_base"] + index,
                    }
                )
    return courses


def entry_features(path: Path, frame: int) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as trace:
        feet = np.asarray(trace["pre_step_foot_link_position_w"])[frame, 0]
        foot_velocity = np.asarray(trace["pre_step_foot_linear_velocity_w"])[frame, 0]
        qpos = np.asarray(trace["pre_step_focal_qpos"])[frame, 0]
        qvel = np.asarray(trace["pre_step_focal_qvel"])[frame, 0]
    if feet.shape != (2, 3) or qpos.shape != (43,) or qvel.shape != (41,):
        raise ValueError("incomplete measured same-frame entry observation")
    features = np.concatenate(
        (qpos[:3], qvel[:3], qpos[36:39], qvel[35:38], feet.ravel(), foot_velocity.ravel())
    )
    if features.shape != (24,) or not np.all(np.isfinite(features)):
        raise ValueError("invalid causal entry features")
    return {"values": features.tolist(), "hash": hash_bytes(features.astype("<f8").tobytes())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--arm", required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("output directory already exists")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    arms = protocol["arms"]
    arm = next((item for item in arms if item["name"] == args.arm), None)
    if (
        protocol.get("schema") != "rsi_team_contextual_action_dataset_protocol_v21"
        or protocol.get("development_only") is not True
        or protocol.get("promotion_authorized") is not False
        or protocol.get("frames") != 250
        or protocol.get("focal_agent_id") != "red.playmaker"
        or arm is None
        or len(courses_from_protocol(protocol)) != 24
        or len({item["name"] for item in arms}) != len(arms)
    ):
        raise ValueError("invalid frozen SIM_ONLY development protocol")
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
        Path(__file__).with_name("rsi_team_intercept_navigation_search.py"),
        root / "src/rosclaw_soccer/rsi/team_intercept_navigation.py",
        root / "src/rosclaw_soccer/rsi/team_phase_intercept_navigation.py",
        root / "src/rosclaw_soccer/rsi/taskspace_swing_probe.py",
        root / "src/rosclaw_soccer/skills/team/independent_team_world.py",
        root / "src/rosclaw_soccer/skills/team/foot_kinematics.py",
        root / "src/rosclaw_soccer/skills/team/contact_point_velocity.py",
    )
    source_hashes = {
        str(path.relative_to(root)): hash_bytes(path.read_bytes()) for path in source_paths
    }
    args.output_dir.mkdir(parents=True)
    rows = []
    for course in courses_from_protocol(protocol):
        scenario = IndependentTeamWorldScenario(
            course["scenario_id"],
            tuple(course["ball_initial_position_m"]),
            tuple(course["ball_initial_velocity_mps"]),
            course["seed"],
        )
        navigation = None
        if arm["name"] != "parent":
            nav_type = (
                TeamPhaseInterceptNavigation
                if arm["foot_selection"] == "phase"
                else TeamInterceptNavigation
            )
            navigation = nav_type(
                agent_id=protocol["focal_agent_id"],
                foundation_hash=policy_hash,
                foundation_config_hash=config_hash,
                forward_gain=float(arm["forward_gain"]),
                lateral_gain=float(arm["lateral_gain"]),
            )
        folder = args.output_dir / course["name"]
        report = _run_one(
            mode="parent" if arm["name"] == "parent" else "candidate",
            asset_root=args.asset_root,
            output_dir=folder,
            fixture=fixture,
            scenario=scenario,
            protocol=protocol,
            navigation_policy=navigation,
        )
        mode = report["mode"]
        outcome = _score(report, folder / mode / "trajectory.npz")
        outcome.update(
            {
                "course": course["name"],
                "entry": entry_features(folder / mode / "taskspace_trace.npz", 30),
                "navigation_contract_hash": None
                if navigation is None
                else navigation.contract_hash,
            }
        )
        rows.append(outcome)
        print(
            f"{args.arm} {course['name']} safe={outcome['safe']} useful={outcome['useful_pass']}",
            flush=True,
        )
    if source_hashes != {
        str(path.relative_to(root)): hash_bytes(path.read_bytes()) for path in source_paths
    }:
        raise ValueError("source changed during frozen physical dataset collection")
    summary = {
        "schema": "rsi_team_contextual_action_dataset_arm_v21",
        "activation_ceiling": "SIM_ONLY",
        "protocol_hash": hash_bytes(args.protocol.read_bytes()),
        "source_hashes": source_hashes,
        "asset_body_hash": qualification.body_hash,
        "arm": arm,
        "rows": rows,
        "safe_contact_count": sum(bool(r["safe"] and r["foot_contact_frames"]) for r in rows),
        "useful_pass_count": sum(bool(r["useful_pass"]) for r in rows),
        "unsafe_count": sum(not r["safe"] for r in rows),
        "promotion_authorized": False,
    }
    summary["report_hash"] = hash_json(summary)
    (args.output_dir / "report.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "RSI_TEAM_CONTEXTUAL_ARM="
        + json.dumps(
            {
                k: summary[k]
                for k in (
                    "arm",
                    "safe_contact_count",
                    "useful_pass_count",
                    "unsafe_count",
                    "report_hash",
                )
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
