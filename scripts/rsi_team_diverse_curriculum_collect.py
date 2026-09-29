"""Collect physically varied SIM_ONLY 3v3 first-touch curriculum episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.rsi.team_adaptive_intercept_navigation import TeamAdaptiveInterceptNavigation
from rosclaw_soccer.rsi.team_intercept_navigation import TeamInterceptNavigation
from rosclaw_soccer.rsi.team_phase_intercept_navigation import TeamPhaseInterceptNavigation
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.independent_team_world import IndependentTeamWorldScenario
from rosclaw_soccer.training.independent_team_growth import build_independent_three_vs_three_fixture
from scripts.rsi_team_contextual_action_dataset import entry_features
from scripts.rsi_team_intercept_navigation_search import _score
from scripts.rsi_team_taskspace_first_touch import _run_one


def training_courses(
    protocol: dict[str, Any], batch_index: int = 0
) -> list[IndependentTeamWorldScenario]:
    curriculum = protocol["curriculum"]
    if type(batch_index) is not int or not 0 <= batch_index < curriculum.get("batch_count", 1):
        raise ValueError("invalid committed physical batch index")
    generator = np.random.default_rng(curriculum["coordinate_seed"] + batch_index)
    scenarios = []
    for index in range(curriculum["training_scene_count"]):
        x = round(float(generator.uniform(*curriculum["ball_x_m"])), 4)
        y = round(float(generator.uniform(*curriculum["ball_y_m"])), 4)
        vx = round(float(generator.uniform(*curriculum["ball_vx_mps"])), 4)
        scenarios.append(
            IndependentTeamWorldScenario(
                (
                    f"s199.rsi-diverse-b{batch_index:02d}-t{index:03d}"
                    if protocol["schema"] == "rsi_team_diverse_curriculum_protocol_v38"
                    else f"s199.rsi-diverse-training-t{index:03d}"
                ),
                (x, y, 0.115),
                (vx, 0.0, 0.0),
                curriculum["identity_seed_base"]
                + batch_index * curriculum["training_scene_count"]
                + index,
            )
        )
    return scenarios


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--batch-index", type=int, default=0)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("curriculum arm output already exists")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    arms = protocol["arms"]
    arm = next((item for item in arms if item["name"] == args.arm), None)
    scenarios = training_courses(protocol, args.batch_index)
    if (
        protocol.get("schema")
        not in {
            "rsi_team_diverse_curriculum_protocol_v29",
            "rsi_team_diverse_curriculum_protocol_v31",
            "rsi_team_diverse_curriculum_protocol_v38",
        }
        or protocol.get("development_only") is not True
        or protocol.get("promotion_authorized") is not False
        or protocol.get("frames") != 250
        or arm is None
        or len(scenarios) not in {32, 64, 128}
        or len({item["name"] for item in arms}) != len(arms)
        or len(arms) not in {3, 7, 8}
        or (
            len(arms) == 3
            and (
                [item["name"] for item in arms]
                not in (
                    ["parent", "baseline", "gate22_cap10"],
                    ["parent", "gate22_cap10", "gate22_revalidate"],
                )
                or protocol["curriculum"].get("batch_count") != 16
            )
        )
        or len({(s.ball_initial_position_m, s.ball_initial_velocity_mps) for s in scenarios})
        != len(scenarios)
    ):
        raise ValueError("uncommitted physically diverse SIM_ONLY training curriculum")
    adaptive_model = json.loads(Path(protocol["adaptive_model_path"]).read_text())
    if (
        adaptive_model.get("model_hash") != protocol["adaptive_model_hash"]
        or hash_json({key: value for key, value in adaptive_model.items() if key != "model_hash"})
        != adaptive_model["model_hash"]
    ):
        raise ValueError("frozen adaptive-intercept model mismatch")
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
        root / "src/rosclaw_soccer/rsi/team_adaptive_intercept_navigation.py",
        root / "src/rosclaw_soccer/rsi/team_intercept_navigation.py",
        root / "src/rosclaw_soccer/rsi/team_phase_intercept_navigation.py",
        root / "src/rosclaw_soccer/skills/team/independent_team_world.py",
    )
    source_hashes = {
        str(path.relative_to(root)): hash_bytes(path.read_bytes()) for path in source_paths
    }
    args.output_dir.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    for index, scenario in enumerate(scenarios):
        name = f"t{index:03d}"
        navigation: TeamAdaptiveInterceptNavigation | TeamInterceptNavigation | None = None
        if arm["name"] == "adaptive_c19":
            navigation = TeamAdaptiveInterceptNavigation(
                agent_id=protocol["focal_agent_id"],
                foundation_hash=policy_hash,
                foundation_config_hash=config_hash,
                **adaptive_model["parameters"],
            )
        elif arm["name"] != "parent":
            nav_type = (
                TeamPhaseInterceptNavigation
                if arm["foot_selection"] == "phase"
                else TeamInterceptNavigation
            )
            navigation_kwargs: dict[str, Any] = {}
            if "near_ball_lateral_hold_gap_m" in arm:
                if nav_type is not TeamPhaseInterceptNavigation:
                    raise ValueError("near-ball lateral hold requires phase foot selection")
                navigation_kwargs["near_ball_lateral_hold_gap_m"] = arm[
                    "near_ball_lateral_hold_gap_m"
                ]
            navigation = nav_type(
                agent_id=protocol["focal_agent_id"],
                foundation_hash=policy_hash,
                foundation_config_hash=config_hash,
                forward_gain=float(arm["forward_gain"]),
                lateral_gain=float(arm["lateral_gain"]),
                **navigation_kwargs,
            )
        episode_protocol = dict(protocol)
        action = dict(protocol["candidate_action"])
        action["strike_through_m"] = arm["strike_through_m"]
        for key in (
            "lateral_cap_m",
            "swing_foot_acquisition_gap_m",
            "swing_acquisition_max_lateral_gap_m",
            "revalidate_swing_side",
            "joint_risk_guard_margin_rad",
        ):
            if key in arm:
                action[key] = arm[key]
        episode_protocol["candidate_action"] = action
        mode = "parent" if arm["name"] == "parent" else "candidate"
        folder = args.output_dir / name
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
            if str(error) not in {
                "incomplete paired motor episode",
                "task-space side used future contact or altered support leg",
            }:
                raise
            rows.append(
                {
                    "scene": name,
                    "scenario_hash": scenario.scenario_hash,
                    "status": (
                        "INCOMPLETE"
                        if str(error) == "incomplete paired motor episode"
                        else "AUDIT_REJECTED"
                    ),
                    "safe": False,
                    "foot_contact_frames": [],
                    "useful_pass": False,
                    "entry_hash": None,
                    "minimum_foot_ball_gap_m": None,
                }
            )
            print(f"{args.arm} {name} INCOMPLETE", flush=True)
            continue
        trace_path = folder / mode / "taskspace_trace.npz"
        entry = entry_features(trace_path, 30)
        with np.load(trace_path, allow_pickle=False) as trace:
            feet = np.asarray(trace["pre_step_foot_link_position_w"])[:, 0]
            ball = np.asarray(trace["pre_step_ball_position_local_m"])[:, 0]
        if mode == "candidate":
            outcome = _score(report, folder / mode / "trajectory.npz")
        else:
            contacts = report["focal_foot_contact_frames"]
            first = contacts[0] if contacts else None
            with np.load(folder / mode / "trajectory.npz", allow_pickle=False) as physics:
                ball_velocity = np.asarray(physics["ball_velocity"])
            outgoing = (
                float(np.max(ball_velocity[first : min(first + 5, len(ball_velocity)), 0]))
                if first is not None
                else None
            )
            safe = bool(
                report["world_result"]["safe"] and not report["focal_nonfoot_contact_frames"]
            )
            outcome = {
                "safe": safe,
                "foot_contact_frames": contacts,
                "useful_pass": bool(safe and outgoing is not None and outgoing >= 0.5),
                "outgoing_ball_vx_mps": outgoing,
            }
        outcome.update(
            {
                "scene": name,
                "scenario_hash": scenario.scenario_hash,
                "status": "COMPLETE",
                "entry_hash": entry["hash"],
                "minimum_foot_ball_gap_m": float(
                    np.min(np.linalg.norm(feet - ball[:, None, :], axis=2))
                ),
            }
        )
        rows.append(outcome)
        print(
            f"{args.arm} {name} safe={outcome['safe']} useful={outcome['useful_pass']}", flush=True
        )
    if source_hashes != {
        str(path.relative_to(root)): hash_bytes(path.read_bytes()) for path in source_paths
    }:
        raise ValueError("physical curriculum source changed during collection")
    result = {
        "schema": "rsi_team_diverse_curriculum_arm_v29",
        "activation_ceiling": "SIM_ONLY",
        "protocol_hash": hash_bytes(args.protocol.read_bytes()),
        "source_hashes": source_hashes,
        "asset_body_hash": qualification.body_hash,
        "adaptive_model_hash": protocol["adaptive_model_hash"],
        "arm": arm,
        "batch_index": args.batch_index,
        "rows": rows,
        "safe_count": sum(bool(row["safe"]) for row in rows),
        "safe_contact_count": sum(bool(row["safe"] and row["foot_contact_frames"]) for row in rows),
        "useful_pass_count": sum(bool(row["useful_pass"]) for row in rows),
        "incomplete_count": sum(row["status"] != "COMPLETE" for row in rows),
        "fresh_holdout": False,
        "promotion_authorized": False,
    }
    result["report_hash"] = hash_json(result)
    (args.output_dir / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "RSI_TEAM_DIVERSE_CURRICULUM="
        + json.dumps(
            {
                key: result[key]
                for key in (
                    "arm",
                    "report_hash",
                    "safe_count",
                    "safe_contact_count",
                    "useful_pass_count",
                    "incomplete_count",
                )
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
