"""Paired CPU MuJoCo measurements of purposeful multi-player engagement."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.growth.competitive_match_assessment import assess_competitive_match_trajectory
from rosclaw_soccer.growth.locomotion_contact_teacher import G1RollingOptionBridgeConfig
from rosclaw_soccer.growth.near_ball_residual import NearBallResidualPolicy
from rosclaw_soccer.growth.owned_ball_contact import OwnedBallContactPolicy
from rosclaw_soccer.growth.pass_failure_feedback import diagnose_passes
from rosclaw_soccer.growth.role_self_model import MatchRole
from rosclaw_soccer.growth.strike_phase_controller import StrikePhaseConfig
from rosclaw_soccer.providers.g1.asset_qualification import trajectory_digest
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.independent_team_world import simulate_independent_team_world
from rosclaw_soccer.training.continuous_competitive_match_growth import (
    build_continuous_competitive_fixture,
    default_continuous_match_config,
    default_continuous_match_scenario,
)
from rosclaw_soccer.training.four_vs_four_match import build_four_vs_four_fixture
from rosclaw_soccer.training.phase_conditioned_strike_growth import (
    default_phase_strike_controller,
    default_phase_strike_option,
    default_phase_strike_teacher,
)
from rosclaw_soccer.world.match_boundary import ball_exit_reason


def run_probe(
    *,
    asset_root: Path,
    output: Path,
    active: bool,
    duration: float,
    four_vs_four: bool = False,
    blue_kickoff: bool = False,
    contact_policy: OwnedBallContactPolicy | None = None,
    kickoff_offset_m: float = 0.0,
    pass_stroke_duration_sec: float = 0.0,
    bilateral_kick_options: bool = False,
    kick_observation_warmstart: bool = False,
    arrival_radius_m: float | None = None,
    anticipatory_contact: bool = False,
    kick_entry_frame: int | None = None,
    pass_swing_amplitude: float | None = None,
    forward_receiver_lane: bool = False,
    pass_reference_distance_m: float = 0.0,
    near_ball_policy: NearBallResidualPolicy | None = None,
    near_ball_seed: int = 0,
    near_ball_explore: bool = False,
    all_role_clearance: bool = False,
    basic_ball_play: bool = False,
    kickoff_role: str = "playmaker",
    contact_preferred_foot: str = "nearest",
    strike_ankle_lateral_m: float | None = None,
    receive_ankle_lateral_m: float | None = None,
    strict_receive_handoff: bool = False,
) -> dict[str, Any]:
    if type(strict_receive_handoff) is not bool or (strict_receive_handoff and not four_vs_four):
        raise ValueError("strict receive handoff requires the bilateral 4v4 world")
    if receive_ankle_lateral_m is not None and (
        not four_vs_four or not 0.10 <= receive_ankle_lateral_m <= 0.24
    ):
        raise ValueError("receive ankle ablation must remain in the bounded 4v4 course")
    if strike_ankle_lateral_m is not None and (
        not four_vs_four or not 0.08 <= strike_ankle_lateral_m <= 0.24
    ):
        raise ValueError("strike ankle ablation must remain in the bounded 4v4 course")
    if contact_preferred_foot not in {"nearest", "left", "right"} or (
        contact_preferred_foot != "nearest" and not four_vs_four
    ):
        raise ValueError("contact foot must be an explicit bilateral 4v4 ablation")
    if (
        type(basic_ball_play) is not bool
        or (basic_ball_play and not four_vs_four)
        or kickoff_role not in {"playmaker", "finisher", "defender", "goalkeeper"}
        or (kickoff_role != "playmaker" and not basic_ball_play)
    ):
        raise ValueError("role kickoff requires an explicit basic-ball-play 4v4 curriculum")
    if near_ball_policy is not None and not four_vs_four:
        raise ValueError("private residual actors require the symmetric 4v4 fixture")
    if four_vs_four and not active:
        raise ValueError("4v4 requires active role objectives")
    if pass_reference_distance_m != 0.0 and not bilateral_kick_options:
        raise ValueError("pass motor reference requires bilateral neural options")
    if not isinstance(forward_receiver_lane, bool) or (forward_receiver_lane and not four_vs_four):
        raise ValueError("forward receiver curriculum requires 4v4")
    if (
        any(v is not None for v in (kick_entry_frame, pass_swing_amplitude))
        and not bilateral_kick_options
    ):
        raise ValueError("kick parameters require bilateral neural options")
    if pass_stroke_duration_sec != 0.0 and not four_vs_four:
        raise ValueError("contact strokes require the 4v4 fixture")
    if not isinstance(anticipatory_contact, bool) or (anticipatory_contact and not four_vs_four):
        raise ValueError("anticipatory contact requires the bilateral 4v4 fixture")
    if not isinstance(bilateral_kick_options, bool) or (
        bilateral_kick_options and not four_vs_four
    ):
        raise ValueError("bilateral kick options require the 4v4 fixture")
    if not isinstance(kick_observation_warmstart, bool) or (
        kick_observation_warmstart and not bilateral_kick_options
    ):
        raise ValueError("kick warmstart requires bilateral kick options")
    if blue_kickoff and not four_vs_four:
        raise ValueError("mirrored kickoff requires the symmetric 4v4 fixture")
    if not -0.20 <= kickoff_offset_m <= 0.20 or (contact_policy is not None and not four_vs_four):
        raise ValueError("contact learning needs bounded 4v4 scenarios")
    if output.exists():
        raise FileExistsError(output)
    source_root = Path(__file__).parents[1]
    implementation = {
        str(p.relative_to(source_root)): hash_bytes(p.read_bytes())
        for relative in (
            "training/active_team_probe.py",
            "training/four_vs_four_match.py",
            "training/independent_team_growth.py",
            "growth/independent_agent_cell.py",
            "growth/role_self_model.py",
            "growth/pass_handoff.py",
            "growth/competitive_match_assessment.py",
            "growth/locomotion_contact_teacher.py",
            "growth/contact_stroke.py",
            "growth/near_ball_residual.py",
            "providers/g1/kick_warmstart.py",
            "growth/owned_ball_contact.py",
            "growth/pass_failure_feedback.py",
            "skills/team/independent_team_world.py",
            "world/field.py",
            "world/multi_player.py",
            "world/bilateral_net.py",
            "world/match_boundary.py",
            "world/player_clearance.py",
        )
        for p in (source_root / relative,)
    }
    fixture = (
        build_four_vs_four_fixture(
            asset_root, forward_receiver_lane=forward_receiver_lane, basic_ball_play=basic_ball_play
        )
        if four_vs_four
        else build_continuous_competitive_fixture(asset_root)
    )
    cells = tuple(
        replace(
            cell,
            tactical_profile=replace(
                cell.tactical_profile,
                active_competition=active,
                anticipatory_contact=anticipatory_contact,
            ),
        )
        for cell in fixture.cells
    )
    config = replace(default_continuous_match_config(), simulation_duration_sec=duration)
    config = replace(
        config,
        all_role_clearance=all_role_clearance,
        strict_receive_handoff=strict_receive_handoff,
    )
    if arrival_radius_m is not None:
        config = replace(config, arrival_radius_m=arrival_radius_m)
    if active:
        config = replace(
            config,
            minimum_player_separation_m=1.20,
            collision_avoidance_gain=3.0,
            duel_lateral_offset_m=0.45,
        )
    scenario = default_continuous_match_scenario()
    teacher = default_phase_strike_teacher()
    option: G1RollingOptionBridgeConfig | None = default_phase_strike_option()
    phase: StrikePhaseConfig | None = default_phase_strike_controller()
    if four_vs_four:
        config = replace(
            config,
            bilateral_goals=True,
            stationary_ball_acquisition=True,
            receive_pocket_depth_m=0.18,
            predictive_separation=True,
            stop_on_ball_exit=True,
            contact_possession_hold_sec=0.60,
            owned_contact_policy=contact_policy,
        )
        # The historical warm-start kick option only supports yaw zero.
        # Use the same world-frame foot-contact teacher on BOTH teams instead.
        option = None
        phase = None
        teacher = replace(
            teacher,
            receive_ankle_lateral_offset_m=0.12,
            receive_follow_through_speed_mps=0.35,
            pass_strike_foot_speed_mps=1.50,
            shot_strike_foot_speed_mps=2.50,
            one_touch_finish_aim_yaw_bias_rad=0.0,
            committed_receive_aim_yaw_bias_rad=0.0,
            pass_stroke_duration_sec=pass_stroke_duration_sec,
            preferred_foot=contact_preferred_foot,
        )
        scenario = replace(
            scenario,
            scenario_id="s199.s212.4v4.blue" if blue_kickoff else "s199.s212.4v4.red",
            ball_initial_position_m=(4.0, 1.20, 0.115) if blue_kickoff else (2.0, -1.20, 0.115),
        )
        if strike_ankle_lateral_m is not None:
            teacher = replace(teacher, ankle_lateral_offset_m=strike_ankle_lateral_m)
        if receive_ankle_lateral_m is not None:
            teacher = replace(
                teacher, committed_receive_ankle_lateral_offset_m=receive_ankle_lateral_m
            )
        x, y, z = scenario.ball_initial_position_m
        if forward_receiver_lane:
            x, y = (3.70, 1.22) if blue_kickoff else (2.30, -1.22)
        if kickoff_role != "playmaker":
            team = "blue" if blue_kickoff else "red"
            owner = next(
                player for player in fixture.players if player.agent_id == f"{team}.{kickoff_role}"
            )
            sign = -1.0 if blue_kickoff else 1.0
            x, y = owner.origin_m[0] + 0.55 * sign, owner.origin_m[1] - 0.12 * sign
        scenario = replace(
            scenario,
            ball_initial_position_m=(
                x,
                y + (-kickoff_offset_m if blue_kickoff else kickoff_offset_m),
                z,
            ),
        )
        if contact_policy is not None:
            teacher = replace(teacher, pass_strike_foot_speed_mps=contact_policy.pass_speed_mps)
        if bilateral_kick_options:
            option = replace(
                default_phase_strike_option(), pass_enabled=True, bilateral_enabled=True
            )
            option = replace(option, observation_warmstart=kick_observation_warmstart)
            option = replace(option, prospective_enabled=anticipatory_contact)
            option = replace(option, pass_reference_distance_m=pass_reference_distance_m)
            if kick_entry_frame is not None:
                option = replace(option, entry_policy_frame=kick_entry_frame)
            if pass_swing_amplitude is not None:
                option = replace(
                    option,
                    pass_parameters=replace(
                        option.pass_parameters, swing_amplitude=pass_swing_amplitude
                    ),
                )
            config = replace(config, contact_possession_hold_sec=1.50)
    results, trajectories = [], []
    for _ in range(2):
        result, trajectory = simulate_independent_team_world(
            asset_root=asset_root,
            roster=fixture.roster,
            cells=cells,
            players=fixture.players,
            scenario=scenario,
            goal=fixture.goal,
            config=config,
            contact_teacher_config=teacher,
            option_bridge_config=option,
            strike_phase_config=phase,
            near_ball_policy=near_ball_policy,
            near_ball_seed=near_ball_seed,
            near_ball_explore=near_ball_explore,
        )
        results.append(result.to_dict())
        trajectories.append(trajectory)
    rows = engagement_rows(trajectories[0], tuple(sorted(c.agent_id for c in cells)))
    exact = trajectory_digest(trajectories[0]) == trajectory_digest(trajectories[1])
    output.mkdir(parents=True)
    if near_ball_policy is not None:
        near_ball_policy.save(output / "residual-policy.npz")
    for name, trajectory in zip(("primary", "replay"), trajectories, strict=True):
        np.savez_compressed(output / f"{name}.npz", **trajectory)  # type: ignore[arg-type]
    report = {
        "schema_version": "rosclaw_soccer.active_team_probe.v1",
        "active_competition": active,
        "world_config": asdict(config),
        "contact_teacher_config": asdict(teacher),
        "option_config": None if option is None else asdict(option),
        "strike_phase_config": None if phase is None else asdict(phase),
        "scenario": asdict(scenario),
        "goal": asdict(fixture.goal),
        "cells": [c.to_dict() for c in cells],
        "players": [asdict(p) for p in fixture.players],
        "four_vs_four": four_vs_four,
        "forward_receiver_lane": forward_receiver_lane,
        "basic_ball_play": basic_ball_play,
        "kickoff_role": kickoff_role,
        "near_ball_residual": None
        if near_ball_policy is None
        else {
            "policy_hash": near_ball_policy.policy_hash,
            "artifact_hash": hash_bytes((output / "residual-policy.npz").read_bytes()),
            "seed": near_ball_seed,
            "explore": near_ball_explore,
            "agent_ids": list(near_ball_policy.agent_ids),
        },
        "fixture_hash": fixture.fixture_hash,
        "implementation": implementation,
        "results": results,
        "engagement": rows,
        "exact_replay": exact,
        "trajectory_digests": [trajectory_digest(t) for t in trajectories],
        "artifacts": {
            name: hash_bytes((output / name).read_bytes()) for name in ("primary.npz", "replay.npz")
        },
        "activation_ceiling": "SIM_ONLY",
        "promotion_eligible": False,
        "hardware_command_sent": False,
    }
    report["termination"] = {
        "requested_duration_sec": duration,
        "actual_duration_sec": float(trajectories[0]["time"][-1]),
        "reason": (
            ball_exit_reason(
                tuple(float(v) for v in trajectories[0]["ball_pose"][-1, :3]),
                left_x=config.left_goal_plane_x_m,
                right_x=fixture.goal.plane_x_m,
                radius=fixture.goal.ball_radius_m,
                goal_width=fixture.goal.width_m,
                goal_height=fixture.goal.height_m,
            )
            if config.stop_on_ball_exit
            else None
        )
        or "TIME_LIMIT",
        "automatic_restart_implemented": False,
    }
    report["assessment"] = assess_competitive_match_trajectory(
        trajectory=trajectories[0],
        trajectory_hash=trajectory_digest(trajectories[0]),
        agent_ids=tuple(sorted(c.agent_id for c in cells)),
        roles={c.agent_id: c.self_model.primary_role for c in cells},
        strict_replay=exact,
        world_safe=results[0]["safe"],
    ).to_dict()
    report["causal_pass_feedback"] = diagnose_passes(
        trajectories[0],
        tuple(sorted(c.agent_id for c in cells)),
        launch_relative=strict_receive_handoff,
    )
    report["pass_feedback_contract"] = (
        "launch_relative_foot_only_v1" if strict_receive_handoff else "intent_relative_v1"
    )
    report["report_hash"] = hash_json(report)
    (output / "probe.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def engagement_rows(trajectory: dict[str, Any], agent_ids: tuple[str, ...]) -> list[dict[str, Any]]:
    time = np.asarray(trajectory["time"])
    dt = np.diff(time)
    if len(time) < 2 or np.any(dt <= 0):
        raise ValueError("engagement needs strictly increasing physical time")
    rows = []
    for code, agent_id in enumerate(agent_ids, 1):
        key = agent_id.replace(".", "_")
        xy = np.asarray(trajectory[key + "_pelvis_pose"])[:, :2]
        steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        moving = steps / dt > 0.15
        idle = longest = 0.0
        for moving_now, elapsed in zip(moving, dt, strict=True):
            idle = 0.0 if moving_now else idle + float(elapsed)
            longest = max(longest, idle)
        contact = np.asarray(trajectory["ball_contact_agent_code"]) == code
        rows.append(
            {
                "agent_id": agent_id,
                "path_length_m": float(steps.sum()),
                "moving_fraction": float(np.sum(dt[moving]) / dt.sum()),
                "longest_stationary_sec": longest,
                "physical_ball_contact_frames": int(contact.sum()),
            }
        )
    return rows


def validate_probe(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text())
    if not isinstance(report, dict):
        raise ValueError("active team report must be an object")
    digest = report.pop("report_hash")
    if digest != hash_json(report):
        raise ValueError("active team report hash changed")
    if report.get("near_ball_residual") is not None:
        residual = report["near_ball_residual"]
        artifact = path.parent / "residual-policy.npz"
        policy = NearBallResidualPolicy.load(artifact)
        if (
            hash_bytes(artifact.read_bytes()) != residual["artifact_hash"]
            or policy.policy_hash != residual["policy_hash"]
            or list(policy.agent_ids) != residual["agent_ids"]
        ):
            raise ValueError("private residual policy identity changed")
    traces = []
    for name in ("primary.npz", "replay.npz"):
        source = path.parent / name
        if hash_bytes(source.read_bytes()) != report["artifacts"][name]:
            raise ValueError("active team trajectory file changed")
        with np.load(source, allow_pickle=False) as archive:
            traces.append({k: archive[k] for k in archive.files})
    digests = [trajectory_digest(t) for t in traces]
    ids = tuple(sorted(c["self_model"]["agent_id"] for c in report["cells"]))
    feedback_contract = report.get("pass_feedback_contract", "intent_relative_v1")
    if feedback_contract not in {"intent_relative_v1", "launch_relative_foot_only_v1"}:
        raise ValueError("unknown pass feedback contract")
    launch_relative = feedback_contract == "launch_relative_foot_only_v1"
    if launch_relative and (
        not report["world_config"].get("strict_receive_handoff")
        or not all(
            "pass_feedback_launch_relative" in t
            and np.shape(t["pass_feedback_launch_relative"]) == np.shape(t["time"])
            and np.asarray(t["pass_feedback_launch_relative"]).dtype == np.bool_
            and np.all(t["pass_feedback_launch_relative"])
            for t in traces
        )
    ):
        raise ValueError("pass feedback is not bound to its physical handoff contract")
    if "causal_pass_feedback" in report and report["causal_pass_feedback"] != diagnose_passes(
        traces[0], ids, launch_relative=launch_relative
    ):
        raise ValueError("causal pass evidence changed")
    if "assessment" in report:
        assessment = assess_competitive_match_trajectory(
            trajectory=traces[0],
            trajectory_hash=digests[0],
            agent_ids=ids,
            roles={
                c["self_model"]["agent_id"]: MatchRole(c["self_model"]["primary_role"])
                for c in report["cells"]
            },
            strict_replay=digests[0] == digests[1],
            world_safe=report["results"][0]["safe"],
        ).to_dict()
        if json.loads(json.dumps(assessment)) != report["assessment"]:
            raise ValueError("physical match assessment changed")
    if "termination" in report:
        config, goal = report["world_config"], report["goal"]
        reason = (
            ball_exit_reason(
                traces[0]["ball_pose"][-1, :3],
                left_x=config["left_goal_plane_x_m"],
                right_x=goal["plane_x_m"],
                radius=goal["ball_radius_m"],
                goal_width=goal["width_m"],
                goal_height=goal["height_m"],
            )
            if config.get("stop_on_ball_exit", False)
            else None
        ) or "TIME_LIMIT"
        if report["termination"] != {
            "requested_duration_sec": config["simulation_duration_sec"],
            "actual_duration_sec": float(traces[0]["time"][-1]),
            "reason": reason,
            "automatic_restart_implemented": False,
        }:
            raise ValueError("match termination differs from physics")
    if (
        report["schema_version"] != "rosclaw_soccer.active_team_probe.v1"
        or report["trajectory_digests"] != digests
        or report["exact_replay"] is not True
        or digests[0] != digests[1]
        or report["results"][0] != report["results"][1]
        or any(r["trajectory_hash"] != d for r, d in zip(report["results"], digests, strict=True))
        or report["engagement"] != engagement_rows(traces[0], ids)
        or report["activation_ceiling"] != "SIM_ONLY"
        or report["hardware_command_sent"] is not False
        or report["promotion_eligible"] is not False
    ):
        raise ValueError("active team evidence failed validation")
    report["report_hash"] = digest
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--active", action="store_true")
    parser.add_argument("--four-vs-four", action="store_true")
    parser.add_argument("--blue-kickoff", action="store_true")
    parser.add_argument("--contact-policy", type=Path)
    parser.add_argument("--kickoff-offset", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--pass-stroke-duration", type=float, default=0.0)
    parser.add_argument("--bilateral-kick-options", action="store_true")
    parser.add_argument("--kick-observation-warmstart", action="store_true")
    parser.add_argument("--anticipatory-contact", action="store_true")
    parser.add_argument("--arrival-radius", type=float)
    parser.add_argument("--kick-entry-frame", type=int)
    parser.add_argument("--pass-swing-amplitude", type=float)
    parser.add_argument("--forward-receiver-lane", action="store_true")
    parser.add_argument("--pass-reference-distance", type=float, default=0.0)
    parser.add_argument("--near-ball-policy", type=Path)
    parser.add_argument("--near-ball-seed", type=int, default=0)
    parser.add_argument("--near-ball-explore", action="store_true")
    parser.add_argument("--all-role-clearance", action="store_true")
    parser.add_argument("--basic-ball-play", action="store_true")
    parser.add_argument("--strict-receive-handoff", action="store_true")
    parser.add_argument("--strike-ankle-lateral", type=float)
    parser.add_argument("--receive-ankle-lateral", type=float)
    parser.add_argument(
        "--contact-preferred-foot", choices=("nearest", "left", "right"), default="nearest"
    )
    parser.add_argument(
        "--kickoff-role",
        choices=("playmaker", "finisher", "defender", "goalkeeper"),
        default="playmaker",
    )
    args = parser.parse_args()
    report = run_probe(
        asset_root=args.asset_root,
        output=args.output,
        active=args.active,
        duration=args.duration,
        four_vs_four=args.four_vs_four,
        blue_kickoff=args.blue_kickoff,
        contact_policy=None
        if args.contact_policy is None
        else OwnedBallContactPolicy(**json.loads(args.contact_policy.read_text())),
        kickoff_offset_m=args.kickoff_offset,
        pass_stroke_duration_sec=args.pass_stroke_duration,
        bilateral_kick_options=args.bilateral_kick_options,
        kick_observation_warmstart=args.kick_observation_warmstart,
        anticipatory_contact=args.anticipatory_contact,
        arrival_radius_m=args.arrival_radius,
        kick_entry_frame=args.kick_entry_frame,
        pass_swing_amplitude=args.pass_swing_amplitude,
        forward_receiver_lane=args.forward_receiver_lane,
        pass_reference_distance_m=args.pass_reference_distance,
        near_ball_policy=NearBallResidualPolicy.load(args.near_ball_policy)
        if args.near_ball_policy is not None
        else None,
        near_ball_seed=args.near_ball_seed,
        near_ball_explore=args.near_ball_explore,
        all_role_clearance=args.all_role_clearance,
        basic_ball_play=args.basic_ball_play,
        kickoff_role=args.kickoff_role,
        contact_preferred_foot=args.contact_preferred_foot,
        strike_ankle_lateral_m=args.strike_ankle_lateral,
        receive_ankle_lateral_m=args.receive_ankle_lateral,
        strict_receive_handoff=args.strict_receive_handoff,
    )
    print(
        json.dumps(
            {
                "engagement": report["engagement"],
                "exact_replay": report["exact_replay"],
                "safe": report["results"][0]["safe"],
                "peak_ball_speed_mps": report["results"][0]["peak_ball_speed_mps"],
                "assessment": report["assessment"],
            }
        )
    )


if __name__ == "__main__":
    main()
