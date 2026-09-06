"""Strict-replay evidence loop for one continuous physical 3v3 match chain.

This is deliberately a hard qualification exam rather than a highlight
renderer.  Every credited skill is reconstructed from foot/glove contacts,
ball dynamics, and causal tactical commitments in one MuJoCo trajectory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.growth.competitive_match_assessment import (
    CompetitiveMatchThresholds,
    assess_competitive_match_trajectory,
)
from rosclaw_soccer.growth.independent_agent_cell import (
    RosclawSoccerAgentCell,
    build_independent_agent_cell,
)
from rosclaw_soccer.growth.locomotion_contact_teacher import (
    G1LocomotionContactTeacherConfig,
    G1RollingOptionBridgeConfig,
)
from rosclaw_soccer.growth.role_self_model import MatchRole, TeamRoleRoster
from rosclaw_soccer.providers.g1.asset_qualification import trajectory_digest
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.independent_team_world import (
    IndependentTeamWorldConfig,
    IndependentTeamWorldScenario,
    simulate_independent_team_world,
)
from rosclaw_soccer.training.independent_team_growth import (
    IndependentTeamFixture,
    build_independent_three_vs_three_fixture,
)
from rosclaw_soccer.world.multi_player import G1PitchPlayerSpec

_LAYOUT = (
    ("red.goalkeeper", "red", MatchRole.GOALKEEPER, (0.20, 0.00, 0.0), "", 0.0),
    (
        "red.playmaker",
        "red",
        MatchRole.PLAYMAKER,
        (1.70, -0.70, 0.0),
        "red_playmaker_",
        0.0,
    ),
    (
        "red.finisher",
        "red",
        MatchRole.FINISHER,
        (3.00, -0.25, 0.0),
        "red_finisher_",
        0.0,
    ),
    (
        "blue.goalkeeper",
        "blue",
        MatchRole.GOALKEEPER,
        (7.10, 0.00, 0.0),
        "blue_goalkeeper_",
        math.pi,
    ),
    (
        "blue.playmaker",
        "blue",
        MatchRole.PLAYMAKER,
        (4.80, -1.50, 0.0),
        "blue_playmaker_",
        math.pi,
    ),
    (
        "blue.finisher",
        "blue",
        MatchRole.FINISHER,
        (3.80, 1.40, 0.0),
        "blue_finisher_",
        math.pi,
    ),
)


def build_continuous_competitive_fixture(asset_root: Path) -> IndependentTeamFixture:
    """Place six private ROSClaw cells in a collision-safe transition exam."""

    foundation = build_independent_three_vs_three_fixture(asset_root)
    team_ids = {
        team_id: tuple(agent_id for agent_id, team, *_ in _LAYOUT if team == team_id)
        for team_id in ("red", "blue")
    }
    cells: tuple[RosclawSoccerAgentCell, ...] = tuple(
        build_independent_agent_cell(
            agent_id=agent_id,
            team_id=team_id,
            primary_role=role,
            teammate_ids=tuple(value for value in team_ids[team_id] if value != agent_id),
            opponent_ids=team_ids["blue" if team_id == "red" else "red"],
            body_hash=foundation.cells[0].growth_scope.body_hash,
            foundation_policy_hash=foundation.foundation_policy_hash,
            home_position_m=origin,
        )
        for agent_id, team_id, role, origin, _, _ in _LAYOUT
    )
    players = tuple(
        G1PitchPlayerSpec(
            agent_id=agent_id,
            body_prefix=prefix,
            origin_m=origin,
            yaw_rad=yaw,
            goalkeeper_gloves=role is MatchRole.GOALKEEPER,
        )
        for agent_id, _, role, origin, prefix, yaw in _LAYOUT
    )
    return IndependentTeamFixture(
        roster=TeamRoleRoster(
            "s207.continuous.competitive.3v3", tuple(c.self_model for c in cells)
        ),
        cells=cells,
        players=players,
        goal=foundation.goal,
        foundation_policy_hash=foundation.foundation_policy_hash,
    )


def default_continuous_match_config() -> IndependentTeamWorldConfig:
    return IndependentTeamWorldConfig(
        simulation_duration_sec=10.0,
        maximum_speed_mps=0.70,
        goalkeeper_maximum_speed_mps=0.50,
        maximum_acceleration_mps2=2.0,
        position_gain=1.60,
        minimum_player_separation_m=1.10,
        collision_avoidance_gain=2.0,
        maximum_collision_correction_mps=0.30,
        contact_possession_hold_sec=0.20,
        receive_run_onto_horizon_sec=3.0,
    )


def default_continuous_match_scenario() -> IndependentTeamWorldScenario:
    return IndependentTeamWorldScenario(
        scenario_id="s199.s207.continuous-competitive-match",
        ball_initial_position_m=(1.92, -0.80, 0.115),
        ball_initial_velocity_mps=(0.0, 0.0, 0.0),
        seed=207_200,
    )


def run_continuous_competitive_match_growth(
    *,
    evidence_dir: Path,
    asset_root: Path,
    world_config: IndependentTeamWorldConfig | None = None,
    teacher_config: G1LocomotionContactTeacherConfig | None = None,
    option_config: G1RollingOptionBridgeConfig | None = None,
    thresholds: CompetitiveMatchThresholds | None = None,
) -> dict[str, Any]:
    """Run, replay, assess, persist, and revalidate one hard match exam."""

    root = evidence_dir.expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("continuous-match evidence directory must be empty")
    root.mkdir(parents=True, exist_ok=True)
    fixture = build_continuous_competitive_fixture(asset_root)
    scenario = default_continuous_match_scenario()
    world = world_config or default_continuous_match_config()
    teacher = teacher_config or G1LocomotionContactTeacherConfig()
    option = option_config or G1RollingOptionBridgeConfig(
        minimum_strike_stance_depth_m=0.30,
        maximum_strike_lateral_error_m=0.60,
    )
    active_thresholds = thresholds or CompetitiveMatchThresholds()
    primary, primary_trace = simulate_independent_team_world(
        asset_root=asset_root,
        roster=fixture.roster,
        cells=fixture.cells,
        players=fixture.players,
        scenario=scenario,
        goal=fixture.goal,
        config=world,
        contact_teacher_config=teacher,
        option_bridge_config=option,
    )
    replay, replay_trace = simulate_independent_team_world(
        asset_root=asset_root,
        roster=fixture.roster,
        cells=fixture.cells,
        players=fixture.players,
        scenario=scenario,
        goal=fixture.goal,
        config=world,
        contact_teacher_config=teacher,
        option_bridge_config=option,
    )
    primary_artifact = _write_trajectory(root / "primary.npz", primary_trace)
    replay_artifact = _write_trajectory(root / "replay.npz", replay_trace)
    exact_replay = bool(
        primary.to_dict() == replay.to_dict()
        and primary.trajectory_hash == replay.trajectory_hash
        and primary_artifact["trajectory_digest"] == replay_artifact["trajectory_digest"]
    )
    agent_ids = tuple(sorted(agent.agent_id for agent in fixture.roster.agents))
    roles = {agent.agent_id: agent.primary_role for agent in fixture.roster.agents}
    primary_assessment = assess_competitive_match_trajectory(
        trajectory=primary_trace,
        trajectory_hash=str(primary_artifact["trajectory_digest"]),
        agent_ids=agent_ids,
        roles=roles,
        strict_replay=exact_replay,
        world_safe=primary.safe,
        thresholds=active_thresholds,
    )
    replay_assessment = assess_competitive_match_trajectory(
        trajectory=replay_trace,
        trajectory_hash=str(replay_artifact["trajectory_digest"]),
        agent_ids=agent_ids,
        roles=roles,
        strict_replay=exact_replay,
        world_safe=replay.safe,
        thresholds=active_thresholds,
    )
    passed = bool(
        exact_replay
        and primary.passed
        and replay.passed
        and primary_assessment.passed
        and primary_assessment.to_dict() == replay_assessment.to_dict()
    )
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.continuous_competitive_match_growth.v1",
        "status": "PASS_CONTINUOUS_MATCH_CHAIN" if passed else "REJECTED_CONTINUOUS_MATCH_CHAIN",
        "passed": passed,
        "fixture_hash": fixture.fixture_hash,
        "scenario": asdict(scenario),
        "scenario_hash": scenario.scenario_hash,
        "world_config": asdict(world),
        "world_config_hash": world.config_hash,
        "teacher_config": asdict(teacher),
        "teacher_config_hash": teacher.config_hash,
        "option_config": asdict(option),
        "option_config_hash": option.config_hash,
        "thresholds": asdict(active_thresholds),
        "thresholds_hash": active_thresholds.config_hash,
        "primary_result": primary.to_dict(),
        "replay_result": replay.to_dict(),
        "primary_assessment": primary_assessment.to_dict(),
        "replay_assessment": replay_assessment.to_dict(),
        "primary_artifact": primary_artifact,
        "replay_artifact": replay_artifact,
        "exact_replay": exact_replay,
        "evidence_boundary": {
            "physics_authority": "CPU_MUJOCO",
            "activation_ceiling": "SIM_ONLY",
            "whole_body_g1_count": 6,
            "one_rosclaw_cell_per_body": True,
            "root_or_ball_state_writes": False,
            "pixels_used_for_scoring": False,
            "hardware_command_sent": False,
        },
        "implementation_hash": hash_json(
            {
                "runner": hash_bytes(Path(__file__).read_bytes()),
                "assessment": hash_bytes(
                    (
                        Path(__file__).parents[1] / "growth/competitive_match_assessment.py"
                    ).read_bytes()
                ),
                "teacher": hash_bytes(
                    (
                        Path(__file__).parents[1] / "growth/locomotion_contact_teacher.py"
                    ).read_bytes()
                ),
                "cells": hash_bytes(
                    (Path(__file__).parents[1] / "growth/independent_agent_cell.py").read_bytes()
                ),
                "world": hash_bytes(
                    (
                        Path(__file__).parents[1] / "skills/team/independent_team_world.py"
                    ).read_bytes()
                ),
            }
        ),
    }
    report["report_hash"] = hash_json(report)
    destination = root / "continuous-match-exam.json"
    _atomic_json(destination, report)
    return validate_continuous_competitive_match_growth(destination)


def validate_continuous_competitive_match_growth(path: Path) -> dict[str, Any]:
    """Fail closed if a report, trajectory, replay, or authority bound changed."""

    resolved = path.expanduser().resolve()
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("continuous-match evidence must be an object")
    expected = value.pop("report_hash", None)
    try:
        for label in ("primary_artifact", "replay_artifact"):
            artifact = value.get(label)
            if not isinstance(artifact, dict):
                raise ValueError("continuous-match trajectory binding is absent")
            artifact_path = resolved.parent / str(artifact.get("file"))
            if (
                not artifact_path.is_file()
                or hash_bytes(artifact_path.read_bytes()) != artifact.get("file_hash")
                or trajectory_digest(_load_trajectory(artifact_path))
                != artifact.get("trajectory_digest")
            ):
                raise ValueError("continuous-match trajectory binding changed")
        assessment = value.get("primary_assessment")
        boundary = value.get("evidence_boundary")
        passed = bool(
            value.get("exact_replay") is True
            and value.get("primary_result", {}).get("passed") is True
            and value.get("replay_result", {}).get("passed") is True
            and isinstance(assessment, dict)
            and assessment.get("passed") is True
            and assessment == value.get("replay_assessment")
        )
        if (
            expected != hash_json(value)
            or value.get("schema_version")
            != "rosclaw_soccer.continuous_competitive_match_growth.v1"
            or value.get("passed") is not passed
            or value.get("status")
            != ("PASS_CONTINUOUS_MATCH_CHAIN" if passed else "REJECTED_CONTINUOUS_MATCH_CHAIN")
            or not isinstance(boundary, dict)
            or boundary.get("activation_ceiling") != "SIM_ONLY"
            or boundary.get("hardware_command_sent") is not False
            or boundary.get("root_or_ball_state_writes") is not False
        ):
            raise ValueError("continuous-match integrity or authority contract is invalid")
    finally:
        if expected is not None:
            value["report_hash"] = expected
    return cast(dict[str, Any], value)


def _write_trajectory(path: Path, trajectory: dict[str, NDArray[Any]]) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as descriptor:
        temporary = Path(descriptor.name)
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **trajectory)  # type: ignore[arg-type]
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "file": path.name,
        "file_hash": hash_bytes(path.read_bytes()),
        "trajectory_digest": trajectory_digest(trajectory),
    }


def _load_trajectory(path: Path) -> dict[str, NDArray[Any]]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".json", mode="w", encoding="utf-8", delete=False
    ) as descriptor:
        temporary = Path(descriptor.name)
        try:
            json.dump(value, descriptor, ensure_ascii=False, indent=2, sort_keys=True)
            descriptor.write("\n")
            descriptor.flush()
            os.fsync(descriptor.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    value = run_continuous_competitive_match_growth(
        evidence_dir=arguments.evidence_dir,
        asset_root=arguments.asset_root,
    )
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "build_continuous_competitive_fixture",
    "default_continuous_match_config",
    "default_continuous_match_scenario",
    "run_continuous_competitive_match_growth",
    "validate_continuous_competitive_match_growth",
]
