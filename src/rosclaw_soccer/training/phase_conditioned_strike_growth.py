"""Strict-replay evidence loop for a phase-conditioned receive-to-strike chain."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.growth.locomotion_contact_teacher import (
    G1LocomotionContactTeacherConfig,
    G1RollingOptionBridgeConfig,
)
from rosclaw_soccer.growth.phase_conditioned_strike_assessment import (
    PhaseConditionedStrikeThresholds,
    assess_phase_conditioned_strike,
)
from rosclaw_soccer.growth.role_self_model import MatchRole
from rosclaw_soccer.growth.strike_phase_controller import StrikePhaseConfig
from rosclaw_soccer.providers.g1.asset_qualification import trajectory_digest
from rosclaw_soccer.sim.contracts import ShotParameters, hash_bytes, hash_json
from rosclaw_soccer.skills.team.independent_team_world import (
    IndependentTeamWorldConfig,
    IndependentTeamWorldScenario,
    simulate_independent_team_world,
)
from rosclaw_soccer.training.continuous_competitive_match_growth import (
    build_continuous_competitive_fixture,
    default_continuous_match_config,
    default_continuous_match_scenario,
)
from rosclaw_soccer.world.field import G1TrainingGoalSpec

_STATUS_PASS = "PASS_PHASE_CONDITIONED_STRIKE_SAVE"
_STATUS_REJECTED = "REJECTED_PHASE_CONDITIONED_STRIKE_SAVE"


def default_phase_strike_teacher() -> G1LocomotionContactTeacherConfig:
    """Preserve the qualified pass receive while removing forward follow-through."""

    return replace(
        G1LocomotionContactTeacherConfig(),
        committed_receive_follow_through_speed_mps=0.0,
    )


def default_phase_strike_option() -> G1RollingOptionBridgeConfig:
    """Bounded high shot option selected from physical failure replay."""

    return G1RollingOptionBridgeConfig(
        entry_policy_frame=235,
        exit_policy_frame=310,
        blend_frames=15,
        minimum_strike_stance_depth_m=0.30,
        maximum_strike_lateral_error_m=0.60,
        shoot_parameters=ShotParameters(
            swing_amplitude=1.0,
            foot_yaw_offset=0.12,
            foot_pitch_offset=0.08,
            loft_synergy=0.15,
            recovery_step_length=0.04,
            policy_type="parameter",
        ),
    )


def default_phase_strike_controller() -> StrikePhaseConfig:
    """Use prediction and failure-derived calibration without widening the goal."""

    return StrikePhaseConfig(
        target_stance_lateral_m=-0.30,
        maximum_yaw_rate_radps=0.80,
        strike_aim_lateral_bias_m=0.65,
    )


def run_phase_conditioned_strike_growth(
    *,
    evidence_dir: Path,
    asset_root: Path,
    thresholds: PhaseConditionedStrikeThresholds | None = None,
) -> dict[str, Any]:
    """Run, replay, assess, persist, and revalidate the S208 exam."""

    root = evidence_dir.expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("phase-conditioned strike evidence directory must be empty")
    root.mkdir(parents=True, exist_ok=True)
    fixture = build_continuous_competitive_fixture(asset_root)
    scenario = default_continuous_match_scenario()
    goal = replace(fixture.goal, target_y_m=0.80)
    world = replace(default_continuous_match_config(), simulation_duration_sec=8.60)
    teacher = default_phase_strike_teacher()
    option = default_phase_strike_option()
    phase = default_phase_strike_controller()
    active_thresholds = thresholds or PhaseConditionedStrikeThresholds()
    primary, primary_trace = simulate_independent_team_world(
        asset_root=asset_root,
        roster=fixture.roster,
        cells=fixture.cells,
        players=fixture.players,
        scenario=scenario,
        goal=goal,
        config=world,
        contact_teacher_config=teacher,
        option_bridge_config=option,
        strike_phase_config=phase,
    )
    replay, replay_trace = simulate_independent_team_world(
        asset_root=asset_root,
        roster=fixture.roster,
        cells=fixture.cells,
        players=fixture.players,
        scenario=scenario,
        goal=goal,
        config=world,
        contact_teacher_config=teacher,
        option_bridge_config=option,
        strike_phase_config=phase,
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
    primary_assessment = assess_phase_conditioned_strike(
        trajectory=primary_trace,
        trajectory_hash=str(primary_artifact["trajectory_digest"]),
        agent_ids=agent_ids,
        roles=roles,
        goal=goal,
        strict_replay=exact_replay,
        world_safe=primary.safe,
        thresholds=active_thresholds,
    )
    replay_assessment = assess_phase_conditioned_strike(
        trajectory=replay_trace,
        trajectory_hash=str(replay_artifact["trajectory_digest"]),
        agent_ids=agent_ids,
        roles=roles,
        goal=goal,
        strict_replay=exact_replay,
        world_safe=replay.safe,
        thresholds=active_thresholds,
    )
    primary_value = primary_assessment.to_dict()
    replay_value = replay_assessment.to_dict()
    passed = bool(
        exact_replay
        and primary.safe
        and replay.safe
        and primary_assessment.passed
        and primary_value == replay_value
    )
    implementation_files = _implementation_files()
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.phase_conditioned_strike_growth.v1",
        "status": _STATUS_PASS if passed else _STATUS_REJECTED,
        "passed": passed,
        "fixture_hash": fixture.fixture_hash,
        "agent_ids": list(agent_ids),
        "agent_roles": {agent_id: roles[agent_id].value for agent_id in agent_ids},
        "scenario": asdict(scenario),
        "scenario_hash": scenario.scenario_hash,
        "goal": asdict(goal),
        "goal_hash": goal.spec_hash,
        "world_config": asdict(world),
        "world_config_hash": world.config_hash,
        "teacher_config": asdict(teacher),
        "teacher_config_hash": teacher.config_hash,
        "option_config": asdict(option),
        "option_config_hash": option.config_hash,
        "phase_config": asdict(phase),
        "phase_config_hash": phase.config_hash,
        "thresholds": asdict(active_thresholds),
        "thresholds_hash": active_thresholds.config_hash,
        "primary_result": primary.to_dict(),
        "replay_result": replay.to_dict(),
        "primary_assessment": primary_value,
        "replay_assessment": replay_value,
        "primary_artifact": primary_artifact,
        "replay_artifact": replay_artifact,
        "exact_replay": exact_replay,
        "growth_claim": {
            "parent_rung": "S207_CONTINUOUS_MATCH_WITHOUT_SHOT",
            "new_physical_chain": [
                "TEAMMATE_FOOT_PASS",
                "FINISHER_FOOT_RECEIVE",
                "CAPTURE_ORIENT_PLANT_STRIKE_RECOVER_COMPLETE",
                "SUSTAINED_ON_TARGET_FOOT_SHOT",
                "OPPONENT_GOALKEEPER_GLOVE_SAVE",
            ],
            "goal_claimed": False,
            "why_goal_not_claimed": "the opponent goalkeeper physically saved the shot",
        },
        "evidence_boundary": {
            "physics_authority": "CPU_MUJOCO",
            "activation_ceiling": "SIM_ONLY",
            "whole_body_g1_count": 6,
            "one_rosclaw_cell_per_body": True,
            "root_or_ball_state_writes": False,
            "pixels_used_for_scoring": False,
            "hardware_command_sent": False,
        },
        "implementation_files": implementation_files,
        "implementation_hash": hash_json(implementation_files),
    }
    report["report_hash"] = hash_json(report)
    destination = root / "phase-conditioned-strike-exam.json"
    _atomic_json(destination, report)
    return validate_phase_conditioned_strike_growth(destination)


def validate_phase_conditioned_strike_growth(path: Path) -> dict[str, Any]:
    """Fail closed if evidence, replay, implementation, or authority changed."""

    resolved = path.expanduser().resolve()
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("phase-conditioned strike evidence must be an object")
    expected = value.pop("report_hash", None)
    try:
        trajectories: dict[str, dict[str, NDArray[Any]]] = {}
        for label in ("primary_artifact", "replay_artifact"):
            artifact = value.get(label)
            if not isinstance(artifact, dict):
                raise ValueError("phase-conditioned strike trajectory binding is absent")
            artifact_path = resolved.parent / str(artifact.get("file"))
            if not artifact_path.is_file():
                raise ValueError("phase-conditioned strike trajectory binding changed")
            trajectory = _load_trajectory(artifact_path)
            if hash_bytes(artifact_path.read_bytes()) != artifact.get(
                "file_hash"
            ) or trajectory_digest(trajectory) != artifact.get("trajectory_digest"):
                raise ValueError("phase-conditioned strike trajectory binding changed")
            trajectories[label] = trajectory
        primary = value.get("primary_assessment")
        replay = value.get("replay_assessment")
        boundary = value.get("evidence_boundary")
        files = value.get("implementation_files")
        implementation_files = _implementation_files()
        raw_agent_ids = value.get("agent_ids")
        raw_roles = value.get("agent_roles")
        if (
            not isinstance(raw_agent_ids, list)
            or not raw_agent_ids
            or any(not isinstance(item, str) for item in raw_agent_ids)
            or raw_agent_ids != sorted(set(raw_agent_ids))
            or not isinstance(raw_roles, dict)
            or set(raw_roles) != set(raw_agent_ids)
        ):
            raise ValueError("phase-conditioned strike roster binding is invalid")
        agent_ids = tuple(cast(list[str], raw_agent_ids))
        try:
            roles = {agent_id: MatchRole(str(raw_roles[agent_id])) for agent_id in agent_ids}
            scenario = IndependentTeamWorldScenario(**cast(dict[str, Any], value.get("scenario")))
            goal = G1TrainingGoalSpec(**cast(dict[str, Any], value.get("goal")))
            world = IndependentTeamWorldConfig(**cast(dict[str, Any], value.get("world_config")))
            teacher = G1LocomotionContactTeacherConfig(
                **cast(dict[str, Any], value.get("teacher_config"))
            )
            raw_option = dict(cast(dict[str, Any], value.get("option_config")))
            raw_option["pass_parameters"] = ShotParameters(
                **cast(dict[str, Any], raw_option.get("pass_parameters"))
            )
            raw_option["shoot_parameters"] = ShotParameters(
                **cast(dict[str, Any], raw_option.get("shoot_parameters"))
            )
            option = G1RollingOptionBridgeConfig(**raw_option)
            phase = StrikePhaseConfig(**cast(dict[str, Any], value.get("phase_config")))
            thresholds = PhaseConditionedStrikeThresholds(
                **cast(dict[str, Any], value.get("thresholds"))
            )
        except (TypeError, ValueError, KeyError) as error:
            raise ValueError("phase-conditioned strike assessment config is invalid") from error
        recomputed_primary = assess_phase_conditioned_strike(
            trajectory=trajectories["primary_artifact"],
            trajectory_hash=str(value["primary_artifact"]["trajectory_digest"]),
            agent_ids=agent_ids,
            roles=roles,
            goal=goal,
            strict_replay=value.get("exact_replay") is True,
            world_safe=value.get("primary_result", {}).get("safe") is True,
            thresholds=thresholds,
        ).to_dict()
        recomputed_replay = assess_phase_conditioned_strike(
            trajectory=trajectories["replay_artifact"],
            trajectory_hash=str(value["replay_artifact"]["trajectory_digest"]),
            agent_ids=agent_ids,
            roles=roles,
            goal=goal,
            strict_replay=value.get("exact_replay") is True,
            world_safe=value.get("replay_result", {}).get("safe") is True,
            thresholds=thresholds,
        ).to_dict()
        passed = bool(
            value.get("exact_replay") is True
            and value.get("primary_result", {}).get("safe") is True
            and value.get("replay_result", {}).get("safe") is True
            and isinstance(primary, dict)
            and primary.get("passed") is True
            and primary == replay
            and primary == recomputed_primary
            and replay == recomputed_replay
        )
        if (
            expected != hash_json(value)
            or value.get("schema_version") != "rosclaw_soccer.phase_conditioned_strike_growth.v1"
            or value.get("passed") is not passed
            or value.get("status") != (_STATUS_PASS if passed else _STATUS_REJECTED)
            or not isinstance(boundary, dict)
            or boundary.get("physics_authority") != "CPU_MUJOCO"
            or boundary.get("activation_ceiling") != "SIM_ONLY"
            or boundary.get("root_or_ball_state_writes") is not False
            or boundary.get("pixels_used_for_scoring") is not False
            or boundary.get("hardware_command_sent") is not False
            or boundary.get("whole_body_g1_count") != len(agent_ids)
            or boundary.get("one_rosclaw_cell_per_body") is not True
            or value.get("scenario_hash") != scenario.scenario_hash
            or value.get("goal_hash") != goal.spec_hash
            or value.get("world_config_hash") != world.config_hash
            or value.get("teacher_config_hash") != teacher.config_hash
            or value.get("option_config_hash") != option.config_hash
            or value.get("phase_config_hash") != phase.config_hash
            or value.get("thresholds_hash") != thresholds.config_hash
            or value.get("primary_result", {}).get("trajectory_hash")
            != value.get("primary_artifact", {}).get("trajectory_digest")
            or value.get("replay_result", {}).get("trajectory_hash")
            != value.get("replay_artifact", {}).get("trajectory_digest")
            or files != implementation_files
            or value.get("implementation_hash") != hash_json(implementation_files)
        ):
            raise ValueError("phase-conditioned strike integrity or authority is invalid")
    finally:
        if expected is not None:
            value["report_hash"] = expected
    return cast(dict[str, Any], value)


def _implementation_files() -> dict[str, str]:
    root = Path(__file__).parents[1]
    paths = {
        "runner": Path(__file__),
        "assessment": root / "growth/phase_conditioned_strike_assessment.py",
        "phase_controller": root / "growth/strike_phase_controller.py",
        "world": root / "skills/team/independent_team_world.py",
    }
    return {name: hash_bytes(path.read_bytes()) for name, path in paths.items()}


def _write_trajectory(path: Path, trajectory: dict[str, NDArray[Any]]) -> dict[str, str]:
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
    value = run_phase_conditioned_strike_growth(
        evidence_dir=arguments.evidence_dir,
        asset_root=arguments.asset_root,
    )
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "default_phase_strike_controller",
    "default_phase_strike_option",
    "default_phase_strike_teacher",
    "run_phase_conditioned_strike_growth",
    "validate_phase_conditioned_strike_growth",
]
