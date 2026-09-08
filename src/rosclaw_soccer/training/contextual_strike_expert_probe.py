"""Run one S210 contextual-expert strike probe in CPU MuJoCo."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.growth.contextual_strike_experts import (
    ContextualStrikeExpertMemory,
    StrikeTaskContext,
)
from rosclaw_soccer.growth.dynamic_strike_coordination import (
    StrikeCoordinationObservation,
)
from rosclaw_soccer.growth.phase_conditioned_strike_assessment import (
    assess_phase_conditioned_strike,
)
from rosclaw_soccer.growth.role_self_model import MatchRole
from rosclaw_soccer.providers.g1.asset_qualification import trajectory_digest
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.independent_team_world import (
    IndependentTeamWorldScenario,
    simulate_independent_team_world,
)
from rosclaw_soccer.training.contextual_strike_expert_learning import (
    load_contextual_strike_expert_memory,
)
from rosclaw_soccer.training.continuous_competitive_match_growth import (
    build_continuous_competitive_fixture,
    default_continuous_match_config,
    default_continuous_match_scenario,
)
from rosclaw_soccer.training.dynamic_strike_coordination_probe import _project_shot
from rosclaw_soccer.training.phase_conditioned_strike_growth import (
    default_phase_strike_controller,
    default_phase_strike_option,
    default_phase_strike_teacher,
)
from rosclaw_soccer.world.field import G1TrainingGoalSpec

_SELECTION_CODES = {"verified-expert": 1, "negative-memory": 2, "out-of-support": 3}


def run_contextual_strike_expert_probe(
    *,
    asset_root: Path,
    memory_artifact_dir: Path,
    scenario: IndependentTeamWorldScenario | None = None,
    goal_target_y_m: float = 0.80,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Evaluate one data-bound memory without granting promotion."""

    memory_root = memory_artifact_dir.expanduser().resolve()
    memory, training = load_contextual_strike_expert_memory(memory_root)
    fixture = build_continuous_competitive_fixture(asset_root)
    active_scenario = scenario or default_continuous_match_scenario()
    goal = replace(fixture.goal, target_y_m=goal_target_y_m)
    result, trajectory = simulate_independent_team_world(
        asset_root=asset_root,
        roster=fixture.roster,
        cells=fixture.cells,
        players=fixture.players,
        scenario=active_scenario,
        goal=goal,
        config=replace(default_continuous_match_config(), simulation_duration_sec=8.60),
        contact_teacher_config=default_phase_strike_teacher(),
        option_bridge_config=default_phase_strike_option(),
        strike_phase_config=default_phase_strike_controller(),
        contextual_strike_memory=memory,
    )
    digest = trajectory_digest(trajectory)
    agent_ids = tuple(sorted(agent.agent_id for agent in fixture.roster.agents))
    roles = {agent.agent_id: agent.primary_role for agent in fixture.roster.agents}
    assessment = assess_phase_conditioned_strike(
        trajectory=trajectory,
        trajectory_hash=digest,
        agent_ids=agent_ids,
        roles=roles,
        goal=goal,
        strict_replay=False,
        world_safe=result.safe,
    ).to_dict()
    shot_projection = _projection(trajectory, assessment=assessment, goal=goal)
    route_summary = _route_summary(trajectory)
    assessment_gates = cast(dict[str, bool], assessment["gates"])
    candidate_success = bool(
        result.safe
        and assessment.get("phase_sequence") == [1, 2, 3, 4, 5, 6]
        and shot_projection["whole_ball_inside_goal"] is True
        and assessment_gates.get("physical_teammate_pass_received") is True
        and assessment_gates.get("physical_foot_strike_in_strike_phase") is True
        and assessment_gates.get("stable_recovery_completed") is True
        and route_summary["selected_frames"] > 0
    )
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.contextual_strike_expert_probe.v1",
        "memory_artifact_path": str(memory_root),
        "memory_training_report_hash": training["report_hash"],
        "memory_training_file_hash": hash_bytes((memory_root / "training.json").read_bytes()),
        "memory_hash": memory.memory_hash,
        "dataset_snapshot_hash": memory.dataset_snapshot_hash,
        "agent_ids": list(agent_ids),
        "agent_roles": {agent_id: roles[agent_id].value for agent_id in agent_ids},
        "scenario": active_scenario.scenario_id,
        "scenario_spec": asdict(active_scenario),
        "scenario_hash": active_scenario.scenario_hash,
        "goal_spec": asdict(goal),
        "world_result": result.to_dict(),
        "world_safe": result.safe,
        "phase_timeline": _phase_timeline(trajectory),
        "route_summary": route_summary,
        "shot_projection": shot_projection,
        "assessment": assessment,
        "candidate_success": candidate_success,
        "trajectory_digest": digest,
        "activation_ceiling": "SIM_ONLY",
        "promotion_eligible": False,
        "hardware_command_sent": False,
    }
    if output_dir is not None:
        root = output_dir.expanduser().resolve()
        checkout = Path(__file__).parents[3]
        if root == checkout or checkout in root.parents:
            raise ValueError("contextual strike probe evidence must remain outside checkout")
        if root.exists():
            raise FileExistsError("contextual strike probe output already exists")
        root.mkdir(parents=True)
        trajectory_path = root / "trajectory.npz"
        _write_trajectory(trajectory_path, trajectory)
        report["trajectory_artifact"] = {
            "file": trajectory_path.name,
            "file_hash": hash_bytes(trajectory_path.read_bytes()),
            "trajectory_digest": digest,
        }
        report["report_hash"] = hash_json(report)
        _atomic_json(root / "probe.json", report)
    return report


def validate_contextual_strike_expert_probe(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    value = _read_json(resolved)
    declared = value.pop("report_hash", None)
    try:
        artifact = value.get("trajectory_artifact")
        if not isinstance(artifact, dict) or artifact.get("file") != "trajectory.npz":
            raise ValueError("contextual strike trajectory binding is absent")
        trajectory_path = resolved.parent / "trajectory.npz"
        if not trajectory_path.is_file() or hash_bytes(
            trajectory_path.read_bytes()
        ) != artifact.get("file_hash"):
            raise ValueError("contextual strike trajectory file changed")
        with np.load(trajectory_path, allow_pickle=False) as archive:
            trajectory = {name: np.asarray(archive[name]) for name in archive.files}
        digest = trajectory_digest(trajectory)
        memory_root = Path(str(value.get("memory_artifact_path"))).expanduser().resolve()
        memory, training = load_contextual_strike_expert_memory(memory_root)
        scenario = IndependentTeamWorldScenario(**cast(dict[str, Any], value["scenario_spec"]))
        goal = G1TrainingGoalSpec(**cast(dict[str, Any], value["goal_spec"]))
        agent_ids = tuple(str(item) for item in cast(list[Any], value["agent_ids"]))
        raw_roles = cast(dict[str, str], value["agent_roles"])
        roles = {agent_id: MatchRole(raw_roles[agent_id]) for agent_id in agent_ids}
        world = cast(dict[str, Any], value["world_result"])
        assessment = assess_phase_conditioned_strike(
            trajectory=trajectory,
            trajectory_hash=digest,
            agent_ids=agent_ids,
            roles=roles,
            goal=goal,
            strict_replay=False,
            world_safe=bool(world.get("safe")),
        ).to_dict()
        shot_projection = _projection(trajectory, assessment=assessment, goal=goal)
        route_summary = _route_summary(trajectory)
        _validate_route_trace(memory, trajectory)
        assessment_gates = cast(dict[str, bool], assessment["gates"])
        success = bool(
            world.get("safe") is True
            and assessment.get("phase_sequence") == [1, 2, 3, 4, 5, 6]
            and shot_projection["whole_ball_inside_goal"] is True
            and assessment_gates.get("physical_teammate_pass_received") is True
            and assessment_gates.get("physical_foot_strike_in_strike_phase") is True
            and assessment_gates.get("stable_recovery_completed") is True
            and route_summary["selected_frames"] > 0
        )
        if (
            declared != hash_json(value)
            or value.get("schema_version") != "rosclaw_soccer.contextual_strike_expert_probe.v1"
            or digest != artifact.get("trajectory_digest")
            or digest != value.get("trajectory_digest")
            or value.get("memory_training_report_hash") != training.get("report_hash")
            or value.get("memory_training_file_hash")
            != hash_bytes((memory_root / "training.json").read_bytes())
            or value.get("memory_hash") != memory.memory_hash
            or value.get("dataset_snapshot_hash") != memory.dataset_snapshot_hash
            or value.get("agent_ids") != sorted(set(value.get("agent_ids", [])))
            or set(raw_roles) != set(agent_ids)
            or value.get("scenario") != scenario.scenario_id
            or value.get("scenario_hash") != scenario.scenario_hash
            or world.get("trajectory_hash") != digest
            or world.get("scenario_hash") != scenario.scenario_hash
            or world.get("physics_authority") != "CPU_MUJOCO"
            or world.get("hardware_command_sent") is not False
            or value.get("world_safe") != world.get("safe")
            or value.get("phase_timeline") != _phase_timeline(trajectory)
            or value.get("route_summary") != route_summary
            or value.get("shot_projection") != shot_projection
            or value.get("assessment") != assessment
            or value.get("candidate_success") is not success
            or value.get("activation_ceiling") != "SIM_ONLY"
            or value.get("promotion_eligible") is not False
            or value.get("hardware_command_sent") is not False
        ):
            raise ValueError("contextual strike probe integrity or authority changed")
    finally:
        value["report_hash"] = declared
    return value


def _validate_route_trace(
    memory: ContextualStrikeExpertMemory, trajectory: dict[str, NDArray[Any]]
) -> None:
    consulted = np.asarray(trajectory["strike_context_memory_consulted"], dtype=np.bool_)
    contexts = np.asarray(trajectory["strike_context_observation"], dtype=np.float64)
    observations = np.asarray(trajectory["strike_coordination_observation"], dtype=np.float64)
    indices = np.asarray(trajectory["strike_context_expert_index"], dtype=np.int64)
    distances = np.asarray(trajectory["strike_context_normalized_distance"], dtype=np.float64)
    codes = np.asarray(trajectory["strike_context_selection_code"], dtype=np.int64)
    abstained = np.asarray(trajectory["strike_context_abstained"], dtype=np.bool_)
    active = np.asarray(trajectory["strike_coordination_actor_active"], dtype=np.bool_)
    stance = np.asarray(trajectory["strike_coordination_stance_blend"], dtype=np.float64)
    yaw = np.asarray(trajectory["strike_coordination_goal_yaw_blend"], dtype=np.float64)
    for frame in np.flatnonzero(consulted):
        decision = memory.select(
            context=StrikeTaskContext(*contexts[frame]),
            observation=StrikeCoordinationObservation(*observations[frame]),
        )
        expected_code = _SELECTION_CODES[decision.reason]
        if (
            indices[frame] != decision.expert_index
            or not math.isclose(distances[frame], decision.normalized_distance, abs_tol=1.0e-10)
            or codes[frame] != expected_code
        ):
            raise ValueError("contextual strike routing trace changed")
        expected_abstained = decision.action is None
        if (
            bool(abstained[frame]) is not expected_abstained
            or bool(active[frame]) is expected_abstained
        ):
            raise ValueError("contextual strike authority trace changed")
        if decision.action is not None and (
            not math.isclose(stance[frame], decision.action.stance_blend, abs_tol=1.0e-10)
            or not math.isclose(yaw[frame], decision.action.goal_yaw_blend, abs_tol=1.0e-10)
        ):
            raise ValueError("contextual strike expert action trace changed")
    for frame in np.flatnonzero(active):
        expert_index = int(indices[frame])
        if not 0 <= expert_index < len(memory.experts) or codes[frame] != 1:
            raise ValueError("contextual strike active expert index changed")
        action = memory.experts[expert_index].actor.act(
            StrikeCoordinationObservation(*observations[frame])
        )
        if (
            bool(abstained[frame])
            or not math.isclose(stance[frame], action.stance_blend, abs_tol=1.0e-10)
            or not math.isclose(yaw[frame], action.goal_yaw_blend, abs_tol=1.0e-10)
        ):
            raise ValueError("contextual strike latched expert action changed")


def _projection(
    trajectory: dict[str, NDArray[Any]],
    *,
    assessment: dict[str, Any],
    goal: G1TrainingGoalSpec,
) -> dict[str, float | int | bool | str | None]:
    y_value, z_value, error, sample_frame = _project_shot(
        trajectory=trajectory,
        strike_frame=int(cast(dict[str, Any], assessment["events"])["strike_frame"]),
        goal_plane_x_m=goal.plane_x_m,
        target_y_m=goal.target_y_m,
        target_z_m=goal.target_z_m,
    )
    return {
        "method": "peak_forward_velocity_within_120ms_ballistic_v1",
        "sample_frame": sample_frame,
        "goal_plane_y_m": y_value,
        "goal_plane_z_m": z_value,
        "target_error_m": error,
        "whole_ball_inside_goal": bool(
            y_value is not None
            and z_value is not None
            and abs(y_value) <= goal.width_m / 2.0 - goal.ball_radius_m
            and goal.ball_radius_m <= z_value <= goal.height_m - goal.ball_radius_m
        ),
    }


def _phase_timeline(trajectory: dict[str, NDArray[Any]]) -> list[dict[str, float | int]]:
    phase_codes = np.asarray(trajectory["strike_phase_code"], dtype=np.int64)
    changes = np.flatnonzero(np.r_[True, phase_codes[1:] != phase_codes[:-1]])
    return [
        {
            "frame": int(frame),
            "time_sec": float(trajectory["time"][frame]),
            "code": int(phase_codes[frame]),
        }
        for frame in changes
        if phase_codes[frame] > 0
    ]


def _route_summary(trajectory: dict[str, NDArray[Any]]) -> dict[str, Any]:
    consulted = np.asarray(trajectory["strike_context_memory_consulted"], dtype=np.bool_)
    active = np.asarray(trajectory["strike_coordination_actor_active"], dtype=np.bool_)
    abstained = np.asarray(trajectory["strike_context_abstained"], dtype=np.bool_)
    indices = np.asarray(trajectory["strike_context_expert_index"], dtype=np.int64)
    distances = np.asarray(trajectory["strike_context_normalized_distance"], dtype=np.float64)
    selected = indices[active]
    return {
        "consulted_frames": int(np.sum(consulted)),
        "selected_frames": int(np.sum(active)),
        "abstained_frames": int(np.sum(abstained)),
        "selected_expert_indices": sorted(set(int(value) for value in selected)),
        "maximum_selected_distance": (
            None if not np.any(active) else float(np.max(distances[active]))
        ),
    }


def _write_trajectory(path: Path, trajectory: dict[str, NDArray[Any]]) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with temporary.open("wb") as output:
            np.savez_compressed(output, **trajectory)  # type: ignore[arg-type]
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".json", mode="w", encoding="utf-8", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--memory-artifact", required=True, type=Path)
    parser.add_argument("--goal-target-y", required=True, type=float)
    parser.add_argument("--ball-y-offset", default=0.0, type=float)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    if not -0.25 <= arguments.ball_y_offset <= 0.25:
        raise ValueError("ball y offset must remain inside [-0.25, 0.25] m")
    if not -1.20 <= arguments.goal_target_y <= 1.20:
        raise ValueError("goal target y must remain inside [-1.20, 1.20] m")
    default = default_continuous_match_scenario()
    seed = 210_000 + round(1000 * arguments.goal_target_y) + round(10_000 * arguments.ball_y_offset)
    report = run_contextual_strike_expert_probe(
        asset_root=arguments.asset_root,
        memory_artifact_dir=arguments.memory_artifact,
        goal_target_y_m=arguments.goal_target_y,
        scenario=replace(
            default,
            scenario_id=(
                f"s199.s210.contextual-strike.y{arguments.goal_target_y:+.3f}."
                f"ball{arguments.ball_y_offset:+.3f}.seed{seed}"
            ),
            ball_initial_position_m=(
                default.ball_initial_position_m[0],
                default.ball_initial_position_m[1] + arguments.ball_y_offset,
                default.ball_initial_position_m[2],
            ),
            seed=seed,
        ),
        output_dir=arguments.output_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "run_contextual_strike_expert_probe",
    "validate_contextual_strike_expert_probe",
]
