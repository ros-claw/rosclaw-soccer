"""Run one non-promotional S209 dynamic strike-coordination physics probe."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.growth.dynamic_strike_coordination import (
    DynamicStrikeCoordinationActor,
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
from rosclaw_soccer.training.continuous_competitive_match_growth import (
    build_continuous_competitive_fixture,
    default_continuous_match_config,
    default_continuous_match_scenario,
)
from rosclaw_soccer.training.phase_conditioned_strike_growth import (
    default_phase_strike_controller,
    default_phase_strike_option,
    default_phase_strike_teacher,
)
from rosclaw_soccer.world.field import G1TrainingGoalSpec


def run_dynamic_strike_coordination_probe(
    *,
    asset_root: Path,
    actor: DynamicStrikeCoordinationActor,
    scenario: IndependentTeamWorldScenario | None = None,
    goal_target_y_m: float = 0.80,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Evaluate one parameter or learned actor without granting promotion."""

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
        strike_coordination_actor=actor,
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
    gates = dict(assessment["gates"])
    gates.pop("strict_replay", None)
    strike_frame = int(assessment["events"]["strike_frame"])
    projected_y, projected_z, target_error, projection_frame = _project_shot(
        trajectory=trajectory,
        strike_frame=strike_frame,
        goal_plane_x_m=goal.plane_x_m,
        target_y_m=goal.target_y_m,
        target_z_m=goal.target_z_m,
    )
    whole_ball_inside = bool(
        projected_y is not None
        and projected_z is not None
        and abs(projected_y) <= goal.width_m / 2.0 - goal.ball_radius_m
        and goal.ball_radius_m <= projected_z <= goal.height_m - goal.ball_radius_m
    )
    phase_codes = np.asarray(trajectory["strike_phase_code"], dtype=np.int64)
    phase_changes = np.flatnonzero(np.r_[True, phase_codes[1:] != phase_codes[:-1]])
    phase_timeline = [
        {
            "frame": int(frame),
            "time_sec": float(trajectory["time"][frame]),
            "code": int(phase_codes[frame]),
        }
        for frame in phase_changes
        if phase_codes[frame] > 0
    ]
    actor_active = np.asarray(trajectory["strike_coordination_actor_active"], dtype=np.bool_)
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.dynamic_strike_coordination_probe.v1",
        "actor_hash": actor.actor_hash,
        "actor_policy_type": actor.policy_type,
        "actor": asdict(actor),
        "agent_ids": list(agent_ids),
        "agent_roles": {agent_id: roles[agent_id].value for agent_id in agent_ids},
        "scenario": active_scenario.scenario_id,
        "scenario_spec": asdict(active_scenario),
        "scenario_hash": active_scenario.scenario_hash,
        "goal_spec": asdict(goal),
        "world_result": result.to_dict(),
        "world_safe": result.safe,
        "non_replay_physics_gates_passed": bool(gates and all(gates.values())),
        "phase_timeline": phase_timeline,
        "coordination_active_frames": int(np.sum(actor_active)),
        "shot_projection": {
            "method": "peak_forward_velocity_within_120ms_ballistic_v1",
            "sample_frame": projection_frame,
            "goal_plane_y_m": projected_y,
            "goal_plane_z_m": projected_z,
            "target_error_m": target_error,
            "whole_ball_inside_goal": whole_ball_inside,
        },
        "assessment": assessment,
        "trajectory_digest": digest,
        "activation_ceiling": "SIM_ONLY",
        "promotion_eligible": False,
        "hardware_command_sent": False,
    }
    if output_dir is not None:
        root = output_dir.expanduser().resolve()
        checkout = Path(__file__).parents[3]
        if root == checkout or checkout in root.parents:
            raise ValueError("dynamic strike probe evidence must remain outside the checkout")
        if root.exists():
            raise FileExistsError("dynamic strike probe output already exists")
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


def validate_dynamic_strike_coordination_probe(path: Path) -> dict[str, Any]:
    """Recompute a probe's physics-derived fields and byte commitments."""

    resolved = path.expanduser().resolve()
    value = _read_json(resolved)
    declared = value.pop("report_hash", None)
    try:
        artifact = value.get("trajectory_artifact")
        if not isinstance(artifact, dict) or artifact.get("file") != "trajectory.npz":
            raise ValueError("dynamic strike probe trajectory binding is absent")
        trajectory_path = resolved.parent / "trajectory.npz"
        if not trajectory_path.is_file() or hash_bytes(
            trajectory_path.read_bytes()
        ) != artifact.get("file_hash"):
            raise ValueError("dynamic strike probe trajectory file changed")
        with np.load(trajectory_path, allow_pickle=False) as archive:
            trajectory = {name: np.asarray(archive[name]) for name in archive.files}
        digest = trajectory_digest(trajectory)
        actor = _actor_from_mapping(value.get("actor"))
        scenario = IndependentTeamWorldScenario(**cast(dict[str, Any], value.get("scenario_spec")))
        goal = G1TrainingGoalSpec(**cast(dict[str, Any], value.get("goal_spec")))
        raw_agent_ids = value.get("agent_ids")
        raw_roles = value.get("agent_roles")
        if (
            not isinstance(raw_agent_ids, list)
            or raw_agent_ids != sorted(set(raw_agent_ids))
            or not isinstance(raw_roles, dict)
            or set(raw_roles) != set(raw_agent_ids)
        ):
            raise ValueError("dynamic strike probe roster binding is invalid")
        agent_ids = tuple(str(agent_id) for agent_id in raw_agent_ids)
        roles = {agent_id: MatchRole(str(raw_roles[agent_id])) for agent_id in agent_ids}
        world_result = value.get("world_result")
        if not isinstance(world_result, dict):
            raise ValueError("dynamic strike probe world result is absent")
        assessment = assess_phase_conditioned_strike(
            trajectory=trajectory,
            trajectory_hash=digest,
            agent_ids=agent_ids,
            roles=roles,
            goal=goal,
            strict_replay=False,
            world_safe=bool(world_result.get("safe")),
        ).to_dict()
        gates = dict(assessment["gates"])
        gates.pop("strict_replay", None)
        strike_frame = int(assessment["events"]["strike_frame"])
        projected_y, projected_z, target_error, projection_frame = _project_shot(
            trajectory=trajectory,
            strike_frame=strike_frame,
            goal_plane_x_m=goal.plane_x_m,
            target_y_m=goal.target_y_m,
            target_z_m=goal.target_z_m,
        )
        shot_projection = {
            "method": "peak_forward_velocity_within_120ms_ballistic_v1",
            "sample_frame": projection_frame,
            "goal_plane_y_m": projected_y,
            "goal_plane_z_m": projected_z,
            "target_error_m": target_error,
            "whole_ball_inside_goal": bool(
                projected_y is not None
                and projected_z is not None
                and abs(projected_y) <= goal.width_m / 2.0 - goal.ball_radius_m
                and goal.ball_radius_m <= projected_z <= goal.height_m - goal.ball_radius_m
            ),
        }
        phase_codes = np.asarray(trajectory["strike_phase_code"], dtype=np.int64)
        phase_changes = np.flatnonzero(np.r_[True, phase_codes[1:] != phase_codes[:-1]])
        phase_timeline = [
            {
                "frame": int(frame),
                "time_sec": float(trajectory["time"][frame]),
                "code": int(phase_codes[frame]),
            }
            for frame in phase_changes
            if phase_codes[frame] > 0
        ]
        active = np.asarray(trajectory["strike_coordination_actor_active"], dtype=np.bool_)
        if (
            declared != hash_json(value)
            or value.get("schema_version") != "rosclaw_soccer.dynamic_strike_coordination_probe.v1"
            or digest != artifact.get("trajectory_digest")
            or digest != value.get("trajectory_digest")
            or actor.actor_hash != value.get("actor_hash")
            or actor.policy_type != value.get("actor_policy_type")
            or scenario.scenario_hash != value.get("scenario_hash")
            or scenario.scenario_id != value.get("scenario")
            or world_result.get("trajectory_hash") != digest
            or world_result.get("scenario_hash") != scenario.scenario_hash
            or world_result.get("physics_authority") != "CPU_MUJOCO"
            or world_result.get("hardware_command_sent") is not False
            or value.get("world_safe") != world_result.get("safe")
            or value.get("assessment") != assessment
            or value.get("non_replay_physics_gates_passed") != bool(gates and all(gates.values()))
            or value.get("phase_timeline") != phase_timeline
            or value.get("coordination_active_frames") != int(np.sum(active))
            or value.get("shot_projection") != shot_projection
            or value.get("activation_ceiling") != "SIM_ONLY"
            or value.get("promotion_eligible") is not False
            or value.get("hardware_command_sent") is not False
        ):
            raise ValueError("dynamic strike probe integrity or authority changed")
    finally:
        value["report_hash"] = declared
    return value


def _project_shot(
    *,
    trajectory: dict[str, NDArray[Any]],
    strike_frame: int,
    goal_plane_x_m: float,
    target_y_m: float,
    target_z_m: float,
) -> tuple[float | None, float | None, float | None, int | None]:
    if strike_frame < 0:
        return (None, None, None, None)
    time = np.asarray(trajectory["time"], dtype=np.float64)
    pose = np.asarray(trajectory["ball_pose"], dtype=np.float64)
    velocity_trace = np.asarray(trajectory["ball_velocity"], dtype=np.float64)
    end_frame = min(
        len(time),
        int(np.searchsorted(time, time[strike_frame] + 0.12, side="right")),
    )
    candidate_frames = np.arange(strike_frame, end_frame, dtype=np.int64)
    candidate_frames = candidate_frames[
        (velocity_trace[candidate_frames, 0] > 1.0e-6)
        & (pose[candidate_frames, 0] < goal_plane_x_m)
    ]
    if len(candidate_frames) == 0:
        return (None, None, None, None)
    sample_frame = int(candidate_frames[np.argmax(velocity_trace[candidate_frames, 0])])
    position = np.asarray(pose[sample_frame, :3], dtype=np.float64)
    velocity = np.asarray(velocity_trace[sample_frame, :3], dtype=np.float64)
    if velocity[0] <= 1.0e-6 or position[0] >= goal_plane_x_m:
        return (None, None, None, None)
    horizon = float((goal_plane_x_m - position[0]) / velocity[0])
    if not 0.0 < horizon <= 2.0:
        return (None, None, None, None)
    projected_y = float(position[1] + horizon * velocity[1])
    projected_z = float(position[2] + horizon * velocity[2] - 0.5 * 9.81 * horizon * horizon)
    target_error = float(math.hypot(projected_y - target_y_m, projected_z - target_z_m))
    return (projected_y, projected_z, target_error, sample_frame)


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


def _actor_from_mapping(raw: object) -> DynamicStrikeCoordinationActor:
    if not isinstance(raw, Mapping):
        raise ValueError("dynamic strike probe actor payload is invalid")
    values = dict(raw)
    for name in ("weights", "biases", "feature_mean", "feature_std"):
        vector = values.get(name)
        if not isinstance(vector, list):
            raise ValueError("dynamic strike probe actor vectors must be JSON arrays")
        values[name] = tuple(float(item) for item in vector)
    return DynamicStrikeCoordinationActor(**values)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--actor-artifact", type=Path)
    parser.add_argument("--stance-blend", type=float)
    parser.add_argument("--goal-yaw-blend", type=float)
    parser.add_argument("--stance-end-blend", type=float)
    parser.add_argument("--goal-yaw-end-blend", type=float)
    parser.add_argument("--ball-y-offset", type=float, default=0.0)
    parser.add_argument("--goal-target-y", type=float, default=0.80)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    if not -0.25 <= arguments.ball_y_offset <= 0.25:
        raise ValueError("ball y offset must remain inside [-0.25, 0.25] m")
    if not -1.20 <= arguments.goal_target_y <= 1.20:
        raise ValueError("goal target y must remain inside [-1.20, 1.20] m")
    scenario_seed = 209000 + round(1000 * arguments.ball_y_offset)
    constant_arguments = (arguments.stance_blend, arguments.goal_yaw_blend)
    ramp_arguments = (arguments.stance_end_blend, arguments.goal_yaw_end_blend)
    if arguments.actor_artifact is not None and any(
        value is not None for value in (*constant_arguments, *ramp_arguments)
    ):
        raise ValueError("actor artifact cannot be combined with parameter teacher arguments")
    if arguments.actor_artifact is None and any(value is None for value in constant_arguments):
        raise ValueError("both start blends are required for a parameter teacher")
    if any(value is None for value in ramp_arguments) and not all(
        value is None for value in ramp_arguments
    ):
        raise ValueError("both phase-ramp end blends must be provided together")
    if arguments.actor_artifact is not None:
        from rosclaw_soccer.training.dynamic_strike_coordination_learning import (
            load_dynamic_strike_coordination_actor_artifact,
        )

        actor, _ = load_dynamic_strike_coordination_actor_artifact(arguments.actor_artifact)
    elif arguments.stance_end_blend is None:
        actor = DynamicStrikeCoordinationActor.constant(
            stance_blend=arguments.stance_blend,
            goal_yaw_blend=arguments.goal_yaw_blend,
        )
    else:
        actor = DynamicStrikeCoordinationActor.phase_ramp(
            stance_start_blend=arguments.stance_blend,
            stance_end_blend=arguments.stance_end_blend,
            goal_yaw_start_blend=arguments.goal_yaw_blend,
            goal_yaw_end_blend=arguments.goal_yaw_end_blend,
        )
    result = run_dynamic_strike_coordination_probe(
        asset_root=arguments.asset_root,
        actor=actor,
        goal_target_y_m=arguments.goal_target_y,
        scenario=replace(
            default_continuous_match_scenario(),
            scenario_id=(
                f"s199.s209.dynamic-strike.y{arguments.ball_y_offset:+.3f}.seed{scenario_seed}"
            ),
            ball_initial_position_m=(
                default_continuous_match_scenario().ball_initial_position_m[0],
                default_continuous_match_scenario().ball_initial_position_m[1]
                + arguments.ball_y_offset,
                default_continuous_match_scenario().ball_initial_position_m[2],
            ),
            seed=scenario_seed,
        ),
        output_dir=arguments.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "run_dynamic_strike_coordination_probe",
    "validate_dynamic_strike_coordination_probe",
]
