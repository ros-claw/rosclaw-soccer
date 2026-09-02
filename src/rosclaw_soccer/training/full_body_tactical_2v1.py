"""Full-body G1 bridge for the bounded 2v1 PASS/SHOOT decision.

Unlike :mod:`tactical_2v1_physics`, every role in this module is a qualified
29-DoF G1 in one CPU MuJoCo model.  The carrier owns the frozen target-
conditioned kick policy, the team-mate owns a frozen locomotion stance, and
the opponent owns the frozen causal locomotion controller.  The learner still
owns only the high-level PASS/SHOOT choice.

The first bridge intentionally ends a pass at controlled foot reception.  It
does not claim a one-touch finish or a continuous match.  This makes the
remaining skill-composition boundary measurable instead of hiding it behind a
scripted second kick.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.growth.tactical_2v1 import (
    MatchedTacticalRollout,
    TacticalAction,
    TacticalRewardWeights,
    TwoVsOneDecisionEvidence,
    TwoVsOneState,
)
from rosclaw_soccer.providers.g1.asset_qualification import trajectory_digest
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.shared_world import (
    G1GoalkeeperConfig,
    G1TacticalMovementConfig,
    simulate_shared_world,
    trained_three_role_skill_simulation_kwargs,
)
from rosclaw_soccer.training.tactical_2v1_physics import FrozenTacticalSkillBundle
from rosclaw_soccer.world.field import G1TrainingGoalSpec

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.:-]{0,127}$")
_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_ACTIONS = (TacticalAction.PASS, TacticalAction.SHOOT)


def _clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _point_segment_distance(
    point: NDArray[np.float64],
    start: NDArray[np.float64],
    end: NDArray[np.float64],
) -> float:
    segment = end - start
    denominator = float(np.dot(segment, segment))
    if denominator <= 1.0e-12:
        return float(np.linalg.norm(point - start))
    fraction = float(np.clip(np.dot(point - start, segment) / denominator, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + fraction * segment)))


@dataclass(frozen=True)
class FullBodyTwoVsOneConfig:
    simulation_duration_sec: float = 7.0
    ball_ground_friction: float = 0.10
    maximum_foot_reception_distance_m: float = 0.28
    minimum_pelvis_height_m: float = 0.60
    maximum_tilt_rad: float = 0.75
    maximum_robot_contact_steps: int = 80
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.full_body_2v1_config.v1"

    def __post_init__(self) -> None:
        values = (
            self.simulation_duration_sec,
            self.ball_ground_friction,
            self.maximum_foot_reception_distance_m,
            self.minimum_pelvis_height_m,
            self.maximum_tilt_rad,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("full-body 2v1 config must be finite")
        if (
            not 6.5 <= self.simulation_duration_sec <= 8.0
            or not 0.05 <= self.ball_ground_friction <= 0.20
            or not 0.15 <= self.maximum_foot_reception_distance_m <= 0.30
            or not 0.55 <= self.minimum_pelvis_height_m <= 0.70
            or not 0.50 <= self.maximum_tilt_rad <= 0.90
            or isinstance(self.maximum_robot_contact_steps, bool)
            or not 0 <= self.maximum_robot_contact_steps <= 100
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("full-body 2v1 config violates its safety envelope")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass(frozen=True)
class FullBodyRoleMovementPlan:
    """Action-conditioned full-body locomotion plan with no pose authority."""

    teammate_origin_m: tuple[float, float, float]
    teammate_movement: G1TacticalMovementConfig
    defender_movement: G1TacticalMovementConfig
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.full_body_role_movement_plan.v1"

    def __post_init__(self) -> None:
        origin = np.asarray(self.teammate_origin_m, dtype=np.float64)
        if (
            origin.shape != (3,)
            or not np.all(np.isfinite(origin))
            or not 3.5 <= origin[0] <= 6.0
            or abs(float(origin[1])) > 1.5
            or abs(float(origin[2])) > 1.0e-12
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("full-body role movement plan violates its SIM-only field envelope")

    @property
    def plan_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass(frozen=True)
class FullBodyTwoVsOneScenario:
    scenario_id: str
    seed: int
    teammate_origin_m: tuple[float, float, float]
    defender_origin_m: tuple[float, float, float]
    ball_ground_friction: float = 0.10
    schema_version: str = "rosclaw_soccer.full_body_2v1_scenario.v1"

    def __post_init__(self) -> None:
        values = (*self.teammate_origin_m, *self.defender_origin_m, self.ball_ground_friction)
        if (
            not _IDENTIFIER.fullmatch(self.scenario_id)
            or isinstance(self.seed, bool)
            or not 0 <= self.seed <= 2**32 - 1
            or any(not math.isfinite(value) for value in values)
        ):
            raise ValueError("full-body 2v1 scenario identity or values are invalid")
        teammate_x, teammate_y, teammate_z = self.teammate_origin_m
        defender_x, defender_y, defender_z = self.defender_origin_m
        if (
            not 5.20 <= teammate_x <= 5.80
            or not 0.25 <= abs(teammate_y) <= 0.65
            or abs(teammate_z) > 1.0e-12
            or not 0.80 <= defender_x <= 4.90
            or not 0.25 <= abs(defender_y) <= 0.80
            or abs(defender_z) > 1.0e-12
            or not 0.05 <= self.ball_ground_friction <= 0.20
        ):
            raise ValueError("full-body 2v1 layout exceeds the qualified curriculum")

    @property
    def scenario_hash(self) -> str:
        return str(hash_json(asdict(self)))

    def state(
        self,
        *,
        skill_bundle: FrozenTacticalSkillBundle,
        config: FullBodyTwoVsOneConfig,
    ) -> TwoVsOneState:
        carrier = np.asarray((0.0, 0.0), dtype=np.float64)
        ball = np.asarray((1.25, 0.0), dtype=np.float64)
        teammate = np.asarray(self.teammate_origin_m[:2], dtype=np.float64)
        defender = np.asarray(self.defender_origin_m[:2], dtype=np.float64)
        goal = np.asarray((7.50, -0.80), dtype=np.float64)
        pressure = _clamp01(1.0 - float(np.linalg.norm(defender - carrier)) / 5.0)
        teammate_open = _clamp01(_point_segment_distance(defender, ball, teammate) / 1.50)
        shot_open = _clamp01(_point_segment_distance(defender, ball, goal) / 1.50)
        world = {
            "carrier": carrier.tolist(),
            "ball": ball.tolist(),
            "teammate": teammate.tolist(),
            "defender": defender.tolist(),
            "goal": goal.tolist(),
        }
        return TwoVsOneState(
            state_id=f"state.{self.scenario_id}",
            seed=self.seed,
            self_state_hash=hash_json({"carrier": world["carrier"], "ball": world["ball"]}),
            world_state_hash=hash_json(world),
            scenario_hash=self.scenario_hash,
            environment_hash=hash_json(
                {
                    "config_hash": config.config_hash,
                    "world": "full_body_g1_shared_world_2v1_v1",
                    "implementation_hash": hash_bytes(Path(__file__).read_bytes()),
                }
            ),
            frozen_foundation_hash=skill_bundle.athlete_foundation_hash,
            frozen_skill_bundle_hash=skill_bundle.bundle_hash,
            frozen_defender_hash=hash_json(
                {
                    "controller": "g1_causal_locomotion_interceptor_v1",
                    "origin_m": self.defender_origin_m,
                }
            ),
            carrier_pressure=pressure,
            teammate_lane_openness=teammate_open,
            shot_lane_openness=shot_open,
            goal_progress=float(ball[0] / goal[0]),
            teammate_progress=float(teammate[0] / goal[0]),
        )


@dataclass(frozen=True)
class FullBodyTwoVsOneResult:
    action: TacticalAction
    focal_teammate_present: bool
    carrier_contact_observed: bool
    carrier_contact_time_sec: float | None
    teammate_contact_observed: bool
    teammate_contact_time_sec: float | None
    teammate_foot_reception_distance_m: float | None
    defender_contact_observed: bool
    defender_contact_time_sec: float | None
    pass_completed: bool
    goal_scored: bool
    goal_crossing_time_sec: float | None
    goal_crossing_y_m: float | None
    goal_crossing_z_m: float | None
    finite_state: bool
    carrier_minimum_pelvis_height_m: float
    teammate_minimum_pelvis_height_m: float
    defender_minimum_pelvis_height_m: float | None
    carrier_joint_limit_violation: bool
    teammate_joint_limit_violation: bool
    defender_joint_limit_violation: bool
    torque_limit_violation: bool
    actuator_saturation: bool
    robot_robot_contact_count: int
    possession_progress: float
    team_reward: float
    role_reward: float
    safety_cost: float
    trajectory_hash: str
    action_trace_hash: str
    whole_body_g1_count: int = 3
    physics_authority: str = "CPU_MUJOCO"
    activation_ceiling: str = "SIM_ONLY"
    hardware_command_sent: bool = False
    schema_version: str = "rosclaw_soccer.full_body_2v1_result.v1"

    @property
    def task_succeeded(self) -> bool:
        return self.pass_completed if self.action == TacticalAction.PASS else self.goal_scored

    @property
    def safe(self) -> bool:
        return self.safety_cost == 0.0

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["action"] = self.action.value
        value["task_succeeded"] = self.task_succeeded
        value["safe"] = self.safe
        return value


def _goal() -> G1TrainingGoalSpec:
    return G1TrainingGoalSpec(
        plane_x_m=7.50,
        width_m=3.0,
        height_m=2.0,
        depth_m=1.2,
        target_y_m=-0.80,
        target_z_m=0.35,
        precision_radius_m=0.30,
    )


def _crossing_time(trajectory: dict[str, NDArray[Any]], plane_x_m: float) -> float | None:
    pose = np.asarray(trajectory["ball_pose"], dtype=np.float64)
    indices = np.flatnonzero((pose[:-1, 0] < plane_x_m) & (pose[1:, 0] >= plane_x_m))
    if not indices.size:
        return None
    return float(np.asarray(trajectory["time"], dtype=np.float64)[int(indices[0] + 1)])


def _foot_reception_distance(
    trajectory: dict[str, NDArray[Any]], contact_time_sec: float | None
) -> float | None:
    if contact_time_sec is None:
        return None
    time = np.asarray(trajectory["time"], dtype=np.float64)
    index = min(int(np.searchsorted(time, contact_time_sec)), len(time) - 1)
    ball = np.asarray(trajectory["ball_pose"], dtype=np.float64)[index, :3]
    left = np.asarray(trajectory["passer_left_foot_position"], dtype=np.float64)[index]
    right = np.asarray(trajectory["passer_right_foot_position"], dtype=np.float64)[index]
    return float(min(np.linalg.norm(ball - left), np.linalg.norm(ball - right)))


def simulate_full_body_two_vs_one(
    *,
    asset_root: Path,
    scenario: FullBodyTwoVsOneScenario,
    action: TacticalAction,
    skill_bundle: FrozenTacticalSkillBundle,
    config: FullBodyTwoVsOneConfig | None = None,
    focal_teammate_present: bool = True,
    movement_plan: FullBodyRoleMovementPlan | None = None,
) -> tuple[FullBodyTwoVsOneResult, dict[str, NDArray[Any]]]:
    """Execute one frozen high-level option with three complete G1 bodies."""

    if action not in _ACTIONS:
        raise ValueError("full-body bridge supports only PASS or SHOOT")
    active = config or FullBodyTwoVsOneConfig()
    goal = _goal()
    teammate_y = float(scenario.teammate_origin_m[1])
    policy_target = (
        (goal.plane_x_m, 2.0 * teammate_y, goal.ball_radius_m)
        if action == TacticalAction.PASS
        else (goal.plane_x_m, goal.target_y_m, 0.50)
    )
    kwargs = trained_three_role_skill_simulation_kwargs()
    kwargs.update(
        {
            "shooter_ball_initial_position_m": (1.25, 0.0, goal.ball_radius_m),
            "shooter_origin": (0.0, 0.0, 0.0),
            "shooter_start_sec": 0.0,
            "shooter_target": (goal.plane_x_m, goal.target_y_m, goal.target_z_m),
            "shooter_policy_target": policy_target,
            "passer_origin": (
                scenario.teammate_origin_m
                if movement_plan is None
                else movement_plan.teammate_origin_m
            ),
            "passer_yaw_rad": 0.0,
            "passer_start_sec": 100.0,
            "passer_collision_enabled": focal_teammate_present,
            "passer_policy_target_m": (2.0, 0.0, 0.20),
            "pass_reception_target_m": (1.0, 0.0, goal.ball_radius_m),
            "passer_tactical_movement_config": (
                None if movement_plan is None else movement_plan.teammate_movement
            ),
            "ball_ground_friction": scenario.ball_ground_friction,
            "goal_spec": goal,
            "goalkeeper_config": G1GoalkeeperConfig(
                initial_lateral_position_m=float(scenario.defender_origin_m[1]),
                reaction_delay_sec=0.35,
                ready_shuffle_speed_mps=0.0,
                maximum_depth_correction_mps=0.05,
            ),
            "goalkeeper_origin_override_m": scenario.defender_origin_m,
            "goalkeeper_tactical_movement_config": (
                None if movement_plan is None else movement_plan.defender_movement
            ),
            "goalkeeper_threat_role": "shooter",
            "simulation_duration_sec": active.simulation_duration_sec,
        }
    )
    result, shared_trajectory = simulate_shared_world(asset_root, **kwargs)
    trajectory = dict(shared_trajectory)
    trajectory["focal_teammate_present"] = np.full(
        np.asarray(trajectory["time"]).shape,
        focal_teammate_present,
        dtype=np.bool_,
    )
    foot_distance = _foot_reception_distance(trajectory, result.pass_contact_time_sec)
    crossing_time = _crossing_time(trajectory, goal.plane_x_m)
    carrier_contact = bool(result.shot_contact_observed)
    teammate_contact = bool(result.pass_contact_observed and focal_teammate_present)
    defender_contact = bool(result.goalkeeper_ball_contact_observed)
    pass_completed = bool(
        action == TacticalAction.PASS
        and carrier_contact
        and teammate_contact
        and result.shot_contact_time_sec is not None
        and result.pass_contact_time_sec is not None
        and result.shot_contact_time_sec < result.pass_contact_time_sec
        and foot_distance is not None
        and foot_distance <= active.maximum_foot_reception_distance_m
        and (
            result.goalkeeper_ball_contact_time_sec is None
            or result.pass_contact_time_sec < result.goalkeeper_ball_contact_time_sec
        )
    )
    goal_scored = bool(
        action == TacticalAction.SHOOT
        and carrier_contact
        and result.goal_crossed
        and crossing_time is not None
        and (result.pass_contact_time_sec is None or crossing_time < result.pass_contact_time_sec)
        and (
            result.goalkeeper_ball_contact_time_sec is None
            or crossing_time < result.goalkeeper_ball_contact_time_sec
        )
    )
    primary_roles_safe = bool(
        result.finite_state
        and result.shooter_min_pelvis_height_m >= active.minimum_pelvis_height_m
        and result.shooter_roll_peak_rad <= active.maximum_tilt_rad
        and result.shooter_pitch_peak_rad <= active.maximum_tilt_rad
        and not result.shooter_joint_limit_violation
        and result.goalkeeper_min_pelvis_height_m is not None
        and result.goalkeeper_min_pelvis_height_m >= active.minimum_pelvis_height_m
        and not result.goalkeeper_joint_limit_violation
        and not result.torque_limit_violation
        and not result.actuator_saturation
        and result.robot_robot_contact_count <= active.maximum_robot_contact_steps
    )
    teammate_safe = bool(
        not focal_teammate_present
        or (
            result.passer_min_pelvis_height_m >= active.minimum_pelvis_height_m
            and not result.passer_joint_limit_violation
        )
    )
    safe = primary_roles_safe and teammate_safe
    pose = np.asarray(trajectory["ball_pose"], dtype=np.float64)
    progress = _clamp01((float(np.max(pose[:, 0])) - 1.25) / (goal.plane_x_m - 1.25))
    task_success = bool((pass_completed or goal_scored) and safe)
    trajectory_hash = trajectory_digest(trajectory)
    action_trace_hash = hash_json(
        {
            "scenario_hash": scenario.scenario_hash,
            "action": action.value,
            "skill_bundle_hash": skill_bundle.bundle_hash,
            "focal_teammate_present": focal_teammate_present,
            "policy_target": policy_target,
            "config_hash": active.config_hash,
            "movement_plan_hash": None if movement_plan is None else movement_plan.plan_hash,
            "implementation_hash": hash_bytes(Path(__file__).read_bytes()),
        }
    )
    wrapped = FullBodyTwoVsOneResult(
        action=action,
        focal_teammate_present=focal_teammate_present,
        carrier_contact_observed=carrier_contact,
        carrier_contact_time_sec=result.shot_contact_time_sec,
        teammate_contact_observed=teammate_contact,
        teammate_contact_time_sec=(result.pass_contact_time_sec if teammate_contact else None),
        teammate_foot_reception_distance_m=foot_distance if teammate_contact else None,
        defender_contact_observed=defender_contact,
        defender_contact_time_sec=result.goalkeeper_ball_contact_time_sec,
        pass_completed=pass_completed,
        goal_scored=goal_scored,
        goal_crossing_time_sec=crossing_time,
        goal_crossing_y_m=result.goal_crossing_y_m,
        goal_crossing_z_m=result.goal_crossing_z_m,
        finite_state=result.finite_state,
        carrier_minimum_pelvis_height_m=result.shooter_min_pelvis_height_m,
        teammate_minimum_pelvis_height_m=result.passer_min_pelvis_height_m,
        defender_minimum_pelvis_height_m=result.goalkeeper_min_pelvis_height_m,
        carrier_joint_limit_violation=result.shooter_joint_limit_violation,
        teammate_joint_limit_violation=result.passer_joint_limit_violation,
        defender_joint_limit_violation=result.goalkeeper_joint_limit_violation,
        torque_limit_violation=result.torque_limit_violation,
        actuator_saturation=result.actuator_saturation,
        robot_robot_contact_count=result.robot_robot_contact_count,
        possession_progress=progress,
        team_reward=1.0 if task_success else 0.0,
        role_reward=1.0 if task_success else 0.0,
        safety_cost=0.0 if safe else 1.0,
        trajectory_hash=trajectory_hash,
        action_trace_hash=action_trace_hash,
    )
    return wrapped, trajectory


def matched_full_body_two_vs_one_decision(
    *,
    asset_root: Path,
    scenario: FullBodyTwoVsOneScenario,
    action: TacticalAction,
    policy_hash: str,
    skill_bundle: FrozenTacticalSkillBundle,
    config: FullBodyTwoVsOneConfig | None = None,
    weights: TacticalRewardWeights | None = None,
) -> tuple[
    TwoVsOneDecisionEvidence,
    FullBodyTwoVsOneResult,
    FullBodyTwoVsOneResult,
    dict[str, NDArray[Any]],
    dict[str, NDArray[Any]],
]:
    """Run an option and the same snapshot with team-mate collision removed."""

    if not _HASH.fullmatch(policy_hash):
        raise ValueError("full-body policy identity must be a SHA-256 hash")
    active = config or FullBodyTwoVsOneConfig()
    primary, primary_trace = simulate_full_body_two_vs_one(
        asset_root=asset_root,
        scenario=scenario,
        action=action,
        skill_bundle=skill_bundle,
        config=active,
        focal_teammate_present=True,
    )
    ablated, ablated_trace = simulate_full_body_two_vs_one(
        asset_root=asset_root,
        scenario=scenario,
        action=action,
        skill_bundle=skill_bundle,
        config=active,
        focal_teammate_present=False,
    )
    state = scenario.state(skill_bundle=skill_bundle, config=active)
    rollout = MatchedTacticalRollout(
        state_hash=state.state_hash,
        policy_hash=policy_hash,
        action=action,
        action_trace_hash=primary.action_trace_hash,
        trajectory_hash=primary.trajectory_hash,
        ablation_action_trace_hash=ablated.action_trace_hash,
        ablated_trajectory_hash=ablated.trajectory_hash,
        team_reward=primary.team_reward,
        role_reward=primary.role_reward,
        ablated_team_reward=ablated.team_reward,
        possession_progress=primary.possession_progress,
        safety_cost=primary.safety_cost,
        ablation_mode="focal_agent_removed",
    )
    evidence = TwoVsOneDecisionEvidence(
        state=state,
        rollout=rollout,
        weights=weights or TacticalRewardWeights(),
    )
    return evidence, primary, ablated, primary_trace, ablated_trace


__all__ = [
    "FullBodyRoleMovementPlan",
    "FullBodyTwoVsOneConfig",
    "FullBodyTwoVsOneResult",
    "FullBodyTwoVsOneScenario",
    "matched_full_body_two_vs_one_decision",
    "simulate_full_body_two_vs_one",
]
