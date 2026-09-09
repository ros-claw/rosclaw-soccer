"""Shared G1 role-autonomy arena backed by independent ROSClaw agent cells.

The arena is a locomotion and coordination bridge, not a complete learned
match.  Every G1 receives its own egocentric observation, private policy
identity, memory namespaces, and role-authorized decision.  Decisions become
bounded velocity commands for the frozen RoboNaldo locomotion policy.  The
module never writes a root pose after initialization, never writes football
state after initialization, and exposes no hardware path.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.growth.contact_stroke import ContactStroke
from rosclaw_soccer.growth.contextual_strike_experts import (
    ContextualStrikeExpertMemory,
    build_strike_task_context,
)
from rosclaw_soccer.growth.dynamic_strike_coordination import (
    DynamicStrikeCoordinationActor,
    StrikeCoordinationAction,
    StrikeCoordinationObservation,
)
from rosclaw_soccer.growth.independent_agent_cell import (
    AgentCellDecision,
    AgentCellObservation,
    AgentPhysicalState,
    RosclawSoccerAgentCell,
    build_team_coordination_frame,
)
from rosclaw_soccer.growth.locomotion_contact_teacher import (
    G1LocomotionContactTeacherConfig,
    G1RollingOptionBridgeConfig,
    locomotion_contact_teacher_effect,
)
from rosclaw_soccer.growth.near_ball_residual import NearBallResidualPolicy, bounded_residual
from rosclaw_soccer.growth.owned_ball_contact import OwnedBallContactPolicy
from rosclaw_soccer.growth.pass_handoff import PassHandoff
from rosclaw_soccer.growth.role_self_model import (
    MatchRole,
    TacticalIntent,
    TeamRoleRoster,
)
from rosclaw_soccer.growth.strike_phase_controller import (
    StrikePhase,
    StrikePhaseConfig,
    StrikePhaseState,
)
from rosclaw_soccer.providers.g1.asset_qualification import (
    qualify_g1_assets,
    trajectory_digest,
)
from rosclaw_soccer.providers.g1.keeper_frame import KeeperFrame
from rosclaw_soccer.providers.g1.locomotion_action_frame import remap_previous_locomotion_action
from rosclaw_soccer.providers.g1.mujoco_primitives import (
    adapt_shot_target,
    load_robonaldo,
    mirror_g1_joint_gains,
    mirror_g1_joint_positions,
)
from rosclaw_soccer.providers.g1.shared_keeper_reach import (
    SharedKeeperReach,
    SharedKeeperReachConfig,
)
from rosclaw_soccer.sim.contracts import (
    G1_DDS_JOINT_NAMES,
    G1_HARD_TORQUE_LIMITS,
    ShotParameters,
    hash_json,
)
from rosclaw_soccer.skills.team.motor_option import (
    TeamMotorObservation,
    TeamMotorOption,
    TeamMotorPhysicsObservation,
    TeamMotorPhysicsObserver,
    TeamMotorTarget,
    motor_blocks_residual,
)
from rosclaw_soccer.world.field import (
    G1CompliantGoalNetState,
    G1TrainingGoalSpec,
    apply_g1_compliant_goal_net_force,
)
from rosclaw_soccer.world.goalkeeper_glove_material import GoalkeeperGloveMaterial
from rosclaw_soccer.world.multi_player import (
    G1PitchPlayerSpec,
    build_g1_multi_player_stadium_model,
)
from rosclaw_soccer.world.player_clearance import propose_clearance_velocity

_CONTROL_DT = 0.02
_PHYSICS_DT = 0.002
_SUBSTEPS = 10
_MOTION_REL = Path("policy/robonaldo/model/freekick_motion.npz")
_INTENT_CODES = {intent: index for index, intent in enumerate(TacticalIntent)}
_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.:-]{0,127}$")


@dataclass(frozen=True)
class IndependentTeamWorldConfig:
    simulation_duration_sec: float = 10.0
    decision_period_sec: float = 0.10
    maximum_speed_mps: float = 0.38
    goalkeeper_maximum_speed_mps: float = 0.30
    maximum_acceleration_mps2: float = 0.60
    maximum_yaw_rate_radps: float = 0.80
    yaw_gain: float = 1.20
    receive_open_body_angle_rad: float = math.pi / 2.0
    position_gain: float = 0.85
    arrival_radius_m: float = 0.14
    minimum_player_separation_m: float = 0.95
    collision_avoidance_gain: float = 1.20
    maximum_collision_correction_mps: float = 0.28
    possession_radius_m: float = 1.25
    contact_possession_hold_sec: float = 0.20
    receive_intercept_horizon_sec: float = 0.45
    receive_pocket_depth_m: float = 0.28
    duel_lateral_offset_m: float = 0.32
    receive_run_onto_horizon_sec: float = 2.5
    receive_run_onto_lane_radius_m: float = 0.65
    receive_runthrough_distance_m: float = 0.80
    receive_pacing_ratio: float = 0.45
    receive_braking_distance_m: float = 0.55
    post_receive_hold_sec: float = 0.20
    strike_bypass_lateral_m: float = 0.55
    minimum_ball_chaser_lease_sec: float = 2.5
    ball_chaser_handoff_margin_m: float = 0.40
    minimum_receive_lease_progress_m: float = 0.50
    minimum_receive_lease_ball_speed_mps: float = 0.50
    left_goal_plane_x_m: float = -1.50
    bilateral_goals: bool = False
    stationary_ball_acquisition: bool = False
    predictive_separation: bool = False
    all_role_clearance: bool = False
    strict_receive_handoff: bool = False
    receiver_commitment_priority: bool = False
    strike_residual_enabled: bool = False
    strike_stance_lateral_m: float | None = None
    keeper_reach: SharedKeeperReachConfig | None = None
    glove_material: GoalkeeperGloveMaterial | None = None
    stop_on_ball_exit: bool = False
    owned_contact_policy: OwnedBallContactPolicy | None = None
    minimum_pelvis_height_m: float = 0.55
    maximum_tilt_rad: float = 0.80
    joint_guard_margin_rad: float = 0.04
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.independent_team_world_config.v1"
    motor_approach_standoff_m: float | None = None
    motor_idle_residual_fallback: bool = False
    pass_stance_bypass: bool = False
    receive_lateral_braking: bool = False
    locomotion_action_frame_sync: bool = False

    def __post_init__(self) -> None:
        if type(self.locomotion_action_frame_sync) is not bool:
            raise ValueError("locomotion action-frame synchronization must be explicit")
        if type(self.receive_lateral_braking) is not bool or (
            self.receive_lateral_braking and not self.strict_receive_handoff
        ):
            raise ValueError("lateral receive braking requires strict physical handoff")
        if type(self.pass_stance_bypass) is not bool or (
            self.pass_stance_bypass and self.owned_contact_policy is None
        ):
            raise ValueError("pass bypass requires explicit owned-contact stance policy")
        if self.motor_approach_standoff_m is not None and (
            type(self.motor_approach_standoff_m) not in (int, float)
            or not math.isfinite(self.motor_approach_standoff_m)
            or not 0.20 <= self.motor_approach_standoff_m <= 0.50
        ):
            raise ValueError("bounded optional learned-motor approach standoff required")
        values = (
            self.simulation_duration_sec,
            self.decision_period_sec,
            self.maximum_speed_mps,
            self.goalkeeper_maximum_speed_mps,
            self.maximum_acceleration_mps2,
            self.maximum_yaw_rate_radps,
            self.yaw_gain,
            self.receive_open_body_angle_rad,
            self.position_gain,
            self.arrival_radius_m,
            self.minimum_player_separation_m,
            self.collision_avoidance_gain,
            self.maximum_collision_correction_mps,
            self.possession_radius_m,
            self.contact_possession_hold_sec,
            self.receive_intercept_horizon_sec,
            self.receive_pocket_depth_m,
            self.duel_lateral_offset_m,
            self.receive_run_onto_horizon_sec,
            self.receive_run_onto_lane_radius_m,
            self.receive_runthrough_distance_m,
            self.receive_pacing_ratio,
            self.receive_braking_distance_m,
            self.post_receive_hold_sec,
            self.strike_bypass_lateral_m,
            self.minimum_ball_chaser_lease_sec,
            self.ball_chaser_handoff_margin_m,
            self.minimum_receive_lease_progress_m,
            self.minimum_receive_lease_ball_speed_mps,
            self.left_goal_plane_x_m,
            self.minimum_pelvis_height_m,
            self.maximum_tilt_rad,
            self.joint_guard_margin_rad,
        )
        if (
            not isinstance(self.bilateral_goals, bool)
            or (
                self.owned_contact_policy is not None
                and not isinstance(self.owned_contact_policy, OwnedBallContactPolicy)
            )
            or not isinstance(self.stationary_ball_acquisition, bool)
            or not isinstance(self.predictive_separation, bool)
            or not isinstance(self.all_role_clearance, bool)
            or not isinstance(self.strict_receive_handoff, bool)
            or type(self.receiver_commitment_priority) is not bool
            or (self.receiver_commitment_priority and not self.strict_receive_handoff)
            or type(self.strike_residual_enabled) is not bool
            or type(self.motor_idle_residual_fallback) is not bool
            or (
                self.strike_stance_lateral_m is not None
                and (
                    type(self.strike_stance_lateral_m) not in {int, float}
                    or not math.isfinite(self.strike_stance_lateral_m)
                    or not -0.18 <= self.strike_stance_lateral_m <= 0.18
                )
            )
            or (
                self.keeper_reach is not None
                and not isinstance(self.keeper_reach, SharedKeeperReachConfig)
            )
            or (
                self.glove_material is not None
                and not isinstance(self.glove_material, GoalkeeperGloveMaterial)
            )
            or not isinstance(self.stop_on_ball_exit, bool)
            or any(not math.isfinite(value) for value in values)
            or not 5.0 <= self.simulation_duration_sec <= 25.0
            or not 0.08 <= self.decision_period_sec <= 0.20
            or not 0.20 <= self.maximum_speed_mps <= 0.70
            or not 0.20 <= self.goalkeeper_maximum_speed_mps <= 0.60
            or not 0.30 <= self.maximum_acceleration_mps2 <= 3.0
            or not 0.30 <= self.maximum_yaw_rate_radps <= 1.50
            or not 0.20 <= self.yaw_gain <= 3.0
            or not 0.50 <= self.receive_open_body_angle_rad <= 1.70
            or not 0.50 <= self.position_gain <= 3.0
            or not 0.05 <= self.arrival_radius_m <= 0.25
            or not 0.55 <= self.minimum_player_separation_m <= 1.20
            or not 0.20 <= self.collision_avoidance_gain <= 3.0
            or not 0.05 <= self.maximum_collision_correction_mps <= 0.30
            # Possession is measured from the ball centre to an ankle body.
            # The previous 0.50 m floor let an agent claim the ball before a
            # foot could possibly touch it, turning PASS/SHOOT into gestures
            # behind a stationary ball.  Keep the legacy default for frozen
            # S199 evidence while allowing contact-scale match profiles.
            or not 0.15 <= self.possession_radius_m <= 1.50
            or not 0.0 <= self.contact_possession_hold_sec <= 1.50
            or not 0.15 <= self.receive_intercept_horizon_sec <= 0.80
            or not 0.18 <= self.receive_pocket_depth_m <= 0.45
            or not 0.20 <= self.duel_lateral_offset_m <= 0.45
            or not 1.0 <= self.receive_run_onto_horizon_sec <= 3.0
            or not 0.30 <= self.receive_run_onto_lane_radius_m <= 0.80
            or not 0.30 <= self.receive_runthrough_distance_m <= 1.20
            or not 0.20 <= self.receive_pacing_ratio <= 0.80
            or not 0.30 <= self.receive_braking_distance_m <= 0.80
            or not 0.20 <= self.post_receive_hold_sec <= 1.00
            or not 0.35 <= self.strike_bypass_lateral_m <= 0.80
            or not 0.50 <= self.minimum_ball_chaser_lease_sec <= 5.0
            or not 0.10 <= self.ball_chaser_handoff_margin_m <= 0.80
            or not 0.20 <= self.minimum_receive_lease_progress_m <= 1.50
            or not 0.20 <= self.minimum_receive_lease_ball_speed_mps <= 1.50
            or not -5.0 <= self.left_goal_plane_x_m <= 0.0
            or not 0.45 <= self.minimum_pelvis_height_m <= 0.70
            or not 0.45 <= self.maximum_tilt_rad <= 1.00
            or not 0.04 <= self.joint_guard_margin_rad <= 0.10
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("independent team world violates its SIM-only envelope")

    @property
    def config_hash(self) -> str:
        value = asdict(self)
        if self.keeper_reach is None:
            value.pop("keeper_reach")  # Preserve historical disabled configuration identities.
        if self.glove_material is None:
            value.pop("glove_material")
        if self.joint_guard_margin_rad == 0.04:
            value.pop("joint_guard_margin_rad")
        if not self.strike_residual_enabled:
            value.pop("strike_residual_enabled")
        if self.strike_stance_lateral_m is None:
            value.pop("strike_stance_lateral_m")
        if self.motor_approach_standoff_m is None:
            value.pop("motor_approach_standoff_m")
        if not self.motor_idle_residual_fallback:
            value.pop("motor_idle_residual_fallback")
        if not self.receiver_commitment_priority:
            value.pop("receiver_commitment_priority")
        if not self.pass_stance_bypass:
            value.pop("pass_stance_bypass")
        if not self.receive_lateral_braking:
            value.pop("receive_lateral_braking")
        if not self.locomotion_action_frame_sync:
            value.pop("locomotion_action_frame_sync")
        return str(hash_json(value))


@dataclass(frozen=True)
class IndependentTeamWorldScenario:
    scenario_id: str
    ball_initial_position_m: tuple[float, float, float]
    ball_initial_velocity_mps: tuple[float, float, float]
    seed: int
    schema_version: str = "rosclaw_soccer.independent_team_world_scenario.v1"

    def __post_init__(self) -> None:
        values = (*self.ball_initial_position_m, *self.ball_initial_velocity_mps)
        if (
            not self.scenario_id.startswith("s199.")
            or isinstance(self.seed, bool)
            or not 0 <= self.seed <= 2**32 - 1
            or len(self.ball_initial_position_m) != 3
            or len(self.ball_initial_velocity_mps) != 3
            or any(not math.isfinite(value) for value in values)
            or self.ball_initial_position_m[2] <= 0.0
        ):
            raise ValueError("independent team scenario is invalid")

    @property
    def scenario_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass(frozen=True)
class AgentWorldQuality:
    agent_id: str
    role: MatchRole
    active_fraction: float
    displacement_m: float
    distinct_intent_count: int
    intent_switch_count: int
    decision_count: int
    pass_intent_count: int
    shot_intent_count: int
    save_intent_count: int
    distribution_intent_count: int
    minimum_pelvis_height_m: float
    maximum_tilt_rad: float
    joint_limit_violation: bool
    torque_limit_violation: bool
    required_minimum_pelvis_height_m: float
    allowed_maximum_tilt_rad: float
    schema_version: str = "rosclaw_soccer.agent_world_quality.v1"

    def __post_init__(self) -> None:
        counts = (
            self.distinct_intent_count,
            self.intent_switch_count,
            self.decision_count,
            self.pass_intent_count,
            self.shot_intent_count,
            self.save_intent_count,
            self.distribution_intent_count,
        )
        values = (
            self.active_fraction,
            self.displacement_m,
            self.minimum_pelvis_height_m,
            self.maximum_tilt_rad,
            self.required_minimum_pelvis_height_m,
            self.allowed_maximum_tilt_rad,
        )
        if (
            not _IDENTIFIER.fullmatch(self.agent_id)
            or not isinstance(self.role, MatchRole)
            or any(isinstance(value, bool) or value < 0 for value in counts)
            or any(not math.isfinite(value) for value in values)
            or not 0.0 <= self.active_fraction <= 1.0
            or self.displacement_m < 0.0
            or self.minimum_pelvis_height_m < 0.0
            or self.maximum_tilt_rad < 0.0
            or not 0.45 <= self.required_minimum_pelvis_height_m <= 0.70
            or not 0.45 <= self.allowed_maximum_tilt_rad <= 1.00
            or not isinstance(self.joint_limit_violation, bool)
            or not isinstance(self.torque_limit_violation, bool)
        ):
            raise ValueError("agent world quality contract is invalid")

    @property
    def safe(self) -> bool:
        return bool(
            self.minimum_pelvis_height_m >= self.required_minimum_pelvis_height_m
            and self.maximum_tilt_rad <= self.allowed_maximum_tilt_rad
            and not self.joint_limit_violation
            and not self.torque_limit_violation
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["role"] = self.role.value
        value["safe"] = self.safe
        return value


@dataclass(frozen=True)
class IndependentTeamWorldResult:
    scenario_hash: str
    roster_hash: str
    config_hash: str
    trajectory_hash: str
    player_count: int
    red_player_count: int
    blue_player_count: int
    decision_frame_count: int
    coordination_frame_hashes: tuple[str, ...]
    qualities: tuple[AgentWorldQuality, ...]
    pass_handshake_count: int
    pass_intent_count: int
    shot_intent_count: int
    save_intent_count: int
    distribution_intent_count: int
    finite_state: bool
    robot_robot_contact_count: int
    rolling_distance_m: float
    peak_ball_speed_mps: float
    keeper_policy_hashes: tuple[tuple[str, str], ...] = ()
    activation_ceiling: str = "SIM_ONLY"
    physics_authority: str = "CPU_MUJOCO"
    hardware_command_sent: bool = False
    pixels_used_for_scoring: bool = False
    schema_version: str = "rosclaw_soccer.independent_team_world_result.v1"
    motor_policy_hashes: tuple[tuple[str, str], ...] = ()
    motor_fault_agents: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        counts = (
            self.player_count,
            self.red_player_count,
            self.blue_player_count,
            self.decision_frame_count,
            self.pass_handshake_count,
            self.pass_intent_count,
            self.shot_intent_count,
            self.save_intent_count,
            self.distribution_intent_count,
            self.robot_robot_contact_count,
        )
        quality_ids = tuple(value.agent_id for value in self.qualities)
        if (
            len({agent for agent, _ in self.motor_policy_hashes}) != len(self.motor_policy_hashes)
            or not set(self.motor_fault_agents).issubset(dict(self.motor_policy_hashes))
            or len(set(self.motor_fault_agents)) != len(self.motor_fault_agents)
            or any(
                agent not in quality_ids or not _HASH.fullmatch(value)
                for agent, value in (*self.keeper_policy_hashes, *self.motor_policy_hashes)
            )
            or any(
                not _HASH.fullmatch(value)
                for value in (
                    self.scenario_hash,
                    self.roster_hash,
                    self.config_hash,
                    self.trajectory_hash,
                )
            )
            or any(isinstance(value, bool) or value < 0 for value in counts)
            or self.player_count != self.red_player_count + self.blue_player_count
            or self.player_count != len(self.qualities)
            or len(quality_ids) != len(set(quality_ids))
            or self.red_player_count != sum(agent_id.startswith("red.") for agent_id in quality_ids)
            or self.blue_player_count
            != sum(agent_id.startswith("blue.") for agent_id in quality_ids)
            or len(self.coordination_frame_hashes) != self.decision_frame_count
            or any(not _HASH.fullmatch(value) for value in self.coordination_frame_hashes)
            or self.pass_handshake_count != self.pass_intent_count
            or self.pass_intent_count != sum(value.pass_intent_count for value in self.qualities)
            or self.shot_intent_count != sum(value.shot_intent_count for value in self.qualities)
            or self.save_intent_count != sum(value.save_intent_count for value in self.qualities)
            or self.distribution_intent_count
            != sum(value.distribution_intent_count for value in self.qualities)
            or not math.isfinite(self.rolling_distance_m)
            or not math.isfinite(self.peak_ball_speed_mps)
            or self.rolling_distance_m < 0.0
            or self.peak_ball_speed_mps < 0.0
            or not isinstance(self.finite_state, bool)
            or not isinstance(self.hardware_command_sent, bool)
            or not isinstance(self.pixels_used_for_scoring, bool)
            or self.activation_ceiling != "SIM_ONLY"
            or self.physics_authority != "CPU_MUJOCO"
            or self.hardware_command_sent
            or self.pixels_used_for_scoring
        ):
            raise ValueError("independent team result contract is invalid")

    @property
    def all_agents_active(self) -> bool:
        return all(value.active_fraction >= 0.20 for value in self.qualities)

    @property
    def all_roles_autonomous(self) -> bool:
        return all(
            value.distinct_intent_count >= 1 and value.decision_count == self.decision_frame_count
            for value in self.qualities
        )

    @property
    def role_complete_both_teams(self) -> bool:
        required = {MatchRole.GOALKEEPER, MatchRole.PLAYMAKER, MatchRole.FINISHER}
        return all(
            required.issubset(
                {value.role for value in self.qualities if value.agent_id.startswith(f"{team_id}.")}
            )
            for team_id in ("red", "blue")
        )

    @property
    def safe(self) -> bool:
        return bool(
            self.finite_state
            and all(value.safe for value in self.qualities)
            and self.robot_robot_contact_count == 0
            and not self.hardware_command_sent
            and not self.pixels_used_for_scoring
        )

    @property
    def passed(self) -> bool:
        return bool(
            self.player_count >= 6
            and self.red_player_count >= 3
            and self.blue_player_count >= 3
            and self.decision_frame_count >= 10
            and len(self.coordination_frame_hashes) == self.decision_frame_count
            and len(set(self.coordination_frame_hashes)) == self.decision_frame_count
            and self.all_agents_active
            and self.all_roles_autonomous
            and self.role_complete_both_teams
            and self.safe
            and not self.motor_fault_agents
        )

    @property
    def result_hash(self) -> str:
        return str(hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        return {
            **(
                {"keeper_policy_hashes": dict(self.keeper_policy_hashes)}
                if self.keeper_policy_hashes
                else {}
            ),
            **(
                {
                    "motor_policy_hashes": dict(self.motor_policy_hashes),
                    "motor_fault_agents": list(self.motor_fault_agents),
                }
                if self.motor_policy_hashes
                else {}
            ),
            "schema_version": self.schema_version,
            "scenario_hash": self.scenario_hash,
            "roster_hash": self.roster_hash,
            "config_hash": self.config_hash,
            "trajectory_hash": self.trajectory_hash,
            "player_count": self.player_count,
            "red_player_count": self.red_player_count,
            "blue_player_count": self.blue_player_count,
            "decision_frame_count": self.decision_frame_count,
            "coordination_frame_hashes": list(self.coordination_frame_hashes),
            "qualities": [value.to_dict() for value in self.qualities],
            "pass_handshake_count": self.pass_handshake_count,
            "pass_intent_count": self.pass_intent_count,
            "shot_intent_count": self.shot_intent_count,
            "save_intent_count": self.save_intent_count,
            "distribution_intent_count": self.distribution_intent_count,
            "finite_state": self.finite_state,
            "robot_robot_contact_count": self.robot_robot_contact_count,
            "rolling_distance_m": self.rolling_distance_m,
            "peak_ball_speed_mps": self.peak_ball_speed_mps,
            "all_agents_active": self.all_agents_active,
            "all_roles_autonomous": self.all_roles_autonomous,
            "safe": self.safe,
            "passed": self.passed,
            "activation_ceiling": self.activation_ceiling,
            "physics_authority": self.physics_authority,
            "hardware_command_sent": self.hardware_command_sent,
            "pixels_used_for_scoring": self.pixels_used_for_scoring,
        }


@dataclass
class _PlayerController:
    spec: G1PitchPlayerSpec
    cell: RosclawSoccerAgentCell
    qpos_base: int
    qvel_base: int
    joint_ids: NDArray[np.int64]
    joint_qpos: NDArray[np.int64]
    joint_qvel: NDArray[np.int64]
    actuators: NDArray[np.int64]
    pelvis_body: int
    torso_body: int
    left_ankle_body: int
    right_ankle_body: int
    robot_geoms: frozenset[int]
    left_foot_geoms: frozenset[int]
    right_foot_geoms: frozenset[int]
    left_glove_geoms: frozenset[int]
    right_glove_geoms: frozenset[int]
    state: Any
    output: Any
    policy: Any
    decision: AgentCellDecision | None = None
    last_world_command: NDArray[np.float64] | None = None
    locomotion_reflection_frame: bool | None = None
    locomotion_frame_switch_count: int = 0
    clearance_feasible: bool = True
    clearance_constrained: bool = False
    current_intent: TacticalIntent | None = None
    intent_switch_count: int = 0
    seen_intents: set[TacticalIntent] | None = None
    active_frames: int = 0
    decision_count: int = 0
    pass_intent_count: int = 0
    shot_intent_count: int = 0
    save_intent_count: int = 0
    distribution_intent_count: int = 0
    minimum_pelvis_height_m: float = math.inf
    maximum_tilt_rad: float = 0.0
    joint_limit_violation: bool = False
    torque_limit_violation: bool = False
    kick_output: Any | None = None
    kick_policy: Any | None = None
    option_active: bool = False
    option_activation_frame: int | None = None
    option_origin_target: NDArray[np.float64] | None = None
    option_origin_kp: NDArray[np.float64] | None = None
    option_origin_kd: NDArray[np.float64] | None = None
    option_parameters: ShotParameters | None = None
    option_contact_observed: bool = False
    option_completed: bool = False
    last_ball_contact_foot: str | None = None
    post_receive_joint_target: NDArray[np.float64] | None = None
    strike_phase: StrikePhaseState = field(default_factory=StrikePhaseState)
    pass_stroke: ContactStroke = field(default_factory=ContactStroke)
    keeper_reach: SharedKeeperReach | None = None

    def __post_init__(self) -> None:
        if self.seen_intents is None:
            self.seen_intents = set()


def simulate_independent_team_world(
    *,
    asset_root: Path,
    roster: TeamRoleRoster,
    cells: tuple[RosclawSoccerAgentCell, ...],
    players: tuple[G1PitchPlayerSpec, ...],
    scenario: IndependentTeamWorldScenario,
    goal: G1TrainingGoalSpec,
    config: IndependentTeamWorldConfig | None = None,
    contact_teacher_config: G1LocomotionContactTeacherConfig | None = None,
    option_bridge_config: G1RollingOptionBridgeConfig | None = None,
    strike_phase_config: StrikePhaseConfig | None = None,
    strike_coordination_actor: DynamicStrikeCoordinationActor | None = None,
    contextual_strike_memory: ContextualStrikeExpertMemory | None = None,
    near_ball_policy: NearBallResidualPolicy | None = None,
    near_ball_seed: int = 0,
    near_ball_explore: bool = False,
    near_ball_exploration_agent_ids: tuple[str, ...] | None = None,
    motor_options: Mapping[str, TeamMotorOption] | None = None,
) -> tuple[IndependentTeamWorldResult, dict[str, NDArray[Any]]]:
    """Run all agent cells and all neural locomotion bodies in one clock."""

    active = config or IndependentTeamWorldConfig()
    cell_by_id = {cell.agent_id: cell for cell in cells}
    player_by_id = {player.agent_id: player for player in players}
    roster_ids = {agent.agent_id for agent in roster.agents}
    exploration_mask: NDArray[np.bool_] | None = None
    if near_ball_exploration_agent_ids is not None:
        scope = near_ball_exploration_agent_ids
        if (
            near_ball_explore is not True
            or near_ball_policy is None
            or type(scope) is not tuple
            or not scope
            or any(type(agent) is not str for agent in scope)
            or tuple(sorted(set(scope))) != scope
            or not set(scope).issubset(roster_ids)
        ):
            raise ValueError("scoped exploration requires a policy and explicit roster subset")
        exploration_mask = np.asarray(
            [agent in scope for agent in near_ball_policy.agent_ids], dtype=bool
        )
    motors = dict(motor_options or {})
    if (
        not set(motors).issubset(roster_ids)
        or len({id(motor) for motor in motors.values()}) != len(motors)
        or any(not _HASH.fullmatch(motor.contract_hash) for motor in motors.values())
        or motors
        and option_bridge_config is not None
        or active.motor_idle_residual_fallback
        and (not motors or near_ball_policy is None)
        or active.keeper_reach is not None
        and any(
            agent.agent_id in motors and agent.primary_role is MatchRole.GOALKEEPER
            for agent in roster.agents
        )
    ):
        raise ValueError("per-player motor identity, binding or override ownership differs")
    motor_hashes = tuple(sorted((key, motor.contract_hash) for key, motor in motors.items()))
    motor_faults: set[str] = set()
    if (
        len(roster.agents) < 6
        or set(cell_by_id) != roster_ids
        or set(player_by_id) != roster_ids
        or len(cell_by_id) != len(cells)
        or len(player_by_id) != len(players)
        or (strike_coordination_actor is not None and strike_phase_config is None)
        or (contextual_strike_memory is not None and strike_phase_config is None)
        or (strike_coordination_actor is not None and contextual_strike_memory is not None)
    ):
        raise ValueError("independent team world roster/cell/body identities differ")
    for cell in cells:
        if cell.self_model.self_model_hash != roster.agent(cell.agent_id).self_model_hash:
            raise ValueError("independent team world cell changed after roster commitment")
        if abs(cell.tactical_profile.decision_period_sec - active.decision_period_sec) > 1.0e-12:
            raise ValueError("agent and world decision clocks differ")
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    residual_ids = tuple(sorted(cell_by_id))
    if near_ball_policy is not None and (
        near_ball_policy.agent_ids != residual_ids
        or near_ball_policy.body_hash != qualification.body_hash
    ):
        raise ValueError("residual policy differs from the qualified body/roster")
    residual_rng = np.random.default_rng(near_ball_seed)
    residual_previous = np.zeros((len(residual_ids), 12), dtype=np.float64)
    prospective_team_contact = any(cell.tactical_profile.anticipatory_contact for cell in cells)

    import mujoco

    model = build_g1_multi_player_stadium_model(
        asset_root,
        players=players,
        spec=goal,
        left_goal_plane_x_m=active.left_goal_plane_x_m if active.bilateral_goals else None,
    )
    model.opt.timestep = _PHYSICS_DT
    if active.glove_material is not None:
        for player in players:
            if player.goalkeeper_gloves:
                active.glove_material.apply(model, prefix=player.body_prefix)
    data = mujoco.MjData(model)
    state_type, output_type, kick_type, _ = load_robonaldo(qualification.asset_root)
    loco_type = importlib.import_module("policy.loco_mode.LocoMode").LocoMode
    with np.load(qualification.asset_root / _MOTION_REL) as motion:
        pelvis_height = float(np.asarray(motion["body_pos_w"])[0, 0, 2])
    controllers = tuple(
        _make_player_controller(
            model=model,
            data=data,
            spec=player_by_id[agent.agent_id],
            cell=cell_by_id[agent.agent_id],
            pelvis_height=pelvis_height,
            state_type=state_type,
            output_type=output_type,
            loco_type=loco_type,
            kick_type=kick_type if option_bridge_config is not None else None,
        )
        for agent in sorted(roster.agents, key=lambda item: item.agent_id)
    )
    ball_body = _id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    if active.keeper_reach is not None:
        for controller in controllers:
            if controller.cell.self_model.primary_role is MatchRole.GOALKEEPER:
                if not controller.spec.goalkeeper_gloves:
                    raise ValueError("keeper reach requires standard goalkeeper glove geometry")
                controller.keeper_reach = SharedKeeperReach(
                    asset_root=asset_root,
                    goal=goal,
                    frame=KeeperFrame(controller.spec.origin_m[:2], controller.spec.yaw_rad),
                    prefix=controller.spec.body_prefix,
                    config=active.keeper_reach,
                )
    ball_geom = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
    ball_joint = _id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
    ball_qpos = int(model.jnt_qposadr[ball_joint])
    ball_qvel = int(model.jnt_dofadr[ball_joint])
    data.qpos[ball_qpos : ball_qpos + 3] = scenario.ball_initial_position_m
    data.qpos[ball_qpos + 3 : ball_qpos + 7] = (1.0, 0.0, 0.0, 0.0)
    data.qvel[ball_qvel : ball_qvel + 3] = scenario.ball_initial_velocity_mps
    data.qvel[ball_qvel + 3 : ball_qvel + 6] = 0.0
    mujoco.mj_forward(model, data)

    total_frames = int(round(active.simulation_duration_sec / _CONTROL_DT))
    decision_stride = max(1, int(round(active.decision_period_sec / _CONTROL_DT)))
    hard_limits = np.asarray(G1_HARD_TORQUE_LIMITS, dtype=np.float64)
    guarded_limits = 0.85 * hard_limits
    trace: dict[str, list[Any]] = {
        "time": [],
        "ball_pose": [],
        "ball_velocity": [],
        "coordination_frame_index": [],
        "possession_agent_code": [],
        "ball_chaser_agent_code": [],
        "receive_lease_agent_code": [],
        "receive_lease_active": [],
        "post_receive_hold_active": [],
        "pass_source_agent_code": [],
        "pass_target_agent_code": [],
        "strike_lease_agent_code": [],
        "first_touch_strike_agent_code": [],
        "strike_stance_depth_m": [],
        "strike_stance_lateral_error_m": [],
        "strike_stance_yaw_error_rad": [],
        "ball_contact_agent_code": [],
        "ball_contact_effector_code": [],
        "ball_contact_foot_code": [],
        "ball_contact_force_n": [],
        "ball_nonfoot_contact_agent_code": [],
        "ball_nonfoot_contact_geom_id": [],
        "ball_nonfoot_contact_force_n": [],
        "robot_robot_contact_count": [],
        "robot_robot_contact_first_code": [],
        "robot_robot_contact_second_code": [],
        "robot_robot_contact_force_n": [],
        "contact_teacher_agent_code": [],
        "contact_teacher_active": [],
        "contact_teacher_peak_torque_nm": [],
        "contact_teacher_last_substep_valid": [],
        "contact_teacher_baseline_torque_nm": [],
        "contact_teacher_residual_torque_nm": [],
        "contact_teacher_tracking_adjustment_nm": [],
        "contact_teacher_guarded_torque_nm": [],
        "contact_teacher_task_force_n": [],
        "contact_teacher_ankle_position_m": [],
        "contact_teacher_mode_code": [],
        "contact_teacher_foot_code": [],
        "contact_teacher_target_m": [],
        "option_agent_code": [],
        "option_policy_frame": [],
        "strike_phase_agent_code": [],
        "strike_phase_code": [],
        "strike_phase_elapsed_sec": [],
        "strike_phase_transition_count": [],
        "strike_phase_abort_code": [],
        "strike_phase_approach_yaw_error_rad": [],
        "strike_phase_predicted_stance_depth_m": [],
        "strike_phase_predicted_stance_lateral_error_m": [],
        "strike_phase_predicted_stance_yaw_error_rad": [],
        "strike_coordination_actor_active": [],
        "strike_coordination_observation": [],
        "strike_coordination_stance_blend": [],
        "strike_coordination_goal_yaw_blend": [],
        "strike_context_memory_consulted": [],
        "strike_context_observation": [],
        "strike_context_expert_index": [],
        "strike_context_normalized_distance": [],
        "strike_context_selection_code": [],
        "strike_context_abstained": [],
    }
    for controller in controllers:
        key = _agent_key(controller.cell.agent_id)
        if controller.keeper_reach is not None:
            for suffix in ("keeper_active", "keeper_residual", "keeper_punch_torque"):
                trace[f"{key}_{suffix}"] = []
        trace.update(
            {
                f"{key}_pelvis_pose": [],
                f"{key}_joint_position": [],
                f"{key}_joint_velocity": [],
                f"{key}_joint_torque": [],
                f"{key}_joint_safety_margin_rad": [],
                f"{key}_left_foot_position": [],
                f"{key}_right_foot_position": [],
                f"{key}_intent_code": [],
                f"{key}_target_position": [],
                f"{key}_world_command": [],
                f"{key}_movement_active": [],
            }
        )
    if active.all_role_clearance:
        for controller in controllers:
            key = _agent_key(controller.cell.agent_id)
            trace[f"{key}_clearance_feasible"] = []
            trace[f"{key}_clearance_constrained"] = []
    initial_positions = {
        controller.cell.agent_id: np.asarray(
            data.qpos[controller.qpos_base : controller.qpos_base + 3], dtype=np.float64
        ).copy()
        for controller in controllers
    }
    coordination_hashes: list[str] = []
    intent_counts = {intent: 0 for intent in TacticalIntent}
    pass_handshake_count = 0
    current_coordination_index = -1
    finite = True
    robot_contact_count = 0
    peak_ball_speed = float(np.linalg.norm(data.qvel[ball_qvel : ball_qvel + 3]))
    initial_ball = np.asarray(data.qpos[ball_qpos : ball_qpos + 3], dtype=np.float64).copy()
    net_state = G1CompliantGoalNetState()
    opposite_net_state = G1CompliantGoalNetState()
    agent_codes = {
        controller.cell.agent_id: index + 1 for index, controller in enumerate(controllers)
    }
    last_ball_contact_agent_id: str | None = None
    last_ball_contact_time_sec = -math.inf
    current_possession_agent_id: str | None = None
    loose_ball_chaser_agent_id: str | None = None
    ball_chaser_lease_start_sec = -math.inf
    receive_lease_agent_id: str | None = None
    receive_lease_source_agent_id: str | None = None
    receive_lease_origin_m: NDArray[np.float64] | None = None
    receive_lease_target_m: tuple[float, float] | None = None
    receive_lease_active = False
    flight_tracking_agent_id: str | None = None
    receive_handoff: PassHandoff | None = None
    handoff_cancellations = 0
    last_receive_contact_agent_id: str | None = None
    last_receive_contact_time_sec = -math.inf
    pass_source_agent_id: str | None = None
    pass_target_agent_id: str | None = None
    strike_lease_agent_id: str | None = None
    strike_lease_start_sec = -math.inf
    assigned_ball_chaser_agent_id: str | None = None
    contextual_strike_key: tuple[str, float] | None = None
    contextual_strike_expert_index = -1
    contextual_strike_distance = 0.0
    contextual_strike_selection_code = 0
    contextual_strike_context = np.zeros(5, dtype=np.float64)

    if active.strict_receive_handoff:
        trace["pass_handoff_source_contact_sec"] = []
        trace["pass_handoff_active"] = []
        trace["pass_handoff_cancellations"] = []
        trace["pass_feedback_launch_relative"] = []

    if near_ball_policy is not None:
        for name in (
            "residual_observations",
            "residual_latent",
            "residual_log_probability",
            "residual_value",
            "residual_active",
            "residual_applied",
        ):
            trace[name] = []
    for frame in range(total_frames):
        for controller in controllers:
            _fill_locomotion_state(controller, data, ball_body, ball_qvel)
        if frame % decision_stride == 0:
            physical_states = tuple(
                _physical_state(controller, data, world_velocity=active.bilateral_goals)
                for controller in controllers
            )
            state_by_id = {state.agent_id: state for state in physical_states}
            if receive_handoff is not None and receive_handoff.expired(
                float(data.time), receiver_stable=state_by_id[receive_handoff.receiver].stable
            ):
                receive_lease_agent_id = None
                receive_lease_source_agent_id = None
                receive_lease_origin_m = None
                receive_lease_target_m = None
                receive_lease_active = False
                receive_handoff = None
                handoff_cancellations += 1
            proximity_possession = _infer_possession(
                controllers=controllers,
                data=data,
                ball_position=np.asarray(data.qpos[ball_qpos : ball_qpos + 3]),
                maximum_distance_m=active.possession_radius_m,
            )
            current_possession_agent_id = (
                last_ball_contact_agent_id
                if last_ball_contact_agent_id is not None
                and float(data.time) - last_ball_contact_time_sec
                <= active.contact_possession_hold_sec
                # Training with a physical contact teacher uses a strict
                # contact-derived owner.  Proximity may still elect a chaser,
                # but it cannot launch a PASS/SHOOT option by itself.
                else proximity_possession
                if contact_teacher_config is None
                else None
            )
            assigned_ball_chaser_agent_id = None
            if contact_teacher_config is not None:
                if current_possession_agent_id is None:
                    if (
                        receive_lease_agent_id is not None
                        and receive_lease_origin_m is not None
                        and state_by_id[receive_lease_agent_id].stable
                        and (
                            not active.strict_receive_handoff
                            or (
                                receive_handoff is not None
                                and receive_handoff.can_activate(float(data.time), progressed=True)
                            )
                        )
                        and (
                            receive_lease_active
                            or (
                                active.strict_receive_handoff
                                and float(np.linalg.norm(data.qvel[ball_qvel : ball_qvel + 2]))
                                >= 0.10
                            )
                            or float(
                                np.linalg.norm(
                                    np.asarray(
                                        data.qpos[ball_qpos : ball_qpos + 2], dtype=np.float64
                                    )
                                    - receive_lease_origin_m
                                )
                            )
                            >= active.minimum_receive_lease_progress_m
                            or float(
                                np.linalg.norm(
                                    np.asarray(
                                        data.qvel[ball_qvel : ball_qvel + 2], dtype=np.float64
                                    )
                                )
                            )
                            >= active.minimum_receive_lease_ball_speed_mps
                        )
                    ):
                        receive_lease_active = True
                        if loose_ball_chaser_agent_id != receive_lease_agent_id:
                            loose_ball_chaser_agent_id = receive_lease_agent_id
                            ball_chaser_lease_start_sec = float(data.time)
                    else:
                        (
                            loose_ball_chaser_agent_id,
                            ball_chaser_lease_start_sec,
                        ) = _select_loose_ball_chaser(
                            controllers=controllers,
                            states=state_by_id,
                            ball_position=np.asarray(
                                data.qpos[ball_qpos : ball_qpos + 2], dtype=np.float64
                            ),
                            current_agent_id=loose_ball_chaser_agent_id,
                            lease_start_sec=ball_chaser_lease_start_sec,
                            time_sec=float(data.time),
                            config=active,
                        )
                    assigned_ball_chaser_agent_id = loose_ball_chaser_agent_id
                else:
                    assigned_ball_chaser_agent_id = _select_pressing_chaser(
                        controllers=controllers,
                        states=state_by_id,
                        possession_agent_id=current_possession_agent_id,
                        ball_position=np.asarray(
                            data.qpos[ball_qpos : ball_qpos + 2], dtype=np.float64
                        ),
                    )
            flight_tracking_agent_id = None
            if (
                active.receiver_commitment_priority
                and receive_handoff is not None
                and receive_lease_agent_id == receive_handoff.receiver
                and current_possession_agent_id in {None, receive_handoff.source}
                and receive_handoff.can_track_incoming_ball(
                    float(data.time),
                    ball_xy=(float(data.qpos[ball_qpos]), float(data.qpos[ball_qpos + 1])),
                    ball_velocity_xy=(float(data.qvel[ball_qvel]), float(data.qvel[ball_qvel + 1])),
                    receiver_xy=state_by_id[receive_handoff.receiver].position_m[:2],
                )
            ):
                flight_tracking_agent_id = receive_handoff.receiver
            observations = tuple(
                _agent_observation(
                    controller=controller,
                    roster=roster,
                    state_by_id=state_by_id,
                    data=data,
                    ball_qpos=ball_qpos,
                    ball_qvel=ball_qvel,
                    goal=goal,
                    left_goal_plane_x_m=active.left_goal_plane_x_m,
                    possession_agent_id=current_possession_agent_id,
                    ball_chaser_agent_id=assigned_ball_chaser_agent_id,
                    active_receive_source_agent_id=(
                        receive_lease_source_agent_id
                        if active.receiver_commitment_priority
                        and (
                            receive_lease_active
                            or flight_tracking_agent_id == controller.cell.agent_id
                        )
                        and receive_lease_agent_id == controller.cell.agent_id
                        else None
                    ),
                )
                for controller in controllers
            )
            decisions = tuple(
                cell_by_id[observation.observer_agent_id].decide(observation)
                for observation in observations
            )
            coordination = build_team_coordination_frame(
                roster=roster,
                cells=cells,
                observations=observations,
                decisions=decisions,
                frame_index=len(coordination_hashes),
            )
            coordination_hashes.append(coordination.frame_hash)
            pass_handshake_count += len(coordination.pass_receive_handshakes)
            pass_source_agent_id = None
            pass_target_agent_id = None
            if current_possession_agent_id is not None or prospective_team_contact:
                current_handshake = next(
                    (
                        handshake
                        for handshake in coordination.pass_receive_handshakes
                        if handshake.passer_agent_id
                        == (current_possession_agent_id or assigned_ball_chaser_agent_id)
                    ),
                    None,
                )
                if current_handshake is not None and (
                    receive_lease_source_agent_id != current_handshake.passer_agent_id
                    or receive_lease_agent_id != current_handshake.receiver_agent_id
                    or receive_lease_origin_m is None
                ):
                    receive_lease_source_agent_id = current_handshake.passer_agent_id
                    receive_lease_agent_id = current_handshake.receiver_agent_id
                    receive_lease_origin_m = np.asarray(
                        data.qpos[ball_qpos : ball_qpos + 2], dtype=np.float64
                    ).copy()
                    receive_lease_active = False
                    flight_tracking_agent_id = None
                    receive_lease_target_m = current_handshake.pass_target_m[:2]
                    if active.strict_receive_handoff:
                        receive_handoff = PassHandoff(
                            current_handshake.passer_agent_id,
                            current_handshake.receiver_agent_id,
                            float(data.time),
                        )
                if current_handshake is not None:
                    pass_source_agent_id = current_handshake.passer_agent_id
                    pass_target_agent_id = current_handshake.receiver_agent_id
            for decision in decisions:
                intent_counts[decision.intent] += 1
                if (
                    decision.intent is TacticalIntent.SHOOT
                    and decision.agent_id == current_possession_agent_id
                    and strike_lease_agent_id != decision.agent_id
                ):
                    strike_lease_agent_id = decision.agent_id
                    strike_lease_start_sec = float(data.time)
            current_coordination_index += 1
            for decision in decisions:
                controller = next(
                    item for item in controllers if item.cell.agent_id == decision.agent_id
                )
                if controller.current_intent is not None and (
                    controller.current_intent is not decision.intent
                ):
                    controller.intent_switch_count += 1
                controller.current_intent = decision.intent
                if controller.decision is None or (
                    controller.decision.intent != decision.intent
                    or controller.decision.target_agent_id != decision.target_agent_id
                ):
                    controller.pass_stroke.reset()
                assert controller.seen_intents is not None
                controller.seen_intents.add(decision.intent)
                controller.decision = decision
                controller.decision_count += 1
                controller.pass_intent_count += int(decision.intent is TacticalIntent.PASS)
                controller.shot_intent_count += int(decision.intent is TacticalIntent.SHOOT)
                controller.save_intent_count += int(decision.intent is TacticalIntent.SAVE)
                controller.distribution_intent_count += int(
                    decision.intent is TacticalIntent.DISTRIBUTE
                )

        positions = {
            controller.cell.agent_id: np.asarray(
                data.qpos[controller.qpos_base : controller.qpos_base + 2], dtype=np.float64
            ).copy()
            for controller in controllers
        }
        teacher_controller = _select_contact_teacher_controller(
            controllers=controllers,
            data=data,
            ball_position=np.asarray(data.qpos[ball_qpos : ball_qpos + 3]),
            current_possession_agent_id=current_possession_agent_id,
            preferred_agent_id=assigned_ball_chaser_agent_id,
            config=contact_teacher_config,
            nearest_contact_first=active.stationary_ball_acquisition,
        )
        teacher_direction = (
            np.zeros(2, dtype=np.float64)
            if teacher_controller is None
            else _contact_teacher_direction(
                teacher_controller,
                data=data,
                ball_qpos=ball_qpos,
                goal=goal,
                left_goal_plane_x_m=active.left_goal_plane_x_m,
            )
        )
        frame_teacher_active = False
        frame_teacher_peak_torque_nm = 0.0
        frame_teacher_last_substep_valid = False
        frame_teacher_baseline = np.zeros(29, dtype=np.float64)
        frame_teacher_residual = np.zeros(29, dtype=np.float64)
        frame_teacher_adjustment = np.zeros(29, dtype=np.float64)
        frame_teacher_guarded = np.zeros(29, dtype=np.float64)
        frame_teacher_force = np.zeros(3, dtype=np.float64)
        frame_teacher_ankle = np.zeros(3, dtype=np.float64)
        frame_teacher_mode_code = 0
        frame_teacher_foot_code = 0
        frame_teacher_target_m: NDArray[np.float64] = np.zeros(3, dtype=np.float64)
        if (
            option_bridge_config is not None
            and strike_lease_agent_id is not None
            and float(data.time) - strike_lease_start_sec
            > option_bridge_config.strike_lease_duration_sec
            and not (
                strike_phase_config is not None
                and any(
                    controller.cell.agent_id == strike_lease_agent_id
                    and controller.strike_phase.active
                    for controller in controllers
                )
            )
        ):
            strike_lease_agent_id = None
        phase_controller = next(
            (
                controller
                for controller in controllers
                if controller.strike_phase.phase
                not in {StrikePhase.IDLE, StrikePhase.COMPLETE, StrikePhase.ABORTED}
            ),
            None,
        )
        frame_phase_approach_yaw_error = 0.0
        phase_metrics = (0.0, 0.0, 0.0)
        frame_coordination_observation = np.zeros(6, dtype=np.float64)
        frame_coordination_action = StrikeCoordinationAction(0.0, 0.0)
        frame_coordination_actor_active = False
        frame_context_observation = np.zeros(5, dtype=np.float64)
        frame_context_memory_consulted = False
        frame_context_expert_index = -1
        frame_context_normalized_distance = 0.0
        frame_context_selection_code = 0
        frame_context_abstained = False
        if phase_controller is not None and strike_phase_config is not None:
            phase_metrics = _controller_strike_stance_metrics(
                controller=phase_controller,
                data=data,
                ball_qpos=ball_qpos,
                goal=goal,
                ball_qvel=ball_qvel,
                prediction_horizon_sec=strike_phase_config.strike_contact_horizon_sec,
            )
            ball_speed = float(np.linalg.norm(data.qvel[ball_qvel : ball_qvel + 3]))
            if (
                strike_coordination_actor is not None or contextual_strike_memory is not None
            ) and phase_controller.strike_phase.phase is StrikePhase.ORIENT:
                base_approach_yaw_error = _strike_tracking_yaw_error(
                    controller=phase_controller,
                    data=data,
                    ball_qpos=ball_qpos,
                    ball_qvel=ball_qvel,
                    goal=goal,
                    config=strike_phase_config,
                )
                observation = StrikeCoordinationObservation(
                    phase_progress=min(
                        2.0,
                        max(
                            0.0,
                            (float(data.time) - phase_controller.strike_phase.phase_enter_time_sec)
                            / strike_phase_config.orient_timeout_sec,
                        ),
                    ),
                    stance_depth_m=phase_metrics[0],
                    stance_lateral_error_m=phase_metrics[1],
                    stance_yaw_error_rad=phase_metrics[2],
                    approach_yaw_error_rad=base_approach_yaw_error,
                    ball_speed_mps=ball_speed,
                )
                frame_coordination_observation = observation.vector()
                if strike_coordination_actor is not None:
                    frame_coordination_action = strike_coordination_actor.act(observation)
                    frame_coordination_actor_active = True
                else:
                    assert contextual_strike_memory is not None
                    selection_key = (
                        phase_controller.cell.agent_id,
                        phase_controller.strike_phase.phase_enter_time_sec,
                    )
                    if contextual_strike_key != selection_key:
                        task_context = build_strike_task_context(
                            goal_target_m=(goal.plane_x_m, goal.target_y_m),
                            ball_position_m=np.asarray(
                                data.qpos[ball_qpos : ball_qpos + 2], dtype=np.float64
                            ),
                            opponent_positions_m=np.asarray(
                                [
                                    positions[agent_id]
                                    for agent_id in phase_controller.cell.self_model.opponent_ids
                                ],
                                dtype=np.float64,
                            ),
                        )
                        selection = contextual_strike_memory.select(
                            context=task_context,
                            observation=observation,
                        )
                        contextual_strike_key = selection_key
                        contextual_strike_expert_index = selection.expert_index
                        contextual_strike_distance = selection.normalized_distance
                        contextual_strike_selection_code = {
                            "verified-expert": 1,
                            "negative-memory": 2,
                            "out-of-support": 3,
                        }[selection.reason]
                        contextual_strike_context = task_context.vector()
                        frame_context_memory_consulted = True
                    frame_context_observation = contextual_strike_context
                    frame_context_expert_index = contextual_strike_expert_index
                    frame_context_normalized_distance = contextual_strike_distance
                    frame_context_selection_code = contextual_strike_selection_code
                    frame_context_abstained = contextual_strike_expert_index < 0
                    if contextual_strike_expert_index >= 0:
                        frame_coordination_action = contextual_strike_memory.experts[
                            contextual_strike_expert_index
                        ].actor.act(observation)
                        frame_coordination_actor_active = True
            approach_yaw_error = _strike_tracking_yaw_error(
                controller=phase_controller,
                data=data,
                ball_qpos=ball_qpos,
                ball_qvel=ball_qvel,
                goal=goal,
                config=strike_phase_config,
                goal_yaw_blend=frame_coordination_action.goal_yaw_blend,
            )
            frame_phase_approach_yaw_error = approach_yaw_error
            ball_position = np.asarray(data.qpos[ball_qpos : ball_qpos + 3], dtype=np.float64)
            ball_distance = min(
                float(np.linalg.norm(data.xpos[phase_controller.left_ankle_body] - ball_position)),
                float(np.linalg.norm(data.xpos[phase_controller.right_ankle_body] - ball_position)),
            )
            phase_controller.strike_phase.advance(
                time_sec=float(data.time),
                stable=bool(
                    data.qpos[phase_controller.qpos_base + 2] >= active.minimum_pelvis_height_m
                    and max(
                        abs(value)
                        for value in _roll_pitch(
                            np.asarray(
                                data.qpos[
                                    phase_controller.qpos_base + 3 : phase_controller.qpos_base + 7
                                ],
                                dtype=np.float64,
                            )
                        )
                    )
                    <= active.maximum_tilt_rad
                ),
                stance_depth_m=phase_metrics[0],
                stance_lateral_error_m=phase_metrics[1],
                stance_yaw_error_rad=phase_metrics[2],
                approach_yaw_error_rad=approach_yaw_error,
                ball_speed_mps=ball_speed,
                ball_distance_m=ball_distance,
                option_active=phase_controller.option_active,
                option_contact_observed=phase_controller.option_contact_observed,
                option_completed=phase_controller.option_completed,
                config=strike_phase_config,
            )
        _activate_rolling_option(
            controllers=controllers,
            current_possession_agent_id=current_possession_agent_id,
            last_ball_contact_agent_id=last_ball_contact_agent_id,
            strike_lease_agent_id=strike_lease_agent_id,
            frame=frame,
            data=data,
            ball_qpos=ball_qpos,
            goal=goal,
            config=option_bridge_config,
            phase_config=strike_phase_config,
            left_goal_plane_x_m=active.left_goal_plane_x_m if active.bilateral_goals else None,
            prospective_agent_id=(
                assigned_ball_chaser_agent_id if current_possession_agent_id is None else None
            ),
        )
        option_controller = next(
            (controller for controller in controllers if controller.option_active),
            None,
        )
        option_policy_frame = 0
        option_override: (
            tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]] | None
        ) = None
        for controller in controllers:
            current_decision = controller.decision
            if current_decision is None:
                raise RuntimeError("independent agent has no current decision")
            command = _movement_command(
                controller=controller,
                decision=current_decision,
                positions=positions,
                data=data,
                ball_qpos=ball_qpos,
                ball_qvel=ball_qvel,
                possession_agent_id=current_possession_agent_id,
                committed_receiver=bool(receive_lease_agent_id == controller.cell.agent_id),
                active_receiver=bool(
                    (receive_lease_active or flight_tracking_agent_id == controller.cell.agent_id)
                    and receive_lease_agent_id == controller.cell.agent_id
                ),
                post_receive_hold=bool(
                    last_receive_contact_agent_id == controller.cell.agent_id
                    and float(data.time) - last_receive_contact_time_sec
                    <= active.post_receive_hold_sec
                    and not (
                        contact_teacher_config is not None
                        and contact_teacher_config.one_touch_finish_enabled
                        and controller.cell.self_model.primary_role is MatchRole.FINISHER
                    )
                ),
                receive_foot_lateral_offset_m=(
                    0.18
                    if contact_teacher_config is None
                    else contact_teacher_config.committed_receive_ankle_lateral_offset_m
                ),
                strike_target_position_m=(
                    None
                    if strike_lease_agent_id != controller.cell.agent_id
                    else (
                        goal.plane_x_m
                        if controller.cell.self_model.team_id == "red"
                        else active.left_goal_plane_x_m,
                        0.0 if active.bilateral_goals else goal.target_y_m,
                    )
                ),
                strike_phase=(
                    controller.strike_phase.phase
                    if strike_phase_config is not None
                    and controller.strike_phase.phase is not StrikePhase.IDLE
                    else None
                ),
                strike_phase_config=strike_phase_config,
                strike_phase_owner_agent_id=(
                    None if phase_controller is None else phase_controller.cell.agent_id
                ),
                strike_coordination_action=(
                    frame_coordination_action
                    if phase_controller is controller and frame_coordination_actor_active
                    else None
                ),
                config=active,
                prospective_contact=prospective_team_contact,
                continuous_motor_approach=bool(
                    controller.cell.agent_id in motors
                    and controller.cell.agent_id not in motor_faults
                ),
                pending_receive_target_m=(
                    receive_lease_target_m
                    if prospective_team_contact
                    and receive_lease_agent_id == controller.cell.agent_id
                    else None
                ),
            )
            current_yaw = _pelvis_yaw(
                np.asarray(
                    data.qpos[controller.qpos_base + 3 : controller.qpos_base + 7],
                    dtype=np.float64,
                )
            )
            local_command = _rotate_z(command, -current_yaw)
            controller.state.vel_cmd = _normalized_locomotion_command(
                controller.policy, local_command
            )
            _run_locomotion(
                controller,
                synchronize_action_frame=active.locomotion_action_frame_sync,
                mirror=bool(local_command[1] < -1.0e-6),
                correct_mirrored_yaw=bool(
                    active.bilateral_goals
                    or (strike_phase_config is not None and controller.strike_phase.active)
                ),
            )
            controller.last_world_command = command.copy()
            if controller.keeper_reach is not None:
                controller.output.actions = controller.keeper_reach.step(
                    model, data, np.asarray(controller.output.actions, dtype=np.float64)
                )
                controller.output.kps, controller.output.kds = controller.keeper_reach.impedance(
                    controller.output.kps, controller.output.kds
                )
            controller.active_frames += int(float(np.linalg.norm(command[:2])) >= 0.04)
            if controller is option_controller:
                option_override, option_policy_frame = _rolling_option_target(
                    controller,
                    frame=frame,
                    config=option_bridge_config,
                )
        if active.locomotion_action_frame_sync:
            trace.setdefault("locomotion_reflection_frame", []).append(
                [c.locomotion_reflection_frame is True for c in controllers]
            )
            trace.setdefault("locomotion_frame_switch_count", []).append(
                [c.locomotion_frame_switch_count for c in controllers]
            )
        motor_targets: dict[str, TeamMotorTarget] = {}
        for controller in controllers:
            agent_id = controller.cell.agent_id
            if agent_id not in motors or agent_id in motor_faults:
                continue
            assert controller.decision is not None
            intent = controller.decision.intent.value
            q = np.r_[
                data.qpos[controller.qpos_base : controller.qpos_base + 7],
                data.qpos[controller.joint_qpos],
                data.qpos[ball_qpos : ball_qpos + 7],
            ]
            v = np.r_[
                data.qvel[controller.qvel_base : controller.qvel_base + 6],
                data.qvel[controller.joint_qvel],
                data.qvel[ball_qvel : ball_qvel + 6],
            ]
            try:
                if controller.last_world_command is None:
                    raise ValueError("shared navigation command is unavailable")
                proposal = motors[agent_id].propose(
                    TeamMotorObservation(
                        agent_id=agent_id,
                        frame=frame,
                        time_sec=float(data.time),
                        intent=intent if intent in {"shoot", "pass", "carry"} else "other",
                        prospective_owner=agent_id
                        in {current_possession_agent_id, assigned_ball_chaser_agent_id},
                        qpos=tuple(float(x) for x in q),
                        qvel=tuple(float(x) for x in v),
                        target_position_m=(
                            controller.decision.target_position_m[0],
                            controller.decision.target_position_m[1],
                            controller.decision.target_position_m[2],
                        ),
                        navigation_command=(
                            float(controller.last_world_command[0]),
                            float(controller.last_world_command[1]),
                            float(controller.last_world_command[2]),
                        ),
                        # Preparation starts at the accepted pass handshake,
                        # not only after the source has physically launched it.
                        committed_receiver=receive_lease_agent_id == agent_id,
                    )
                )
                if proposal is not None:
                    if not isinstance(proposal, TeamMotorTarget):
                        raise ValueError("motor returned a non-contract proposal")
                    motor_targets[agent_id] = TeamMotorTarget(
                        proposal.target_rad, proposal.kp, proposal.kd
                    )
            except (ValueError, TypeError, FloatingPointError):
                # The frozen locomotion controller remains the fallback. This
                # motor is latched off for the match; no silent candidate retry.
                motor_faults.add(agent_id)
        if motors:
            trace.setdefault("full_body_motor_active", []).append(
                [c.cell.agent_id in motor_targets for c in controllers]
            )
            trace.setdefault("full_body_motor_fault", []).append(
                [c.cell.agent_id in motor_faults for c in controllers]
            )
        residual_by_id: dict[str, NDArray[np.float64]] = {}
        residual_blocked = {
            agent_id
            for agent_id in motors
            if motor_blocks_residual(
                registered=True,
                proposed=agent_id in motor_targets,
                faulted=agent_id in motor_faults,
                allow_idle_fallback=active.motor_idle_residual_fallback,
            )
        }
        if near_ball_policy is not None:
            ordered = tuple(
                next(c for c in controllers if c.cell.agent_id == agent_id)
                for agent_id in residual_ids
            )
            observations_np = np.stack(
                [
                    _near_ball_observation(
                        c,
                        data=data,
                        ball_qpos=ball_qpos,
                        ball_qvel=ball_qvel,
                        previous=residual_previous[i],
                    )
                    for i, c in enumerate(ordered)
                ]
            )
            residual_active = np.asarray(
                [
                    c is teacher_controller
                    and c is not option_controller
                    and c.cell.agent_id not in residual_blocked
                    and not (
                        last_receive_contact_agent_id == c.cell.agent_id
                        and float(data.time) - last_receive_contact_time_sec
                        <= active.post_receive_hold_sec
                    )
                    and c.decision is not None
                    and _residual_intent_enabled(
                        c.decision.intent, strike_enabled=active.strike_residual_enabled
                    )
                    and data.qpos[c.qpos_base + 2] >= active.minimum_pelvis_height_m
                    and min(
                        np.linalg.norm(
                            data.xpos[c.left_ankle_body] - data.qpos[ball_qpos : ball_qpos + 3]
                        ),
                        np.linalg.norm(
                            data.xpos[c.right_ankle_body] - data.qpos[ball_qpos : ball_qpos + 3]
                        ),
                    )
                    <= 0.75
                    for c in ordered
                ],
                dtype=bool,
            )
            latent, log_probability, values = near_ball_policy.act(
                observations_np,
                residual_rng,
                explore=near_ball_explore,
                **({"exploration_mask": exploration_mask} if exploration_mask is not None else {}),
            )
            if exploration_mask is not None:
                trace.setdefault("residual_exploration_mask", []).append(exploration_mask.copy())
            residual_previous = bounded_residual(latent, residual_previous, residual_active)
            for i, c in enumerate(ordered):
                if (
                    c.cell.agent_id in residual_blocked
                    or c is option_controller
                    or (
                        last_receive_contact_agent_id == c.cell.agent_id
                        and float(data.time) - last_receive_contact_time_sec
                        <= active.post_receive_hold_sec
                    )
                ):
                    residual_previous[i] = 0
                residual_by_id[c.cell.agent_id] = residual_previous[i].copy()
            trace["residual_observations"].append(observations_np)
            trace["residual_latent"].append(latent)
            trace["residual_log_probability"].append(log_probability)
            trace["residual_value"].append(values)
            trace["residual_active"].append(residual_active)
            trace["residual_applied"].append(residual_previous.copy())
        frame_contact_agent_code = 0
        frame_contact_effector_code = 0
        frame_contact_foot_code = 0
        frame_contact_force_n = 0.0
        frame_nonfoot_contact_agent_code = 0
        frame_nonfoot_contact_geom_id = -1
        frame_nonfoot_contact_force_n = 0.0
        frame_robot_contact_count = 0
        frame_robot_contact_first_code = 0
        frame_robot_contact_second_code = 0
        frame_robot_contact_force_n = 0.0
        frame_first_touch_strike_agent_id = next(
            (
                controller.cell.agent_id
                for controller in controllers
                if receive_lease_active
                and receive_lease_agent_id == controller.cell.agent_id
                and controller.cell.self_model.primary_role is MatchRole.FINISHER
            ),
            None,
        )
        for _ in range(_SUBSTEPS):
            frame_teacher_last_substep_valid = False
            frame_teacher_baseline = np.zeros(29, dtype=np.float64)
            frame_teacher_residual = np.zeros(29, dtype=np.float64)
            frame_teacher_adjustment = np.zeros(29, dtype=np.float64)
            frame_teacher_guarded = np.zeros(29, dtype=np.float64)
            frame_teacher_force = np.zeros(3, dtype=np.float64)
            frame_teacher_ankle = np.zeros(3, dtype=np.float64)
            # The learned actor updates targets at 50 Hz, while its high-gain
            # PD loop must close at the 500 Hz physics rate.  Holding one
            # torque sample for the full 20 ms control frame destabilizes the
            # G1 even at zero command; this is the same two-rate contract used
            # by the already-qualified shared-world runner.
            for controller in controllers:
                post_receive_stabilizing = bool(
                    last_receive_contact_agent_id == controller.cell.agent_id
                    and float(data.time) - last_receive_contact_time_sec
                    <= active.post_receive_hold_sec
                    and not (
                        contact_teacher_config is not None
                        and contact_teacher_config.one_touch_finish_enabled
                        and controller.cell.self_model.primary_role is MatchRole.FINISHER
                    )
                )
                one_touch_finishing = bool(
                    contact_teacher_config is not None
                    and contact_teacher_config.one_touch_finish_enabled
                    and controller.cell.self_model.primary_role is MatchRole.FINISHER
                    and last_receive_contact_agent_id == controller.cell.agent_id
                    and float(data.time) - last_receive_contact_time_sec
                    <= contact_teacher_config.contact_memory_sec
                )
                if controller.cell.agent_id in motor_targets:
                    motor_target = motor_targets[controller.cell.agent_id]
                    target, kp, kd = (
                        np.asarray(values, dtype=np.float64)
                        for values in (motor_target.target_rad, motor_target.kp, motor_target.kd)
                    )
                elif controller is option_controller and option_override is not None:
                    target, kp, kd = option_override
                elif (
                    post_receive_stabilizing or one_touch_finishing
                ) and controller.post_receive_joint_target is not None:
                    target = controller.post_receive_joint_target
                    kp = np.asarray(controller.output.kps, dtype=np.float64)
                    kd = np.asarray(controller.output.kds, dtype=np.float64)
                    if one_touch_finishing and contact_teacher_config is not None:
                        kd = kd * contact_teacher_config.one_touch_support_damping_scale
                else:
                    target = np.asarray(controller.output.actions, dtype=np.float64)
                    kp = np.asarray(controller.output.kps, dtype=np.float64)
                    kd = np.asarray(controller.output.kds, dtype=np.float64)
                q = np.asarray(data.qpos[controller.joint_qpos], dtype=np.float64)
                residual = residual_by_id.get(controller.cell.agent_id)
                if (
                    controller.cell.agent_id not in residual_blocked
                    and controller.cell.agent_id not in motor_faults
                    and residual is not None
                    and np.any(residual)
                    and not post_receive_stabilizing
                ):
                    target = target.copy()
                    target[:12] += residual
                dq = np.asarray(data.qvel[controller.joint_qvel], dtype=np.float64)
                raw_torque = kp * (target - q) - kd * dq
                teacher_effect_observed = False
                if controller.keeper_reach is not None:
                    raw_torque += controller.keeper_reach.torque_nm
                if (
                    controller is teacher_controller
                    and controller.cell.agent_id not in motor_targets
                    and controller is not option_controller
                    and not post_receive_stabilizing
                    and not (
                        option_bridge_config is not None
                        and option_bridge_config.prospective_enabled
                        and controller.decision is not None
                        and controller.decision.intent
                        in {TacticalIntent.PASS, TacticalIntent.SHOOT}
                    )
                    and (
                        strike_phase_config is None
                        or not controller.strike_phase.active
                        or controller.strike_phase.phase is StrikePhase.STRIKE
                    )
                    and (
                        controller.cell.agent_id != strike_lease_agent_id
                        or _strike_teacher_stance_ready(
                            controller=controller,
                            data=data,
                            ball_qpos=ball_qpos,
                            goal=goal,
                            minimum_depth_m=0.30,
                            target_xy=(
                                (
                                    goal.plane_x_m
                                    if controller.cell.self_model.team_id == "red"
                                    else active.left_goal_plane_x_m
                                ),
                                0.0,
                            )
                            if active.bilateral_goals
                            else None,
                        )
                    )
                    and contact_teacher_config is not None
                ):
                    ball_position = np.asarray(
                        data.qpos[ball_qpos : ball_qpos + 3], dtype=np.float64
                    )
                    left_distance = float(
                        np.linalg.norm(data.xpos[controller.left_ankle_body] - ball_position)
                    )
                    right_distance = float(
                        np.linalg.norm(data.xpos[controller.right_ankle_body] - ball_position)
                    )
                    use_left = (
                        controller.last_ball_contact_foot == "left"
                        if (post_receive_stabilizing or one_touch_finishing)
                        and controller.last_ball_contact_foot is not None
                        else contact_teacher_config.preferred_foot == "left"
                        or contact_teacher_config.preferred_foot == "nearest"
                        and left_distance < right_distance
                    )
                    contact_recent = bool(
                        last_ball_contact_agent_id == controller.cell.agent_id
                        and float(data.time) - last_ball_contact_time_sec
                        <= contact_teacher_config.contact_memory_sec
                    )
                    ball_linear_velocity = np.asarray(
                        data.qvel[ball_qvel : ball_qvel + 3], dtype=np.float64
                    )
                    receive_intent = bool(
                        controller.decision is not None
                        and controller.decision.intent
                        in {TacticalIntent.RECEIVE, TacticalIntent.INTERCEPT}
                    )
                    is_committed_receiver = bool(
                        receive_lease_active and receive_lease_agent_id == controller.cell.agent_id
                    )
                    contact_mode = (
                        "strike"
                        if one_touch_finishing
                        else "receive"
                        if post_receive_stabilizing
                        or (
                            receive_intent
                            and (
                                is_committed_receiver
                                or active.stationary_ball_acquisition
                                or float(np.linalg.norm(ball_linear_velocity[:2]))
                                >= contact_teacher_config.minimum_receive_ball_speed_mps
                            )
                        )
                        else "strike"
                    )
                    effect_config = contact_teacher_config
                    if one_touch_finishing:
                        effect_config = replace(
                            effect_config,
                            strike_foot_speed_mps=(
                                contact_teacher_config.shot_strike_foot_speed_mps
                            ),
                            aim_yaw_bias_rad=(
                                contact_teacher_config.one_touch_finish_aim_yaw_bias_rad
                            ),
                        )
                    elif (
                        is_committed_receiver or post_receive_stabilizing
                    ) and contact_mode == "receive":
                        effect_config = replace(
                            effect_config,
                            receive_follow_through_speed_mps=(
                                contact_teacher_config.committed_receive_follow_through_speed_mps
                            ),
                            receive_ankle_lateral_offset_m=(
                                contact_teacher_config.committed_receive_ankle_lateral_offset_m
                            ),
                            velocity_damping_n_per_mps=(
                                contact_teacher_config.committed_receive_velocity_damping_n_per_mps
                            ),
                            maximum_task_force_n=(
                                contact_teacher_config.committed_receive_maximum_task_force_n
                            ),
                            maximum_joint_residual_nm=(
                                contact_teacher_config.committed_receive_maximum_joint_residual_nm
                            ),
                            aim_yaw_bias_rad=(
                                contact_teacher_config.committed_receive_aim_yaw_bias_rad
                            ),
                        )
                    if controller.cell.agent_id == strike_lease_agent_id:
                        effect_config = replace(
                            effect_config,
                            strike_foot_speed_mps=(
                                contact_teacher_config.shot_strike_foot_speed_mps
                            ),
                        )
                    elif (
                        controller.decision is not None
                        and controller.decision.intent is TacticalIntent.PASS
                    ):
                        effect_config = replace(
                            effect_config,
                            strike_foot_speed_mps=(
                                contact_teacher_config.pass_strike_foot_speed_mps
                            ),
                        )
                    stroke_progress = None
                    effect_direction = teacher_direction.copy()
                    lateral_sign = _contact_foot_lateral_sign(
                        controller=controller,
                        data=data,
                        desired_direction_xy=effect_direction,
                        use_left=use_left,
                    )
                    if (
                        effect_config.pass_stroke_duration_sec > 0
                        and contact_mode == "strike"
                        and controller.decision is not None
                        and controller.decision.intent is TacticalIntent.PASS
                        and min(left_distance, right_distance)
                        <= effect_config.maximum_foot_ball_distance_m
                    ):
                        stroke_progress = controller.pass_stroke.step(
                            time_sec=float(data.time),
                            duration_sec=effect_config.pass_stroke_duration_sec,
                            use_left=use_left,
                            direction_xy=(float(effect_direction[0]), float(effect_direction[1])),
                            lateral_sign=lateral_sign,
                        )
                        use_left = controller.pass_stroke.use_left
                        effect_direction = np.asarray(controller.pass_stroke.direction_xy)
                        lateral_sign = controller.pass_stroke.lateral_sign
                    effect = locomotion_contact_teacher_effect(
                        model=model,
                        data=data,
                        ankle_body_id=(
                            controller.left_ankle_body if use_left else controller.right_ankle_body
                        ),
                        actuated_dof_indices=controller.joint_qvel,
                        ball_position_m=ball_position,
                        ball_velocity_mps=ball_linear_velocity,
                        desired_ball_direction_xy=effect_direction,
                        contact_mode=contact_mode,
                        # Recompute anatomical left after every yaw change.
                        # A fixed sign is only valid in the birth frame and
                        # crosses the selected ankle through the support leg
                        # after a receiver turns to face an incoming pass.
                        local_lateral_sign=lateral_sign,
                        contact_recent=contact_recent,
                        config=effect_config,
                        strike_progress=stroke_progress,
                    )
                    frame_teacher_mode_code = 1 if contact_mode == "receive" else 2
                    frame_teacher_foot_code = 1 if use_left else 2
                    frame_teacher_target_m = effect.ankle_target_m.copy()
                    # Read-only decomposition at the final 2 ms substep of
                    # this 50 Hz frame. Never count opposing torques as
                    # successful contact merely because a teacher is active.
                    teacher_effect_observed = True
                    frame_teacher_last_substep_valid = True
                    frame_teacher_baseline = raw_torque.copy()
                    if effect.active and effect_config.contact_leg_stiffness_scale < 1:
                        from rosclaw_soccer.growth.locomotion_contact_teacher import (
                            contact_tracking_adjustment,
                        )

                        frame_teacher_adjustment = contact_tracking_adjustment(
                            kp * (target - q),
                            use_left=bool(use_left),
                            scale=effect_config.contact_leg_stiffness_scale,
                        )
                        raw_torque += frame_teacher_adjustment
                    frame_teacher_residual = effect.torque_nm.copy()
                    frame_teacher_force = effect.task_force_n.copy()
                    frame_teacher_ankle = data.xpos[
                        controller.left_ankle_body if use_left else controller.right_ankle_body
                    ].copy()
                    raw_torque += effect.torque_nm
                    frame_teacher_active = frame_teacher_active or effect.active
                    frame_teacher_peak_torque_nm = max(
                        frame_teacher_peak_torque_nm,
                        float(np.max(np.abs(effect.torque_nm))),
                    )
                projected_torque = _project_joint_safe_torque(
                    joint_position=q,
                    joint_velocity=dq,
                    commanded_torque=raw_torque,
                    joint_ranges=np.asarray(model.jnt_range[controller.joint_ids]),
                    limited=model.jnt_limited[controller.joint_ids].astype(bool),
                    margin_rad=active.joint_guard_margin_rad,
                )
                torque = np.clip(projected_torque, -guarded_limits, guarded_limits)
                if teacher_effect_observed:
                    frame_teacher_guarded = torque.copy()
                data.ctrl[controller.actuators] = torque
                controller.torque_limit_violation = bool(
                    controller.torque_limit_violation or np.any(np.abs(torque) > hard_limits)
                )
            apply_g1_compliant_goal_net_force(
                data,
                ball_body_id=ball_body,
                ball_qpos=ball_qpos,
                ball_qvel=ball_qvel,
                spec=goal,
                capture_depth_m=max(0.20, 0.80 * goal.depth_m),
                stiffness_n_m=180.0,
                damping_n_s_m=10.0,
                state=net_state,
            )
            if active.bilateral_goals:
                from rosclaw_soccer.world.bilateral_net import apply_opposite_goal_net_force

                apply_opposite_goal_net_force(
                    data,
                    ball_body_id=ball_body,
                    ball_qpos=ball_qpos,
                    ball_qvel=ball_qvel,
                    spec=goal,
                    left_goal_plane_x_m=active.left_goal_plane_x_m,
                    state=opposite_net_state,
                )
            mujoco.mj_step(model, data)
            if motor_targets:
                _observe_team_motor_physics(
                    model,
                    data,
                    controllers,
                    motors,
                    motor_targets,
                    motor_faults,
                    ball_geom,
                    ball_qpos,
                    ball_qvel,
                )
            (
                substep_robot_contacts,
                substep_first_agent,
                substep_second_agent,
                substep_robot_contact_force_n,
            ) = _robot_robot_contact_observation(model, data, controllers)
            robot_contact_count += substep_robot_contacts
            frame_robot_contact_count += substep_robot_contacts
            if substep_robot_contact_force_n >= frame_robot_contact_force_n:
                frame_robot_contact_first_code = (
                    0 if substep_first_agent is None else agent_codes[substep_first_agent]
                )
                frame_robot_contact_second_code = (
                    0 if substep_second_agent is None else agent_codes[substep_second_agent]
                )
                frame_robot_contact_force_n = substep_robot_contact_force_n
            for contact_index in range(int(data.ncon)):
                contact = data.contact[contact_index]
                pair = {int(contact.geom1), int(contact.geom2)}
                if ball_geom not in pair:
                    continue
                other = next(value for value in pair if value != ball_geom)
                for controller in controllers:
                    if other not in controller.robot_geoms:
                        continue
                    effector_code = (
                        1
                        if other in controller.left_foot_geoms
                        else 2
                        if other in controller.right_foot_geoms
                        else 3
                        if other in controller.left_glove_geoms
                        else 4
                        if other in controller.right_glove_geoms
                        else 0
                    )
                    wrench: NDArray[np.float64] = np.zeros(6, dtype=np.float64)
                    mujoco.mj_contactForce(model, data, contact_index, wrench)
                    force = float(np.linalg.norm(wrench[:3]))
                    if (
                        controller.keeper_reach is not None
                        and effector_code in (3, 4)
                        and force > 1e-6
                    ):
                        controller.keeper_reach.notify_glove_contact(float(data.time))
                    if active.strict_receive_handoff:
                        # A geometrically listed contact with zero force is
                        # not a possession, launch, or reception observation.
                        if force <= 1e-6:
                            break
                        if receive_handoff is not None:
                            receive_handoff = receive_handoff.observe_contact(
                                agent_id=controller.cell.agent_id,
                                foot=effector_code in (1, 2),
                                force_n=force,
                                time_sec=float(data.time),
                            )
                            if receive_handoff.interrupted:
                                receive_lease_agent_id = None
                                receive_lease_source_agent_id = None
                                receive_lease_origin_m = None
                                receive_lease_target_m = None
                                receive_lease_active = False
                                receive_handoff = None
                                handoff_cancellations += 1
                    if not effector_code:
                        if force >= frame_nonfoot_contact_force_n:
                            frame_nonfoot_contact_agent_code = agent_codes[controller.cell.agent_id]
                            frame_nonfoot_contact_geom_id = other
                            frame_nonfoot_contact_force_n = force
                        break
                    if force >= frame_contact_force_n:
                        frame_contact_agent_code = agent_codes[controller.cell.agent_id]
                        frame_contact_effector_code = effector_code
                        frame_contact_foot_code = effector_code if effector_code <= 2 else 0
                        frame_contact_force_n = force
                    last_ball_contact_agent_id = controller.cell.agent_id
                    last_ball_contact_time_sec = float(data.time)
                    if loose_ball_chaser_agent_id != controller.cell.agent_id:
                        loose_ball_chaser_agent_id = controller.cell.agent_id
                        ball_chaser_lease_start_sec = float(data.time)
                    if receive_lease_agent_id == controller.cell.agent_id and (
                        not active.strict_receive_handoff or effector_code in (1, 2)
                    ):
                        last_receive_contact_agent_id = controller.cell.agent_id
                        last_receive_contact_time_sec = float(data.time)
                        controller.post_receive_joint_target = np.asarray(
                            data.qpos[controller.joint_qpos], dtype=np.float64
                        ).copy()
                        if (
                            strike_phase_config is not None
                            and controller.cell.self_model.primary_role is MatchRole.FINISHER
                        ):
                            controller.strike_phase.begin_capture(float(data.time))
                        receive_lease_agent_id = None
                        receive_lease_source_agent_id = None
                        receive_lease_origin_m = None
                        receive_lease_target_m = None
                        receive_lease_active = False
                        receive_handoff = None
                    if (
                        strike_lease_agent_id is not None
                        and strike_lease_agent_id != controller.cell.agent_id
                    ):
                        strike_lease_agent_id = None
                    current_possession_agent_id = controller.cell.agent_id
                    controller.last_ball_contact_foot = (
                        "left" if effector_code == 1 else "right" if effector_code == 2 else None
                    )
                    if controller is option_controller:
                        controller.option_contact_observed = True
                    break
        peak_ball_speed = max(
            peak_ball_speed,
            float(np.linalg.norm(data.qvel[ball_qvel : ball_qvel + 3])),
        )
        finite = bool(
            np.all(np.isfinite(data.qpos))
            and np.all(np.isfinite(data.qvel))
            and np.all(np.isfinite(data.ctrl))
        )
        for controller in controllers:
            q = np.asarray(data.qpos[controller.joint_qpos], dtype=np.float64)
            limited = model.jnt_limited[controller.joint_ids].astype(bool)
            ranges = model.jnt_range[controller.joint_ids]
            controller.joint_limit_violation = bool(
                controller.joint_limit_violation
                or np.any(q[limited] < ranges[limited, 0] - 1.0e-5)
                or np.any(q[limited] > ranges[limited, 1] + 1.0e-5)
            )
            pelvis = np.asarray(
                data.qpos[controller.qpos_base : controller.qpos_base + 7], dtype=np.float64
            )
            roll, pitch = _roll_pitch(pelvis[3:7])
            controller.minimum_pelvis_height_m = min(
                controller.minimum_pelvis_height_m, float(pelvis[2])
            )
            controller.maximum_tilt_rad = max(controller.maximum_tilt_rad, abs(roll), abs(pitch))
            _append_player_trace(trace, controller=controller, data=data, model=model)
        trace["time"].append(float(data.time))
        trace["ball_pose"].append(data.qpos[ball_qpos : ball_qpos + 7].copy())
        trace["ball_velocity"].append(data.qvel[ball_qvel : ball_qvel + 6].copy())
        trace["coordination_frame_index"].append(current_coordination_index)
        trace["possession_agent_code"].append(
            0 if current_possession_agent_id is None else agent_codes[current_possession_agent_id]
        )
        trace["ball_chaser_agent_code"].append(
            0 if loose_ball_chaser_agent_id is None else agent_codes[loose_ball_chaser_agent_id]
        )
        trace["receive_lease_agent_code"].append(
            0 if receive_lease_agent_id is None else agent_codes[receive_lease_agent_id]
        )
        trace["receive_lease_active"].append(receive_lease_active)
        if active.receiver_commitment_priority:
            trace.setdefault("receive_flight_tracking_agent_code", []).append(
                agent_codes[flight_tracking_agent_id]
                if flight_tracking_agent_id is not None
                and flight_tracking_agent_id == receive_lease_agent_id
                else 0
            )
        last_receiver = next(
            (
                controller
                for controller in controllers
                if controller.cell.agent_id == last_receive_contact_agent_id
            ),
            None,
        )
        trace["post_receive_hold_active"].append(
            bool(
                last_receiver is not None
                and float(data.time) - last_receive_contact_time_sec <= active.post_receive_hold_sec
                and not (
                    contact_teacher_config is not None
                    and contact_teacher_config.one_touch_finish_enabled
                    and last_receiver.cell.self_model.primary_role is MatchRole.FINISHER
                )
            )
        )
        trace["pass_source_agent_code"].append(
            0 if pass_source_agent_id is None else agent_codes[pass_source_agent_id]
        )
        trace["pass_target_agent_code"].append(
            0 if pass_target_agent_id is None else agent_codes[pass_target_agent_id]
        )
        if active.strict_receive_handoff:
            trace["pass_feedback_launch_relative"].append(True)
            trace["pass_handoff_source_contact_sec"].append(
                -1.0
                if receive_handoff is None or receive_handoff.source_foot_contact_sec is None
                else receive_handoff.source_foot_contact_sec
            )
            trace["pass_handoff_active"].append(receive_lease_active)
            trace["pass_handoff_cancellations"].append(handoff_cancellations)
        trace["strike_lease_agent_code"].append(
            agent_codes[frame_first_touch_strike_agent_id]
            if frame_first_touch_strike_agent_id is not None
            else 0
            if strike_lease_agent_id is None
            else agent_codes[strike_lease_agent_id]
        )
        trace["first_touch_strike_agent_code"].append(
            0
            if frame_first_touch_strike_agent_id is None
            else agent_codes[frame_first_touch_strike_agent_id]
        )
        strike_metrics = _strike_stance_metrics(
            controllers=controllers,
            strike_lease_agent_id=strike_lease_agent_id,
            data=data,
            ball_qpos=ball_qpos,
            goal=goal,
        )
        trace["strike_stance_depth_m"].append(strike_metrics[0])
        trace["strike_stance_lateral_error_m"].append(strike_metrics[1])
        trace["strike_stance_yaw_error_rad"].append(strike_metrics[2])
        trace["ball_contact_agent_code"].append(frame_contact_agent_code)
        trace["ball_contact_effector_code"].append(frame_contact_effector_code)
        trace["ball_contact_foot_code"].append(frame_contact_foot_code)
        trace["ball_contact_force_n"].append(frame_contact_force_n)
        trace["ball_nonfoot_contact_agent_code"].append(frame_nonfoot_contact_agent_code)
        trace["ball_nonfoot_contact_geom_id"].append(frame_nonfoot_contact_geom_id)
        trace["ball_nonfoot_contact_force_n"].append(frame_nonfoot_contact_force_n)
        trace["robot_robot_contact_count"].append(frame_robot_contact_count)
        trace["robot_robot_contact_first_code"].append(frame_robot_contact_first_code)
        trace["robot_robot_contact_second_code"].append(frame_robot_contact_second_code)
        trace["robot_robot_contact_force_n"].append(frame_robot_contact_force_n)
        trace["contact_teacher_agent_code"].append(
            0 if teacher_controller is None else agent_codes[teacher_controller.cell.agent_id]
        )
        trace["contact_teacher_active"].append(frame_teacher_active)
        trace["contact_teacher_peak_torque_nm"].append(frame_teacher_peak_torque_nm)
        trace["contact_teacher_last_substep_valid"].append(frame_teacher_last_substep_valid)
        trace["contact_teacher_baseline_torque_nm"].append(frame_teacher_baseline)
        trace["contact_teacher_residual_torque_nm"].append(frame_teacher_residual)
        trace["contact_teacher_tracking_adjustment_nm"].append(frame_teacher_adjustment)
        trace["contact_teacher_guarded_torque_nm"].append(frame_teacher_guarded)
        trace["contact_teacher_task_force_n"].append(frame_teacher_force)
        trace["contact_teacher_ankle_position_m"].append(frame_teacher_ankle)
        trace["contact_teacher_mode_code"].append(frame_teacher_mode_code)
        trace["contact_teacher_foot_code"].append(frame_teacher_foot_code)
        trace["contact_teacher_target_m"].append(frame_teacher_target_m)
        trace["option_agent_code"].append(
            0 if option_controller is None else agent_codes[option_controller.cell.agent_id]
        )
        trace["option_policy_frame"].append(option_policy_frame)
        recorded_phase_controller = next(
            (
                controller
                for controller in controllers
                if controller.strike_phase.phase is not StrikePhase.IDLE
            ),
            None,
        )
        trace["strike_phase_agent_code"].append(
            0
            if recorded_phase_controller is None
            else agent_codes[recorded_phase_controller.cell.agent_id]
        )
        trace["strike_phase_code"].append(
            0 if recorded_phase_controller is None else recorded_phase_controller.strike_phase.code
        )
        trace["strike_phase_elapsed_sec"].append(
            0.0
            if recorded_phase_controller is None
            else max(
                0.0,
                float(data.time) - recorded_phase_controller.strike_phase.phase_enter_time_sec,
            )
        )
        trace["strike_phase_transition_count"].append(
            0
            if recorded_phase_controller is None
            else recorded_phase_controller.strike_phase.transition_count
        )
        trace["strike_phase_abort_code"].append(
            _strike_phase_abort_code(
                None
                if recorded_phase_controller is None
                else recorded_phase_controller.strike_phase.abort_reason
            )
        )
        trace["strike_phase_approach_yaw_error_rad"].append(frame_phase_approach_yaw_error)
        trace["strike_phase_predicted_stance_depth_m"].append(phase_metrics[0])
        trace["strike_phase_predicted_stance_lateral_error_m"].append(phase_metrics[1])
        trace["strike_phase_predicted_stance_yaw_error_rad"].append(phase_metrics[2])
        trace["strike_coordination_actor_active"].append(frame_coordination_actor_active)
        trace["strike_coordination_observation"].append(frame_coordination_observation)
        trace["strike_coordination_stance_blend"].append(frame_coordination_action.stance_blend)
        trace["strike_coordination_goal_yaw_blend"].append(frame_coordination_action.goal_yaw_blend)
        trace["strike_context_memory_consulted"].append(frame_context_memory_consulted)
        trace["strike_context_observation"].append(frame_context_observation)
        trace["strike_context_expert_index"].append(frame_context_expert_index)
        trace["strike_context_normalized_distance"].append(frame_context_normalized_distance)
        trace["strike_context_selection_code"].append(frame_context_selection_code)
        trace["strike_context_abstained"].append(frame_context_abstained)
        if not finite:
            break
        if active.stop_on_ball_exit:
            from rosclaw_soccer.world.match_boundary import ball_exit_reason

            if (
                ball_exit_reason(
                    tuple(float(v) for v in data.qpos[ball_qpos : ball_qpos + 3]),
                    left_x=active.left_goal_plane_x_m,
                    right_x=goal.plane_x_m,
                    radius=goal.ball_radius_m,
                    goal_width=goal.width_m,
                    goal_height=goal.height_m,
                )
                is not None
            ):
                break

    trajectory = {name: np.asarray(values) for name, values in trace.items()}
    qualities = tuple(
        AgentWorldQuality(
            agent_id=controller.cell.agent_id,
            role=controller.cell.self_model.primary_role,
            active_fraction=controller.active_frames / max(1, len(trace["time"])),
            displacement_m=float(
                np.linalg.norm(
                    np.asarray(
                        data.qpos[controller.qpos_base : controller.qpos_base + 2],
                        dtype=np.float64,
                    )
                    - initial_positions[controller.cell.agent_id][:2]
                )
            ),
            distinct_intent_count=len(controller.seen_intents or ()),
            intent_switch_count=controller.intent_switch_count,
            decision_count=controller.decision_count,
            pass_intent_count=controller.pass_intent_count,
            shot_intent_count=controller.shot_intent_count,
            save_intent_count=controller.save_intent_count,
            distribution_intent_count=controller.distribution_intent_count,
            minimum_pelvis_height_m=controller.minimum_pelvis_height_m,
            maximum_tilt_rad=controller.maximum_tilt_rad,
            joint_limit_violation=controller.joint_limit_violation,
            torque_limit_violation=controller.torque_limit_violation,
            required_minimum_pelvis_height_m=active.minimum_pelvis_height_m,
            allowed_maximum_tilt_rad=active.maximum_tilt_rad,
        )
        for controller in controllers
    )
    if tuple(sorted((key, motor.contract_hash) for key, motor in motors.items())) != motor_hashes:
        raise ValueError("motor contract changed during the shared match")
    result = IndependentTeamWorldResult(
        scenario_hash=scenario.scenario_hash,
        roster_hash=roster.roster_hash,
        config_hash=active.config_hash,
        trajectory_hash=trajectory_digest(trajectory),
        player_count=len(controllers),
        red_player_count=sum(cell.self_model.team_id == "red" for cell in cells),
        blue_player_count=sum(cell.self_model.team_id == "blue" for cell in cells),
        decision_frame_count=len(coordination_hashes),
        coordination_frame_hashes=tuple(coordination_hashes),
        qualities=qualities,
        pass_handshake_count=pass_handshake_count,
        pass_intent_count=intent_counts[TacticalIntent.PASS],
        shot_intent_count=intent_counts[TacticalIntent.SHOOT],
        save_intent_count=intent_counts[TacticalIntent.SAVE],
        distribution_intent_count=intent_counts[TacticalIntent.DISTRIBUTE],
        finite_state=finite,
        robot_robot_contact_count=robot_contact_count,
        rolling_distance_m=float(
            np.linalg.norm(
                np.asarray(data.qpos[ball_qpos : ball_qpos + 2], dtype=np.float64)
                - initial_ball[:2]
            )
        ),
        peak_ball_speed_mps=peak_ball_speed,
        keeper_policy_hashes=tuple(
            (c.cell.agent_id, c.keeper_reach.policy_hash)
            for c in controllers
            if c.keeper_reach is not None
        ),
        motor_policy_hashes=motor_hashes,
        motor_fault_agents=tuple(sorted(motor_faults)),
    )
    return result, trajectory


def _observe_team_motor_physics(
    model: Any,
    data: Any,
    controllers: tuple[_PlayerController, ...],
    motors: dict[str, TeamMotorOption],
    targets: dict[str, TeamMotorTarget],
    faults: set[str],
    ball_geom: int,
    ball_qpos: int,
    ball_qvel: int,
) -> None:
    import mujoco

    safe = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
    for controller in controllers:
        pose = data.qpos[controller.qpos_base : controller.qpos_base + 7]
        joints = data.qpos[controller.joint_qpos]
        ranges = model.jnt_range[controller.joint_ids]
        limited = model.jnt_limited[controller.joint_ids].astype(bool)
        excess = np.maximum(ranges[:, 0] - joints, joints - ranges[:, 1])
        safe = bool(
            safe
            and pose[2] > 0.58
            and 1 - 2 * (pose[4] ** 2 + pose[5] ** 2) > 0.8
            and np.all(excess[limited] < 0.025)
        )
    measured: list[tuple[int, float]] = []
    for index in range(data.ncon):
        contact = data.contact[index]
        pair = (int(contact.geom1), int(contact.geom2))
        if ball_geom not in pair:
            continue
        other = pair[1] if pair[0] == ball_geom else pair[0]
        if model.geom(other).name in {"floor", "ground", "pitch"}:
            continue
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, wrench)
        if not np.isfinite(wrench).all():
            faults.update(targets)
            targets.clear()
            return
        measured.append((other, max(0.0, float(wrench[0]))))
    for controller in controllers:
        agent = controller.cell.agent_id
        if agent not in targets or not isinstance(motors[agent], TeamMotorPhysicsObserver):
            continue
        feet = controller.left_foot_geoms | controller.right_foot_geoms
        q = np.r_[
            data.qpos[controller.qpos_base : controller.qpos_base + 7],
            data.qpos[controller.joint_qpos],
            data.qpos[ball_qpos : ball_qpos + 7],
        ]
        v = np.r_[
            data.qvel[controller.qvel_base : controller.qvel_base + 6],
            data.qvel[controller.joint_qvel],
            data.qvel[ball_qvel : ball_qvel + 6],
        ]
        observer = motors[agent]
        assert isinstance(observer, TeamMotorPhysicsObserver)
        try:
            observer.observe_physics(
                TeamMotorPhysicsObservation(
                    time_sec=float(data.time),
                    qpos=tuple(float(x) for x in q),
                    qvel=tuple(float(x) for x in v),
                    world_bodies_safe=safe,
                    foot_normal_force_n=max(
                        (force for geom, force in measured if geom in feet), default=0.0
                    ),
                    other_non_ground_normal_force_n=max(
                        (force for geom, force in measured if geom not in feet), default=0.0
                    ),
                )
            )
        except (ValueError, TypeError, FloatingPointError):
            faults.add(agent)
            targets.pop(agent)  # Remove this override before the next 2 ms step.


def _make_player_controller(
    *,
    model: Any,
    data: Any,
    spec: G1PitchPlayerSpec,
    cell: RosclawSoccerAgentCell,
    pelvis_height: float,
    state_type: Any,
    output_type: Any,
    loco_type: Any,
    kick_type: Any | None = None,
) -> _PlayerController:
    import mujoco

    prefix = spec.body_prefix
    free_joint = _id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + "floating_base_joint")
    qpos_base = int(model.jnt_qposadr[free_joint])
    qvel_base = int(model.jnt_dofadr[free_joint])
    joint_ids = np.asarray(
        [_id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + name) for name in G1_DDS_JOINT_NAMES],
        dtype=np.int64,
    )
    joint_qpos = np.asarray(model.jnt_qposadr[joint_ids], dtype=np.int64)
    joint_qvel = np.asarray(model.jnt_dofadr[joint_ids], dtype=np.int64)
    actuators = np.asarray(
        [_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, prefix + name) for name in G1_DDS_JOINT_NAMES],
        dtype=np.int64,
    )
    state = state_type(29)
    output = output_type(29)
    with contextlib.redirect_stdout(io.StringIO()):
        policy = loco_type(state, output)
        policy.enter()
    kick_output = None
    kick_policy = None
    if kick_type is not None:
        kick_output = output_type(29)
        with contextlib.redirect_stdout(io.StringIO()):
            kick_policy = kick_type(state, kick_output)
    data.qpos[qpos_base : qpos_base + 3] = (
        spec.origin_m[0],
        spec.origin_m[1],
        pelvis_height,
    )
    half_yaw = 0.5 * spec.yaw_rad
    data.qpos[qpos_base + 3 : qpos_base + 7] = (
        math.cos(half_yaw),
        0.0,
        0.0,
        math.sin(half_yaw),
    )
    data.qpos[joint_qpos] = np.asarray(policy.default_angles_reorder, dtype=np.float64)
    pelvis_body = _id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "pelvis")
    left_ankle_body = _id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "left_ankle_roll_link")
    right_ankle_body = _id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "right_ankle_roll_link")
    return _PlayerController(
        spec=spec,
        cell=cell,
        qpos_base=qpos_base,
        qvel_base=qvel_base,
        joint_ids=joint_ids,
        joint_qpos=joint_qpos,
        joint_qvel=joint_qvel,
        actuators=actuators,
        pelvis_body=pelvis_body,
        torso_body=_id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "torso_link"),
        left_ankle_body=left_ankle_body,
        right_ankle_body=right_ankle_body,
        robot_geoms=_robot_geom_ids(model, pelvis_body),
        left_foot_geoms=_robot_geom_ids(model, left_ankle_body),
        right_foot_geoms=_robot_geom_ids(model, right_ankle_body),
        left_glove_geoms=_goalkeeper_glove_geoms(
            model,
            prefix=prefix,
            side="left",
            enabled=spec.goalkeeper_gloves,
        ),
        right_glove_geoms=_goalkeeper_glove_geoms(
            model,
            prefix=prefix,
            side="right",
            enabled=spec.goalkeeper_gloves,
        ),
        state=state,
        output=output,
        policy=policy,
        kick_output=kick_output,
        kick_policy=kick_policy,
    )


def _fill_locomotion_state(
    controller: _PlayerController, data: Any, ball_body: int, ball_qvel: int
) -> None:
    state = controller.state
    state.q = data.qpos[controller.joint_qpos].copy()
    state.dq = data.qvel[controller.joint_qvel].copy()
    state.tau_est = data.ctrl[controller.actuators].copy()
    state.root_lin_vel_b = data.qvel[controller.qvel_base : controller.qvel_base + 3].copy()
    state.root_ang_vel_b = data.qvel[controller.qvel_base + 3 : controller.qvel_base + 6].copy()
    state.torso_pos_w = data.xpos[controller.torso_body].copy()
    state.torso_quat_w = data.xquat[controller.torso_body].copy()
    state.pelvis_pos_w = data.qpos[controller.qpos_base : controller.qpos_base + 3].copy()
    state.pelvis_quat_w = data.qpos[controller.qpos_base + 3 : controller.qpos_base + 7].copy()
    state.ball_pos_w = data.xpos[ball_body].copy()
    state.ball_vel_w = data.qvel[ball_qvel : ball_qvel + 3].copy()
    state.ball_valid = True
    state.gravity_ori = _gravity_orientation(state.pelvis_quat_w)
    state.ang_vel = state.root_ang_vel_b.copy()


def _residual_intent_enabled(intent: TacticalIntent, *, strike_enabled: bool) -> bool:
    """Opt-in leg learning coverage; never override option/body/contact guards.

    SAVE is deliberately not enabled: a twelve-leg residual is not a learned
    arm interception policy. DISTRIBUTE covers the keeper's foot distribution.
    """
    if type(strike_enabled) is not bool:
        raise ValueError("explicit residual strike coverage required")
    return intent in {
        TacticalIntent.RECEIVE,
        TacticalIntent.INTERCEPT,
        TacticalIntent.PASS,
        TacticalIntent.CARRY,
    } or (strike_enabled and intent in {TacticalIntent.SHOOT, TacticalIntent.DISTRIBUTE})


def _owned_stance_policy(
    config: IndependentTeamWorldConfig, intent: TacticalIntent
) -> OwnedBallContactPolicy:
    policy = config.owned_contact_policy
    if policy is None:
        raise ValueError("owned stance requires a contact policy")
    if intent is TacticalIntent.SHOOT and config.strike_stance_lateral_m is not None:
        return replace(policy, lateral_m=config.strike_stance_lateral_m)
    return policy


def _near_ball_observation(
    controller: _PlayerController,
    *,
    data: Any,
    ball_qpos: int,
    ball_qvel: int,
    previous: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Measured proprioception and local task context; no future trajectory input."""
    state = controller.state
    ball = np.asarray(data.qpos[ball_qpos : ball_qpos + 3])
    yaw = _pelvis_yaw(state.pelvis_quat_w)
    target = (
        ball[:2]
        if controller.decision is None
        else np.asarray(controller.decision.target_position_m)[:2]
    )
    direction = target - ball[:2]
    direction = direction / max(float(np.linalg.norm(direction)), 1e-6)
    observation = np.concatenate(
        (
            state.gravity_ori,
            state.ang_vel,
            state.q[:12] - np.asarray(controller.policy.default_angles_reorder)[:12],
            state.dq[:12] * 0.05,
            _rotate_z(ball - state.pelvis_pos_w, -yaw),
            _rotate_z(data.qvel[ball_qvel : ball_qvel + 3], -yaw),
            _rotate_z(np.r_[direction, 0.0], -yaw)[:2],
            _rotate_z(data.xpos[controller.left_ankle_body] - ball, -yaw),
            _rotate_z(data.xpos[controller.right_ankle_body] - ball, -yaw),
            previous,
        )
    )
    return np.asarray(np.clip(observation, -5, 5), dtype=np.float64)


def _physical_state(
    controller: _PlayerController, data: Any, *, world_velocity: bool = False
) -> AgentPhysicalState:
    pose = np.asarray(data.qpos[controller.qpos_base : controller.qpos_base + 7], dtype=np.float64)
    velocity = _rotate_z(
        np.asarray(data.qvel[controller.qvel_base : controller.qvel_base + 3]),
        0.0 if world_velocity else controller.spec.yaw_rad,
    )
    roll, pitch = _roll_pitch(pose[3:7])
    tilt = max(abs(roll), abs(pitch))
    return AgentPhysicalState(
        agent_id=controller.cell.agent_id,
        position_m=(float(pose[0]), float(pose[1]), float(pose[2])),
        velocity_mps=(float(velocity[0]), float(velocity[1]), float(velocity[2])),
        pelvis_height_m=float(pose[2]),
        tilt_rad=tilt,
        stable=bool(pose[2] >= 0.55 and tilt <= 0.80),
    )


def _agent_observation(
    *,
    controller: _PlayerController,
    roster: TeamRoleRoster,
    state_by_id: dict[str, AgentPhysicalState],
    data: Any,
    ball_qpos: int,
    ball_qvel: int,
    goal: G1TrainingGoalSpec,
    left_goal_plane_x_m: float,
    possession_agent_id: str | None,
    ball_chaser_agent_id: str | None,
    active_receive_source_agent_id: str | None = None,
) -> AgentCellObservation:
    model = roster.agent(controller.cell.agent_id)
    own_goal_x = goal.plane_x_m if model.team_id == "blue" else left_goal_plane_x_m
    opponent_goal_x = goal.plane_x_m if model.team_id == "red" else left_goal_plane_x_m
    ball_position = data.qpos[ball_qpos : ball_qpos + 3]
    ball_velocity = data.qvel[ball_qvel : ball_qvel + 3]
    return AgentCellObservation(
        observer_agent_id=model.agent_id,
        time_sec=float(data.time),
        ball_position_m=(
            float(ball_position[0]),
            float(ball_position[1]),
            float(ball_position[2]),
        ),
        ball_velocity_mps=(
            float(ball_velocity[0]),
            float(ball_velocity[1]),
            float(ball_velocity[2]),
        ),
        own_goal_m=(own_goal_x, 0.0, 0.0),
        opponent_goal_m=(opponent_goal_x, 0.0, 0.0),
        possession_agent_id=possession_agent_id,
        self_state=state_by_id[model.agent_id],
        teammate_states=tuple(state_by_id[agent_id] for agent_id in model.teammate_ids),
        opponent_states=tuple(state_by_id[agent_id] for agent_id in model.opponent_ids),
        ball_chaser_agent_id=ball_chaser_agent_id,
        active_receive_source_agent_id=active_receive_source_agent_id,
    )


def _infer_possession(
    *,
    controllers: tuple[_PlayerController, ...],
    data: Any,
    ball_position: NDArray[Any],
    maximum_distance_m: float,
) -> str | None:
    distances = []
    for controller in controllers:
        feet = (
            np.asarray(data.xpos[controller.left_ankle_body], dtype=np.float64),
            np.asarray(data.xpos[controller.right_ankle_body], dtype=np.float64),
        )
        distance = min(float(np.linalg.norm(foot - ball_position)) for foot in feet)
        distances.append((distance, controller.cell.agent_id))
    distance, agent_id = min(distances)
    return agent_id if distance <= maximum_distance_m else None


def _movement_command(
    *,
    controller: _PlayerController,
    decision: AgentCellDecision,
    positions: dict[str, NDArray[np.float64]],
    data: Any,
    ball_qpos: int,
    ball_qvel: int,
    possession_agent_id: str | None,
    committed_receiver: bool,
    active_receiver: bool,
    post_receive_hold: bool,
    receive_foot_lateral_offset_m: float,
    strike_target_position_m: tuple[float, float] | None,
    config: IndependentTeamWorldConfig,
    strike_phase: StrikePhase | None = None,
    strike_phase_config: StrikePhaseConfig | None = None,
    strike_phase_owner_agent_id: str | None = None,
    strike_coordination_action: StrikeCoordinationAction | None = None,
    prospective_contact: bool = False,
    continuous_motor_approach: bool = False,
    pending_receive_target_m: tuple[float, float] | None = None,
) -> NDArray[np.float64]:
    current = positions[controller.cell.agent_id]
    target = np.asarray(decision.target_position_m[:2], dtype=np.float64)
    ball = np.asarray(data.qpos[ball_qpos : ball_qpos + 2], dtype=np.float64)
    phase_stance_target: NDArray[np.float64] | None = None
    phase_approach_direction: NDArray[np.float64] | None = None
    if committed_receiver and not active_receiver and decision.intent is not TacticalIntent.RECEIVE:
        # A negotiated receiver must stop its generic run-in-behind before the
        # ball is launched.  It may still move laterally to put its nearest
        # foot into the pass corridor; freezing both axes lets the ball pass
        # alongside a receiver that began off the lane.
        attack_direction = np.asarray(
            (1.0, 0.0) if controller.cell.self_model.team_id == "red" else (-1.0, 0.0),
            dtype=np.float64,
        )
        lateral = np.asarray((-attack_direction[1], attack_direction[0]), dtype=np.float64)
        pending_feet = (
            np.asarray(data.xpos[controller.left_ankle_body, :2], dtype=np.float64),
            np.asarray(data.xpos[controller.right_ankle_body, :2], dtype=np.float64),
        )
        receiving_foot = min(pending_feet, key=lambda foot: float(np.linalg.norm(foot - ball)))
        lateral_error = float(np.dot(ball - receiving_foot, lateral))
        target = (
            current.copy()
            if abs(float(np.dot(ball - current, lateral))) <= 0.25
            else current + lateral_error * lateral
        )
    if committed_receiver and not active_receiver and pending_receive_target_m is not None:
        # A diagonal pass is aimed at the negotiated receiver point. Do not
        # drag that receiver onto the passer's current horizontal ball lane.
        target = np.asarray(pending_receive_target_m, dtype=np.float64)
    if decision.intent is TacticalIntent.RECEIVE:
        ball_velocity = np.asarray(data.qvel[ball_qvel : ball_qvel + 2], dtype=np.float64)
        speed = float(np.linalg.norm(ball_velocity))
        travel_direction = (
            ball_velocity / speed
            if speed > 0.10
            else np.asarray(
                (1.0, 0.0) if controller.cell.self_model.team_id == "red" else (-1.0, 0.0),
                dtype=np.float64,
            )
        )
        time_to_player = (
            float(np.dot(current - ball, ball_velocity)) / (speed * speed)
            if speed > 0.10
            else math.inf
        )
        opponent_has_ball = possession_agent_id in controller.cell.self_model.opponent_ids
        can_run_onto_pass = bool(
            not opponent_has_ball
            and 0.0 <= time_to_player <= config.receive_run_onto_horizon_sec
            and float(np.linalg.norm(ball + max(0.0, time_to_player) * ball_velocity - current))
            <= config.receive_run_onto_lane_radius_m
        )
        if can_run_onto_pass:
            predicted_ball = ball + max(0.0, time_to_player) * ball_velocity
            lateral = np.asarray((-travel_direction[1], travel_direction[0]), dtype=np.float64)
            feet = (
                (
                    np.asarray(data.xpos[controller.left_ankle_body, :2], dtype=np.float64),
                    1.0,
                ),
                (
                    np.asarray(data.xpos[controller.right_ankle_body, :2], dtype=np.float64),
                    -1.0,
                ),
            )
            receiving_foot, lateral_sign = min(
                feet,
                key=lambda item: float(np.linalg.norm(item[0] - predicted_ball)),
            )
            desired_foot = predicted_ball + lateral_sign * receive_foot_lateral_offset_m * lateral
            lateral_error = float(np.dot(desired_foot - receiving_foot, lateral))
            if committed_receiver:
                # A committed receiver owns a moving capture point, not a
                # velocity chase.  Keeping the pelvis behind the predicted
                # ball arrival prevents the locomotion feet from overrunning
                # the ball and reflecting it backwards during the transition
                # into a strike stance.
                target = (
                    predicted_ball
                    - config.receive_pocket_depth_m * travel_direction
                    + lateral_error * lateral
                )
            else:
                # Stay ahead of an incoming ball but deliberately run slower
                # than it until the handshake becomes a physical receive
                # lease.  Driving at the same velocity kept the receiver a
                # constant 0.8--1.0 m in front of the ball.
                pacing_distance = min(
                    config.receive_runthrough_distance_m,
                    config.receive_pacing_ratio * speed / config.position_gain,
                )
                target = current + pacing_distance * travel_direction + lateral_error * lateral
        else:
            target = (
                ball
                + config.receive_intercept_horizon_sec * ball_velocity
                - config.receive_pocket_depth_m * travel_direction
            )
        if opponent_has_ball:
            lateral = np.asarray((-travel_direction[1], travel_direction[0]), dtype=np.float64)
            duel_side = -1.0 if controller.cell.self_model.team_id == "red" else 1.0
            target += duel_side * config.duel_lateral_offset_m * lateral
        if committed_receiver:
            ball_xyz = np.asarray(data.qpos[ball_qpos : ball_qpos + 3], dtype=np.float64)
            foot_distance = min(
                float(np.linalg.norm(data.xpos[controller.left_ankle_body] - ball_xyz)),
                float(np.linalg.norm(data.xpos[controller.right_ankle_body] - ball_xyz)),
            )
            if foot_distance <= config.receive_braking_distance_m:
                target = current.copy()
                if config.receive_lateral_braking and active_receiver and speed > 0.10:
                    # Stop longitudinal pursuit, not the last lateral capture
                    # correction. Align a real foot with the measured ball ray.
                    lateral_axis = np.asarray((-travel_direction[1], travel_direction[0]))
                    capture_foot = min(
                        (
                            data.xpos[controller.left_ankle_body],
                            data.xpos[controller.right_ankle_body],
                        ),
                        key=lambda foot: abs(float(np.dot(ball - foot[:2], lateral_axis))),
                    )
                    lateral_error = float(np.dot(ball - capture_foot[:2], lateral_axis))
                    target += np.clip(lateral_error, -0.30, 0.30) * lateral_axis
    if strike_target_position_m is not None and strike_phase_config is not None:
        destination = np.asarray(strike_target_position_m, dtype=np.float64)
        ball_velocity = np.asarray(data.qvel[ball_qvel : ball_qvel + 2], dtype=np.float64)
        phase_ball = ball + strike_phase_config.strike_contact_horizon_sec * ball_velocity
        direction = destination - phase_ball
        direction /= max(float(np.linalg.norm(direction)), 1.0e-9)
        lateral = np.asarray((-direction[1], direction[0]), dtype=np.float64)
        phase_stance_target = (
            phase_ball
            - strike_phase_config.target_stance_depth_m * direction
            - strike_phase_config.target_stance_lateral_m * lateral
        )
        ball_speed = float(np.linalg.norm(ball_velocity))
        signed_lateral_error = float(np.dot(phase_ball - current, lateral))
        phase_approach_velocity = (
            ball_velocity
            - strike_phase_config.orient_goalward_lag_mps * direction
            + strike_phase_config.orient_lateral_gain_per_sec
            * (signed_lateral_error - strike_phase_config.target_stance_lateral_m)
            * lateral
            if ball_speed > 0.10
            else direction * min(0.12, strike_phase_config.orient_goalward_lag_mps)
        )
        approach_speed = float(np.linalg.norm(phase_approach_velocity))
        phase_approach_direction = (
            phase_approach_velocity / approach_speed
            if approach_speed > 1.0e-9
            else direction.copy()
        )
        if strike_phase in {StrikePhase.CAPTURE, StrikePhase.STRIKE}:
            # Capture and orient without the old discontinuous lateral hop.
            # During STRIKE the locomotion base yields to the frozen whole-body
            # option rather than walking through its planted support foot.
            target = current.copy()
        elif strike_phase is StrikePhase.ORIENT:
            # Pace behind the received ball instead of stopping and making a
            # 180-degree turn.  The slower pursuit lets a usable stance depth
            # open while preserving the receiver's contact-side relationship.
            pacing_target = current + phase_approach_velocity / config.position_gain
            if strike_coordination_action is None:
                # Preserve the frozen parent path without even a zero-weight
                # floating-point blend; long MuJoCo rollouts are sensitive to
                # last-bit changes.
                target = pacing_target
            else:
                blend = strike_coordination_action.stance_blend
                target = (1.0 - blend) * pacing_target + blend * phase_stance_target
        elif strike_phase is StrikePhase.PLANT:
            target = phase_stance_target
        elif strike_phase in {StrikePhase.RECOVER, StrikePhase.COMPLETE, StrikePhase.ABORTED}:
            target = current.copy()
    elif strike_target_position_m is not None:
        destination = np.asarray(strike_target_position_m, dtype=np.float64)
        direction = destination - ball
        direction /= max(float(np.linalg.norm(direction)), 1.0e-9)
        lateral = np.asarray((-direction[1], direction[0]), dtype=np.float64)
        ball_from_player = ball - current
        stance_depth = float(np.dot(ball_from_player, direction))
        signed_lateral_error = float(np.dot(ball_from_player, lateral))
        if stance_depth < 0.30 and abs(signed_lateral_error) < config.strike_bypass_lateral_m:
            bypass_sign = -1.0 if signed_lateral_error >= 0.0 else 1.0
            target = current + bypass_sign * config.strike_bypass_lateral_m * lateral
        elif stance_depth < 0.30:
            # Back up in a lane parallel to the shot line while preserving
            # the already-safe lateral separation from the football.
            target = ball - 0.72 * direction - signed_lateral_error * lateral
        else:
            target = ball - 0.72 * direction
    elif decision.intent in {
        TacticalIntent.PASS,
        TacticalIntent.SHOOT,
        TacticalIntent.DISTRIBUTE,
    }:
        destination = np.asarray(decision.target_position_m[:2], dtype=np.float64)
        direction = destination - ball
        direction /= max(float(np.linalg.norm(direction)), 1.0e-9)
        target = ball - 0.72 * direction
    motor_approach = bool(
        continuous_motor_approach
        and config.motor_approach_standoff_m is not None
        and prospective_contact
        and possession_agent_id is None
        and decision.intent is TacticalIntent.SHOOT
        and not post_receive_hold
    )
    if motor_approach:
        # An admitted moving-handoff motor cannot be reached by the legacy
        # standing-shot target 0.72 m behind the ball. This is an explicit
        # locomotion objective, not a root/ball pose write or success label.
        destination = np.asarray(decision.target_position_m[:2], dtype=np.float64)
        direction = destination - ball
        direction /= max(float(np.linalg.norm(direction)), 1.0e-9)
        assert config.motor_approach_standoff_m is not None
        target = ball - config.motor_approach_standoff_m * direction
    if post_receive_hold:
        target = current.copy()
    error = target - current
    command: NDArray[np.float64] = np.zeros(3, dtype=np.float64)
    command[:2] = config.position_gain * error
    speed_limit = (
        config.goalkeeper_maximum_speed_mps
        if controller.cell.self_model.primary_role is MatchRole.GOALKEEPER
        else config.maximum_speed_mps
    )
    speed = float(np.linalg.norm(command[:2]))
    if speed > speed_limit:
        command[:2] *= speed_limit / speed
    if float(np.linalg.norm(error)) <= config.arrival_radius_m:
        command[:2] = 0.0
    desired_yaw: float | None = None
    if (
        config.owned_contact_policy is not None
        and not motor_approach
        and not post_receive_hold
        and strike_phase_config is None
        and (
            possession_agent_id == controller.cell.agent_id
            or (prospective_contact and possession_agent_id is None)
        )
        and decision.intent
        in {TacticalIntent.PASS, TacticalIntent.SHOOT, TacticalIntent.DISTRIBUTE}
        and float(np.linalg.norm(np.asarray(decision.target_position_m[:2]) - ball)) > 1.0e-6
    ):
        stance, desired_yaw = _owned_stance_policy(config, decision.intent).stance(
            (float(ball[0]), float(ball[1])),
            (decision.target_position_m[0], decision.target_position_m[1]),
        )
        if config.pass_stance_bypass and decision.intent is TacticalIntent.PASS:
            stance = _owned_stance_policy(config, decision.intent).approach_waypoint(
                (float(current[0]), float(current[1])),
                (float(ball[0]), float(ball[1])),
                (decision.target_position_m[0], decision.target_position_m[1]),
                lateral_clearance_m=config.strike_bypass_lateral_m,
            )
        # The destination belongs to the ball, not to the player's pelvis.
        error = np.asarray(stance, dtype=np.float64) - current
        command[:2] = config.position_gain * error
        if float(np.linalg.norm(error)) <= config.arrival_radius_m:
            command[:2] = 0.0
    if (
        phase_approach_direction is not None
        and strike_phase is StrikePhase.ORIENT
        and not post_receive_hold
    ):
        if strike_coordination_action is not None:
            yaw_blend = strike_coordination_action.goal_yaw_blend
            strike_direction = direction
            phase_approach_direction = (
                1.0 - yaw_blend
            ) * phase_approach_direction + yaw_blend * strike_direction
            phase_approach_direction /= max(float(np.linalg.norm(phase_approach_direction)), 1.0e-9)
        desired_yaw = math.atan2(
            float(phase_approach_direction[1]),
            float(phase_approach_direction[0]),
        )
    elif not post_receive_hold and (
        strike_target_position_m is not None
        or (prospective_contact and decision.intent in {TacticalIntent.PASS, TacticalIntent.SHOOT})
    ):
        destination = np.asarray(
            strike_target_position_m
            if strike_target_position_m is not None
            else decision.target_position_m[:2],
            dtype=np.float64,
        )
        desired_yaw = math.atan2(destination[1] - ball[1], destination[0] - ball[0])
    elif (
        committed_receiver
        and not post_receive_hold
        and float(np.linalg.norm(ball - current)) > 1.0e-6
    ):
        ball_bearing = math.atan2(ball[1] - current[1], ball[0] - current[0])
        desired_yaw = ball_bearing + (
            -config.receive_open_body_angle_rad
            if controller.cell.self_model.team_id == "red"
            else config.receive_open_body_angle_rad
        )
    if desired_yaw is not None:
        current_yaw = _pelvis_yaw(
            np.asarray(
                data.qpos[controller.qpos_base + 3 : controller.qpos_base + 7], dtype=np.float64
            )
        )
        yaw_error = math.atan2(
            math.sin(desired_yaw - current_yaw), math.cos(desired_yaw - current_yaw)
        )
        yaw_rate_limit = (
            strike_phase_config.maximum_yaw_rate_radps
            if strike_phase_config is not None
            and strike_phase is not None
            and strike_phase not in {StrikePhase.IDLE, StrikePhase.COMPLETE, StrikePhase.ABORTED}
            else config.maximum_yaw_rate_radps
        )
        command[2] = float(
            np.clip(
                config.yaw_gain * yaw_error,
                -yaw_rate_limit,
                yaw_rate_limit,
            )
        )
    for other_id, other in positions.items():
        if other_id == controller.cell.agent_id:
            continue
        delta = current - other
        separation = float(np.linalg.norm(delta))
        if 1.0e-9 < separation < config.minimum_player_separation_m:
            correction = min(
                config.maximum_collision_correction_mps,
                config.collision_avoidance_gain * (config.minimum_player_separation_m - separation),
            )
            command[:2] += correction * delta / separation
    if (
        strike_phase_config is not None
        and strike_phase_owner_agent_id is not None
        and controller.cell.agent_id != strike_phase_owner_agent_id
        and strike_phase_owner_agent_id in controller.cell.self_model.teammate_ids
    ):
        owner = positions[strike_phase_owner_agent_id]
        delta = current - owner
        separation = float(np.linalg.norm(delta))
        if 1.0e-9 < separation < strike_phase_config.support_lane_clearance_m:
            correction = min(
                config.maximum_collision_correction_mps,
                config.collision_avoidance_gain
                * (strike_phase_config.support_lane_clearance_m - separation),
            )
            command[:2] += correction * delta / separation
    if config.predictive_separation and decision.intent in {
        TacticalIntent.SUPPORT,
        TacticalIntent.RUN_IN_BEHIND,
        TacticalIntent.COVER,
    }:
        # Project approach speed instead of adding a repulsion weaker than the
        # 0.70 m/s approach command. Tangential motion remains available.
        for other_id, other in sorted(positions.items()):
            if other_id == controller.cell.agent_id:
                continue
            away = current - other
            separation = float(np.linalg.norm(away))
            if 1.0e-9 < separation < config.minimum_player_separation_m:
                away /= separation
                approach = float(np.dot(command[:2], away))
                lower_bound = -max(0.0, 0.60 * (separation - 0.80))
                if approach < lower_bound:
                    command[:2] += (lower_bound - approach) * away
    previous = (
        np.zeros(3, dtype=np.float64)
        if controller.last_world_command is None
        else controller.last_world_command
    )
    delta = command - previous
    maximum_delta = config.maximum_acceleration_mps2 * _CONTROL_DT
    delta_norm = float(np.linalg.norm(delta[:2]))
    if delta_norm > maximum_delta:
        command[:2] = previous[:2] + delta[:2] * maximum_delta / delta_norm
    speed = float(np.linalg.norm(command[:2]))
    if speed > speed_limit:
        command[:2] *= speed_limit / speed
    if config.all_role_clearance:
        proposal = propose_clearance_velocity(
            command[:2],
            np.asarray(
                [
                    other - current
                    for agent, other in sorted(positions.items())
                    if agent != controller.cell.agent_id
                ]
            ).reshape((-1, 2)),
            maximum_speed_mps=speed_limit,
        )
        command[:2] = proposal.velocity_mps
        controller.clearance_feasible = proposal.feasible
        controller.clearance_constrained = proposal.constrained
    return command


def _select_loose_ball_chaser(
    *,
    controllers: tuple[_PlayerController, ...],
    states: dict[str, AgentPhysicalState],
    ball_position: NDArray[np.float64],
    current_agent_id: str | None,
    lease_start_sec: float,
    time_sec: float,
    config: IndependentTeamWorldConfig,
) -> tuple[str | None, float]:
    """Keep one outfielder on the ball until a materially better handoff exists."""

    candidates: tuple[str, ...] = tuple(
        controller.cell.agent_id
        for controller in controllers
        if (
            controller.cell.self_model.primary_role is not MatchRole.GOALKEEPER
            or (
                controller.cell.self_model.basic_ball_play
                and np.linalg.norm(
                    np.asarray(states[controller.cell.agent_id].position_m[:2]) - ball_position
                )
                <= 0.8
                and np.linalg.norm(
                    np.asarray(controller.cell.tactical_profile.home_position_m[:2]) - ball_position
                )
                <= 1.3
            )
        )
        and states[controller.cell.agent_id].stable
    )
    if not candidates:
        return None, time_sec

    def distance(agent_id: str) -> float:
        return float(
            np.linalg.norm(
                np.asarray(states[agent_id].position_m[:2], dtype=np.float64) - ball_position
            )
        )

    candidate = min(candidates, key=lambda agent_id: (distance(agent_id), agent_id))
    if current_agent_id not in candidates:
        return candidate, time_sec
    assert current_agent_id is not None
    if time_sec - lease_start_sec < config.minimum_ball_chaser_lease_sec:
        return current_agent_id, lease_start_sec
    if distance(candidate) + config.ball_chaser_handoff_margin_m < distance(current_agent_id):
        return candidate, time_sec
    return current_agent_id, lease_start_sec


def _select_pressing_chaser(
    *,
    controllers: tuple[_PlayerController, ...],
    states: dict[str, AgentPhysicalState],
    possession_agent_id: str,
    ball_position: NDArray[np.float64],
) -> str | None:
    """Assign exactly one stable opposing outfielder to a contact owner."""

    owner = next(
        controller for controller in controllers if controller.cell.agent_id == possession_agent_id
    )
    opponent_ids = set(owner.cell.self_model.opponent_ids)
    candidates: tuple[str, ...] = tuple(
        controller.cell.agent_id
        for controller in controllers
        if controller.cell.agent_id in opponent_ids
        and controller.cell.self_model.primary_role is not MatchRole.GOALKEEPER
        and states[controller.cell.agent_id].stable
    )
    if not candidates:
        return None
    return str(
        min(
            candidates,
            key=lambda agent_id: (
                float(
                    np.linalg.norm(
                        np.asarray(states[agent_id].position_m[:2], dtype=np.float64)
                        - ball_position
                    )
                ),
                agent_id,
            ),
        )
    )


def _select_contact_teacher_controller(
    *,
    controllers: tuple[_PlayerController, ...],
    data: Any,
    ball_position: NDArray[np.float64],
    current_possession_agent_id: str | None,
    preferred_agent_id: str | None,
    config: G1LocomotionContactTeacherConfig | None,
    nearest_contact_first: bool = False,
) -> _PlayerController | None:
    """Grant one training-only foot-contact lease on the current frame."""

    if config is None:
        return None
    contact_intents = {
        TacticalIntent.RECEIVE,
        TacticalIntent.CARRY,
        TacticalIntent.PASS,
        TacticalIntent.SHOOT,
        TacticalIntent.INTERCEPT,
        TacticalIntent.SAVE,
        TacticalIntent.DISTRIBUTE,
    }
    candidates = tuple(
        controller
        for controller in controllers
        if controller.decision is not None and controller.decision.intent in contact_intents
    )
    if not candidates:
        return None
    preferred = next(
        (controller for controller in candidates if controller.cell.agent_id == preferred_agent_id),
        None,
    )
    if (
        not nearest_contact_first
        and preferred is not None
        and min(
            float(np.linalg.norm(data.xpos[preferred.left_ankle_body] - ball_position)),
            float(np.linalg.norm(data.xpos[preferred.right_ankle_body] - ball_position)),
        )
        <= config.maximum_foot_ball_distance_m
    ):
        return preferred
    if current_possession_agent_id is not None:
        owner = next(
            (
                controller
                for controller in candidates
                if controller.cell.agent_id == current_possession_agent_id
            ),
            None,
        )
        # A remembered contact owner can already be metres from the moving
        # ball. In physical acquisition mode it must not starve a reachable
        # opponent/receiver of the contact teacher during the possession hold.
        # Keep the legacy owner-first experiment contract otherwise unchanged.
        if owner is not None and (
            not nearest_contact_first
            or min(
                float(np.linalg.norm(data.xpos[owner.left_ankle_body] - ball_position)),
                float(np.linalg.norm(data.xpos[owner.right_ankle_body] - ball_position)),
            )
            <= config.maximum_foot_ball_distance_m
        ):
            return owner
    return min(
        candidates,
        key=lambda controller: min(
            float(np.linalg.norm(data.xpos[controller.left_ankle_body] - ball_position)),
            float(np.linalg.norm(data.xpos[controller.right_ankle_body] - ball_position)),
        ),
    )


def _activate_rolling_option(
    *,
    controllers: tuple[_PlayerController, ...],
    current_possession_agent_id: str | None,
    last_ball_contact_agent_id: str | None,
    strike_lease_agent_id: str | None,
    frame: int,
    data: Any,
    ball_qpos: int,
    goal: G1TrainingGoalSpec,
    config: G1RollingOptionBridgeConfig | None,
    phase_config: StrikePhaseConfig | None = None,
    left_goal_plane_x_m: float | None = None,
    prospective_agent_id: str | None = None,
) -> None:
    """Warm-start one owned PASS/SHOOT option after contact-derived possession."""

    if config is None or any(controller.option_active for controller in controllers):
        return
    leased_shot = bool(
        strike_lease_agent_id is not None and strike_lease_agent_id == last_ball_contact_agent_id
    )
    owned_option = current_possession_agent_id == last_ball_contact_agent_id
    prospective = bool(
        config.prospective_enabled
        and current_possession_agent_id is None
        and prospective_agent_id is not None
    )
    if not leased_shot and not owned_option and not prospective:
        return
    candidate = next(
        (
            controller
            for controller in controllers
            if controller.cell.agent_id
            == (
                strike_lease_agent_id
                if leased_shot
                else prospective_agent_id
                if prospective
                else current_possession_agent_id
            )
            and controller.decision is not None
            and (
                leased_shot
                or controller.decision.intent is TacticalIntent.SHOOT
                or config.pass_enabled
                and controller.decision.intent is TacticalIntent.PASS
            )
            and not controller.option_completed
        ),
        None,
    )
    if candidate is None:
        return
    phase_required = phase_config is not None
    if phase_required and candidate.strike_phase.phase is not StrikePhase.STRIKE:
        return
    if abs(candidate.spec.yaw_rad) > 1.0e-9 and not (
        config.bilateral_enabled and left_goal_plane_x_m is not None
    ):
        # This first bridge only certifies the canonical red attacking frame.
        # Mirrored/bilateral strike entry needs its own matched retention set.
        return
    if candidate.kick_policy is None or candidate.kick_output is None:
        raise RuntimeError("rolling option bridge was requested without a kick policy")
    assert candidate.decision is not None
    attacking_goal = (
        (left_goal_plane_x_m, -goal.target_y_m, goal.target_z_m)
        if config.bilateral_enabled
        and left_goal_plane_x_m is not None
        and candidate.cell.self_model.team_id == "blue"
        else (goal.plane_x_m, goal.target_y_m, goal.target_z_m)
    )
    ball = np.asarray(data.qpos[ball_qpos : ball_qpos + 2], dtype=np.float64)
    pelvis = np.asarray(data.qpos[candidate.qpos_base : candidate.qpos_base + 2], dtype=np.float64)
    target = np.asarray(
        attacking_goal[:2] if leased_shot else candidate.decision.target_position_m[:2],
        dtype=np.float64,
    )
    direction = target - ball
    direction /= max(float(np.linalg.norm(direction)), 1.0e-9)
    target_yaw = math.atan2(float(direction[1]), float(direction[0]))
    current_yaw = _pelvis_yaw(
        np.asarray(data.qpos[candidate.qpos_base + 3 : candidate.qpos_base + 7], dtype=np.float64)
    )
    yaw_error = abs(
        math.atan2(math.sin(target_yaw - current_yaw), math.cos(target_yaw - current_yaw))
    )
    lateral = np.asarray((-direction[1], direction[0]), dtype=np.float64)
    stance_depth = float(np.dot(ball - pelvis, direction))
    lateral_error = abs(float(np.dot(ball - pelvis, lateral)))
    if not phase_required and (
        not config.minimum_strike_stance_depth_m
        <= stance_depth
        <= config.maximum_strike_stance_depth_m
        or lateral_error > config.maximum_strike_lateral_error_m
        or yaw_error > config.maximum_strike_yaw_error_rad
    ):
        return
    with contextlib.redirect_stdout(io.StringIO()):
        candidate.kick_policy.enter()
    is_pass = not leased_shot and candidate.decision.intent is TacticalIntent.PASS
    preferred_target = (
        candidate.decision.target_position_m
        if is_pass
        else (
            attacking_goal[0],
            attacking_goal[1]
            + (0.0 if phase_config is None else phase_config.strike_aim_lateral_bias_m),
            attacking_goal[2],
        )
    )
    candidate.kick_policy.target_pos_w = np.asarray(preferred_target, dtype=np.float32)
    if is_pass and config.pass_reference_distance_m > 0:
        # Calibrated motor reference is distinct from the tactical receiver.
        # The receiver remains the physical scoring target; no ball state changes.
        waypoint_xy = pelvis + config.pass_reference_distance_m * direction
        candidate.kick_policy.target_pos_w = np.asarray(
            (waypoint_xy[0], waypoint_xy[1], 0.20), dtype=np.float32
        )
    candidate.kick_policy.time_step = (
        int(candidate.kick_policy.WARMUP_STEPS) + config.entry_policy_frame
    )
    if config.observation_warmstart:
        from rosclaw_soccer.providers.g1.kick_warmstart import prepare_kick_handoff

        prepare_kick_handoff(candidate.kick_policy, entry_frame=config.entry_policy_frame)
    parameters = config.pass_parameters if is_pass else config.shoot_parameters
    candidate.option_active = True
    candidate.option_activation_frame = frame
    candidate.option_origin_target = None
    candidate.option_origin_kp = None
    candidate.option_origin_kd = None
    candidate.option_parameters = parameters
    candidate.option_contact_observed = False


def _strike_teacher_stance_ready(
    *,
    controller: _PlayerController,
    data: Any,
    ball_qpos: int,
    goal: G1TrainingGoalSpec,
    minimum_depth_m: float,
    target_xy: tuple[float, float] | None = None,
) -> bool:
    """Only let the strike residual act once the pelvis is behind the ball."""

    ball = np.asarray(data.qpos[ball_qpos : ball_qpos + 2], dtype=np.float64)
    pelvis = np.asarray(
        data.qpos[controller.qpos_base : controller.qpos_base + 2], dtype=np.float64
    )
    destination = np.asarray(
        (goal.plane_x_m, goal.target_y_m) if target_xy is None else target_xy,
        dtype=np.float64,
    )
    direction = destination - ball
    direction /= max(float(np.linalg.norm(direction)), 1.0e-9)
    return bool(float(np.dot(ball - pelvis, direction)) >= minimum_depth_m)


def _strike_stance_metrics(
    *,
    controllers: tuple[_PlayerController, ...],
    strike_lease_agent_id: str | None,
    data: Any,
    ball_qpos: int,
    goal: G1TrainingGoalSpec,
) -> tuple[float, float, float]:
    if strike_lease_agent_id is None:
        return (0.0, 0.0, 0.0)
    controller = next(
        value for value in controllers if value.cell.agent_id == strike_lease_agent_id
    )
    return _controller_strike_stance_metrics(
        controller=controller,
        data=data,
        ball_qpos=ball_qpos,
        goal=goal,
    )


def _controller_strike_stance_metrics(
    *,
    controller: _PlayerController,
    data: Any,
    ball_qpos: int,
    goal: G1TrainingGoalSpec,
    ball_qvel: int | None = None,
    prediction_horizon_sec: float = 0.0,
) -> tuple[float, float, float]:
    ball = np.asarray(data.qpos[ball_qpos : ball_qpos + 2], dtype=np.float64)
    if ball_qvel is not None and prediction_horizon_sec > 0.0:
        ball = ball + prediction_horizon_sec * np.asarray(
            data.qvel[ball_qvel : ball_qvel + 2], dtype=np.float64
        )
    pelvis = np.asarray(
        data.qpos[controller.qpos_base : controller.qpos_base + 2], dtype=np.float64
    )
    direction = np.asarray((goal.plane_x_m, goal.target_y_m), dtype=np.float64) - ball
    direction /= max(float(np.linalg.norm(direction)), 1.0e-9)
    lateral = np.asarray((-direction[1], direction[0]), dtype=np.float64)
    target_yaw = math.atan2(float(direction[1]), float(direction[0]))
    current_yaw = _pelvis_yaw(
        np.asarray(data.qpos[controller.qpos_base + 3 : controller.qpos_base + 7], dtype=np.float64)
    )
    yaw_error = abs(
        math.atan2(math.sin(target_yaw - current_yaw), math.cos(target_yaw - current_yaw))
    )
    return (
        float(np.dot(ball - pelvis, direction)),
        abs(float(np.dot(ball - pelvis, lateral))),
        yaw_error,
    )


def _strike_tracking_yaw_error(
    *,
    controller: _PlayerController,
    data: Any,
    ball_qpos: int,
    ball_qvel: int,
    goal: G1TrainingGoalSpec,
    config: StrikePhaseConfig,
    goal_yaw_blend: float = 0.0,
) -> float:
    # MuJoCo slices are writable views into ``data.qvel``.  This function is an
    # observation-only error calculation, so own the buffer before normalizing.
    ball_velocity = np.array(data.qvel[ball_qvel : ball_qvel + 2], dtype=np.float64, copy=True)
    # ``data.qpos[...]`` is a MuJoCo-backed view.  Use an out-of-place sum so
    # prediction can never mutate authoritative football state.
    phase_ball = np.asarray(data.qpos[ball_qpos : ball_qpos + 2], dtype=np.float64).copy()
    phase_ball = phase_ball + config.strike_contact_horizon_sec * ball_velocity
    direction = np.asarray((goal.plane_x_m, goal.target_y_m), dtype=np.float64) - phase_ball
    direction /= max(float(np.linalg.norm(direction)), 1.0e-9)
    pelvis = np.asarray(
        data.qpos[controller.qpos_base : controller.qpos_base + 2], dtype=np.float64
    )
    lateral = np.asarray((-direction[1], direction[0]), dtype=np.float64)
    signed_lateral_error = float(np.dot(phase_ball - pelvis, lateral))
    velocity = (
        ball_velocity
        - config.orient_goalward_lag_mps * direction
        + config.orient_lateral_gain_per_sec
        * (signed_lateral_error - config.target_stance_lateral_m)
        * lateral
    )
    if float(np.linalg.norm(velocity)) <= 1.0e-9:
        velocity = direction
    if goal_yaw_blend <= 0.0:
        # Preserve the frozen S208 path bit-for-bit when no actor has authority.
        target_yaw = math.atan2(float(velocity[1]), float(velocity[0]))
    else:
        velocity /= max(float(np.linalg.norm(velocity)), 1.0e-9)
        blended_direction = (1.0 - goal_yaw_blend) * velocity + goal_yaw_blend * direction
        blended_direction /= max(float(np.linalg.norm(blended_direction)), 1.0e-9)
        target_yaw = math.atan2(float(blended_direction[1]), float(blended_direction[0]))
    current_yaw = _pelvis_yaw(
        np.asarray(data.qpos[controller.qpos_base + 3 : controller.qpos_base + 7], dtype=np.float64)
    )
    return abs(
        math.atan2(
            math.sin(target_yaw - current_yaw),
            math.cos(target_yaw - current_yaw),
        )
    )


def _strike_phase_abort_code(reason: str | None) -> int:
    return {
        None: 0,
        "NONFINITE_STATE": 1,
        "BODY_UNSTABLE": 2,
        "BALL_ESCAPED": 3,
        "ORIENT_TIMEOUT": 4,
        "PLANT_TIMEOUT": 5,
        "STRIKE_TIMEOUT": 6,
        "OPTION_ENDED_WITHOUT_CONTACT": 7,
        "RECOVERY_TIMEOUT": 8,
        "CLOCK_ROLLBACK": 9,
    }.get(reason, 255)


def _rolling_option_target(
    controller: _PlayerController,
    *,
    frame: int,
    config: G1RollingOptionBridgeConfig | None,
) -> tuple[
    tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    int,
]:
    if (
        config is None
        or controller.kick_policy is None
        or controller.kick_output is None
        or controller.option_activation_frame is None
        or controller.option_parameters is None
    ):
        raise RuntimeError("rolling option state is incomplete")
    if controller.option_origin_target is None:
        controller.option_origin_target = np.asarray(
            controller.output.actions, dtype=np.float64
        ).copy()
        controller.option_origin_kp = np.asarray(controller.output.kps, dtype=np.float64).copy()
        controller.option_origin_kd = np.asarray(controller.output.kds, dtype=np.float64).copy()
    with contextlib.redirect_stdout(io.StringIO()):
        controller.kick_policy.run()
    policy_frame = max(
        0,
        int(controller.kick_policy.time_step) - int(controller.kick_policy.WARMUP_STEPS),
    )
    target = adapt_shot_target(
        target=np.asarray(controller.kick_output.actions, dtype=np.float64),
        default=np.asarray(controller.kick_policy.default_q_mj, dtype=np.float64),
        parameters=controller.option_parameters,
        policy_frame=policy_frame,
    )
    kp = np.asarray(controller.kick_output.kps, dtype=np.float64)
    kd = np.asarray(controller.kick_output.kds, dtype=np.float64)
    blend = min(
        1.0,
        (frame - controller.option_activation_frame + 1) / config.blend_frames,
    )
    origin_target = controller.option_origin_target
    origin_kp = controller.option_origin_kp
    origin_kd = controller.option_origin_kd
    if origin_target is None or origin_kp is None or origin_kd is None:
        raise RuntimeError("rolling strike transition origin is incomplete")
    blended = (
        (1.0 - blend) * origin_target + blend * target,
        (1.0 - blend) * origin_kp + blend * kp,
        (1.0 - blend) * origin_kd + blend * kd,
    )
    if policy_frame >= config.exit_policy_frame and (
        controller.option_contact_observed or policy_frame >= config.exit_policy_frame + 30
    ):
        controller.option_active = False
        controller.option_completed = True
    return blended, policy_frame


def _contact_teacher_direction(
    controller: _PlayerController,
    *,
    data: Any,
    ball_qpos: int,
    goal: G1TrainingGoalSpec,
    left_goal_plane_x_m: float,
) -> NDArray[np.float64]:
    decision = controller.decision
    if decision is None:
        raise RuntimeError("contact teacher has no current agent decision")
    ball = np.asarray(data.qpos[ball_qpos : ball_qpos + 2], dtype=np.float64)
    target = np.asarray(decision.target_position_m[:2], dtype=np.float64)
    if decision.intent in {TacticalIntent.RECEIVE, TacticalIntent.INTERCEPT}:
        target = np.asarray(
            (
                goal.plane_x_m
                if controller.cell.self_model.team_id == "red"
                else left_goal_plane_x_m,
                0.0,
            ),
            dtype=np.float64,
        )
    elif decision.intent is TacticalIntent.SAVE:
        target = ball + np.asarray(
            (-1.0, 0.0) if controller.cell.self_model.team_id == "blue" else (1.0, 0.0),
            dtype=np.float64,
        )
    direction = target - ball
    if float(np.linalg.norm(direction)) <= 1.0e-9:
        direction[0] = 1.0 if controller.cell.self_model.team_id == "red" else -1.0
    return direction


def _contact_foot_lateral_sign(
    *,
    controller: _PlayerController,
    data: Any,
    desired_direction_xy: NDArray[Any],
    use_left: bool,
) -> float:
    direction = np.asarray(desired_direction_xy, dtype=np.float64)
    direction /= max(float(np.linalg.norm(direction)), 1.0e-9)
    desired_lateral = np.asarray((-direction[1], direction[0]), dtype=np.float64)
    yaw = _pelvis_yaw(
        np.asarray(data.qpos[controller.qpos_base + 3 : controller.qpos_base + 7], dtype=np.float64)
    )
    anatomical_left = np.asarray((-math.sin(yaw), math.cos(yaw)), dtype=np.float64)
    left_sign = 1.0 if float(np.dot(anatomical_left, desired_lateral)) >= 0.0 else -1.0
    return left_sign if use_left else -left_sign


def _normalized_locomotion_command(policy: Any, physical: NDArray[Any]) -> NDArray[np.float64]:
    command = np.asarray(physical, dtype=np.float64)
    ranges = np.asarray((policy.range_velx, policy.range_vely, policy.range_velz), dtype=np.float64)
    command = np.clip(command, ranges[:, 0], ranges[:, 1])
    widths = ranges[:, 1] - ranges[:, 0]
    if command.shape != (3,) or np.any(widths <= 0.0):
        raise ValueError("locomotion command contract changed")
    return np.asarray(-1.0 + 2.0 * (command - ranges[:, 0]) / widths, dtype=np.float64)


def _run_locomotion(
    controller: _PlayerController,
    *,
    mirror: bool,
    correct_mirrored_yaw: bool = False,
    synchronize_action_frame: bool = False,
) -> None:
    """Keep the previous private action in the same frame as current inputs.

    This fixes changes between ordinary and sagittally reflected inference.
    It does not import another controller's unexecuted history or claim that
    this shadow locomotion proposal was physically applied while SONIC owned
    the body. Sensor restoration remains in the existing inference adapter.
    """
    if not synchronize_action_frame:
        _run_locomotion_in_frame(
            controller, mirror=mirror, correct_mirrored_yaw=correct_mirrored_yaw
        )
        return
    previous = np.asarray(controller.policy.action).copy()
    previous_frame = controller.locomotion_reflection_frame
    controller.policy.action = remap_previous_locomotion_action(
        previous,
        default_angles=np.asarray(controller.policy.default_angles),
        action_scale=controller.policy.action_scale,
        joint_to_motor=np.asarray(controller.policy.joint2motor_idx),
        previous_reflected=mirror if previous_frame is None else previous_frame,
        next_reflected=mirror,
    )
    try:
        _run_locomotion_in_frame(
            controller, mirror=mirror, correct_mirrored_yaw=correct_mirrored_yaw
        )
    except Exception:
        controller.policy.action = previous
        raise
    controller.locomotion_reflection_frame = mirror
    if previous_frame is not None and previous_frame != mirror:
        controller.locomotion_frame_switch_count += 1


def _run_locomotion_in_frame(
    controller: _PlayerController,
    *,
    mirror: bool,
    correct_mirrored_yaw: bool = False,
) -> None:
    """Reuse the qualified sagittal mirror for the actor's weak -y half-space."""

    keeper_reach = getattr(controller, "keeper_reach", None)
    if keeper_reach is not None and keeper_reach.config.neutralize_foundation_arm_observation:
        q, dq = controller.state.q.copy(), controller.state.dq.copy()
        try:
            # Explicit policy-input ablation, never a change to measured state
            # or physics. Gravity, angular velocity and leg proprioception remain real.
            controller.state.q[15:] = controller.policy.default_angles_reorder[15:]
            controller.state.dq[15:] = 0
            _run_locomotion_unmasked(
                controller, mirror=mirror, correct_mirrored_yaw=correct_mirrored_yaw
            )
        finally:
            controller.state.q, controller.state.dq = q, dq
        return
    _run_locomotion_unmasked(controller, mirror=mirror, correct_mirrored_yaw=correct_mirrored_yaw)


def _run_locomotion_unmasked(
    controller: _PlayerController, *, mirror: bool, correct_mirrored_yaw: bool
) -> None:

    if not mirror:
        with contextlib.redirect_stdout(io.StringIO()):
            controller.policy.run()
        return
    state = controller.state
    original = {
        name: np.asarray(getattr(state, name), dtype=np.float64).copy()
        for name in ("q", "dq", "gravity_ori", "ang_vel", "vel_cmd")
    }
    try:
        state.q = mirror_g1_joint_positions(original["q"])
        state.dq = mirror_g1_joint_positions(original["dq"])
        gravity = original["gravity_ori"].copy()
        gravity[1] *= -1.0
        state.gravity_ori = gravity
        angular = original["ang_vel"].copy()
        angular[(0, 2),] *= -1.0
        state.ang_vel = angular
        command = original["vel_cmd"].copy()
        command[1] *= -1.0
        # A sagittal reflection reverses both lateral translation and yaw.
        # Gate the correction to the new phase controller so frozen receiving
        # behavior retains its qualified distribution while the new transition
        # no longer turns opposite its world command.
        if correct_mirrored_yaw:
            command[2] *= -1.0
        state.vel_cmd = command
        with contextlib.redirect_stdout(io.StringIO()):
            controller.policy.run()
        controller.output.actions = mirror_g1_joint_positions(
            np.asarray(controller.output.actions, dtype=np.float64)
        )
        controller.output.kps = mirror_g1_joint_gains(
            np.asarray(controller.output.kps, dtype=np.float64)
        )
        controller.output.kds = mirror_g1_joint_gains(
            np.asarray(controller.output.kds, dtype=np.float64)
        )
    finally:
        for name, value in original.items():
            setattr(state, name, value)


def _append_player_trace(
    trace: dict[str, list[Any]], *, controller: _PlayerController, data: Any, model: Any = None
) -> None:
    decision = controller.decision
    command = controller.last_world_command
    if decision is None or command is None:
        raise RuntimeError("cannot trace an undecided independent agent")
    key = _agent_key(controller.cell.agent_id)
    trace[f"{key}_pelvis_pose"].append(
        data.qpos[controller.qpos_base : controller.qpos_base + 7].copy()
    )
    trace[f"{key}_joint_position"].append(data.qpos[controller.joint_qpos].copy())
    trace[f"{key}_joint_velocity"].append(data.qvel[controller.joint_qvel].copy())
    trace[f"{key}_joint_torque"].append(data.ctrl[controller.actuators].copy())
    if f"{key}_joint_safety_margin_rad" in trace:
        if model is None:
            raise ValueError("joint margin tracing requires the physical model")
        q = np.asarray(data.qpos[controller.joint_qpos], dtype=np.float64)
        ranges = np.asarray(model.jnt_range[controller.joint_ids])
        margin = np.minimum(q - ranges[:, 0], ranges[:, 1] - q)
        margin[~model.jnt_limited[controller.joint_ids].astype(bool)] = 1.0
        trace[f"{key}_joint_safety_margin_rad"].append(margin)
    trace[f"{key}_left_foot_position"].append(data.xpos[controller.left_ankle_body].copy())
    trace[f"{key}_right_foot_position"].append(data.xpos[controller.right_ankle_body].copy())
    trace[f"{key}_intent_code"].append(_INTENT_CODES[decision.intent])
    trace[f"{key}_target_position"].append(decision.target_position_m)
    trace[f"{key}_world_command"].append(command.copy())
    trace[f"{key}_movement_active"].append(float(np.linalg.norm(command[:2])) >= 0.04)
    if controller.keeper_reach is not None:
        trace[f"{key}_keeper_active"].append(controller.keeper_reach.active)
        trace[f"{key}_keeper_residual"].append(controller.keeper_reach.previous.copy())
        trace[f"{key}_keeper_punch_torque"].append(controller.keeper_reach.torque_nm.copy())
    if f"{key}_clearance_feasible" in trace:
        trace[f"{key}_clearance_feasible"].append(controller.clearance_feasible)
        trace[f"{key}_clearance_constrained"].append(controller.clearance_constrained)


def _robot_robot_contact_observation(
    model: Any,
    data: Any,
    controllers: tuple[_PlayerController, ...],
) -> tuple[int, str | None, str | None, float]:
    owner: dict[int, str] = {}
    for controller in controllers:
        for geom in controller.robot_geoms:
            owner[geom] = controller.cell.agent_id
    count = 0
    peak_first: str | None = None
    peak_second: str | None = None
    peak_force_n = 0.0
    wrench: NDArray[np.float64] = np.zeros(6, dtype=np.float64)
    import mujoco

    for index in range(int(data.ncon)):
        contact = data.contact[index]
        first = owner.get(int(contact.geom1))
        second = owner.get(int(contact.geom2))
        if first is None or second is None or first == second:
            continue
        count += 1
        mujoco.mj_contactForce(model, data, index, wrench)
        force_n = float(np.linalg.norm(wrench[:3]))
        if force_n >= peak_force_n:
            peak_first = first
            peak_second = second
            peak_force_n = force_n
    return count, peak_first, peak_second, peak_force_n


def _robot_geom_ids(model: Any, root_body: int) -> frozenset[int]:
    values: set[int] = set()
    for geom in range(int(model.ngeom)):
        body = int(model.geom_bodyid[geom])
        while body > 0 and body != root_body:
            body = int(model.body_parentid[body])
        if body == root_body:
            values.add(geom)
    return frozenset(values)


def _goalkeeper_glove_geoms(
    model: Any,
    *,
    prefix: str,
    side: str,
    enabled: bool,
) -> frozenset[int]:
    """Resolve the explicit regulation glove geom without broad hand aliases."""

    if not enabled:
        return frozenset()
    if side not in {"left", "right"}:
        raise ValueError("goalkeeper glove side is invalid")
    import mujoco

    geom = _id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        f"{prefix}{side}_goalkeeper_glove",
    )
    return frozenset((geom,))


def _gravity_orientation(quaternion: NDArray[Any]) -> NDArray[np.float64]:
    value = np.asarray(quaternion, dtype=np.float64)
    if value.shape != (4,) or not np.all(np.isfinite(value)):
        raise ValueError("pelvis quaternion is invalid")
    qw, qx, qy, qz = map(float, value)
    return np.asarray(
        (
            2.0 * (-qz * qx + qw * qy),
            -2.0 * (qz * qy + qw * qx),
            1.0 - 2.0 * (qw * qw + qz * qz),
        ),
        dtype=np.float64,
    )


def _project_joint_safe_torque(
    *,
    joint_position: NDArray[Any],
    joint_velocity: NDArray[Any],
    commanded_torque: NDArray[Any],
    joint_ranges: NDArray[Any],
    limited: NDArray[Any],
    margin_rad: float = 0.04,
    prediction_horizon_sec: float = 0.08,
    boundary_kp: float = 80.0,
    boundary_kd: float = 6.0,
) -> NDArray[np.float64]:
    """Project outward torque away from predicted G1 joint-limit threats."""

    q = np.asarray(joint_position, dtype=np.float64)
    dq = np.asarray(joint_velocity, dtype=np.float64)
    torque = np.asarray(commanded_torque, dtype=np.float64)
    ranges = np.asarray(joint_ranges, dtype=np.float64)
    constrained = np.asarray(limited, dtype=bool)
    if (
        q.shape != (29,)
        or dq.shape != (29,)
        or torque.shape != (29,)
        or ranges.shape != (29, 2)
        or constrained.shape != (29,)
        or not all(np.all(np.isfinite(value)) for value in (q, dq, torque, ranges))
    ):
        raise ValueError("joint guard requires finite 29-DoF G1 state")
    projected = torque.copy()
    predicted = q + prediction_horizon_sec * dq
    lower = ranges[:, 0] + margin_rad
    upper = ranges[:, 1] - margin_rad
    lower_threat = constrained & (predicted < lower)
    upper_threat = constrained & (predicted > upper)
    lower_brake = boundary_kp * (lower - q) - boundary_kd * dq
    upper_brake = boundary_kp * (upper - q) - boundary_kd * dq
    projected[lower_threat] = np.maximum(projected[lower_threat], lower_brake[lower_threat])
    projected[upper_threat] = np.minimum(projected[upper_threat], upper_brake[upper_threat])
    return projected


def _roll_pitch(quaternion: NDArray[Any]) -> tuple[float, float]:
    value = np.array(quaternion, dtype=np.float64, copy=True)
    value /= max(float(np.linalg.norm(value)), 1.0e-12)
    w, x, y, z = map(float, value)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))
    return roll, pitch


def _pelvis_yaw(quaternion: NDArray[Any]) -> float:
    value = np.array(quaternion, dtype=np.float64, copy=True)
    value /= max(float(np.linalg.norm(value)), 1.0e-12)
    w, x, y, z = map(float, value)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _rotate_z(vector: NDArray[Any], yaw: float) -> NDArray[np.float64]:
    value = np.asarray(vector, dtype=np.float64)
    if value.shape != (3,):
        raise ValueError("planar locomotion vector must be xyz")
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return np.asarray(
        (
            cosine * value[0] - sine * value[1],
            sine * value[0] + cosine * value[1],
            value[2],
        ),
        dtype=np.float64,
    )


def _agent_key(agent_id: str) -> str:
    return agent_id.replace(".", "_").replace(":", "_").replace("-", "_")


def _id(model: Any, object_type: Any, name: str) -> int:
    import mujoco

    value = int(mujoco.mj_name2id(model, object_type, name))
    if value < 0:
        raise ValueError(f"multi-player G1 model is missing {name}")
    return value


__all__ = [
    "AgentWorldQuality",
    "IndependentTeamWorldConfig",
    "IndependentTeamWorldResult",
    "IndependentTeamWorldScenario",
    "simulate_independent_team_world",
]
