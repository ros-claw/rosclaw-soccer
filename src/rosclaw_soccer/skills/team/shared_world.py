"""Multi-role football rollout in one deterministic MuJoCo world.

This module is the Soccer-owned successor to the historical Core experiment.
It runs one passer, one shooter, one goalkeeper and one physical ball under a
single CPU MuJoCo solver.  It never opens ROS, DDS, a vendor SDK or hardware.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from rosclaw.simforge.g1_cerebellar_recovery import G1CerebellarRecoveryConfig
from rosclaw.simforge.models import Partition
from rosclaw.simforge.tasks.g1_goalforge.scenario import GoalForgeScenario

from rosclaw_soccer.growth.ballistic_contact_impulse_actor import (
    G1BallisticContactImpulseActor,
    g1_ballistic_contact_impulse_effect,
    load_g1_ballistic_contact_impulse_actor,
    select_g1_ballistic_contact_effect,
)
from rosclaw_soccer.growth.ballistic_contact_residual import (
    G1BallisticContactResidualConfig,
    blend_g1_ballistic_contact_target,
)
from rosclaw_soccer.growth.ballistic_contact_torque_residual import (
    G1BallisticContactTorqueResidualConfig,
    g1_ballistic_contact_torque_residual,
)
from rosclaw_soccer.growth.causal_skill_transition import (
    CausalSkillTransitionActor,
    CausalTransitionDecision,
    causal_transition_features,
    load_causal_skill_transition_actor,
)
from rosclaw_soccer.growth.causal_strike_option import (
    CausalStrikeOptionDecision,
    CausalStrikeOptionObservation,
    CausalStrikeOptionPhase,
    G1CausalStrikeOptionConfig,
    G1CausalStrikeOptionController,
)
from rosclaw_soccer.growth.causal_strike_router import CausalStrikeRouteDecision
from rosclaw_soccer.growth.first_touch_interception import (
    FirstTouchInterceptionConfig,
    first_touch_interception_effect,
)
from rosclaw_soccer.growth.football_motion_prior import (
    G1FootballMotionPrior,
    blend_g1_football_motion_prior_displacement,
    blend_g1_football_motion_prior_right_leg_velocity,
    blend_g1_football_motion_prior_target,
    blend_g1_football_motion_prior_velocity,
    load_g1_football_motion_prior,
)
from rosclaw_soccer.growth.mosaic_agility_prior import (
    G1MosaicAgilityPrior,
    blend_g1_mosaic_agility_target,
    blend_g1_mosaic_agility_velocity,
    load_g1_mosaic_agility_prior,
)
from rosclaw_soccer.growth.mosaic_g1_contract import (
    MOSAIC_G1_ISAACLAB_TO_CANONICAL_DOF,
)
from rosclaw_soccer.growth.mosaic_gmt import (
    G1MosaicGMTOverheadSkill,
    MosaicGMTContract,
    MosaicGMTTorchController,
    load_g1_mosaic_gmt_overhead_skill,
    load_mosaic_gmt_torch,
)
from rosclaw_soccer.growth.mosaic_overhead_reach_prior import (
    G1MosaicOverheadReachPrior,
    blend_g1_mosaic_overhead_reach_target,
    load_g1_mosaic_overhead_reach_prior,
)
from rosclaw_soccer.growth.neural_contact_actor import (
    G1NeuralContactActor,
    evaluate_neural_contact_actor,
    load_g1_neural_contact_actor,
    neural_contact_features,
)
from rosclaw_soccer.growth.reactive_route_actor import (
    reactive_route_features,
)
from rosclaw_soccer.growth.runtime_causal_strike_router import (
    G1RuntimeCausalStrikeRouter,
    load_runtime_causal_strike_router,
    runtime_causal_strike_features,
)
from rosclaw_soccer.growth.runtime_contact_mode_actor import (
    G1RuntimeContactModeActor,
    RuntimeContactModeAction,
    RuntimeContactModeDecision,
    load_runtime_contact_mode_actor,
)
from rosclaw_soccer.growth.runtime_contact_target_actor import (
    G1RuntimeContactTargetActor,
    RuntimeContactTargetDecision,
    load_runtime_contact_target_actor,
)
from rosclaw_soccer.growth.runtime_finish_plan_actor import (
    G1RuntimeFinishPlanActor,
    RuntimeFinishPlanDecision,
    load_runtime_finish_plan_actor,
    prepared_finish_plan_features,
)
from rosclaw_soccer.growth.runtime_receive_actor import (
    G1RuntimeReceiveActor,
    RuntimeReceiveAction,
    RuntimeReceiveDecision,
    load_runtime_receive_actor,
    runtime_receive_features,
)
from rosclaw_soccer.growth.target_velocity_contact_actor import (
    G1TargetVelocityContactActor,
    g1_target_velocity_contact_effect,
    load_g1_target_velocity_contact_actor,
)
from rosclaw_soccer.growth.temporal_route_actor import (
    G1TemporalRouteActor,
    RouteActor,
    TemporalRouteMemory,
    load_route_actor,
)
from rosclaw_soccer.growth.three_axis_contact_actor import (
    G1ThreeAxisContactActor,
    g1_three_axis_contact_effect,
    load_g1_three_axis_contact_actor,
)
from rosclaw_soccer.providers.g1.asset_qualification import (
    G1AssetQualification,
    qualify_g1_assets,
)
from rosclaw_soccer.providers.g1.iql_artifact import (
    IQLResidualDecision,
    IQLResidualGuardConfig,
    NumpyIQLActor,
    SupportBoundIQLResidualActor,
)
from rosclaw_soccer.providers.g1.mujoco_primitives import (
    adapt_shot_target as _adapt_target,
)
from rosclaw_soccer.providers.g1.mujoco_primitives import (
    build_shared_recovery_controller,
    shared_post_impact_recovery_config,
)
from rosclaw_soccer.providers.g1.mujoco_primitives import (
    load_robonaldo as _load_robonaldo,
)
from rosclaw_soccer.providers.g1.mujoco_primitives import (
    mirror_g1_joint_gains as _mirror_g1_joint_gains,
)
from rosclaw_soccer.providers.g1.mujoco_primitives import (
    mirror_g1_joint_positions as _mirror_g1_joint_positions,
)
from rosclaw_soccer.providers.g1.mujoco_primitives import (
    policy_repeat_count as _policy_repeat_count,
)
from rosclaw_soccer.providers.g1.mujoco_primitives import (
    quaternion_multiply as _quaternion_multiply,
)
from rosclaw_soccer.providers.g1.mujoco_primitives import (
    roll_pitch as _roll_pitch,
)
from rosclaw_soccer.providers.g1.transition_bridge import (
    G1TransitionBridgeConfig,
    G1VelocityMatchedTransitionBridge,
)
from rosclaw_soccer.sim.contracts import (
    G1_DDS_JOINT_NAMES,
    G1_HARD_TORQUE_LIMITS,
    ShotParameters,
    hash_bytes,
    hash_json,
)
from rosclaw_soccer.skills.goalkeeper_v2.external_reference import (
    HumanoidGoalkeeperReferenceAction,
    NumpyHumanoidGoalkeeperReferenceActor,
    load_humanoid_goalkeeper_reference_actor,
)
from rosclaw_soccer.skills.goalkeeper_v2.observations import (
    GoalkeeperActorObservation,
    GoalkeeperActorObserver,
)
from rosclaw_soccer.skills.goalkeeper_v2.policy import (
    NumpyGoalkeeperActor,
    load_goalkeeper_actor_artifact,
)
from rosclaw_soccer.skills.shoot.loft_teacher import (
    G1LoftTeacherConfig,
    g1_loft_teacher_effect,
)
from rosclaw_soccer.training.goalkeeper_dive_option import (
    GoalkeeperBalancedDiveSeed,
    balanced_dive_qualified_impedance,
    build_balanced_dive_imitation_seed,
    load_official_goalkeeper_dive_atlas,
)
from rosclaw_soccer.training.goalkeeper_whole_body_reach import (
    G1WholeBodyReachAtlas,
    load_g1_whole_body_reach_atlas,
    whole_body_reach_from_target_numpy,
)
from rosclaw_soccer.world.field import (
    G1_GOALKEEPER_GLOVE_CENTER_M,
    G1_GOALKEEPER_GLOVE_HALF_EXTENTS_M,
    G1CompliantGoalNetState,
    G1TrainingGoalSpec,
    apply_g1_compliant_goal_net_force,
    build_g1_coupled_stadium_model,
    build_g1_four_player_two_ball_stadium_model,
    build_g1_three_player_stadium_model,
    g1_ball_inside_goal_mouth,
)

_MOTION_REL = Path("policy/robonaldo/model/freekick_motion.npz")
_PASSER_ORIGIN = np.asarray((3.7452039279962945, -0.16406006503921598, 0.0))
_PASSER_YAW = math.pi
_SHOOTER_START_SEC = 2.02
_CONTROL_DT = 0.02
_PHYSICS_DT = 0.002
_SUBSTEPS = 10
_TOTAL_TIME_SEC = 15.0


@dataclass(frozen=True)
class G1JointGuardConfig:
    margin_rad: float = 0.04
    prediction_horizon_sec: float = 0.08
    boundary_kp: float = 80.0
    boundary_kd: float = 6.0
    schema_version: str = "rosclaw.growth.g1_joint_guard_config.v1"

    def __post_init__(self) -> None:
        values = (
            self.margin_rad,
            self.prediction_horizon_sec,
            self.boundary_kp,
            self.boundary_kd,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("joint guard config must be finite")
        if not 0.01 <= self.margin_rad <= 0.10:
            raise ValueError("joint guard margin must be in [0.01, 0.10]")
        if not 0.02 <= self.prediction_horizon_sec <= 0.20:
            raise ValueError("joint guard horizon must be in [0.02, 0.20]")
        if not 20.0 <= self.boundary_kp <= 200.0:
            raise ValueError("joint guard kp must be in [20, 200]")
        if not 1.0 <= self.boundary_kd <= 20.0:
            raise ValueError("joint guard kd must be in [1, 20]")


@dataclass(frozen=True)
class G1MovementWaypoint:
    """One immutable SIM-only world-frame waypoint for learned locomotion."""

    time_sec: float
    position_m: tuple[float, float, float]
    schema_version: str = "rosclaw.soccer.g1_movement_waypoint.v1"

    def __post_init__(self) -> None:
        position = np.asarray(self.position_m, dtype=np.float64)
        if not math.isfinite(self.time_sec) or not 0.0 <= self.time_sec <= 120.0:
            raise ValueError("movement waypoint time must be in [0, 120] seconds")
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("movement waypoint position must be a finite xyz vector")
        if not -2.0 <= position[0] <= 12.0 or abs(float(position[1])) > 4.0:
            raise ValueError("movement waypoint must remain inside the bounded soccer field")
        if abs(float(position[2])) > 1.0e-12:
            raise ValueError("movement waypoint must remain on the ground plane")


@dataclass(frozen=True)
class G1TacticalMovementConfig:
    """Bounded tactical target stream executed by the frozen locomotion actor.

    This contract never writes body pose or joint state.  It produces a causal
    velocity command for the already loaded RoboNaldo locomotion policy in the
    shared MuJoCo world.  It has no REAL/hardware execution authority.
    """

    waypoints: tuple[G1MovementWaypoint, ...]
    maximum_speed_mps: float = 0.55
    maximum_acceleration_mps2: float = 1.20
    position_gain: float = 1.50
    arrival_radius_m: float = 0.04
    execution_mode: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw.soccer.g1_tactical_movement_config.v1"

    def __post_init__(self) -> None:
        if not 2 <= len(self.waypoints) <= 16:
            raise ValueError("tactical movement requires between 2 and 16 waypoints")
        times = tuple(waypoint.time_sec for waypoint in self.waypoints)
        if any(later <= earlier for earlier, later in zip(times, times[1:], strict=False)):
            raise ValueError("tactical movement waypoint times must be strictly increasing")
        values = (
            self.maximum_speed_mps,
            self.maximum_acceleration_mps2,
            self.position_gain,
            self.arrival_radius_m,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("tactical movement limits must be finite")
        if not 0.10 <= self.maximum_speed_mps <= 0.70:
            raise ValueError("tactical movement speed must be in [0.10, 0.70] m/s")
        if not 0.20 <= self.maximum_acceleration_mps2 <= 3.0:
            raise ValueError("tactical movement acceleration must be in [0.20, 3.0] m/s^2")
        if not 0.20 <= self.position_gain <= 4.0:
            raise ValueError("tactical movement position gain must be in [0.20, 4.0]")
        if not 0.02 <= self.arrival_radius_m <= 0.20:
            raise ValueError("tactical movement arrival radius must be in [0.02, 0.20] m")
        if self.execution_mode != "SIM_ONLY" or self.hardware_authorized:
            raise ValueError("tactical movement is permanently SIM_ONLY")

    @property
    def config_hash(self) -> str:
        return hash_json(asdict(self))


@dataclass(frozen=True)
class G1ReactiveMovementConfig:
    """Content-bound observation actor routed into frozen locomotion."""

    actor_artifact_path: str
    actor_hash: str
    target_position_m: tuple[float, float, float]
    action: str
    role: str
    maximum_speed_mps: float = 0.55
    maximum_acceleration_mps2: float = 1.20
    arrival_radius_m: float = 0.04
    minimum_role_separation_m: float = 0.90
    collision_avoidance_gain: float = 1.50
    maximum_collision_correction_mps: float = 0.25
    post_reception_follow_through_m: float = 0.35
    far_target_activation_distance_m: float = 0.55
    far_target_speed_gain: float = 1.20
    velocity_braking_gain: float = 0.45
    maximum_velocity_braking_correction_mps: float = 0.18
    diagonal_braking_target_dx_start_m: float = 0.40
    diagonal_braking_target_dx_full_m: float = 0.60
    diagonal_braking_target_dy_start_m: float = 0.35
    diagonal_braking_target_dy_full_m: float = 0.50
    execution_mode: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw.soccer.g1_reactive_movement_config.v1"

    def __post_init__(self) -> None:
        target = np.asarray(self.target_position_m, dtype=np.float64)
        if not self.actor_hash.startswith("sha256:") or len(self.actor_hash) != 71:
            raise ValueError("reactive movement actor must be content bound")
        if not self.actor_artifact_path or not Path(self.actor_artifact_path).is_file():
            raise ValueError("reactive movement actor artifact does not exist")
        if (
            target.shape != (3,)
            or not np.all(np.isfinite(target))
            or not -2.0 <= target[0] <= 12.0
            or abs(float(target[1])) > 4.0
            or abs(float(target[2])) > 1.0e-12
        ):
            raise ValueError("reactive movement target must remain on the bounded field")
        if self.action not in {"pass", "shoot"} or self.role not in {
            "teammate",
            "defender",
        }:
            raise ValueError("reactive movement action or role is unsupported")
        values = (
            self.maximum_speed_mps,
            self.maximum_acceleration_mps2,
            self.arrival_radius_m,
            self.minimum_role_separation_m,
            self.collision_avoidance_gain,
            self.maximum_collision_correction_mps,
            self.post_reception_follow_through_m,
            self.far_target_activation_distance_m,
            self.far_target_speed_gain,
            self.velocity_braking_gain,
            self.maximum_velocity_braking_correction_mps,
            self.diagonal_braking_target_dx_start_m,
            self.diagonal_braking_target_dx_full_m,
            self.diagonal_braking_target_dy_start_m,
            self.diagonal_braking_target_dy_full_m,
        )
        if (
            not all(math.isfinite(value) for value in values)
            or not 0.10 <= self.maximum_speed_mps <= 0.70
            or not 0.20 <= self.maximum_acceleration_mps2 <= 3.0
            or not 0.02 <= self.arrival_radius_m <= 0.20
            or not 0.50 <= self.minimum_role_separation_m <= 1.20
            or not 0.20 <= self.collision_avoidance_gain <= 3.0
            or not 0.05 <= self.maximum_collision_correction_mps <= 0.30
            or not 0.0 <= self.post_reception_follow_through_m <= 0.60
            or not 0.30 <= self.far_target_activation_distance_m <= 1.00
            or not 1.0 <= self.far_target_speed_gain <= 1.40
            or not 0.0 <= self.velocity_braking_gain <= 1.0
            or not 0.0 <= self.maximum_velocity_braking_correction_mps <= 0.25
            or not 0.20 <= self.diagonal_braking_target_dx_start_m <= 0.60
            or not 0.40 <= self.diagonal_braking_target_dx_full_m <= 0.80
            or self.diagonal_braking_target_dx_start_m >= self.diagonal_braking_target_dx_full_m
            or not 0.20 <= self.diagonal_braking_target_dy_start_m <= 0.60
            or not 0.40 <= self.diagonal_braking_target_dy_full_m <= 0.80
            or self.diagonal_braking_target_dy_start_m >= self.diagonal_braking_target_dy_full_m
        ):
            raise ValueError("reactive movement limits violate the locomotion envelope")
        if self.execution_mode != "SIM_ONLY" or self.hardware_authorized:
            raise ValueError("reactive movement is permanently SIM_ONLY")

    @property
    def config_hash(self) -> str:
        return hash_json(asdict(self))


@dataclass(frozen=True)
class G1GoalkeeperConfig:
    """Causal locomotion/reach adapter for the goalkeeper policy instance."""

    depth_from_goal_line_m: float = 0.48
    initial_lateral_position_m: float = 0.0
    reaction_delay_sec: float = 0.12
    lateral_position_gain: float = 1.35
    maximum_lateral_speed_mps: float = 0.38
    depth_position_gain: float = 1.20
    maximum_depth_correction_mps: float = 0.20
    ready_shuffle_speed_mps: float = 0.10
    ready_shuffle_period_sec: float = 2.4
    prestrike_ball_positioning_enabled: bool = False
    prestrike_ball_lateral_blend: float = 0.85
    canonical_locomotion_mirror_enabled: bool = False
    regulation_goal_positioning_enabled: bool = False
    post_contact_stabilization_enabled: bool = False
    post_contact_ready_recovery_enabled: bool = False
    post_contact_ready_recovery_delay_sec: float = 4.0
    post_contact_ready_yaw_gain: float = 0.4
    post_contact_ready_maximum_yaw_rate_rad_s: float = 0.12
    post_contact_ready_lateral_position_gain: float = 0.6
    post_contact_ready_maximum_lateral_speed_mps: float = 0.12
    post_contact_ready_lateral_deadband_m: float = 0.04
    recovery_athlete_checkpoint_path: Path | None = None
    recovery_athlete_exam_path: Path | None = None
    recovery_athlete_blend: float = 0.0
    recovery_athlete_authority_envelope_enabled: bool = False
    successor_lateral_probe_enabled: bool = False
    successor_lateral_probe_delay_sec: float = 8.0
    successor_lateral_probe_duration_sec: float = 0.8
    successor_lateral_probe_command_mps: float = 0.0
    arm_spread_rad: float = 0.24
    maximum_waist_lean_rad: float = 0.08
    actor_observation_mode: str = "legacy_shooter_phase"
    actor_artifact_path: Path | None = None
    actor_minimum_target_height_m: float = 0.0
    actor_minimum_current_ball_height_m: float = 0.0
    actor_minimum_incoming_ball_speed_mps: float = 0.10
    actor_threat_warmup_sec: float = 0.12
    actor_minimum_intercept_confidence: float = 0.50
    actor_operational_space_reach_local_x_m: float = -0.28
    actor_operational_space_reach_side_offset_m: float = 0.0
    actor_bimanual_reach_enabled: bool = False
    actor_bimanual_reach_local_x_m: float = 0.0
    actor_bimanual_reach_half_span_m: float = 0.08
    actor_bimanual_reach_height_offset_m: float = 0.0
    actor_bimanual_reach_minimum_fraction: float = 0.0
    actor_bimanual_reach_gain_scale: float = 1.0
    actor_bimanual_reach_memory_decay: float = 0.82
    actor_bimanual_reach_memory_maximum_rad: float = 0.40
    actor_bimanual_support_arm_blend: float = 0.0
    actor_bimanual_support_arm_overhead_bias_rad: float = 0.0
    actor_bimanual_punch_force_n: float = 0.0
    actor_bimanual_punch_vertical_force_scale: float = 0.0
    actor_bimanual_punch_outward_force_scale: float = 0.0
    actor_bimanual_punch_central_boost_n_per_m: float = 0.0
    actor_bimanual_punch_reference_abs_lateral_m: float = 0.45
    actor_bimanual_punch_window_sec: float = 0.30
    glove_contact_time_constant_sec: float = 0.02
    glove_contact_damping_ratio: float = 1.0
    joint_guard_margin_rad: float = 0.08
    joint_guard_prediction_horizon_sec: float = 0.16
    joint_guard_boundary_kp: float = 140.0
    joint_guard_boundary_kd: float = 12.0
    joint_guard_impact_lead_sec: float = 0.0
    external_reference_manifest_path: Path | None = None
    external_reference_blend: float = 0.0
    overhead_reach_prior_path: Path | None = None
    overhead_reach_blend: float = 0.0
    overhead_reach_minimum_target_height_m: float = 1.10
    overhead_reach_full_target_height_m: float = 1.30
    overhead_reach_waist_scale: float = 0.20
    overhead_reach_arm_scale: float = 1.0
    whole_body_reach_atlas_path: Path | None = None
    whole_body_reach_blend: float = 0.0
    whole_body_reach_minimum_target_height_m: float = 1.0
    whole_body_reach_full_target_height_m: float = 1.25
    whole_body_reach_timing_lead_sec: float = 0.10
    whole_body_reach_waist_scale: float = 1.0
    whole_body_reach_target_arm_scale: float = 1.0
    whole_body_reach_support_arm_scale: float = 1.0
    mosaic_gmt_model_path: Path | None = None
    mosaic_gmt_skill_path: Path | None = None
    mosaic_gmt_blend: float = 0.0
    mosaic_gmt_minimum_target_height_m: float = 1.0
    mosaic_gmt_full_target_height_m: float = 1.25
    mosaic_gmt_maximum_lateral_error_m: float = 1.50
    mosaic_gmt_timing_lead_sec: float = 0.10
    mosaic_gmt_lower_body_scale: float = 1.0
    mosaic_gmt_waist_scale: float = 1.0
    mosaic_gmt_arm_scale: float = 1.0
    mosaic_gmt_mirror_by_intercept: bool = True
    balanced_dive_source_checkout: Path | None = None
    balanced_dive_blend: float = 0.0
    balanced_dive_minimum_lateral_error_m: float = 0.30
    balanced_dive_activation_lead_sec: float = 0.0
    balanced_dive_initial_phase: float = 0.0
    balanced_dive_phase_at_arrival: float = 0.60
    balanced_dive_peak_phase: float = 1.0
    balanced_dive_lower_body_scale: float = 1.0
    balanced_dive_waist_scale: float = 1.0
    balanced_dive_arm_scale: float = 1.0
    balanced_dive_blend_in_sec: float = 0.24
    balanced_dive_recovery_tail_sec: float = 0.40
    balanced_dive_landing_capture_enabled: bool = False
    balanced_dive_landing_capture_sec: float = 0.80
    balanced_dive_landing_damping_scale: float = 1.50
    post_contact_proprioceptive_capture_enabled: bool = False
    post_contact_proprioceptive_capture_delay_sec: float = 0.80
    post_contact_proprioceptive_capture_window_sec: float = 1.50
    post_contact_proprioceptive_capture_maximum_root_speed_mps: float = 0.40
    post_contact_proprioceptive_capture_duration_sec: float = 0.80
    dive_athlete_checkpoint_path: Path | None = None
    dive_athlete_exam_path: Path | None = None
    dive_athlete_blend: float = 0.0
    anticipation_enabled: bool = False
    anticipation_start_policy_frame: int = 230
    anticipation_minimum_foot_ball_distance_m: float = 0.35
    anticipation_maximum_foot_ball_distance_m: float = 1.30
    anticipation_target_blend: float = 0.65
    anticipation_velocity_scale: float = 0.70
    block_action_enabled: bool = False
    block_action_timing_mode: str = "auto"
    block_action_start_policy_frame: int = 218
    block_action_blend_frames: int = 16
    block_action_hold_frames: int = 46
    block_action_waist_yaw_rad: float = 0.0
    block_action_waist_roll_rad: float = 0.0
    block_action_waist_pitch_rad: float = 0.0
    block_action_hip_pitch_rad: float = 0.0
    block_action_hip_roll_rad: float = 0.0
    block_action_knee_flex_rad: float = 0.0
    block_action_shoulder_pitch_rad: float = 0.0
    block_action_shoulder_roll_rad: float = 0.0
    block_action_elbow_flex_rad: float = 0.0
    schema_version: str = "rosclaw_soccer.g1_goalkeeper_config.v32"

    def __post_init__(self) -> None:
        values = (
            self.depth_from_goal_line_m,
            self.initial_lateral_position_m,
            self.reaction_delay_sec,
            self.lateral_position_gain,
            self.maximum_lateral_speed_mps,
            self.depth_position_gain,
            self.maximum_depth_correction_mps,
            self.ready_shuffle_speed_mps,
            self.ready_shuffle_period_sec,
            self.prestrike_ball_lateral_blend,
            self.arm_spread_rad,
            self.maximum_waist_lean_rad,
            self.actor_minimum_target_height_m,
            self.actor_minimum_current_ball_height_m,
            self.actor_minimum_incoming_ball_speed_mps,
            self.actor_threat_warmup_sec,
            self.actor_minimum_intercept_confidence,
            self.actor_operational_space_reach_local_x_m,
            self.actor_operational_space_reach_side_offset_m,
            self.actor_bimanual_reach_local_x_m,
            self.actor_bimanual_reach_half_span_m,
            self.actor_bimanual_reach_height_offset_m,
            self.actor_bimanual_reach_minimum_fraction,
            self.actor_bimanual_reach_gain_scale,
            self.actor_bimanual_reach_memory_decay,
            self.actor_bimanual_reach_memory_maximum_rad,
            self.actor_bimanual_support_arm_blend,
            self.actor_bimanual_support_arm_overhead_bias_rad,
            self.actor_bimanual_punch_force_n,
            self.actor_bimanual_punch_vertical_force_scale,
            self.actor_bimanual_punch_outward_force_scale,
            self.actor_bimanual_punch_central_boost_n_per_m,
            self.actor_bimanual_punch_reference_abs_lateral_m,
            self.actor_bimanual_punch_window_sec,
            self.glove_contact_time_constant_sec,
            self.glove_contact_damping_ratio,
            self.joint_guard_margin_rad,
            self.joint_guard_prediction_horizon_sec,
            self.joint_guard_boundary_kp,
            self.joint_guard_boundary_kd,
            self.joint_guard_impact_lead_sec,
            self.anticipation_minimum_foot_ball_distance_m,
            self.anticipation_maximum_foot_ball_distance_m,
            self.anticipation_target_blend,
            self.anticipation_velocity_scale,
            self.external_reference_blend,
            self.overhead_reach_blend,
            self.overhead_reach_minimum_target_height_m,
            self.overhead_reach_full_target_height_m,
            self.overhead_reach_waist_scale,
            self.overhead_reach_arm_scale,
            self.whole_body_reach_blend,
            self.whole_body_reach_minimum_target_height_m,
            self.whole_body_reach_full_target_height_m,
            self.whole_body_reach_timing_lead_sec,
            self.whole_body_reach_waist_scale,
            self.whole_body_reach_target_arm_scale,
            self.whole_body_reach_support_arm_scale,
            self.mosaic_gmt_blend,
            self.mosaic_gmt_minimum_target_height_m,
            self.mosaic_gmt_full_target_height_m,
            self.mosaic_gmt_maximum_lateral_error_m,
            self.mosaic_gmt_timing_lead_sec,
            self.mosaic_gmt_lower_body_scale,
            self.mosaic_gmt_waist_scale,
            self.mosaic_gmt_arm_scale,
            self.balanced_dive_blend,
            self.balanced_dive_minimum_lateral_error_m,
            self.balanced_dive_activation_lead_sec,
            self.balanced_dive_initial_phase,
            self.balanced_dive_phase_at_arrival,
            self.balanced_dive_peak_phase,
            self.balanced_dive_lower_body_scale,
            self.balanced_dive_waist_scale,
            self.balanced_dive_arm_scale,
            self.balanced_dive_blend_in_sec,
            self.balanced_dive_recovery_tail_sec,
            self.balanced_dive_landing_capture_sec,
            self.balanced_dive_landing_damping_scale,
            self.post_contact_proprioceptive_capture_delay_sec,
            self.post_contact_proprioceptive_capture_window_sec,
            self.post_contact_proprioceptive_capture_maximum_root_speed_mps,
            self.post_contact_proprioceptive_capture_duration_sec,
            self.dive_athlete_blend,
            self.post_contact_ready_recovery_delay_sec,
            self.post_contact_ready_yaw_gain,
            self.post_contact_ready_maximum_yaw_rate_rad_s,
            self.post_contact_ready_lateral_position_gain,
            self.post_contact_ready_maximum_lateral_speed_mps,
            self.post_contact_ready_lateral_deadband_m,
            self.recovery_athlete_blend,
            self.successor_lateral_probe_delay_sec,
            self.successor_lateral_probe_duration_sec,
            self.successor_lateral_probe_command_mps,
            self.block_action_waist_yaw_rad,
            self.block_action_waist_roll_rad,
            self.block_action_waist_pitch_rad,
            self.block_action_hip_pitch_rad,
            self.block_action_hip_roll_rad,
            self.block_action_knee_flex_rad,
            self.block_action_shoulder_pitch_rad,
            self.block_action_shoulder_roll_rad,
            self.block_action_elbow_flex_rad,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("goalkeeper config must be finite")
        if not 0.25 <= self.depth_from_goal_line_m <= 0.80:
            raise ValueError("goalkeeper depth must be in [0.25, 0.80] m")
        lateral_limit = 3.44 if self.regulation_goal_positioning_enabled else 1.50
        if not -lateral_limit <= self.initial_lateral_position_m <= lateral_limit:
            raise ValueError("goalkeeper initial lateral position exceeds its qualified pocket")
        if not 0.04 <= self.reaction_delay_sec <= 0.35:
            raise ValueError("goalkeeper reaction delay must be in [0.04, 0.35] s")
        if not 0.5 <= self.lateral_position_gain <= 2.5:
            raise ValueError("goalkeeper position gain must be in [0.5, 2.5]")
        if not 0.20 <= self.maximum_lateral_speed_mps <= 0.40:
            raise ValueError("goalkeeper lateral speed must be in [0.20, 0.40] m/s")
        if not 0.50 <= self.depth_position_gain <= 2.50:
            raise ValueError("goalkeeper depth gain must be in [0.50, 2.50]")
        if not 0.05 <= self.maximum_depth_correction_mps <= 0.30:
            raise ValueError("goalkeeper depth correction must be in [0.05, 0.30] m/s")
        if not 0.0 <= self.ready_shuffle_speed_mps <= 0.20:
            raise ValueError("goalkeeper ready shuffle must be in [0, 0.20] m/s")
        if not 1.5 <= self.ready_shuffle_period_sec <= 4.0:
            raise ValueError("goalkeeper shuffle period must be in [1.5, 4.0] s")
        if not isinstance(self.prestrike_ball_positioning_enabled, bool):
            raise ValueError("goalkeeper prestrike positioning flag must be boolean")
        if not 0.0 <= self.prestrike_ball_lateral_blend <= 1.0:
            raise ValueError("goalkeeper prestrike ball blend must be in [0, 1]")
        if not isinstance(self.canonical_locomotion_mirror_enabled, bool):
            raise ValueError("goalkeeper canonical locomotion mirror flag must be boolean")
        if not isinstance(self.regulation_goal_positioning_enabled, bool):
            raise ValueError("goalkeeper regulation positioning flag must be boolean")
        if not 0.10 <= self.arm_spread_rad <= 0.45:
            raise ValueError("goalkeeper arm spread must be in [0.10, 0.45] rad")
        if not 0.0 <= self.maximum_waist_lean_rad <= 0.15:
            raise ValueError("goalkeeper waist lean must be in [0, 0.15] rad")
        if self.actor_observation_mode not in {
            "legacy_shooter_phase",
            "visible_ball_history_v3",
        }:
            raise ValueError("unsupported goalkeeper actor observation mode")
        if not 0.0 <= self.actor_minimum_target_height_m <= 1.60:
            raise ValueError("goalkeeper actor height router is invalid")
        if not 0.0 <= self.actor_minimum_current_ball_height_m <= 1.0:
            raise ValueError("goalkeeper actor current-height router is invalid")
        if not 0.10 <= self.actor_minimum_incoming_ball_speed_mps <= 15.0:
            raise ValueError("goalkeeper actor incoming-speed router is invalid")
        if not 0.04 <= self.actor_threat_warmup_sec <= 0.20:
            raise ValueError("goalkeeper actor threat warmup is invalid")
        if not 0.25 <= self.actor_minimum_intercept_confidence <= 0.80:
            raise ValueError("goalkeeper actor intercept confidence gate is invalid")
        if not -0.35 <= self.actor_operational_space_reach_local_x_m <= 0.35:
            raise ValueError("goalkeeper actor operational reach depth is invalid")
        if not -0.20 <= self.actor_operational_space_reach_side_offset_m <= 0.20:
            raise ValueError("goalkeeper actor side-calibrated reach depth is invalid")
        if any(
            not -0.35 <= value <= 0.35
            for value in (
                self.actor_operational_space_reach_local_x_m
                - self.actor_operational_space_reach_side_offset_m,
                self.actor_operational_space_reach_local_x_m
                + self.actor_operational_space_reach_side_offset_m,
            )
        ):
            raise ValueError("goalkeeper actor side-calibrated reach exceeds its envelope")
        if not -0.35 <= self.actor_bimanual_reach_local_x_m <= 0.35:
            raise ValueError("goalkeeper bimanual reach depth is invalid")
        if not 0.04 <= self.actor_bimanual_reach_half_span_m <= 0.20:
            raise ValueError("goalkeeper bimanual reach half-span is invalid")
        if not -0.12 <= self.actor_bimanual_reach_height_offset_m <= 0.12:
            raise ValueError("goalkeeper bimanual reach height offset is invalid")
        if not 0.0 <= self.actor_bimanual_reach_minimum_fraction <= 1.0:
            raise ValueError("goalkeeper bimanual reach minimum fraction is invalid")
        if not 0.5 <= self.actor_bimanual_reach_gain_scale <= 4.0:
            raise ValueError("goalkeeper bimanual reach gain scale is invalid")
        if not 0.75 <= self.actor_bimanual_reach_memory_decay <= 0.98:
            raise ValueError("goalkeeper bimanual reach memory decay is invalid")
        if not 0.20 <= self.actor_bimanual_reach_memory_maximum_rad <= 0.80:
            raise ValueError("goalkeeper bimanual reach memory bound is invalid")
        if not 0.0 <= self.actor_bimanual_support_arm_blend <= 1.0:
            raise ValueError("goalkeeper bimanual support-arm blend is invalid")
        if not 0.0 <= self.actor_bimanual_support_arm_overhead_bias_rad <= 0.35:
            raise ValueError("goalkeeper bimanual support-arm overhead bias is invalid")
        if not 0.0 <= self.actor_bimanual_punch_force_n <= 160.0:
            raise ValueError("goalkeeper bimanual punch force is invalid")
        if not 0.0 <= self.actor_bimanual_punch_vertical_force_scale <= 1.0:
            raise ValueError("goalkeeper bimanual vertical punch scale is invalid")
        if not 0.0 <= self.actor_bimanual_punch_outward_force_scale <= 0.75:
            raise ValueError("goalkeeper bimanual outward punch scale is invalid")
        if not 0.0 <= self.actor_bimanual_punch_central_boost_n_per_m <= 80.0:
            raise ValueError("goalkeeper bimanual central punch boost is invalid")
        if not 0.20 <= self.actor_bimanual_punch_reference_abs_lateral_m <= 0.60:
            raise ValueError("goalkeeper bimanual punch lateral reference is invalid")
        peak_scalar_force = (
            self.actor_bimanual_punch_force_n
            + self.actor_bimanual_punch_central_boost_n_per_m
            * self.actor_bimanual_punch_reference_abs_lateral_m
        )
        peak_vector_scale = math.sqrt(
            1.0
            + self.actor_bimanual_punch_vertical_force_scale**2
            + self.actor_bimanual_punch_outward_force_scale**2
        )
        if peak_scalar_force * peak_vector_scale > 160.0:
            raise ValueError("goalkeeper adaptive bimanual punch exceeds its force envelope")
        if not 0.08 <= self.actor_bimanual_punch_window_sec <= 0.60:
            raise ValueError("goalkeeper bimanual punch window is invalid")
        if not 0.003 <= self.glove_contact_time_constant_sec <= 0.04:
            raise ValueError("goalkeeper glove contact time constant is invalid")
        if not 0.10 <= self.glove_contact_damping_ratio <= 2.0:
            raise ValueError("goalkeeper glove contact damping ratio is invalid")
        if not 0.04 <= self.joint_guard_margin_rad <= 0.10:
            raise ValueError("goalkeeper joint guard margin is invalid")
        if not 0.08 <= self.joint_guard_prediction_horizon_sec <= 0.20:
            raise ValueError("goalkeeper joint guard horizon is invalid")
        if not 80.0 <= self.joint_guard_boundary_kp <= 200.0:
            raise ValueError("goalkeeper joint guard kp is invalid")
        if not 6.0 <= self.joint_guard_boundary_kd <= 20.0:
            raise ValueError("goalkeeper joint guard kd is invalid")
        if not 0.0 <= self.joint_guard_impact_lead_sec <= 0.08:
            raise ValueError("goalkeeper impact guard lead is invalid")
        if self.block_action_timing_mode not in {"auto", "observation", "shooter_phase"}:
            raise ValueError("goalkeeper block timing mode is invalid")
        if self.actor_artifact_path is not None and (
            self.actor_observation_mode != "visible_ball_history_v3"
            or not self.actor_artifact_path.is_file()
        ):
            raise ValueError("goalkeeper actor artifact requires a readable visible-ball V3 policy")
        if (
            self.actor_bimanual_reach_enabled
            or self.actor_bimanual_punch_force_n > 0.0
            or self.actor_bimanual_support_arm_blend > 0.0
            or self.actor_bimanual_support_arm_overhead_bias_rad > 0.0
        ) and self.actor_artifact_path is None:
            raise ValueError("goalkeeper bimanual skills require a deployed actor artifact")
        if (
            self.actor_bimanual_support_arm_blend > 0.0
            or self.actor_bimanual_support_arm_overhead_bias_rad > 0.0
        ) and not self.actor_bimanual_reach_enabled:
            raise ValueError("goalkeeper support-arm posture requires bimanual reach")
        if (self.external_reference_manifest_path is None) != (
            self.external_reference_blend == 0.0
        ):
            raise ValueError(
                "goalkeeper external reference requires a path and positive blend together"
            )
        if self.external_reference_manifest_path is not None and (
            self.actor_artifact_path is not None
            or self.actor_observation_mode != "visible_ball_history_v3"
            or not self.external_reference_manifest_path.is_file()
            or not 0.0 < self.external_reference_blend <= 1.0
        ):
            raise ValueError("goalkeeper external reference is an isolated visible-ball V3 teacher")
        if (self.overhead_reach_prior_path is None) != (self.overhead_reach_blend == 0.0):
            raise ValueError(
                "goalkeeper overhead reach requires a prior path and positive blend together"
            )
        if self.overhead_reach_prior_path is not None and (
            self.actor_observation_mode != "visible_ball_history_v3"
            or not self.overhead_reach_prior_path.is_file()
            or not 0.0 < self.overhead_reach_blend <= 1.0
        ):
            raise ValueError("goalkeeper overhead reach requires a readable visible-ball V3 prior")
        if not (
            0.80
            <= self.overhead_reach_minimum_target_height_m
            < self.overhead_reach_full_target_height_m
            <= 1.60
        ):
            raise ValueError("goalkeeper overhead reach height curriculum is invalid")
        if not 0.0 <= self.overhead_reach_waist_scale <= 0.5:
            raise ValueError("goalkeeper overhead reach waist scale must be in [0, 0.5]")
        if not 0.0 <= self.overhead_reach_arm_scale <= 1.0:
            raise ValueError("goalkeeper overhead reach arm scale must be in [0, 1]")
        if (self.whole_body_reach_atlas_path is None) != (self.whole_body_reach_blend == 0.0):
            raise ValueError(
                "goalkeeper whole-body reach requires an atlas path and positive blend together"
            )
        if self.whole_body_reach_atlas_path is not None and (
            self.actor_observation_mode != "visible_ball_history_v3"
            or not self.whole_body_reach_atlas_path.is_file()
            or not 0.0 < self.whole_body_reach_blend <= 1.0
        ):
            raise ValueError(
                "goalkeeper whole-body reach requires a readable visible-ball V3 atlas"
            )
        if (
            self.overhead_reach_prior_path is not None
            and self.whole_body_reach_atlas_path is not None
            and self.overhead_reach_blend + self.whole_body_reach_blend > 1.0
        ):
            raise ValueError("goalkeeper fused aerial reach authority exceeds one")
        if not (
            0.80
            <= self.whole_body_reach_minimum_target_height_m
            < self.whole_body_reach_full_target_height_m
            <= 1.60
        ):
            raise ValueError("goalkeeper whole-body reach height curriculum is invalid")
        if not 0.0 <= self.whole_body_reach_timing_lead_sec <= 0.25:
            raise ValueError("goalkeeper whole-body reach timing lead must be in [0, 0.25] s")
        if not 0.0 <= self.whole_body_reach_waist_scale <= 1.0:
            raise ValueError("goalkeeper whole-body reach waist scale must be in [0, 1]")
        if not 0.0 <= self.whole_body_reach_target_arm_scale <= 2.0:
            raise ValueError("goalkeeper whole-body reach target-arm scale must be in [0, 2]")
        if not 0.0 <= self.whole_body_reach_support_arm_scale <= 1.0:
            raise ValueError("goalkeeper whole-body reach support-arm scale must be in [0, 1]")
        gmt_paths = (
            self.mosaic_gmt_model_path is not None and self.mosaic_gmt_skill_path is not None
        )
        if gmt_paths != (self.mosaic_gmt_blend > 0.0):
            raise ValueError("goalkeeper GMT requires model, skill, and positive blend together")
        if gmt_paths and (
            self.actor_observation_mode != "visible_ball_history_v3"
            or not self.mosaic_gmt_model_path.is_file()  # type: ignore[union-attr]
            or not self.mosaic_gmt_skill_path.is_file()  # type: ignore[union-attr]
            or not 0.0 < self.mosaic_gmt_blend <= 1.0
        ):
            raise ValueError("goalkeeper GMT requires readable visible-ball V3 artifacts")
        if gmt_paths and (
            self.overhead_reach_prior_path is not None
            or self.whole_body_reach_atlas_path is not None
        ):
            raise ValueError("goalkeeper GMT requires exclusive aerial teacher authority")
        if not (
            0.80
            <= self.mosaic_gmt_minimum_target_height_m
            < self.mosaic_gmt_full_target_height_m
            <= 1.60
        ):
            raise ValueError("goalkeeper GMT height curriculum is invalid")
        if not 0.10 <= self.mosaic_gmt_maximum_lateral_error_m <= 1.50:
            raise ValueError("goalkeeper GMT lateral router is invalid")
        if not 0.0 <= self.mosaic_gmt_timing_lead_sec <= 0.25:
            raise ValueError("goalkeeper GMT timing lead must be in [0, 0.25] s")
        if not isinstance(self.mosaic_gmt_mirror_by_intercept, bool):
            raise ValueError("goalkeeper GMT lateral mirror flag must be boolean")
        for value, label in (
            (self.mosaic_gmt_lower_body_scale, "lower-body"),
            (self.mosaic_gmt_waist_scale, "waist"),
            (self.mosaic_gmt_arm_scale, "arm"),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"goalkeeper GMT {label} scale must be in [0, 1]")
        if (self.balanced_dive_source_checkout is None) != (self.balanced_dive_blend == 0.0):
            raise ValueError("goalkeeper balanced dive requires a source and positive blend")
        if self.balanced_dive_source_checkout is not None and (
            not self.balanced_dive_source_checkout.is_dir()
            or not 0.0 < self.balanced_dive_blend <= 1.0
            or not gmt_paths
        ):
            raise ValueError("goalkeeper balanced dive requires a GMT-routed readable source")
        if not 0.20 <= self.balanced_dive_minimum_lateral_error_m <= 0.80:
            raise ValueError("goalkeeper balanced dive lateral gate is invalid")
        if not 0.0 <= self.balanced_dive_activation_lead_sec <= 1.0:
            raise ValueError("goalkeeper balanced dive activation lead is invalid")
        if not 0.0 <= self.balanced_dive_initial_phase <= 0.30:
            raise ValueError("goalkeeper balanced dive initial phase is invalid")
        if not 0.40 <= self.balanced_dive_phase_at_arrival <= 0.85:
            raise ValueError("goalkeeper balanced dive arrival phase is invalid")
        if self.balanced_dive_initial_phase >= self.balanced_dive_phase_at_arrival:
            raise ValueError("goalkeeper balanced dive initial phase must precede arrival")
        if not self.balanced_dive_phase_at_arrival <= self.balanced_dive_peak_phase <= 1.0:
            raise ValueError("goalkeeper balanced dive peak phase is invalid")
        if not all(
            0.0 <= value <= 1.0
            for value in (
                self.balanced_dive_lower_body_scale,
                self.balanced_dive_waist_scale,
                self.balanced_dive_arm_scale,
            )
        ):
            raise ValueError("goalkeeper balanced dive joint-group scale is invalid")
        if self.balanced_dive_source_checkout is not None and (
            max(
                self.balanced_dive_lower_body_scale,
                self.balanced_dive_waist_scale,
                self.balanced_dive_arm_scale,
            )
            <= 0.0
        ):
            raise ValueError("goalkeeper balanced dive requires non-zero joint authority")
        if not 0.10 <= self.balanced_dive_blend_in_sec <= 0.50:
            raise ValueError("goalkeeper balanced dive blend-in is invalid")
        if not 0.20 <= self.balanced_dive_recovery_tail_sec <= 0.80:
            raise ValueError("goalkeeper balanced dive recovery tail is invalid")
        if not isinstance(self.balanced_dive_landing_capture_enabled, bool):
            raise ValueError("goalkeeper landing capture flag must be boolean")
        if not 0.20 <= self.balanced_dive_landing_capture_sec <= 1.50:
            raise ValueError("goalkeeper landing capture duration is invalid")
        if not 1.0 <= self.balanced_dive_landing_damping_scale <= 3.0:
            raise ValueError("goalkeeper landing capture damping is invalid")
        if (
            self.balanced_dive_landing_capture_enabled
            and self.balanced_dive_source_checkout is None
        ):
            raise ValueError("goalkeeper landing capture requires a balanced dive source")
        if not isinstance(self.post_contact_proprioceptive_capture_enabled, bool):
            raise ValueError("goalkeeper proprioceptive capture flag must be boolean")
        if not 0.0 <= self.post_contact_proprioceptive_capture_delay_sec <= 1.20:
            raise ValueError("goalkeeper proprioceptive capture delay is invalid")
        if not 0.20 <= self.post_contact_proprioceptive_capture_window_sec <= 1.50:
            raise ValueError("goalkeeper proprioceptive capture window is invalid")
        if (
            self.post_contact_proprioceptive_capture_delay_sec
            >= self.post_contact_proprioceptive_capture_window_sec
        ):
            raise ValueError("goalkeeper proprioceptive capture window is empty")
        if not 0.05 <= self.post_contact_proprioceptive_capture_maximum_root_speed_mps <= 1.50:
            raise ValueError("goalkeeper proprioceptive capture speed gate is invalid")
        if not 0.20 <= self.post_contact_proprioceptive_capture_duration_sec <= 1.50:
            raise ValueError("goalkeeper proprioceptive capture duration is invalid")
        if self.post_contact_proprioceptive_capture_enabled and not (
            self.balanced_dive_landing_capture_enabled and self.post_contact_stabilization_enabled
        ):
            raise ValueError(
                "goalkeeper proprioceptive capture requires landing and contact stabilization"
            )
        dive_athlete_paths = (
            self.dive_athlete_checkpoint_path,
            self.dive_athlete_exam_path,
        )
        if not 0.0 <= self.dive_athlete_blend <= 1.0:
            raise ValueError("goalkeeper dive athlete blend is invalid")
        if (any(path is not None for path in dive_athlete_paths)) != (
            self.dive_athlete_blend > 0.0
        ) or (
            self.dive_athlete_blend > 0.0
            and (
                not all(path is not None and path.is_file() for path in dive_athlete_paths)
                or self.balanced_dive_source_checkout is None
            )
        ):
            raise ValueError(
                "goalkeeper dive athlete requires checkpoint, passing exam and dive source"
            )
        if not isinstance(self.post_contact_ready_recovery_enabled, bool):
            raise ValueError("goalkeeper post-contact ready recovery flag must be boolean")
        if self.post_contact_ready_recovery_enabled and not self.post_contact_stabilization_enabled:
            raise ValueError(
                "goalkeeper post-contact ready recovery requires contact stabilization"
            )
        if not 2.0 <= self.post_contact_ready_recovery_delay_sec <= 5.5:
            raise ValueError("goalkeeper post-contact ready recovery delay is invalid")
        if not 0.1 <= self.post_contact_ready_yaw_gain <= 4.0:
            raise ValueError("goalkeeper post-contact ready yaw gain is invalid")
        if not 0.02 <= self.post_contact_ready_maximum_yaw_rate_rad_s <= 0.8:
            raise ValueError("goalkeeper post-contact ready yaw rate is invalid")
        if not 0.1 <= self.post_contact_ready_lateral_position_gain <= 2.0:
            raise ValueError("goalkeeper post-contact ready lateral gain is invalid")
        if (
            not 0.05 <= self.post_contact_ready_maximum_lateral_speed_mps <= 0.20
            or self.post_contact_ready_maximum_lateral_speed_mps > self.maximum_lateral_speed_mps
        ):
            raise ValueError("goalkeeper post-contact ready lateral speed is invalid")
        if not 0.02 <= self.post_contact_ready_lateral_deadband_m <= 0.20:
            raise ValueError("goalkeeper post-contact ready lateral deadband is invalid")
        recovery_athlete_paths = (
            self.recovery_athlete_checkpoint_path,
            self.recovery_athlete_exam_path,
        )
        if not 0.0 <= self.recovery_athlete_blend <= 1.0:
            raise ValueError("goalkeeper recovery athlete blend is invalid")
        if not isinstance(self.recovery_athlete_authority_envelope_enabled, bool):
            raise ValueError("goalkeeper recovery athlete authority envelope flag is invalid")
        if (any(path is not None for path in recovery_athlete_paths)) != (
            self.recovery_athlete_blend > 0.0
        ) or (
            self.recovery_athlete_blend > 0.0
            and (
                not all(path is not None and path.is_file() for path in recovery_athlete_paths)
                or not self.post_contact_stabilization_enabled
                or not self.post_contact_ready_recovery_enabled
            )
        ):
            raise ValueError(
                "goalkeeper recovery athlete requires checkpoint, exam and ready recovery"
            )
        if self.recovery_athlete_authority_envelope_enabled and self.recovery_athlete_blend <= 0.0:
            raise ValueError("goalkeeper recovery athlete authority envelope requires an actor")
        if not isinstance(self.successor_lateral_probe_enabled, bool):
            raise ValueError("goalkeeper successor lateral probe flag must be boolean")
        if not 2.0 <= self.successor_lateral_probe_delay_sec <= 10.0:
            raise ValueError("goalkeeper successor lateral probe delay is invalid")
        if not 0.4 <= self.successor_lateral_probe_duration_sec <= 1.5:
            raise ValueError("goalkeeper successor lateral probe duration is invalid")
        probe_command_configured = abs(self.successor_lateral_probe_command_mps) > 1.0e-12
        probe_command_active = abs(self.successor_lateral_probe_command_mps) >= 0.05
        if (
            self.successor_lateral_probe_enabled != probe_command_configured
            or (self.successor_lateral_probe_enabled and not probe_command_active)
            or abs(self.successor_lateral_probe_command_mps) > self.maximum_lateral_speed_mps
            or (
                self.successor_lateral_probe_enabled
                and (
                    not self.post_contact_stabilization_enabled
                    or not self.post_contact_ready_recovery_enabled
                    or self.successor_lateral_probe_delay_sec
                    < self.post_contact_ready_recovery_delay_sec + 0.5
                )
            )
        ):
            raise ValueError(
                "goalkeeper successor probe requires bounded post-contact ready recovery"
            )
        if not 180 <= self.anticipation_start_policy_frame <= 270:
            raise ValueError("goalkeeper anticipation frame must be in [180, 270]")
        if not (
            0.20
            <= self.anticipation_minimum_foot_ball_distance_m
            < self.anticipation_maximum_foot_ball_distance_m
            <= 4.0
        ):
            raise ValueError("goalkeeper anticipation distance window is invalid")
        if not 0.0 <= self.anticipation_target_blend <= 1.0:
            raise ValueError("goalkeeper anticipation blend must be in [0, 1]")
        if not 0.25 <= self.anticipation_velocity_scale <= 1.0:
            raise ValueError("goalkeeper anticipation velocity scale must be in [0.25, 1]")
        if not 180 <= self.block_action_start_policy_frame <= 270:
            raise ValueError("goalkeeper block-action frame must be in [180, 270]")
        if not 4 <= self.block_action_blend_frames <= 40:
            raise ValueError("goalkeeper block-action blend must be in [4, 40] frames")
        if not 10 <= self.block_action_hold_frames <= 100:
            raise ValueError("goalkeeper block-action hold must be in [10, 100] frames")
        bounds = (
            (self.block_action_waist_yaw_rad, 0.70, "waist yaw"),
            (self.block_action_waist_roll_rad, 0.35, "waist roll"),
            (self.block_action_waist_pitch_rad, 0.35, "waist pitch"),
            (self.block_action_hip_pitch_rad, 0.50, "hip pitch"),
            (self.block_action_hip_roll_rad, 0.35, "hip roll"),
            (self.block_action_knee_flex_rad, 0.70, "knee flex"),
            (self.block_action_shoulder_pitch_rad, 1.00, "shoulder pitch"),
            (self.block_action_shoulder_roll_rad, 0.80, "shoulder roll"),
            (self.block_action_elbow_flex_rad, 0.60, "elbow flex"),
        )
        for value, limit, label in bounds:
            if abs(value) > limit:
                raise ValueError(f"goalkeeper block-action {label} exceeds its bound")


@dataclass(frozen=True)
class G1SecondThreatConfig:
    """SIM-only live-ball launcher for a continuous second-threat curriculum."""

    launch_time_sec: float = 17.0
    rearm_lead_sec: float = 0.60
    force_duration_sec: float = 0.08
    flight_time_sec: float = 0.82
    target_depth_before_goal_m: float = 0.72
    target_y_m: float = 0.45
    target_z_m: float = 1.10
    goalkeeper_punch_force_n: float = 85.0
    goalkeeper_punch_outward_force_scale: float = 0.75
    maximum_force_n: float = 80.0
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.g1_second_threat_config.v1"

    def __post_init__(self) -> None:
        values = tuple(
            value
            for value in asdict(self).values()
            if isinstance(value, int | float) and not isinstance(value, bool)
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("second-threat settings must be finite")
        if not 14.0 <= self.launch_time_sec <= 21.0:
            raise ValueError("second-threat launch time is invalid")
        if not 0.30 <= self.rearm_lead_sec <= 1.20:
            raise ValueError("second-threat rearm lead is invalid")
        if not 0.04 <= self.force_duration_sec <= 0.15:
            raise ValueError("second-threat force duration is invalid")
        if not 0.55 <= self.flight_time_sec <= 1.20:
            raise ValueError("second-threat flight time is invalid")
        if not 0.50 <= self.target_depth_before_goal_m <= 1.00:
            raise ValueError("second-threat target depth is invalid")
        if not -2.5 <= self.target_y_m <= 2.5 or not 1.10 <= self.target_z_m <= 1.80:
            raise ValueError("second-threat target is invalid")
        if not 20.0 <= self.goalkeeper_punch_force_n <= 120.0:
            raise ValueError("second-threat punch force is invalid")
        if not 0.0 <= self.goalkeeper_punch_outward_force_scale <= 0.75:
            raise ValueError("second-threat outward punch scale is invalid")
        if not 40.0 <= self.maximum_force_n <= 120.0:
            raise ValueError("second-threat force limit is invalid")
        if self.activation_ceiling != "SIM_ONLY" or self.hardware_authorized:
            raise ValueError("second-threat curriculum must remain SIM_ONLY")


@dataclass(frozen=True)
class G1PhysicalSecondStrikerConfig:
    """Fourth-G1/second-football provider for a no-reset second threat.

    Both the robot and football exist from model compilation onward.  The
    fourth G1 starts its frozen RoboNaldo policy late enough for the first
    goalkeeper save to finish, while its exact foot contact—not a timer—owns
    the causal handoff to the goalkeeper observer.
    """

    policy_start_time_sec: float = 11.49
    observer_rearm_earliest_sec: float = 16.0
    origin_m: tuple[float, float, float] = (0.50, 0.30, 0.0)
    ball_origin_m: tuple[float, float, float] = (1.785, 0.282, 0.115)
    # ROSClaw exposes each robot's sensed world state in its attached actor
    # frame.  This is therefore the inverse-calibrated FreeKick target in that
    # same policy frame, not the physical scoring target at the goal plane.
    policy_target_m: tuple[float, float, float] = (7.50, 0.70, 1.20)
    # A target-conditioned contact actor needs the physical second-shot aim,
    # not the first-shot marker.  The complete successor is a high-glove save;
    # feeding the pitch-level ball-radius marker creates a contradictory low
    # shot objective.
    ballistic_target_depth_before_goal_m: float = 0.83
    ballistic_target_y_m: float = 0.19
    ballistic_target_z_m: float = 1.44
    kick_foot: str = "right"
    swing_amplitude: float = 1.0
    swing_speed_scale: float = 0.90
    foot_yaw_offset: float = 0.01
    foot_pitch_offset: float = 0.10
    loft_synergy: float = 0.10
    contact_phase_offset: float = 0.0
    # Preserve the S109 role-shared actor envelope as the bilateral control.
    # Individual frontier cases may tighten or widen it within the validated
    # [0.15, 0.30] m range.
    ballistic_actor_proximity_m: float = 0.25
    goalkeeper_punch_force_n: float = 90.0
    goalkeeper_punch_outward_force_scale: float = 0.0
    post_policy_frame: int = 275
    post_policy_blend_frames: int = 0
    minimum_contact_force_n: float = 20.0
    maximum_contact_force_n: float = 1500.0
    minimum_post_contact_speed_gain_mps: float = 4.0
    minimum_forward_ball_speed_mps: float = 3.0
    minimum_pelvis_height_m: float = 0.60
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.g1_physical_second_striker_config.v3"

    def __post_init__(self) -> None:
        values = (
            self.policy_start_time_sec,
            self.observer_rearm_earliest_sec,
            *self.origin_m,
            *self.ball_origin_m,
            *self.policy_target_m,
            self.ballistic_target_depth_before_goal_m,
            self.ballistic_target_y_m,
            self.ballistic_target_z_m,
            self.swing_amplitude,
            self.swing_speed_scale,
            self.foot_yaw_offset,
            self.foot_pitch_offset,
            self.loft_synergy,
            self.contact_phase_offset,
            self.ballistic_actor_proximity_m,
            self.goalkeeper_punch_force_n,
            self.goalkeeper_punch_outward_force_scale,
            self.minimum_contact_force_n,
            self.maximum_contact_force_n,
            self.minimum_post_contact_speed_gain_mps,
            self.minimum_forward_ball_speed_mps,
            self.minimum_pelvis_height_m,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("physical second-striker settings must be finite")
        if not 9.0 <= self.policy_start_time_sec <= 15.0:
            raise ValueError("physical second-striker start time is invalid")
        if not 14.0 <= self.observer_rearm_earliest_sec <= 20.0:
            raise ValueError("physical second-striker observer rearm time is invalid")
        if self.observer_rearm_earliest_sec <= self.policy_start_time_sec:
            raise ValueError("physical second-striker rearm must follow policy start")
        local_ball = np.asarray(self.ball_origin_m) - np.asarray(self.origin_m)
        if not (
            1.15 <= local_ball[0] <= 1.40
            and -0.25 <= local_ball[1] <= 0.15
            and 0.105 <= local_ball[2] <= 0.130
        ):
            raise ValueError("physical second football is outside the qualified strike pocket")
        if not 4.0 <= self.policy_target_m[0] <= 12.0 or not (
            -3.2 <= self.policy_target_m[1] <= 3.2 and 0.115 <= self.policy_target_m[2] <= 1.2
        ):
            raise ValueError("physical second-striker policy target is invalid")
        if not 0.30 <= self.ballistic_target_depth_before_goal_m <= 1.20 or not (
            -3.5 <= self.ballistic_target_y_m <= 3.5 and 1.10 <= self.ballistic_target_z_m <= 1.80
        ):
            raise ValueError("physical second-striker ballistic target is invalid")
        if self.kick_foot not in {"left", "right"}:
            raise ValueError("physical second-striker kick foot must be left or right")
        if not 0.15 <= self.ballistic_actor_proximity_m <= 0.30:
            raise ValueError("physical second-striker actor proximity is invalid")
        ShotParameters(
            kick_foot=self.kick_foot,
            swing_amplitude=self.swing_amplitude,
            swing_speed_scale=self.swing_speed_scale,
            foot_yaw_offset=self.foot_yaw_offset,
            foot_pitch_offset=self.foot_pitch_offset,
            loft_synergy=self.loft_synergy,
            contact_phase_offset=self.contact_phase_offset,
        )
        if not 20.0 <= self.goalkeeper_punch_force_n <= 120.0:
            raise ValueError("physical second-striker goalkeeper punch force is invalid")
        if not 0.0 <= self.goalkeeper_punch_outward_force_scale <= 0.75:
            raise ValueError("physical second-striker goalkeeper punch scale is invalid")
        if not 220 <= self.post_policy_frame <= 340:
            raise ValueError("physical second-striker post-policy frame is invalid")
        if not 0 <= self.post_policy_blend_frames <= 40:
            raise ValueError("physical second-striker handoff blend is invalid")
        if not 5.0 <= self.minimum_contact_force_n <= 100.0 or not (
            200.0 <= self.maximum_contact_force_n <= 3000.0
        ):
            raise ValueError("physical second-striker contact-force gate is invalid")
        if self.maximum_contact_force_n <= self.minimum_contact_force_n:
            raise ValueError("physical second-striker force interval is empty")
        if not 2.0 <= self.minimum_post_contact_speed_gain_mps <= 8.0:
            raise ValueError("physical second-striker speed-gain gate is invalid")
        if not 2.0 <= self.minimum_forward_ball_speed_mps <= 8.0:
            raise ValueError("physical second-striker forward-speed gate is invalid")
        if not 0.50 <= self.minimum_pelvis_height_m <= 0.72:
            raise ValueError("physical second-striker pelvis-height gate is invalid")
        if self.activation_ceiling != "SIM_ONLY" or self.hardware_authorized:
            raise ValueError("physical second-striker provider must remain SIM_ONLY")


def shared_post_impact_simulation_kwargs() -> dict[str, Any]:
    recovery = shared_post_impact_recovery_config()
    guard = G1JointGuardConfig()
    return {
        "passer_recovery_config": recovery,
        "passer_post_policy_frame": 265,
        "passer_post_policy_blend_frames": 2,
        "passer_joint_guard_enabled": True,
        "passer_post_policy_neutral_velocity_enabled": True,
        "passer_joint_guard_config": guard,
        "passer_post_policy_recovery_enabled": True,
        "shooter_recovery_config": recovery,
        "shooter_post_policy_frame": 275,
        "shooter_post_policy_blend_frames": 0,
        "shooter_joint_guard_enabled": True,
        "shooter_post_policy_neutral_velocity_enabled": True,
        "shooter_joint_guard_config": guard,
        "shooter_post_policy_recovery_enabled": True,
    }


def trained_coupled_skill_simulation_kwargs() -> dict[str, Any]:
    values = shared_post_impact_simulation_kwargs()
    values.update(
        {
            "shooter_parameter_overrides": {
                "foot_yaw_offset": 0.085,
                "foot_pitch_offset": 0.010,
            },
            "shooter_early_arrival_parameter_overrides": {
                "foot_yaw_offset": 0.115,
                "foot_pitch_offset": 0.025,
            },
        }
    )
    return values


def trained_three_role_skill_simulation_kwargs() -> dict[str, Any]:
    """Return the retained safe finisher plus causal goalkeeper anticipation."""

    values = trained_coupled_skill_simulation_kwargs()
    values.update(
        {
            "shooter_start_sec": 2.19,
            "shooter_joint_guard_config": G1JointGuardConfig(
                margin_rad=0.10,
                prediction_horizon_sec=0.20,
                boundary_kp=180.0,
                boundary_kd=18.0,
            ),
            "goalkeeper_config": G1GoalkeeperConfig(
                reaction_delay_sec=0.08,
                lateral_position_gain=2.5,
                maximum_lateral_speed_mps=0.40,
                arm_spread_rad=0.28,
                maximum_waist_lean_rad=0.08,
                anticipation_enabled=True,
            ),
            "unified_stadium_scene": True,
        }
    )
    return values


@dataclass(frozen=True)
class G1SharedWorldResult:
    finite_state: bool
    pass_contact_observed: bool
    shot_contact_observed: bool
    pass_contact_time_sec: float | None
    shot_contact_time_sec: float | None
    pass_peak_ball_speed_mps: float
    shot_peak_ball_speed_mps: float
    goal_crossed: bool
    goal_plane_crossed: bool
    goal_crossing_y_m: float | None
    goal_crossing_z_m: float | None
    target_error_m: float | None
    passer_min_pelvis_height_m: float
    shooter_min_pelvis_height_m: float
    passer_roll_peak_rad: float
    passer_pitch_peak_rad: float
    shooter_roll_peak_rad: float
    shooter_pitch_peak_rad: float
    passer_tail_wobble_index: float
    shooter_tail_wobble_index: float
    receiver_phase_hold_frames: int
    receiver_phase_advance_frames: int
    receiver_max_ball_phase_error_m: float
    robot_robot_contact_count: int
    joint_limit_violation: bool
    torque_limit_violation: bool
    actuator_saturation: bool
    physics_steps: int
    passer_support_foot_slip_m: float = 0.0
    shooter_support_foot_slip_m: float = 0.0
    passer_post_contact_support_foot_slip_m: float = 0.0
    shooter_post_contact_support_foot_slip_m: float = 0.0
    passer_contact_impulse_ns: float = 0.0
    shooter_contact_impulse_ns: float = 0.0
    passer_post_kick_fall: bool = False
    shooter_post_kick_fall: bool = False
    shooter_learned_torque_fraction: float = 0.0
    shooter_learned_torque_fallback_fraction: float = 0.0
    shooter_learned_torque_mean_confidence: float = 0.0
    shooter_learned_torque_peak_residual_nm: float = 0.0
    shooter_learned_torque_support_rms_peak: float = 0.0
    shooter_joint_guard_fraction: float = 0.0
    shooter_joint_guard_route: str = "disabled"
    pass_reception_target_m: tuple[float, float, float] = (1.00, 0.0, 0.115)
    pass_delivery_position_m: tuple[float, float, float] | None = None
    pass_delivery_error_m: float | None = None
    pass_delivery_lateral_error_m: float | None = None
    passer_joint_guard_fraction: float = 0.0
    passer_joint_guard_route: str = "disabled"
    passer_recovery_active_fraction: float = 0.0
    shooter_recovery_active_fraction: float = 0.0
    passer_recovery_peak_blend_fraction: float = 0.0
    shooter_recovery_peak_blend_fraction: float = 0.0
    shooter_transition_actor_hash: str | None = None
    shooter_transition_actor_accepted: bool = False
    shooter_transition_triggered: bool = False
    shooter_transition_trigger_time_sec: float | None = None
    shooter_transition_trigger_policy_frame: int | None = None
    shooter_transition_residual_frames: int = 0
    shooter_transition_support_distance: float | None = None
    shooter_transition_predicted_safe_probability: float | None = None
    shooter_transition_predicted_chain_probability: float | None = None
    shooter_transition_ensemble_probability_spread: float | None = None
    shooter_transition_used_parent_fallback: bool = False
    shooter_causal_strike_option_enabled: bool = False
    shooter_causal_strike_option_config_hash: str | None = None
    shooter_causal_strike_option_final_phase: str | None = None
    shooter_causal_strike_option_reason: str | None = None
    shooter_causal_strike_bridge_started: bool = False
    shooter_causal_strike_bridge_start_time_sec: float | None = None
    shooter_causal_strike_bridge_predecessor_policy_frame: int | None = None
    shooter_causal_strike_selected_phase_start_frame: int | None = None
    shooter_causal_strike_bridge_peak_target_velocity_rms_rad_s: float = 0.0
    shooter_causal_strike_abort_recovery_activated: bool = False
    shooter_runtime_strike_router_hash: str | None = None
    shooter_runtime_strike_route_decided: bool = False
    shooter_runtime_strike_route_accepted: bool = False
    shooter_runtime_strike_route: str | None = None
    shooter_runtime_strike_route_time_sec: float | None = None
    shooter_runtime_strike_route_support_distance: float | None = None
    shooter_runtime_strike_route_advance_frames: int | None = None
    shooter_runtime_receive_actor_hash: str | None = None
    shooter_runtime_receive_decided: bool = False
    shooter_runtime_receive_accepted: bool = False
    shooter_runtime_receive_route: str | None = None
    shooter_runtime_receive_time_sec: float | None = None
    shooter_runtime_receive_support_distance: float | None = None
    shooter_runtime_receive_alignment_tolerance_sec: float | None = None
    shooter_runtime_receive_stance_offset_y_m: float | None = None
    shooter_runtime_receive_foot_yaw_offset_rad: float | None = None
    shooter_runtime_contact_target_actor_hash: str | None = None
    shooter_runtime_contact_target_decided: bool = False
    shooter_runtime_contact_target_accepted: bool = False
    shooter_runtime_contact_target_route: str | None = None
    shooter_runtime_contact_target_time_sec: float | None = None
    shooter_runtime_contact_target_support_distance: float | None = None
    shooter_runtime_contact_target_velocity_xyz_mps: tuple[float, float, float] | None = None
    shooter_runtime_finish_plan_actor_hash: str | None = None
    shooter_runtime_finish_plan_decided: bool = False
    shooter_runtime_finish_plan_accepted: bool = False
    shooter_runtime_finish_plan_route: str | None = None
    shooter_runtime_finish_plan_time_sec: float | None = None
    shooter_runtime_finish_plan_support_distance: float | None = None
    shooter_aim_expert_route: str = "nominal"
    shooter_early_arrival_expert_fraction: float = 0.0
    shooter_ballistic_actor_active_fraction: float = 0.0
    shooter_ballistic_actor_peak_torque_nm: float = 0.0
    shooter_ballistic_actor_hash: str | None = None
    shooter_motion_prior_hash: str | None = None
    shooter_motion_prior_position_active_fraction: float = 0.0
    shooter_motion_prior_velocity_active_fraction: float = 0.0
    shooter_motion_prior_strike_leg_scale: float = 1.0
    shooter_motion_prior_joint_scales: tuple[float, ...] = (1.0,) * 29
    shooter_motion_prior_velocity_joint_scales: tuple[float, ...] = (1.0,) * 29
    shooter_motion_prior_peak_target_delta_rad: float = 0.0
    shooter_motion_prior_peak_velocity_delta_rad_s: float = 0.0
    shooter_agility_prior_hash: str | None = None
    shooter_agility_prior_active_fraction: float = 0.0
    shooter_agility_prior_peak_target_delta_rad: float = 0.0
    shooter_agility_prior_peak_velocity_delta_rad_s: float = 0.0
    shooter_agility_prior_joint_scales: tuple[float, ...] = (0.0,) * 12 + (1.0,) * 17
    shooter_contact_prior_hash: str | None = None
    shooter_contact_prior_active_fraction: float = 0.0
    shooter_contact_prior_peak_target_delta_rad: float = 0.0
    shooter_contact_prior_peak_velocity_delta_rad_s: float = 0.0
    shooter_contact_prior_joint_scales: tuple[float, ...] = (1.0,) * 6
    goalkeeper_enabled: bool = False
    goalkeeper_reaction_active_fraction: float = 0.0
    goalkeeper_anticipation_active_fraction: float = 0.0
    goalkeeper_canonical_locomotion_mirror_active_fraction: float = 0.0
    goalkeeper_block_action_active_fraction: float = 0.0
    goalkeeper_lateral_displacement_m: float = 0.0
    goalkeeper_peak_lateral_speed_mps: float = 0.0
    goalkeeper_min_pelvis_height_m: float | None = None
    goalkeeper_ball_contact_observed: bool = False
    goalkeeper_ball_contact_time_sec: float | None = None
    goalkeeper_save_observed: bool = False
    goalkeeper_left_glove_contact_observed: bool = False
    goalkeeper_right_glove_contact_observed: bool = False
    goalkeeper_glove_contact_height_m: float | None = None
    goalkeeper_glove_contact_time_sec: float | None = None
    goalkeeper_glove_contact_position_m: tuple[float, float, float] | None = None
    goalkeeper_glove_contact_surface_distance_m: float | None = None
    goalkeeper_glove_contact_side: str | None = None
    goalkeeper_contact_left_hand_height_m: float | None = None
    goalkeeper_contact_right_hand_height_m: float | None = None
    goalkeeper_contact_left_hand_ball_distance_m: float | None = None
    goalkeeper_contact_right_hand_ball_distance_m: float | None = None
    goalkeeper_both_hands_raised_at_contact: bool = False
    goalkeeper_bimanual_window_observed: bool = False
    goalkeeper_bimanual_reach_active_fraction: float = 0.0
    goalkeeper_bimanual_punch_active_fraction: float = 0.0
    goalkeeper_bimanual_punch_peak_torque_nm: float = 0.0
    goalkeeper_reach_memory_peak_rad: float = 0.0
    goalkeeper_overhead_reach_prior_hash: str | None = None
    goalkeeper_overhead_reach_active_fraction: float = 0.0
    goalkeeper_overhead_reach_peak_blend: float = 0.0
    goalkeeper_whole_body_reach_atlas_hash: str | None = None
    goalkeeper_whole_body_reach_active_fraction: float = 0.0
    goalkeeper_whole_body_reach_peak_blend: float = 0.0
    goalkeeper_mosaic_gmt_skill_hash: str | None = None
    goalkeeper_mosaic_gmt_contract_hash: str | None = None
    goalkeeper_mosaic_gmt_active_fraction: float = 0.0
    goalkeeper_mosaic_gmt_peak_blend: float = 0.0
    goalkeeper_balanced_dive_seed_hash: str | None = None
    goalkeeper_balanced_dive_active_fraction: float = 0.0
    goalkeeper_balanced_dive_peak_blend: float = 0.0
    goalkeeper_dive_athlete_checkpoint_hash: str | None = None
    goalkeeper_dive_athlete_blend: float = 0.0
    goalkeeper_recovery_athlete_checkpoint_hash: str | None = None
    goalkeeper_recovery_athlete_blend: float = 0.0
    goalkeeper_recovery_athlete_authority_envelope_enabled: bool = False
    goalkeeper_recovery_athlete_active_fraction: float = 0.0
    goalkeeper_recovery_athlete_suppressed_fraction: float = 0.0
    second_threat_enabled: bool = False
    second_threat_rearmed: bool = False
    second_threat_rearm_time_sec: float | None = None
    second_threat_launch_observed: bool = False
    second_threat_launch_time_sec: float | None = None
    second_threat_launch_position_m: tuple[float, float, float] | None = None
    second_threat_target_velocity_mps: tuple[float, float, float] | None = None
    second_threat_peak_force_n: float = 0.0
    goalkeeper_second_ball_contact_observed: bool = False
    goalkeeper_second_ball_contact_time_sec: float | None = None
    goalkeeper_second_glove_contact_observed: bool = False
    goalkeeper_second_glove_contact_time_sec: float | None = None
    goalkeeper_second_glove_contact_height_m: float | None = None
    goalkeeper_second_glove_contact_position_m: tuple[float, float, float] | None = None
    goalkeeper_second_glove_contact_surface_distance_m: float | None = None
    goalkeeper_second_glove_contact_side: str | None = None
    goalkeeper_second_contact_left_hand_height_m: float | None = None
    goalkeeper_second_contact_right_hand_height_m: float | None = None
    goalkeeper_second_contact_left_hand_ball_distance_m: float | None = None
    goalkeeper_second_contact_right_hand_ball_distance_m: float | None = None
    goalkeeper_second_save_observed: bool = False
    physical_second_striker_enabled: bool = False
    second_striker_ball_existed_from_time_zero: bool = False
    second_striker_contact_observed: bool = False
    second_striker_contact_time_sec: float | None = None
    second_striker_contact_foot: str | None = None
    second_striker_contact_force_peak_n: float = 0.0
    second_striker_precontact_peak_ball_speed_mps: float = 0.0
    second_striker_postcontact_peak_ball_speed_mps: float = 0.0
    second_striker_postcontact_peak_forward_ball_speed_mps: float = 0.0
    second_striker_min_pelvis_height_m: float | None = None
    second_striker_ballistic_actor_active_fraction: float = 0.0
    second_striker_ballistic_actor_peak_torque_nm: float = 0.0
    second_striker_unexpected_precontact_collision_geoms: tuple[str, ...] = ()
    second_striker_joint_limit_violation: bool = False
    second_ball_goal_plane_crossed: bool = False
    second_ball_goal_crossed: bool = False
    second_ball_goal_crossing_y_m: float | None = None
    second_ball_goal_crossing_z_m: float | None = None
    passer_joint_limit_violation: bool = False
    shooter_joint_limit_violation: bool = False
    goalkeeper_joint_limit_violation: bool = False
    schema_version: str = "rosclaw_soccer.g1_shared_world_result.v26"

    @property
    def pass_precision_passed(self) -> bool:
        return bool(
            self.pass_delivery_error_m is not None
            and self.pass_delivery_error_m <= 0.05
            and self.pass_delivery_lateral_error_m is not None
            and self.pass_delivery_lateral_error_m <= 0.03
        )

    @property
    def passed(self) -> bool:
        return bool(
            self.finite_state
            and self.pass_contact_observed
            and self.shot_contact_observed
            and self.pass_contact_time_sec is not None
            and self.shot_contact_time_sec is not None
            and self.pass_contact_time_sec < self.shot_contact_time_sec
            and self.pass_peak_ball_speed_mps >= 0.55
            and self.pass_precision_passed
            and self.shot_peak_ball_speed_mps >= 6.0
            and self.goal_crossed
            and self.target_error_m is not None
            and self.target_error_m <= 0.10
            and self.passer_min_pelvis_height_m >= 0.55
            and self.shooter_min_pelvis_height_m >= 0.55
            and self.passer_roll_peak_rad <= 0.55
            and self.passer_pitch_peak_rad <= 0.65
            and self.shooter_roll_peak_rad <= 0.55
            and self.shooter_pitch_peak_rad <= 0.65
            and not self.joint_limit_violation
            and not self.torque_limit_violation
            and not self.actuator_saturation
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["passed"] = self.passed
        value["pass_precision_passed"] = self.pass_precision_passed
        return value


@dataclass
class _Robot:
    role: str
    prefix: str
    origin: np.ndarray
    world_from_local_quat: np.ndarray
    qpos_base: int
    qvel_base: int
    joint_qpos: np.ndarray
    joint_qvel: np.ndarray
    actuators: np.ndarray
    joint_ids: np.ndarray
    pelvis_body: int
    torso_body: int
    left_hand_body: int
    right_hand_body: int
    left_ankle_body: int
    right_ankle_body: int
    state: Any
    output: Any
    policy: Any
    standby_output: Any | None
    standby_policy: Any | None
    parameters: ShotParameters
    start_sec: float
    hold_target: np.ndarray
    last_target: np.ndarray
    target_velocity: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    motion_prior: G1FootballMotionPrior | None = None
    motion_prior_position_blend: float = 0.0
    motion_prior_velocity_blend: float = 0.0
    motion_prior_strike_leg_scale: float = 1.0
    motion_prior_joint_scales: tuple[float, ...] = (1.0,) * 29
    motion_prior_velocity_joint_scales: tuple[float, ...] = (1.0,) * 29
    motion_prior_contact_policy_frame: int = 253
    motion_prior_position_active_frame_count: int = 0
    motion_prior_velocity_active_frame_count: int = 0
    motion_prior_peak_target_delta_rad: float = 0.0
    motion_prior_peak_velocity_delta_rad_s: float = 0.0
    last_motion_prior_target_delta: np.ndarray | None = None
    last_motion_prior_velocity_delta: np.ndarray | None = None
    last_motion_prior_position_active: bool = False
    last_motion_prior_velocity_active: bool = False
    agility_prior: G1MosaicAgilityPrior | None = None
    agility_prior_position_blend: float = 0.0
    agility_prior_velocity_blend: float = 0.0
    agility_prior_contact_policy_frame: int = 253
    agility_prior_joint_scales: tuple[float, ...] = (0.0,) * 12 + (1.0,) * 17
    agility_prior_active_frame_count: int = 0
    agility_prior_peak_target_delta_rad: float = 0.0
    agility_prior_peak_velocity_delta_rad_s: float = 0.0
    last_agility_prior_target_delta: np.ndarray | None = None
    last_agility_prior_velocity_delta: np.ndarray | None = None
    last_agility_prior_active: bool = False
    contact_prior: G1FootballMotionPrior | None = None
    contact_prior_position_blend: float = 0.0
    contact_prior_velocity_blend: float = 0.0
    contact_prior_contact_policy_frame: int = 253
    contact_prior_active_frame_count: int = 0
    contact_prior_peak_target_delta_rad: float = 0.0
    contact_prior_peak_velocity_delta_rad_s: float = 0.0
    last_contact_prior_target_delta: np.ndarray | None = None
    last_contact_prior_velocity_delta: np.ndarray | None = None
    last_contact_prior_active: bool = False
    contact_prior_joint_scales: tuple[float, ...] = (1.0,) * 6
    recovery_controller: Any | None = None
    phase_hold_frames: int = 0
    phase_hold_remaining: int = 0
    post_policy_frame: int | None = None
    post_policy_blend_frames: int = 0
    post_policy_active: bool = False
    post_policy_transition_step: int = 0
    post_policy_origin_target: np.ndarray | None = None
    post_policy_origin_kp: np.ndarray | None = None
    post_policy_origin_kd: np.ndarray | None = None
    post_policy_blend_fraction: float = 0.0
    post_policy_activation_simulation_frame: int | None = None
    entered: bool = False
    contact_latched: bool = False
    contact_time: float | None = None
    transition_actor: CausalSkillTransitionActor | None = None
    transition_decision: CausalTransitionDecision | None = None
    transition_triggered: bool = False
    transition_trigger_time_sec: float | None = None
    last_transition_features: np.ndarray | None = None
    motion_joint_order: np.ndarray | None = None
    causal_strike_option: G1CausalStrikeOptionController | None = None
    last_causal_strike_option_decision: CausalStrikeOptionDecision | None = None
    causal_strike_bridge: G1VelocityMatchedTransitionBridge | None = None
    causal_strike_bridge_frame: int = 0
    causal_strike_bridge_frames: int = 0
    causal_strike_bridge_phase_start: int = 0
    causal_strike_bridge_kp: np.ndarray | None = None
    causal_strike_bridge_kd: np.ndarray | None = None
    causal_strike_bridge_started: bool = False
    causal_strike_bridge_start_time_sec: float | None = None
    causal_strike_bridge_predecessor_policy_frame: int | None = None
    causal_strike_selected_phase_start_frame: int | None = None
    causal_strike_bridge_peak_target_velocity_rms_rad_s: float = 0.0
    causal_strike_abort_recovery_activated: bool = False
    runtime_strike_router: G1RuntimeCausalStrikeRouter | G1RuntimeContactModeActor | None = None
    runtime_strike_route_decision: CausalStrikeRouteDecision | RuntimeContactModeDecision | None = (
        None
    )
    runtime_strike_route_time_sec: float | None = None
    last_runtime_strike_features: np.ndarray | None = None
    runtime_receive_actor: G1RuntimeReceiveActor | None = None
    runtime_receive_probe_action: RuntimeReceiveAction | None = None
    runtime_receive_decision: RuntimeReceiveDecision | None = None
    runtime_receive_time_sec: float | None = None
    last_runtime_receive_features: np.ndarray | None = None
    runtime_contact_target_actor: G1RuntimeContactTargetActor | None = None
    runtime_contact_target_decision: RuntimeContactTargetDecision | None = None
    runtime_contact_target_time_sec: float | None = None
    runtime_finish_plan_actor: G1RuntimeFinishPlanActor | None = None
    runtime_finish_plan_decision: RuntimeFinishPlanDecision | None = None
    runtime_finish_plan_time_sec: float | None = None
    latest_left_support: bool = False
    latest_right_support: bool = False
    phase_hold_count: int = 0
    phase_advance_count: int = 0
    max_ball_phase_error_m: float = 0.0
    last_phase_correction: int = 0
    phase_sync_enabled: bool = False
    left_support_anchor: np.ndarray | None = None
    right_support_anchor: np.ndarray | None = None
    latest_support_slip_m: float = 0.0
    peak_support_slip_m: float = 0.0
    post_contact_peak_support_slip_m: float = 0.0
    contact_impulse_ns: float = 0.0
    recovery_torque_actor: Any | None = None
    learned_torque_frame_count: int = 0
    learned_torque_fallback_count: int = 0
    learned_torque_confidence_sum: float = 0.0
    learned_torque_peak_residual_nm: float = 0.0
    learned_torque_support_rms_peak: float = 0.0
    joint_guard_enabled: bool = False
    joint_guard_frame_count: int = 0
    post_policy_neutral_velocity_enabled: bool = False
    post_policy_forward_velocity_mps: float = 0.0
    joint_guard_config: G1JointGuardConfig = G1JointGuardConfig()
    joint_guard_late_config: G1JointGuardConfig | None = None
    selected_joint_guard_config: G1JointGuardConfig | None = None
    joint_guard_route: str = "disabled"
    post_policy_recovery_enabled: bool = False
    recovery_active_frame_count: int = 0
    recovery_peak_blend_fraction: float = 0.0
    last_recovery_active: bool = False
    last_recovery_blend_fraction: float = 0.0
    early_arrival_parameters: ShotParameters | None = None
    early_arrival_expert_frame_count: int = 0
    goalkeeper_reach_memory: np.ndarray | None = None
    goalkeeper_reach_memory_side: int = 0
    goalkeeper_reach_memory_peak_rad: float = 0.0
    standby_locomotion_mirror_active: bool = False
    recovery_athlete_active_frame_count: int = 0
    recovery_athlete_suppressed_frame_count: int = 0
    last_recovery_athlete_active: bool = False
    last_recovery_athlete_suppressed: bool = False
    last_recovery_athlete_raw_world_command: np.ndarray | None = None
    last_recovery_athlete_world_command: np.ndarray | None = None
    last_tactical_world_target: np.ndarray | None = None
    last_tactical_world_command: np.ndarray | None = None
    last_tactical_movement_active: bool = False
    last_reactive_route_features: np.ndarray | None = None
    last_reactive_route_support_distance: float = 0.0
    last_reactive_route_accepted: bool = False
    last_temporal_route_memory: TemporalRouteMemory | None = None
    last_reactive_role_separation_m: float = math.inf
    last_reactive_collision_shield_active: bool = False
    last_reactive_velocity_braking_correction: np.ndarray | None = None
    reactive_diagonal_braking_confidence: float | None = None


def _base_scenario() -> GoalForgeScenario:
    return GoalForgeScenario(
        scenario_id="soccer_shared_world_nominal",
        partition=Partition.DEVELOPMENT,
        seed=0,
        seed_commitment=hash_json({"task": "soccer_shared_world", "seed": 0}),
        generation=0,
        ball_x_m=1.0,
        ball_y_m=0.0,
        ball_velocity_x_mps=0.0,
        ball_velocity_y_mps=0.0,
        target_y_m=0.0,
        target_z_m=0.2,
        ball_mass_kg=0.41,
        ball_ground_friction=0.05,
        restitution=0.55,
        support_ground_friction=1.0,
        control_latency_ms=0.0,
        observation_noise_m=0.0,
        joint_zero_bias_rad=0.0,
        disturbance_n=0.0,
    )


def _recovery_config() -> G1CerebellarRecoveryConfig:
    return G1CerebellarRecoveryConfig(
        start_policy_frame=300,
        blend_frames=100,
        standing_pose_blend=0.30,
        roll_posture_bias_rad=-0.05,
        settling_start_policy_frame=400,
        settling_blend_frames=100,
        settling_standing_pose_blend=0.45,
        settling_roll_posture_bias_rad=-0.02,
        settling_waist_pitch_bias_rad=0.12,
        target_smoothing_alpha=0.60,
        target_smoothing_start_policy_frame=300,
        target_smoothing_joint_group="upper_body",
    )


def _balanced_dive_phase_profile(
    *,
    elapsed_sec: float,
    flight_duration_sec: float,
    phase_at_arrival: float,
    peak_phase: float,
    blend_in_sec: float,
    recovery_tail_sec: float,
    initial_phase: float = 0.0,
) -> tuple[float, float, bool]:
    """Return the imitation phase and a continuous ownership envelope.

    The source dive first blends into its initial posture and only then
    advances through the recorded motion.  Recovery must start after both
    intervals have completed.  Subtracting only ``phase_duration`` from the
    wall-clock elapsed time used to include ``blend_in_sec`` twice and caused
    an abrupt ownership drop at the first recovery frame.
    """

    values = (
        elapsed_sec,
        flight_duration_sec,
        phase_at_arrival,
        peak_phase,
        blend_in_sec,
        recovery_tail_sec,
        initial_phase,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("balanced dive phase settings must be finite")
    if elapsed_sec < 0.0 or flight_duration_sec <= 0.0:
        raise ValueError("balanced dive timing must be non-negative")
    if (
        not 0.0 <= initial_phase < phase_at_arrival <= peak_phase <= 1.0
        or blend_in_sec <= 0.0
        or recovery_tail_sec <= 0.0
    ):
        raise ValueError("balanced dive phase envelope is invalid")
    phase_elapsed = max(0.0, elapsed_sec - blend_in_sec)
    phase_duration = max(
        0.20,
        (flight_duration_sec - blend_in_sec) / (phase_at_arrival - initial_phase),
    )
    raw_phase = initial_phase + phase_elapsed / phase_duration
    phase = float(np.clip(raw_phase, 0.0, peak_phase))
    blend_gate = float(np.clip(elapsed_sec / blend_in_sec, 0.0, 1.0))
    phase_complete_sec = blend_in_sec + (peak_phase - initial_phase) * phase_duration
    release_sec = phase_complete_sec + recovery_tail_sec
    if elapsed_sec > phase_complete_sec:
        blend_gate *= float(
            np.clip(
                1.0 - (elapsed_sec - phase_complete_sec) / recovery_tail_sec,
                0.0,
                1.0,
            )
        )
    return phase, blend_gate, elapsed_sec <= release_sec + 1.0e-12


def _landing_capture_profile(*, elapsed_sec: float, duration_sec: float) -> tuple[float, bool]:
    """Blend a landed posture into the ready pose without a target jump."""

    if (
        not math.isfinite(elapsed_sec)
        or not math.isfinite(duration_sec)
        or elapsed_sec < 0.0
        or duration_sec <= 0.0
    ):
        raise ValueError("landing capture timing must be finite and positive")
    phase = float(np.clip(elapsed_sec / duration_sec, 0.0, 1.0))
    blend = phase * phase * (3.0 - 2.0 * phase)
    return blend, elapsed_sec <= duration_sec + 1.0e-12


def _latch_gmt_mirror_direction(
    current: bool | None,
    *,
    skill_active: bool,
    local_intercept_y_m: float,
    mirror_enabled: bool,
) -> bool | None:
    """Choose one sagittal mirror at skill onset and retain it for the event."""

    if not math.isfinite(local_intercept_y_m):
        raise ValueError("GMT mirror intercept must be finite")
    if current is None and skill_active:
        return bool(mirror_enabled and local_intercept_y_m > 0.0)
    return current


def _mirror_gmt_proprioception(
    joint_position: NDArray[np.float64],
    joint_velocity: NDArray[np.float64],
    torso_quaternion_wxyz: NDArray[np.float64],
    angular_velocity_body_rad_s: NDArray[np.float64],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Reflect live proprioception into the frozen GMT exemplar's half-space."""

    quaternion = np.asarray(torso_quaternion_wxyz, dtype=np.float64)
    angular_velocity = np.asarray(angular_velocity_body_rad_s, dtype=np.float64)
    if quaternion.shape != (4,) or angular_velocity.shape != (3,):
        raise ValueError("GMT mirrored proprioception shapes are invalid")
    mirrored_quaternion = quaternion.copy()
    mirrored_quaternion[(1, 3),] *= -1.0
    mirrored_angular_velocity = angular_velocity.copy()
    mirrored_angular_velocity[(0, 2),] *= -1.0
    return (
        _mirror_g1_joint_positions(joint_position),
        _mirror_g1_joint_positions(joint_velocity),
        mirrored_quaternion,
        mirrored_angular_velocity,
    )


def _build_recovery_controller(
    qualification: G1AssetQualification,
    scenario: GoalForgeScenario,
    config: Any,
) -> Any:
    resolved = config or _recovery_config()
    from rosclaw.simforge.g1_cerebellar_recovery import (
        evaluate_g1_cerebellar_recovery_regime,
    )

    eligible, reasons = evaluate_g1_cerebellar_recovery_regime(
        support_friction=scenario.support_ground_friction,
        control_latency_ms=scenario.control_latency_ms,
        disturbance_n=scenario.disturbance_n,
        config=resolved,
    )
    return build_shared_recovery_controller(
        qualification,
        regime_commitment=scenario.scenario_commitment,
        regime_eligible=eligible,
        regime_reasons=reasons,
        config=resolved,
    )


def simulate_shared_world(
    asset_root: Path,
    **simulation_kwargs: Any,
) -> tuple[G1SharedWorldResult, dict[str, np.ndarray]]:
    """Run one fail-closed SIM-only shared-ball three-role rollout."""

    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    return _simulate_shared_world(
        qualification.asset_root,
        qualification,
        **simulation_kwargs,
    )


def _simulate_shared_world(
    asset_root: Path,
    qualification: G1AssetQualification,
    *,
    shooter_start_sec: float = _SHOOTER_START_SEC,
    shooter_target: tuple[float, float, float] = (5.0, 1.10, 1.09),
    shooter_policy_target: tuple[float, float, float] | None = None,
    shooter_parameter_overrides: dict[str, float] | None = None,
    shooter_early_arrival_parameter_overrides: dict[str, float] | None = None,
    passer_parameter_overrides: dict[str, float] | None = None,
    passer_recovery_config: Any | None = None,
    passer_post_policy_frame: int | None = 290,
    passer_post_policy_blend_frames: int = 0,
    passer_joint_guard_enabled: bool = False,
    passer_precontact_joint_guard_enabled: bool = False,
    passer_waist_pitch_target_margin_rad: float = 0.0,
    passer_post_policy_neutral_velocity_enabled: bool = False,
    passer_joint_guard_config: G1JointGuardConfig | None = None,
    passer_post_policy_recovery_enabled: bool = False,
    ball_ground_friction: float = 0.10,
    receiver_phase_sync_enabled: bool = True,
    shooter_transition_actor_path: Path | None = None,
    shooter_causal_strike_option_config: G1CausalStrikeOptionConfig | None = None,
    shooter_runtime_strike_router_path: Path | None = None,
    shooter_runtime_contact_mode_actor_path: Path | None = None,
    shooter_runtime_receive_actor_path: Path | None = None,
    shooter_runtime_receive_probe_action: RuntimeReceiveAction | None = None,
    shooter_runtime_contact_target_actor_path: Path | None = None,
    shooter_runtime_finish_plan_actor_path: Path | None = None,
    shooter_recovery_candidate_path: Path | None = None,
    shooter_recovery_residual_config: IQLResidualGuardConfig | None = None,
    shooter_recovery_config: Any | None = None,
    shooter_post_policy_frame: int | None = 430,
    shooter_post_policy_blend_frames: int = 0,
    shooter_joint_guard_enabled: bool = False,
    shooter_precontact_joint_guard_enabled: bool = False,
    shooter_post_policy_neutral_velocity_enabled: bool = False,
    shooter_post_policy_forward_velocity_mps: float = 0.0,
    shooter_joint_guard_config: G1JointGuardConfig | None = None,
    shooter_joint_guard_late_config: G1JointGuardConfig | None = None,
    shooter_post_policy_recovery_enabled: bool = False,
    shooter_motion_prior_path: Path | None = None,
    shooter_motion_prior_position_blend: float = 0.0,
    shooter_motion_prior_velocity_blend: float = 0.0,
    shooter_motion_prior_strike_leg_scale: float = 1.0,
    shooter_motion_prior_joint_scales: tuple[float, ...] = (1.0,) * 29,
    shooter_motion_prior_velocity_joint_scales: tuple[float, ...] | None = None,
    shooter_motion_prior_contact_policy_frame: int = 253,
    shooter_agility_prior_path: Path | None = None,
    shooter_agility_prior_position_blend: float = 0.0,
    shooter_agility_prior_velocity_blend: float = 0.0,
    shooter_agility_prior_contact_policy_frame: int = 253,
    shooter_agility_prior_joint_scales: tuple[float, ...] = (0.0,) * 12 + (1.0,) * 17,
    shooter_contact_prior_path: Path | None = None,
    shooter_contact_prior_position_blend: float = 0.0,
    shooter_contact_prior_velocity_blend: float = 0.0,
    shooter_contact_prior_contact_policy_frame: int = 253,
    shooter_contact_prior_joint_scales: tuple[float, ...] = (1.0,) * 6,
    shooter_ballistic_actor_path: Path | None = None,
    shooter_ballistic_actor_proximity_m: float | None = None,
    shooter_three_axis_contact_actor_path: Path | None = None,
    shooter_target_velocity_contact_actor_path: Path | None = None,
    shooter_target_foot_velocity_xyz_mps: tuple[float, float, float] | None = None,
    shooter_neural_contact_actor_path: Path | None = None,
    shooter_neural_contact_policy_frame: int | None = None,
    shooter_neural_contact_target_velocity_xyz_mps: tuple[float, float, float] | None = None,
    shooter_ballistic_contact_config: G1BallisticContactResidualConfig | None = None,
    shooter_ballistic_contact_torque_config: G1BallisticContactTorqueResidualConfig | None = None,
    shooter_first_touch_interception_config: FirstTouchInterceptionConfig | None = None,
    passer_reception_interception_config: FirstTouchInterceptionConfig | None = None,
    shooter_loft_teacher_config: G1LoftTeacherConfig | None = None,
    shooter_origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    passer_start_sec: float = 0.0,
    passer_collision_enabled: bool = True,
    passer_origin: tuple[float, float, float] | None = None,
    passer_yaw_rad: float = _PASSER_YAW,
    passer_ball_local_xy: tuple[float, float] = (1.205, -0.16),
    passer_policy_target_m: tuple[float, float, float] = (5.0, 0.0, 0.20),
    pass_reception_target_m: tuple[float, float, float] = (1.00, 0.0, 0.115),
    passer_tactical_movement_config: G1TacticalMovementConfig | None = None,
    passer_reactive_movement_config: G1ReactiveMovementConfig | None = None,
    goal_spec: G1TrainingGoalSpec | None = None,
    goalkeeper_config: G1GoalkeeperConfig | None = None,
    goalkeeper_tactical_movement_config: G1TacticalMovementConfig | None = None,
    goalkeeper_reactive_movement_config: G1ReactiveMovementConfig | None = None,
    goalkeeper_origin_override_m: tuple[float, float, float] | None = None,
    goalkeeper_threat_role: str = "shooter",
    second_threat_config: G1SecondThreatConfig | None = None,
    physical_second_striker_config: G1PhysicalSecondStrikerConfig | None = None,
    second_striker_ballistic_actor_path: Path | None = None,
    second_striker_loft_teacher_config: G1LoftTeacherConfig | None = None,
    second_striker_ballistic_contact_config: G1BallisticContactResidualConfig | None = None,
    second_striker_ballistic_contact_torque_config: (
        G1BallisticContactTorqueResidualConfig | None
    ) = None,
    second_ball_mass_kg: float | None = None,
    second_ball_ground_friction: float | None = None,
    unified_stadium_scene: bool = True,
    shooter_ball_initial_position_m: tuple[float, float, float] | None = None,
    ball_launcher_position_m: tuple[float, float, float] | None = None,
    ball_launcher_velocity_mps: tuple[float, float, float] | None = None,
    launcher_receiver_enabled: bool = False,
    simulation_duration_sec: float = _TOTAL_TIME_SEC,
) -> tuple[G1SharedWorldResult, dict[str, np.ndarray]]:
    import mujoco

    if shooter_recovery_candidate_path is not None:
        raise ValueError(
            "shared-world recovery actors must use the Soccer 110-D approach/strike "
            "contract; legacy 74-D recovery artifacts are not accepted"
        )
    if shooter_transition_actor_path is not None and not receiver_phase_sync_enabled:
        raise ValueError("causal shooter transition requires receiver phase synchronization")
    if shooter_causal_strike_option_config is not None and (
        shooter_transition_actor_path is not None or receiver_phase_sync_enabled
    ):
        raise ValueError(
            "causal strike option requires an exclusive clock and disabled legacy phase sync"
        )
    runtime_route_paths = (
        shooter_runtime_strike_router_path,
        shooter_runtime_contact_mode_actor_path,
        shooter_runtime_receive_actor_path,
        shooter_runtime_contact_target_actor_path,
        shooter_runtime_finish_plan_actor_path,
    )
    if sum(path is not None for path in runtime_route_paths) > 1:
        raise ValueError("runtime strike, body-contact and RECEIVE actors are mutually exclusive")
    if (
        shooter_runtime_receive_actor_path is not None
        and shooter_runtime_receive_probe_action is not None
    ):
        raise ValueError("runtime RECEIVE actor and training intervention are mutually exclusive")
    runtime_router_requested = (
        shooter_runtime_strike_router_path is not None
        or shooter_runtime_contact_mode_actor_path is not None
    )
    if runtime_router_requested and (
        shooter_causal_strike_option_config is None
        or shooter_ballistic_contact_torque_config is None
    ):
        raise ValueError(
            "runtime strike routing requires the causal option and bounded contact muscle memory"
        )
    runtime_receive_requested = (
        shooter_runtime_receive_actor_path is not None
        or shooter_runtime_receive_probe_action is not None
        or shooter_runtime_contact_target_actor_path is not None
        or shooter_runtime_finish_plan_actor_path is not None
    )
    if runtime_receive_requested and (
        shooter_causal_strike_option_config is None or shooter_neural_contact_actor_path is None
    ):
        raise ValueError(
            "runtime RECEIVE control requires the causal option and neural contact muscle memory"
        )
    if shooter_runtime_contact_mode_actor_path is not None and (
        shooter_three_axis_contact_actor_path is None
        and shooter_target_velocity_contact_actor_path is None
    ):
        raise ValueError("runtime body-contact routing requires a task-space contact actor")
    if shooter_ballistic_actor_path is not None and shooter_loft_teacher_config is not None:
        raise ValueError("shooter ballistic actor and loft teacher cannot share torque authority")
    if shooter_three_axis_contact_actor_path is not None and (
        shooter_ballistic_actor_path is not None
        or shooter_loft_teacher_config is not None
        or shooter_target_velocity_contact_actor_path is not None
    ):
        raise ValueError(
            "three-axis contact actor requires exclusive learned/teacher torque authority"
        )
    if (shooter_target_velocity_contact_actor_path is None) != (
        shooter_target_foot_velocity_xyz_mps is None
    ):
        raise ValueError("target-velocity actor and committed target must be configured together")
    if shooter_target_velocity_contact_actor_path is not None and (
        shooter_ballistic_actor_path is not None or shooter_loft_teacher_config is not None
    ):
        raise ValueError("target-velocity actor requires exclusive learned torque authority")
    if shooter_neural_contact_actor_path is None:
        if (
            shooter_neural_contact_policy_frame is not None
            or shooter_neural_contact_target_velocity_xyz_mps is not None
        ):
            raise ValueError("neural contact settings require a neural contact actor")
    elif (
        shooter_runtime_contact_target_actor_path is None
        and shooter_runtime_finish_plan_actor_path is None
    ):
        if (
            shooter_neural_contact_policy_frame is None
            or shooter_neural_contact_target_velocity_xyz_mps is None
        ):
            raise ValueError("neural contact actor, contact frame and target are one commitment")
    elif shooter_runtime_contact_target_actor_path is not None and (
        shooter_neural_contact_policy_frame is None
        or shooter_neural_contact_target_velocity_xyz_mps is not None
    ):
        raise ValueError(
            "runtime contact target actor requires a fixed contact frame and owns the target"
        )
    elif shooter_runtime_finish_plan_actor_path is not None and (
        shooter_neural_contact_policy_frame is not None
        or shooter_neural_contact_target_velocity_xyz_mps is not None
    ):
        raise ValueError("runtime finish plan actor owns the contact frame and target")
    if shooter_neural_contact_actor_path is not None and any(
        value is not None
        for value in (
            shooter_ballistic_actor_path,
            shooter_three_axis_contact_actor_path,
            shooter_target_velocity_contact_actor_path,
            shooter_loft_teacher_config,
            shooter_ballistic_contact_torque_config,
        )
    ):
        raise ValueError("neural contact actor requires exclusive contact-torque authority")
    if shooter_loft_teacher_config is not None and not shooter_loft_teacher_config.enabled:
        raise ValueError("shooter loft teacher must contain a non-zero task-space target")
    if (ball_launcher_position_m is None) != (ball_launcher_velocity_mps is None):
        raise ValueError("ball launcher position and velocity must be configured together")
    if shooter_ball_initial_position_m is not None and ball_launcher_position_m is not None:
        raise ValueError("direct-shot and launcher ball initializations are mutually exclusive")
    if launcher_receiver_enabled and ball_launcher_position_m is None:
        raise ValueError("launcher receiver requires a configured moving-ball launcher")
    if second_threat_config is not None and (
        goalkeeper_config is None
        or not goalkeeper_config.post_contact_ready_recovery_enabled
        or goalkeeper_config.successor_lateral_probe_enabled
        or ball_launcher_position_m is not None
        or shooter_ball_initial_position_m is not None
        or simulation_duration_sec
        < second_threat_config.launch_time_sec + second_threat_config.flight_time_sec + 2.0
    ):
        raise ValueError(
            "second-threat curriculum requires a recovered goalkeeper, continuous first chain "
            "and enough unprobed simulation time"
        )
    if physical_second_striker_config is not None and (
        goalkeeper_config is None
        or not goalkeeper_config.post_contact_ready_recovery_enabled
        or goalkeeper_config.successor_lateral_probe_enabled
        or second_threat_config is not None
        or ball_launcher_position_m is not None
        or shooter_ball_initial_position_m is not None
        or shooter_ballistic_actor_path is None
        or simulation_duration_sec
        < physical_second_striker_config.observer_rearm_earliest_sec + 4.0
    ):
        raise ValueError(
            "physical second striker requires a recovered goalkeeper, ballistic contact "
            "actor, continuous first chain, no launcher and enough simulation time"
        )
    if physical_second_striker_config is None and (
        second_striker_ballistic_actor_path is not None
        or second_striker_loft_teacher_config is not None
        or second_ball_mass_kg is not None
        or second_ball_ground_friction is not None
    ):
        raise ValueError(
            "second-striker actor, teacher and ball physics require a physical second striker"
        )
    if second_striker_loft_teacher_config is not None and (
        not second_striker_loft_teacher_config.enabled
        or second_striker_ballistic_actor_path is not None
    ):
        raise ValueError(
            "second-striker loft teacher must be enabled and cannot share plastic authority"
        )
    if goalkeeper_threat_role not in {"shooter", "passer"}:
        raise ValueError("goalkeeper threat role must be shooter or passer")
    if goalkeeper_threat_role != "shooter" and (
        second_threat_config is not None or physical_second_striker_config is not None
    ):
        raise ValueError("alternate goalkeeper threat roles cannot own a second-threat chain")
    if not math.isfinite(passer_start_sec) or not 0.0 <= passer_start_sec <= 120.0:
        raise ValueError("passer start time must be in [0, 120] seconds")
    if not isinstance(passer_collision_enabled, bool):
        raise ValueError("passer collision flag must be boolean")
    if goalkeeper_tactical_movement_config is not None and goalkeeper_config is None:
        raise ValueError("goalkeeper tactical movement requires a goalkeeper configuration")
    if goalkeeper_reactive_movement_config is not None and goalkeeper_config is None:
        raise ValueError("goalkeeper reactive movement requires a goalkeeper configuration")
    if passer_tactical_movement_config is not None and passer_start_sec <= 1.0e-12:
        raise ValueError("passer tactical movement requires a delayed passer policy start")
    if passer_reactive_movement_config is not None and passer_start_sec <= 1.0e-12:
        raise ValueError("passer reactive movement requires a delayed passer policy start")
    if (
        passer_tactical_movement_config is not None and passer_reactive_movement_config is not None
    ) or (
        goalkeeper_tactical_movement_config is not None
        and goalkeeper_reactive_movement_config is not None
    ):
        raise ValueError("fixed and reactive movement routes are mutually exclusive")
    if passer_reactive_movement_config is not None and (
        passer_reactive_movement_config.role != "teammate"
    ):
        raise ValueError("passer reactive movement must use the teammate role")
    if goalkeeper_reactive_movement_config is not None and (
        goalkeeper_reactive_movement_config.role != "defender"
    ):
        raise ValueError("goalkeeper reactive movement must use the defender role")
    if second_ball_mass_kg is not None and (
        not math.isfinite(second_ball_mass_kg) or not 0.40 <= second_ball_mass_kg <= 0.46
    ):
        raise ValueError("second football mass must be in [0.40, 0.46] kg")
    if second_ball_ground_friction is not None and (
        not math.isfinite(second_ball_ground_friction)
        or not 0.03 <= second_ball_ground_friction <= 0.80
    ):
        raise ValueError("second football ground friction must be in [0.03, 0.80]")
    if second_threat_config is not None and goalkeeper_config is not None:
        second_peak_punch_vector = second_threat_config.goalkeeper_punch_force_n * math.sqrt(
            1.0
            + goalkeeper_config.actor_bimanual_punch_vertical_force_scale**2
            + second_threat_config.goalkeeper_punch_outward_force_scale**2
        )
        if second_peak_punch_vector > 160.0:
            raise ValueError("second-threat punch exceeds the goalkeeper force envelope")
    direct_shot_position = (
        None
        if shooter_ball_initial_position_m is None
        else np.asarray(shooter_ball_initial_position_m, dtype=np.float64)
    )
    if direct_shot_position is not None and (
        direct_shot_position.shape != (3,)
        or not np.all(np.isfinite(direct_shot_position))
        or direct_shot_position[2] <= 0.0
    ):
        raise ValueError("direct-shot ball position must contain a finite above-ground xyz")
    launcher_position = (
        None
        if ball_launcher_position_m is None
        else np.asarray(ball_launcher_position_m, dtype=np.float64)
    )
    launcher_velocity = (
        None
        if ball_launcher_velocity_mps is None
        else np.asarray(ball_launcher_velocity_mps, dtype=np.float64)
    )
    if launcher_position is not None and (
        launcher_position.shape != (3,)
        or launcher_velocity is None
        or launcher_velocity.shape != (3,)
        or not np.all(np.isfinite(launcher_position))
        or not np.all(np.isfinite(launcher_velocity))
        or float(np.linalg.norm(launcher_velocity)) <= 0.10
    ):
        raise ValueError("ball launcher state must contain finite xyz position and velocity")
    if not math.isfinite(simulation_duration_sec) or not 2.0 <= simulation_duration_sec <= 25.0:
        raise ValueError("shared-world duration must be in [2, 25] seconds")

    for label, blend in (
        ("position", shooter_motion_prior_position_blend),
        ("velocity", shooter_motion_prior_velocity_blend),
    ):
        if not math.isfinite(blend) or not 0.0 <= blend <= 0.50:
            raise ValueError(f"shooter motion-prior {label} blend must be in [0, 0.50]")
    motion_prior_requested = (
        shooter_motion_prior_position_blend > 0.0 or shooter_motion_prior_velocity_blend > 0.0
    )
    if (shooter_motion_prior_path is None) != (not motion_prior_requested):
        raise ValueError("shooter motion-prior path and non-zero blend must be configured together")
    if not 220 <= shooter_motion_prior_contact_policy_frame <= 290:
        raise ValueError("shooter motion-prior contact frame must be in [220, 290]")
    if not math.isfinite(shooter_motion_prior_strike_leg_scale) or not (
        0.0 <= shooter_motion_prior_strike_leg_scale <= 1.0
    ):
        raise ValueError("shooter motion-prior strike-leg scale must be in [0, 1]")
    if len(shooter_motion_prior_joint_scales) != 29 or not all(
        math.isfinite(value) and 0.0 <= value <= 2.0 for value in shooter_motion_prior_joint_scales
    ):
        raise ValueError("shooter motion-prior joint scales must contain 29 values in [0, 2]")
    velocity_joint_scales = (
        shooter_motion_prior_joint_scales
        if shooter_motion_prior_velocity_joint_scales is None
        else shooter_motion_prior_velocity_joint_scales
    )
    if len(velocity_joint_scales) != 29 or not all(
        math.isfinite(value) and 0.0 <= value <= 2.0 for value in velocity_joint_scales
    ):
        raise ValueError(
            "shooter motion-prior velocity joint scales must contain 29 values in [0, 2]"
        )
    if not math.isfinite(shooter_agility_prior_position_blend) or not (
        0.0 <= shooter_agility_prior_position_blend <= 0.50
    ):
        raise ValueError("shooter agility-prior position blend must be in [0, 0.50]")
    if not math.isfinite(shooter_agility_prior_velocity_blend) or not (
        0.0 <= shooter_agility_prior_velocity_blend <= 0.10
    ):
        raise ValueError("shooter agility-prior velocity blend must be in [0, 0.10]")
    agility_prior_requested = (
        shooter_agility_prior_position_blend > 0.0 or shooter_agility_prior_velocity_blend > 0.0
    )
    if (shooter_agility_prior_path is None) != (not agility_prior_requested):
        raise ValueError(
            "shooter agility-prior path and non-zero blend must be configured together"
        )
    if not 220 <= shooter_agility_prior_contact_policy_frame <= 290:
        raise ValueError("shooter agility-prior contact frame must be in [220, 290]")
    if len(shooter_agility_prior_joint_scales) != 29 or not all(
        math.isfinite(value) and 0.0 <= value <= 1.0 for value in shooter_agility_prior_joint_scales
    ):
        raise ValueError("shooter agility-prior joint scales must contain 29 values in [0, 1]")
    if any(value != 0.0 for value in shooter_agility_prior_joint_scales[:12]):
        raise ValueError("shooter agility-prior cannot modify either leg")
    if not math.isfinite(shooter_contact_prior_position_blend) or not (
        0.0 <= shooter_contact_prior_position_blend <= 0.20
    ):
        raise ValueError("shooter contact-prior position blend must be in [0, 0.20]")
    if not math.isfinite(shooter_contact_prior_velocity_blend) or not (
        0.0 <= shooter_contact_prior_velocity_blend <= 0.10
    ):
        raise ValueError("shooter contact-prior velocity blend must be in [0, 0.10]")
    contact_prior_requested = (
        shooter_contact_prior_position_blend > 0.0 or shooter_contact_prior_velocity_blend > 0.0
    )
    if (shooter_contact_prior_path is None) != (not contact_prior_requested):
        raise ValueError(
            "shooter contact-prior path and non-zero blend must be configured together"
        )
    if not 220 <= shooter_contact_prior_contact_policy_frame <= 290:
        raise ValueError("shooter contact-prior contact frame must be in [220, 290]")
    if len(shooter_contact_prior_joint_scales) != 6 or not all(
        math.isfinite(value) and 0.0 <= value <= 1.0 for value in shooter_contact_prior_joint_scales
    ):
        raise ValueError("shooter contact-prior joint scales must contain six values in [0, 1]")
    if not math.isfinite(shooter_post_policy_forward_velocity_mps) or not (
        0.0 <= shooter_post_policy_forward_velocity_mps <= 0.15
    ):
        raise ValueError("shooter post-policy forward velocity must be in [0, 0.15] m/s")

    root = asset_root.expanduser().resolve()
    active_shooter_origin = np.asarray(shooter_origin, dtype=np.float64)
    if active_shooter_origin.shape != (3,) or not np.all(np.isfinite(active_shooter_origin)):
        raise ValueError("shooter origin must be a finite xyz vector")
    if (
        not -2.0 <= float(active_shooter_origin[0]) <= 6.0
        or abs(float(active_shooter_origin[1])) > 4.00
        or not math.isclose(float(active_shooter_origin[2]), 0.0, abs_tol=1e-12)
    ):
        raise ValueError("shared-world shooter origin exceeds the qualified pitch envelope")
    active_passer_origin = np.asarray(
        _PASSER_ORIGIN if passer_origin is None else passer_origin,
        dtype=np.float64,
    )
    if active_passer_origin.shape != (3,) or not np.all(np.isfinite(active_passer_origin)):
        raise ValueError("passer origin must be a finite xyz vector")
    if not math.isfinite(passer_yaw_rad) or not -math.pi <= passer_yaw_rad <= math.pi:
        raise ValueError("passer yaw must be finite and in [-pi, pi]")
    active_passer_ball_xy = np.asarray(passer_ball_local_xy, dtype=np.float64)
    if (
        active_passer_ball_xy.shape != (2,)
        or not np.all(np.isfinite(active_passer_ball_xy))
        or not 1.05 <= active_passer_ball_xy[0] <= 1.25
        or not -0.30 <= active_passer_ball_xy[1] <= -0.08
    ):
        raise ValueError("passer ball pocket must be finite and inside the qualified envelope")
    active_passer_policy_target = np.asarray(passer_policy_target_m, dtype=np.float64)
    if (
        active_passer_policy_target.shape != (3,)
        or not np.all(np.isfinite(active_passer_policy_target))
        or not 1.0 <= active_passer_policy_target[0] <= 12.0
        or abs(float(active_passer_policy_target[1])) > 4.0
        or not 0.105 <= active_passer_policy_target[2] <= 1.20
    ):
        raise ValueError("passer policy target exceeds the qualified tactical envelope")
    active_pass_reception_target = np.asarray(pass_reception_target_m, dtype=np.float64)
    if active_pass_reception_target.shape != (3,) or not np.all(
        np.isfinite(active_pass_reception_target)
    ):
        raise ValueError("pass reception target must be a finite xyz vector")
    local_pass_reception_target = active_pass_reception_target - active_shooter_origin
    if (
        not 0.80 <= local_pass_reception_target[0] <= 1.40
        or not -0.30 <= local_pass_reception_target[1] <= 0.30
        or not 0.105 <= active_pass_reception_target[2] <= 0.130
    ):
        raise ValueError("pass reception target exceeds the shooter-local strike pocket")
    active_goal = goal_spec or G1TrainingGoalSpec(
        plane_x_m=shooter_target[0],
        target_y_m=shooter_target[1],
        target_z_m=shooter_target[2],
    )
    if not np.allclose(
        np.asarray(shooter_target, dtype=np.float64),
        np.asarray(
            (active_goal.plane_x_m, active_goal.target_y_m, active_goal.target_z_m),
            dtype=np.float64,
        ),
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError("shooter target and shared stadium goal target must match")
    active_policy_target = np.asarray(
        shooter_target if shooter_policy_target is None else shooter_policy_target,
        dtype=np.float64,
    )
    if active_policy_target.shape != (3,) or not np.all(np.isfinite(active_policy_target)):
        raise ValueError("shooter policy target must be a finite xyz vector")
    if goalkeeper_config is not None and not unified_stadium_scene:
        raise ValueError("goalkeeper requires the unified stadium physics scene")
    active_goalkeeper_origin_override: np.ndarray | None = None
    if goalkeeper_origin_override_m is not None:
        active_goalkeeper_origin_override = np.asarray(
            goalkeeper_origin_override_m, dtype=np.float64
        )
        if (
            goalkeeper_config is None
            or active_goalkeeper_origin_override.shape != (3,)
            or not np.all(np.isfinite(active_goalkeeper_origin_override))
            or not -2.0 <= active_goalkeeper_origin_override[0] <= active_goal.plane_x_m
            or abs(float(active_goalkeeper_origin_override[1])) > 4.0
            or not math.isclose(float(active_goalkeeper_origin_override[2]), 0.0, abs_tol=1e-12)
        ):
            raise ValueError("goalkeeper origin override exceeds the shared pitch envelope")
    if passer_waist_pitch_target_margin_rad != 0.0 and not (
        0.005 <= passer_waist_pitch_target_margin_rad <= 0.05
    ):
        raise ValueError("passer waist-pitch margin must be zero or in [0.005, 0.05]")
    model = _coupled_model(
        root,
        passer_origin=active_passer_origin,
        passer_yaw_rad=passer_yaw_rad,
        goal=active_goal,
        goalkeeper_config=goalkeeper_config,
        goalkeeper_origin_override=(
            None
            if active_goalkeeper_origin_override is None
            else (
                float(active_goalkeeper_origin_override[0]),
                float(active_goalkeeper_origin_override[1]),
                float(active_goalkeeper_origin_override[2]),
            )
        ),
        physical_second_striker_config=physical_second_striker_config,
        second_ball_mass_kg=second_ball_mass_kg,
        unified_stadium_scene=unified_stadium_scene,
    )
    data = mujoco.MjData(model)
    model.opt.timestep = _PHYSICS_DT
    scenario = _base_scenario()
    ball_body = _id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    ball_geom = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
    ball_joint = int(model.body_jntadr[ball_body])
    ball_qpos = int(model.jnt_qposadr[ball_joint])
    ball_qvel = int(model.jnt_dofadr[ball_joint])
    second_ball_body: int | None = None
    second_ball_geom: int | None = None
    second_ball_qpos: int | None = None
    second_ball_qvel: int | None = None
    if physical_second_striker_config is not None:
        second_ball_body = _id(model, mujoco.mjtObj.mjOBJ_BODY, "second_ball")
        second_ball_geom = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "second_ball_geom")
        second_ball_joint = _id(model, mujoco.mjtObj.mjOBJ_JOINT, "second_ball_free")
        second_ball_qpos = int(model.jnt_qposadr[second_ball_joint])
        second_ball_qvel = int(model.jnt_dofadr[second_ball_joint])
    floor_geom = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if not 0.03 <= ball_ground_friction <= 0.80:
        raise ValueError("coupled relay ball friction must be in [0.03, 0.80]")
    model.geom_friction[ball_geom] = (
        active_goal.ball_contact_sliding_friction,
        active_goal.ball_torsional_friction,
        active_goal.ball_rolling_friction,
    )
    model.geom_friction[floor_geom, 0] = scenario.support_ground_friction
    for pair_index in range(int(model.npair)):
        pair_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_PAIR, pair_index) or ""
        if pair_name == "ball_floor":
            model.pair_friction[pair_index] = (
                ball_ground_friction,
                ball_ground_friction,
                active_goal.ball_torsional_friction,
                active_goal.ball_rolling_friction,
                active_goal.ball_rolling_friction,
            )
        elif pair_name == "second_ball_floor" and second_ball_ground_friction is not None:
            model.pair_friction[pair_index] = (
                second_ball_ground_friction,
                second_ball_ground_friction,
                active_goal.ball_torsional_friction,
                active_goal.ball_rolling_friction,
                active_goal.ball_rolling_friction,
            )

    state_type, output_type, policy_type, mujoco_to_isaac = _load_robonaldo(root)
    shooter_transition_actor = (
        None
        if shooter_transition_actor_path is None
        else load_causal_skill_transition_actor(shooter_transition_actor_path)
    )
    shooter_runtime_strike_router = (
        None
        if shooter_runtime_strike_router_path is None
        else load_runtime_causal_strike_router(shooter_runtime_strike_router_path)
    )
    shooter_runtime_contact_mode_actor = (
        None
        if shooter_runtime_contact_mode_actor_path is None
        else load_runtime_contact_mode_actor(shooter_runtime_contact_mode_actor_path)
    )
    active_runtime_strike_router = (
        shooter_runtime_contact_mode_actor or shooter_runtime_strike_router
    )
    if active_runtime_strike_router is not None and (
        active_runtime_strike_router.body_hash != qualification.body_hash
        or active_runtime_strike_router.kick_prior_hash != qualification.kick_prior_hash
    ):
        raise ValueError("runtime strike router asset identity changed")
    shooter_runtime_receive_actor = (
        None
        if shooter_runtime_receive_actor_path is None
        else load_runtime_receive_actor(shooter_runtime_receive_actor_path)
    )
    if shooter_runtime_receive_actor is not None and (
        shooter_runtime_receive_actor.body_hash != qualification.body_hash
        or shooter_runtime_receive_actor.kick_prior_hash != qualification.kick_prior_hash
    ):
        raise ValueError("runtime RECEIVE actor asset identity changed")
    shooter_runtime_contact_target_actor = (
        None
        if shooter_runtime_contact_target_actor_path is None
        else load_runtime_contact_target_actor(shooter_runtime_contact_target_actor_path)
    )
    if shooter_runtime_contact_target_actor is not None and (
        shooter_runtime_contact_target_actor.body_hash != qualification.body_hash
        or shooter_runtime_contact_target_actor.kick_prior_hash != qualification.kick_prior_hash
    ):
        raise ValueError("runtime contact target actor asset identity changed")
    shooter_runtime_finish_plan_actor = (
        None
        if shooter_runtime_finish_plan_actor_path is None
        else load_runtime_finish_plan_actor(shooter_runtime_finish_plan_actor_path)
    )
    if shooter_runtime_finish_plan_actor is not None and (
        shooter_runtime_finish_plan_actor.body_hash != qualification.body_hash
        or shooter_runtime_finish_plan_actor.kick_prior_hash != qualification.kick_prior_hash
    ):
        raise ValueError("runtime finish plan actor asset identity changed")
    shooter_motion_prior: G1FootballMotionPrior | None = None
    if shooter_motion_prior_path is not None:
        shooter_motion_prior = load_g1_football_motion_prior(shooter_motion_prior_path)
        if shooter_motion_prior.body_hash != qualification.body_hash:
            raise ValueError("motion-prior Body hash does not match coupled G1")
        if (
            shooter_motion_prior_velocity_blend > 0.0
            and not shooter_motion_prior.whole_body_velocity_reference_rad_s
        ):
            raise ValueError("motion-prior velocity blend requires a velocity-aware artifact")
    shooter_agility_prior: G1MosaicAgilityPrior | None = None
    if shooter_agility_prior_path is not None:
        shooter_agility_prior = load_g1_mosaic_agility_prior(shooter_agility_prior_path)
    shooter_contact_prior: G1FootballMotionPrior | None = None
    if shooter_contact_prior_path is not None:
        shooter_contact_prior = load_g1_football_motion_prior(shooter_contact_prior_path)
        if shooter_contact_prior.body_hash != qualification.body_hash:
            raise ValueError("contact-prior Body hash does not match coupled G1")
        if (
            shooter_contact_prior.schema_version
            not in {
                "rosclaw.growth.g1_football_motion_prior.v1",
                "rosclaw.growth.g1_football_motion_prior.v7",
            }
            or shooter_contact_prior.source_dataset != "OmniContact"
            or shooter_contact_prior.whole_body_reference_rad
        ):
            raise ValueError("contact prior must be the train-only OmniContact right-leg artifact")
        if (
            shooter_contact_prior_velocity_blend > 0.0
            and not shooter_contact_prior.right_leg_velocity_reference_rad_s
        ):
            raise ValueError("contact-prior velocity blend requires a velocity-aware artifact")
    shooter_ballistic_actor: G1BallisticContactImpulseActor | None = None
    shooter_three_axis_contact_actor: G1ThreeAxisContactActor | None = None
    shooter_target_velocity_contact_actor: G1TargetVelocityContactActor | None = None
    shooter_neural_contact_actor: G1NeuralContactActor | None = None
    second_striker_ballistic_actor: G1BallisticContactImpulseActor | None = None
    second_striker_candidate_actor: G1BallisticContactImpulseActor | None = None
    if shooter_ballistic_actor_path is not None:
        shooter_ballistic_actor = load_g1_ballistic_contact_impulse_actor(
            shooter_ballistic_actor_path
        )
        if shooter_ballistic_actor_proximity_m is not None:
            shooter_ballistic_actor = replace(
                shooter_ballistic_actor,
                maximum_foot_ball_distance_m=shooter_ballistic_actor_proximity_m,
            )
        if shooter_ballistic_actor.body_hash != qualification.body_hash:
            raise ValueError("ballistic contact actor Body hash does not match coupled G1")
        second_striker_ballistic_actor = shooter_ballistic_actor
    if shooter_three_axis_contact_actor_path is not None:
        shooter_three_axis_contact_actor = load_g1_three_axis_contact_actor(
            shooter_three_axis_contact_actor_path
        )
        if shooter_three_axis_contact_actor.body_hash != qualification.body_hash:
            raise ValueError("three-axis contact actor Body hash does not match coupled G1")
    if shooter_target_velocity_contact_actor_path is not None:
        shooter_target_velocity_contact_actor = load_g1_target_velocity_contact_actor(
            shooter_target_velocity_contact_actor_path
        )
        if shooter_target_velocity_contact_actor.body_hash != qualification.body_hash:
            raise ValueError("target-velocity contact actor Body hash does not match coupled G1")
        assert shooter_target_foot_velocity_xyz_mps is not None
        if not shooter_target_velocity_contact_actor.target_supported(
            shooter_target_foot_velocity_xyz_mps
        ):
            raise ValueError("committed target foot velocity is outside learned support")
    if shooter_neural_contact_actor_path is not None:
        shooter_neural_contact_actor = load_g1_neural_contact_actor(
            shooter_neural_contact_actor_path
        )
        if shooter_neural_contact_actor.body_hash != qualification.body_hash:
            raise ValueError("neural contact actor Body hash does not match coupled G1")
        if (
            shooter_runtime_contact_target_actor is None
            and shooter_runtime_finish_plan_actor is None
        ):
            assert shooter_neural_contact_target_velocity_xyz_mps is not None
            if not shooter_neural_contact_actor.target_supported(
                shooter_neural_contact_target_velocity_xyz_mps
            ):
                raise ValueError("neural contact target velocity is outside learned support")
        elif shooter_runtime_contact_target_actor is not None and (
            shooter_runtime_contact_target_actor.neural_contact_actor_hash
            != shooter_neural_contact_actor.actor_hash
            or any(
                not shooter_neural_contact_actor.target_supported(
                    memory.action.target_foot_velocity_xyz_mps
                )
                for memory in (
                    *shooter_runtime_contact_target_actor.successful_memories,
                    *shooter_runtime_contact_target_actor.failed_memories,
                )
            )
        ):
            raise ValueError("runtime contact target actor neural lineage changed")
        if shooter_runtime_finish_plan_actor is not None and (
            shooter_runtime_finish_plan_actor.neural_contact_actor_hash
            != shooter_neural_contact_actor.actor_hash
            or any(
                not shooter_neural_contact_actor.target_supported(
                    memory.action.target.target_foot_velocity_xyz_mps
                )
                for memory in (
                    *shooter_runtime_finish_plan_actor.successful_memories,
                    *shooter_runtime_finish_plan_actor.failed_memories,
                )
            )
        ):
            raise ValueError("runtime finish plan actor neural lineage changed")
    if second_striker_ballistic_actor_path is not None:
        second_striker_candidate_actor = load_g1_ballistic_contact_impulse_actor(
            second_striker_ballistic_actor_path
        )
        if second_striker_candidate_actor.body_hash != qualification.body_hash:
            raise ValueError("second-striker contact actor Body hash does not match coupled G1")
        if not second_striker_candidate_actor.target_conditioned:
            raise ValueError("second-striker plastic candidate must be target-conditioned")
    if physical_second_striker_config is not None:
        proximity = physical_second_striker_config.ballistic_actor_proximity_m
        if second_striker_ballistic_actor is not None:
            second_striker_ballistic_actor = replace(
                second_striker_ballistic_actor,
                maximum_foot_ball_distance_m=proximity,
            )
        if second_striker_candidate_actor is not None:
            second_striker_candidate_actor = replace(
                second_striker_candidate_actor,
                maximum_foot_ball_distance_m=proximity,
            )
    with np.load(root / _MOTION_REL) as motion:
        initial_position = np.asarray(motion["body_pos_w"][0, 0], dtype=np.float64)
        initial_quaternion = np.asarray(motion["body_quat_w"][0, 0], dtype=np.float64)
        initial_joints = np.asarray(motion["joint_pos"][0][mujoco_to_isaac], dtype=np.float64)
    standby_target = np.asarray(
        (
            -0.2,
            0.0,
            0.0,
            0.42,
            -0.23,
            0.0,
            -0.2,
            0.0,
            0.0,
            0.42,
            -0.23,
            0.0,
            0.0,
            0.0,
            0.0,
            0.35,
            0.18,
            0.0,
            0.87,
            0.0,
            0.0,
            0.0,
            0.35,
            -0.18,
            0.0,
            0.87,
            0.0,
            0.0,
            0.0,
        ),
        dtype=np.float64,
    )
    standby_kp = np.asarray(
        (
            100,
            100,
            100,
            150,
            40,
            40,
            100,
            100,
            100,
            150,
            40,
            40,
            300,
            300,
            300,
            100,
            100,
            50,
            50,
            20,
            20,
            20,
            100,
            100,
            50,
            50,
            20,
            20,
            20,
        ),
        dtype=np.float64,
    )
    standby_kd = np.asarray(
        (
            2,
            2,
            2,
            4,
            2,
            2,
            2,
            2,
            2,
            4,
            2,
            2,
            3,
            3,
            3,
            2,
            2,
            2,
            2,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            1,
            1,
            1,
        ),
        dtype=np.float64,
    )

    shooter_parameters = ShotParameters(
        stance_offset_y=-0.06,
        pelvis_yaw_offset=0.175,
        com_shift_y=-0.065,
        swing_speed_scale=0.90,
        foot_yaw_offset=0.03025,
        recovery_step_length=0.055,
        policy_type="parameter",
    )
    if shooter_parameter_overrides:
        shooter_parameters = replace(
            shooter_parameters,
            **cast(Any, shooter_parameter_overrides),
        )
    shooter_early_arrival_parameters = (
        replace(
            shooter_parameters,
            **cast(Any, shooter_early_arrival_parameter_overrides),
        )
        if shooter_early_arrival_parameter_overrides
        else None
    )
    second_striker_parameters = (
        replace(
            shooter_parameters,
            kick_foot=physical_second_striker_config.kick_foot,
            stance_offset_y=(
                -shooter_parameters.stance_offset_y
                if physical_second_striker_config.kick_foot == "left"
                else shooter_parameters.stance_offset_y
            ),
            swing_amplitude=physical_second_striker_config.swing_amplitude,
            swing_speed_scale=physical_second_striker_config.swing_speed_scale,
            foot_yaw_offset=physical_second_striker_config.foot_yaw_offset,
            foot_pitch_offset=physical_second_striker_config.foot_pitch_offset,
            loft_synergy=physical_second_striker_config.loft_synergy,
            contact_phase_offset=physical_second_striker_config.contact_phase_offset,
        )
        if physical_second_striker_config is not None
        else None
    )
    passer_parameters = ShotParameters(
        stance_offset_y=-0.04,
        pelvis_yaw_offset=0.10,
        com_shift_y=-0.04,
        swing_amplitude=0.75,
        swing_speed_scale=0.80,
        recovery_step_length=0.03,
        policy_type="parameter",
    )
    if passer_parameter_overrides:
        passer_parameters = replace(
            passer_parameters,
            **cast(Any, passer_parameter_overrides),
        )
    prepared_finish_plan_decision: RuntimeFinishPlanDecision | None = None
    if shooter_runtime_finish_plan_actor is not None:
        prepared_finish_plan_decision = shooter_runtime_finish_plan_actor.decide(
            prepared_finish_plan_features(
                receiver_lane_m=float(active_shooter_origin[1]),
                reception_target_x_m=float(local_pass_reception_target[0]),
                passer_ball_local_xy_m=(
                    float(active_passer_ball_xy[0]),
                    float(active_passer_ball_xy[1]),
                ),
                ball_ground_friction=ball_ground_friction,
                passer_yaw_rad=passer_yaw_rad,
                passer_stance_offset_xy_m=(
                    passer_parameters.stance_offset_x,
                    passer_parameters.stance_offset_y,
                ),
                passer_swing_speed_scale=passer_parameters.swing_speed_scale,
            )
        )
        prepared_action = prepared_finish_plan_decision.action
        if prepared_action is not None:
            shooter_parameters = replace(
                shooter_parameters,
                stance_offset_x=prepared_action.receive.stance_offset_x_m,
                stance_offset_y=prepared_action.receive.stance_offset_y_m,
                foot_yaw_offset=prepared_action.receive.foot_yaw_offset_rad,
                foot_pitch_offset=prepared_action.receive.foot_pitch_offset_rad,
            )
            if shooter_early_arrival_parameters is not None:
                shooter_early_arrival_parameters = replace(
                    shooter_early_arrival_parameters,
                    stance_offset_x=prepared_action.receive.stance_offset_x_m,
                    stance_offset_y=prepared_action.receive.stance_offset_y_m,
                    foot_yaw_offset=prepared_action.receive.foot_yaw_offset_rad,
                    foot_pitch_offset=prepared_action.receive.foot_pitch_offset_rad,
                )
            shooter_neural_contact_policy_frame = prepared_action.receive.contact_policy_frame
            shooter_neural_contact_target_velocity_xyz_mps = (
                prepared_action.target.target_foot_velocity_xyz_mps
            )
            shooter_post_policy_frame = (
                prepared_action.receive.contact_policy_frame
                + shooter_runtime_finish_plan_actor.contact_handoff_offset_frames
            )
            assert shooter_causal_strike_option_config is not None
            shooter_causal_strike_option_config = replace(
                shooter_causal_strike_option_config,
                maximum_arrival_advance_frames=(
                    prepared_action.receive.maximum_arrival_advance_frames
                ),
                arrival_alignment_tolerance_sec=(
                    prepared_action.receive.arrival_alignment_tolerance_sec
                ),
            )
    passer_scenario = replace(
        scenario,
        scenario_id="goalforge-coupled-g1-a-soft-back-pass",
        ball_x_m=float(active_passer_ball_xy[0]),
        ball_y_m=float(active_passer_ball_xy[1]),
        ball_velocity_x_mps=0.0,
        ball_velocity_y_mps=0.0,
        ball_launch_delay_sec=0.0,
        ball_ground_friction=ball_ground_friction,
        target_y_m=0.0,
        target_z_m=0.20,
    )
    shooter_scenario = replace(
        scenario,
        scenario_id="goalforge-coupled-g1-b-fast-high-finish",
        ball_x_m=1.25,
        ball_y_m=0.0,
        ball_velocity_x_mps=-0.59,
        ball_velocity_y_mps=0.0,
        ball_launch_delay_sec=0.0,
        ball_ground_friction=ball_ground_friction,
        # GoalForgeScenario v1 caps only the recovery controller's context at
        # |y|<=1.2 m and z<=1.2 m.  The separately bound stadium target can
        # extend to the complete 3 m goal mouth.
        target_y_m=float(np.clip(shooter_target[1], -1.20, 1.20)),
        target_z_m=min(shooter_target[2], 1.20),
    )
    passer_recovery = _build_recovery_controller(
        qualification, passer_scenario, passer_recovery_config
    )
    shooter_recovery = _build_recovery_controller(
        qualification, shooter_scenario, shooter_recovery_config
    )
    second_striker_recovery = (
        None
        if physical_second_striker_config is None
        else _build_recovery_controller(
            qualification,
            replace(
                shooter_scenario,
                scenario_id="goalforge-fourth-g1-physical-second-strike",
                ball_velocity_x_mps=0.0,
                ball_velocity_y_mps=0.0,
            ),
            shooter_recovery_config,
        )
    )
    passer_recovery.reset()
    shooter_recovery.reset()
    if second_striker_recovery is not None:
        second_striker_recovery.reset()
    if shooter_recovery_candidate_path is not None:
        shooter_recovery_actor: NumpyIQLActor | SupportBoundIQLResidualActor | None
        if shooter_recovery_residual_config is None:
            shooter_recovery_actor = NumpyIQLActor.load(shooter_recovery_candidate_path)
        else:
            shooter_recovery_actor = SupportBoundIQLResidualActor.load(
                shooter_recovery_candidate_path,
                shooter_recovery_residual_config,
            )
    else:
        shooter_recovery_actor = None
    shooter = _make_robot(
        model=model,
        data=data,
        role="shooter",
        prefix="",
        origin=active_shooter_origin,
        yaw=0.0,
        state_type=state_type,
        output_type=output_type,
        policy_type=policy_type,
        motion_joint_order=mujoco_to_isaac,
        parameters=shooter_parameters,
        start_sec=shooter_start_sec,
        initial_position=initial_position,
        initial_quaternion=initial_quaternion,
        initial_joints=standby_target,
        target_local=np.asarray(active_policy_target, dtype=np.float32),
        phase_hold_frames=0,
        standby_target=standby_target,
        standby_kp=standby_kp,
        standby_kd=standby_kd,
        use_locomotion_standby=True,
        recovery_controller=shooter_recovery,
        post_policy_frame=shooter_post_policy_frame,
        post_policy_blend_frames=shooter_post_policy_blend_frames,
        phase_sync_enabled=(receiver_phase_sync_enabled and shooter_transition_actor is None),
        recovery_torque_actor=shooter_recovery_actor,
        joint_guard_enabled=shooter_joint_guard_enabled,
        post_policy_neutral_velocity_enabled=shooter_post_policy_neutral_velocity_enabled,
        post_policy_forward_velocity_mps=shooter_post_policy_forward_velocity_mps,
        joint_guard_config=shooter_joint_guard_config or G1JointGuardConfig(),
        joint_guard_late_config=shooter_joint_guard_late_config,
        post_policy_recovery_enabled=shooter_post_policy_recovery_enabled,
        early_arrival_parameters=shooter_early_arrival_parameters,
        motion_prior=shooter_motion_prior,
        motion_prior_position_blend=shooter_motion_prior_position_blend,
        motion_prior_velocity_blend=shooter_motion_prior_velocity_blend,
        motion_prior_strike_leg_scale=shooter_motion_prior_strike_leg_scale,
        motion_prior_joint_scales=shooter_motion_prior_joint_scales,
        motion_prior_velocity_joint_scales=velocity_joint_scales,
        motion_prior_contact_policy_frame=shooter_motion_prior_contact_policy_frame,
        agility_prior=shooter_agility_prior,
        agility_prior_position_blend=shooter_agility_prior_position_blend,
        agility_prior_velocity_blend=shooter_agility_prior_velocity_blend,
        agility_prior_contact_policy_frame=shooter_agility_prior_contact_policy_frame,
        agility_prior_joint_scales=shooter_agility_prior_joint_scales,
        contact_prior=shooter_contact_prior,
        contact_prior_position_blend=shooter_contact_prior_position_blend,
        contact_prior_velocity_blend=shooter_contact_prior_velocity_blend,
        contact_prior_contact_policy_frame=shooter_contact_prior_contact_policy_frame,
        contact_prior_joint_scales=shooter_contact_prior_joint_scales,
    )
    shooter.transition_actor = shooter_transition_actor
    shooter.causal_strike_option = (
        None
        if shooter_causal_strike_option_config is None
        else G1CausalStrikeOptionController(shooter_causal_strike_option_config)
    )
    shooter.runtime_strike_router = active_runtime_strike_router
    shooter.runtime_receive_actor = shooter_runtime_receive_actor
    shooter.runtime_receive_probe_action = shooter_runtime_receive_probe_action
    shooter.runtime_contact_target_actor = shooter_runtime_contact_target_actor
    shooter.runtime_finish_plan_actor = shooter_runtime_finish_plan_actor
    shooter.runtime_finish_plan_decision = prepared_finish_plan_decision
    if prepared_finish_plan_decision is not None:
        shooter.runtime_finish_plan_time_sec = 0.0
        prepared_action = prepared_finish_plan_decision.action
        shooter.runtime_receive_decision = RuntimeReceiveDecision(
            accepted=prepared_finish_plan_decision.accepted,
            route=prepared_finish_plan_decision.route,
            confidence=prepared_finish_plan_decision.confidence,
            nearest_success_distance=(prepared_finish_plan_decision.nearest_success_distance),
            nearest_same_action_failure_distance=(
                prepared_finish_plan_decision.nearest_same_action_failure_distance
            ),
            selected_context_hash=prepared_finish_plan_decision.selected_context_hash,
            action=None if prepared_action is None else prepared_action.receive,
            actor_hash=prepared_finish_plan_decision.actor_hash,
        )
        shooter.runtime_receive_time_sec = 0.0
        shooter.runtime_contact_target_decision = RuntimeContactTargetDecision(
            accepted=prepared_finish_plan_decision.accepted,
            route=prepared_finish_plan_decision.route,
            confidence=prepared_finish_plan_decision.confidence,
            nearest_success_distance=(prepared_finish_plan_decision.nearest_success_distance),
            nearest_same_action_failure_distance=(
                prepared_finish_plan_decision.nearest_same_action_failure_distance
            ),
            selected_context_hash=prepared_finish_plan_decision.selected_context_hash,
            action=None if prepared_action is None else prepared_action.target,
            actor_hash=prepared_finish_plan_decision.actor_hash,
        )
        shooter.runtime_contact_target_time_sec = 0.0
    if shooter.runtime_contact_target_actor is not None:
        required = shooter.runtime_contact_target_actor.required_receive_action
        configured = shooter_causal_strike_option_config
        if (
            configured is None
            or shooter_neural_contact_policy_frame != required.contact_policy_frame
            or configured.maximum_arrival_advance_frames != required.maximum_arrival_advance_frames
            or not math.isclose(
                configured.arrival_alignment_tolerance_sec,
                required.arrival_alignment_tolerance_sec,
                abs_tol=1.0e-12,
            )
            or not math.isclose(
                shooter.parameters.stance_offset_x,
                required.stance_offset_x_m,
                abs_tol=1.0e-12,
            )
            or not math.isclose(
                shooter.parameters.stance_offset_y,
                required.stance_offset_y_m,
                abs_tol=1.0e-12,
            )
            or not math.isclose(
                shooter.parameters.foot_yaw_offset,
                required.foot_yaw_offset_rad,
                abs_tol=1.0e-12,
            )
            or not math.isclose(
                shooter.parameters.foot_pitch_offset,
                required.foot_pitch_offset_rad,
                abs_tol=1.0e-12,
            )
        ):
            raise ValueError("runtime contact target actor RECEIVE-law commitment changed")
    if (
        shooter.runtime_strike_router is not None
        or shooter.runtime_receive_actor is not None
        or shooter.runtime_receive_probe_action is not None
        or shooter.runtime_contact_target_actor is not None
    ):
        if shooter.causal_strike_option is None:
            raise RuntimeError("runtime receive/strike actor initialized without causal option")
        shooter.causal_strike_option.arm_runtime_route()
    passer = _make_robot(
        model=model,
        data=data,
        role="passer",
        prefix="passer_",
        origin=active_passer_origin,
        yaw=passer_yaw_rad,
        state_type=state_type,
        output_type=output_type,
        policy_type=policy_type,
        motion_joint_order=mujoco_to_isaac,
        parameters=passer_parameters,
        start_sec=passer_start_sec,
        initial_position=initial_position,
        initial_quaternion=initial_quaternion,
        initial_joints=initial_joints,
        target_local=np.asarray(active_passer_policy_target, dtype=np.float32),
        phase_hold_frames=0,
        standby_target=None,
        standby_kp=None,
        standby_kd=None,
        use_locomotion_standby=True,
        recovery_controller=passer_recovery,
        post_policy_frame=passer_post_policy_frame,
        post_policy_blend_frames=passer_post_policy_blend_frames,
        phase_sync_enabled=False,
        recovery_torque_actor=None,
        joint_guard_enabled=passer_joint_guard_enabled,
        post_policy_neutral_velocity_enabled=passer_post_policy_neutral_velocity_enabled,
        post_policy_forward_velocity_mps=0.0,
        joint_guard_config=passer_joint_guard_config or G1JointGuardConfig(),
        joint_guard_late_config=None,
        post_policy_recovery_enabled=passer_post_policy_recovery_enabled,
        early_arrival_parameters=None,
        motion_prior=None,
        motion_prior_position_blend=0.0,
        motion_prior_velocity_blend=0.0,
        motion_prior_strike_leg_scale=1.0,
        motion_prior_joint_scales=(1.0,) * 29,
        motion_prior_velocity_joint_scales=(1.0,) * 29,
        motion_prior_contact_policy_frame=253,
        contact_prior=None,
        contact_prior_position_blend=0.0,
        contact_prior_velocity_blend=0.0,
        contact_prior_contact_policy_frame=253,
        contact_prior_joint_scales=(1.0,) * 6,
    )
    goalkeeper: _Robot | None = None
    goalkeeper_observer: GoalkeeperActorObserver | None = None
    goalkeeper_actor: NumpyGoalkeeperActor | None = None
    goalkeeper_reference_actor: NumpyHumanoidGoalkeeperReferenceActor | None = None
    goalkeeper_overhead_reach_prior: G1MosaicOverheadReachPrior | None = None
    goalkeeper_whole_body_reach_atlas: G1WholeBodyReachAtlas | None = None
    goalkeeper_mosaic_gmt_controller: MosaicGMTTorchController | None = None
    goalkeeper_mosaic_gmt_contract: MosaicGMTContract | None = None
    goalkeeper_mosaic_gmt_skill: G1MosaicGMTOverheadSkill | None = None
    goalkeeper_balanced_dive_seed: GoalkeeperBalancedDiveSeed | None = None
    goalkeeper_balanced_dive_kp: NDArray[np.float64] | None = None
    goalkeeper_balanced_dive_kd: NDArray[np.float64] | None = None
    goalkeeper_dive_athlete_torch: Any | None = None
    goalkeeper_dive_athlete_model: Any | None = None
    goalkeeper_dive_athlete_checkpoint: dict[str, Any] | None = None
    goalkeeper_dive_athlete_checkpoint_hash: str | None = None
    goalkeeper_recovery_athlete_torch: Any | None = None
    goalkeeper_recovery_athlete_model: Any | None = None
    goalkeeper_recovery_athlete_checkpoint: dict[str, Any] | None = None
    goalkeeper_recovery_athlete_checkpoint_hash: str | None = None
    goalkeeper_origin: np.ndarray | None = None
    if goalkeeper_config is not None:
        goalkeeper_origin = np.asarray(
            (
                (
                    active_goal.plane_x_m - goalkeeper_config.depth_from_goal_line_m,
                    goalkeeper_config.initial_lateral_position_m,
                    0.0,
                )
                if active_goalkeeper_origin_override is None
                else active_goalkeeper_origin_override
            ),
            dtype=np.float64,
        )
        goalkeeper_initial_position, goalkeeper_initial_quaternion = _goalkeeper_neutral_root_pose(
            initial_position
        )
        goalkeeper = _make_robot(
            model=model,
            data=data,
            role="goalkeeper",
            prefix="goalkeeper_",
            origin=goalkeeper_origin,
            yaw=math.pi,
            state_type=state_type,
            output_type=output_type,
            policy_type=policy_type,
            motion_joint_order=mujoco_to_isaac,
            parameters=ShotParameters(policy_type="parameter"),
            start_sec=math.inf,
            # The free-kick clip starts with a ~73-degree global yaw because
            # it was authored for the shooter.  Reusing that global root pose
            # here silently rotates every goalkeeper-local locomotion command
            # into a fore/aft drift.  A goalkeeper uses the clip's qualified
            # pelvis height and joint convention, but owns an independent
            # upright goal-facing root pose.
            initial_position=goalkeeper_initial_position,
            initial_quaternion=goalkeeper_initial_quaternion,
            initial_joints=standby_target,
            target_local=np.asarray((-5.0, 0.0, 0.2), dtype=np.float32),
            phase_hold_frames=0,
            standby_target=standby_target,
            standby_kp=standby_kp,
            standby_kd=standby_kd,
            use_locomotion_standby=True,
            recovery_controller=None,
            post_policy_frame=None,
            post_policy_blend_frames=0,
            phase_sync_enabled=False,
            recovery_torque_actor=None,
            joint_guard_enabled=True,
            post_policy_neutral_velocity_enabled=False,
            post_policy_forward_velocity_mps=0.0,
            joint_guard_config=G1JointGuardConfig(
                margin_rad=0.08,
                prediction_horizon_sec=0.16,
                boundary_kp=140.0,
                boundary_kd=12.0,
            ),
            joint_guard_late_config=G1JointGuardConfig(
                margin_rad=goalkeeper_config.joint_guard_margin_rad,
                prediction_horizon_sec=goalkeeper_config.joint_guard_prediction_horizon_sec,
                boundary_kp=goalkeeper_config.joint_guard_boundary_kp,
                boundary_kd=goalkeeper_config.joint_guard_boundary_kd,
            ),
            post_policy_recovery_enabled=False,
            early_arrival_parameters=None,
            motion_prior=None,
            motion_prior_position_blend=0.0,
            motion_prior_velocity_blend=0.0,
            motion_prior_strike_leg_scale=1.0,
            motion_prior_joint_scales=(1.0,) * 29,
            motion_prior_velocity_joint_scales=(1.0,) * 29,
            motion_prior_contact_policy_frame=253,
            contact_prior=None,
            contact_prior_position_blend=0.0,
            contact_prior_velocity_blend=0.0,
            contact_prior_contact_policy_frame=253,
            contact_prior_joint_scales=(1.0,) * 6,
        )
        if goalkeeper_config.actor_observation_mode == "visible_ball_history_v3":
            goalkeeper_observer = GoalkeeperActorObserver(
                control_dt_sec=_CONTROL_DT,
                flight_velocity_threshold_mps=(
                    goalkeeper_config.actor_minimum_incoming_ball_speed_mps
                ),
            )
        if goalkeeper_config.actor_artifact_path is not None:
            artifact = load_goalkeeper_actor_artifact(goalkeeper_config.actor_artifact_path)
            if artifact.body_hash != qualification.body_hash:
                raise ValueError("goalkeeper actor artifact Body hash does not match coupled G1")
            if (
                goalkeeper_observer is None
                or artifact.actor_observation_contract_hash
                != goalkeeper_observer.spec.actor_contract_hash
            ):
                raise ValueError("goalkeeper actor artifact observation contract changed")
            goalkeeper_actor = NumpyGoalkeeperActor(artifact)
        if goalkeeper_config.external_reference_manifest_path is not None:
            goalkeeper_reference_actor = load_humanoid_goalkeeper_reference_actor(
                goalkeeper_config.external_reference_manifest_path
            )
        if goalkeeper_config.overhead_reach_prior_path is not None:
            goalkeeper_overhead_reach_prior = load_g1_mosaic_overhead_reach_prior(
                goalkeeper_config.overhead_reach_prior_path
            )
            source_scene = root / "g1_description" / "scene_with_ball.xml"
            if (
                goalkeeper_overhead_reach_prior.body_hash != qualification.body_hash
                or goalkeeper_overhead_reach_prior.physics_scene_hash
                != hash_bytes(source_scene.read_bytes())
            ):
                raise ValueError("goalkeeper overhead reach prior does not match coupled G1")
        if goalkeeper_config.whole_body_reach_atlas_path is not None:
            goalkeeper_whole_body_reach_atlas = load_g1_whole_body_reach_atlas(
                goalkeeper_config.whole_body_reach_atlas_path
            )
            if goalkeeper_whole_body_reach_atlas.body_hash != qualification.body_hash:
                raise ValueError("goalkeeper whole-body reach atlas does not match coupled G1")
        if (
            goalkeeper_config.mosaic_gmt_model_path is not None
            and goalkeeper_config.mosaic_gmt_skill_path is not None
        ):
            import torch

            gmt_policy, goalkeeper_mosaic_gmt_contract = load_mosaic_gmt_torch(
                goalkeeper_config.mosaic_gmt_model_path,
                device=torch.device("cpu"),
            )
            goalkeeper_mosaic_gmt_skill = load_g1_mosaic_gmt_overhead_skill(
                goalkeeper_config.mosaic_gmt_skill_path
            )
            goalkeeper_mosaic_gmt_controller = MosaicGMTTorchController(
                policy=gmt_policy,
                contract=goalkeeper_mosaic_gmt_contract,
                skill=goalkeeper_mosaic_gmt_skill,
                environment_count=1,
                device=torch.device("cpu"),
            )
        if goalkeeper_config.balanced_dive_source_checkout is not None:
            goalkeeper_balanced_dive_seed = build_balanced_dive_imitation_seed(
                load_official_goalkeeper_dive_atlas(
                    checkout=goalkeeper_config.balanced_dive_source_checkout
                )
            )
            (
                goalkeeper_balanced_dive_kp,
                goalkeeper_balanced_dive_kd,
            ) = balanced_dive_qualified_impedance()
        if (
            goalkeeper_config.dive_athlete_checkpoint_path is not None
            and goalkeeper_config.dive_athlete_exam_path is not None
        ):
            import torch

            from rosclaw_soccer.training.dive_athlete_cpu_exam import (
                validate_dive_athlete_cpu_exam_report,
            )
            from rosclaw_soccer.training.dive_athlete_expert import (
                load_dive_athlete_expert,
            )

            athlete_checkpoint_path = goalkeeper_config.dive_athlete_checkpoint_path
            athlete_source_checkout = goalkeeper_config.balanced_dive_source_checkout
            if athlete_source_checkout is None:
                raise ValueError("goalkeeper dive athlete source is unavailable")
            athlete_exam = validate_dive_athlete_cpu_exam_report(
                goalkeeper_config.dive_athlete_exam_path
            )
            goalkeeper_dive_athlete_checkpoint_hash = hash_bytes(
                athlete_checkpoint_path.read_bytes()
            )
            if athlete_exam.get("checkpoint_hash") != goalkeeper_dive_athlete_checkpoint_hash:
                raise ValueError("goalkeeper dive athlete checkpoint/exam binding changed")
            (
                goalkeeper_dive_athlete_model,
                goalkeeper_dive_athlete_checkpoint,
            ) = load_dive_athlete_expert(
                checkpoint_path=athlete_checkpoint_path,
                asset_root=root,
                dive_source_checkout=athlete_source_checkout,
                device=torch.device("cpu"),
            )
            goalkeeper_dive_athlete_torch = torch
        if (
            goalkeeper_config.recovery_athlete_checkpoint_path is not None
            and goalkeeper_config.recovery_athlete_exam_path is not None
        ):
            import torch

            from rosclaw_soccer.training.recovery_athlete_cpu_exam import (
                validate_recovery_athlete_cpu_exam,
            )
            from rosclaw_soccer.training.recovery_athlete_student import (
                load_recovery_athlete_student,
            )

            recovery_checkpoint_path = goalkeeper_config.recovery_athlete_checkpoint_path
            recovery_exam = validate_recovery_athlete_cpu_exam(
                goalkeeper_config.recovery_athlete_exam_path
            )
            goalkeeper_recovery_athlete_checkpoint_hash = hash_bytes(
                recovery_checkpoint_path.read_bytes()
            )
            if recovery_exam.get("checkpoint_hash") != goalkeeper_recovery_athlete_checkpoint_hash:
                raise ValueError("goalkeeper recovery athlete checkpoint/exam binding changed")
            locomotion_policy_path = root / "policy" / "loco_mode" / "model" / "policy_29dof.pt"
            (
                goalkeeper_recovery_athlete_model,
                goalkeeper_recovery_athlete_checkpoint,
            ) = load_recovery_athlete_student(
                checkpoint_path=recovery_checkpoint_path,
                locomotion_policy_path=locomotion_policy_path,
                device=torch.device("cpu"),
            )
            goalkeeper_recovery_athlete_torch = torch
    second_striker: _Robot | None = None
    if physical_second_striker_config is not None:
        second_striker = _make_robot(
            model=model,
            data=data,
            role="second_striker",
            prefix="second_striker_",
            origin=np.asarray(physical_second_striker_config.origin_m, dtype=np.float64),
            yaw=0.0,
            state_type=state_type,
            output_type=output_type,
            policy_type=policy_type,
            motion_joint_order=mujoco_to_isaac,
            parameters=cast(ShotParameters, second_striker_parameters),
            start_sec=physical_second_striker_config.policy_start_time_sec,
            initial_position=initial_position,
            initial_quaternion=initial_quaternion,
            initial_joints=standby_target,
            target_local=np.asarray(
                physical_second_striker_config.policy_target_m,
                dtype=np.float32,
            ),
            phase_hold_frames=0,
            standby_target=standby_target,
            standby_kp=standby_kp,
            standby_kd=standby_kd,
            use_locomotion_standby=True,
            recovery_controller=second_striker_recovery,
            post_policy_frame=physical_second_striker_config.post_policy_frame,
            post_policy_blend_frames=(physical_second_striker_config.post_policy_blend_frames),
            phase_sync_enabled=False,
            recovery_torque_actor=None,
            joint_guard_enabled=True,
            post_policy_neutral_velocity_enabled=False,
            post_policy_forward_velocity_mps=0.0,
            joint_guard_config=shooter_joint_guard_config or G1JointGuardConfig(),
            joint_guard_late_config=shooter_joint_guard_late_config,
            post_policy_recovery_enabled=True,
            early_arrival_parameters=shooter_early_arrival_parameters,
            motion_prior=None,
            motion_prior_position_blend=0.0,
            motion_prior_velocity_blend=0.0,
            motion_prior_strike_leg_scale=1.0,
            motion_prior_joint_scales=(1.0,) * 29,
            motion_prior_velocity_joint_scales=(1.0,) * 29,
            motion_prior_contact_policy_frame=253,
            contact_prior=None,
            contact_prior_position_blend=0.0,
            contact_prior_velocity_blend=0.0,
            contact_prior_contact_policy_frame=253,
            contact_prior_joint_scales=(1.0,) * 6,
        )
    robots = tuple(
        robot for robot in (passer, shooter, goalkeeper, second_striker) if robot is not None
    )
    passer_reactive_actor: RouteActor | None = None
    goalkeeper_reactive_actor: RouteActor | None = None
    if passer_reactive_movement_config is not None:
        passer_reactive_actor = load_route_actor(
            Path(passer_reactive_movement_config.actor_artifact_path)
        )
        if passer_reactive_actor.actor_hash != passer_reactive_movement_config.actor_hash:
            raise ValueError("passer reactive route actor changed after plan construction")
    if goalkeeper_reactive_movement_config is not None:
        goalkeeper_reactive_actor = load_route_actor(
            Path(goalkeeper_reactive_movement_config.actor_artifact_path)
        )
        if goalkeeper_reactive_actor.actor_hash != goalkeeper_reactive_movement_config.actor_hash:
            raise ValueError("goalkeeper reactive route actor changed after plan construction")
    passer_geoms = _robot_geom_ids(model, passer.pelvis_body)
    shooter_geoms = _robot_geom_ids(model, shooter.pelvis_body)
    goalkeeper_geoms = (
        frozenset() if goalkeeper is None else _robot_geom_ids(model, goalkeeper.pelvis_body)
    )
    goalkeeper_left_glove_geoms: frozenset[int] = frozenset()
    goalkeeper_right_glove_geoms: frozenset[int] = frozenset()
    second_striker_geoms = (
        frozenset()
        if second_striker is None
        else _robot_geom_ids(model, second_striker.pelvis_body)
    )
    if not passer_collision_enabled:
        passer_geom_indices = np.asarray(sorted(passer_geoms), dtype=np.int64)
        # Counterfactual removal keeps the focal body numerically supported by
        # the floor but removes its coupling to the ball and other players.
        # Let category 2 mean "ablated focal body" and add that category only
        # to the floor.  Zeroing all masks would make the body free-fall and
        # turn a bounded causal replay into a poorly conditioned simulation.
        model.geom_contype[passer_geom_indices] = 2
        model.geom_conaffinity[passer_geom_indices] = 2
        model.geom_contype[floor_geom] = int(model.geom_contype[floor_geom]) | 2
        model.geom_conaffinity[floor_geom] = int(model.geom_conaffinity[floor_geom]) | 2
    if goalkeeper is not None and goalkeeper_config is not None:
        goalkeeper_gloves = _goalkeeper_glove_geoms(
            model=model,
            goalkeeper=goalkeeper,
        )
        goalkeeper_geoms = goalkeeper_geoms | goalkeeper_gloves
        goalkeeper_left_glove_geoms = frozenset(
            geom
            for geom in goalkeeper_gloves
            if model.geom(geom).name == "goalkeeper_left_goalkeeper_glove"
        )
        goalkeeper_right_glove_geoms = frozenset(
            geom
            for geom in goalkeeper_gloves
            if model.geom(geom).name == "goalkeeper_right_goalkeeper_glove"
        )
    # One immutable shared ball.  This is the pass scenario's local initial
    # state transformed into the coupled world; there is no later teleport.
    data.qpos[ball_qpos : ball_qpos + 3] = (
        direct_shot_position
        if direct_shot_position is not None
        else (
            active_passer_origin
            + _rotate_z(
                np.asarray((*active_passer_ball_xy, active_goal.ball_radius_m), dtype=np.float64),
                passer_yaw_rad,
            )
            if launcher_position is None
            else launcher_position
        )
    )
    data.qpos[ball_qpos + 3 : ball_qpos + 7] = (1.0, 0.0, 0.0, 0.0)
    data.qvel[ball_qvel : ball_qvel + 6] = 0.0
    if launcher_velocity is not None:
        data.qvel[ball_qvel : ball_qvel + 3] = launcher_velocity
    mujoco.mj_forward(model, data)

    for robot in robots:
        if robot is second_striker:
            assert second_ball_body is not None and second_ball_qvel is not None
            _fill_local_state(robot, data, second_ball_body, second_ball_qvel)
        else:
            _fill_local_state(robot, data, ball_body, ball_qvel)
    if launcher_position is None and direct_shot_position is None and passer_start_sec <= 1.0e-12:
        _enter_policy(passer)

    hard_limits = np.asarray(G1_HARD_TORQUE_LIMITS, dtype=np.float64)
    guarded_limits = hard_limits * 0.85
    total_frames = int(round(simulation_duration_sec / _CONTROL_DT))
    trace: dict[str, list[Any]] = {
        "time": [],
        "ball_pose": [],
        "ball_velocity": [],
        "passer_pelvis_pose": [],
        "shooter_pelvis_pose": [],
        "passer_torso_quaternion": [],
        "shooter_torso_quaternion": [],
        "passer_joint_position": [],
        "shooter_joint_position": [],
        "passer_joint_velocity": [],
        "shooter_joint_velocity": [],
        "passer_joint_torque": [],
        "shooter_joint_torque": [],
        "passer_commanded_torque": [],
        "shooter_commanded_torque": [],
        "passer_safety_projected_torque": [],
        "shooter_safety_projected_torque": [],
        "passer_executed_torque": [],
        "shooter_executed_torque": [],
        "passer_policy_action": [],
        "shooter_policy_action": [],
        "shooter_target_velocity": [],
        "shooter_motion_prior_target_delta": [],
        "shooter_motion_prior_velocity_delta": [],
        "shooter_motion_prior_position_active": [],
        "shooter_motion_prior_velocity_active": [],
        "shooter_agility_prior_velocity_delta": [],
        "shooter_agility_prior_target_delta": [],
        "shooter_agility_prior_active": [],
        "shooter_contact_prior_target_delta": [],
        "shooter_contact_prior_velocity_delta": [],
        "shooter_contact_prior_active": [],
        "passer_com_position": [],
        "shooter_com_position": [],
        "passer_left_foot_position": [],
        "passer_right_foot_position": [],
        "shooter_left_foot_position": [],
        "shooter_right_foot_position": [],
        "passer_support_foot_slip": [],
        "shooter_support_foot_slip": [],
        "passer_contact_impulse": [],
        "shooter_contact_impulse": [],
        "shooter_learned_torque_active": [],
        "passer_joint_guard_active": [],
        "shooter_joint_guard_active": [],
        "passer_post_policy_blend_fraction": [],
        "shooter_post_policy_blend_fraction": [],
        "passer_recovery_active": [],
        "shooter_recovery_active": [],
        "passer_recovery_blend_fraction": [],
        "shooter_recovery_blend_fraction": [],
        "passer_policy_frame": [],
        "shooter_policy_frame": [],
        "shooter_phase_correction": [],
        "shooter_transition_features": [],
        "shooter_transition_actor_accepted": [],
        "shooter_transition_support_distance": [],
        "shooter_transition_trigger_policy_frame": [],
        "shooter_transition_residual_frames": [],
        "shooter_transition_predicted_safe_probability": [],
        "shooter_transition_predicted_chain_probability": [],
        "shooter_transition_ensemble_probability_spread": [],
        "shooter_transition_used_parent_fallback": [],
        "shooter_transition_triggered": [],
        "shooter_causal_strike_option_phase": [],
        "shooter_causal_strike_option_ready": [],
        "shooter_causal_strike_option_begin_bridge": [],
        "shooter_causal_strike_option_incoming_ball": [],
        "shooter_causal_strike_option_ball_arrival_eta_sec": [],
        "shooter_causal_strike_option_incoming_observation_count": [],
        "shooter_causal_strike_selected_phase_start_frame": [],
        "shooter_causal_strike_bridge_fraction": [],
        "shooter_ball_local_position": [],
        "shooter_ball_local_velocity": [],
        "shooter_pelvis_local_position": [],
        "shooter_runtime_strike_features": [],
        "shooter_runtime_strike_route_decided": [],
        "shooter_runtime_strike_route_accepted": [],
        "shooter_runtime_strike_route_support_distance": [],
        "shooter_runtime_strike_route_advance_frames": [],
        "shooter_runtime_receive_features": [],
        "shooter_runtime_receive_decided": [],
        "shooter_runtime_receive_accepted": [],
        "shooter_runtime_receive_support_distance": [],
        "shooter_runtime_receive_alignment_tolerance_sec": [],
        "shooter_runtime_receive_stance_offset_y_m": [],
        "shooter_runtime_receive_foot_yaw_offset_rad": [],
        "shooter_runtime_contact_target_decided": [],
        "shooter_runtime_contact_target_accepted": [],
        "shooter_runtime_contact_target_support_distance": [],
        "shooter_runtime_contact_target_velocity_xyz_mps": [],
        "shooter_ballistic_actor_active": [],
        "shooter_ballistic_actor_torque": [],
        "shooter_three_axis_contact_actor_active": [],
        "shooter_three_axis_contact_actor_torque": [],
        "shooter_three_axis_contact_actor_force_xyz_n": [],
        "shooter_three_axis_contact_actor_foot_velocity_xyz_mps": [],
        "shooter_target_velocity_contact_actor_active": [],
        "shooter_target_velocity_contact_actor_torque": [],
        "shooter_target_velocity_contact_actor_force_xyz_n": [],
        "shooter_target_velocity_contact_actor_foot_velocity_xyz_mps": [],
        "shooter_target_velocity_contact_actor_target_xyz_mps": [],
        "shooter_target_velocity_contact_actor_target_supported": [],
        "shooter_neural_contact_actor_active": [],
        "shooter_neural_contact_actor_torque": [],
        "shooter_neural_contact_actor_supported": [],
        "shooter_neural_contact_actor_ood_distance": [],
        "shooter_ballistic_contact_active": [],
        "shooter_ballistic_contact_target_delta": [],
        "shooter_ballistic_contact_torque_active": [],
        "shooter_ballistic_contact_torque": [],
        "shooter_first_touch_interception_active": [],
        "shooter_first_touch_interception_torque": [],
        "shooter_first_touch_interception_force": [],
        "shooter_first_touch_interception_error": [],
        "passer_reception_interception_active": [],
        "passer_reception_interception_torque": [],
        "passer_reception_interception_force": [],
        "passer_reception_interception_error": [],
        "shooter_loft_teacher_active": [],
        "shooter_loft_teacher_torque": [],
        "shooter_loft_teacher_force_xyz_n": [],
        "shooter_loft_teacher_foot_velocity_xyz_mps": [],
        "passer_foot_contact": [],
        "shooter_foot_contact": [],
        # -1 = anatomical left foot, +1 = anatomical right foot, 0 = no
        # foot-ball contact in this control frame.
        "shooter_ball_contact_foot": [],
        "ball_contact_role": [],
        "second_threat_rearmed": [],
        "second_threat_launcher_active": [],
        "second_threat_launcher_force": [],
        "robot_robot_contact_count": [],
    }
    if passer_tactical_movement_config is not None or passer_reactive_movement_config is not None:
        trace.update(
            {
                "passer_tactical_world_target": [],
                "passer_tactical_world_command": [],
                "passer_tactical_movement_active": [],
            }
        )
    if passer_reactive_movement_config is not None:
        trace.update(
            {
                "passer_reactive_route_features": [],
                "passer_reactive_route_support_distance": [],
                "passer_reactive_route_accepted": [],
                "passer_reactive_role_separation_m": [],
                "passer_reactive_collision_shield_active": [],
                "passer_reactive_velocity_braking_correction": [],
            }
        )
    if (
        goalkeeper_tactical_movement_config is not None
        or goalkeeper_reactive_movement_config is not None
    ):
        trace.update(
            {
                "goalkeeper_tactical_world_target": [],
                "goalkeeper_tactical_world_command": [],
                "goalkeeper_tactical_movement_active": [],
            }
        )
    if goalkeeper_reactive_movement_config is not None:
        trace.update(
            {
                "goalkeeper_reactive_route_features": [],
                "goalkeeper_reactive_route_support_distance": [],
                "goalkeeper_reactive_route_accepted": [],
                "goalkeeper_reactive_role_separation_m": [],
                "goalkeeper_reactive_collision_shield_active": [],
                "goalkeeper_reactive_velocity_braking_correction": [],
            }
        )
    if second_striker is not None:
        trace.update(
            {
                "second_ball_pose": [],
                "second_ball_velocity": [],
                "second_striker_pelvis_pose": [],
                "second_striker_left_foot_position": [],
                "second_striker_right_foot_position": [],
                "second_striker_joint_position": [],
                "second_striker_joint_velocity": [],
                "second_striker_commanded_torque": [],
                "second_striker_safety_projected_torque": [],
                "second_striker_executed_torque": [],
                "second_striker_policy_action": [],
                "second_striker_policy_frame": [],
                "second_striker_foot_contact": [],
                "second_striker_contact_force_n": [],
                "second_striker_ballistic_actor_active": [],
                "second_striker_ballistic_actor_torque": [],
                "second_striker_ballistic_actor_force_yz_n": [],
                "second_striker_ballistic_actor_foot_velocity_yz_mps": [],
                "second_striker_ballistic_actor_foot_ball_distance_m": [],
                "second_striker_ballistic_actor_desired_launch_velocity_yz_mps": [],
                "second_striker_ballistic_actor_target_conditioned": [],
                "second_striker_ballistic_actor_launch_envelope_supported": [],
                "second_striker_ballistic_actor_candidate_selected": [],
                "second_striker_loft_teacher_active": [],
                "second_striker_loft_teacher_force_yz_n": [],
                "second_striker_loft_teacher_foot_velocity_yz_mps": [],
                "second_striker_ballistic_contact_active": [],
                "second_striker_ballistic_contact_target_delta": [],
                "second_striker_ballistic_contact_torque_active": [],
                "second_striker_ballistic_contact_torque": [],
            }
        )
    if goalkeeper is not None:
        trace.update(
            {
                "goalkeeper_pelvis_pose": [],
                "goalkeeper_root_velocity": [],
                "goalkeeper_torso_quaternion": [],
                "goalkeeper_left_foot_position": [],
                "goalkeeper_right_foot_position": [],
                "goalkeeper_foot_contact": [],
                "goalkeeper_support_foot_slip": [],
                "goalkeeper_left_hand_position": [],
                "goalkeeper_right_hand_position": [],
                "goalkeeper_joint_position": [],
                "goalkeeper_joint_velocity": [],
                "goalkeeper_joint_torque": [],
                "goalkeeper_commanded_torque": [],
                "goalkeeper_safety_projected_torque": [],
                "goalkeeper_executed_torque": [],
                "goalkeeper_policy_action": [],
                "goalkeeper_target_velocity": [],
                "goalkeeper_policy_frame": [],
                "goalkeeper_command_mps": [],
                "goalkeeper_predicted_target_y_m": [],
                "goalkeeper_estimated_ball_velocity_mps": [],
                "goalkeeper_estimated_intercept": [],
                "goalkeeper_actor_height_routed": [],
                "goalkeeper_mosaic_gmt_central_routed": [],
                "goalkeeper_observed_flight_active": [],
                "goalkeeper_observed_flight_start_sec": [],
                "goalkeeper_reaction_active": [],
                "goalkeeper_useful_reaction_active": [],
                "goalkeeper_anticipation_active": [],
                "goalkeeper_canonical_locomotion_mirror_active": [],
                "goalkeeper_block_action_active": [],
                "goalkeeper_overhead_reach_blend": [],
                "goalkeeper_overhead_reach_target_delta": [],
                "goalkeeper_whole_body_reach_blend": [],
                "goalkeeper_whole_body_reach_target_delta": [],
                "goalkeeper_mosaic_gmt_blend": [],
                "goalkeeper_mosaic_gmt_mirrored": [],
                "goalkeeper_mosaic_gmt_target_delta": [],
                "goalkeeper_balanced_dive_blend": [],
                "goalkeeper_balanced_dive_target_delta": [],
                "goalkeeper_landing_capture_active": [],
                "goalkeeper_landing_capture_blend": [],
                "goalkeeper_proprioceptive_capture_active": [],
                "goalkeeper_recovery_athlete_active": [],
                "goalkeeper_recovery_athlete_suppressed": [],
                "goalkeeper_recovery_athlete_raw_world_command": [],
                "goalkeeper_recovery_athlete_world_command": [],
                "goalkeeper_contact_epoch": [],
                "goalkeeper_second_ball_contact": [],
                "goalkeeper_second_left_glove_contact": [],
                "goalkeeper_second_right_glove_contact": [],
                "goalkeeper_ball_contact": [],
                "goalkeeper_left_glove_contact": [],
                "goalkeeper_right_glove_contact": [],
                "goalkeeper_bimanual_reach_active": [],
                "goalkeeper_bimanual_punch_active": [],
                "goalkeeper_bimanual_punch_torque": [],
            }
        )
    # A separate 500 Hz window preserves the actual impact state for visual
    # replay and geometry audits.  The regular trajectory intentionally stays
    # at the 50 Hz controller rate.
    goalkeeper_contact_trace: dict[str, list[Any]] = {
        "goalkeeper_contact_window_time": [],
        "goalkeeper_contact_window_ball_pose": [],
        "goalkeeper_contact_window_ball_velocity": [],
        "goalkeeper_contact_window_pelvis_pose": [],
        "goalkeeper_contact_window_joint_position": [],
        "goalkeeper_contact_window_left_hand_position": [],
        "goalkeeper_contact_window_right_hand_position": [],
        "goalkeeper_contact_window_left_glove_contact": [],
        "goalkeeper_contact_window_right_glove_contact": [],
        "goalkeeper_contact_window_left_surface_distance_m": [],
        "goalkeeper_contact_window_right_surface_distance_m": [],
    }
    finite = True
    joint_violation = False
    role_joint_violation = {robot.role: False for robot in robots}
    torque_violation = False
    actuator_saturation = False
    robot_robot_contact_count = 0
    passer_min_height = math.inf
    shooter_min_height = math.inf
    passer_roll_peak = 0.0
    passer_pitch_peak = 0.0
    shooter_roll_peak = 0.0
    shooter_pitch_peak = 0.0
    second_striker_min_height = math.inf
    second_striker_contact_force_peak = 0.0
    second_striker_precontact_peak_speed = 0.0
    second_striker_postcontact_peak_speed = 0.0
    second_striker_postcontact_peak_forward_speed = 0.0
    second_striker_contact_foot: str | None = None
    second_striker_unexpected_precontact_collision_geoms: set[str] = set()
    pass_peak_speed = 0.0
    shot_peak_speed = 0.0
    goal_crossed = False
    goal_plane_crossed = False
    crossing_y: float | None = None
    crossing_z: float | None = None
    pass_delivery_position: np.ndarray | None = None
    goalkeeper_contact_time: float | None = None
    goalkeeper_left_glove_contact_time: float | None = None
    goalkeeper_right_glove_contact_time: float | None = None
    goalkeeper_glove_contact_height: float | None = None
    goalkeeper_glove_contact_time: float | None = None
    goalkeeper_glove_contact_position: np.ndarray | None = None
    goalkeeper_glove_contact_surface_distance: float | None = None
    goalkeeper_glove_contact_side: str | None = None
    goalkeeper_contact_left_hand_height: float | None = None
    goalkeeper_contact_right_hand_height: float | None = None
    goalkeeper_contact_left_hand_ball_distance: float | None = None
    goalkeeper_contact_right_hand_ball_distance: float | None = None
    goalkeeper_bimanual_window_observed = False
    goalkeeper_bimanual_reach_frames = 0
    goalkeeper_bimanual_punch_frames = 0
    goalkeeper_bimanual_punch_peak_torque = 0.0
    goalkeeper_min_height = math.inf
    goalkeeper_peak_lateral_speed = 0.0
    goalkeeper_reaction_frames = 0
    goalkeeper_anticipation_frames = 0
    goalkeeper_block_action_frames = 0
    goalkeeper_overhead_reach_frames = 0
    goalkeeper_overhead_reach_peak_blend = 0.0
    goalkeeper_whole_body_reach_frames = 0
    goalkeeper_whole_body_reach_peak_blend = 0.0
    goalkeeper_whole_body_reach_memory = 0.0
    goalkeeper_mosaic_gmt_frames = 0
    goalkeeper_mosaic_gmt_peak_blend = 0.0
    goalkeeper_mosaic_gmt_arrival_time_sec: float | None = None
    goalkeeper_mosaic_gmt_target_height_m = 0.0
    goalkeeper_mosaic_gmt_route: bool | None = None
    goalkeeper_mosaic_gmt_mirror_latch: bool | None = None
    goalkeeper_balanced_dive_frames = 0
    goalkeeper_balanced_dive_peak_blend = 0.0
    goalkeeper_balanced_dive_flight_start_sec: float | None = None
    goalkeeper_balanced_dive_flight_duration_sec: float | None = None
    goalkeeper_balanced_dive_direction_index: int | None = None
    goalkeeper_balanced_dive_anchor_target: NDArray[np.float64] | None = None
    goalkeeper_balanced_dive_was_airborne = False
    goalkeeper_landing_capture_start_sec: float | None = None
    goalkeeper_landing_capture_anchor: NDArray[np.float64] | None = None
    goalkeeper_landing_capture_duration_sec: float | None = None
    goalkeeper_landing_capture_proprioceptive = False
    goalkeeper_initial_y = 0.0 if goalkeeper is None else float(data.qpos[goalkeeper.qpos_base + 1])
    previous_ball_x = float(data.qpos[ball_qpos])
    goal_net_state = G1CompliantGoalNetState()
    previous_second_ball_x = (
        None if second_ball_qpos is None else float(data.qpos[second_ball_qpos])
    )
    second_goal_net_state = G1CompliantGoalNetState()
    second_ball_goal_plane_crossed = False
    second_ball_goal_crossed = False
    second_ball_crossing_y: float | None = None
    second_ball_crossing_z: float | None = None
    goalkeeper_previous_actor_residual: NDArray[np.float64] = np.zeros(29, dtype=np.float64)
    goalkeeper_contact_epoch = 0
    second_threat_rearmed = False
    second_threat_rearm_time: float | None = None
    second_threat_launch_time: float | None = None
    second_threat_launch_position: NDArray[np.float64] | None = None
    second_threat_target_velocity: NDArray[np.float64] | None = None
    second_threat_force: NDArray[np.float64] | None = None
    second_threat_force_stop_sec: float | None = None
    second_threat_peak_force = 0.0
    first_goal_crossed_before_second_threat = False
    goalkeeper_second_contact_time: float | None = None
    goalkeeper_second_glove_contact_time: float | None = None
    goalkeeper_second_glove_contact_height: float | None = None
    goalkeeper_second_glove_contact_position: NDArray[np.float64] | None = None
    goalkeeper_second_glove_contact_surface_distance: float | None = None
    goalkeeper_second_glove_contact_side: str | None = None
    goalkeeper_second_contact_left_hand_height: float | None = None
    goalkeeper_second_contact_right_hand_height: float | None = None
    goalkeeper_second_contact_left_hand_ball_distance: float | None = None
    goalkeeper_second_contact_right_hand_ball_distance: float | None = None
    physical_rearm_earliest = (
        None
        if physical_second_striker_config is None
        else physical_second_striker_config.observer_rearm_earliest_sec
    )

    for frame in range(total_frames):
        for robot in robots:
            if robot is second_striker or (
                robot is goalkeeper and second_striker is not None and second_threat_rearmed
            ):
                assert second_ball_body is not None and second_ball_qvel is not None
                _fill_local_state(robot, data, second_ball_body, second_ball_qvel)
            else:
                _fill_local_state(robot, data, ball_body, ball_qvel)
        shooter.last_transition_features = causal_transition_features(
            receiver_pelvis_world_m=np.asarray(
                data.qpos[shooter.qpos_base : shooter.qpos_base + 3], dtype=np.float64
            ),
            predecessor_pelvis_world_m=np.asarray(
                data.qpos[passer.qpos_base : passer.qpos_base + 3], dtype=np.float64
            ),
            receiver_ball_local_m=np.asarray(shooter.state.ball_pos_w, dtype=np.float64),
            receiver_reception_target_local_m=np.asarray(
                local_pass_reception_target, dtype=np.float64
            ),
            receiver_shot_target_local_m=np.asarray(active_policy_target, dtype=np.float64),
            predecessor_swing_speed_scale=passer.parameters.swing_speed_scale,
            ball_ground_friction=ball_ground_friction,
            predecessor_yaw_rad=passer_yaw_rad,
            receiver_kick_foot=shooter.parameters.kick_foot,
        )
        if shooter.transition_actor is not None and shooter.transition_decision is None:
            shooter.transition_decision = shooter.transition_actor.decide(
                shooter.last_transition_features
            )
        predecessor_frame = max(
            0,
            int(passer.policy.time_step) - int(passer.policy.WARMUP_STEPS),
        )
        if shooter.causal_strike_option is not None:
            pelvis_roll, pelvis_pitch = _roll_pitch(
                np.asarray(shooter.state.pelvis_quat_w, dtype=np.float64)
            )
            option_observation = CausalStrikeOptionObservation(
                timestamp_sec=float(data.time),
                predecessor_policy_frame=predecessor_frame,
                receiver_pelvis_height_m=float(shooter.state.pelvis_pos_w[2]),
                receiver_roll_rad=pelvis_roll,
                receiver_pitch_rad=pelvis_pitch,
                receiver_joint_velocity_rms_rad_s=float(
                    np.sqrt(np.mean(np.square(shooter.state.dq)))
                ),
                receiver_ball_local_x_m=float(shooter.state.ball_pos_w[0]),
                receiver_ball_local_vx_mps=float(shooter.state.ball_vel_w[0]),
            )
            shooter.last_causal_strike_option_decision = shooter.causal_strike_option.step(
                option_observation
            )
            if (
                (
                    shooter.runtime_receive_actor is not None
                    or shooter.runtime_receive_probe_action is not None
                    or shooter.runtime_contact_target_actor is not None
                )
                and shooter.runtime_receive_decision is None
                and shooter.runtime_contact_target_decision is None
                and shooter.causal_strike_option.stable_incoming_observed
            ):
                arrival_eta = shooter.last_causal_strike_option_decision.ball_arrival_eta_sec
                if arrival_eta is None:
                    raise RuntimeError("stable incoming RECEIVE observation has no arrival ETA")
                shooter_policy_frame = max(
                    0,
                    int(shooter.policy.time_step) - int(shooter.policy.WARMUP_STEPS),
                )
                receive_features = runtime_receive_features(
                    ball_local_position_m=shooter.state.ball_pos_w,
                    ball_local_velocity_mps=shooter.state.ball_vel_w,
                    ball_arrival_eta_sec=arrival_eta,
                    pelvis_local_position_m=shooter.state.pelvis_pos_w,
                    joint_velocity_rad_s=shooter.state.dq,
                    policy_frame=shooter_policy_frame,
                )
                shooter.last_runtime_receive_features = np.asarray(
                    receive_features, dtype=np.float64
                )
                if shooter.runtime_contact_target_actor is not None:
                    target_decision = shooter.runtime_contact_target_actor.decide(receive_features)
                    route_latch_open = shooter.causal_strike_option.runtime_route_latch_open
                    if not route_latch_open:
                        target_decision = replace(
                            target_decision,
                            accepted=False,
                            route="CONTACT_TARGET_CAUSAL_WINDOW_CLOSED_FALLBACK",
                            confidence=0.0,
                            selected_context_hash=None,
                            action=None,
                        )
                    shooter.runtime_contact_target_decision = target_decision
                    shooter.runtime_contact_target_time_sec = float(data.time)
                    target_action = target_decision.action
                    if target_action is None:
                        if route_latch_open:
                            shooter.causal_strike_option.reject_runtime_route()
                        else:
                            shooter.causal_strike_option.expire_runtime_route()
                    else:
                        required = shooter.runtime_contact_target_actor.required_receive_action
                        shooter.causal_strike_option.select_arrival_route(
                            required.maximum_arrival_advance_frames,
                            arrival_alignment_tolerance_sec=(
                                required.arrival_alignment_tolerance_sec
                            ),
                        )
                        shooter_neural_contact_target_velocity_xyz_mps = (
                            target_action.target_foot_velocity_xyz_mps
                        )
                else:
                    receive_decision = (
                        shooter.runtime_receive_actor.decide(receive_features)
                        if shooter.runtime_receive_actor is not None
                        else RuntimeReceiveDecision(
                            accepted=True,
                            route="TRAINING_RUNTIME_RECEIVE_INTERVENTION",
                            confidence=1.0,
                            nearest_success_distance=0.0,
                            nearest_same_action_failure_distance=None,
                            selected_context_hash=None,
                            action=shooter.runtime_receive_probe_action,
                            actor_hash=cast(
                                RuntimeReceiveAction, shooter.runtime_receive_probe_action
                            ).action_hash,
                        )
                    )
                    route_latch_open = shooter.causal_strike_option.runtime_route_latch_open
                    if not route_latch_open:
                        receive_decision = replace(
                            receive_decision,
                            accepted=False,
                            route="RUNTIME_RECEIVE_CAUSAL_WINDOW_CLOSED_FALLBACK",
                            confidence=0.0,
                            selected_context_hash=None,
                            action=None,
                        )
                    shooter.runtime_receive_decision = receive_decision
                    shooter.runtime_receive_time_sec = float(data.time)
                    receive_action = receive_decision.action
                    if receive_action is None:
                        if route_latch_open:
                            shooter.causal_strike_option.reject_runtime_route()
                        else:
                            shooter.causal_strike_option.expire_runtime_route()
                    else:
                        shooter.causal_strike_option.select_arrival_route(
                            receive_action.maximum_arrival_advance_frames,
                            arrival_alignment_tolerance_sec=(
                                receive_action.arrival_alignment_tolerance_sec
                            ),
                        )
                        shooter.parameters = replace(
                            shooter.parameters,
                            stance_offset_x=receive_action.stance_offset_x_m,
                            stance_offset_y=receive_action.stance_offset_y_m,
                            foot_yaw_offset=receive_action.foot_yaw_offset_rad,
                            foot_pitch_offset=receive_action.foot_pitch_offset_rad,
                        )
                        shooter_neural_contact_policy_frame = receive_action.contact_policy_frame
            if (
                shooter.runtime_strike_router is not None
                and shooter.runtime_strike_route_decision is None
                and shooter.causal_strike_option.stable_incoming_observed
            ):
                arrival_eta = shooter.last_causal_strike_option_decision.ball_arrival_eta_sec
                if arrival_eta is None:
                    raise RuntimeError("stable incoming route observation has no arrival ETA")
                runtime_features = runtime_causal_strike_features(
                    ball_local_position_m=shooter.state.ball_pos_w,
                    ball_local_velocity_mps=shooter.state.ball_vel_w,
                    ball_arrival_eta_sec=arrival_eta,
                    pelvis_local_position_m=shooter.state.pelvis_pos_w,
                    joint_velocity_rad_s=shooter.state.dq,
                )
                shooter.last_runtime_strike_features = np.asarray(
                    runtime_features, dtype=np.float64
                )
                route_decision = shooter.runtime_strike_router.decide(runtime_features)
                route_latch_open = shooter.causal_strike_option.runtime_route_latch_open
                if not route_latch_open:
                    route_decision = replace(
                        route_decision,
                        accepted=False,
                        route="RUNTIME_STRIKE_CAUSAL_WINDOW_CLOSED_FALLBACK",
                        confidence=0.0,
                        selected_context_hash=None,
                        action=None,
                    )
                shooter.runtime_strike_route_decision = route_decision
                shooter.runtime_strike_route_time_sec = float(data.time)
                route_action = route_decision.action
                maximum_advance = (
                    0 if route_action is None else route_action.maximum_arrival_advance_frames
                )
                if route_action is None:
                    if route_latch_open:
                        shooter.causal_strike_option.reject_runtime_route()
                    else:
                        shooter.causal_strike_option.expire_runtime_route()
                else:
                    shooter.causal_strike_option.select_arrival_route(maximum_advance)
                    shooter.parameters = replace(
                        shooter.parameters,
                        foot_yaw_offset=route_action.foot_yaw_offset_rad,
                        foot_pitch_offset=route_action.foot_pitch_offset_rad,
                    )
                    if isinstance(route_action, RuntimeContactModeAction):
                        shooter.parameters = replace(
                            shooter.parameters,
                            stance_offset_x=route_action.stance_offset_x_m,
                            stance_offset_y=route_action.stance_offset_y_m,
                        )
                        assert shooter_ballistic_contact_torque_config is not None
                        shooter_ballistic_contact_torque_config = replace(
                            shooter_ballistic_contact_torque_config,
                            contact_policy_frame=route_action.contact_policy_frame,
                        )
            if shooter.last_causal_strike_option_decision.begin_bridge:
                strike_phase_start_frame = (
                    shooter.last_causal_strike_option_decision.strike_phase_start_frame
                )
                if strike_phase_start_frame is None:
                    raise RuntimeError("causal strike option committed without a strike phase")
                _begin_causal_strike_bridge(
                    shooter,
                    timestamp_sec=float(data.time),
                    predecessor_policy_frame=predecessor_frame,
                    strike_phase_start_frame=strike_phase_start_frame,
                )
        if (
            (second_threat_config is not None or physical_rearm_earliest is not None)
            and goalkeeper is not None
            and goalkeeper_observer is not None
            and not second_threat_rearmed
            and goalkeeper_contact_time is not None
            and float(data.time)
            >= (
                second_threat_config.launch_time_sec - second_threat_config.rearm_lead_sec
                if second_threat_config is not None
                else cast(float, physical_rearm_earliest)
            )
            and _goalkeeper_ready_for_second_threat(goalkeeper, data=data)
        ):
            # Rearm only controller memory.  Physical qpos/qvel, the live ball
            # and the simulation clock remain untouched.
            goalkeeper.contact_latched = False
            goalkeeper.contact_time = None
            goalkeeper_observer.rearm()
            goalkeeper_previous_actor_residual.fill(0.0)
            if goalkeeper.goalkeeper_reach_memory is not None:
                goalkeeper.goalkeeper_reach_memory.fill(0.0)
            goalkeeper_mosaic_gmt_arrival_time_sec = None
            goalkeeper_mosaic_gmt_target_height_m = 0.0
            goalkeeper_mosaic_gmt_route = None
            goalkeeper_mosaic_gmt_mirror_latch = None
            goalkeeper_balanced_dive_flight_start_sec = None
            goalkeeper_balanced_dive_flight_duration_sec = None
            goalkeeper_balanced_dive_direction_index = None
            goalkeeper_balanced_dive_anchor_target = None
            goalkeeper_balanced_dive_was_airborne = False
            goalkeeper_landing_capture_start_sec = None
            goalkeeper_landing_capture_anchor = None
            goalkeeper_landing_capture_duration_sec = None
            goalkeeper_landing_capture_proprioceptive = False
            second_threat_rearmed = True
            second_threat_rearm_time = float(data.time)
        if (
            second_threat_config is not None
            and second_threat_rearmed
            and second_threat_launch_time is None
            and float(data.time) + 1.0e-12 >= second_threat_config.launch_time_sec
        ):
            if goalkeeper_observer is None:
                raise RuntimeError("second-threat launch lost its causal observer")
            # The readiness event occurs before the launch.  Clear only the
            # causal ball history again at the exact force epoch so residual
            # motion from the first save cannot masquerade as threat two.
            goalkeeper_observer.rearm()
            goalkeeper_previous_actor_residual.fill(0.0)
            position = np.asarray(data.qpos[ball_qpos : ball_qpos + 3], dtype=np.float64)
            current_velocity = np.asarray(data.qvel[ball_qvel : ball_qvel + 3], dtype=np.float64)
            duration = second_threat_config.flight_time_sec
            target = np.asarray(
                (
                    active_goal.plane_x_m - second_threat_config.target_depth_before_goal_m,
                    second_threat_config.target_y_m,
                    second_threat_config.target_z_m,
                ),
                dtype=np.float64,
            )
            # A goal threat must travel from the pitch toward the goal.  A
            # first save can legitimately deflect the live ball behind the
            # goal line but outside the posts.  Relaunching that ball back
            # toward the field would be a physically real force and still be
            # the wrong football event, so reject it before computing or
            # applying any impulse.
            if position[0] + active_goal.ball_radius_m >= target[0]:
                raise RuntimeError("second-threat live ball is not in a field-side launch pocket")
            target_velocity = (target - position) / duration
            target_velocity[2] += 0.5 * 9.81 * duration
            if target_velocity[0] <= 0.0:
                raise RuntimeError("second-threat live ball is not travelling toward goal")
            force = active_goal.ball_mass_kg * (
                (target_velocity - current_velocity) / second_threat_config.force_duration_sec
                + np.asarray((0.0, 0.0, 9.81), dtype=np.float64)
            )
            force_norm = float(np.linalg.norm(force))
            if force_norm > second_threat_config.maximum_force_n:
                raise RuntimeError("second-threat live-ball launch exceeded its force bound")
            second_threat_launch_time = float(data.time)
            second_threat_launch_position = position.copy()
            second_threat_target_velocity = target_velocity.copy()
            second_threat_force = force.copy()
            second_threat_force_stop_sec = (
                float(data.time) + second_threat_config.force_duration_sec
            )
            second_threat_peak_force = force_norm
            first_goal_crossed_before_second_threat = goal_crossed
        if (
            (launcher_position is None or launcher_receiver_enabled)
            and not shooter.entered
            and shooter.causal_strike_option is None
        ):
            if shooter.transition_actor is None:
                if data.time + 1e-12 >= shooter.start_sec:
                    _enter_policy(shooter)
            else:
                transition_decision = shooter.transition_decision
                if transition_decision is None:
                    raise RuntimeError("causal shooter transition decision is unavailable")
                if predecessor_frame >= transition_decision.trigger_policy_frame:
                    _enter_policy(shooter)
                    shooter.transition_triggered = True
                    shooter.transition_trigger_time_sec = float(data.time)
        if not passer.entered and data.time + 1e-12 >= passer.start_sec:
            _enter_policy(passer)
        if (
            second_striker is not None
            and not second_striker.entered
            and data.time + 1e-12 >= second_striker.start_sec
        ):
            _enter_policy(second_striker)
        policy_frames: dict[str, int] = {}
        goalkeeper_command_mps = 0.0
        goalkeeper_target_y_m = 0.0
        goalkeeper_reaction_active = False
        goalkeeper_anticipation_active = False
        goalkeeper_block_action_active = False
        goalkeeper_overhead_reach_blend = 0.0
        goalkeeper_overhead_reach_target_delta: NDArray[np.float64] = np.zeros(29, dtype=np.float64)
        goalkeeper_whole_body_reach_blend = 0.0
        goalkeeper_whole_body_reach_target_delta: NDArray[np.float64] = np.zeros(
            29, dtype=np.float64
        )
        goalkeeper_mosaic_gmt_blend = 0.0
        goalkeeper_mosaic_gmt_mirrored = False
        goalkeeper_mosaic_gmt_target_delta: NDArray[np.float64] = np.zeros(29, dtype=np.float64)
        goalkeeper_balanced_dive_blend = 0.0
        goalkeeper_balanced_dive_target_delta: NDArray[np.float64] = np.zeros(29, dtype=np.float64)
        goalkeeper_landing_capture_active = False
        goalkeeper_landing_capture_blend = 0.0
        goalkeeper_proprioceptive_capture_active = False
        goalkeeper_bimanual_reach_active = False
        goalkeeper_actor_observation: GoalkeeperActorObservation | None = None
        goalkeeper_actor_height_routed = False
        goalkeeper_mosaic_gmt_central_routed = False
        goalkeeper_learned_action = None
        goalkeeper_foundation_target: NDArray[np.float64] | None = None
        goalkeeper_foundation_kp: NDArray[np.float64] | None = None
        goalkeeper_foundation_kd: NDArray[np.float64] | None = None
        goalkeeper_reference_action: HumanoidGoalkeeperReferenceAction | None = None
        shooter_ballistic_contact_active = False
        shooter_ballistic_contact_target_delta: NDArray[np.float64] = np.zeros(29, dtype=np.float64)
        second_striker_ballistic_contact_active = False
        second_striker_ballistic_contact_target_delta: NDArray[np.float64] = np.zeros(
            29, dtype=np.float64
        )
        if goalkeeper is not None and goalkeeper_config is not None:
            physical_second_tracking = bool(second_striker is not None and second_threat_rearmed)
            keeper_shooter: _Robot = (
                cast(_Robot, second_striker)
                if physical_second_tracking
                else passer
                if goalkeeper_threat_role == "passer"
                else shooter
            )
            keeper_ball_qpos = (
                cast(int, second_ball_qpos) if physical_second_tracking else ball_qpos
            )
            keeper_ball_qvel = (
                cast(int, second_ball_qvel) if physical_second_tracking else ball_qvel
            )
            if goalkeeper_reactive_movement_config is not None:
                assert goalkeeper_reactive_actor is not None
                tactical_target, tactical_command, tactical_active = _command_reactive_movement(
                    goalkeeper,
                    carrier=shooter,
                    other_role=passer,
                    data=data,
                    ball_qpos=ball_qpos,
                    config=goalkeeper_reactive_movement_config,
                    actor=goalkeeper_reactive_actor,
                )
                goalkeeper_command_mps = float(tactical_command[1])
                goalkeeper_target_y_m = float(tactical_target[1])
                goalkeeper_reaction_active = tactical_active
            elif goalkeeper_tactical_movement_config is not None:
                tactical_target, tactical_command, tactical_active = _command_tactical_movement(
                    goalkeeper,
                    data=data,
                    config=goalkeeper_tactical_movement_config,
                    timestamp_sec=float(data.time),
                )
                goalkeeper_command_mps = float(tactical_command[1])
                goalkeeper_target_y_m = float(tactical_target[1])
                goalkeeper_reaction_active = tactical_active
            else:
                (
                    goalkeeper_command_mps,
                    goalkeeper_target_y_m,
                    goalkeeper_reaction_active,
                    goalkeeper_anticipation_active,
                    goalkeeper_actor_observation,
                ) = _command_goalkeeper(
                    goalkeeper,
                    shooter=keeper_shooter,
                    data=data,
                    ball_qpos=keeper_ball_qpos,
                    ball_qvel=keeper_ball_qvel,
                    goal=active_goal,
                    config=goalkeeper_config,
                    shot_contact_time=keeper_shooter.contact_time,
                    observer=goalkeeper_observer,
                    learned_actor=goalkeeper_actor,
                    previous_actor_residual=goalkeeper_previous_actor_residual,
                    recovery_athlete_torch=goalkeeper_recovery_athlete_torch,
                    recovery_athlete_model=goalkeeper_recovery_athlete_model,
                    recovery_athlete_checkpoint=goalkeeper_recovery_athlete_checkpoint,
                )
            goalkeeper.standby_locomotion_mirror_active = bool(
                goalkeeper_config.canonical_locomotion_mirror_enabled
                and goalkeeper_command_mps > 1.0e-6
            )
            goalkeeper_reaction_frames += int(goalkeeper_reaction_active)
            goalkeeper_anticipation_frames += int(goalkeeper_anticipation_active)
            if goalkeeper_reference_actor is not None and goalkeeper_actor_observation is not None:
                intercept = goalkeeper_actor_observation.estimated_intercept
                reference_region = _external_reference_region(
                    local_y_m=intercept[1],
                    height_m=intercept[2],
                )
                goalkeeper_reference_action = goalkeeper_reference_actor.action(
                    ball_position_local_m=np.asarray(goalkeeper.state.ball_pos_w, dtype=np.float64),
                    ball_visible=bool(
                        goalkeeper_actor_observation.observed_flight_start_sec is not None
                    ),
                    angular_velocity_rad_s=np.asarray(goalkeeper.state.ang_vel, dtype=np.float64),
                    projected_gravity=np.asarray(goalkeeper.state.gravity_ori, dtype=np.float64),
                    joint_position_rad=np.asarray(goalkeeper.state.q, dtype=np.float64),
                    joint_velocity_rad_s=np.asarray(goalkeeper.state.dq, dtype=np.float64),
                    region_override=reference_region,
                )
        for robot in robots:
            if (
                not robot.entered
                and robot.standby_policy is not None
                and robot.role != "goalkeeper"
            ):
                if robot is passer and passer_reactive_movement_config is not None:
                    if goalkeeper is None or passer_reactive_actor is None:
                        raise RuntimeError("reactive teammate route requires its defender context")
                    _command_reactive_movement(
                        robot,
                        carrier=shooter,
                        other_role=goalkeeper,
                        data=data,
                        ball_qpos=ball_qpos,
                        config=passer_reactive_movement_config,
                        actor=passer_reactive_actor,
                    )
                elif robot is passer and passer_tactical_movement_config is not None:
                    _command_tactical_movement(
                        robot,
                        data=data,
                        config=passer_tactical_movement_config,
                        timestamp_sec=float(data.time),
                    )
                else:
                    robot.state.vel_cmd = _normalized_zero_locomotion_command(robot.standby_policy)
            policy_frames[robot.role] = _update_policy(robot, frame, timestamp_sec=float(data.time))
            role_contact_config = (
                second_striker_ballistic_contact_config
                if robot.role == "second_striker"
                and second_striker_ballistic_contact_config is not None
                else shooter_ballistic_contact_config
            )
            if robot.role in {"shooter", "second_striker"} and role_contact_config is not None:
                (
                    robot.last_target,
                    contact_target_delta,
                    contact_target_active,
                ) = blend_g1_ballistic_contact_target(
                    target=robot.last_target,
                    policy_frame=policy_frames[robot.role],
                    control_dt_sec=_CONTROL_DT,
                    config=role_contact_config,
                    kick_foot=robot.parameters.kick_foot,
                )
                if robot.role == "second_striker":
                    second_striker_ballistic_contact_target_delta = contact_target_delta
                    second_striker_ballistic_contact_active = contact_target_active
                else:
                    shooter_ballistic_contact_target_delta = contact_target_delta
                    shooter_ballistic_contact_active = contact_target_active
            robot.motion_prior_position_active_frame_count += int(
                robot.last_motion_prior_position_active
            )
            robot.motion_prior_velocity_active_frame_count += int(
                robot.last_motion_prior_velocity_active
            )
            robot.agility_prior_active_frame_count += int(robot.last_agility_prior_active)
            robot.contact_prior_active_frame_count += int(robot.last_contact_prior_active)
            if robot.role == "passer" and passer_waist_pitch_target_margin_rad > 0.0:
                joint_index = 14
                joint_range = model.jnt_range[robot.joint_ids[joint_index]]
                robot.last_target[joint_index] = float(
                    np.clip(
                        robot.last_target[joint_index],
                        joint_range[0] + passer_waist_pitch_target_margin_rad,
                        joint_range[1] - passer_waist_pitch_target_margin_rad,
                    )
                )
            if robot.role == "goalkeeper" and goalkeeper_config is not None:
                goalkeeper_foundation_target = robot.last_target.copy()
                goalkeeper_foundation_kp = robot.kp.copy()
                goalkeeper_foundation_kd = robot.kd.copy()
                goalkeeper_actor_height_routed = bool(
                    goalkeeper_actor is not None
                    and goalkeeper_actor_observation is not None
                    and _goalkeeper_actor_route_active(
                        robot,
                        observation=goalkeeper_actor_observation,
                        config=goalkeeper_config,
                        timestamp_sec=float(data.time),
                    )
                )
                if (
                    goalkeeper_actor_height_routed
                    and goalkeeper_actor is not None
                    and goalkeeper_actor_observation is not None
                ):
                    goalkeeper_learned_action = goalkeeper_actor.action(
                        goalkeeper_actor_observation
                    )
                    goalkeeper_previous_actor_residual = np.asarray(
                        goalkeeper_learned_action.joint_position_residual_rad,
                        dtype=np.float64,
                    )
                if (
                    goalkeeper_mosaic_gmt_route is None
                    and goalkeeper_mosaic_gmt_controller is not None
                    and goalkeeper_actor_observation is not None
                    and _goalkeeper_actor_route_active(
                        robot,
                        observation=goalkeeper_actor_observation,
                        config=goalkeeper_config,
                        timestamp_sec=float(data.time),
                    )
                ):
                    current_relative_intercept_y = _goalkeeper_current_relative_intercept_y(
                        robot,
                        anchor_relative_intercept_y_m=float(
                            goalkeeper_actor_observation.estimated_intercept[1]
                        ),
                    )
                    goalkeeper_mosaic_gmt_route = bool(
                        goalkeeper_actor_observation.estimated_intercept[2]
                        >= goalkeeper_config.mosaic_gmt_minimum_target_height_m
                        and abs(current_relative_intercept_y)
                        <= goalkeeper_config.mosaic_gmt_maximum_lateral_error_m
                    )
                goalkeeper_mosaic_gmt_central_routed = goalkeeper_mosaic_gmt_route is True
                if goalkeeper_reference_action is not None:
                    flight_start = (
                        None
                        if goalkeeper_actor_observation is None
                        else goalkeeper_actor_observation.observed_flight_start_sec
                    )
                    if flight_start is not None:
                        elapsed = max(0.0, float(data.time) - flight_start)
                        ramp = float(np.clip(elapsed / 0.12, 0.0, 1.0))
                        ramp = ramp * ramp * (3.0 - 2.0 * ramp)
                        blend = goalkeeper_config.external_reference_blend * ramp
                        reference_target = np.asarray(
                            goalkeeper_reference_action.target_joint_position_rad,
                            dtype=np.float64,
                        )
                        ranges = model.jnt_range[robot.joint_ids]
                        limited = model.jnt_limited[robot.joint_ids].astype(bool)
                        clipped = reference_target.copy()
                        clipped[limited] = np.clip(
                            clipped[limited],
                            ranges[limited, 0] + 0.08,
                            ranges[limited, 1] - 0.08,
                        )
                        # The published teacher's leg commands were trained in
                        # Isaac Gym and destabilize this MuJoCo locomotion
                        # foundation.  Keep its data-driven waist/arm reach as
                        # a research teacher while the qualified local
                        # cerebellum retains exclusive ownership of both legs.
                        upper = slice(12, 29)
                        robot.last_target[upper] = (1.0 - blend) * robot.last_target[
                            upper
                        ] + blend * clipped[upper]
                if (
                    goalkeeper_learned_action is not None
                    and goalkeeper_reaction_active
                    and not goalkeeper_mosaic_gmt_central_routed
                ):
                    learned_residual = np.asarray(
                        goalkeeper_learned_action.joint_position_residual_rad,
                        dtype=np.float64,
                    )
                    # The qualified locomotion policy remains the foundation;
                    # the actor contributes only artifact-bounded residuals
                    # (at most 0.045 rad on a leg joint in the S11 candidate).
                    # With the goalkeeper root frame corrected, this small
                    # whole-body channel can express a split step or leg load
                    # without rotating the lateral command into fore/aft drift.
                    robot.last_target += learned_residual
                    if (
                        goalkeeper_actor is not None
                        and goalkeeper_actor_observation is not None
                        and goalkeeper_actor.artifact.operational_space_reach_enabled
                    ):
                        elapsed_sec = max(
                            0.0,
                            float(data.time)
                            - float(
                                goalkeeper_actor_observation.observed_flight_start_sec or data.time
                            ),
                        )
                        reach_fraction = abs(
                            goalkeeper_learned_action.operational_space_reach_fraction
                        )
                        if goalkeeper_config.actor_bimanual_reach_enabled:
                            goalkeeper_bimanual_reach_active = (
                                _apply_goalkeeper_bimanual_operational_space_reach(
                                    robot,
                                    model=model,
                                    data=data,
                                    observation=goalkeeper_actor_observation,
                                    artifact=goalkeeper_actor.artifact,
                                    target_local_x_m=(
                                        goalkeeper_config.actor_bimanual_reach_local_x_m
                                    ),
                                    half_span_m=(
                                        goalkeeper_config.actor_bimanual_reach_half_span_m
                                    ),
                                    height_offset_m=(
                                        goalkeeper_config.actor_bimanual_reach_height_offset_m
                                    ),
                                    reach_fraction=max(
                                        reach_fraction,
                                        goalkeeper_config.actor_bimanual_reach_minimum_fraction,
                                    ),
                                    gain_scale=(goalkeeper_config.actor_bimanual_reach_gain_scale),
                                    memory_decay=(
                                        goalkeeper_config.actor_bimanual_reach_memory_decay
                                    ),
                                    memory_maximum_rad=(
                                        goalkeeper_config.actor_bimanual_reach_memory_maximum_rad
                                    ),
                                    elapsed_sec=elapsed_sec,
                                )
                            )
                        else:
                            _apply_goalkeeper_operational_space_reach(
                                robot,
                                model=model,
                                data=data,
                                observation=goalkeeper_actor_observation,
                                artifact=goalkeeper_actor.artifact,
                                target_local_x_m=(
                                    goalkeeper_config.actor_operational_space_reach_local_x_m
                                ),
                                target_side_x_offset_m=(
                                    goalkeeper_config.actor_operational_space_reach_side_offset_m
                                ),
                                reach_fraction=reach_fraction,
                                elapsed_sec=elapsed_sec,
                            )
                        goalkeeper_bimanual_reach_frames += int(goalkeeper_bimanual_reach_active)
                observation_block_timing = bool(
                    goalkeeper_config.block_action_timing_mode == "observation"
                    or (
                        goalkeeper_config.block_action_timing_mode == "auto"
                        and goalkeeper_config.actor_observation_mode == "visible_ball_history_v3"
                    )
                )
                if observation_block_timing:
                    flight_start = (
                        None
                        if goalkeeper_actor_observation is None
                        else goalkeeper_actor_observation.observed_flight_start_sec
                    )
                    block_frame = (
                        -1
                        if flight_start is None
                        else int(
                            math.floor(
                                max(
                                    0.0,
                                    float(data.time)
                                    - flight_start
                                    - goalkeeper_config.reaction_delay_sec,
                                )
                                / _CONTROL_DT
                            )
                        )
                    )
                else:
                    block_frame = (
                        policy_frames["shooter"] - goalkeeper_config.block_action_start_policy_frame
                    )
                # A deployed actor owns the reach residual.  Keeping the old
                # hand-authored reach on top would make the CPU exam measure
                # two superposed controllers rather than the learned policy.
                if goalkeeper_actor is None or not goalkeeper_actor_height_routed:
                    goalkeeper_block_action_active = _apply_goalkeeper_reach(
                        robot,
                        target_y_m=goalkeeper_target_y_m,
                        current_y_m=float(data.qpos[robot.qpos_base + 1]),
                        reaction_active=goalkeeper_reaction_active,
                        block_frame=block_frame,
                        config=goalkeeper_config,
                    )
                if (
                    goalkeeper_overhead_reach_prior is not None
                    and goalkeeper_actor_observation is not None
                    and goalkeeper_actor_observation.observed_flight_start_sec is not None
                ):
                    # A height-routed motion expert preserves the low-shot
                    # champion while replaying a semantically reconstructed
                    # MOSAIC overhead-set trajectory for aerial shots.  Legs
                    # remain under the qualified locomotion cerebellum; the
                    # waist/arms provide data-driven counterbalanced reach.
                    intercept = goalkeeper_actor_observation.estimated_intercept
                    (
                        robot.last_target,
                        goalkeeper_overhead_reach_target_delta,
                        goalkeeper_overhead_reach_blend,
                    ) = blend_g1_mosaic_overhead_reach_target(
                        target=robot.last_target,
                        prior=goalkeeper_overhead_reach_prior,
                        time_to_arrival_sec=intercept[0],
                        target_height_m=intercept[2],
                        blend=goalkeeper_config.overhead_reach_blend,
                        minimum_target_height_m=(
                            goalkeeper_config.overhead_reach_minimum_target_height_m
                        ),
                        full_target_height_m=(
                            goalkeeper_config.overhead_reach_full_target_height_m
                        ),
                        joint_scales=(
                            (0.0,) * 12
                            + (goalkeeper_config.overhead_reach_waist_scale,) * 3
                            + (goalkeeper_config.overhead_reach_arm_scale,) * 14
                        ),
                    )
                    goalkeeper_overhead_reach_frames += int(goalkeeper_overhead_reach_blend > 0.0)
                    goalkeeper_overhead_reach_peak_blend = max(
                        goalkeeper_overhead_reach_peak_blend,
                        goalkeeper_overhead_reach_blend,
                    )
                if (
                    goalkeeper_whole_body_reach_atlas is not None
                    and goalkeeper_actor_observation is not None
                ):
                    intercept = goalkeeper_actor_observation.estimated_intercept
                    height_span = (
                        goalkeeper_config.whole_body_reach_full_target_height_m
                        - goalkeeper_config.whole_body_reach_minimum_target_height_m
                    )
                    height_phase = float(
                        np.clip(
                            (
                                intercept[2]
                                - goalkeeper_config.whole_body_reach_minimum_target_height_m
                            )
                            / height_span,
                            0.0,
                            1.0,
                        )
                    )
                    height_gate = height_phase * height_phase * (3.0 - 2.0 * height_phase)
                    adjusted_tta = max(
                        0.0,
                        intercept[0] - goalkeeper_config.whole_body_reach_timing_lead_sec,
                    )
                    timing_phase = float(np.clip((0.80 - adjusted_tta) / 0.40, 0.0, 1.0))
                    timing_gate = timing_phase * timing_phase * (3.0 - 2.0 * timing_phase)
                    requested_blend = (
                        goalkeeper_config.whole_body_reach_blend * height_gate * timing_gate
                    )
                    if goalkeeper_actor_observation.observed_flight_start_sec is None:
                        requested_blend = 0.0
                    # Retain a short proprioceptive muscle-memory tail instead
                    # of dropping the upper body pose in one 20 ms frame when
                    # the ball crosses the keeper anchor.
                    goalkeeper_whole_body_reach_memory = max(
                        requested_blend,
                        goalkeeper_whole_body_reach_memory * 0.92,
                    )
                    goalkeeper_whole_body_reach_blend = goalkeeper_whole_body_reach_memory
                    if goalkeeper_whole_body_reach_blend > 1.0e-6:
                        current_y = float(data.qpos[robot.qpos_base + 1])
                        current_z = float(data.qpos[robot.qpos_base + 2])
                        target_x = float(
                            np.median(
                                np.asarray(
                                    goalkeeper_whole_body_reach_atlas.target_relative_m,
                                    dtype=np.float64,
                                )[:, 0]
                            )
                        )
                        atlas_delta = whole_body_reach_from_target_numpy(
                            target_relative=np.asarray(
                                (
                                    (
                                        target_x,
                                        goalkeeper_target_y_m - current_y,
                                        intercept[2] - current_z,
                                    ),
                                ),
                                dtype=np.float64,
                            ),
                            model=goalkeeper_whole_body_reach_atlas,
                        )[0]
                        scaled_atlas_delta = atlas_delta.copy()
                        scaled_atlas_delta[:3] *= goalkeeper_config.whole_body_reach_waist_scale
                        target_arm = slice(10, 17)
                        support_arm = slice(3, 10)
                        if goalkeeper_target_y_m - current_y < 0.0:
                            target_arm, support_arm = support_arm, target_arm
                        scaled_atlas_delta[target_arm] *= (
                            goalkeeper_config.whole_body_reach_target_arm_scale
                        )
                        scaled_atlas_delta[support_arm] *= (
                            goalkeeper_config.whole_body_reach_support_arm_scale
                        )
                        goalkeeper_whole_body_reach_target_delta[12:] = (
                            goalkeeper_whole_body_reach_blend * scaled_atlas_delta
                        )
                        robot.last_target[12:] += goalkeeper_whole_body_reach_target_delta[12:]
                        goalkeeper_whole_body_reach_frames += 1
                        goalkeeper_whole_body_reach_peak_blend = max(
                            goalkeeper_whole_body_reach_peak_blend,
                            goalkeeper_whole_body_reach_blend,
                        )
                if (
                    goalkeeper_balanced_dive_seed is not None
                    and goalkeeper_actor_observation is not None
                    and goalkeeper_actor_observation.observed_flight_start_sec is not None
                ):
                    intercept = goalkeeper_actor_observation.estimated_intercept
                    current_y = float(data.qpos[robot.qpos_base + 1])
                    lateral_error = goalkeeper_target_y_m - current_y
                    height_routed = bool(
                        intercept[2] >= goalkeeper_config.mosaic_gmt_minimum_target_height_m
                    )
                    if (
                        goalkeeper_balanced_dive_flight_start_sec is None
                        and height_routed
                        and abs(lateral_error)
                        >= goalkeeper_config.balanced_dive_minimum_lateral_error_m
                        and (
                            goalkeeper_config.balanced_dive_activation_lead_sec <= 0.0
                            or 0.0
                            < intercept[0]
                            <= goalkeeper_config.balanced_dive_activation_lead_sec
                        )
                    ):
                        goalkeeper_balanced_dive_flight_start_sec = float(data.time)
                        goalkeeper_balanced_dive_flight_duration_sec = max(
                            0.40,
                            intercept[0],
                        )
                        goalkeeper_balanced_dive_direction_index = 0 if lateral_error < 0.0 else 1
                        # The qualification always transitions from the
                        # canonical ready pose, not from an arbitrary neural
                        # locomotion action sampled on the trigger frame.
                        goalkeeper_balanced_dive_anchor_target = robot.hold_target.copy()
                    if (
                        goalkeeper_balanced_dive_flight_start_sec is not None
                        and goalkeeper_balanced_dive_flight_duration_sec is not None
                        and goalkeeper_balanced_dive_direction_index is not None
                        and goalkeeper_balanced_dive_anchor_target is not None
                        and goalkeeper_balanced_dive_kp is not None
                        and goalkeeper_balanced_dive_kd is not None
                    ):
                        elapsed = max(
                            0.0,
                            float(data.time) - goalkeeper_balanced_dive_flight_start_sec,
                        )
                        # The source seed was physically qualified with a
                        # 0.5 s move into its first pose *before* advancing
                        # the recorded trajectory.  Advancing the clip while
                        # that blend was still in progress mixed two distant
                        # postures and made the same qualified seed collapse
                        # inside the shared-world rollout.  Preserve that
                        # temporal contract, then time-warp only the remaining
                        # trajectory so the configured phase is reached when
                        # the ball arrives.
                        phase, blend_gate, dive_owns_joints = _balanced_dive_phase_profile(
                            elapsed_sec=elapsed,
                            flight_duration_sec=(goalkeeper_balanced_dive_flight_duration_sec),
                            phase_at_arrival=(goalkeeper_config.balanced_dive_phase_at_arrival),
                            peak_phase=goalkeeper_config.balanced_dive_peak_phase,
                            blend_in_sec=goalkeeper_config.balanced_dive_blend_in_sec,
                            recovery_tail_sec=(goalkeeper_config.balanced_dive_recovery_tail_sec),
                            initial_phase=goalkeeper_config.balanced_dive_initial_phase,
                        )
                        frame_position = phase * (
                            goalkeeper_balanced_dive_seed.joint_position_rad.shape[1] - 1
                        )
                        lower_frame = int(math.floor(frame_position))
                        upper_frame = min(
                            lower_frame + 1,
                            goalkeeper_balanced_dive_seed.joint_position_rad.shape[1] - 1,
                        )
                        frame_alpha = frame_position - lower_frame
                        # The qualified source-left action moves the MuJoCo
                        # root toward world -y; its manufactured mirror moves
                        # toward +y.  Select by the measured world-frame error.
                        source_dive_target = (
                            1.0 - frame_alpha
                        ) * goalkeeper_balanced_dive_seed.joint_position_rad[
                            goalkeeper_balanced_dive_direction_index, lower_frame
                        ] + frame_alpha * goalkeeper_balanced_dive_seed.joint_position_rad[
                            goalkeeper_balanced_dive_direction_index, upper_frame
                        ]
                        dive_target = source_dive_target
                        if (
                            goalkeeper_dive_athlete_torch is not None
                            and goalkeeper_dive_athlete_model is not None
                            and goalkeeper_dive_athlete_checkpoint is not None
                        ):
                            from rosclaw_soccer.training.dive_athlete_expert import (
                                decode_dive_athlete_target,
                                dive_athlete_features_numpy,
                            )

                            athlete_features = dive_athlete_features_numpy(
                                phase=np.asarray((phase,), dtype=np.float64),
                                target_lateral_m=np.asarray(
                                    (abs(lateral_error),), dtype=np.float64
                                ),
                                target_height_m=np.asarray((intercept[2],), dtype=np.float64),
                                duration_sec=np.asarray(
                                    (goalkeeper_balanced_dive_flight_duration_sec,),
                                    dtype=np.float64,
                                ),
                                contact_phase=np.asarray(
                                    (goalkeeper_config.balanced_dive_phase_at_arrival,),
                                    dtype=np.float64,
                                ),
                            )
                            athlete_direction = -1.0 if lateral_error < 0.0 else 1.0
                            with goalkeeper_dive_athlete_torch.inference_mode():
                                decoded_target = decode_dive_athlete_target(
                                    torch=goalkeeper_dive_athlete_torch,
                                    model=goalkeeper_dive_athlete_model,
                                    checkpoint=goalkeeper_dive_athlete_checkpoint,
                                    features=goalkeeper_dive_athlete_torch.as_tensor(
                                        athlete_features,
                                        dtype=goalkeeper_dive_athlete_torch.float32,
                                    ),
                                    direction=goalkeeper_dive_athlete_torch.as_tensor(
                                        (athlete_direction,),
                                        dtype=goalkeeper_dive_athlete_torch.float32,
                                    ),
                                )
                            neural_dive_target = np.asarray(
                                decoded_target[0].cpu(), dtype=np.float64
                            )
                            dive_target = source_dive_target + (
                                goalkeeper_config.dive_athlete_blend
                                * (neural_dive_target - source_dive_target)
                            )
                        # Match the qualification's linear blend-in, retain
                        # full ownership through the recorded trajectory and
                        # then hand back continuously over the recovery tail.
                        goalkeeper_balanced_dive_blend = (
                            goalkeeper_config.balanced_dive_blend * blend_gate
                        )
                        # The physically qualified dive reference first owns
                        # the complete pose.  A bounded GMT blend may then
                        # refine its arms for high shots; retaining the source
                        # arm counter-motion is essential to angular-momentum
                        # balance during take-off.
                        if dive_owns_joints:
                            joint_group_scale = np.asarray(
                                (goalkeeper_config.balanced_dive_lower_body_scale,) * 12
                                + (goalkeeper_config.balanced_dive_waist_scale,) * 3
                                + (goalkeeper_config.balanced_dive_arm_scale,) * 14,
                                dtype=np.float64,
                            )
                            qualified_target = goalkeeper_balanced_dive_anchor_target + (
                                goalkeeper_balanced_dive_blend
                                * joint_group_scale
                                * (dive_target - goalkeeper_balanced_dive_anchor_target)
                            )
                            goalkeeper_balanced_dive_target_delta[:] = (
                                qualified_target - robot.last_target
                            )
                            robot.last_target += goalkeeper_balanced_dive_target_delta
                            # The CPU qualification owns a zero-velocity PD
                            # target and these exact gains while it has joint
                            # authority.  Once the envelope reaches zero, do
                            # not keep replacing the feedback locomotion
                            # target with a fixed pose: the stable foundation
                            # must regain complete target and impedance
                            # ownership for recovery.
                            robot.target_velocity.fill(0.0)
                            robot.kp = goalkeeper_balanced_dive_kp.copy()
                            robot.kd = goalkeeper_balanced_dive_kd.copy()
                            if goalkeeper_balanced_dive_blend > 0.0:
                                goalkeeper_balanced_dive_frames += 1
                                goalkeeper_balanced_dive_peak_blend = max(
                                    goalkeeper_balanced_dive_peak_blend,
                                    goalkeeper_balanced_dive_blend,
                                )
                if (
                    goalkeeper_mosaic_gmt_controller is not None
                    and goalkeeper_mosaic_gmt_contract is not None
                    and goalkeeper_mosaic_gmt_skill is not None
                    and goalkeeper_actor_observation is not None
                ):
                    torch = goalkeeper_mosaic_gmt_controller.torch
                    intercept = goalkeeper_actor_observation.estimated_intercept
                    if (
                        goalkeeper_actor_observation.observed_flight_start_sec is not None
                        and goalkeeper_actor_observation.intercept_confidence > 0.0
                        and intercept[0] > 0.0
                    ):
                        arrival_estimate = float(data.time) + intercept[0]
                        goalkeeper_mosaic_gmt_arrival_time_sec = (
                            arrival_estimate
                            if goalkeeper_mosaic_gmt_arrival_time_sec is None
                            else 0.85 * goalkeeper_mosaic_gmt_arrival_time_sec
                            + 0.15 * arrival_estimate
                        )
                        goalkeeper_mosaic_gmt_target_height_m = intercept[2]
                    height_span = (
                        goalkeeper_config.mosaic_gmt_full_target_height_m
                        - goalkeeper_config.mosaic_gmt_minimum_target_height_m
                    )
                    height_phase = float(
                        np.clip(
                            (
                                goalkeeper_mosaic_gmt_target_height_m
                                - goalkeeper_config.mosaic_gmt_minimum_target_height_m
                            )
                            / height_span,
                            0.0,
                            1.0,
                        )
                    )
                    height_gate = height_phase * height_phase * (3.0 - 2.0 * height_phase)
                    relative_time = (
                        float(goalkeeper_mosaic_gmt_skill.relative_times_sec[0])
                        if goalkeeper_mosaic_gmt_arrival_time_sec is None
                        else float(data.time)
                        - goalkeeper_mosaic_gmt_arrival_time_sec
                        + goalkeeper_config.mosaic_gmt_timing_lead_sec
                    )
                    skill_active = bool(
                        not robot.contact_latched
                        and height_gate > 0.0
                        and goalkeeper_mosaic_gmt_route is True
                        and goalkeeper_actor_observation.observed_flight_start_sec is not None
                        and relative_time
                        >= float(goalkeeper_mosaic_gmt_skill.relative_times_sec[0])
                        and relative_time
                        <= float(goalkeeper_mosaic_gmt_skill.relative_times_sec[-1]) + 0.06
                    )
                    goalkeeper_mosaic_gmt_mirror_latch = _latch_gmt_mirror_direction(
                        goalkeeper_mosaic_gmt_mirror_latch,
                        skill_active=skill_active,
                        local_intercept_y_m=float(intercept[1]),
                        mirror_enabled=goalkeeper_config.mosaic_gmt_mirror_by_intercept,
                    )
                    gmt_joint_position = np.asarray(robot.state.q, dtype=np.float64)
                    gmt_joint_velocity = np.asarray(robot.state.dq, dtype=np.float64)
                    gmt_torso_quaternion = np.asarray(
                        data.xquat[robot.torso_body], dtype=np.float64
                    )
                    gmt_angular_velocity = np.asarray(robot.state.ang_vel, dtype=np.float64)
                    if goalkeeper_mosaic_gmt_mirror_latch:
                        (
                            gmt_joint_position,
                            gmt_joint_velocity,
                            gmt_torso_quaternion,
                            gmt_angular_velocity,
                        ) = _mirror_gmt_proprioception(
                            gmt_joint_position,
                            gmt_joint_velocity,
                            gmt_torso_quaternion,
                            gmt_angular_velocity,
                        )
                    gmt_target_tensor, _ = goalkeeper_mosaic_gmt_controller.target(
                        canonical_joint_position=torch.as_tensor(
                            np.asarray(gmt_joint_position, dtype=np.float32)[None, :]
                        ),
                        canonical_joint_velocity=torch.as_tensor(
                            np.asarray(gmt_joint_velocity, dtype=np.float32)[None, :]
                        ),
                        torso_quaternion_wxyz=torch.as_tensor(
                            np.asarray(gmt_torso_quaternion, dtype=np.float32)[None, :]
                        ),
                        base_angular_velocity_body_rad_s=torch.as_tensor(
                            np.asarray(gmt_angular_velocity, dtype=np.float32)[None, :]
                        ),
                        heading_quaternion_wxyz=torch.as_tensor(
                            ((0.0, 0.0, 0.0, 1.0),),
                            dtype=torch.float32,
                        ),
                        relative_time_sec=torch.as_tensor(
                            (relative_time,),
                            dtype=torch.float32,
                        ),
                        active=torch.as_tensor((skill_active,)),
                    )
                    goalkeeper_mosaic_gmt_blend = (
                        goalkeeper_config.mosaic_gmt_blend * height_gate if skill_active else 0.0
                    )
                    if goalkeeper_mosaic_gmt_blend > 0.0:
                        gmt_target = np.asarray(
                            gmt_target_tensor[0].detach().cpu(),
                            dtype=np.float64,
                        )
                        goalkeeper_mosaic_gmt_mirrored = bool(goalkeeper_mosaic_gmt_mirror_latch)
                        if goalkeeper_mosaic_gmt_mirrored:
                            gmt_target = _mirror_g1_joint_positions(gmt_target)
                        group_scales = np.asarray(
                            (goalkeeper_config.mosaic_gmt_lower_body_scale,) * 12
                            + (goalkeeper_config.mosaic_gmt_waist_scale,) * 3
                            + (goalkeeper_config.mosaic_gmt_arm_scale,) * 14,
                            dtype=np.float64,
                        )
                        joint_blend = goalkeeper_mosaic_gmt_blend * group_scales
                        goalkeeper_mosaic_gmt_target_delta = joint_blend * (
                            gmt_target - robot.last_target
                        )
                        robot.last_target += goalkeeper_mosaic_gmt_target_delta
                        canonical_order = np.asarray(
                            MOSAIC_G1_ISAACLAB_TO_CANONICAL_DOF,
                            dtype=np.int64,
                        )
                        gmt_kp = np.asarray(
                            goalkeeper_mosaic_gmt_contract.joint_stiffness,
                            dtype=np.float64,
                        )[canonical_order]
                        gmt_kd = np.asarray(
                            goalkeeper_mosaic_gmt_contract.joint_damping,
                            dtype=np.float64,
                        )[canonical_order]
                        if goalkeeper_mosaic_gmt_mirrored:
                            # The frozen GMT skill is the local -y/right-side
                            # exemplar.  Reusing it unchanged for local +y
                            # made the arm/waist counter-motion reinforce the
                            # wrong support side and toppled otherwise stable
                            # leftward locomotion.  Mirror targets *and*
                            # impedance together in the canonical G1 order.
                            gmt_kp = _mirror_g1_joint_gains(gmt_kp)
                            gmt_kd = _mirror_g1_joint_gains(gmt_kd)
                        robot.kp = (1.0 - joint_blend) * robot.kp + joint_blend * gmt_kp
                        robot.kd = (1.0 - joint_blend) * robot.kd + joint_blend * gmt_kd
                        goalkeeper_mosaic_gmt_frames += 1
                        goalkeeper_mosaic_gmt_peak_blend = max(
                            goalkeeper_mosaic_gmt_peak_blend,
                            goalkeeper_mosaic_gmt_blend,
                        )
                if (
                    goalkeeper_config.actor_bimanual_reach_enabled
                    and not robot.contact_latched
                    and goalkeeper_mosaic_gmt_central_routed
                    and goalkeeper_actor is not None
                    and goalkeeper_actor_observation is not None
                    and goalkeeper_actor_observation.observed_flight_start_sec is not None
                ):
                    learned_fraction = (
                        0.0
                        if goalkeeper_learned_action is None
                        else abs(goalkeeper_learned_action.operational_space_reach_fraction)
                    )
                    goalkeeper_bimanual_reach_active = (
                        _apply_goalkeeper_bimanual_operational_space_reach(
                            robot,
                            model=model,
                            data=data,
                            observation=goalkeeper_actor_observation,
                            artifact=goalkeeper_actor.artifact,
                            target_local_x_m=(goalkeeper_config.actor_bimanual_reach_local_x_m),
                            half_span_m=(goalkeeper_config.actor_bimanual_reach_half_span_m),
                            height_offset_m=(
                                goalkeeper_config.actor_bimanual_reach_height_offset_m
                            ),
                            reach_fraction=max(
                                learned_fraction,
                                goalkeeper_config.actor_bimanual_reach_minimum_fraction,
                            ),
                            gain_scale=(goalkeeper_config.actor_bimanual_reach_gain_scale),
                            memory_decay=(goalkeeper_config.actor_bimanual_reach_memory_decay),
                            memory_maximum_rad=(
                                goalkeeper_config.actor_bimanual_reach_memory_maximum_rad
                            ),
                            elapsed_sec=max(
                                0.0,
                                float(data.time)
                                - goalkeeper_actor_observation.observed_flight_start_sec,
                            ),
                        )
                    )
                    goalkeeper_bimanual_reach_frames += int(goalkeeper_bimanual_reach_active)
                    _apply_goalkeeper_bimanual_support_arm(
                        robot,
                        local_intercept_y_m=float(
                            goalkeeper_actor_observation.estimated_intercept[1]
                        ),
                        blend=goalkeeper_config.actor_bimanual_support_arm_blend,
                        overhead_bias_rad=(
                            goalkeeper_config.actor_bimanual_support_arm_overhead_bias_rad
                        ),
                    )
                if (
                    goalkeeper_config.balanced_dive_landing_capture_enabled
                    and goalkeeper_landing_capture_start_sec is not None
                    and goalkeeper_landing_capture_anchor is not None
                    and goalkeeper_landing_capture_duration_sec is not None
                    and goalkeeper_balanced_dive_kp is not None
                    and goalkeeper_balanced_dive_kd is not None
                    and goalkeeper_foundation_target is not None
                    and goalkeeper_foundation_kp is not None
                    and goalkeeper_foundation_kd is not None
                ):
                    goalkeeper_landing_capture_blend, goalkeeper_landing_capture_active = (
                        _landing_capture_profile(
                            elapsed_sec=max(
                                0.0,
                                float(data.time) - goalkeeper_landing_capture_start_sec,
                            ),
                            duration_sec=goalkeeper_landing_capture_duration_sec,
                        )
                    )
                    if goalkeeper_landing_capture_active:
                        goalkeeper_proprioceptive_capture_active = (
                            goalkeeper_landing_capture_proprioceptive
                        )
                        # A proprioceptive landing reflex owns legs and waist
                        # only.  It captures the measured first-contact pose,
                        # then returns continuously to the feedback locomotion
                        # foundation with increased early damping.  Arms
                        # remain free to finish the save; no root pose or
                        # velocity is written.
                        lower_body = slice(0, 15)
                        blend = goalkeeper_landing_capture_blend
                        robot.last_target[lower_body] = (
                            1.0 - blend
                        ) * goalkeeper_landing_capture_anchor[
                            lower_body
                        ] + blend * goalkeeper_foundation_target[lower_body]
                        robot.target_velocity[lower_body] = 0.0
                        robot.kp[lower_body] = (1.0 - blend) * goalkeeper_balanced_dive_kp[
                            lower_body
                        ] + blend * goalkeeper_foundation_kp[lower_body]
                        robot.kd[lower_body] = (
                            (1.0 - blend)
                            * goalkeeper_balanced_dive_kd[lower_body]
                            * goalkeeper_config.balanced_dive_landing_damping_scale
                            + blend * goalkeeper_foundation_kd[lower_body]
                        )
                goalkeeper_block_action_frames += int(goalkeeper_block_action_active)
        learned_torque: dict[str, np.ndarray | IQLResidualDecision | None] = {
            robot.role: None for robot in robots
        }
        joint_guard_active: dict[str, bool] = {robot.role: False for robot in robots}
        for robot in robots:
            if robot.recovery_torque_actor is not None and robot.contact_latched:
                actor_state = _recovery_actor_state(
                    robot,
                    data,
                    ball_body=ball_body,
                    timestamp_sec=float(data.time),
                )
                if isinstance(robot.recovery_torque_actor, SupportBoundIQLResidualActor):
                    # A residual is meaningful only after the measured-contact
                    # structured recovery controller has taken ownership.
                    if not robot.post_policy_active:
                        continue
                    q = data.qpos[robot.joint_qpos]
                    dq = data.qvel[robot.joint_qvel]
                    baseline_torque = (robot.last_target - q) * robot.kp + (
                        robot.target_velocity - dq
                    ) * robot.kd
                    residual_decision = robot.recovery_torque_actor.action(
                        actor_state,
                        baseline_torque,
                    )
                    robot.learned_torque_support_rms_peak = max(
                        robot.learned_torque_support_rms_peak,
                        residual_decision.standardized_rms,
                    )
                    if residual_decision.accepted:
                        learned_torque[robot.role] = residual_decision
                        robot.learned_torque_frame_count += 1
                        robot.learned_torque_confidence_sum += residual_decision.confidence
                        robot.learned_torque_peak_residual_nm = max(
                            robot.learned_torque_peak_residual_nm,
                            residual_decision.peak_residual_nm,
                        )
                    else:
                        robot.learned_torque_fallback_count += 1
                else:
                    learned_torque[robot.role] = robot.recovery_torque_actor.action(actor_state)
                    robot.learned_torque_frame_count += 1

        contact_role = 0
        shooter_contact_foot = 0
        frame_robot_contacts = 0
        commanded_torque: dict[str, NDArray[np.float64]] = {
            robot.role: np.zeros(29, dtype=np.float64) for robot in robots
        }
        projected_torque: dict[str, NDArray[np.float64]] = {
            robot.role: np.zeros(29, dtype=np.float64) for robot in robots
        }
        executed_torque: dict[str, NDArray[np.float64]] = {
            robot.role: np.zeros(29, dtype=np.float64) for robot in robots
        }
        frame_contact_impulse = {robot.role: 0.0 for robot in robots}
        passer_support = (False, False)
        shooter_support = (False, False)
        goalkeeper_support = (False, False)
        frame_ballistic_actor_active = False
        frame_ballistic_actor_torque: NDArray[np.float64] = np.zeros(29, dtype=np.float64)
        frame_three_axis_contact_actor_active = False
        frame_three_axis_contact_actor_torque: NDArray[np.float64] = np.zeros(29, dtype=np.float64)
        frame_three_axis_contact_actor_force_xyz_n: NDArray[np.float64] = np.zeros(
            3, dtype=np.float64
        )
        frame_three_axis_contact_actor_foot_velocity_xyz_mps: NDArray[np.float64] = np.zeros(
            3, dtype=np.float64
        )
        frame_target_velocity_contact_actor_active = False
        frame_target_velocity_contact_actor_target_supported = False
        frame_target_velocity_contact_actor_torque: NDArray[np.float64] = np.zeros(
            29, dtype=np.float64
        )
        frame_target_velocity_contact_actor_force_xyz_n: NDArray[np.float64] = np.zeros(
            3, dtype=np.float64
        )
        frame_target_velocity_contact_actor_foot_velocity_xyz_mps: NDArray[np.float64] = np.zeros(
            3, dtype=np.float64
        )
        frame_target_velocity_contact_actor_target_xyz_mps: NDArray[np.float64] = np.zeros(
            3, dtype=np.float64
        )
        frame_neural_contact_actor_active = False
        frame_neural_contact_actor_supported = False
        frame_neural_contact_actor_torque: NDArray[np.float64] = np.zeros(29, dtype=np.float64)
        frame_neural_contact_actor_ood_distance = math.inf
        frame_second_striker_ballistic_actor_active = False
        frame_second_striker_ballistic_actor_torque: NDArray[np.float64] = np.zeros(
            29, dtype=np.float64
        )
        frame_second_striker_ballistic_actor_force_yz_n: NDArray[np.float64] = np.zeros(
            2, dtype=np.float64
        )
        frame_second_striker_ballistic_actor_foot_velocity_yz_mps: NDArray[np.float64] = np.zeros(
            2, dtype=np.float64
        )
        frame_second_striker_ballistic_actor_foot_ball_distance_m = math.nan
        frame_second_actor_desired_launch_yz: NDArray[np.float64] = np.full(
            2, math.nan, dtype=np.float64
        )
        frame_second_striker_ballistic_actor_target_conditioned = False
        frame_second_striker_ballistic_actor_launch_envelope_supported = False
        frame_second_striker_ballistic_actor_candidate_selected = False
        frame_second_striker_loft_teacher_active = False
        frame_second_striker_loft_teacher_force_yz_n: NDArray[np.float64] = np.zeros(
            2, dtype=np.float64
        )
        frame_second_striker_loft_teacher_foot_velocity_yz_mps: NDArray[np.float64] = np.zeros(
            2, dtype=np.float64
        )
        frame_second_striker_contact = False
        frame_second_striker_contact_force_n = 0.0
        frame_loft_teacher_active = False
        frame_loft_teacher_torque: NDArray[np.float64] = np.zeros(29, dtype=np.float64)
        frame_loft_teacher_force_xyz_n: NDArray[np.float64] = np.zeros(3, dtype=np.float64)
        frame_loft_teacher_foot_velocity_xyz_mps: NDArray[np.float64] = np.zeros(
            3, dtype=np.float64
        )
        frame_ballistic_contact_torque_active = False
        frame_ballistic_contact_torque: NDArray[np.float64] = np.zeros(29, dtype=np.float64)
        frame_first_touch_interception_active = False
        frame_first_touch_interception_torque: NDArray[np.float64] = np.zeros(29, dtype=np.float64)
        frame_first_touch_interception_force: NDArray[np.float64] = np.zeros(3, dtype=np.float64)
        frame_first_touch_interception_error: NDArray[np.float64] = np.zeros(3, dtype=np.float64)
        frame_passer_reception_interception_active = False
        frame_passer_reception_interception_torque: NDArray[np.float64] = np.zeros(
            29, dtype=np.float64
        )
        frame_passer_reception_interception_force: NDArray[np.float64] = np.zeros(
            3, dtype=np.float64
        )
        frame_passer_reception_interception_error: NDArray[np.float64] = np.zeros(
            3, dtype=np.float64
        )
        frame_second_striker_ballistic_contact_torque_active = False
        frame_second_striker_ballistic_contact_torque: NDArray[np.float64] = np.zeros(
            29, dtype=np.float64
        )
        frame_goalkeeper_bimanual_punch_active = False
        frame_goalkeeper_bimanual_punch_torque: NDArray[np.float64] = np.zeros(29, dtype=np.float64)
        for substep_index in range(_SUBSTEPS):
            data.xfrc_applied[ball_body, :3] = 0.0
            if second_ball_body is not None:
                data.xfrc_applied[second_ball_body, :3] = 0.0
            for robot in robots:
                q = data.qpos[robot.joint_qpos]
                dq = data.qvel[robot.joint_qvel]
                actor_torque = learned_torque[robot.role]
                baseline_torque = (robot.last_target - q) * robot.kp + (
                    robot.target_velocity - dq
                ) * robot.kd
                if isinstance(actor_torque, IQLResidualDecision):
                    raw_torque = baseline_torque + actor_torque.residual_torque
                elif actor_torque is not None:
                    raw_torque = actor_torque
                else:
                    raw_torque = baseline_torque
                if (
                    robot.role == "goalkeeper"
                    and goalkeeper_config is not None
                    and goalkeeper_actor_observation is not None
                    and goalkeeper_actor_height_routed
                    and goalkeeper_config.actor_bimanual_punch_force_n > 0.0
                ):
                    central_boost = (
                        goalkeeper_config.actor_bimanual_punch_central_boost_n_per_m
                        * max(
                            0.0,
                            goalkeeper_config.actor_bimanual_punch_reference_abs_lateral_m
                            - abs(float(goalkeeper_actor_observation.estimated_intercept[1])),
                        )
                    )
                    punch_torque, punch_active = _goalkeeper_bimanual_punch_torque(
                        robot,
                        model=model,
                        data=data,
                        observation=goalkeeper_actor_observation,
                        force_n=(
                            physical_second_striker_config.goalkeeper_punch_force_n
                            if physical_second_striker_config is not None
                            and second_threat_rearmed
                            and goalkeeper_contact_epoch == 1
                            else (
                                second_threat_config.goalkeeper_punch_force_n
                                if second_threat_config is not None
                                and second_threat_rearmed
                                and goalkeeper_contact_epoch == 1
                                else goalkeeper_config.actor_bimanual_punch_force_n + central_boost
                            )
                        ),
                        vertical_force_scale=(
                            goalkeeper_config.actor_bimanual_punch_vertical_force_scale
                        ),
                        outward_force_scale=(
                            physical_second_striker_config.goalkeeper_punch_outward_force_scale
                            if physical_second_striker_config is not None
                            and second_threat_rearmed
                            and goalkeeper_contact_epoch == 1
                            else (
                                second_threat_config.goalkeeper_punch_outward_force_scale
                                if second_threat_config is not None
                                and second_threat_rearmed
                                and goalkeeper_contact_epoch == 1
                                else goalkeeper_config.actor_bimanual_punch_outward_force_scale
                            )
                        ),
                        window_sec=goalkeeper_config.actor_bimanual_punch_window_sec,
                    )
                    raw_torque = raw_torque + punch_torque
                    frame_goalkeeper_bimanual_punch_active = (
                        frame_goalkeeper_bimanual_punch_active or punch_active
                    )
                    if float(np.max(np.abs(punch_torque))) >= float(
                        np.max(np.abs(frame_goalkeeper_bimanual_punch_torque))
                    ):
                        frame_goalkeeper_bimanual_punch_torque = punch_torque.copy()
                role_ballistic_actor = (
                    second_striker_ballistic_actor
                    if robot.role == "second_striker"
                    else shooter_ballistic_actor
                )
                if robot.role in {"shooter", "second_striker"} and (
                    role_ballistic_actor is not None
                ):
                    actor_ball_qpos = (
                        cast(int, second_ball_qpos) if robot.role == "second_striker" else ball_qpos
                    )
                    actor_ball_qvel = (
                        cast(int, second_ball_qvel) if robot.role == "second_striker" else ball_qvel
                    )
                    bilateral_actor_kwargs: dict[str, Any] = {}
                    if robot.parameters.kick_foot == "left":
                        bilateral_actor_kwargs = {
                            "striking_ankle_body_id": robot.left_ankle_body,
                            "lateral_mirror_sign": -1.0,
                        }
                    actor_effect_kwargs: dict[str, Any] = {
                        "model": model,
                        "data": data,
                        "right_ankle_body_id": robot.right_ankle_body,
                        "policy_frame": policy_frames[robot.role],
                        "contact_observed": robot.contact_latched,
                        "ball_position": np.asarray(
                            data.qpos[actor_ball_qpos : actor_ball_qpos + 3],
                            dtype=np.float64,
                        ),
                        "ball_velocity": np.asarray(
                            data.qvel[actor_ball_qvel : actor_ball_qvel + 3],
                            dtype=np.float64,
                        ),
                        "goal_plane_x_m": (
                            active_goal.plane_x_m
                            - physical_second_striker_config.ballistic_target_depth_before_goal_m
                            if robot.role == "second_striker"
                            and physical_second_striker_config is not None
                            else active_goal.plane_x_m
                        ),
                        "target_y_m": (
                            physical_second_striker_config.ballistic_target_y_m
                            if robot.role == "second_striker"
                            and physical_second_striker_config is not None
                            else active_goal.target_y_m
                        ),
                        "target_z_m": (
                            physical_second_striker_config.ballistic_target_z_m
                            if robot.role == "second_striker"
                            and physical_second_striker_config is not None
                            else active_goal.target_z_m
                        ),
                        "actuated_dof_indices": np.asarray(robot.joint_qvel, dtype=np.int64),
                        **bilateral_actor_kwargs,
                    }
                    candidate_effect = (
                        g1_ballistic_contact_impulse_effect(
                            actor=second_striker_candidate_actor,
                            **actor_effect_kwargs,
                        )
                        if robot.role == "second_striker"
                        and second_striker_candidate_actor is not None
                        else None
                    )
                    parent_effect = g1_ballistic_contact_impulse_effect(
                        actor=role_ballistic_actor,
                        **actor_effect_kwargs,
                    )
                    selection = select_g1_ballistic_contact_effect(
                        parent=parent_effect,
                        candidate=candidate_effect,
                    )
                    effect = selection.effect
                    teacher_owns_second_contact = bool(
                        robot.role == "second_striker"
                        and second_striker_loft_teacher_config is not None
                    )
                    if robot.role == "second_striker":
                        frame_second_striker_ballistic_actor_candidate_selected = (
                            frame_second_striker_ballistic_actor_candidate_selected
                            or selection.candidate_selected
                        )
                    if not teacher_owns_second_contact:
                        raw_torque = raw_torque + effect.torque
                    if robot.role == "second_striker":
                        diagnostic_effect = candidate_effect or effect
                        if diagnostic_effect.foot_ball_distance_m is not None:
                            frame_second_striker_ballistic_actor_foot_ball_distance_m = (
                                diagnostic_effect.foot_ball_distance_m
                            )
                            if (
                                diagnostic_effect.target_conditioned
                                and diagnostic_effect.foot_ball_distance_m
                                <= (
                                    second_striker_candidate_actor.maximum_foot_ball_distance_m
                                    if second_striker_candidate_actor is not None
                                    else role_ballistic_actor.maximum_foot_ball_distance_m
                                )
                            ):
                                frame_second_striker_ballistic_actor_target_conditioned = True
                                frame_second_striker_ballistic_actor_launch_envelope_supported = (
                                    diagnostic_effect.launch_envelope_supported
                                )
                                desired_launch_velocity = np.asarray(
                                    (
                                        diagnostic_effect.desired_lateral_launch_speed_mps,
                                        diagnostic_effect.desired_vertical_launch_speed_mps,
                                    ),
                                    dtype=np.float64,
                                )
                                frame_second_actor_desired_launch_yz = desired_launch_velocity
                        frame_second_striker_ballistic_actor_active = bool(
                            frame_second_striker_ballistic_actor_active
                            or (effect.active and not teacher_owns_second_contact)
                        )
                        if not teacher_owns_second_contact and float(
                            np.max(np.abs(effect.torque))
                        ) >= float(np.max(np.abs(frame_second_striker_ballistic_actor_torque))):
                            frame_second_striker_ballistic_actor_torque = effect.torque.copy()
                            frame_second_striker_ballistic_actor_force_yz_n = np.asarray(
                                (effect.lateral_force_n, effect.vertical_force_n),
                                dtype=np.float64,
                            )
                            frame_second_striker_ballistic_actor_foot_velocity_yz_mps = np.asarray(
                                (
                                    effect.foot_lateral_speed_mps,
                                    effect.foot_vertical_speed_mps,
                                ),
                                dtype=np.float64,
                            )
                    else:
                        frame_ballistic_actor_active = frame_ballistic_actor_active or effect.active
                        if float(np.max(np.abs(effect.torque))) >= float(
                            np.max(np.abs(frame_ballistic_actor_torque))
                        ):
                            frame_ballistic_actor_torque = effect.torque.copy()
                strike_route_authorized = bool(
                    robot.runtime_strike_router is None
                    or (
                        robot.runtime_strike_route_decision is not None
                        and robot.runtime_strike_route_decision.accepted
                    )
                )
                receive_route_authorized = bool(
                    (
                        robot.runtime_receive_actor is None
                        and robot.runtime_receive_probe_action is None
                    )
                    or (
                        robot.runtime_receive_decision is not None
                        and robot.runtime_receive_decision.accepted
                    )
                )
                target_route_authorized = bool(
                    robot.runtime_contact_target_actor is None
                    or (
                        robot.runtime_contact_target_decision is not None
                        and robot.runtime_contact_target_decision.accepted
                    )
                )
                finish_plan_authorized = bool(
                    robot.runtime_finish_plan_actor is None
                    or (
                        robot.runtime_finish_plan_decision is not None
                        and robot.runtime_finish_plan_decision.accepted
                    )
                )
                three_axis_contact_authorized = bool(
                    strike_route_authorized
                    and receive_route_authorized
                    and target_route_authorized
                    and finish_plan_authorized
                )
                if (
                    robot.role == "shooter"
                    and shooter_three_axis_contact_actor is not None
                    and three_axis_contact_authorized
                ):
                    bilateral_actor_kwargs = {}
                    if robot.parameters.kick_foot == "left":
                        bilateral_actor_kwargs = {
                            "striking_ankle_body_id": robot.left_ankle_body,
                            "lateral_mirror_sign": -1.0,
                        }
                    three_axis_effect = g1_three_axis_contact_effect(
                        model=model,
                        data=data,
                        right_ankle_body_id=robot.right_ankle_body,
                        actor=shooter_three_axis_contact_actor,
                        policy_frame=policy_frames[robot.role],
                        contact_observed=robot.contact_latched,
                        ball_position=np.asarray(
                            data.qpos[ball_qpos : ball_qpos + 3], dtype=np.float64
                        ),
                        actuated_dof_indices=np.asarray(robot.joint_qvel, dtype=np.int64),
                        **bilateral_actor_kwargs,
                    )
                    raw_torque = raw_torque + three_axis_effect.torque
                    frame_three_axis_contact_actor_active = bool(
                        frame_three_axis_contact_actor_active or three_axis_effect.active
                    )
                    if float(np.max(np.abs(three_axis_effect.torque))) >= float(
                        np.max(np.abs(frame_three_axis_contact_actor_torque))
                    ):
                        frame_three_axis_contact_actor_torque = three_axis_effect.torque.copy()
                        frame_three_axis_contact_actor_force_xyz_n = (
                            three_axis_effect.force_xyz_n.copy()
                        )
                        frame_three_axis_contact_actor_foot_velocity_xyz_mps = (
                            three_axis_effect.foot_velocity_xyz_mps.copy()
                        )
                if (
                    robot.role == "shooter"
                    and shooter_target_velocity_contact_actor is not None
                    and three_axis_contact_authorized
                ):
                    assert shooter_target_foot_velocity_xyz_mps is not None
                    target_actor_kwargs: dict[str, Any] = {}
                    if robot.parameters.kick_foot == "left":
                        target_actor_kwargs = {
                            "striking_ankle_body_id": robot.left_ankle_body,
                            "lateral_mirror_sign": -1.0,
                        }
                    target_effect = g1_target_velocity_contact_effect(
                        model=model,
                        data=data,
                        right_ankle_body_id=robot.right_ankle_body,
                        actor=shooter_target_velocity_contact_actor,
                        target_velocity_xyz_mps=np.asarray(
                            shooter_target_foot_velocity_xyz_mps, dtype=np.float64
                        ),
                        policy_frame=policy_frames[robot.role],
                        contact_observed=robot.contact_latched,
                        ball_position=np.asarray(
                            data.qpos[ball_qpos : ball_qpos + 3], dtype=np.float64
                        ),
                        actuated_dof_indices=np.asarray(robot.joint_qvel, dtype=np.int64),
                        **target_actor_kwargs,
                    )
                    raw_torque = raw_torque + target_effect.torque
                    frame_target_velocity_contact_actor_active = bool(
                        frame_target_velocity_contact_actor_active or target_effect.active
                    )
                    frame_target_velocity_contact_actor_target_supported = bool(
                        frame_target_velocity_contact_actor_target_supported
                        or target_effect.target_supported
                    )
                    if float(np.max(np.abs(target_effect.torque))) >= float(
                        np.max(np.abs(frame_target_velocity_contact_actor_torque))
                    ):
                        frame_target_velocity_contact_actor_torque = target_effect.torque.copy()
                        frame_target_velocity_contact_actor_force_xyz_n = (
                            target_effect.force_xyz_n.copy()
                        )
                        frame_target_velocity_contact_actor_foot_velocity_xyz_mps = (
                            target_effect.foot_velocity_xyz_mps.copy()
                        )
                        frame_target_velocity_contact_actor_target_xyz_mps = (
                            target_effect.target_velocity_xyz_mps.copy()
                        )
                if (
                    robot.role == "shooter"
                    and shooter_neural_contact_actor is not None
                    and three_axis_contact_authorized
                ):
                    assert shooter_neural_contact_policy_frame is not None
                    assert shooter_neural_contact_target_velocity_xyz_mps is not None
                    # Source labels are control-rate residuals.  Evaluate once
                    # and hold across physics substeps, as a real low-level
                    # command buffer would.
                    if substep_index == 0:
                        neural_features = neural_contact_features(
                            phase_offset_frames=float(
                                policy_frames[robot.role] - shooter_neural_contact_policy_frame
                            ),
                            target_velocity_xyz_mps=(
                                shooter_neural_contact_target_velocity_xyz_mps
                            ),
                            ball_local_position_m=robot.state.ball_pos_w,
                            ball_local_velocity_mps=robot.state.ball_vel_w,
                            joint_position_rad=q,
                            joint_velocity_rad_s=dq,
                        )
                        neural_effect = evaluate_neural_contact_actor(
                            actor=shooter_neural_contact_actor,
                            features=neural_features,
                        )
                        frame_neural_contact_actor_torque = neural_effect.torque.copy()
                        frame_neural_contact_actor_active = neural_effect.active
                        frame_neural_contact_actor_supported = neural_effect.supported
                        frame_neural_contact_actor_ood_distance = (
                            neural_effect.normalized_ood_distance
                        )
                    raw_torque = raw_torque + frame_neural_contact_actor_torque
                if robot.role == "shooter" and shooter_loft_teacher_config is not None:
                    teacher_effect = g1_loft_teacher_effect(
                        model=model,
                        data=data,
                        right_ankle_body_id=robot.right_ankle_body,
                        config=shooter_loft_teacher_config,
                        policy_frame=policy_frames[robot.role],
                        contact_observed=robot.contact_latched,
                        ball_position=np.asarray(
                            data.qpos[ball_qpos : ball_qpos + 3],
                            dtype=np.float64,
                        ),
                    )
                    raw_torque = raw_torque + teacher_effect.torque
                    frame_loft_teacher_active = frame_loft_teacher_active or teacher_effect.active
                    if float(np.max(np.abs(teacher_effect.torque))) >= float(
                        np.max(np.abs(frame_loft_teacher_torque))
                    ):
                        frame_loft_teacher_torque = teacher_effect.torque.copy()
                        frame_loft_teacher_force_xyz_n = np.asarray(
                            (
                                teacher_effect.forward_force_n,
                                teacher_effect.lateral_force_n,
                                teacher_effect.vertical_force_n,
                            ),
                            dtype=np.float64,
                        )
                        frame_loft_teacher_foot_velocity_xyz_mps = np.asarray(
                            (
                                teacher_effect.foot_forward_speed_mps,
                                teacher_effect.foot_lateral_speed_mps,
                                teacher_effect.foot_vertical_speed_mps,
                            ),
                            dtype=np.float64,
                        )
                if (
                    robot.role == "second_striker"
                    and second_striker_loft_teacher_config is not None
                    and second_ball_qpos is not None
                ):
                    second_teacher_effect = g1_loft_teacher_effect(
                        model=model,
                        data=data,
                        right_ankle_body_id=robot.right_ankle_body,
                        config=second_striker_loft_teacher_config,
                        policy_frame=policy_frames[robot.role],
                        contact_observed=robot.contact_latched,
                        ball_position=np.asarray(
                            data.qpos[second_ball_qpos : second_ball_qpos + 3],
                            dtype=np.float64,
                        ),
                        actuated_dof_indices=np.asarray(robot.joint_qvel, dtype=np.int64),
                    )
                    raw_torque = raw_torque + second_teacher_effect.torque
                    frame_second_striker_loft_teacher_active = (
                        frame_second_striker_loft_teacher_active or second_teacher_effect.active
                    )
                    if np.linalg.norm(
                        (
                            second_teacher_effect.lateral_force_n,
                            second_teacher_effect.vertical_force_n,
                        )
                    ) >= np.linalg.norm(frame_second_striker_loft_teacher_force_yz_n):
                        frame_second_striker_loft_teacher_force_yz_n = np.asarray(
                            (
                                second_teacher_effect.lateral_force_n,
                                second_teacher_effect.vertical_force_n,
                            ),
                            dtype=np.float64,
                        )
                        frame_second_striker_loft_teacher_foot_velocity_yz_mps = np.asarray(
                            (
                                second_teacher_effect.foot_lateral_speed_mps,
                                second_teacher_effect.foot_vertical_speed_mps,
                            ),
                            dtype=np.float64,
                        )
                role_contact_torque_config = (
                    second_striker_ballistic_contact_torque_config
                    if robot.role == "second_striker"
                    and second_striker_ballistic_contact_torque_config is not None
                    else shooter_ballistic_contact_torque_config
                )
                runtime_contact_authorized = bool(
                    robot.role != "shooter"
                    or robot.runtime_strike_router is None
                    or (
                        robot.runtime_strike_route_decision is not None
                        and robot.runtime_strike_route_decision.accepted
                    )
                )
                if robot.role in {"shooter", "second_striker"} and (
                    role_contact_torque_config is not None and runtime_contact_authorized
                ):
                    contact_torque, contact_torque_active = g1_ballistic_contact_torque_residual(
                        policy_frame=policy_frames[robot.role],
                        control_dt_sec=_CONTROL_DT,
                        config=role_contact_torque_config,
                        kick_foot=robot.parameters.kick_foot,
                    )
                    raw_torque = raw_torque + contact_torque
                    if robot.role == "second_striker":
                        frame_second_striker_ballistic_contact_torque_active = (
                            frame_second_striker_ballistic_contact_torque_active
                            or contact_torque_active
                        )
                        if float(np.max(np.abs(contact_torque))) >= float(
                            np.max(np.abs(frame_second_striker_ballistic_contact_torque))
                        ):
                            frame_second_striker_ballistic_contact_torque = contact_torque.copy()
                    else:
                        frame_ballistic_contact_torque_active = (
                            frame_ballistic_contact_torque_active or contact_torque_active
                        )
                        if float(np.max(np.abs(contact_torque))) >= float(
                            np.max(np.abs(frame_ballistic_contact_torque))
                        ):
                            frame_ballistic_contact_torque = contact_torque.copy()
                if robot.role == "shooter" and shooter_first_touch_interception_config is not None:
                    striking_ankle = (
                        robot.left_ankle_body
                        if robot.parameters.kick_foot == "left"
                        else robot.right_ankle_body
                    )
                    interception = first_touch_interception_effect(
                        model=model,
                        data=data,
                        striking_ankle_body_id=striking_ankle,
                        actuated_dof_indices=np.asarray(robot.joint_qvel, dtype=np.int64),
                        ball_position_m=np.asarray(
                            data.qpos[ball_qpos : ball_qpos + 3], dtype=np.float64
                        ),
                        ball_velocity_mps=np.asarray(
                            data.qvel[ball_qvel : ball_qvel + 3], dtype=np.float64
                        ),
                        policy_frame=policy_frames[robot.role],
                        contact_observed=robot.contact_latched,
                        kick_foot=robot.parameters.kick_foot,
                        config=shooter_first_touch_interception_config,
                    )
                    raw_torque = raw_torque + interception.torque
                    frame_first_touch_interception_active = (
                        frame_first_touch_interception_active or interception.active
                    )
                    if float(np.max(np.abs(interception.torque))) >= float(
                        np.max(np.abs(frame_first_touch_interception_torque))
                    ):
                        frame_first_touch_interception_torque = interception.torque.copy()
                        frame_first_touch_interception_force = interception.task_force_n.copy()
                        frame_first_touch_interception_error = interception.position_error_m.copy()
                if robot.role == "passer" and passer_reception_interception_config is not None:
                    left_distance = float(
                        np.linalg.norm(
                            np.asarray(data.xpos[robot.left_ankle_body], dtype=np.float64)
                            - np.asarray(data.qpos[ball_qpos : ball_qpos + 3], dtype=np.float64)
                        )
                    )
                    right_distance = float(
                        np.linalg.norm(
                            np.asarray(data.xpos[robot.right_ankle_body], dtype=np.float64)
                            - np.asarray(data.qpos[ball_qpos : ball_qpos + 3], dtype=np.float64)
                        )
                    )
                    reception_foot = "left" if left_distance <= right_distance else "right"
                    reception_ankle = (
                        robot.left_ankle_body
                        if reception_foot == "left"
                        else robot.right_ankle_body
                    )
                    reception = first_touch_interception_effect(
                        model=model,
                        data=data,
                        striking_ankle_body_id=reception_ankle,
                        actuated_dof_indices=np.asarray(robot.joint_qvel, dtype=np.int64),
                        ball_position_m=np.asarray(
                            data.qpos[ball_qpos : ball_qpos + 3], dtype=np.float64
                        ),
                        ball_velocity_mps=np.asarray(
                            data.qvel[ball_qvel : ball_qvel + 3], dtype=np.float64
                        ),
                        policy_frame=(passer_reception_interception_config.start_policy_frame),
                        contact_observed=robot.contact_latched,
                        kick_foot=reception_foot,
                        config=passer_reception_interception_config,
                    )
                    raw_torque = raw_torque + reception.torque
                    frame_passer_reception_interception_active = (
                        frame_passer_reception_interception_active or reception.active
                    )
                    if float(np.max(np.abs(reception.torque))) >= float(
                        np.max(np.abs(frame_passer_reception_interception_torque))
                    ):
                        frame_passer_reception_interception_torque = reception.torque.copy()
                        frame_passer_reception_interception_force = reception.task_force_n.copy()
                        frame_passer_reception_interception_error = (
                            reception.position_error_m.copy()
                        )
                safety_projected = raw_torque
                # The candidate is a post-impact recovery module.  Keeping the
                # guard behind the measured contact latch preserves the frozen
                # kick prior and prevents recovery search from gaming accuracy.
                if robot.joint_guard_enabled and (
                    robot.contact_latched
                    or robot.role == "goalkeeper"
                    or (robot.role == "passer" and passer_precontact_joint_guard_enabled)
                    or (robot.role == "shooter" and shooter_precontact_joint_guard_enabled)
                ):
                    goalkeeper_impact_imminent = bool(
                        robot.role == "goalkeeper"
                        and goalkeeper_config is not None
                        and goalkeeper_actor_observation is not None
                        and goalkeeper_config.joint_guard_impact_lead_sec > 0.0
                        and 0.0
                        <= goalkeeper_actor_observation.estimated_intercept[0]
                        <= goalkeeper_config.joint_guard_impact_lead_sec
                    )
                    if (
                        robot.role == "goalkeeper"
                        and (robot.contact_latched or goalkeeper_impact_imminent)
                        and robot.joint_guard_late_config is not None
                    ):
                        robot.selected_joint_guard_config = robot.joint_guard_late_config
                        robot.joint_guard_route = "goalkeeper_impact"
                    elif robot.selected_joint_guard_config is None:
                        (
                            robot.selected_joint_guard_config,
                            robot.joint_guard_route,
                        ) = _select_joint_guard_config(
                            standard=robot.joint_guard_config,
                            late_arrival=robot.joint_guard_late_config,
                            phase_advance_count=robot.phase_advance_count,
                        )
                    guard_config = robot.selected_joint_guard_config
                    safety_projected, active = _project_joint_safe_torque(
                        joint_position=q,
                        joint_velocity=dq,
                        commanded_torque=raw_torque,
                        joint_ranges=model.jnt_range[robot.joint_ids],
                        limited=model.jnt_limited[robot.joint_ids].astype(bool),
                        margin_rad=guard_config.margin_rad,
                        prediction_horizon_sec=guard_config.prediction_horizon_sec,
                        boundary_kp=guard_config.boundary_kp,
                        boundary_kd=guard_config.boundary_kd,
                    )
                    joint_guard_active[robot.role] = joint_guard_active[robot.role] or active
                torque = np.clip(safety_projected, -guarded_limits, guarded_limits)
                commanded_torque[robot.role] = raw_torque.copy()
                projected_torque[robot.role] = torque.copy()
                torque_violation = torque_violation or bool(np.any(np.abs(torque) > hard_limits))
                actuator_saturation = actuator_saturation or bool(
                    np.any(np.abs(torque) >= hard_limits * 0.999)
                )
                data.ctrl[robot.actuators] = torque
            if unified_stadium_scene:
                apply_g1_compliant_goal_net_force(
                    data,
                    ball_body_id=ball_body,
                    ball_qpos=ball_qpos,
                    ball_qvel=ball_qvel,
                    spec=active_goal,
                    capture_depth_m=max(0.20, 0.80 * active_goal.depth_m),
                    stiffness_n_m=180.0,
                    damping_n_s_m=10.0,
                    state=goal_net_state,
                )
                if (
                    second_ball_body is not None
                    and second_ball_qpos is not None
                    and second_ball_qvel is not None
                ):
                    apply_g1_compliant_goal_net_force(
                        data,
                        ball_body_id=second_ball_body,
                        ball_qpos=second_ball_qpos,
                        ball_qvel=second_ball_qvel,
                        spec=active_goal,
                        capture_depth_m=max(0.20, 0.80 * active_goal.depth_m),
                        stiffness_n_m=180.0,
                        damping_n_s_m=10.0,
                        state=second_goal_net_state,
                    )
            if (
                second_threat_force is not None
                and second_threat_force_stop_sec is not None
                and float(data.time) < second_threat_force_stop_sec - 1.0e-12
            ):
                # The compliant net owns the base ball-force buffer and clears
                # it on every physics substep.  Add the bounded curriculum
                # impulse afterwards so the recorded launcher event is also a
                # real force in the MuJoCo integration, while preserving any
                # simultaneous net response.
                data.xfrc_applied[ball_body, :3] += second_threat_force
            mujoco.mj_step(model, data)
            for robot in robots:
                executed_torque[robot.role] = data.actuator_force[robot.actuators].copy()
            observation = _contacts(
                model=model,
                data=data,
                ball_geom=ball_geom,
                floor_geom=floor_geom,
                passer_geoms=passer_geoms,
                shooter_geoms=shooter_geoms,
                goalkeeper_geoms=goalkeeper_geoms,
                goalkeeper_left_glove_geoms=goalkeeper_left_glove_geoms,
                goalkeeper_right_glove_geoms=goalkeeper_right_glove_geoms,
            )
            second_observation: dict[str, Any] | None = None
            if second_ball_geom is not None and second_striker is not None:
                second_observation = _contacts(
                    model=model,
                    data=data,
                    ball_geom=second_ball_geom,
                    floor_geom=floor_geom,
                    passer_geoms=passer_geoms | shooter_geoms,
                    shooter_geoms=second_striker_geoms,
                    goalkeeper_geoms=goalkeeper_geoms,
                    goalkeeper_left_glove_geoms=goalkeeper_left_glove_geoms,
                    goalkeeper_right_glove_geoms=goalkeeper_right_glove_geoms,
                    shooter_geom_prefix="second_striker_",
                )
            passer_support = (
                passer_support[0] or observation["passer_left"],
                passer_support[1] or observation["passer_right"],
            )
            shooter_support = (
                shooter_support[0] or observation["shooter_left"],
                shooter_support[1] or observation["shooter_right"],
            )
            goalkeeper_support = (
                goalkeeper_support[0] or observation["goalkeeper_left"],
                goalkeeper_support[1] or observation["goalkeeper_right"],
            )
            frame_robot_contacts += int(observation["robot_robot"])
            if observation["ball_passer"]:
                contact_role = 1
                impulse = float(observation["ball_passer_force_n"]) * _PHYSICS_DT
                passer.contact_impulse_ns += impulse
                frame_contact_impulse["passer"] += impulse
                if not passer.contact_latched:
                    passer.contact_latched = True
                    passer.contact_time = float(data.time)
                    _reset_post_contact_support_anchors(passer)
            if observation["ball_shooter"]:
                contact_role = 2
                if observation["ball_shooter_left"]:
                    shooter_contact_foot = -1
                elif observation["ball_shooter_right"]:
                    shooter_contact_foot = 1
                impulse = float(observation["ball_shooter_force_n"]) * _PHYSICS_DT
                shooter.contact_impulse_ns += impulse
                frame_contact_impulse["shooter"] += impulse
                if not shooter.contact_latched:
                    pass_delivery_position = np.asarray(
                        data.qpos[ball_qpos : ball_qpos + 3], dtype=np.float64
                    ).copy()
                    shooter.contact_latched = True
                    shooter.contact_time = float(data.time)
                    if shooter.causal_strike_option is not None:
                        shooter.causal_strike_option.observe_contact()
                    _reset_post_contact_support_anchors(shooter)
            if (
                goalkeeper is not None
                and observation["ball_goalkeeper"]
                and (second_striker is None or not second_striker.contact_latched)
            ):
                contact_role = 3
                if not goalkeeper.contact_latched:
                    goalkeeper.contact_latched = True
                    goalkeeper.contact_time = float(data.time)
                    goalkeeper_contact_epoch += 1
                    if goalkeeper_contact_epoch == 1:
                        goalkeeper_contact_time = float(data.time)
                    elif goalkeeper_contact_epoch == 2:
                        goalkeeper_second_contact_time = float(data.time)
                if goalkeeper_contact_epoch == 1 and observation["ball_goalkeeper_left_glove"]:
                    goalkeeper_left_glove_contact_time = (
                        float(data.time)
                        if goalkeeper_left_glove_contact_time is None
                        else goalkeeper_left_glove_contact_time
                    )
                if goalkeeper_contact_epoch == 1 and observation["ball_goalkeeper_right_glove"]:
                    goalkeeper_right_glove_contact_time = (
                        float(data.time)
                        if goalkeeper_right_glove_contact_time is None
                        else goalkeeper_right_glove_contact_time
                    )
                if (
                    goalkeeper_contact_epoch == 2
                    and observation["ball_goalkeeper_glove"]
                    and goalkeeper_second_glove_contact_time is None
                ):
                    goalkeeper_second_glove_contact_time = float(data.time)
                    goalkeeper_second_glove_contact_position = np.asarray(
                        data.qpos[ball_qpos : ball_qpos + 3], dtype=np.float64
                    ).copy()
                    goalkeeper_second_glove_contact_height = float(
                        goalkeeper_second_glove_contact_position[2]
                    )
                    second_left = bool(observation["ball_goalkeeper_left_glove"])
                    second_right = bool(observation["ball_goalkeeper_right_glove"])
                    goalkeeper_second_glove_contact_side = (
                        "both"
                        if second_left and second_right
                        else "left"
                        if second_left
                        else "right"
                    )
                    second_surface_distances = [
                        float(value)
                        for value in (
                            observation["ball_goalkeeper_left_glove_surface_distance_m"],
                            observation["ball_goalkeeper_right_glove_surface_distance_m"],
                        )
                        if value is not None
                    ]
                    goalkeeper_second_glove_contact_surface_distance = min(second_surface_distances)
                    second_left_hand = np.asarray(
                        data.xpos[goalkeeper.left_hand_body], dtype=np.float64
                    )
                    second_right_hand = np.asarray(
                        data.xpos[goalkeeper.right_hand_body], dtype=np.float64
                    )
                    goalkeeper_second_contact_left_hand_height = float(second_left_hand[2])
                    goalkeeper_second_contact_right_hand_height = float(second_right_hand[2])
                    goalkeeper_second_contact_left_hand_ball_distance = float(
                        np.linalg.norm(second_left_hand - goalkeeper_second_glove_contact_position)
                    )
                    goalkeeper_second_contact_right_hand_ball_distance = float(
                        np.linalg.norm(second_right_hand - goalkeeper_second_glove_contact_position)
                    )
                if (
                    goalkeeper_contact_epoch == 1
                    and observation["ball_goalkeeper_glove"]
                    and goalkeeper_glove_contact_time is None
                ):
                    goalkeeper_glove_contact_time = float(data.time)
                    goalkeeper_glove_contact_position = np.asarray(
                        data.qpos[ball_qpos : ball_qpos + 3], dtype=np.float64
                    ).copy()
                    goalkeeper_glove_contact_height = float(goalkeeper_glove_contact_position[2])
                    left_active = bool(observation["ball_goalkeeper_left_glove"])
                    right_active = bool(observation["ball_goalkeeper_right_glove"])
                    if left_active and right_active:
                        goalkeeper_glove_contact_side = "both"
                    elif left_active:
                        goalkeeper_glove_contact_side = "left"
                    else:
                        goalkeeper_glove_contact_side = "right"
                    surface_distances = [
                        float(value)
                        for value in (
                            observation["ball_goalkeeper_left_glove_surface_distance_m"],
                            observation["ball_goalkeeper_right_glove_surface_distance_m"],
                        )
                        if value is not None
                    ]
                    goalkeeper_glove_contact_surface_distance = min(surface_distances)
                    ball_at_contact = goalkeeper_glove_contact_position
                    left_hand_at_contact = np.asarray(
                        data.xpos[goalkeeper.left_hand_body], dtype=np.float64
                    )
                    right_hand_at_contact = np.asarray(
                        data.xpos[goalkeeper.right_hand_body], dtype=np.float64
                    )
                    goalkeeper_contact_left_hand_height = float(left_hand_at_contact[2])
                    goalkeeper_contact_right_hand_height = float(right_hand_at_contact[2])
                    goalkeeper_contact_left_hand_ball_distance = float(
                        np.linalg.norm(left_hand_at_contact - ball_at_contact)
                    )
                    goalkeeper_contact_right_hand_ball_distance = float(
                        np.linalg.norm(right_hand_at_contact - ball_at_contact)
                    )
                    left_distance = float(
                        np.linalg.norm(
                            np.asarray(data.xpos[goalkeeper.left_hand_body], dtype=np.float64)
                            - np.asarray(data.qpos[ball_qpos : ball_qpos + 3], dtype=np.float64)
                        )
                    )
                    right_distance = float(
                        np.linalg.norm(
                            np.asarray(data.xpos[goalkeeper.right_hand_body], dtype=np.float64)
                            - np.asarray(data.qpos[ball_qpos : ball_qpos + 3], dtype=np.float64)
                        )
                    )
                    goalkeeper_bimanual_window_observed = bool(
                        float(data.xpos[goalkeeper.left_hand_body, 2]) >= 1.05
                        and float(data.xpos[goalkeeper.right_hand_body, 2]) >= 1.05
                        and max(left_distance, right_distance) <= 0.38
                    )
            if second_observation is not None and second_striker is not None:
                frame_robot_contacts += int(second_observation["robot_robot"])
                if not second_striker.contact_latched and second_ball_geom is not None:
                    for contact_index in range(int(data.ncon)):
                        candidate_contact = data.contact[contact_index]
                        geoms = {
                            int(candidate_contact.geom1),
                            int(candidate_contact.geom2),
                        }
                        if second_ball_geom not in geoms:
                            continue
                        other_geom = next(geom for geom in geoms if geom != second_ball_geom)
                        if other_geom == floor_geom or other_geom in second_striker_geoms:
                            continue
                        other_name = (
                            mujoco.mj_id2name(
                                model,
                                mujoco.mjtObj.mjOBJ_GEOM,
                                other_geom,
                            )
                            or f"geom-{other_geom}"
                        )
                        second_striker_unexpected_precontact_collision_geoms.add(other_name)
                if second_observation["ball_shooter"]:
                    contact_role = 4
                    frame_second_striker_contact = True
                    frame_second_striker_contact_force_n = max(
                        frame_second_striker_contact_force_n,
                        float(second_observation["ball_shooter_force_n"]),
                    )
                    second_striker_contact_force_peak = max(
                        second_striker_contact_force_peak,
                        float(second_observation["ball_shooter_force_n"]),
                    )
                    if not second_striker.contact_latched:
                        second_striker.contact_latched = True
                        second_striker.contact_time = float(data.time)
                        second_striker_contact_foot = (
                            "left"
                            if second_observation["ball_shooter_left"]
                            else "right"
                            if second_observation["ball_shooter_right"]
                            else None
                        )
                        _reset_post_contact_support_anchors(second_striker)
                        if goalkeeper_observer is not None:
                            # The physical foot collision—not a configured
                            # timestamp—starts threat epoch two.
                            goalkeeper_observer.rearm()
                            goalkeeper_previous_actor_residual.fill(0.0)
                if (
                    goalkeeper is not None
                    and second_striker.contact_latched
                    and second_observation["ball_goalkeeper"]
                ):
                    if not goalkeeper.contact_latched:
                        goalkeeper.contact_latched = True
                        goalkeeper.contact_time = float(data.time)
                        goalkeeper_contact_epoch = 2
                        goalkeeper_second_contact_time = float(data.time)
                    if (
                        second_observation["ball_goalkeeper_glove"]
                        and goalkeeper_second_glove_contact_time is None
                    ):
                        assert second_ball_qpos is not None
                        goalkeeper_second_glove_contact_time = float(data.time)
                        goalkeeper_second_glove_contact_position = np.asarray(
                            data.qpos[second_ball_qpos : second_ball_qpos + 3],
                            dtype=np.float64,
                        ).copy()
                        goalkeeper_second_glove_contact_height = float(
                            goalkeeper_second_glove_contact_position[2]
                        )
                        second_left = bool(second_observation["ball_goalkeeper_left_glove"])
                        second_right = bool(second_observation["ball_goalkeeper_right_glove"])
                        goalkeeper_second_glove_contact_side = (
                            "both"
                            if second_left and second_right
                            else "left"
                            if second_left
                            else "right"
                        )
                        second_surface_distances = [
                            float(value)
                            for value in (
                                second_observation["ball_goalkeeper_left_glove_surface_distance_m"],
                                second_observation[
                                    "ball_goalkeeper_right_glove_surface_distance_m"
                                ],
                            )
                            if value is not None
                        ]
                        goalkeeper_second_glove_contact_surface_distance = min(
                            second_surface_distances
                        )
                        second_left_hand = np.asarray(
                            data.xpos[goalkeeper.left_hand_body], dtype=np.float64
                        )
                        second_right_hand = np.asarray(
                            data.xpos[goalkeeper.right_hand_body], dtype=np.float64
                        )
                        goalkeeper_second_contact_left_hand_height = float(second_left_hand[2])
                        goalkeeper_second_contact_right_hand_height = float(second_right_hand[2])
                        goalkeeper_second_contact_left_hand_ball_distance = float(
                            np.linalg.norm(
                                second_left_hand - goalkeeper_second_glove_contact_position
                            )
                        )
                        goalkeeper_second_contact_right_hand_ball_distance = float(
                            np.linalg.norm(
                                second_right_hand - goalkeeper_second_glove_contact_position
                            )
                        )
            active_contact_observation = (
                second_observation
                if second_striker is not None and second_striker.contact_latched
                else observation
            )
            active_contact_qpos = (
                cast(int, second_ball_qpos)
                if second_striker is not None and second_striker.contact_latched
                else ball_qpos
            )
            active_contact_qvel = (
                cast(int, second_ball_qvel)
                if second_striker is not None and second_striker.contact_latched
                else ball_qvel
            )
            active_contact_time = (
                goalkeeper_second_glove_contact_time
                if second_striker is not None and second_striker.contact_latched
                else goalkeeper_glove_contact_time
            )
            active_shot_latched = bool(
                shooter.contact_latched
                or (second_striker is not None and second_striker.contact_latched)
            )
            if (
                goalkeeper is not None
                and active_shot_latched
                and active_contact_observation is not None
            ):
                ball_keeper_separation_x = abs(
                    float(data.qpos[active_contact_qpos]) - float(data.qpos[goalkeeper.qpos_base])
                )
                recent_glove_contact = bool(
                    active_contact_time is not None
                    and float(data.time) <= active_contact_time + 0.80
                )
                if ball_keeper_separation_x <= 1.40 or recent_glove_contact:
                    goalkeeper_contact_trace["goalkeeper_contact_window_time"].append(
                        float(data.time)
                    )
                    goalkeeper_contact_trace["goalkeeper_contact_window_ball_pose"].append(
                        data.qpos[active_contact_qpos : active_contact_qpos + 7].copy()
                    )
                    goalkeeper_contact_trace["goalkeeper_contact_window_ball_velocity"].append(
                        data.qvel[active_contact_qvel : active_contact_qvel + 6].copy()
                    )
                    goalkeeper_contact_trace["goalkeeper_contact_window_pelvis_pose"].append(
                        data.qpos[goalkeeper.qpos_base : goalkeeper.qpos_base + 7].copy()
                    )
                    goalkeeper_contact_trace["goalkeeper_contact_window_joint_position"].append(
                        data.qpos[goalkeeper.joint_qpos].copy()
                    )
                    goalkeeper_contact_trace["goalkeeper_contact_window_left_hand_position"].append(
                        data.xpos[goalkeeper.left_hand_body].copy()
                    )
                    goalkeeper_contact_trace[
                        "goalkeeper_contact_window_right_hand_position"
                    ].append(data.xpos[goalkeeper.right_hand_body].copy())
                    goalkeeper_contact_trace["goalkeeper_contact_window_left_glove_contact"].append(
                        bool(active_contact_observation["ball_goalkeeper_left_glove"])
                    )
                    goalkeeper_contact_trace[
                        "goalkeeper_contact_window_right_glove_contact"
                    ].append(bool(active_contact_observation["ball_goalkeeper_right_glove"]))
                    goalkeeper_contact_trace[
                        "goalkeeper_contact_window_left_surface_distance_m"
                    ].append(
                        np.nan
                        if active_contact_observation[
                            "ball_goalkeeper_left_glove_surface_distance_m"
                        ]
                        is None
                        else float(
                            active_contact_observation[
                                "ball_goalkeeper_left_glove_surface_distance_m"
                            ]
                        )
                    )
                    goalkeeper_contact_trace[
                        "goalkeeper_contact_window_right_surface_distance_m"
                    ].append(
                        np.nan
                        if active_contact_observation[
                            "ball_goalkeeper_right_glove_surface_distance_m"
                        ]
                        is None
                        else float(
                            active_contact_observation[
                                "ball_goalkeeper_right_glove_surface_distance_m"
                            ]
                        )
                    )
        goalkeeper_bimanual_punch_frames += int(frame_goalkeeper_bimanual_punch_active)
        goalkeeper_bimanual_punch_peak_torque = max(
            goalkeeper_bimanual_punch_peak_torque,
            float(np.max(np.abs(frame_goalkeeper_bimanual_punch_torque))),
        )
        passer.latest_left_support, passer.latest_right_support = passer_support
        shooter.latest_left_support, shooter.latest_right_support = shooter_support
        if goalkeeper is not None:
            goalkeeper.latest_left_support, goalkeeper.latest_right_support = goalkeeper_support
        _update_support_slip(passer, data, passer_support)
        _update_support_slip(shooter, data, shooter_support)
        if goalkeeper is not None:
            _update_support_slip(goalkeeper, data, goalkeeper_support)
            if goalkeeper_balanced_dive_flight_start_sec is not None and not any(
                goalkeeper_support
            ):
                goalkeeper_balanced_dive_was_airborne = True
            proprioceptive_capture = bool(
                goalkeeper_config is not None
                and goalkeeper_config.post_contact_proprioceptive_capture_enabled
                and goalkeeper.contact_latched
                and goalkeeper.contact_time is not None
                and float(data.time) - goalkeeper.contact_time
                >= goalkeeper_config.post_contact_proprioceptive_capture_delay_sec
                and float(data.time) - goalkeeper.contact_time
                <= goalkeeper_config.post_contact_proprioceptive_capture_window_sec
                and all(goalkeeper_support)
                and float(
                    np.linalg.norm(
                        np.asarray(
                            data.qvel[goalkeeper.qvel_base : goalkeeper.qvel_base + 3],
                            dtype=np.float64,
                        )
                    )
                )
                <= goalkeeper_config.post_contact_proprioceptive_capture_maximum_root_speed_mps
            )
            if (
                goalkeeper_config is not None
                and goalkeeper_config.balanced_dive_landing_capture_enabled
                and goalkeeper.contact_latched
                and goalkeeper_landing_capture_start_sec is None
                and (
                    (goalkeeper_balanced_dive_was_airborne and any(goalkeeper_support))
                    or proprioceptive_capture
                )
            ):
                goalkeeper_landing_capture_start_sec = float(data.time)
                goalkeeper_landing_capture_anchor = np.asarray(
                    data.qpos[goalkeeper.joint_qpos],
                    dtype=np.float64,
                ).copy()
                goalkeeper_landing_capture_duration_sec = (
                    goalkeeper_config.balanced_dive_landing_capture_sec
                    if goalkeeper_balanced_dive_was_airborne
                    else goalkeeper_config.post_contact_proprioceptive_capture_duration_sec
                )
                goalkeeper_landing_capture_proprioceptive = bool(
                    proprioceptive_capture and not goalkeeper_balanced_dive_was_airborne
                )
        for robot in robots:
            robot.joint_guard_frame_count += int(joint_guard_active[robot.role])
        robot_robot_contact_count += frame_robot_contacts

        ball_position = data.qpos[ball_qpos : ball_qpos + 3].copy()
        ball_velocity = data.qvel[ball_qvel : ball_qvel + 3].copy()
        ball_speed = float(np.linalg.norm(ball_velocity))
        if passer.contact_latched and not shooter.contact_latched:
            pass_peak_speed = max(pass_peak_speed, ball_speed)
        if shooter.contact_latched:
            shot_peak_speed = max(shot_peak_speed, ball_speed)
        if not goal_plane_crossed and previous_ball_x < active_goal.plane_x_m <= float(
            ball_position[0]
        ):
            goal_plane_crossed = True
            crossing_y = float(ball_position[1])
            crossing_z = float(ball_position[2])
            goal_crossed = g1_ball_inside_goal_mouth(
                active_goal,
                ball_y_m=crossing_y,
                ball_z_m=crossing_z,
            )
        previous_ball_x = float(ball_position[0])

        if (
            second_striker is not None
            and second_ball_qpos is not None
            and second_ball_qvel is not None
            and previous_second_ball_x is not None
        ):
            second_position = data.qpos[second_ball_qpos : second_ball_qpos + 3].copy()
            second_velocity = data.qvel[second_ball_qvel : second_ball_qvel + 3].copy()
            second_speed = float(np.linalg.norm(second_velocity))
            if second_striker.contact_latched:
                second_striker_postcontact_peak_speed = max(
                    second_striker_postcontact_peak_speed,
                    second_speed,
                )
                second_striker_postcontact_peak_forward_speed = max(
                    second_striker_postcontact_peak_forward_speed,
                    float(second_velocity[0]),
                )
            else:
                second_striker_precontact_peak_speed = max(
                    second_striker_precontact_peak_speed,
                    second_speed,
                )
            if (
                not second_ball_goal_plane_crossed
                and previous_second_ball_x < active_goal.plane_x_m <= float(second_position[0])
            ):
                second_ball_goal_plane_crossed = True
                second_ball_crossing_y = float(second_position[1])
                second_ball_crossing_z = float(second_position[2])
                second_ball_goal_crossed = g1_ball_inside_goal_mouth(
                    active_goal,
                    ball_y_m=second_ball_crossing_y,
                    ball_z_m=second_ball_crossing_z,
                )
            previous_second_ball_x = float(second_position[0])
            second_striker_min_height = min(
                second_striker_min_height,
                float(data.qpos[second_striker.qpos_base + 2]),
            )

        heights = (float(data.qpos[passer.qpos_base + 2]), float(data.qpos[2]))
        passer_min_height = min(passer_min_height, heights[0])
        shooter_min_height = min(shooter_min_height, heights[1])
        if goalkeeper is not None:
            goalkeeper_min_height = min(
                goalkeeper_min_height,
                float(data.qpos[goalkeeper.qpos_base + 2]),
            )
            goalkeeper_peak_lateral_speed = max(
                goalkeeper_peak_lateral_speed,
                abs(float(data.qvel[goalkeeper.qvel_base + 1])),
            )
        passer_roll, passer_pitch = _roll_pitch(data.xquat[passer.torso_body])
        shooter_roll, shooter_pitch = _roll_pitch(data.xquat[shooter.torso_body])
        passer_roll_peak = max(passer_roll_peak, abs(passer_roll))
        passer_pitch_peak = max(passer_pitch_peak, abs(passer_pitch))
        shooter_roll_peak = max(shooter_roll_peak, abs(shooter_roll))
        shooter_pitch_peak = max(shooter_pitch_peak, abs(shooter_pitch))
        for robot in robots:
            ranges = model.jnt_range[robot.joint_ids]
            limited = model.jnt_limited[robot.joint_ids].astype(bool)
            q = data.qpos[robot.joint_qpos]
            role_violation = bool(
                np.any(q[limited] < ranges[limited, 0] - 1e-5)
                or np.any(q[limited] > ranges[limited, 1] + 1e-5)
            )
            role_joint_violation[robot.role] = role_joint_violation[robot.role] or role_violation
            joint_violation = joint_violation or role_violation
        finite = finite and all(
            np.all(np.isfinite(value)) for value in (data.qpos, data.qvel, data.ctrl, ball_position)
        )
        trace["shooter_ballistic_actor_active"].append(frame_ballistic_actor_active)
        trace["shooter_ballistic_actor_torque"].append(frame_ballistic_actor_torque)
        trace["shooter_three_axis_contact_actor_active"].append(
            frame_three_axis_contact_actor_active
        )
        trace["shooter_three_axis_contact_actor_torque"].append(
            frame_three_axis_contact_actor_torque
        )
        trace["shooter_three_axis_contact_actor_force_xyz_n"].append(
            frame_three_axis_contact_actor_force_xyz_n
        )
        trace["shooter_three_axis_contact_actor_foot_velocity_xyz_mps"].append(
            frame_three_axis_contact_actor_foot_velocity_xyz_mps
        )
        trace["shooter_target_velocity_contact_actor_active"].append(
            frame_target_velocity_contact_actor_active
        )
        trace["shooter_target_velocity_contact_actor_torque"].append(
            frame_target_velocity_contact_actor_torque
        )
        trace["shooter_target_velocity_contact_actor_force_xyz_n"].append(
            frame_target_velocity_contact_actor_force_xyz_n
        )
        trace["shooter_target_velocity_contact_actor_foot_velocity_xyz_mps"].append(
            frame_target_velocity_contact_actor_foot_velocity_xyz_mps
        )
        trace["shooter_target_velocity_contact_actor_target_xyz_mps"].append(
            frame_target_velocity_contact_actor_target_xyz_mps
        )
        trace["shooter_target_velocity_contact_actor_target_supported"].append(
            frame_target_velocity_contact_actor_target_supported
        )
        trace["shooter_neural_contact_actor_active"].append(frame_neural_contact_actor_active)
        trace["shooter_neural_contact_actor_torque"].append(frame_neural_contact_actor_torque)
        trace["shooter_neural_contact_actor_supported"].append(frame_neural_contact_actor_supported)
        trace["shooter_neural_contact_actor_ood_distance"].append(
            frame_neural_contact_actor_ood_distance
        )
        trace["shooter_ballistic_contact_active"].append(shooter_ballistic_contact_active)
        trace["shooter_ballistic_contact_target_delta"].append(
            shooter_ballistic_contact_target_delta
        )
        trace["shooter_ballistic_contact_torque_active"].append(
            frame_ballistic_contact_torque_active
        )
        trace["shooter_ballistic_contact_torque"].append(frame_ballistic_contact_torque)
        trace["shooter_first_touch_interception_active"].append(
            frame_first_touch_interception_active
        )
        trace["shooter_first_touch_interception_torque"].append(
            frame_first_touch_interception_torque
        )
        trace["shooter_first_touch_interception_force"].append(frame_first_touch_interception_force)
        trace["shooter_first_touch_interception_error"].append(frame_first_touch_interception_error)
        trace["passer_reception_interception_active"].append(
            frame_passer_reception_interception_active
        )
        trace["passer_reception_interception_torque"].append(
            frame_passer_reception_interception_torque
        )
        trace["passer_reception_interception_force"].append(
            frame_passer_reception_interception_force
        )
        trace["passer_reception_interception_error"].append(
            frame_passer_reception_interception_error
        )
        trace["shooter_loft_teacher_active"].append(frame_loft_teacher_active)
        trace["shooter_loft_teacher_torque"].append(frame_loft_teacher_torque)
        trace["shooter_loft_teacher_force_xyz_n"].append(frame_loft_teacher_force_xyz_n)
        trace["shooter_loft_teacher_foot_velocity_xyz_mps"].append(
            frame_loft_teacher_foot_velocity_xyz_mps
        )
        if (
            second_striker is not None
            and second_ball_qpos is not None
            and second_ball_qvel is not None
        ):
            trace["second_ball_pose"].append(
                data.qpos[second_ball_qpos : second_ball_qpos + 7].copy()
            )
            trace["second_ball_velocity"].append(
                data.qvel[second_ball_qvel : second_ball_qvel + 6].copy()
            )
            trace["second_striker_pelvis_pose"].append(
                data.qpos[second_striker.qpos_base : second_striker.qpos_base + 7].copy()
            )
            trace["second_striker_left_foot_position"].append(
                data.xpos[second_striker.left_ankle_body].copy()
            )
            trace["second_striker_right_foot_position"].append(
                data.xpos[second_striker.right_ankle_body].copy()
            )
            trace["second_striker_joint_position"].append(
                data.qpos[second_striker.joint_qpos].copy()
            )
            trace["second_striker_joint_velocity"].append(
                data.qvel[second_striker.joint_qvel].copy()
            )
            trace["second_striker_commanded_torque"].append(
                commanded_torque["second_striker"].copy()
            )
            trace["second_striker_safety_projected_torque"].append(
                projected_torque["second_striker"].copy()
            )
            trace["second_striker_executed_torque"].append(executed_torque["second_striker"].copy())
            trace["second_striker_policy_action"].append(second_striker.last_target.copy())
            trace["second_striker_policy_frame"].append(policy_frames["second_striker"])
            trace["second_striker_foot_contact"].append(frame_second_striker_contact)
            trace["second_striker_contact_force_n"].append(frame_second_striker_contact_force_n)
            trace["second_striker_ballistic_actor_active"].append(
                frame_second_striker_ballistic_actor_active
            )
            trace["second_striker_ballistic_actor_torque"].append(
                frame_second_striker_ballistic_actor_torque
            )
            trace["second_striker_ballistic_actor_force_yz_n"].append(
                frame_second_striker_ballistic_actor_force_yz_n
            )
            trace["second_striker_ballistic_actor_foot_velocity_yz_mps"].append(
                frame_second_striker_ballistic_actor_foot_velocity_yz_mps
            )
            trace["second_striker_ballistic_actor_foot_ball_distance_m"].append(
                frame_second_striker_ballistic_actor_foot_ball_distance_m
            )
            trace["second_striker_ballistic_actor_desired_launch_velocity_yz_mps"].append(
                frame_second_actor_desired_launch_yz
            )
            trace["second_striker_ballistic_actor_target_conditioned"].append(
                frame_second_striker_ballistic_actor_target_conditioned
            )
            trace["second_striker_ballistic_actor_launch_envelope_supported"].append(
                frame_second_striker_ballistic_actor_launch_envelope_supported
            )
            trace["second_striker_ballistic_actor_candidate_selected"].append(
                frame_second_striker_ballistic_actor_candidate_selected
            )
            trace["second_striker_loft_teacher_active"].append(
                frame_second_striker_loft_teacher_active
            )
            trace["second_striker_loft_teacher_force_yz_n"].append(
                frame_second_striker_loft_teacher_force_yz_n
            )
            trace["second_striker_loft_teacher_foot_velocity_yz_mps"].append(
                frame_second_striker_loft_teacher_foot_velocity_yz_mps
            )
            trace["second_striker_ballistic_contact_active"].append(
                second_striker_ballistic_contact_active
            )
            trace["second_striker_ballistic_contact_target_delta"].append(
                second_striker_ballistic_contact_target_delta
            )
            trace["second_striker_ballistic_contact_torque_active"].append(
                frame_second_striker_ballistic_contact_torque_active
            )
            trace["second_striker_ballistic_contact_torque"].append(
                frame_second_striker_ballistic_contact_torque
            )
        if goalkeeper is not None:
            trace["goalkeeper_pelvis_pose"].append(
                data.qpos[goalkeeper.qpos_base : goalkeeper.qpos_base + 7].copy()
            )
            trace["goalkeeper_root_velocity"].append(
                data.qvel[goalkeeper.qvel_base : goalkeeper.qvel_base + 6].copy()
            )
            trace["goalkeeper_torso_quaternion"].append(data.xquat[goalkeeper.torso_body].copy())
            trace["goalkeeper_left_foot_position"].append(
                data.xpos[goalkeeper.left_ankle_body].copy()
            )
            trace["goalkeeper_right_foot_position"].append(
                data.xpos[goalkeeper.right_ankle_body].copy()
            )
            trace["goalkeeper_foot_contact"].append(goalkeeper_support)
            trace["goalkeeper_support_foot_slip"].append(goalkeeper.latest_support_slip_m)
            trace["goalkeeper_left_hand_position"].append(
                data.xpos[goalkeeper.left_hand_body].copy()
            )
            trace["goalkeeper_right_hand_position"].append(
                data.xpos[goalkeeper.right_hand_body].copy()
            )
            trace["goalkeeper_joint_position"].append(data.qpos[goalkeeper.joint_qpos].copy())
            trace["goalkeeper_joint_velocity"].append(data.qvel[goalkeeper.joint_qvel].copy())
            trace["goalkeeper_joint_torque"].append(data.ctrl[goalkeeper.actuators].copy())
            trace["goalkeeper_commanded_torque"].append(commanded_torque["goalkeeper"].copy())
            trace["goalkeeper_safety_projected_torque"].append(
                projected_torque["goalkeeper"].copy()
            )
            trace["goalkeeper_executed_torque"].append(executed_torque["goalkeeper"].copy())
            trace["goalkeeper_policy_action"].append(goalkeeper.last_target.copy())
            trace["goalkeeper_target_velocity"].append(goalkeeper.target_velocity.copy())
            trace["goalkeeper_policy_frame"].append(policy_frames["goalkeeper"])
            trace["goalkeeper_command_mps"].append(goalkeeper_command_mps)
            trace["goalkeeper_predicted_target_y_m"].append(goalkeeper_target_y_m)
            trace["goalkeeper_estimated_ball_velocity_mps"].append(
                np.zeros(3, dtype=np.float64)
                if goalkeeper_actor_observation is None
                else np.asarray(
                    goalkeeper_actor_observation.estimated_ball_velocity_mps,
                    dtype=np.float64,
                )
            )
            trace["goalkeeper_estimated_intercept"].append(
                np.zeros(3, dtype=np.float64)
                if goalkeeper_actor_observation is None
                else np.asarray(
                    goalkeeper_actor_observation.estimated_intercept,
                    dtype=np.float64,
                )
            )
            trace["goalkeeper_actor_height_routed"].append(goalkeeper_actor_height_routed)
            trace["goalkeeper_mosaic_gmt_central_routed"].append(
                goalkeeper_mosaic_gmt_central_routed
            )
            trace["goalkeeper_observed_flight_active"].append(
                goalkeeper_actor_observation is not None
                and goalkeeper_actor_observation.observed_flight_start_sec is not None
            )
            trace["goalkeeper_observed_flight_start_sec"].append(
                math.nan
                if goalkeeper_actor_observation is None
                or goalkeeper_actor_observation.observed_flight_start_sec is None
                else goalkeeper_actor_observation.observed_flight_start_sec
            )
            trace["goalkeeper_reaction_active"].append(goalkeeper_reaction_active)
            lateral_error = goalkeeper_target_y_m - float(data.qpos[goalkeeper.qpos_base + 1])
            trace["goalkeeper_useful_reaction_active"].append(
                bool(
                    goalkeeper_reaction_active
                    and (
                        abs(lateral_error) <= 0.10
                        or (
                            abs(goalkeeper_command_mps) >= 0.08
                            and goalkeeper_command_mps * lateral_error > 0.0
                        )
                    )
                )
            )
            trace["goalkeeper_anticipation_active"].append(goalkeeper_anticipation_active)
            trace["goalkeeper_canonical_locomotion_mirror_active"].append(
                goalkeeper.standby_locomotion_mirror_active
            )
            trace["goalkeeper_block_action_active"].append(goalkeeper_block_action_active)
            trace["goalkeeper_overhead_reach_blend"].append(goalkeeper_overhead_reach_blend)
            trace["goalkeeper_overhead_reach_target_delta"].append(
                goalkeeper_overhead_reach_target_delta.copy()
            )
            trace["goalkeeper_whole_body_reach_blend"].append(goalkeeper_whole_body_reach_blend)
            trace["goalkeeper_whole_body_reach_target_delta"].append(
                goalkeeper_whole_body_reach_target_delta.copy()
            )
            trace["goalkeeper_mosaic_gmt_blend"].append(goalkeeper_mosaic_gmt_blend)
            trace["goalkeeper_mosaic_gmt_mirrored"].append(goalkeeper_mosaic_gmt_mirrored)
            trace["goalkeeper_mosaic_gmt_target_delta"].append(
                goalkeeper_mosaic_gmt_target_delta.copy()
            )
            trace["goalkeeper_balanced_dive_blend"].append(goalkeeper_balanced_dive_blend)
            trace["goalkeeper_balanced_dive_target_delta"].append(
                goalkeeper_balanced_dive_target_delta.copy()
            )
            trace["goalkeeper_landing_capture_active"].append(goalkeeper_landing_capture_active)
            trace["goalkeeper_landing_capture_blend"].append(goalkeeper_landing_capture_blend)
            trace["goalkeeper_proprioceptive_capture_active"].append(
                goalkeeper_proprioceptive_capture_active
            )
            trace["goalkeeper_recovery_athlete_active"].append(
                goalkeeper.last_recovery_athlete_active
            )
            trace["goalkeeper_recovery_athlete_suppressed"].append(
                goalkeeper.last_recovery_athlete_suppressed
            )
            trace["goalkeeper_recovery_athlete_raw_world_command"].append(
                np.zeros(3, dtype=np.float64)
                if goalkeeper.last_recovery_athlete_raw_world_command is None
                else goalkeeper.last_recovery_athlete_raw_world_command.copy()
            )
            trace["goalkeeper_recovery_athlete_world_command"].append(
                np.zeros(3, dtype=np.float64)
                if goalkeeper.last_recovery_athlete_world_command is None
                else goalkeeper.last_recovery_athlete_world_command.copy()
            )
            trace["goalkeeper_contact_epoch"].append(goalkeeper_contact_epoch)
            trace["goalkeeper_second_ball_contact"].append(
                goalkeeper_second_contact_time is not None
                and abs(float(data.time) - goalkeeper_second_contact_time) <= _CONTROL_DT + 1e-9
            )
            trace["goalkeeper_second_left_glove_contact"].append(
                goalkeeper_second_glove_contact_time is not None
                and goalkeeper_second_glove_contact_side in {"left", "both"}
                and abs(float(data.time) - goalkeeper_second_glove_contact_time)
                <= _CONTROL_DT + 1e-9
            )
            trace["goalkeeper_second_right_glove_contact"].append(
                goalkeeper_second_glove_contact_time is not None
                and goalkeeper_second_glove_contact_side in {"right", "both"}
                and abs(float(data.time) - goalkeeper_second_glove_contact_time)
                <= _CONTROL_DT + 1e-9
            )
            trace["goalkeeper_ball_contact"].append(
                goalkeeper_contact_time is not None
                and abs(float(data.time) - goalkeeper_contact_time) <= _CONTROL_DT + 1e-9
            )
            trace["goalkeeper_left_glove_contact"].append(
                goalkeeper_left_glove_contact_time is not None
                and abs(float(data.time) - goalkeeper_left_glove_contact_time) <= _CONTROL_DT + 1e-9
            )
            trace["goalkeeper_right_glove_contact"].append(
                goalkeeper_right_glove_contact_time is not None
                and abs(float(data.time) - goalkeeper_right_glove_contact_time)
                <= _CONTROL_DT + 1e-9
            )
            trace["goalkeeper_bimanual_reach_active"].append(goalkeeper_bimanual_reach_active)
            trace["goalkeeper_bimanual_punch_active"].append(frame_goalkeeper_bimanual_punch_active)
            trace["goalkeeper_bimanual_punch_torque"].append(
                frame_goalkeeper_bimanual_punch_torque.copy()
            )
        if (
            passer_tactical_movement_config is not None
            or passer_reactive_movement_config is not None
        ):
            trace["passer_tactical_world_target"].append(
                np.zeros(3, dtype=np.float64)
                if passer.last_tactical_world_target is None
                else passer.last_tactical_world_target.copy()
            )
            trace["passer_tactical_world_command"].append(
                np.zeros(3, dtype=np.float64)
                if passer.last_tactical_world_command is None
                else passer.last_tactical_world_command.copy()
            )
            trace["passer_tactical_movement_active"].append(passer.last_tactical_movement_active)
        if passer_reactive_movement_config is not None:
            trace["passer_reactive_route_features"].append(
                np.zeros(14, dtype=np.float64)
                if passer.last_reactive_route_features is None
                else passer.last_reactive_route_features.copy()
            )
            trace["passer_reactive_route_support_distance"].append(
                passer.last_reactive_route_support_distance
            )
            trace["passer_reactive_route_accepted"].append(passer.last_reactive_route_accepted)
            trace["passer_reactive_role_separation_m"].append(
                passer.last_reactive_role_separation_m
            )
            trace["passer_reactive_collision_shield_active"].append(
                passer.last_reactive_collision_shield_active
            )
            trace["passer_reactive_velocity_braking_correction"].append(
                np.zeros(2, dtype=np.float64)
                if passer.last_reactive_velocity_braking_correction is None
                else passer.last_reactive_velocity_braking_correction.copy()
            )
        if (
            goalkeeper_tactical_movement_config is not None
            or goalkeeper_reactive_movement_config is not None
        ):
            assert goalkeeper is not None
            trace["goalkeeper_tactical_world_target"].append(
                np.zeros(3, dtype=np.float64)
                if goalkeeper.last_tactical_world_target is None
                else goalkeeper.last_tactical_world_target.copy()
            )
            trace["goalkeeper_tactical_world_command"].append(
                np.zeros(3, dtype=np.float64)
                if goalkeeper.last_tactical_world_command is None
                else goalkeeper.last_tactical_world_command.copy()
            )
            trace["goalkeeper_tactical_movement_active"].append(
                goalkeeper.last_tactical_movement_active
            )
        if goalkeeper_reactive_movement_config is not None:
            assert goalkeeper is not None
            trace["goalkeeper_reactive_route_features"].append(
                np.zeros(14, dtype=np.float64)
                if goalkeeper.last_reactive_route_features is None
                else goalkeeper.last_reactive_route_features.copy()
            )
            trace["goalkeeper_reactive_route_support_distance"].append(
                goalkeeper.last_reactive_route_support_distance
            )
            trace["goalkeeper_reactive_route_accepted"].append(
                goalkeeper.last_reactive_route_accepted
            )
            trace["goalkeeper_reactive_role_separation_m"].append(
                goalkeeper.last_reactive_role_separation_m
            )
            trace["goalkeeper_reactive_collision_shield_active"].append(
                goalkeeper.last_reactive_collision_shield_active
            )
            trace["goalkeeper_reactive_velocity_braking_correction"].append(
                np.zeros(2, dtype=np.float64)
                if goalkeeper.last_reactive_velocity_braking_correction is None
                else goalkeeper.last_reactive_velocity_braking_correction.copy()
            )
        second_launcher_active = bool(
            second_threat_force is not None
            and second_threat_force_stop_sec is not None
            and float(data.time) <= second_threat_force_stop_sec + 1.0e-12
        )
        trace["second_threat_rearmed"].append(second_threat_rearmed)
        trace["second_threat_launcher_active"].append(second_launcher_active)
        trace["second_threat_launcher_force"].append(
            np.zeros(3, dtype=np.float64)
            if not second_launcher_active or second_threat_force is None
            else second_threat_force.copy()
        )
        _append_trace(
            trace,
            data=data,
            ball_qpos=ball_qpos,
            ball_qvel=ball_qvel,
            passer=passer,
            shooter=shooter,
            policy_frames=policy_frames,
            support=(passer_support, shooter_support),
            contact_role=contact_role,
            shooter_contact_foot=shooter_contact_foot,
            robot_contacts=frame_robot_contacts,
            commanded_torque=commanded_torque,
            projected_torque=projected_torque,
            executed_torque=executed_torque,
            contact_impulse=frame_contact_impulse,
            learned_torque=learned_torque,
            joint_guard_active=joint_guard_active,
        )
        if not finite:
            break

    target_error = (
        math.hypot(crossing_y - shooter_target[1], crossing_z - shooter_target[2])
        if crossing_y is not None and crossing_z is not None
        else None
    )
    pass_delivery_error = (
        float(np.linalg.norm(pass_delivery_position - active_pass_reception_target))
        if pass_delivery_position is not None
        else None
    )
    pass_delivery_lateral_error = (
        abs(float(pass_delivery_position[1] - active_pass_reception_target[1]))
        if pass_delivery_position is not None
        else None
    )
    trajectory = {name: np.asarray(values) for name, values in trace.items()}
    if goalkeeper is not None:
        trajectory.update(
            {name: np.asarray(values) for name, values in goalkeeper_contact_trace.items()}
        )
    result = G1SharedWorldResult(
        finite_state=finite,
        pass_contact_observed=passer.contact_latched,
        shot_contact_observed=shooter.contact_latched,
        pass_contact_time_sec=passer.contact_time,
        shot_contact_time_sec=shooter.contact_time,
        pass_peak_ball_speed_mps=pass_peak_speed,
        shot_peak_ball_speed_mps=shot_peak_speed,
        goal_crossed=goal_crossed,
        goal_plane_crossed=goal_plane_crossed,
        goal_crossing_y_m=crossing_y,
        goal_crossing_z_m=crossing_z,
        target_error_m=target_error,
        passer_min_pelvis_height_m=passer_min_height,
        shooter_min_pelvis_height_m=shooter_min_height,
        passer_roll_peak_rad=passer_roll_peak,
        passer_pitch_peak_rad=passer_pitch_peak,
        shooter_roll_peak_rad=shooter_roll_peak,
        shooter_pitch_peak_rad=shooter_pitch_peak,
        passer_tail_wobble_index=_tail_wobble(trajectory, "passer"),
        shooter_tail_wobble_index=_tail_wobble(trajectory, "shooter"),
        receiver_phase_hold_frames=shooter.phase_hold_count,
        receiver_phase_advance_frames=shooter.phase_advance_count,
        receiver_max_ball_phase_error_m=shooter.max_ball_phase_error_m,
        robot_robot_contact_count=robot_robot_contact_count,
        joint_limit_violation=joint_violation,
        torque_limit_violation=torque_violation,
        actuator_saturation=actuator_saturation,
        physics_steps=len(trace["time"]) * _SUBSTEPS,
        passer_support_foot_slip_m=passer.peak_support_slip_m,
        shooter_support_foot_slip_m=shooter.peak_support_slip_m,
        passer_post_contact_support_foot_slip_m=passer.post_contact_peak_support_slip_m,
        shooter_post_contact_support_foot_slip_m=shooter.post_contact_peak_support_slip_m,
        passer_contact_impulse_ns=passer.contact_impulse_ns,
        shooter_contact_impulse_ns=shooter.contact_impulse_ns,
        passer_post_kick_fall=passer_min_height < 0.55,
        shooter_post_kick_fall=shooter_min_height < 0.55,
        shooter_learned_torque_fraction=(
            shooter.learned_torque_frame_count
            / max(
                1,
                total_frames - int(round((shooter.contact_time or _TOTAL_TIME_SEC) / _CONTROL_DT)),
            )
        ),
        shooter_learned_torque_fallback_fraction=(
            shooter.learned_torque_fallback_count
            / max(
                1,
                total_frames - int(round((shooter.contact_time or _TOTAL_TIME_SEC) / _CONTROL_DT)),
            )
        ),
        shooter_learned_torque_mean_confidence=(
            shooter.learned_torque_confidence_sum / max(1, shooter.learned_torque_frame_count)
        ),
        shooter_learned_torque_peak_residual_nm=shooter.learned_torque_peak_residual_nm,
        shooter_learned_torque_support_rms_peak=shooter.learned_torque_support_rms_peak,
        shooter_joint_guard_fraction=shooter.joint_guard_frame_count / max(1, total_frames),
        shooter_joint_guard_route=shooter.joint_guard_route,
        pass_reception_target_m=(
            float(active_pass_reception_target[0]),
            float(active_pass_reception_target[1]),
            float(active_pass_reception_target[2]),
        ),
        pass_delivery_position_m=(
            (
                float(pass_delivery_position[0]),
                float(pass_delivery_position[1]),
                float(pass_delivery_position[2]),
            )
            if pass_delivery_position is not None
            else None
        ),
        pass_delivery_error_m=pass_delivery_error,
        pass_delivery_lateral_error_m=pass_delivery_lateral_error,
        passer_joint_guard_fraction=passer.joint_guard_frame_count / max(1, total_frames),
        passer_joint_guard_route=passer.joint_guard_route,
        passer_recovery_active_fraction=passer.recovery_active_frame_count / max(1, total_frames),
        shooter_recovery_active_fraction=shooter.recovery_active_frame_count / max(1, total_frames),
        passer_recovery_peak_blend_fraction=passer.recovery_peak_blend_fraction,
        shooter_recovery_peak_blend_fraction=shooter.recovery_peak_blend_fraction,
        shooter_transition_actor_hash=(
            None if shooter.transition_actor is None else shooter.transition_actor.actor_hash
        ),
        shooter_transition_actor_accepted=bool(
            shooter.transition_decision is not None and shooter.transition_decision.accepted
        ),
        shooter_transition_triggered=shooter.transition_triggered,
        shooter_transition_trigger_time_sec=shooter.transition_trigger_time_sec,
        shooter_transition_trigger_policy_frame=(
            None
            if shooter.transition_decision is None
            else shooter.transition_decision.trigger_policy_frame
        ),
        shooter_transition_residual_frames=(
            0
            if shooter.transition_decision is None
            else shooter.transition_decision.residual_frames
        ),
        shooter_transition_support_distance=(
            None
            if shooter.transition_decision is None
            else shooter.transition_decision.support_distance
        ),
        shooter_transition_predicted_safe_probability=(
            None
            if shooter.transition_decision is None
            else shooter.transition_decision.predicted_safe_probability
        ),
        shooter_transition_predicted_chain_probability=(
            None
            if shooter.transition_decision is None
            else shooter.transition_decision.predicted_chain_probability
        ),
        shooter_transition_ensemble_probability_spread=(
            None
            if shooter.transition_decision is None
            else shooter.transition_decision.ensemble_probability_spread
        ),
        shooter_transition_used_parent_fallback=bool(
            shooter.transition_decision is not None
            and shooter.transition_decision.used_parent_fallback
        ),
        shooter_causal_strike_option_enabled=shooter.causal_strike_option is not None,
        shooter_causal_strike_option_config_hash=(
            None
            if shooter.causal_strike_option is None
            else shooter.causal_strike_option.config.config_hash
        ),
        shooter_causal_strike_option_final_phase=(
            None
            if shooter.causal_strike_option is None
            else shooter.causal_strike_option.phase.name
        ),
        shooter_causal_strike_option_reason=(
            None
            if shooter.last_causal_strike_option_decision is None
            else shooter.last_causal_strike_option_decision.reason
        ),
        shooter_causal_strike_bridge_started=shooter.causal_strike_bridge_started,
        shooter_causal_strike_bridge_start_time_sec=(shooter.causal_strike_bridge_start_time_sec),
        shooter_causal_strike_bridge_predecessor_policy_frame=(
            shooter.causal_strike_bridge_predecessor_policy_frame
        ),
        shooter_causal_strike_selected_phase_start_frame=(
            shooter.causal_strike_selected_phase_start_frame
        ),
        shooter_causal_strike_bridge_peak_target_velocity_rms_rad_s=(
            shooter.causal_strike_bridge_peak_target_velocity_rms_rad_s
        ),
        shooter_causal_strike_abort_recovery_activated=(
            shooter.causal_strike_abort_recovery_activated
        ),
        shooter_runtime_strike_router_hash=(
            None
            if shooter.runtime_strike_router is None
            else shooter.runtime_strike_router.actor_hash
        ),
        shooter_runtime_strike_route_decided=(shooter.runtime_strike_route_decision is not None),
        shooter_runtime_strike_route_accepted=bool(
            shooter.runtime_strike_route_decision is not None
            and shooter.runtime_strike_route_decision.accepted
        ),
        shooter_runtime_strike_route=(
            None
            if shooter.runtime_strike_route_decision is None
            else shooter.runtime_strike_route_decision.route
        ),
        shooter_runtime_strike_route_time_sec=shooter.runtime_strike_route_time_sec,
        shooter_runtime_strike_route_support_distance=(
            None
            if shooter.runtime_strike_route_decision is None
            else shooter.runtime_strike_route_decision.nearest_success_distance
        ),
        shooter_runtime_strike_route_advance_frames=(
            None
            if shooter.runtime_strike_route_decision is None
            or shooter.runtime_strike_route_decision.action is None
            else (shooter.runtime_strike_route_decision.action.maximum_arrival_advance_frames)
        ),
        shooter_runtime_receive_actor_hash=(
            shooter.runtime_receive_actor.actor_hash
            if shooter.runtime_receive_actor is not None
            else (
                None
                if shooter.runtime_finish_plan_actor is None
                else shooter.runtime_finish_plan_actor.actor_hash
            )
        ),
        shooter_runtime_receive_decided=(shooter.runtime_receive_decision is not None),
        shooter_runtime_receive_accepted=bool(
            shooter.runtime_receive_decision is not None
            and shooter.runtime_receive_decision.accepted
        ),
        shooter_runtime_receive_route=(
            None
            if shooter.runtime_receive_decision is None
            else shooter.runtime_receive_decision.route
        ),
        shooter_runtime_receive_time_sec=shooter.runtime_receive_time_sec,
        shooter_runtime_receive_support_distance=(
            None
            if shooter.runtime_receive_decision is None
            else shooter.runtime_receive_decision.nearest_success_distance
        ),
        shooter_runtime_receive_alignment_tolerance_sec=(
            None
            if shooter.runtime_receive_decision is None
            or shooter.runtime_receive_decision.action is None
            else shooter.runtime_receive_decision.action.arrival_alignment_tolerance_sec
        ),
        shooter_runtime_receive_stance_offset_y_m=(
            None
            if shooter.runtime_receive_decision is None
            or shooter.runtime_receive_decision.action is None
            else shooter.runtime_receive_decision.action.stance_offset_y_m
        ),
        shooter_runtime_receive_foot_yaw_offset_rad=(
            None
            if shooter.runtime_receive_decision is None
            or shooter.runtime_receive_decision.action is None
            else shooter.runtime_receive_decision.action.foot_yaw_offset_rad
        ),
        shooter_runtime_contact_target_actor_hash=(
            shooter.runtime_contact_target_actor.actor_hash
            if shooter.runtime_contact_target_actor is not None
            else (
                None
                if shooter.runtime_finish_plan_actor is None
                else shooter.runtime_finish_plan_actor.actor_hash
            )
        ),
        shooter_runtime_contact_target_decided=(
            shooter.runtime_contact_target_decision is not None
        ),
        shooter_runtime_contact_target_accepted=bool(
            shooter.runtime_contact_target_decision is not None
            and shooter.runtime_contact_target_decision.accepted
        ),
        shooter_runtime_contact_target_route=(
            None
            if shooter.runtime_contact_target_decision is None
            else shooter.runtime_contact_target_decision.route
        ),
        shooter_runtime_contact_target_time_sec=shooter.runtime_contact_target_time_sec,
        shooter_runtime_contact_target_support_distance=(
            None
            if shooter.runtime_contact_target_decision is None
            else shooter.runtime_contact_target_decision.nearest_success_distance
        ),
        shooter_runtime_contact_target_velocity_xyz_mps=(
            None
            if shooter.runtime_contact_target_decision is None
            or shooter.runtime_contact_target_decision.action is None
            else (shooter.runtime_contact_target_decision.action.target_foot_velocity_xyz_mps)
        ),
        shooter_runtime_finish_plan_actor_hash=(
            None
            if shooter.runtime_finish_plan_actor is None
            else shooter.runtime_finish_plan_actor.actor_hash
        ),
        shooter_runtime_finish_plan_decided=(shooter.runtime_finish_plan_decision is not None),
        shooter_runtime_finish_plan_accepted=bool(
            shooter.runtime_finish_plan_decision is not None
            and shooter.runtime_finish_plan_decision.accepted
        ),
        shooter_runtime_finish_plan_route=(
            None
            if shooter.runtime_finish_plan_decision is None
            else shooter.runtime_finish_plan_decision.route
        ),
        shooter_runtime_finish_plan_time_sec=shooter.runtime_finish_plan_time_sec,
        shooter_runtime_finish_plan_support_distance=(
            None
            if shooter.runtime_finish_plan_decision is None
            else shooter.runtime_finish_plan_decision.nearest_success_distance
        ),
        shooter_aim_expert_route=(
            "early_arrival" if shooter.early_arrival_expert_frame_count > 0 else "nominal"
        ),
        shooter_early_arrival_expert_fraction=(
            shooter.early_arrival_expert_frame_count / max(1, total_frames)
        ),
        shooter_ballistic_actor_active_fraction=(
            float(np.mean(trajectory["shooter_ballistic_actor_active"]))
            if trajectory["shooter_ballistic_actor_active"].size
            else 0.0
        ),
        shooter_ballistic_actor_peak_torque_nm=(
            float(np.max(np.abs(trajectory["shooter_ballistic_actor_torque"])))
            if trajectory["shooter_ballistic_actor_torque"].size
            else 0.0
        ),
        shooter_ballistic_actor_hash=(
            None if shooter_ballistic_actor is None else shooter_ballistic_actor.actor_hash
        ),
        shooter_motion_prior_hash=(
            None if shooter_motion_prior is None else shooter_motion_prior.prior_hash
        ),
        shooter_motion_prior_position_active_fraction=(
            shooter.motion_prior_position_active_frame_count / max(1, total_frames)
        ),
        shooter_motion_prior_velocity_active_fraction=(
            shooter.motion_prior_velocity_active_frame_count / max(1, total_frames)
        ),
        shooter_motion_prior_strike_leg_scale=shooter.motion_prior_strike_leg_scale,
        shooter_motion_prior_joint_scales=shooter.motion_prior_joint_scales,
        shooter_motion_prior_velocity_joint_scales=(shooter.motion_prior_velocity_joint_scales),
        shooter_motion_prior_peak_target_delta_rad=(shooter.motion_prior_peak_target_delta_rad),
        shooter_motion_prior_peak_velocity_delta_rad_s=(
            shooter.motion_prior_peak_velocity_delta_rad_s
        ),
        shooter_agility_prior_hash=(
            None if shooter_agility_prior is None else shooter_agility_prior.prior_hash
        ),
        shooter_agility_prior_active_fraction=(
            shooter.agility_prior_active_frame_count / max(1, total_frames)
        ),
        shooter_agility_prior_peak_target_delta_rad=(shooter.agility_prior_peak_target_delta_rad),
        shooter_agility_prior_peak_velocity_delta_rad_s=(
            shooter.agility_prior_peak_velocity_delta_rad_s
        ),
        shooter_agility_prior_joint_scales=shooter.agility_prior_joint_scales,
        shooter_contact_prior_hash=(
            None if shooter_contact_prior is None else shooter_contact_prior.prior_hash
        ),
        shooter_contact_prior_active_fraction=(
            shooter.contact_prior_active_frame_count / max(1, total_frames)
        ),
        shooter_contact_prior_peak_target_delta_rad=(shooter.contact_prior_peak_target_delta_rad),
        shooter_contact_prior_peak_velocity_delta_rad_s=(
            shooter.contact_prior_peak_velocity_delta_rad_s
        ),
        shooter_contact_prior_joint_scales=shooter.contact_prior_joint_scales,
        goalkeeper_enabled=goalkeeper is not None,
        goalkeeper_reaction_active_fraction=(goalkeeper_reaction_frames / max(1, total_frames)),
        goalkeeper_anticipation_active_fraction=(
            goalkeeper_anticipation_frames / max(1, total_frames)
        ),
        goalkeeper_canonical_locomotion_mirror_active_fraction=(
            float(np.mean(trajectory["goalkeeper_canonical_locomotion_mirror_active"]))
            if goalkeeper is not None
            and trajectory["goalkeeper_canonical_locomotion_mirror_active"].size
            else 0.0
        ),
        goalkeeper_block_action_active_fraction=(
            goalkeeper_block_action_frames / max(1, total_frames)
        ),
        goalkeeper_lateral_displacement_m=(
            0.0
            if goalkeeper is None
            else abs(float(data.qpos[goalkeeper.qpos_base + 1]) - goalkeeper_initial_y)
        ),
        goalkeeper_peak_lateral_speed_mps=goalkeeper_peak_lateral_speed,
        goalkeeper_min_pelvis_height_m=(None if goalkeeper is None else goalkeeper_min_height),
        goalkeeper_ball_contact_observed=goalkeeper_contact_time is not None,
        goalkeeper_ball_contact_time_sec=goalkeeper_contact_time,
        goalkeeper_save_observed=(
            goalkeeper_contact_time is not None
            and not (
                first_goal_crossed_before_second_threat
                if second_threat_launch_time is not None
                else goal_crossed
            )
        ),
        goalkeeper_left_glove_contact_observed=(goalkeeper_left_glove_contact_time is not None),
        goalkeeper_right_glove_contact_observed=(goalkeeper_right_glove_contact_time is not None),
        goalkeeper_glove_contact_height_m=goalkeeper_glove_contact_height,
        goalkeeper_glove_contact_time_sec=goalkeeper_glove_contact_time,
        goalkeeper_glove_contact_position_m=(
            None
            if goalkeeper_glove_contact_position is None
            else (
                float(goalkeeper_glove_contact_position[0]),
                float(goalkeeper_glove_contact_position[1]),
                float(goalkeeper_glove_contact_position[2]),
            )
        ),
        goalkeeper_glove_contact_surface_distance_m=(goalkeeper_glove_contact_surface_distance),
        goalkeeper_glove_contact_side=goalkeeper_glove_contact_side,
        goalkeeper_contact_left_hand_height_m=goalkeeper_contact_left_hand_height,
        goalkeeper_contact_right_hand_height_m=goalkeeper_contact_right_hand_height,
        goalkeeper_contact_left_hand_ball_distance_m=(goalkeeper_contact_left_hand_ball_distance),
        goalkeeper_contact_right_hand_ball_distance_m=(goalkeeper_contact_right_hand_ball_distance),
        goalkeeper_both_hands_raised_at_contact=bool(
            goalkeeper_contact_left_hand_height is not None
            and goalkeeper_contact_right_hand_height is not None
            and goalkeeper_contact_left_hand_height >= 1.10
            and goalkeeper_contact_right_hand_height >= 1.10
        ),
        goalkeeper_bimanual_window_observed=goalkeeper_bimanual_window_observed,
        goalkeeper_bimanual_reach_active_fraction=(
            goalkeeper_bimanual_reach_frames / max(1, total_frames)
        ),
        goalkeeper_bimanual_punch_active_fraction=(
            goalkeeper_bimanual_punch_frames / max(1, total_frames)
        ),
        goalkeeper_bimanual_punch_peak_torque_nm=(goalkeeper_bimanual_punch_peak_torque),
        goalkeeper_reach_memory_peak_rad=(
            0.0 if goalkeeper is None else goalkeeper.goalkeeper_reach_memory_peak_rad
        ),
        goalkeeper_overhead_reach_prior_hash=(
            None
            if goalkeeper_overhead_reach_prior is None
            else goalkeeper_overhead_reach_prior.prior_hash
        ),
        goalkeeper_overhead_reach_active_fraction=(
            goalkeeper_overhead_reach_frames / max(1, total_frames)
        ),
        goalkeeper_overhead_reach_peak_blend=goalkeeper_overhead_reach_peak_blend,
        goalkeeper_whole_body_reach_atlas_hash=(
            None
            if goalkeeper_whole_body_reach_atlas is None
            else goalkeeper_whole_body_reach_atlas.model_hash
        ),
        goalkeeper_whole_body_reach_active_fraction=(
            goalkeeper_whole_body_reach_frames / max(1, total_frames)
        ),
        goalkeeper_whole_body_reach_peak_blend=goalkeeper_whole_body_reach_peak_blend,
        goalkeeper_mosaic_gmt_skill_hash=(
            None if goalkeeper_mosaic_gmt_skill is None else goalkeeper_mosaic_gmt_skill.skill_hash
        ),
        goalkeeper_mosaic_gmt_contract_hash=(
            None
            if goalkeeper_mosaic_gmt_contract is None
            else goalkeeper_mosaic_gmt_contract.contract_hash
        ),
        goalkeeper_mosaic_gmt_active_fraction=(goalkeeper_mosaic_gmt_frames / max(1, total_frames)),
        goalkeeper_mosaic_gmt_peak_blend=goalkeeper_mosaic_gmt_peak_blend,
        goalkeeper_balanced_dive_seed_hash=(
            None
            if goalkeeper_balanced_dive_seed is None
            else goalkeeper_balanced_dive_seed.seed_hash
        ),
        goalkeeper_balanced_dive_active_fraction=(
            goalkeeper_balanced_dive_frames / max(1, total_frames)
        ),
        goalkeeper_balanced_dive_peak_blend=goalkeeper_balanced_dive_peak_blend,
        goalkeeper_dive_athlete_checkpoint_hash=(goalkeeper_dive_athlete_checkpoint_hash),
        goalkeeper_dive_athlete_blend=(
            0.0 if goalkeeper_config is None else goalkeeper_config.dive_athlete_blend
        ),
        goalkeeper_recovery_athlete_checkpoint_hash=(goalkeeper_recovery_athlete_checkpoint_hash),
        goalkeeper_recovery_athlete_blend=(
            0.0 if goalkeeper_config is None else goalkeeper_config.recovery_athlete_blend
        ),
        goalkeeper_recovery_athlete_authority_envelope_enabled=(
            False
            if goalkeeper_config is None
            else goalkeeper_config.recovery_athlete_authority_envelope_enabled
        ),
        goalkeeper_recovery_athlete_active_fraction=(
            0.0
            if goalkeeper is None
            else goalkeeper.recovery_athlete_active_frame_count / max(1, total_frames)
        ),
        goalkeeper_recovery_athlete_suppressed_fraction=(
            0.0
            if goalkeeper is None or goalkeeper.recovery_athlete_active_frame_count <= 0
            else goalkeeper.recovery_athlete_suppressed_frame_count
            / goalkeeper.recovery_athlete_active_frame_count
        ),
        second_threat_enabled=second_threat_config is not None,
        second_threat_rearmed=second_threat_rearmed,
        second_threat_rearm_time_sec=second_threat_rearm_time,
        second_threat_launch_observed=second_threat_launch_time is not None,
        second_threat_launch_time_sec=second_threat_launch_time,
        second_threat_launch_position_m=(
            None
            if second_threat_launch_position is None
            else (
                float(second_threat_launch_position[0]),
                float(second_threat_launch_position[1]),
                float(second_threat_launch_position[2]),
            )
        ),
        second_threat_target_velocity_mps=(
            None
            if second_threat_target_velocity is None
            else (
                float(second_threat_target_velocity[0]),
                float(second_threat_target_velocity[1]),
                float(second_threat_target_velocity[2]),
            )
        ),
        second_threat_peak_force_n=second_threat_peak_force,
        goalkeeper_second_ball_contact_observed=goalkeeper_second_contact_time is not None,
        goalkeeper_second_ball_contact_time_sec=goalkeeper_second_contact_time,
        goalkeeper_second_glove_contact_observed=(goalkeeper_second_glove_contact_time is not None),
        goalkeeper_second_glove_contact_time_sec=goalkeeper_second_glove_contact_time,
        goalkeeper_second_glove_contact_height_m=goalkeeper_second_glove_contact_height,
        goalkeeper_second_glove_contact_position_m=(
            None
            if goalkeeper_second_glove_contact_position is None
            else (
                float(goalkeeper_second_glove_contact_position[0]),
                float(goalkeeper_second_glove_contact_position[1]),
                float(goalkeeper_second_glove_contact_position[2]),
            )
        ),
        goalkeeper_second_glove_contact_surface_distance_m=(
            goalkeeper_second_glove_contact_surface_distance
        ),
        goalkeeper_second_glove_contact_side=goalkeeper_second_glove_contact_side,
        goalkeeper_second_contact_left_hand_height_m=(goalkeeper_second_contact_left_hand_height),
        goalkeeper_second_contact_right_hand_height_m=(goalkeeper_second_contact_right_hand_height),
        goalkeeper_second_contact_left_hand_ball_distance_m=(
            goalkeeper_second_contact_left_hand_ball_distance
        ),
        goalkeeper_second_contact_right_hand_ball_distance_m=(
            goalkeeper_second_contact_right_hand_ball_distance
        ),
        goalkeeper_second_save_observed=(
            goalkeeper_second_glove_contact_time is not None
            and not (
                second_ball_goal_crossed
                if physical_second_striker_config is not None
                else goal_crossed
            )
        ),
        physical_second_striker_enabled=physical_second_striker_config is not None,
        second_striker_ball_existed_from_time_zero=(physical_second_striker_config is not None),
        second_striker_contact_observed=(
            second_striker is not None and second_striker.contact_latched
        ),
        second_striker_contact_time_sec=(
            None if second_striker is None else second_striker.contact_time
        ),
        second_striker_contact_foot=second_striker_contact_foot,
        second_striker_contact_force_peak_n=second_striker_contact_force_peak,
        second_striker_precontact_peak_ball_speed_mps=(second_striker_precontact_peak_speed),
        second_striker_postcontact_peak_ball_speed_mps=(second_striker_postcontact_peak_speed),
        second_striker_postcontact_peak_forward_ball_speed_mps=(
            second_striker_postcontact_peak_forward_speed
        ),
        second_striker_min_pelvis_height_m=(
            None if second_striker is None else second_striker_min_height
        ),
        second_striker_ballistic_actor_active_fraction=(
            float(np.mean(trajectory["second_striker_ballistic_actor_active"]))
            if second_striker is not None
            else 0.0
        ),
        second_striker_ballistic_actor_peak_torque_nm=(
            float(np.max(np.abs(trajectory["second_striker_ballistic_actor_torque"])))
            if second_striker is not None
            else 0.0
        ),
        second_striker_unexpected_precontact_collision_geoms=tuple(
            sorted(second_striker_unexpected_precontact_collision_geoms)
        ),
        second_striker_joint_limit_violation=role_joint_violation.get("second_striker", False),
        second_ball_goal_plane_crossed=second_ball_goal_plane_crossed,
        second_ball_goal_crossed=second_ball_goal_crossed,
        second_ball_goal_crossing_y_m=second_ball_crossing_y,
        second_ball_goal_crossing_z_m=second_ball_crossing_z,
        passer_joint_limit_violation=role_joint_violation["passer"],
        shooter_joint_limit_violation=role_joint_violation["shooter"],
        goalkeeper_joint_limit_violation=role_joint_violation.get("goalkeeper", False),
    )
    return result, trajectory


def _coupled_model(
    root: Path,
    *,
    passer_origin: np.ndarray | None = None,
    passer_yaw_rad: float = _PASSER_YAW,
    goal: G1TrainingGoalSpec | None = None,
    goalkeeper_config: G1GoalkeeperConfig | None = None,
    goalkeeper_origin_override: tuple[float, float, float] | None = None,
    physical_second_striker_config: G1PhysicalSecondStrikerConfig | None = None,
    second_ball_mass_kg: float | None = None,
    unified_stadium_scene: bool = False,
) -> Any:
    """Build the one corrected Soccer-owned stadium physics contract."""

    origin = np.asarray(
        _PASSER_ORIGIN if passer_origin is None else passer_origin,
        dtype=np.float64,
    )
    origin_m = (float(origin[0]), float(origin[1]), float(origin[2]))
    if not unified_stadium_scene:
        raise ValueError("shared-world Soccer rollouts require the corrected stadium scene")
    if goalkeeper_config is None:
        if physical_second_striker_config is not None:
            raise ValueError("physical second striker requires a goalkeeper")
        return build_g1_coupled_stadium_model(
            root,
            passer_origin_m=origin_m,
            passer_yaw_rad=passer_yaw_rad,
            spec=goal,
        )
    active_goal = goal or G1TrainingGoalSpec()
    goalkeeper_origin = (
        (
            active_goal.plane_x_m - goalkeeper_config.depth_from_goal_line_m,
            goalkeeper_config.initial_lateral_position_m,
            0.0,
        )
        if goalkeeper_origin_override is None
        else goalkeeper_origin_override
    )
    if physical_second_striker_config is None:
        model = build_g1_three_player_stadium_model(
            root,
            passer_origin_m=origin_m,
            passer_yaw_rad=passer_yaw_rad,
            goalkeeper_origin_m=goalkeeper_origin,
            spec=active_goal,
        )
    else:
        model = build_g1_four_player_two_ball_stadium_model(
            root,
            passer_origin_m=origin_m,
            passer_yaw_rad=passer_yaw_rad,
            goalkeeper_origin_m=goalkeeper_origin,
            second_striker_origin_m=physical_second_striker_config.origin_m,
            first_ball_origin_m=(1.0, 0.0, active_goal.ball_radius_m),
            second_ball_origin_m=physical_second_striker_config.ball_origin_m,
            second_ball_mass_kg=second_ball_mass_kg,
            spec=active_goal,
        )
    _configure_goalkeeper_glove_contact(model, goalkeeper_config)
    return model


def _configure_goalkeeper_glove_contact(
    model: Any,
    config: G1GoalkeeperConfig,
) -> None:
    """Bind the glove/ball impact response to an explicit physical contract."""

    import mujoco

    for side in ("left", "right"):
        geom_id = _id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            f"goalkeeper_{side}_goalkeeper_glove",
        )
        model.geom_solref[geom_id] = (
            config.glove_contact_time_constant_sec,
            config.glove_contact_damping_ratio,
        )
        # Give the purpose-built glove material precedence over the generic
        # ball material when MuJoCo combines the two contact solvers.
        model.geom_priority[geom_id] = 1
        if (
            config.glove_contact_time_constant_sec != 0.02
            or config.glove_contact_damping_ratio != 1.0
        ):
            wrist_pitch_joint = _id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                f"goalkeeper_{side}_wrist_pitch_joint",
            )
            # The source MJCF leaves joint-limit margin at zero and uses the
            # soft global 20 ms constraint.  A fast, low-damping punch contact
            # can consequently push a wrist through its declared mechanical
            # range before the controller reacts.  Align the physical
            # constraint activation with the configured predictive guard and
            # solve it inside one 50 Hz control interval.  This remains local
            # to the explicitly enabled punch material and never writes qpos.
            model.jnt_margin[wrist_pitch_joint] = max(0.04, config.joint_guard_margin_rad)
            model.jnt_solref[wrist_pitch_joint] = (0.003, 1.0)


def _goalkeeper_neutral_root_pose(
    source_initial_position: np.ndarray,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Reuse the qualified pelvis height without reusing a shooter's global yaw."""

    position = np.asarray(source_initial_position, dtype=np.float64)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("goalkeeper neutral root requires a finite source position")
    return (
        np.asarray((0.0, 0.0, position[2]), dtype=np.float64),
        np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64),
    )


def _kick_initial_pose(
    source_position: np.ndarray,
    source_quaternion: np.ndarray,
    source_joints: np.ndarray,
    *,
    kick_foot: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Put the frozen right-foot initial state in the selected anatomical frame."""

    position = np.asarray(source_position, dtype=np.float64)
    quaternion = np.asarray(source_quaternion, dtype=np.float64)
    joints = np.asarray(source_joints, dtype=np.float64)
    if (
        position.shape != (3,)
        or quaternion.shape != (4,)
        or joints.shape != (29,)
        or not all(np.all(np.isfinite(value)) for value in (position, quaternion, joints))
        or kick_foot not in {"left", "right"}
    ):
        raise ValueError("kick initial pose contract is invalid")
    if kick_foot == "right":
        return position.copy(), quaternion.copy(), joints.copy()
    mirrored_position = position.copy()
    mirrored_position[1] *= -1.0
    mirrored_quaternion = quaternion.copy()
    mirrored_quaternion[(1, 3),] *= -1.0
    return mirrored_position, mirrored_quaternion, _mirror_g1_joint_positions(joints)


def _make_robot(
    *,
    model: Any,
    data: Any,
    role: str,
    prefix: str,
    origin: np.ndarray,
    yaw: float,
    state_type: Any,
    output_type: Any,
    policy_type: Any,
    motion_joint_order: np.ndarray,
    parameters: ShotParameters,
    start_sec: float,
    initial_position: np.ndarray,
    initial_quaternion: np.ndarray,
    initial_joints: np.ndarray,
    target_local: np.ndarray,
    phase_hold_frames: int,
    standby_target: np.ndarray | None,
    standby_kp: np.ndarray | None,
    standby_kd: np.ndarray | None,
    use_locomotion_standby: bool,
    recovery_controller: Any | None,
    post_policy_frame: int | None,
    post_policy_blend_frames: int,
    phase_sync_enabled: bool,
    recovery_torque_actor: Any | None,
    joint_guard_enabled: bool,
    post_policy_neutral_velocity_enabled: bool,
    post_policy_forward_velocity_mps: float,
    joint_guard_config: G1JointGuardConfig,
    joint_guard_late_config: G1JointGuardConfig | None,
    post_policy_recovery_enabled: bool,
    early_arrival_parameters: ShotParameters | None,
    motion_prior: G1FootballMotionPrior | None,
    motion_prior_position_blend: float,
    motion_prior_velocity_blend: float,
    motion_prior_strike_leg_scale: float,
    motion_prior_joint_scales: tuple[float, ...],
    motion_prior_velocity_joint_scales: tuple[float, ...],
    motion_prior_contact_policy_frame: int,
    contact_prior: G1FootballMotionPrior | None,
    contact_prior_position_blend: float,
    contact_prior_velocity_blend: float,
    contact_prior_contact_policy_frame: int,
    contact_prior_joint_scales: tuple[float, ...],
    agility_prior: G1MosaicAgilityPrior | None = None,
    agility_prior_position_blend: float = 0.0,
    agility_prior_velocity_blend: float = 0.0,
    agility_prior_contact_policy_frame: int = 253,
    agility_prior_joint_scales: tuple[float, ...] = (0.0,) * 12 + (1.0,) * 17,
) -> _Robot:
    import mujoco

    if not 0 <= post_policy_blend_frames <= 200:
        raise ValueError("post-policy blend frames must be in [0, 200]")
    order = np.asarray(motion_joint_order, dtype=np.int64)
    if order.shape != (29,) or set(order.tolist()) != set(range(29)):
        raise ValueError("G1 motion joint order must be a 29-DoF permutation")

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
        policy = policy_type(state, output)
    standby_output = None
    standby_policy = None
    if use_locomotion_standby:
        standby_type = importlib.import_module("policy.loco_mode.LocoMode").LocoMode
        standby_output = output_type(29)
        with contextlib.redirect_stdout(io.StringIO()):
            standby_policy = standby_type(state, standby_output)
            standby_policy.enter()
    policy_target = np.asarray(target_local, dtype=np.float32).copy()
    if parameters.kick_foot == "left":
        policy_target[1] *= -1.0
    policy.target_pos_w = policy_target
    half_yaw = 0.5 * yaw
    frame_quaternion = np.asarray(
        (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)), dtype=np.float64
    )
    kick_mirror_sign = -1.0 if parameters.kick_foot == "left" else 1.0
    posture_yaw = 0.5 * kick_mirror_sign * parameters.pelvis_yaw_offset
    posture_quaternion = np.asarray(
        (math.cos(posture_yaw), 0.0, 0.0, math.sin(posture_yaw)), dtype=np.float64
    )
    anatomical_position, anatomical_quaternion, anatomical_joints = _kick_initial_pose(
        initial_position,
        initial_quaternion,
        initial_joints,
        kick_foot=parameters.kick_foot,
    )
    local_quaternion = _quaternion_multiply(posture_quaternion, anatomical_quaternion)
    data.qpos[qpos_base : qpos_base + 3] = origin + _rotate_z(
        anatomical_position
        + np.asarray(
            (
                parameters.stance_offset_x,
                # The relay pass is physically centred at world y=0 rather
                # than mirrored with the strike option.  Preserve the
                # qualified lateral pocket so a left-foot option approaches
                # the same immutable ball instead of shifting away from it.
                parameters.stance_offset_y,
                0.0,
            ),
            dtype=np.float64,
        ),
        yaw,
    )
    data.qpos[qpos_base + 3 : qpos_base + 7] = _quaternion_multiply(
        frame_quaternion, local_quaternion
    )
    data.qpos[joint_qpos] = anatomical_joints
    hold_target = (
        np.asarray(standby_target, dtype=np.float64).copy()
        if standby_target is not None
        else anatomical_joints.copy()
    )
    kp = (
        np.asarray(standby_kp, dtype=np.float64).copy()
        if standby_kp is not None
        else np.asarray(policy.kps, dtype=np.float64).copy()
    )
    kd = (
        np.asarray(standby_kd, dtype=np.float64).copy()
        if standby_kd is not None
        else np.asarray(policy.kds, dtype=np.float64).copy()
    )
    return _Robot(
        role=role,
        prefix=prefix,
        origin=origin.copy(),
        world_from_local_quat=frame_quaternion,
        qpos_base=qpos_base,
        qvel_base=qvel_base,
        joint_qpos=joint_qpos,
        joint_qvel=joint_qvel,
        actuators=actuators,
        joint_ids=joint_ids,
        pelvis_body=_id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "pelvis"),
        torso_body=_id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "torso_link"),
        left_hand_body=_id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "left_wrist_yaw_link"),
        right_hand_body=_id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "right_wrist_yaw_link"),
        left_ankle_body=_id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "left_ankle_roll_link"),
        right_ankle_body=_id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "right_ankle_roll_link"),
        state=state,
        output=output,
        policy=policy,
        motion_joint_order=order.copy(),
        standby_output=standby_output,
        standby_policy=standby_policy,
        parameters=parameters,
        start_sec=start_sec,
        hold_target=hold_target,
        last_target=hold_target.copy(),
        target_velocity=np.zeros(29, dtype=np.float64),
        kp=kp,
        kd=kd,
        motion_prior=motion_prior,
        motion_prior_position_blend=motion_prior_position_blend,
        motion_prior_velocity_blend=motion_prior_velocity_blend,
        motion_prior_strike_leg_scale=motion_prior_strike_leg_scale,
        motion_prior_joint_scales=motion_prior_joint_scales,
        motion_prior_velocity_joint_scales=motion_prior_velocity_joint_scales,
        motion_prior_contact_policy_frame=motion_prior_contact_policy_frame,
        last_motion_prior_target_delta=np.zeros(29, dtype=np.float64),
        last_motion_prior_velocity_delta=np.zeros(29, dtype=np.float64),
        agility_prior=agility_prior,
        agility_prior_position_blend=agility_prior_position_blend,
        agility_prior_velocity_blend=agility_prior_velocity_blend,
        agility_prior_contact_policy_frame=agility_prior_contact_policy_frame,
        agility_prior_joint_scales=agility_prior_joint_scales,
        last_agility_prior_target_delta=np.zeros(29, dtype=np.float64),
        last_agility_prior_velocity_delta=np.zeros(29, dtype=np.float64),
        contact_prior=contact_prior,
        contact_prior_position_blend=contact_prior_position_blend,
        contact_prior_velocity_blend=contact_prior_velocity_blend,
        contact_prior_contact_policy_frame=contact_prior_contact_policy_frame,
        last_contact_prior_target_delta=np.zeros(29, dtype=np.float64),
        last_contact_prior_velocity_delta=np.zeros(29, dtype=np.float64),
        contact_prior_joint_scales=contact_prior_joint_scales,
        recovery_controller=recovery_controller,
        phase_hold_frames=phase_hold_frames,
        phase_hold_remaining=phase_hold_frames,
        post_policy_frame=post_policy_frame,
        post_policy_blend_frames=post_policy_blend_frames,
        phase_sync_enabled=phase_sync_enabled,
        recovery_torque_actor=recovery_torque_actor,
        joint_guard_enabled=joint_guard_enabled,
        post_policy_neutral_velocity_enabled=post_policy_neutral_velocity_enabled,
        post_policy_forward_velocity_mps=post_policy_forward_velocity_mps,
        joint_guard_config=joint_guard_config,
        joint_guard_late_config=joint_guard_late_config,
        post_policy_recovery_enabled=post_policy_recovery_enabled,
        early_arrival_parameters=early_arrival_parameters,
    )


def _fill_local_state(robot: _Robot, data: Any, ball_body: int, ball_qvel: int) -> None:
    inverse_quaternion = robot.world_from_local_quat.copy()
    inverse_quaternion[1:] *= -1.0
    robot.state.q = data.qpos[robot.joint_qpos].copy()
    robot.state.dq = data.qvel[robot.joint_qvel].copy()
    robot.state.tau_est = data.ctrl[robot.actuators].copy()
    # MuJoCo stores a free joint's six tangent velocities in the joint frame.
    # The attached passer frame is therefore already the policy's local frame;
    # rotating these values a second time flips x/y and destabilizes inference.
    robot.state.root_lin_vel_b = data.qvel[robot.qvel_base : robot.qvel_base + 3].copy()
    robot.state.root_ang_vel_b = data.qvel[robot.qvel_base + 3 : robot.qvel_base + 6].copy()
    robot.state.torso_pos_w = _to_local(data.xpos[robot.torso_body], robot)
    robot.state.torso_quat_w = _quaternion_multiply(
        inverse_quaternion, data.xquat[robot.torso_body]
    )
    robot.state.pelvis_pos_w = _to_local(data.qpos[robot.qpos_base : robot.qpos_base + 3], robot)
    robot.state.pelvis_quat_w = _quaternion_multiply(
        inverse_quaternion, data.qpos[robot.qpos_base + 3 : robot.qpos_base + 7]
    )
    robot.state.ball_pos_w = _to_local(data.xpos[ball_body], robot)
    robot.state.ball_vel_w = _rotate_z(data.qvel[ball_qvel : ball_qvel + 3], -_yaw(robot))
    robot.state.ball_valid = True
    quaternion = np.asarray(robot.state.pelvis_quat_w, dtype=np.float64)
    qw, qx, qy, qz = map(float, quaternion)
    robot.state.gravity_ori = np.asarray(
        (
            2.0 * (-qz * qx + qw * qy),
            -2.0 * (qz * qy + qw * qx),
            1.0 - 2.0 * (qw * qw + qz * qz),
        ),
        dtype=np.float64,
    )
    robot.state.ang_vel = robot.state.root_ang_vel_b.copy()


def _begin_causal_strike_bridge(
    robot: _Robot,
    *,
    timestamp_sec: float,
    predecessor_policy_frame: int,
    strike_phase_start_frame: int,
) -> None:
    controller = robot.causal_strike_option
    order = robot.motion_joint_order
    if (
        controller is None
        or order is None
        or robot.causal_strike_bridge is not None
        or robot.entered
    ):
        raise RuntimeError("causal strike bridge lost its exclusive entry contract")
    config = controller.config
    if not (
        config.minimum_strike_phase_start_frame
        <= strike_phase_start_frame
        <= config.maximum_strike_phase_start_frame
    ):
        raise RuntimeError("causal strike option selected an invalid motion phase")
    with _canonical_kick_policy_state(robot), contextlib.redirect_stdout(io.StringIO()):
        robot.policy.enter()
        robot.policy._ref_anchor_world_origin = robot.policy._init_to_world @ (
            robot.policy.motion_body_pos[strike_phase_start_frame, 9].astype(np.float64)
        )
    phase_target = np.asarray(
        robot.policy.motion_joint_pos[strike_phase_start_frame][order],
        dtype=np.float64,
    )
    phase_velocity = np.asarray(
        robot.policy.motion_joint_vel[strike_phase_start_frame][order],
        dtype=np.float64,
    )
    policy_kp = np.asarray(robot.policy.kps, dtype=np.float64)
    policy_kd = np.asarray(robot.policy.kds, dtype=np.float64)
    if robot.parameters.kick_foot == "left":
        phase_target = _mirror_g1_joint_positions(phase_target)
        phase_velocity = _mirror_g1_joint_positions(phase_velocity)
        policy_kp = _mirror_g1_joint_gains(policy_kp)
        policy_kd = _mirror_g1_joint_gains(policy_kd)
    entry_position = np.asarray(robot.state.q, dtype=np.float64)
    entry_velocity = np.asarray(robot.state.dq, dtype=np.float64)
    delta = phase_target - entry_position
    robot.causal_strike_bridge = G1VelocityMatchedTransitionBridge(
        entry_position=entry_position,
        entry_velocity=entry_velocity,
        exit_position=phase_target,
        exit_velocity=phase_velocity,
        config=G1TransitionBridgeConfig(
            duration_sec=config.bridge_duration_sec,
            entry_velocity_scale=config.bridge_entry_velocity_scale,
            exit_velocity_scale=config.bridge_exit_velocity_scale,
            maximum_boundary_velocity_rad_s=config.bridge_boundary_velocity_limit_rad_s,
        ),
    )
    robot.causal_strike_bridge_frames = int(round(config.bridge_duration_sec / _CONTROL_DT))
    robot.causal_strike_bridge_phase_start = strike_phase_start_frame
    robot.causal_strike_bridge_kp = np.minimum(
        policy_kp,
        np.asarray(G1_HARD_TORQUE_LIMITS, dtype=np.float64)
        * 0.58
        / np.maximum(np.abs(delta), 0.05),
    )
    robot.causal_strike_bridge_kd = policy_kd * 0.72
    robot.causal_strike_bridge_started = True
    robot.causal_strike_bridge_start_time_sec = timestamp_sec
    robot.causal_strike_bridge_predecessor_policy_frame = predecessor_policy_frame
    robot.causal_strike_selected_phase_start_frame = strike_phase_start_frame


def _enter_policy(robot: _Robot) -> None:
    with _canonical_kick_policy_state(robot), contextlib.redirect_stdout(io.StringIO()):
        robot.policy.enter()
    robot.entered = True


@contextlib.contextmanager
def _canonical_kick_policy_state(robot: _Robot) -> Any:
    """Expose a mirrored left-foot rollout in the right-foot policy frame."""

    if robot.parameters.kick_foot != "left":
        yield
        return
    state = robot.state
    attributes = (
        "q",
        "dq",
        "ddq",
        "tau_est",
        "gravity_ori",
        "ang_vel",
        "root_lin_vel_b",
        "root_ang_vel_b",
        "torso_pos_w",
        "torso_quat_w",
        "pelvis_pos_w",
        "pelvis_quat_w",
        "ball_pos_w",
        "ball_vel_w",
        "ball_pos_b",
        "target_pos_b",
    )
    original = {name: np.asarray(getattr(state, name)).copy() for name in attributes}
    target_y_bias = float(state.target_y_bias)
    ball_y_bias = float(state.ball_y_bias)
    try:
        for name in ("q", "dq", "ddq", "tau_est"):
            setattr(state, name, _mirror_g1_joint_positions(original[name]))
        for name in (
            "gravity_ori",
            "root_lin_vel_b",
            "torso_pos_w",
            "pelvis_pos_w",
            "ball_pos_w",
            "ball_vel_w",
            "ball_pos_b",
            "target_pos_b",
        ):
            value = original[name].copy()
            value[1] *= -1.0
            setattr(state, name, value)
        for name in ("ang_vel", "root_ang_vel_b"):
            value = original[name].copy()
            value[(0, 2),] *= -1.0
            setattr(state, name, value)
        for name in ("torso_quat_w", "pelvis_quat_w"):
            value = original[name].copy()
            value[(1, 3),] *= -1.0
            setattr(state, name, value)
        state.target_y_bias = -target_y_bias
        state.ball_y_bias = -ball_y_bias
        yield
    finally:
        for name, value in original.items():
            setattr(state, name, value)
        state.target_y_bias = target_y_bias
        state.ball_y_bias = ball_y_bias


def _smooth_policy_handoff(
    *,
    origin_target: np.ndarray,
    origin_kp: np.ndarray,
    origin_kd: np.ndarray,
    destination_target: np.ndarray,
    destination_kp: np.ndarray,
    destination_kd: np.ndarray,
    transition_step: int,
    blend_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Blend a policy handoff without a target or impedance step."""

    if blend_frames <= 0 or not 0 <= transition_step < blend_frames:
        raise ValueError("policy handoff step must be inside a positive blend window")
    values = (
        origin_target,
        origin_kp,
        origin_kd,
        destination_target,
        destination_kp,
        destination_kd,
    )
    if any(value.shape != (29,) or not np.all(np.isfinite(value)) for value in values):
        raise ValueError("policy handoff vectors must be finite 29-joint arrays")
    linear = (transition_step + 1) / blend_frames
    fraction = linear * linear * (3.0 - 2.0 * linear)
    return (
        origin_target + fraction * (destination_target - origin_target),
        origin_kp + fraction * (destination_kp - origin_kp),
        origin_kd + fraction * (destination_kd - origin_kd),
        fraction,
    )


def _project_joint_safe_torque(
    *,
    joint_position: np.ndarray,
    joint_velocity: np.ndarray,
    commanded_torque: np.ndarray,
    joint_ranges: np.ndarray,
    limited: np.ndarray,
    margin_rad: float = 0.04,
    prediction_horizon_sec: float = 0.08,
    boundary_kp: float = 80.0,
    boundary_kd: float = 6.0,
) -> tuple[np.ndarray, bool]:
    """Project outward torque when a velocity-aware joint envelope is threatened."""

    if (
        joint_position.shape != (29,)
        or joint_velocity.shape != (29,)
        or commanded_torque.shape != (29,)
        or joint_ranges.shape != (29, 2)
        or limited.shape != (29,)
    ):
        raise ValueError("joint guard requires the 29-DoF G1 contract")
    if not all(
        np.all(np.isfinite(value))
        for value in (joint_position, joint_velocity, commanded_torque, joint_ranges)
    ):
        raise ValueError("joint guard inputs must be finite")
    if not 0.0 < margin_rad <= 0.10 or not 0.0 < prediction_horizon_sec <= 0.20:
        raise ValueError("joint guard envelope parameters are invalid")
    projected = commanded_torque.copy()
    predicted = joint_position + prediction_horizon_sec * joint_velocity
    lower = joint_ranges[:, 0] + margin_rad
    upper = joint_ranges[:, 1] - margin_rad
    lower_threat = limited & (predicted < lower)
    upper_threat = limited & (predicted > upper)
    lower_brake = boundary_kp * (lower - joint_position) - boundary_kd * joint_velocity
    upper_brake = boundary_kp * (upper - joint_position) - boundary_kd * joint_velocity
    projected[lower_threat] = np.maximum(projected[lower_threat], lower_brake[lower_threat])
    projected[upper_threat] = np.minimum(projected[upper_threat], upper_brake[upper_threat])
    active = not np.array_equal(projected, commanded_torque)
    return projected, active


def _select_joint_guard_config(
    *,
    standard: G1JointGuardConfig,
    late_arrival: G1JointGuardConfig | None,
    phase_advance_count: int,
) -> tuple[G1JointGuardConfig, str]:
    """Latch one guard expert from the causal receiver phase signal."""

    if phase_advance_count < 0:
        raise ValueError("phase advance count must be non-negative")
    if late_arrival is not None and phase_advance_count > 0:
        return late_arrival, "late_arrival"
    return standard, "standard"


def _normalized_zero_locomotion_command(policy: Any) -> np.ndarray:
    """Invert RoboNaldo joystick scaling so the physical command is zero."""

    ranges = np.asarray(
        (policy.range_velx, policy.range_vely, policy.range_velz),
        dtype=np.float64,
    )
    if ranges.shape != (3, 2) or not np.all(np.isfinite(ranges)):
        raise ValueError("locomotion command ranges must be finite [min, max] pairs")
    widths = ranges[:, 1] - ranges[:, 0]
    if np.any(widths <= 0.0) or np.any(ranges[:, 0] > 0.0) or np.any(ranges[:, 1] < 0.0):
        raise ValueError("locomotion command ranges must be ordered and contain zero")
    normalized = -1.0 - 2.0 * ranges[:, 0] / widths
    if np.any(np.abs(normalized) > 1.0 + 1e-12):
        raise ValueError("zero locomotion command is outside the normalized input range")
    result: NDArray[np.float64] = np.asarray(normalized, dtype=np.float64)
    return result


def _normalized_locomotion_command(policy: Any, physical_command: np.ndarray) -> np.ndarray:
    """Map physical ``vx, vy, yaw`` commands into RoboNaldo joystick space."""

    command = np.asarray(physical_command, dtype=np.float64)
    ranges = np.asarray(
        (policy.range_velx, policy.range_vely, policy.range_velz),
        dtype=np.float64,
    )
    if command.shape != (3,) or not np.all(np.isfinite(command)):
        raise ValueError("locomotion command must be a finite physical 3-vector")
    widths = ranges[:, 1] - ranges[:, 0]
    if ranges.shape != (3, 2) or np.any(widths <= 0.0):
        raise ValueError("locomotion command ranges are invalid")
    if np.any(command < ranges[:, 0]) or np.any(command > ranges[:, 1]):
        raise ValueError("physical locomotion command is outside the policy range")
    result: NDArray[np.float64] = np.asarray(
        -1.0 + 2.0 * (command - ranges[:, 0]) / widths, dtype=np.float64
    )
    return result


def _command_tactical_movement(
    robot: _Robot,
    *,
    data: Any,
    config: G1TacticalMovementConfig,
    timestamp_sec: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], bool]:
    """Track a causal waypoint stream through the frozen locomotion policy."""

    if robot.standby_policy is None:
        raise RuntimeError("tactical movement requires a locomotion standby policy")
    times = np.asarray([waypoint.time_sec for waypoint in config.waypoints], dtype=np.float64)
    positions = np.asarray([waypoint.position_m for waypoint in config.waypoints], dtype=np.float64)
    if timestamp_sec <= times[0]:
        target = positions[0]
        feedforward = np.zeros(3, dtype=np.float64)
    elif timestamp_sec >= times[-1]:
        target = positions[-1]
        feedforward = np.zeros(3, dtype=np.float64)
    else:
        upper = int(np.searchsorted(times, timestamp_sec, side="right"))
        lower = upper - 1
        duration = float(times[upper] - times[lower])
        phase = float((timestamp_sec - times[lower]) / duration)
        target = positions[lower] + phase * (positions[upper] - positions[lower])
        feedforward = (positions[upper] - positions[lower]) / duration

    position = np.asarray(data.qpos[robot.qpos_base : robot.qpos_base + 3], dtype=np.float64)
    error = target - position
    error[2] = 0.0
    command = feedforward + config.position_gain * error
    command[2] = 0.0
    speed = float(np.linalg.norm(command[:2]))
    if speed > config.maximum_speed_mps:
        command[:2] *= config.maximum_speed_mps / speed
    if timestamp_sec >= times[-1] and float(np.linalg.norm(error[:2])) <= config.arrival_radius_m:
        command[:2] = 0.0

    previous = (
        np.zeros(3, dtype=np.float64)
        if robot.last_tactical_world_command is None
        else np.asarray(robot.last_tactical_world_command, dtype=np.float64)
    )
    maximum_delta = config.maximum_acceleration_mps2 * _CONTROL_DT
    delta = command - previous
    delta_norm = float(np.linalg.norm(delta[:2]))
    if delta_norm > maximum_delta:
        command[:2] = previous[:2] + delta[:2] * (maximum_delta / delta_norm)

    yaw = _yaw(robot)
    local_command = _rotate_z(command, -yaw)
    ranges = np.asarray(
        (
            robot.standby_policy.range_velx,
            robot.standby_policy.range_vely,
            robot.standby_policy.range_velz,
        ),
        dtype=np.float64,
    )
    local_command = np.clip(local_command, ranges[:, 0], ranges[:, 1])
    actual_world_command = _rotate_z(local_command, yaw)
    if robot.role != "goalkeeper":
        # RoboNaldo's released locomotion actor is materially stronger in its
        # canonical local +y half-space.  Reuse the already qualified mirrored
        # proprioception/action path for a team-mate's local -y route instead
        # of asking the weak half-space to learn around a runtime mismatch.
        robot.standby_locomotion_mirror_active = bool(local_command[1] < -1.0e-6)
    robot.state.vel_cmd = _normalized_locomotion_command(robot.standby_policy, local_command)
    active = bool(float(np.linalg.norm(actual_world_command[:2])) > 1.0e-6)
    robot.last_tactical_world_target = np.asarray(target, dtype=np.float64).copy()
    robot.last_tactical_world_command = actual_world_command.copy()
    robot.last_tactical_movement_active = active
    return robot.last_tactical_world_target, actual_world_command, active


def _command_reactive_movement(
    robot: _Robot,
    *,
    carrier: _Robot,
    other_role: _Robot,
    data: Any,
    ball_qpos: int,
    config: G1ReactiveMovementConfig,
    actor: RouteActor,
) -> tuple[NDArray[np.float64], NDArray[np.float64], bool]:
    """Run one bounded observation update through frozen locomotion authority."""

    if robot.standby_policy is None:
        raise RuntimeError("reactive movement requires a locomotion standby policy")
    position = np.asarray(data.qpos[robot.qpos_base : robot.qpos_base + 2], dtype=np.float64)
    velocity = np.asarray(data.qvel[robot.qvel_base : robot.qvel_base + 2], dtype=np.float64)
    carrier_position = np.asarray(
        data.qpos[carrier.qpos_base : carrier.qpos_base + 2], dtype=np.float64
    )
    other_position = np.asarray(
        data.qpos[other_role.qpos_base : other_role.qpos_base + 2], dtype=np.float64
    )
    ball_position = np.asarray(data.qpos[ball_qpos : ball_qpos + 2], dtype=np.float64)
    target = np.asarray(config.target_position_m, dtype=np.float64)
    if config.role == "teammate" and config.action == "pass" and robot.contact_latched:
        target = target.copy()
        target[0] = min(12.0, target[0] + config.post_reception_follow_through_m)
    features = reactive_route_features(
        target_xy_m=target[:2],
        self_position_xy_m=position,
        self_velocity_xy_mps=velocity,
        ball_position_xy_m=ball_position,
        carrier_position_xy_m=carrier_position,
        other_role_position_xy_m=other_position,
        action=config.action,
        role=config.role,
    )
    decision: Any
    if isinstance(actor, G1TemporalRouteActor):
        decision = actor.decide(features, robot.last_temporal_route_memory)
        robot.last_temporal_route_memory = decision.next_memory
    else:
        decision = actor.decide(features)
    if robot.reactive_diagonal_braking_confidence is None:
        if config.role == "teammate" and config.action == "pass":
            dx_confidence = float(
                np.clip(
                    (features[0] - config.diagonal_braking_target_dx_start_m)
                    / (
                        config.diagonal_braking_target_dx_full_m
                        - config.diagonal_braking_target_dx_start_m
                    ),
                    0.0,
                    1.0,
                )
            )
            dy_confidence = float(
                np.clip(
                    (features[1] - config.diagonal_braking_target_dy_start_m)
                    / (
                        config.diagonal_braking_target_dy_full_m
                        - config.diagonal_braking_target_dy_start_m
                    ),
                    0.0,
                    1.0,
                )
            )
            robot.reactive_diagonal_braking_confidence = dx_confidence * dy_confidence
        else:
            robot.reactive_diagonal_braking_confidence = 0.0
    command = np.zeros(3, dtype=np.float64)
    robot.last_reactive_role_separation_m = float(np.linalg.norm(position - other_position))
    robot.last_reactive_collision_shield_active = False
    robot.last_reactive_velocity_braking_correction = np.zeros(2, dtype=np.float64)
    if decision.accepted:
        command[:2] = decision.world_command_xy_mps
        if robot.reactive_diagonal_braking_confidence > 0.0:
            braking_correction = -config.velocity_braking_gain * velocity
            braking_norm = float(np.linalg.norm(braking_correction))
            if braking_norm > config.maximum_velocity_braking_correction_mps:
                braking_correction *= config.maximum_velocity_braking_correction_mps / braking_norm
            braking_correction *= robot.reactive_diagonal_braking_confidence
            command[:2] += braking_correction
            robot.last_reactive_velocity_braking_correction = braking_correction.copy()
        target_error_distance = float(np.linalg.norm(target[:2] - position))
        far_fraction = float(
            np.clip(
                (target_error_distance - config.far_target_activation_distance_m) / 0.45,
                0.0,
                1.0,
            )
        )
        command[:2] *= 1.0 + far_fraction * (config.far_target_speed_gain - 1.0)
        speed = float(np.linalg.norm(command[:2]))
        if speed > config.maximum_speed_mps:
            command[:2] *= config.maximum_speed_mps / speed
        if float(np.linalg.norm(target[:2] - position)) <= config.arrival_radius_m:
            command[:2] = 0.0

        relative_role_position = position - other_position
        role_separation = float(np.linalg.norm(relative_role_position))
        shield_active = role_separation < config.minimum_role_separation_m
        if shield_active:
            if role_separation <= 1.0e-9:
                relative_role_position = target[:2] - other_position
                role_separation = float(np.linalg.norm(relative_role_position))
            if role_separation > 1.0e-9:
                correction_speed = min(
                    config.maximum_collision_correction_mps,
                    config.collision_avoidance_gain
                    * (config.minimum_role_separation_m - role_separation),
                )
                command[:2] += correction_speed * relative_role_position / role_separation
                speed = float(np.linalg.norm(command[:2]))
                if speed > config.maximum_speed_mps:
                    command[:2] *= config.maximum_speed_mps / speed
        robot.last_reactive_role_separation_m = role_separation
        robot.last_reactive_collision_shield_active = shield_active

        previous = (
            np.zeros(3, dtype=np.float64)
            if robot.last_tactical_world_command is None
            else np.asarray(robot.last_tactical_world_command, dtype=np.float64)
        )
        maximum_delta = config.maximum_acceleration_mps2 * _CONTROL_DT
        delta = command - previous
        delta_norm = float(np.linalg.norm(delta[:2]))
        if delta_norm > maximum_delta:
            command[:2] = previous[:2] + delta[:2] * (maximum_delta / delta_norm)

    yaw = _yaw(robot)
    local_command = _rotate_z(command, -yaw)
    ranges = np.asarray(
        (
            robot.standby_policy.range_velx,
            robot.standby_policy.range_vely,
            robot.standby_policy.range_velz,
        ),
        dtype=np.float64,
    )
    local_command = np.clip(local_command, ranges[:, 0], ranges[:, 1])
    actual_world_command = _rotate_z(local_command, yaw)
    if robot.role != "goalkeeper":
        robot.standby_locomotion_mirror_active = bool(local_command[1] < -1.0e-6)
    robot.state.vel_cmd = _normalized_locomotion_command(robot.standby_policy, local_command)
    active = bool(float(np.linalg.norm(actual_world_command[:2])) > 1.0e-6)
    robot.last_tactical_world_target = target.copy()
    robot.last_tactical_world_command = actual_world_command.copy()
    robot.last_tactical_movement_active = active
    robot.last_reactive_route_features = features.copy()
    robot.last_reactive_route_support_distance = decision.support_distance
    robot.last_reactive_route_accepted = decision.accepted
    return target, actual_world_command, active


def _goalkeeper_actor_route_active(
    robot: _Robot,
    *,
    observation: GoalkeeperActorObservation,
    config: G1GoalkeeperConfig,
    timestamp_sec: float,
) -> bool:
    """Route a segmented, causal incoming threat to the aerial actor."""

    flight_start = observation.observed_flight_start_sec
    return bool(
        not robot.contact_latched
        and flight_start is not None
        and timestamp_sec - flight_start >= config.actor_threat_warmup_sec
        and observation.intercept_confidence >= config.actor_minimum_intercept_confidence
        and observation.estimated_intercept[2] >= config.actor_minimum_target_height_m
        and robot.state.ball_pos_w[2] >= config.actor_minimum_current_ball_height_m
        and np.linalg.norm(observation.estimated_ball_velocity_mps)
        >= config.actor_minimum_incoming_ball_speed_mps
    )


def _command_goalkeeper(
    robot: _Robot,
    *,
    shooter: _Robot,
    data: Any,
    ball_qpos: int,
    ball_qvel: int,
    goal: G1TrainingGoalSpec,
    config: G1GoalkeeperConfig,
    shot_contact_time: float | None,
    observer: GoalkeeperActorObserver | None,
    learned_actor: NumpyGoalkeeperActor | None,
    previous_actor_residual: np.ndarray,
    recovery_athlete_torch: Any | None,
    recovery_athlete_model: Any | None,
    recovery_athlete_checkpoint: dict[str, Any] | None,
    _force_legacy: bool = False,
) -> tuple[float, float, bool, bool, GoalkeeperActorObservation | None]:
    """Issue a causal lateral shuffle from intent, proprioception and ball flight."""

    if robot.standby_policy is None:
        raise RuntimeError("goalkeeper requires the qualified locomotion policy")
    if config.actor_observation_mode == "visible_ball_history_v3" and not _force_legacy:
        if observer is None:
            raise RuntimeError("visible-ball goalkeeper requires its causal observer")
        visible = _command_goalkeeper_visible_ball(
            robot,
            data=data,
            ball_qpos=ball_qpos,
            goal=goal,
            config=config,
            observer=observer,
            learned_actor=learned_actor,
            previous_actor_residual=previous_actor_residual,
            recovery_athlete_torch=recovery_athlete_torch,
            recovery_athlete_model=recovery_athlete_model,
            recovery_athlete_checkpoint=recovery_athlete_checkpoint,
        )
        observation = visible[4]
        if (
            config.block_action_timing_mode == "shooter_phase"
            and not _goalkeeper_actor_route_active(
                robot,
                observation=observation,
                config=config,
                timestamp_sec=float(data.time),
            )
        ):
            legacy = _command_goalkeeper(
                robot,
                shooter=shooter,
                data=data,
                ball_qpos=ball_qpos,
                ball_qvel=ball_qvel,
                goal=goal,
                config=config,
                shot_contact_time=shot_contact_time,
                observer=observer,
                learned_actor=None,
                previous_actor_residual=previous_actor_residual,
                recovery_athlete_torch=recovery_athlete_torch,
                recovery_athlete_model=recovery_athlete_model,
                recovery_athlete_checkpoint=recovery_athlete_checkpoint,
                _force_legacy=True,
            )
            return legacy[0], legacy[1], legacy[2], legacy[3], observation
        return visible
    timestamp = float(data.time)
    ball = np.asarray(data.qpos[ball_qpos : ball_qpos + 3], dtype=np.float64)
    velocity = np.asarray(data.qvel[ball_qvel : ball_qvel + 3], dtype=np.float64)
    flight_reaction_active = bool(
        shot_contact_time is not None
        and timestamp >= shot_contact_time + config.reaction_delay_sec
        and velocity[0] > 0.10
    )
    shooter_policy_frame = max(
        0,
        int(shooter.policy.time_step) - int(shooter.policy.WARMUP_STEPS),
    )
    foot_ball_distance = float(
        np.linalg.norm(np.asarray(data.xpos[shooter.right_ankle_body], dtype=np.float64) - ball)
    )
    anticipation_active = bool(
        config.anticipation_enabled
        and shot_contact_time is None
        and shooter_policy_frame >= config.anticipation_start_policy_frame
        and config.anticipation_minimum_foot_ball_distance_m
        <= foot_ball_distance
        <= config.anticipation_maximum_foot_ball_distance_m
    )
    reaction_active = flight_reaction_active or anticipation_active
    if flight_reaction_active:
        horizon = max(0.0, (goal.plane_x_m - float(ball[0])) / float(velocity[0]))
        target_y = float(ball[1] + velocity[1] * horizon)
        mouth_limit = goal.width_m / 2.0 - 0.22
        target_y = float(np.clip(target_y, -mouth_limit, mouth_limit))
    elif anticipation_active:
        target_y = float(
            config.anticipation_target_blend * goal.target_y_m
            + (1.0 - config.anticipation_target_blend) * float(ball[1])
        )
    else:
        phase = 2.0 * math.pi * timestamp / config.ready_shuffle_period_sec
        target_y = config.initial_lateral_position_m + 0.12 * math.sin(phase)
    current_y = float(data.qpos[robot.qpos_base + 1])
    world_velocity_y = float(
        np.clip(
            config.lateral_position_gain * (target_y - current_y),
            -config.maximum_lateral_speed_mps,
            config.maximum_lateral_speed_mps,
        )
    )
    if not reaction_active:
        world_velocity_y = float(
            np.clip(
                world_velocity_y,
                -config.ready_shuffle_speed_mps,
                config.ready_shuffle_speed_mps,
            )
        )
    elif anticipation_active:
        world_velocity_y *= config.anticipation_velocity_scale
    # The keeper faces -x, so its local +y points toward world -y.
    local_depth_velocity = _goalkeeper_local_depth_velocity(
        robot,
        data=data,
        goal=goal,
        config=config,
    )
    robot.state.vel_cmd = _normalized_locomotion_command(
        robot.standby_policy,
        np.asarray((local_depth_velocity, -world_velocity_y, 0.0), dtype=np.float64),
    )
    return world_velocity_y, target_y, reaction_active, anticipation_active, None


def _command_goalkeeper_visible_ball(
    robot: _Robot,
    *,
    data: Any,
    ball_qpos: int,
    goal: G1TrainingGoalSpec,
    config: G1GoalkeeperConfig,
    observer: GoalkeeperActorObserver,
    learned_actor: NumpyGoalkeeperActor | None,
    previous_actor_residual: np.ndarray,
    recovery_athlete_torch: Any | None,
    recovery_athlete_model: Any | None,
    recovery_athlete_checkpoint: dict[str, Any] | None,
) -> tuple[float, float, bool, bool, GoalkeeperActorObservation]:
    """Issue a causal command without reading shooter state or internal phase."""

    if robot.standby_policy is None:
        raise RuntimeError("goalkeeper requires the qualified locomotion policy")
    timestamp = float(data.time)
    observation = observer.observe(
        timestamp_sec=timestamp,
        # ball_pos_w is expressed in the robot's immutable spawn/goal-local
        # frame.  Using this stable anchor prevents standby body sway from
        # appearing as an incoming shot in the position history.
        ball_relative_position_m=np.asarray(robot.state.ball_pos_w, dtype=np.float64),
        gravity_orientation=np.asarray(robot.state.gravity_ori, dtype=np.float64),
        root_linear_velocity_mps=np.asarray(robot.state.root_lin_vel_b, dtype=np.float64),
        angular_velocity_rad_s=np.asarray(robot.state.ang_vel, dtype=np.float64),
        joint_position_rad=np.asarray(robot.state.q, dtype=np.float64),
        joint_velocity_rad_s=np.asarray(robot.state.dq, dtype=np.float64),
        # This is the previous *learned residual*, not the absolute target of
        # the frozen locomotion foundation.  The training contract uses the
        # same quantity; mixing the two created a severe sim-to-deploy input
        # distribution shift.
        previous_action_rad=np.asarray(previous_actor_residual, dtype=np.float64),
    )
    robot.last_recovery_athlete_active = False
    robot.last_recovery_athlete_suppressed = False
    robot.last_recovery_athlete_raw_world_command = None
    robot.last_recovery_athlete_world_command = None
    if robot.contact_latched and config.post_contact_stabilization_enabled:
        # A deflected ball is no longer an incoming threat.  Continuing to
        # extrapolate its outgoing flight made the keeper reverse direction
        # several times after a successful save.  Let the qualified
        # locomotion foundation absorb the impact with its exact zero command;
        # the bounded dive/reach envelopes release their own joint authority
        # independently and continuously.
        current_y = float(data.qpos[robot.qpos_base + 1])
        if not config.post_contact_ready_recovery_enabled:
            robot.state.vel_cmd = _normalized_zero_locomotion_command(robot.standby_policy)
            return 0.0, current_y, False, False, observation
        contact_time = robot.contact_time
        if contact_time is None:
            raise RuntimeError("goalkeeper post-contact recovery lacks a contact timestamp")
        elapsed_since_contact = max(0.0, timestamp - contact_time)
        if elapsed_since_contact < config.post_contact_ready_recovery_delay_sec:
            robot.state.vel_cmd = _normalized_zero_locomotion_command(robot.standby_policy)
            return 0.0, current_y, False, False, observation
        probe_active = bool(
            config.successor_lateral_probe_enabled
            and config.successor_lateral_probe_delay_sec
            <= elapsed_since_contact
            < config.successor_lateral_probe_delay_sec + config.successor_lateral_probe_duration_sec
        )
        if probe_active:
            world_velocity_y = config.successor_lateral_probe_command_mps
            target_y = current_y + world_velocity_y * config.successor_lateral_probe_duration_sec
        else:
            world_velocity_y = (
                0.0
                if abs(current_y) <= config.post_contact_ready_lateral_deadband_m
                else float(
                    np.clip(
                        -config.post_contact_ready_lateral_position_gain * current_y,
                        -config.post_contact_ready_maximum_lateral_speed_mps,
                        config.post_contact_ready_maximum_lateral_speed_mps,
                    )
                )
            )
            target_y = 0.0
        desired_yaw = math.pi
        current_yaw = _current_root_yaw(robot, data=data)
        yaw_error = math.atan2(
            math.sin(desired_yaw - current_yaw),
            math.cos(desired_yaw - current_yaw),
        )
        yaw_rate = float(
            np.clip(
                config.post_contact_ready_yaw_gain * yaw_error,
                -config.post_contact_ready_maximum_yaw_rate_rad_s,
                config.post_contact_ready_maximum_yaw_rate_rad_s,
            )
        )
        desired_world_x = goal.plane_x_m - config.depth_from_goal_line_m
        current_world_x = float(data.qpos[robot.qpos_base])
        world_velocity_x = float(
            np.clip(
                config.depth_position_gain * (desired_world_x - current_world_x),
                -config.maximum_depth_correction_mps,
                config.maximum_depth_correction_mps,
            )
        )
        if (
            not probe_active
            and recovery_athlete_torch is not None
            and recovery_athlete_model is not None
            and recovery_athlete_checkpoint is not None
        ):
            learned_world_command = _recovery_athlete_world_command(
                robot,
                data=data,
                desired_depth_m=desired_world_x,
                yaw_error_rad=yaw_error,
                elapsed_since_contact_sec=elapsed_since_contact,
                torch=recovery_athlete_torch,
                model=recovery_athlete_model,
                checkpoint=recovery_athlete_checkpoint,
            )
            raw_learned_world_command = learned_world_command.copy()
            if config.recovery_athlete_authority_envelope_enabled:
                learned_world_command = _recovery_athlete_authority_envelope(
                    learned_world_command,
                    depth_error_m=desired_world_x - current_world_x,
                    lateral_position_m=current_y,
                    yaw_error_rad=yaw_error,
                    config=config,
                )
                robot.last_recovery_athlete_suppressed = bool(
                    not np.allclose(
                        learned_world_command,
                        raw_learned_world_command,
                        rtol=0.0,
                        atol=1.0e-9,
                    )
                )
                robot.recovery_athlete_suppressed_frame_count += int(
                    robot.last_recovery_athlete_suppressed
                )
            blend = config.recovery_athlete_blend
            world_velocity_x = float(
                np.clip(
                    (1.0 - blend) * world_velocity_x + blend * learned_world_command[0],
                    -config.maximum_depth_correction_mps,
                    config.maximum_depth_correction_mps,
                )
            )
            world_velocity_y = float(
                np.clip(
                    (1.0 - blend) * world_velocity_y + blend * learned_world_command[1],
                    -config.post_contact_ready_maximum_lateral_speed_mps,
                    config.post_contact_ready_maximum_lateral_speed_mps,
                )
            )
            yaw_rate = float(
                np.clip(
                    (1.0 - blend) * yaw_rate + blend * learned_world_command[2],
                    -config.post_contact_ready_maximum_yaw_rate_rad_s,
                    config.post_contact_ready_maximum_yaw_rate_rad_s,
                )
            )
            robot.last_recovery_athlete_active = True
            robot.last_recovery_athlete_raw_world_command = raw_learned_world_command
            robot.last_recovery_athlete_world_command = learned_world_command.copy()
            robot.recovery_athlete_active_frame_count += 1
        local_velocity = _rotate_z(
            np.asarray((world_velocity_x, world_velocity_y, 0.0), dtype=np.float64),
            -current_yaw,
        )
        robot.state.vel_cmd = _normalized_locomotion_command(
            robot.standby_policy,
            np.asarray((local_velocity[0], local_velocity[1], yaw_rate), dtype=np.float64),
        )
        return world_velocity_y, target_y, False, False, observation
    flight_start = observation.observed_flight_start_sec
    learned_action = (
        None
        if (
            learned_actor is None
            or not _goalkeeper_actor_route_active(
                robot,
                observation=observation,
                config=config,
                timestamp_sec=timestamp,
            )
        )
        else learned_actor.action(observation)
    )
    if learned_action is None:
        reaction_active = bool(
            flight_start is not None and timestamp >= flight_start + config.reaction_delay_sec
        )
    else:
        reaction_active = bool(
            abs(learned_action.lateral_velocity_mps) > 0.02
            or np.linalg.norm(learned_action.joint_position_residual_rad) > 0.02
        )
    ball = np.asarray(data.qpos[ball_qpos : ball_qpos + 3], dtype=np.float64)
    local_velocity = np.asarray(observation.estimated_ball_velocity_mps, dtype=np.float64)
    world_velocity = _rotate_z(local_velocity, _yaw(robot))
    prestrike_positioning_active = bool(
        config.prestrike_ball_positioning_enabled
        and flight_start is None
        and not robot.contact_latched
    )
    if reaction_active and world_velocity[0] > 0.10:
        horizon = max(0.0, (goal.plane_x_m - float(ball[0])) / float(world_velocity[0]))
        target_y = float(ball[1] + world_velocity[1] * horizon)
        target_y = float(np.clip(target_y, -goal.width_m / 2.0 + 0.22, goal.width_m / 2.0 - 0.22))
    elif prestrike_positioning_active:
        target_y = _prestrike_positioning_target(
            ball_y_m=float(ball[1]),
            anchor_y_m=config.initial_lateral_position_m,
            goal_width_m=goal.width_m,
            blend=config.prestrike_ball_lateral_blend,
        )
    else:
        phase = 2.0 * math.pi * timestamp / config.ready_shuffle_period_sec
        target_y = config.initial_lateral_position_m + 0.12 * math.sin(phase)
    current_y = float(data.qpos[robot.qpos_base + 1])
    analytic_velocity_y = float(
        np.clip(
            config.lateral_position_gain * (target_y - current_y),
            -config.maximum_lateral_speed_mps,
            config.maximum_lateral_speed_mps,
        )
    )
    if learned_action is not None and reaction_active:
        # The actor observes the goalkeeper-local frame; the keeper faces -x,
        # so its learned local +y velocity is world -y.
        learned_velocity_y = float(
            np.clip(
                -learned_action.lateral_velocity_mps,
                -config.maximum_lateral_speed_mps,
                config.maximum_lateral_speed_mps,
            )
        )
        # A learned hesitation must not suppress the causal position servo.
        # Keep the actor when it is target-directed, but give the locomotion
        # foundation a minimum useful command proportional to the measured
        # intercept error.  Wrong-way neural commands fail closed to the
        # analytical command rather than moving out of the shot path.
        if learned_velocity_y * analytic_velocity_y > 0.0:
            world_velocity_y = math.copysign(
                max(abs(learned_velocity_y), 0.85 * abs(analytic_velocity_y)),
                analytic_velocity_y,
            )
        else:
            world_velocity_y = analytic_velocity_y
    else:
        world_velocity_y = analytic_velocity_y
    if not reaction_active and not prestrike_positioning_active:
        world_velocity_y = float(
            np.clip(
                world_velocity_y,
                -config.ready_shuffle_speed_mps,
                config.ready_shuffle_speed_mps,
            )
        )
    local_depth_velocity = _goalkeeper_local_depth_velocity(
        robot,
        data=data,
        goal=goal,
        config=config,
    )
    robot.state.vel_cmd = _normalized_locomotion_command(
        robot.standby_policy,
        np.asarray((local_depth_velocity, -world_velocity_y, 0.0), dtype=np.float64),
    )
    return (
        world_velocity_y,
        target_y,
        reaction_active,
        prestrike_positioning_active,
        observation,
    )


def _prestrike_positioning_target(
    *,
    ball_y_m: float,
    anchor_y_m: float,
    goal_width_m: float,
    blend: float,
) -> float:
    """Position from the visible ball lane without reading the planned shot target.

    A regulation goalkeeper cannot cross the full 7.32 m mouth after a hard
    shot is already airborne.  This causal split-step target uses only the
    current physical ball position and the configured neutral anchor.  It is
    deliberately independent of ``goal.target_y_m`` so a hidden future target
    cannot leak into the controller.
    """

    values = (ball_y_m, anchor_y_m, goal_width_m, blend)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("goalkeeper prestrike positioning state must be finite")
    if goal_width_m <= 0.44 or not 0.0 <= blend <= 1.0:
        raise ValueError("goalkeeper prestrike positioning bounds are invalid")
    mouth_limit = goal_width_m / 2.0 - 0.22
    target = anchor_y_m + blend * (ball_y_m - anchor_y_m)
    return float(np.clip(target, -mouth_limit, mouth_limit))


def _goalkeeper_current_relative_intercept_y(
    robot: _Robot,
    *,
    anchor_relative_intercept_y_m: float,
) -> float:
    """Convert a stable-anchor intercept into current-root reach space.

    The visible-ball observer intentionally uses the immutable spawn frame so
    harmless goalkeeper sway cannot look like a new incoming shot.  Once a
    keeper has causally pre-positioned across a regulation goal, however, the
    reachability gate must measure the remaining distance from the *current*
    body.  Keeping these frames separate preserves threat segmentation while
    preventing a reachable corner from being rejected as several metres away.
    """

    if not math.isfinite(anchor_relative_intercept_y_m):
        raise ValueError("goalkeeper intercept must be finite")
    pelvis_local = np.asarray(robot.state.pelvis_pos_w, dtype=np.float64)
    if pelvis_local.shape != (3,) or not np.all(np.isfinite(pelvis_local)):
        raise ValueError("goalkeeper pelvis state must be a finite local position")
    return float(anchor_relative_intercept_y_m - pelvis_local[1])


def _goalkeeper_local_depth_velocity(
    robot: _Robot,
    *,
    data: Any,
    goal: G1TrainingGoalSpec,
    config: G1GoalkeeperConfig,
) -> float:
    """Keep a lateral save inside the goalkeeper's trained depth corridor."""

    desired_world_x = goal.plane_x_m - config.depth_from_goal_line_m
    current_world_x = float(data.qpos[robot.qpos_base])
    world_velocity_x = float(
        np.clip(
            config.depth_position_gain * (desired_world_x - current_world_x),
            -config.maximum_depth_correction_mps,
            config.maximum_depth_correction_mps,
        )
    )
    # The goalkeeper is yawed by pi, so local forward is world -x.
    return -world_velocity_x


def _external_reference_region(*, local_y_m: float, height_m: float) -> int:
    """Map a causal intercept to the six published teacher skill families."""

    if not math.isfinite(local_y_m) or not math.isfinite(height_m):
        raise ValueError("reference teacher region requires a finite intercept")
    side = 0 if local_y_m >= 0.0 else 1
    if height_m >= 1.15:
        return 2 + side
    if height_m <= 0.55:
        return 4 + side
    return side


def _apply_goalkeeper_operational_space_reach(
    robot: _Robot,
    *,
    model: Any,
    data: Any,
    observation: GoalkeeperActorObservation,
    artifact: Any,
    target_local_x_m: float,
    target_side_x_offset_m: float,
    reach_fraction: float,
    elapsed_sec: float,
) -> bool:
    """Turn the learned causal intercept into a bounded whole-arm reach.

    This is a generic damped least-squares task-space controller.  It does not
    encode a named shot or future trajectory.  The neural actor owns the
    activation and causal intercept; the local MuJoCo kinematics own the safe
    joint mapping.  Only one arm is writable, leaving the waist and both legs
    to the qualified locomotion cerebellum.
    """

    import mujoco

    if robot.goalkeeper_reach_memory is None:
        robot.goalkeeper_reach_memory = np.zeros(29, dtype=np.float64)
    if observation.intercept_confidence < 0.20 or elapsed_sec < 0.04:
        robot.goalkeeper_reach_memory *= float(artifact.operational_space_memory_decay)
        return False
    if not math.isfinite(reach_fraction) or not 0.0 <= reach_fraction <= 1.0:
        raise ValueError("goalkeeper operational reach activation is invalid")
    _, local_y, height = observation.estimated_intercept
    if not math.isfinite(target_local_x_m) or not -0.35 <= target_local_x_m <= 0.35:
        raise ValueError("goalkeeper operational reach depth is invalid")
    if not math.isfinite(target_side_x_offset_m) or not -0.20 <= target_side_x_offset_m <= 0.20:
        raise ValueError("goalkeeper operational side reach depth is invalid")
    # Positive local x points toward the incoming ball for the goalkeeper,
    # whose spawn yaw is pi.  Candidate curricula can therefore learn to meet
    # a fast shot in front of the body instead of chasing it toward the line.
    target_world = robot.origin + _rotate_z(
        np.asarray((target_local_x_m, local_y, height), dtype=np.float64),
        _yaw(robot),
    )
    pelvis_y = float(data.qpos[robot.qpos_base + 1])
    target_world[1] = float(np.clip(target_world[1], pelvis_y - 0.52, pelvis_y + 0.52))
    target_world[2] = float(np.clip(target_world[2], 0.24, 1.48))
    # The replicated goalkeeper is yawed by pi: its local +y axis maps to
    # world -y, and its anatomical left arm therefore covers world -y.
    if float(target_world[1]) <= pelvis_y:
        hand_body = robot.left_hand_body
        writable: NDArray[np.int64] = np.arange(15, 22, dtype=np.int64)
        side = -1
    else:
        hand_body = robot.right_hand_body
        writable = np.arange(22, 29, dtype=np.int64)
        side = 1
    side_depth = side * target_side_x_offset_m
    if not -0.35 <= target_local_x_m + side_depth <= 0.35:
        raise ValueError("goalkeeper operational side reach exceeds its envelope")
    target_world += _rotate_z(
        np.asarray((side_depth, 0.0, 0.0), dtype=np.float64),
        _yaw(robot),
    )
    if robot.goalkeeper_reach_memory_side not in (0, side):
        # A causal estimate may switch side during the first noisy frames.
        # Never carry a remembered reach into the opposite arm.
        robot.goalkeeper_reach_memory.fill(0.0)
    robot.goalkeeper_reach_memory_side = side
    # The wrist body's origin sits before the rubber palm.  Control the same
    # offset used by the collision envelope, otherwise inverse kinematics can
    # rotate a wrist without actually moving the contacting surface.
    hand_offset_local = np.asarray(G1_GOALKEEPER_GLOVE_CENTER_M, dtype=np.float64)
    hand_rotation = np.asarray(data.xmat[hand_body], dtype=np.float64).reshape(3, 3)
    hand_offset_world = hand_rotation @ hand_offset_local
    current = np.asarray(data.xpos[hand_body], dtype=np.float64) + hand_offset_world
    error = target_world - current
    if float(np.linalg.norm(error)) <= 0.12:
        robot.goalkeeper_reach_memory *= float(artifact.operational_space_memory_decay)
        robot.last_target += robot.goalkeeper_reach_memory
        return True
    jacobian_position = np.zeros((3, model.nv), dtype=np.float64)
    jacobian_rotation = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jac(
        model,
        data,
        jacobian_position,
        jacobian_rotation,
        current,
        hand_body,
    )
    dofs = robot.joint_qvel[writable]
    jacobian = jacobian_position[:, dofs]
    damping = float(artifact.operational_space_reach_damping)
    normal = jacobian @ jacobian.T + damping * damping * np.eye(3)
    delta = jacobian.T @ np.linalg.solve(normal, error)
    delta *= float(artifact.operational_space_reach_gain) * 0.70
    limit = float(artifact.operational_space_reach_maximum_step_rad)
    delta = np.clip(delta, -limit, limit)
    ramp = float(np.clip(elapsed_sec / float(artifact.operational_space_reach_ramp_sec), 0.0, 1.0))
    ramp = ramp * ramp * (3.0 - 2.0 * ramp)
    decay = float(artifact.operational_space_memory_decay)
    robot.goalkeeper_reach_memory *= decay
    robot.goalkeeper_reach_memory[writable] += ramp * reach_fraction * delta
    memory_limit = float(artifact.operational_space_memory_maximum_rad)
    ranges = model.jnt_range[robot.joint_ids[writable]]
    limited = model.jnt_limited[robot.joint_ids[writable]].astype(bool)
    # Memory is a bounded offset from the current locomotion target.  In
    # addition to the artifact ceiling, project it into the actual qualified
    # joint range before it can reach the PD controller.
    lower_offset = ranges[:, 0] + 0.08 - robot.last_target[writable]
    upper_offset = ranges[:, 1] - 0.08 - robot.last_target[writable]
    memory = np.clip(robot.goalkeeper_reach_memory[writable], -memory_limit, memory_limit)
    memory[limited] = np.clip(memory[limited], lower_offset[limited], upper_offset[limited])
    robot.goalkeeper_reach_memory[writable] = memory
    robot.goalkeeper_reach_memory_peak_rad = max(
        robot.goalkeeper_reach_memory_peak_rad,
        float(np.max(np.abs(robot.goalkeeper_reach_memory))),
    )
    robot.last_target += robot.goalkeeper_reach_memory
    selected = robot.last_target[writable]
    selected[limited] = np.clip(
        selected[limited],
        ranges[limited, 0] + 0.08,
        ranges[limited, 1] - 0.08,
    )
    robot.last_target[writable] = selected
    return True


def _apply_goalkeeper_bimanual_support_arm(
    robot: _Robot,
    *,
    local_intercept_y_m: float,
    blend: float,
    overhead_bias_rad: float,
) -> None:
    """Mirror the GMT target-hand motion into a bounded two-hand save pose."""

    if not all(math.isfinite(value) for value in (local_intercept_y_m, blend, overhead_bias_rad)):
        raise ValueError("goalkeeper support-arm state must be finite")
    if not 0.0 <= blend <= 1.0:
        raise ValueError("goalkeeper support-arm blend is invalid")
    if not 0.0 <= overhead_bias_rad <= 0.35:
        raise ValueError("goalkeeper support-arm overhead bias is invalid")
    if blend <= 0.0 and overhead_bias_rad <= 0.0:
        return
    mirror_sign = np.asarray((1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0))
    if local_intercept_y_m <= 0.0:
        target_slice = slice(22, 29)
        support_slice = slice(15, 22)
        support_shoulder_pitch, support_elbow = 15, 18
    else:
        target_slice = slice(15, 22)
        support_slice = slice(22, 29)
        support_shoulder_pitch, support_elbow = 22, 25
    target_delta = robot.last_target[target_slice] - robot.hold_target[target_slice]
    mirrored_target = robot.hold_target[support_slice] + mirror_sign * target_delta
    robot.last_target[support_slice] = (1.0 - blend) * robot.last_target[
        support_slice
    ] + blend * mirrored_target
    # Preserve the task-space reach as the primary authority while keeping the
    # non-contact glove visibly available above the shoulder.  This is a
    # bounded posture bias, not a rendered animation: it still flows through
    # joint projection, torque limits and the strict CPU physics replay.
    robot.last_target[support_shoulder_pitch] -= overhead_bias_rad
    robot.last_target[support_elbow] -= 0.70 * overhead_bias_rad


def _goalkeeper_bimanual_punch_torque(
    robot: _Robot,
    *,
    model: Any,
    data: Any,
    observation: GoalkeeperActorObservation,
    force_n: float,
    vertical_force_scale: float,
    outward_force_scale: float,
    window_sec: float,
) -> tuple[NDArray[np.float64], bool]:
    """Project a bounded two-glove punch through measured arm Jacobians."""

    import mujoco

    torque: NDArray[np.float64] = np.zeros(29, dtype=np.float64)
    time_to_arrival = float(observation.estimated_intercept[0])
    if (
        not math.isfinite(time_to_arrival)
        or not math.isfinite(force_n)
        or not math.isfinite(vertical_force_scale)
        or not math.isfinite(outward_force_scale)
        or not math.isfinite(window_sec)
    ):
        raise ValueError("goalkeeper bimanual punch state must be finite")
    if not 0.0 <= vertical_force_scale <= 1.0:
        raise ValueError("goalkeeper bimanual vertical punch scale is invalid")
    if not 0.0 <= outward_force_scale <= 0.75:
        raise ValueError("goalkeeper bimanual outward punch scale is invalid")
    if force_n <= 0.0 or time_to_arrival < -0.04 or time_to_arrival > window_sec:
        return torque, False
    phase = float(np.clip(1.0 - max(0.0, time_to_arrival) / window_sec, 0.0, 1.0))
    envelope = phase * phase * (3.0 - 2.0 * phase)
    # Project the learned punch intent in the goalkeeper's local frame.  The
    # asset's arm Jacobian convention has been verified by strict replay: the
    # positive local-X projection dissipates incoming ball speed, whereas the
    # opposite sign amplifies the post-contact speed and destabilises support.
    force_local = _goalkeeper_punch_force_local(
        force_n=force_n * envelope,
        local_intercept_y_m=float(observation.estimated_intercept[1]),
        vertical_force_scale=vertical_force_scale,
        outward_force_scale=outward_force_scale,
    )
    force_world = _rotate_z(force_local, _yaw(robot))
    force_world[2] = force_local[2]
    hand_contracts: tuple[tuple[int, NDArray[np.int64]], ...] = (
        (robot.left_hand_body, np.arange(15, 22, dtype=np.int64)),
        (robot.right_hand_body, np.arange(22, 29, dtype=np.int64)),
    )
    for hand_body, writable in hand_contracts:
        hand_rotation = np.asarray(data.xmat[hand_body], dtype=np.float64).reshape(3, 3)
        palm = np.asarray(data.xpos[hand_body], dtype=np.float64) + hand_rotation @ np.asarray(
            G1_GOALKEEPER_GLOVE_CENTER_M, dtype=np.float64
        )
        jacobian_position = np.zeros((3, model.nv), dtype=np.float64)
        jacobian_rotation = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jac(
            model,
            data,
            jacobian_position,
            jacobian_rotation,
            palm,
            hand_body,
        )
        torque[writable] = jacobian_position[:, robot.joint_qvel[writable]].T @ force_world
    if not np.all(np.isfinite(torque)):
        raise FloatingPointError("goalkeeper bimanual punch emitted non-finite torque")
    return torque, envelope > 1.0e-9


def _goalkeeper_punch_force_local(
    *,
    force_n: float,
    local_intercept_y_m: float,
    vertical_force_scale: float,
    outward_force_scale: float,
) -> NDArray[np.float64]:
    """Build a bounded intercept-conditioned force that clears the near post."""

    values = (force_n, local_intercept_y_m, vertical_force_scale, outward_force_scale)
    if not all(math.isfinite(value) for value in values) or force_n < 0.0:
        raise ValueError("goalkeeper punch force state must be finite and non-negative")
    if not 0.0 <= vertical_force_scale <= 1.0 or not 0.0 <= outward_force_scale <= 0.75:
        raise ValueError("goalkeeper punch force scales are invalid")
    outward_direction = (
        0.0 if abs(local_intercept_y_m) < 1.0e-9 else math.copysign(1.0, local_intercept_y_m)
    )
    return np.asarray(
        (
            force_n,
            outward_direction * force_n * outward_force_scale,
            force_n * vertical_force_scale,
        ),
        dtype=np.float64,
    )


def _apply_goalkeeper_bimanual_operational_space_reach(
    robot: _Robot,
    *,
    model: Any,
    data: Any,
    observation: GoalkeeperActorObservation,
    artifact: Any,
    target_local_x_m: float,
    half_span_m: float,
    height_offset_m: float,
    reach_fraction: float,
    gain_scale: float,
    memory_decay: float,
    memory_maximum_rad: float,
    elapsed_sec: float,
) -> bool:
    """Decode one causal high-ball intent into a bounded two-glove pocket.

    Both arms share the learned actor's activation and intercept, but each arm
    is projected through its measured MuJoCo Jacobian.  Legs and waist remain
    owned by the qualified locomotion policy.  The small target separation is
    expressed in task space rather than as a named animation, so the same
    primitive can close around any visible high intercept inside its envelope.
    """

    import mujoco

    if robot.goalkeeper_reach_memory is None:
        robot.goalkeeper_reach_memory = np.zeros(29, dtype=np.float64)
    if observation.intercept_confidence < 0.20 or elapsed_sec < 0.04:
        robot.goalkeeper_reach_memory *= float(artifact.operational_space_memory_decay)
        return False
    values = (
        target_local_x_m,
        half_span_m,
        height_offset_m,
        reach_fraction,
        gain_scale,
        memory_decay,
        memory_maximum_rad,
        elapsed_sec,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("goalkeeper bimanual reach state must be finite")
    if (
        not -0.35 <= target_local_x_m <= 0.35
        or not 0.04 <= half_span_m <= 0.20
        or not -0.12 <= height_offset_m <= 0.12
        or not 0.0 <= reach_fraction <= 1.0
        or not 0.5 <= gain_scale <= 4.0
        or not 0.75 <= memory_decay <= 0.98
        or not 0.20 <= memory_maximum_rad <= 0.80
    ):
        raise ValueError("goalkeeper bimanual reach state is outside its envelope")
    _, local_y, height = observation.estimated_intercept
    target_center = robot.origin + _rotate_z(
        np.asarray(
            (target_local_x_m, local_y, height + height_offset_m),
            dtype=np.float64,
        ),
        _yaw(robot),
    )
    pelvis_y = float(data.qpos[robot.qpos_base + 1])
    target_center[1] = float(np.clip(target_center[1], pelvis_y - 0.42, pelvis_y + 0.42))
    target_center[2] = float(np.clip(target_center[2], 0.72, 1.55))
    decay = memory_decay
    robot.goalkeeper_reach_memory *= decay
    active = False
    hand_contracts: tuple[tuple[int, NDArray[np.int64], float], ...] = (
        (robot.left_hand_body, np.arange(15, 22, dtype=np.int64), -half_span_m),
        (robot.right_hand_body, np.arange(22, 29, dtype=np.int64), half_span_m),
    )
    for hand_body, writable, lateral_offset in hand_contracts:
        target_world = target_center.copy()
        target_world[1] += lateral_offset
        hand_rotation = np.asarray(data.xmat[hand_body], dtype=np.float64).reshape(3, 3)
        current = np.asarray(data.xpos[hand_body], dtype=np.float64) + hand_rotation @ np.asarray(
            G1_GOALKEEPER_GLOVE_CENTER_M, dtype=np.float64
        )
        error = target_world - current
        if float(np.linalg.norm(error)) <= 0.07:
            active = True
            continue
        jacobian_position = np.zeros((3, model.nv), dtype=np.float64)
        jacobian_rotation = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jac(
            model,
            data,
            jacobian_position,
            jacobian_rotation,
            current,
            hand_body,
        )
        jacobian = jacobian_position[:, robot.joint_qvel[writable]]
        damping = float(artifact.operational_space_reach_damping)
        normal = jacobian @ jacobian.T + damping * damping * np.eye(3)
        delta = jacobian.T @ np.linalg.solve(normal, error)
        delta *= float(artifact.operational_space_reach_gain) * gain_scale
        limit = float(artifact.operational_space_reach_maximum_step_rad)
        delta = np.clip(delta, -limit, limit)
        ramp = float(
            np.clip(elapsed_sec / float(artifact.operational_space_reach_ramp_sec), 0.0, 1.0)
        )
        ramp = ramp * ramp * (3.0 - 2.0 * ramp)
        robot.goalkeeper_reach_memory[writable] += ramp * reach_fraction * delta
        active = True
    memory_limit = memory_maximum_rad
    writable = np.arange(15, 29, dtype=np.int64)
    ranges = model.jnt_range[robot.joint_ids[writable]]
    limited = model.jnt_limited[robot.joint_ids[writable]].astype(bool)
    lower_offset = ranges[:, 0] + 0.08 - robot.last_target[writable]
    upper_offset = ranges[:, 1] - 0.08 - robot.last_target[writable]
    memory = np.clip(robot.goalkeeper_reach_memory[writable], -memory_limit, memory_limit)
    memory[limited] = np.clip(memory[limited], lower_offset[limited], upper_offset[limited])
    robot.goalkeeper_reach_memory[writable] = memory
    robot.goalkeeper_reach_memory_peak_rad = max(
        robot.goalkeeper_reach_memory_peak_rad,
        float(np.max(np.abs(robot.goalkeeper_reach_memory))),
    )
    robot.last_target += robot.goalkeeper_reach_memory
    selected = robot.last_target[writable]
    selected[limited] = np.clip(
        selected[limited],
        ranges[limited, 0] + 0.08,
        ranges[limited, 1] - 0.08,
    )
    robot.last_target[writable] = selected
    return active


def _apply_goalkeeper_reach(
    robot: _Robot,
    *,
    target_y_m: float,
    current_y_m: float,
    reaction_active: bool,
    block_frame: int,
    config: G1GoalkeeperConfig,
) -> bool:
    """Add a bounded reach/lean to the locomotion target after ball reaction."""

    if not reaction_active:
        return False
    direction = float(np.sign(target_y_m - current_y_m))
    magnitude = float(np.clip(abs(target_y_m - current_y_m) / 0.9, 0.0, 1.0))
    robot.last_target[13] += direction * config.maximum_waist_lean_rad * magnitude
    robot.last_target[16] += config.arm_spread_rad
    robot.last_target[23] -= config.arm_spread_rad
    # Reach farther with the arm on the predicted side while keeping both
    # elbows flexed enough to avoid self-collision.
    if direction >= 0.0:
        robot.last_target[22] -= 0.14 * magnitude
        robot.last_target[25] = max(robot.last_target[25] - 0.12 * magnitude, 0.35)
    else:
        robot.last_target[15] -= 0.14 * magnitude
        robot.last_target[18] = max(robot.last_target[18] - 0.12 * magnitude, 0.35)

    block_action_active = bool(
        config.block_action_enabled
        and 0 <= block_frame < config.block_action_blend_frames + config.block_action_hold_frames
    )
    if not block_action_active:
        return False
    blend_phase = min(1.0, (block_frame + 1) / config.block_action_blend_frames)
    block_blend = blend_phase * blend_phase * (3.0 - 2.0 * blend_phase)
    signed_blend = direction * block_blend
    robot.last_target[12] += signed_blend * config.block_action_waist_yaw_rad
    robot.last_target[13] += signed_blend * config.block_action_waist_roll_rad
    robot.last_target[14] += block_blend * config.block_action_waist_pitch_rad
    if direction >= 0.0:
        leg_pitch_index, leg_roll_index, knee_index = 6, 7, 9
        shoulder_pitch_index, shoulder_roll_index, elbow_index = 22, 23, 25
    else:
        leg_pitch_index, leg_roll_index, knee_index = 0, 1, 3
        shoulder_pitch_index, shoulder_roll_index, elbow_index = 15, 16, 18
    robot.last_target[leg_pitch_index] += block_blend * config.block_action_hip_pitch_rad
    robot.last_target[leg_roll_index] -= signed_blend * config.block_action_hip_roll_rad
    robot.last_target[knee_index] += block_blend * config.block_action_knee_flex_rad
    robot.last_target[shoulder_pitch_index] += block_blend * config.block_action_shoulder_pitch_rad
    robot.last_target[shoulder_roll_index] -= signed_blend * config.block_action_shoulder_roll_rad
    robot.last_target[elbow_index] += block_blend * config.block_action_elbow_flex_rad
    return True


def _update_policy(
    robot: _Robot,
    simulation_frame: int,
    *,
    timestamp_sec: float,
) -> int:
    robot.last_recovery_active = False
    robot.last_recovery_blend_fraction = 0.0
    if robot.causal_strike_bridge is not None and not robot.entered:
        controller = robot.causal_strike_option
        bridge_kp = robot.causal_strike_bridge_kp
        bridge_kd = robot.causal_strike_bridge_kd
        if controller is None or bridge_kp is None or bridge_kd is None:
            raise RuntimeError("causal strike bridge state is incomplete")
        _reset_inactive_imitation_channels(robot)
        bridge_frame = robot.causal_strike_bridge_frame
        bridge_frames = robot.causal_strike_bridge_frames
        if not 0 <= bridge_frame < bridge_frames:
            raise RuntimeError("causal strike bridge frame is outside its contract")
        sample = robot.causal_strike_bridge.sample((bridge_frame + 1) * _CONTROL_DT)
        robot.last_target = np.asarray(sample.position, dtype=np.float64).copy()
        robot.target_velocity = np.asarray(sample.velocity, dtype=np.float64).copy()
        robot.kp = bridge_kp.copy()
        robot.kd = bridge_kd.copy()
        velocity_rms = float(np.sqrt(np.mean(np.square(robot.target_velocity))))
        robot.causal_strike_bridge_peak_target_velocity_rms_rad_s = max(
            robot.causal_strike_bridge_peak_target_velocity_rms_rad_s,
            velocity_rms,
        )
        phase_start = robot.causal_strike_bridge_phase_start
        if bridge_frame >= bridge_frames - controller.config.history_prime_frames:
            robot.policy.time_step = (
                robot.policy.WARMUP_STEPS + phase_start - (bridge_frames - 1 - bridge_frame)
            )
            with _canonical_kick_policy_state(robot), contextlib.redirect_stdout(io.StringIO()):
                robot.policy._build_obs()
        robot.causal_strike_bridge_frame += 1
        if robot.causal_strike_bridge_frame == bridge_frames:
            robot.policy.time_step = robot.policy.WARMUP_STEPS + phase_start
            robot.entered = True
        return phase_start
    if not robot.entered:
        robot.target_velocity.fill(0.0)
        robot.last_motion_prior_position_active = False
        robot.last_motion_prior_velocity_active = False
        robot.last_motion_prior_target_delta = np.zeros(29, dtype=np.float64)
        robot.last_motion_prior_velocity_delta = np.zeros(29, dtype=np.float64)
        _reset_inactive_imitation_channels(robot)
        if robot.standby_policy is not None and robot.standby_output is not None:
            _run_standby_locomotion(robot)
            robot.last_target = np.asarray(robot.standby_output.actions, dtype=np.float64).copy()
            robot.kp = np.asarray(robot.standby_output.kps, dtype=np.float64).copy()
            robot.kd = np.asarray(robot.standby_output.kds, dtype=np.float64).copy()
        else:
            robot.last_target = robot.hold_target.copy()
        return 0
    current_policy_frame = max(0, int(robot.policy.time_step) - int(robot.policy.WARMUP_STEPS))
    robot.last_phase_correction = 0
    if robot.causal_strike_option is not None:
        robot.causal_strike_option.observe_policy_progress(current_policy_frame)
    if (
        not robot.post_policy_active
        and robot.causal_strike_option is not None
        and robot.causal_strike_option.phase == CausalStrikeOptionPhase.ABORTED
    ):
        robot.post_policy_active = True
        robot.post_policy_origin_target = robot.last_target.copy()
        robot.post_policy_origin_kp = robot.kp.copy()
        robot.post_policy_origin_kd = robot.kd.copy()
        robot.post_policy_activation_simulation_frame = simulation_frame
        robot.post_policy_blend_frames = max(
            robot.post_policy_blend_frames,
            robot.causal_strike_option.config.abort_recovery_blend_frames,
        )
        robot.causal_strike_abort_recovery_activated = True
    if (
        not robot.post_policy_active
        and robot.post_policy_frame is not None
        and current_policy_frame >= robot.post_policy_frame
        and robot.contact_latched
    ):
        robot.post_policy_active = True
        robot.post_policy_origin_target = robot.last_target.copy()
        robot.post_policy_origin_kp = robot.kp.copy()
        robot.post_policy_origin_kd = robot.kd.copy()
        robot.post_policy_activation_simulation_frame = simulation_frame
    if robot.post_policy_active:
        robot.last_motion_prior_position_active = False
        robot.last_motion_prior_velocity_active = False
        robot.last_motion_prior_target_delta = np.zeros(29, dtype=np.float64)
        robot.last_motion_prior_velocity_delta = np.zeros(29, dtype=np.float64)
        _reset_inactive_imitation_channels(robot)
        if robot.standby_policy is None or robot.standby_output is None:
            raise RuntimeError("post-kick locomotion transition is unavailable")
        if robot.post_policy_neutral_velocity_enabled:
            robot.state.vel_cmd = _normalized_locomotion_command(
                robot.standby_policy,
                np.asarray(
                    (robot.post_policy_forward_velocity_mps, 0.0, 0.0),
                    dtype=np.float64,
                ),
            )
        _run_standby_locomotion(robot)
        standby_target = np.asarray(robot.standby_output.actions, dtype=np.float64)
        standby_kp = np.asarray(robot.standby_output.kps, dtype=np.float64)
        standby_kd = np.asarray(robot.standby_output.kds, dtype=np.float64)
        activation_frame = robot.post_policy_activation_simulation_frame
        if activation_frame is None:
            raise RuntimeError("post-policy activation frame is unavailable")
        recovery_policy_frame = current_policy_frame + max(0, simulation_frame - activation_frame)
        if robot.post_policy_recovery_enabled:
            standby_target, standby_kp, standby_kd = _apply_recovery_controller(
                robot,
                target=standby_target,
                kp=standby_kp,
                kd=standby_kd,
                policy_frame=recovery_policy_frame,
                timestamp_sec=timestamp_sec,
            )
        if robot.post_policy_blend_frames:
            origin_target = robot.post_policy_origin_target
            origin_kp = robot.post_policy_origin_kp
            origin_kd = robot.post_policy_origin_kd
            if origin_target is None or origin_kp is None or origin_kd is None:
                raise RuntimeError("post-policy transition origin is unavailable")
            transition_step = min(
                robot.post_policy_transition_step,
                robot.post_policy_blend_frames - 1,
            )
            (
                robot.last_target,
                robot.kp,
                robot.kd,
                fraction,
            ) = _smooth_policy_handoff(
                origin_target=origin_target,
                origin_kp=origin_kp,
                origin_kd=origin_kd,
                destination_target=standby_target,
                destination_kp=standby_kp,
                destination_kd=standby_kd,
                transition_step=transition_step,
                blend_frames=robot.post_policy_blend_frames,
            )
            robot.post_policy_transition_step += 1
            robot.post_policy_blend_fraction = fraction
        else:
            robot.last_target = standby_target.copy()
            robot.kp = standby_kp.copy()
            robot.kd = standby_kd.copy()
            robot.post_policy_blend_fraction = 1.0
        robot.target_velocity = _apply_agility_prior_velocity(
            robot,
            target_velocity=np.zeros(29, dtype=np.float64),
            policy_frame=recovery_policy_frame,
        )
        robot.last_target = _apply_agility_prior_target(
            robot,
            target=robot.last_target,
            policy_frame=recovery_policy_frame,
        )
        return recovery_policy_frame
    if current_policy_frame >= 185 and robot.phase_hold_remaining:
        robot.phase_hold_remaining -= 1
        return current_policy_frame
    repeat = _policy_repeat_count(
        robot.parameters.swing_speed_scale,
        current_policy_frame,
        simulation_frame,
    )
    if robot.causal_strike_option is not None:
        repeat, correction = robot.causal_strike_option.align_repeat_count(
            policy_frame=current_policy_frame,
            nominal_repeat=repeat,
        )
        if correction < 0:
            robot.phase_hold_count += 1
        elif correction > 0:
            robot.phase_advance_count += 1
        robot.last_phase_correction = correction
    elif (
        robot.phase_sync_enabled
        and robot.role == "shooter"
        and 184 <= current_policy_frame <= 252
        and float(robot.state.ball_vel_w[0]) < -0.05
    ):
        phase = float(current_policy_frame - 184)
        expected_ball_x = 1.32676487e-4 * phase * phase - 2.54767831e-2 * phase + 2.12799256
        phase_error = float(robot.state.ball_pos_w[0]) - expected_ball_x
        robot.max_ball_phase_error_m = max(
            robot.max_ball_phase_error_m,
            abs(phase_error),
        )
        if phase_error > 0.025:
            repeat = 0
            robot.phase_hold_count += 1
            robot.last_phase_correction = -1
        elif phase_error < -0.025:
            repeat = max(2, repeat + 1)
            robot.phase_advance_count += 1
            robot.last_phase_correction = 1
    if repeat:
        with _canonical_kick_policy_state(robot), contextlib.redirect_stdout(io.StringIO()):
            for _ in range(repeat):
                robot.policy.run()
        policy_frame = max(0, int(robot.policy.time_step) - int(robot.policy.WARMUP_STEPS))
        active_parameters = robot.parameters
        if (
            robot.role == "shooter"
            and robot.phase_hold_count > 0
            and robot.early_arrival_parameters is not None
        ):
            active_parameters = robot.early_arrival_parameters
            robot.early_arrival_expert_frame_count += 1
        target = _adapt_target(
            target=np.asarray(robot.output.actions, dtype=np.float64),
            default=np.asarray(robot.policy.default_q_mj, dtype=np.float64),
            parameters=active_parameters,
            policy_frame=policy_frame,
        )
        output_kp = np.asarray(robot.output.kps, dtype=np.float64).copy()
        output_kd = np.asarray(robot.output.kds, dtype=np.float64).copy()
        if active_parameters.kick_foot == "left":
            output_kp = _mirror_g1_joint_gains(output_kp)
            output_kd = _mirror_g1_joint_gains(output_kd)
        target, target_velocity = _apply_motion_prior(
            robot,
            target=target,
            policy_frame=policy_frame,
        )
        target, output_kp, output_kd = _apply_recovery_controller(
            robot,
            target=target,
            kp=output_kp,
            kd=output_kd,
            policy_frame=policy_frame,
            timestamp_sec=timestamp_sec,
        )
        if robot.last_recovery_active:
            target_velocity *= 1.0 - robot.last_recovery_blend_fraction
        robot.last_target = target
        robot.target_velocity = target_velocity
        robot.kp = output_kp
        robot.kd = output_kd
        return policy_frame
    return current_policy_frame


def _run_standby_locomotion(robot: _Robot) -> None:
    """Run the qualified locomotion prior in a mirrored canonical half-space.

    The published policy is substantially safer in local ``+y`` than ``-y``
    over a regulation-goal traversal.  When the explicit mirror route is
    active, proprioception and the physical command are reflected before the
    policy call, while targets and gains are reflected back afterwards.  The
    policy's previous-action memory therefore stays entirely canonical.
    """

    if robot.standby_policy is None or robot.standby_output is None:
        raise RuntimeError("standby locomotion policy is unavailable")
    if not robot.standby_locomotion_mirror_active:
        with contextlib.redirect_stdout(io.StringIO()):
            robot.standby_policy.run()
        return
    state = robot.state
    original = {
        name: np.asarray(getattr(state, name), dtype=np.float64).copy()
        for name in ("q", "dq", "gravity_ori", "ang_vel", "vel_cmd")
    }
    try:
        state.q = _mirror_g1_joint_positions(original["q"])
        state.dq = _mirror_g1_joint_positions(original["dq"])
        gravity = original["gravity_ori"].copy()
        gravity[1] *= -1.0
        state.gravity_ori = gravity
        angular = original["ang_vel"].copy()
        angular[(0, 2),] *= -1.0
        state.ang_vel = angular
        command = original["vel_cmd"].copy()
        command[1] *= -1.0
        state.vel_cmd = command
        with contextlib.redirect_stdout(io.StringIO()):
            robot.standby_policy.run()
        robot.standby_output.actions = _mirror_g1_joint_positions(
            np.asarray(robot.standby_output.actions, dtype=np.float64)
        )
        robot.standby_output.kps = _mirror_g1_joint_gains(
            np.asarray(robot.standby_output.kps, dtype=np.float64)
        )
        robot.standby_output.kds = _mirror_g1_joint_gains(
            np.asarray(robot.standby_output.kds, dtype=np.float64)
        )
    finally:
        for name, value in original.items():
            setattr(state, name, value)


def _apply_motion_prior(
    robot: _Robot,
    *,
    target: np.ndarray,
    policy_frame: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a bounded kinematic teacher through the existing PD contract."""

    target_velocity: NDArray[np.float64] = np.zeros(29, dtype=np.float64)
    target_delta: NDArray[np.float64] = np.zeros(29, dtype=np.float64)
    velocity_delta: NDArray[np.float64] = np.zeros(29, dtype=np.float64)
    position_active = False
    velocity_active = False
    if robot.motion_prior is not None:
        target, target_delta, position_active = blend_g1_football_motion_prior_target(
            target=target,
            prior=robot.motion_prior,
            policy_frame=policy_frame,
            contact_policy_frame=robot.motion_prior_contact_policy_frame,
            control_dt_sec=_CONTROL_DT,
            blend=robot.motion_prior_position_blend,
        )
        target_velocity, velocity_delta, velocity_active = blend_g1_football_motion_prior_velocity(
            target_velocity=target_velocity,
            prior=robot.motion_prior,
            policy_frame=policy_frame,
            contact_policy_frame=robot.motion_prior_contact_policy_frame,
            control_dt_sec=_CONTROL_DT,
            blend=robot.motion_prior_velocity_blend,
        )
        # Preserve the precision-critical contact expert while allowing the
        # support leg, waist and arms to learn whole-body style.  This is the
        # stability/plasticity boundary for a right-foot strike.
        strike_leg = slice(6, 12)
        target[strike_leg] -= target_delta[strike_leg] * (1.0 - robot.motion_prior_strike_leg_scale)
        target_delta[strike_leg] *= robot.motion_prior_strike_leg_scale
        target_velocity[strike_leg] -= velocity_delta[strike_leg] * (
            1.0 - robot.motion_prior_strike_leg_scale
        )
        velocity_delta[strike_leg] *= robot.motion_prior_strike_leg_scale
        position_scales = np.asarray(robot.motion_prior_joint_scales, dtype=np.float64)
        velocity_scales = np.asarray(
            robot.motion_prior_velocity_joint_scales,
            dtype=np.float64,
        )
        target -= target_delta * (1.0 - position_scales)
        target_delta *= position_scales
        target_velocity -= velocity_delta * (1.0 - velocity_scales)
        velocity_delta *= velocity_scales
        position_active = bool(np.any(np.abs(target_delta) > 1e-12))
        velocity_active = bool(np.any(np.abs(velocity_delta) > 1e-12))
    robot.last_motion_prior_target_delta = target_delta
    robot.last_motion_prior_velocity_delta = velocity_delta
    robot.last_motion_prior_position_active = position_active
    robot.last_motion_prior_velocity_active = velocity_active
    robot.motion_prior_peak_target_delta_rad = max(
        robot.motion_prior_peak_target_delta_rad,
        float(np.max(np.abs(target_delta))),
    )
    robot.motion_prior_peak_velocity_delta_rad_s = max(
        robot.motion_prior_peak_velocity_delta_rad_s,
        float(np.max(np.abs(velocity_delta))),
    )
    target = _apply_agility_prior_target(
        robot,
        target=target,
        policy_frame=policy_frame,
    )
    target_velocity = _apply_agility_prior_velocity(
        robot,
        target_velocity=target_velocity,
        policy_frame=policy_frame,
    )
    contact_delta: NDArray[np.float64] = np.zeros(29, dtype=np.float64)
    contact_velocity_delta: NDArray[np.float64] = np.zeros(29, dtype=np.float64)
    contact_active = False
    if robot.contact_prior is not None:
        target, contact_delta, contact_active = blend_g1_football_motion_prior_displacement(
            target=target,
            prior=robot.contact_prior,
            policy_frame=policy_frame,
            contact_policy_frame=robot.contact_prior_contact_policy_frame,
            control_dt_sec=_CONTROL_DT,
            blend=robot.contact_prior_position_blend,
        )
        strike_leg = slice(6, 12)
        scales = np.asarray(robot.contact_prior_joint_scales, dtype=np.float64)
        target[strike_leg] -= contact_delta[strike_leg] * (1.0 - scales)
        contact_delta[strike_leg] *= scales
        target_velocity, contact_velocity_delta, velocity_contact_active = (
            blend_g1_football_motion_prior_right_leg_velocity(
                target_velocity=target_velocity,
                prior=robot.contact_prior,
                policy_frame=policy_frame,
                contact_policy_frame=robot.contact_prior_contact_policy_frame,
                control_dt_sec=_CONTROL_DT,
                blend=robot.contact_prior_velocity_blend,
            )
        )
        target_velocity[strike_leg] -= contact_velocity_delta[strike_leg] * (1.0 - scales)
        contact_velocity_delta[strike_leg] *= scales
        contact_active = bool(np.any(np.abs(contact_delta) > 1e-12))
        contact_active = contact_active or velocity_contact_active
    robot.last_contact_prior_target_delta = contact_delta
    robot.last_contact_prior_velocity_delta = contact_velocity_delta
    robot.last_contact_prior_active = contact_active
    robot.contact_prior_peak_target_delta_rad = max(
        robot.contact_prior_peak_target_delta_rad,
        float(np.max(np.abs(contact_delta))),
    )
    robot.contact_prior_peak_velocity_delta_rad_s = max(
        robot.contact_prior_peak_velocity_delta_rad_s,
        float(np.max(np.abs(contact_velocity_delta))),
    )
    return target, target_velocity


def _reset_inactive_imitation_channels(robot: _Robot) -> None:
    """Clear accounting state when an imitation channel did not run."""

    robot.last_agility_prior_active = False
    robot.last_agility_prior_target_delta = np.zeros(29, dtype=np.float64)
    robot.last_agility_prior_velocity_delta = np.zeros(29, dtype=np.float64)
    robot.last_contact_prior_active = False
    robot.last_contact_prior_target_delta = np.zeros(29, dtype=np.float64)
    robot.last_contact_prior_velocity_delta = np.zeros(29, dtype=np.float64)


def _apply_agility_prior_velocity(
    robot: _Robot,
    *,
    target_velocity: np.ndarray,
    policy_frame: int,
) -> np.ndarray:
    """Apply and account for the isolated MOSAIC velocity channel."""

    agility_velocity_delta: NDArray[np.float64] = np.zeros(29, dtype=np.float64)
    agility_active = False
    if robot.agility_prior is not None:
        target_velocity, agility_velocity_delta, agility_active = blend_g1_mosaic_agility_velocity(
            target_velocity=target_velocity,
            prior=robot.agility_prior,
            policy_frame=policy_frame,
            contact_policy_frame=robot.agility_prior_contact_policy_frame,
            control_dt_sec=_CONTROL_DT,
            blend=robot.agility_prior_velocity_blend,
            joint_scales=robot.agility_prior_joint_scales,
        )
    robot.last_agility_prior_velocity_delta = agility_velocity_delta
    robot.last_agility_prior_active = robot.last_agility_prior_active or agility_active
    robot.agility_prior_peak_velocity_delta_rad_s = max(
        robot.agility_prior_peak_velocity_delta_rad_s,
        float(np.max(np.abs(agility_velocity_delta))),
    )
    return target_velocity


def _apply_agility_prior_target(
    robot: _Robot,
    *,
    target: np.ndarray,
    policy_frame: int,
) -> np.ndarray:
    """Apply and account for the isolated endpoint-neutral pose channel."""

    target_delta: NDArray[np.float64] = np.zeros(29, dtype=np.float64)
    active = False
    if robot.agility_prior is not None:
        target, target_delta, active = blend_g1_mosaic_agility_target(
            target=target,
            prior=robot.agility_prior,
            policy_frame=policy_frame,
            contact_policy_frame=robot.agility_prior_contact_policy_frame,
            control_dt_sec=_CONTROL_DT,
            blend=robot.agility_prior_position_blend,
            joint_scales=robot.agility_prior_joint_scales,
        )
    robot.last_agility_prior_target_delta = target_delta
    robot.last_agility_prior_active = active
    robot.agility_prior_peak_target_delta_rad = max(
        robot.agility_prior_peak_target_delta_rad,
        float(np.max(np.abs(target_delta))),
    )
    return target


def _apply_recovery_controller(
    robot: _Robot,
    *,
    target: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
    policy_frame: int,
    timestamp_sec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply one shared contact-gated cerebellar recovery contract.

    The helper is deliberately used on both sides of a policy handoff.  This
    prevents the recovery state machine from disappearing when the retained
    locomotion expert takes control after a pass or shot.
    """

    if robot.recovery_controller is None:
        return target.copy(), kp.copy(), kd.copy()
    left_foot_mirror = robot.parameters.kick_foot == "left"
    canonical_target = _mirror_g1_joint_positions(target) if left_foot_mirror else target
    canonical_left_support = (
        robot.latest_right_support if left_foot_mirror else robot.latest_left_support
    )
    canonical_right_support = (
        robot.latest_left_support if left_foot_mirror else robot.latest_right_support
    )
    recovery = robot.recovery_controller.adapt_target(
        target=canonical_target,
        policy_frame=policy_frame,
        timestamp_sec=timestamp_sec,
        ball_contact_detected=robot.contact_latched,
        left_support=canonical_left_support,
        right_support=canonical_right_support,
    )
    output_kp = kp.copy()
    output_kd = kd.copy()
    # ROSClaw Core before the optional terminal-damping extension does not
    # expose these effect fields.  Identity defaults preserve the frozen
    # recovery trajectory while allowing a downstream checkout to run against
    # either contract version.
    terminal_group = str(getattr(recovery, "terminal_damping_joint_group", "whole_body"))
    terminal_slice = {
        "whole_body": slice(None),
        "legs": slice(0, 12),
        "upper_body": slice(12, None),
    }[terminal_group]
    output_kp[terminal_slice] *= float(getattr(recovery, "terminal_kp_scale", 1.0))
    output_kd[terminal_slice] *= float(getattr(recovery, "terminal_kd_scale", 1.0))
    robot.last_recovery_active = recovery.active
    robot.last_recovery_blend_fraction = recovery.blend_fraction
    robot.recovery_active_frame_count += int(recovery.active)
    robot.recovery_peak_blend_fraction = max(
        robot.recovery_peak_blend_fraction,
        recovery.blend_fraction,
    )
    recovery_target = (
        _mirror_g1_joint_positions(recovery.target) if left_foot_mirror else recovery.target
    )
    return recovery_target, output_kp, output_kd


def _contacts(
    *,
    model: Any,
    data: Any,
    ball_geom: int,
    floor_geom: int,
    passer_geoms: frozenset[int],
    shooter_geoms: frozenset[int],
    goalkeeper_geoms: frozenset[int] = frozenset(),
    goalkeeper_left_glove_geoms: frozenset[int] = frozenset(),
    goalkeeper_right_glove_geoms: frozenset[int] = frozenset(),
    shooter_geom_prefix: str = "",
) -> dict[str, Any]:
    import mujoco

    result: dict[str, Any] = {
        "passer_left": False,
        "passer_right": False,
        "shooter_left": False,
        "shooter_right": False,
        "goalkeeper_left": False,
        "goalkeeper_right": False,
        "ball_passer": False,
        "ball_shooter": False,
        "ball_shooter_left": False,
        "ball_shooter_right": False,
        "ball_goalkeeper": False,
        "ball_goalkeeper_glove": False,
        "ball_goalkeeper_left_glove": False,
        "ball_goalkeeper_right_glove": False,
        "ball_goalkeeper_left_glove_surface_distance_m": None,
        "ball_goalkeeper_right_glove_surface_distance_m": None,
        "ball_passer_force_n": 0.0,
        "ball_shooter_force_n": 0.0,
        "robot_robot": False,
    }
    force: NDArray[np.float64] = np.zeros(6, dtype=np.float64)
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1) or ""
        name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2) or ""
        if floor_geom in {geom1, geom2}:
            other = name2 if geom1 == floor_geom else name1
            if other.startswith("passer_left_foot"):
                result["passer_left"] = True
            elif other.startswith("passer_right_foot"):
                result["passer_right"] = True
            elif other.startswith("goalkeeper_left_foot"):
                result["goalkeeper_left"] = True
            elif other.startswith("goalkeeper_right_foot"):
                result["goalkeeper_right"] = True
            elif other.startswith(shooter_geom_prefix + "left_foot"):
                result["shooter_left"] = True
            elif other.startswith(shooter_geom_prefix + "right_foot"):
                result["shooter_right"] = True
        if ball_geom in {geom1, geom2}:
            other_geom = geom2 if geom1 == ball_geom else geom1
            if other_geom in passer_geoms:
                result["ball_passer"] = True
                mujoco.mj_contactForce(model, data, index, force)
                result["ball_passer_force_n"] = max(
                    float(result["ball_passer_force_n"]), float(np.linalg.norm(force[:3]))
                )
            elif other_geom in shooter_geoms:
                result["ball_shooter"] = True
                other_name = name2 if geom1 == ball_geom else name1
                result["ball_shooter_left"] = other_name.startswith(
                    shooter_geom_prefix + "left_foot"
                )
                result["ball_shooter_right"] = other_name.startswith(
                    shooter_geom_prefix + "right_foot"
                )
                mujoco.mj_contactForce(model, data, index, force)
                result["ball_shooter_force_n"] = max(
                    float(result["ball_shooter_force_n"]), float(np.linalg.norm(force[:3]))
                )
            elif other_geom in goalkeeper_geoms:
                result["ball_goalkeeper"] = True
                if other_geom in goalkeeper_left_glove_geoms:
                    result["ball_goalkeeper_glove"] = True
                    result["ball_goalkeeper_left_glove"] = True
                    previous = result["ball_goalkeeper_left_glove_surface_distance_m"]
                    result["ball_goalkeeper_left_glove_surface_distance_m"] = (
                        float(contact.dist)
                        if previous is None
                        else min(float(previous), float(contact.dist))
                    )
                elif other_geom in goalkeeper_right_glove_geoms:
                    result["ball_goalkeeper_glove"] = True
                    result["ball_goalkeeper_right_glove"] = True
                    previous = result["ball_goalkeeper_right_glove_surface_distance_m"]
                    result["ball_goalkeeper_right_glove_surface_distance_m"] = (
                        float(contact.dist)
                        if previous is None
                        else min(float(previous), float(contact.dist))
                    )
        if (geom1 in passer_geoms and geom2 in shooter_geoms) or (
            geom2 in passer_geoms and geom1 in shooter_geoms
        ):
            result["robot_robot"] = True
        if goalkeeper_geoms and (
            (geom1 in goalkeeper_geoms and geom2 in (passer_geoms | shooter_geoms))
            or (geom2 in goalkeeper_geoms and geom1 in (passer_geoms | shooter_geoms))
        ):
            result["robot_robot"] = True
    return result


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
    *,
    model: Any,
    goalkeeper: _Robot,
) -> frozenset[int]:
    """Resolve the anatomical glove geoms used by physics and save scoring.

    The unified stadium owns the actual glove geometry.  The controller must
    bind to that exact scene contract instead of silently expecting the older
    spherical reach-envelope names or dimensions.
    """

    names = {
        "goalkeeper_left_goalkeeper_glove": goalkeeper.left_hand_body,
        "goalkeeper_right_goalkeeper_glove": goalkeeper.right_hand_body,
    }
    result: set[int] = set()
    for geom in range(int(model.ngeom)):
        name = model.geom(geom).name
        expected_body = names.get(name)
        if expected_body is None:
            continue
        if int(model.geom_bodyid[geom]) != expected_body:
            raise ValueError("goalkeeper glove is attached to the wrong body")
        expected_size = np.asarray(G1_GOALKEEPER_GLOVE_HALF_EXTENTS_M, dtype=np.float64)
        if not np.allclose(model.geom_size[geom], expected_size, atol=1e-9, rtol=0.0):
            raise ValueError("goalkeeper glove size changed")
        result.add(geom)
    if len(result) != 2:
        raise ValueError("goalkeeper scene lacks two anatomical glove collision geoms")
    return frozenset(result)


def _append_trace(
    trace: dict[str, list[Any]],
    *,
    data: Any,
    ball_qpos: int,
    ball_qvel: int,
    passer: _Robot,
    shooter: _Robot,
    policy_frames: dict[str, int],
    support: tuple[tuple[bool, bool], tuple[bool, bool]],
    contact_role: int,
    shooter_contact_foot: int,
    robot_contacts: int,
    commanded_torque: dict[str, np.ndarray],
    projected_torque: dict[str, np.ndarray],
    executed_torque: dict[str, np.ndarray],
    contact_impulse: dict[str, float],
    learned_torque: dict[str, np.ndarray | IQLResidualDecision | None],
    joint_guard_active: dict[str, bool],
) -> None:
    trace["time"].append(float(data.time))
    trace["ball_pose"].append(data.qpos[ball_qpos : ball_qpos + 7].copy())
    trace["ball_velocity"].append(data.qvel[ball_qvel : ball_qvel + 6].copy())
    for robot in (passer, shooter):
        trace[f"{robot.role}_pelvis_pose"].append(
            data.qpos[robot.qpos_base : robot.qpos_base + 7].copy()
        )
        trace[f"{robot.role}_torso_quaternion"].append(data.xquat[robot.torso_body].copy())
        trace[f"{robot.role}_joint_position"].append(data.qpos[robot.joint_qpos].copy())
        trace[f"{robot.role}_joint_velocity"].append(data.qvel[robot.joint_qvel].copy())
        trace[f"{robot.role}_joint_torque"].append(data.ctrl[robot.actuators].copy())
        trace[f"{robot.role}_commanded_torque"].append(commanded_torque[robot.role].copy())
        trace[f"{robot.role}_safety_projected_torque"].append(projected_torque[robot.role].copy())
        trace[f"{robot.role}_executed_torque"].append(executed_torque[robot.role].copy())
        trace[f"{robot.role}_policy_action"].append(robot.last_target.copy())
        trace[f"{robot.role}_com_position"].append(data.subtree_com[robot.pelvis_body].copy())
        trace[f"{robot.role}_left_foot_position"].append(data.xpos[robot.left_ankle_body].copy())
        trace[f"{robot.role}_right_foot_position"].append(data.xpos[robot.right_ankle_body].copy())
        trace[f"{robot.role}_support_foot_slip"].append(robot.latest_support_slip_m)
        trace[f"{robot.role}_contact_impulse"].append(contact_impulse[robot.role])
        trace[f"{robot.role}_joint_guard_active"].append(joint_guard_active[robot.role])
        trace[f"{robot.role}_post_policy_blend_fraction"].append(robot.post_policy_blend_fraction)
        trace[f"{robot.role}_recovery_active"].append(robot.last_recovery_active)
        trace[f"{robot.role}_recovery_blend_fraction"].append(robot.last_recovery_blend_fraction)
        trace[f"{robot.role}_policy_frame"].append(policy_frames[robot.role])
    trace["shooter_target_velocity"].append(shooter.target_velocity.copy())
    trace["shooter_motion_prior_target_delta"].append(
        np.asarray(shooter.last_motion_prior_target_delta, dtype=np.float64).copy()
    )
    trace["shooter_motion_prior_velocity_delta"].append(
        np.asarray(shooter.last_motion_prior_velocity_delta, dtype=np.float64).copy()
    )
    trace["shooter_motion_prior_position_active"].append(shooter.last_motion_prior_position_active)
    trace["shooter_motion_prior_velocity_active"].append(shooter.last_motion_prior_velocity_active)
    trace["shooter_agility_prior_velocity_delta"].append(
        np.asarray(shooter.last_agility_prior_velocity_delta, dtype=np.float64).copy()
    )
    trace["shooter_agility_prior_target_delta"].append(
        np.asarray(shooter.last_agility_prior_target_delta, dtype=np.float64).copy()
    )
    trace["shooter_agility_prior_active"].append(shooter.last_agility_prior_active)
    trace["shooter_contact_prior_target_delta"].append(
        np.asarray(shooter.last_contact_prior_target_delta, dtype=np.float64).copy()
    )
    trace["shooter_contact_prior_velocity_delta"].append(
        np.asarray(shooter.last_contact_prior_velocity_delta, dtype=np.float64).copy()
    )
    trace["shooter_contact_prior_active"].append(shooter.last_contact_prior_active)
    trace["passer_foot_contact"].append(support[0])
    trace["shooter_foot_contact"].append(support[1])
    trace["shooter_phase_correction"].append(shooter.last_phase_correction)
    transition_features = shooter.last_transition_features
    if transition_features is None:
        raise RuntimeError("shooter transition features were not recorded")
    transition_decision = shooter.transition_decision
    trace["shooter_transition_features"].append(transition_features.copy())
    trace["shooter_transition_actor_accepted"].append(
        False if transition_decision is None else transition_decision.accepted
    )
    trace["shooter_transition_support_distance"].append(
        -1.0 if transition_decision is None else transition_decision.support_distance
    )
    trace["shooter_transition_trigger_policy_frame"].append(
        -1 if transition_decision is None else transition_decision.trigger_policy_frame
    )
    trace["shooter_transition_residual_frames"].append(
        0 if transition_decision is None else transition_decision.residual_frames
    )
    trace["shooter_transition_predicted_safe_probability"].append(
        -1.0
        if transition_decision is None or transition_decision.predicted_safe_probability is None
        else transition_decision.predicted_safe_probability
    )
    trace["shooter_transition_predicted_chain_probability"].append(
        -1.0
        if transition_decision is None or transition_decision.predicted_chain_probability is None
        else transition_decision.predicted_chain_probability
    )
    trace["shooter_transition_ensemble_probability_spread"].append(
        -1.0
        if transition_decision is None or transition_decision.ensemble_probability_spread is None
        else transition_decision.ensemble_probability_spread
    )
    trace["shooter_transition_used_parent_fallback"].append(
        False if transition_decision is None else transition_decision.used_parent_fallback
    )
    trace["shooter_transition_triggered"].append(shooter.transition_triggered)
    option_decision = shooter.last_causal_strike_option_decision
    trace["shooter_causal_strike_option_phase"].append(
        -1 if option_decision is None else int(option_decision.phase)
    )
    trace["shooter_causal_strike_option_ready"].append(
        False if option_decision is None else option_decision.ready
    )
    trace["shooter_causal_strike_option_begin_bridge"].append(
        False if option_decision is None else option_decision.begin_bridge
    )
    trace["shooter_causal_strike_option_incoming_ball"].append(
        False if option_decision is None else option_decision.incoming_ball
    )
    trace["shooter_causal_strike_option_ball_arrival_eta_sec"].append(
        -1.0
        if option_decision is None or option_decision.ball_arrival_eta_sec is None
        else option_decision.ball_arrival_eta_sec
    )
    trace["shooter_causal_strike_option_incoming_observation_count"].append(
        0 if option_decision is None else option_decision.incoming_observation_count
    )
    trace["shooter_causal_strike_selected_phase_start_frame"].append(
        -1
        if shooter.causal_strike_selected_phase_start_frame is None
        else shooter.causal_strike_selected_phase_start_frame
    )
    trace["shooter_causal_strike_bridge_fraction"].append(
        0.0
        if not shooter.causal_strike_bridge_frames
        else min(
            1.0,
            shooter.causal_strike_bridge_frame / shooter.causal_strike_bridge_frames,
        )
    )
    trace["shooter_ball_local_position"].append(
        np.asarray(shooter.state.ball_pos_w, dtype=np.float64).copy()
    )
    trace["shooter_ball_local_velocity"].append(
        np.asarray(shooter.state.ball_vel_w, dtype=np.float64).copy()
    )
    trace["shooter_pelvis_local_position"].append(
        np.asarray(shooter.state.pelvis_pos_w, dtype=np.float64).copy()
    )
    runtime_decision = shooter.runtime_strike_route_decision
    trace["shooter_runtime_strike_features"].append(
        np.zeros(7, dtype=np.float64)
        if shooter.last_runtime_strike_features is None
        else shooter.last_runtime_strike_features.copy()
    )
    trace["shooter_runtime_strike_route_decided"].append(runtime_decision is not None)
    trace["shooter_runtime_strike_route_accepted"].append(
        False if runtime_decision is None else runtime_decision.accepted
    )
    trace["shooter_runtime_strike_route_support_distance"].append(
        -1.0 if runtime_decision is None else runtime_decision.nearest_success_distance
    )
    trace["shooter_runtime_strike_route_advance_frames"].append(
        -1
        if runtime_decision is None or runtime_decision.action is None
        else runtime_decision.action.maximum_arrival_advance_frames
    )
    receive_decision = shooter.runtime_receive_decision
    receive_action = None if receive_decision is None else receive_decision.action
    trace["shooter_runtime_receive_features"].append(
        np.zeros(9, dtype=np.float64)
        if shooter.last_runtime_receive_features is None
        else shooter.last_runtime_receive_features.copy()
    )
    trace["shooter_runtime_receive_decided"].append(receive_decision is not None)
    trace["shooter_runtime_receive_accepted"].append(
        False if receive_decision is None else receive_decision.accepted
    )
    trace["shooter_runtime_receive_support_distance"].append(
        -1.0 if receive_decision is None else receive_decision.nearest_success_distance
    )
    trace["shooter_runtime_receive_alignment_tolerance_sec"].append(
        -1.0 if receive_action is None else receive_action.arrival_alignment_tolerance_sec
    )
    trace["shooter_runtime_receive_stance_offset_y_m"].append(
        0.0 if receive_action is None else receive_action.stance_offset_y_m
    )
    trace["shooter_runtime_receive_foot_yaw_offset_rad"].append(
        0.0 if receive_action is None else receive_action.foot_yaw_offset_rad
    )
    target_decision = shooter.runtime_contact_target_decision
    target_action = None if target_decision is None else target_decision.action
    trace["shooter_runtime_contact_target_decided"].append(target_decision is not None)
    trace["shooter_runtime_contact_target_accepted"].append(
        False if target_decision is None else target_decision.accepted
    )
    trace["shooter_runtime_contact_target_support_distance"].append(
        -1.0 if target_decision is None else target_decision.nearest_success_distance
    )
    trace["shooter_runtime_contact_target_velocity_xyz_mps"].append(
        np.zeros(3, dtype=np.float64)
        if target_action is None
        else np.asarray(target_action.target_foot_velocity_xyz_mps, dtype=np.float64)
    )
    trace["shooter_learned_torque_active"].append(learned_torque["shooter"] is not None)
    trace["shooter_ball_contact_foot"].append(shooter_contact_foot)
    trace["ball_contact_role"].append(contact_role)
    trace["robot_robot_contact_count"].append(robot_contacts)


def _recovery_actor_state(
    robot: _Robot,
    data: Any,
    *,
    ball_body: int,
    timestamp_sec: float,
) -> np.ndarray:
    """Recreate the frozen 74-D recovery feature contract during simulation."""

    from rosclaw.growth.recovery_dataset import STATE_FEATURES

    pelvis = np.asarray(data.qpos[robot.qpos_base : robot.qpos_base + 3], dtype=np.float64)
    pelvis_velocity = np.asarray(data.qvel[robot.qvel_base : robot.qvel_base + 3], dtype=np.float64)
    roll, pitch = _roll_pitch(np.asarray(data.xquat[robot.torso_body], dtype=np.float64))
    com = np.asarray(data.subtree_com[robot.pelvis_body], dtype=np.float64)
    left_foot = np.asarray(data.xpos[robot.left_ankle_body], dtype=np.float64)
    right_foot = np.asarray(data.xpos[robot.right_ankle_body], dtype=np.float64)
    support_count = int(robot.latest_left_support) + int(robot.latest_right_support)
    support_y = (
        (
            left_foot[1] * int(robot.latest_left_support)
            + right_foot[1] * int(robot.latest_right_support)
        )
        / support_count
        if support_count
        else 0.0
    )
    state = np.concatenate(
        (
            np.asarray(data.qpos[robot.joint_qpos], dtype=np.float64),
            np.asarray(data.qvel[robot.joint_qvel], dtype=np.float64),
            np.asarray((pelvis[2],), dtype=np.float64),
            pelvis_velocity,
            np.asarray((roll, pitch, com[1] - support_y), dtype=np.float64),
            np.asarray(
                (float(robot.latest_left_support), float(robot.latest_right_support)),
                dtype=np.float64,
            ),
            np.asarray(data.xpos[ball_body], dtype=np.float64) - pelvis,
            np.asarray(robot.state.ball_vel_w, dtype=np.float64),
            np.asarray(
                (max(0.0, timestamp_sec - float(robot.contact_time or timestamp_sec)),),
                dtype=np.float64,
            ),
        )
    )
    if state.shape != (len(STATE_FEATURES),) or not np.all(np.isfinite(state)):
        raise RuntimeError("live recovery actor state violates the frozen feature contract")
    result: NDArray[np.float64] = np.asarray(state, dtype=np.float64)
    return result


def _update_support_slip(
    robot: _Robot,
    data: Any,
    support: tuple[bool, bool],
) -> None:
    slips: list[float] = []
    for side, active, body in (
        ("left", support[0], robot.left_ankle_body),
        ("right", support[1], robot.right_ankle_body),
    ):
        anchor_name = f"{side}_support_anchor"
        if not active:
            setattr(robot, anchor_name, None)
            continue
        position = np.asarray(data.xpos[body], dtype=np.float64).copy()
        anchor = getattr(robot, anchor_name)
        if anchor is None:
            setattr(robot, anchor_name, position)
            anchor = position
        slips.append(float(np.linalg.norm((position - anchor)[:2])))
    robot.latest_support_slip_m = max(slips, default=0.0)
    robot.peak_support_slip_m = max(
        robot.peak_support_slip_m,
        robot.latest_support_slip_m,
    )
    if robot.contact_latched:
        robot.post_contact_peak_support_slip_m = max(
            robot.post_contact_peak_support_slip_m,
            robot.latest_support_slip_m,
        )


def _reset_post_contact_support_anchors(robot: _Robot) -> None:
    """Start the recovery-slip clock at measured ball contact."""

    robot.left_support_anchor = None
    robot.right_support_anchor = None
    robot.latest_support_slip_m = 0.0


def _tail_wobble(trajectory: dict[str, np.ndarray], role: str) -> float:
    time = trajectory["time"]
    if len(time) < 3:
        return math.inf
    mask = time >= max(0.0, float(time[-1]) - 2.0)
    torso = trajectory[f"{role}_torso_quaternion"][mask]
    tilt = np.asarray([_roll_pitch(quat) for quat in torso], dtype=np.float64)
    velocity = trajectory[f"{role}_joint_velocity"][mask]
    return float(np.mean(np.linalg.norm(np.diff(tilt, axis=0), axis=1))) + float(
        np.mean(np.linalg.norm(velocity[:, :12], axis=1)) * 0.01
    )


def _id(model: Any, object_type: Any, name: str) -> int:
    import mujoco

    value = int(mujoco.mj_name2id(model, object_type, name))
    if value < 0:
        raise ValueError(f"coupled G1 model is missing {name}")
    return value


def _yaw(robot: _Robot) -> float:
    return 2.0 * math.atan2(
        float(robot.world_from_local_quat[3]),
        float(robot.world_from_local_quat[0]),
    )


def _recovery_athlete_authority_envelope(
    command: NDArray[np.float64],
    *,
    depth_error_m: float,
    lateral_position_m: float,
    yaw_error_rad: float,
    config: G1GoalkeeperConfig,
) -> NDArray[np.float64]:
    """Project a learned recovery proposal into a monotone sparse envelope.

    The neural actor still chooses the command magnitude.  This causal safety
    layer removes approximation residue in a qualified deadband, rejects a
    command that would increase the measured error, and caps each component by
    the corresponding continuous recovery field.  It never increases actor
    authority.
    """

    proposal = np.asarray(command, dtype=np.float64)
    state = np.asarray((depth_error_m, lateral_position_m, yaw_error_rad), dtype=np.float64)
    if (
        proposal.shape != (3,)
        or not np.all(np.isfinite(proposal))
        or not np.all(np.isfinite(state))
    ):
        raise ValueError("recovery athlete authority envelope input is invalid")

    def monotone(value: float, error: float, cap: float, *, opposite: bool = False) -> float:
        desired_sign = -math.copysign(1.0, error) if opposite else math.copysign(1.0, error)
        if abs(error) <= 1.0e-12 or value * desired_sign <= 0.0:
            return 0.0
        return desired_sign * min(abs(value), max(0.0, cap))

    lateral_outside = max(
        abs(lateral_position_m) - config.post_contact_ready_lateral_deadband_m,
        0.0,
    )
    projected = np.asarray(
        (
            monotone(
                float(proposal[0]),
                depth_error_m,
                min(
                    config.maximum_depth_correction_mps,
                    config.depth_position_gain * abs(depth_error_m),
                ),
            ),
            monotone(
                float(proposal[1]),
                lateral_position_m,
                min(
                    config.post_contact_ready_maximum_lateral_speed_mps,
                    config.post_contact_ready_lateral_position_gain * lateral_outside,
                ),
                opposite=True,
            ),
            monotone(
                float(proposal[2]),
                yaw_error_rad,
                min(
                    config.post_contact_ready_maximum_yaw_rate_rad_s,
                    config.post_contact_ready_yaw_gain * abs(yaw_error_rad),
                ),
            ),
        ),
        dtype=np.float64,
    )
    if np.any(np.abs(projected) > np.abs(proposal) + 1.0e-12):
        raise RuntimeError("recovery athlete authority envelope increased authority")
    return projected


def _goalkeeper_ready_for_second_threat(robot: _Robot, *, data: Any) -> bool:
    """Return a causal live-state rearm decision for a new threat epoch."""

    pelvis_height = float(data.qpos[robot.qpos_base + 2])
    velocity = np.asarray(data.qvel[robot.qvel_base : robot.qvel_base + 6], dtype=np.float64)
    torso = np.asarray(data.xquat[robot.torso_body], dtype=np.float64)
    if (
        velocity.shape != (6,)
        or torso.shape != (4,)
        or not np.all(np.isfinite(np.concatenate((velocity, torso))))
    ):
        return False
    upright = 1.0 - 2.0 * (torso[1] ** 2 + torso[2] ** 2)
    return bool(
        pelvis_height >= 0.70
        and upright >= 0.90
        and float(np.linalg.norm(velocity[:3])) <= 0.25
        and float(np.linalg.norm(velocity[3:])) <= 0.50
        and robot.latest_left_support
        and robot.latest_right_support
    )


def _recovery_athlete_world_command(
    robot: _Robot,
    *,
    data: Any,
    desired_depth_m: float,
    yaw_error_rad: float,
    elapsed_since_contact_sec: float,
    torch: Any,
    model: Any,
    checkpoint: dict[str, Any],
) -> NDArray[np.float64]:
    """Decode one bounded high-level command from live proprioception."""

    from rosclaw_soccer.training.recovery_athlete_student import (
        decode_recovery_athlete_command,
        recovery_athlete_features_numpy,
    )

    pelvis = np.asarray(data.qpos[robot.qpos_base : robot.qpos_base + 7], dtype=np.float64)
    velocity = np.asarray(data.qvel[robot.qvel_base : robot.qvel_base + 6], dtype=np.float64)
    torso = np.asarray(data.xquat[robot.torso_body], dtype=np.float64)
    upright = 1.0 - 2.0 * (torso[1] ** 2 + torso[2] ** 2)
    features = recovery_athlete_features_numpy(
        depth_error_m=np.asarray((desired_depth_m - pelvis[0],), dtype=np.float64),
        lateral_position_m=np.asarray((pelvis[1],), dtype=np.float64),
        yaw_error_rad=np.asarray((yaw_error_rad,), dtype=np.float64),
        root_velocity=velocity.reshape(1, 6),
        pelvis_height_m=np.asarray((pelvis[2],), dtype=np.float64),
        upright_projection=np.asarray((upright,), dtype=np.float64),
        foot_contact=np.asarray(
            ((robot.latest_left_support, robot.latest_right_support),),
            dtype=np.bool_,
        ),
        elapsed_since_contact_sec=np.asarray((elapsed_since_contact_sec,), dtype=np.float64),
    )
    with torch.inference_mode():
        normalized = (
            decode_recovery_athlete_command(
                torch=torch,
                model=model,
                features=torch.as_tensor(features, dtype=torch.float32),
            )
            .detach()
            .cpu()
            .numpy()[0]
        )
    output_scale = np.asarray(checkpoint.get("output_scale"), dtype=np.float64)
    command = np.asarray(normalized, dtype=np.float64) * output_scale
    if command.shape != (3,) or not np.all(np.isfinite(command)):
        raise RuntimeError("goalkeeper recovery athlete produced an invalid command")
    return command


def _current_root_yaw(robot: _Robot, *, data: Any) -> float:
    """Return measured floating-base yaw instead of the immutable spawn yaw."""

    quaternion = np.asarray(
        data.qpos[robot.qpos_base + 3 : robot.qpos_base + 7],
        dtype=np.float64,
    )
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise RuntimeError("goalkeeper root quaternion must be a finite wxyz vector")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-8:
        raise RuntimeError("goalkeeper root quaternion must be non-degenerate")
    w, x, y, z = quaternion / norm
    return float(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _rotate_z(vector: np.ndarray, yaw: float) -> np.ndarray:
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    x, y, z = np.asarray(vector, dtype=np.float64)
    result: NDArray[np.float64] = np.asarray(
        (cosine * x - sine * y, sine * x + cosine * y, z),
        dtype=np.float64,
    )
    return result


def _to_local(position: np.ndarray, robot: _Robot) -> np.ndarray:
    return _rotate_z(np.asarray(position, dtype=np.float64) - robot.origin, -_yaw(robot))
