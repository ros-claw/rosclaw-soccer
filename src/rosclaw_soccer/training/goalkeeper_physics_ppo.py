"""Synchronized multi-GPU PPO over the MJWarp G1 goalkeeper task."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from rosclaw_soccer.sim.contracts import (
    hash_bytes,
    hash_json,
    sanitize_nonfinite_evidence,
)
from rosclaw_soccer.training.goalkeeper_mjwarp import (
    GoalkeeperMJWarpBatch,
    goalkeeper_world_config,
)
from rosclaw_soccer.training.goalkeeper_mobility_option import (
    GoalkeeperMobilityOptionConfig,
)
from rosclaw_soccer.training.goalkeeper_targeted_dive import (
    _MIRROR_ORDER,
    _MIRROR_SIGN,
)

_CONTROLLER_SEMANTIC_FIELDS = (
    "task_space_reach_blend",
    "task_space_reach_atlas_enabled",
    "runtime_task_space_reach_enabled",
    "runtime_task_space_reach_blend",
    "shot_intent_cue_enabled",
    "targeted_dive_checkpoint",
    "targeted_dive_option_duration_sec",
    "targeted_dive_phase_hold_sec",
    "targeted_dive_prediction_lead_sec",
    "targeted_dive_nominal_shot_flight_time_sec",
    "targeted_dive_intercept_phase_at_arrival",
    "targeted_dive_posture_exception_duration_sec",
    "targeted_dive_root_angular_speed_guard_ceiling_rad_s",
    "targeted_dive_decoder_residual_authority",
    "targeted_dive_decoder_lower_body_residual_authority",
    "targeted_dive_decoder_lower_body_command_scale",
    "targeted_dive_decoder_waist_residual_authority",
    "targeted_dive_decoder_arm_residual_authority",
    "targeted_dive_actor_residual_scale",
    "targeted_dive_actor_recovery_plasticity_sec",
    "targeted_dive_actor_recovery_residual_authority_scale",
    "targeted_dive_post_save_counterstep_enabled",
    "targeted_dive_post_save_counterstep_duration_sec",
    "targeted_dive_post_save_counterstep_command_limit",
    "targeted_dive_post_save_counterstep_capture_horizon_sec",
    "targeted_dive_post_save_counterstep_recenter_weight",
    "targeted_dive_post_save_option_release_sec",
    "targeted_dive_post_save_fall_recovery_enabled",
    "targeted_dive_post_save_fall_recovery_duration_sec",
    "targeted_dive_post_save_fall_minimum_pelvis_height_m",
    "targeted_dive_post_save_fall_minimum_upright_projection",
    "targeted_dive_post_save_fall_maximum_root_linear_speed_mps",
    "targeted_dive_post_save_fall_maximum_root_angular_speed_rad_s",
    "targeted_dive_mosaic_gmt_model",
    "targeted_dive_mosaic_gmt_getup_skill",
    "targeted_dive_mosaic_gmt_getup_blend",
    "targeted_dive_mosaic_gmt_getup_activation_maximum_pelvis_height_m",
    "targeted_dive_mosaic_gmt_getup_blend_in_sec",
    "targeted_dive_mosaic_gmt_getup_reference_feedforward_blend",
    "targeted_dive_mosaic_gmt_getup_lower_body_scale",
    "targeted_dive_mosaic_gmt_getup_waist_scale",
    "targeted_dive_mosaic_gmt_getup_arm_scale",
    "targeted_dive_anchor_lower_body_scale",
    "targeted_dive_anchor_waist_scale",
    "targeted_dive_anchor_arm_scale",
    "targeted_dive_minimum_option_gate",
    "targeted_dive_runtime_reach_blend",
    "targeted_dive_runtime_reach_feedback_blend",
    "targeted_dive_runtime_reach_feedback_gain",
    "targeted_dive_runtime_reach_feedback_maximum_error_m",
    "targeted_dive_runtime_reach_feedback_support_scale",
    "targeted_dive_runtime_contact_support_side_enabled",
    "targeted_dive_actor_contact_support_side_enabled",
    "targeted_dive_actor_recovery_context_enabled",
    "targeted_dive_runtime_whole_body_reach_blend",
    "targeted_dive_lateral_drive_scale",
    "targeted_dive_negative_target_lateral_drive_scale",
    "targeted_dive_lateral_drive_full_activation_gate",
    "targeted_dive_lateral_drive_capture_enabled",
    "targeted_dive_lateral_drive_capture_horizon_sec",
    "targeted_dive_lateral_drive_target_standoff_m",
    "targeted_dive_lateral_drive_capture_scale_m",
    "targeted_dive_lateral_drive_learned_gate_enabled",
    "targeted_dive_canonical_locomotion_mirror_enabled",
    "targeted_dive_runtime_lateral_lunge_blend",
    "targeted_dive_runtime_lateral_lunge_hip_roll_rad",
    "targeted_dive_runtime_lateral_lunge_ankle_roll_rad",
    "targeted_dive_runtime_lateral_lunge_approach_horizon_sec",
    "targeted_dive_substep_upper_body_guard_enabled",
    "targeted_dive_substep_upper_body_guard_onset_rad_s",
    "targeted_dive_substep_upper_body_guard_ceiling_rad_s",
    "targeted_dive_substep_upper_body_minimum_position_scale",
    "targeted_dive_substep_option_lower_body_guard_enabled",
    "targeted_dive_substep_option_lower_body_guard_onset_rad_s",
    "targeted_dive_substep_option_lower_body_guard_ceiling_rad_s",
    "targeted_dive_substep_option_lower_body_minimum_scale",
    "targeted_dive_low_shot_phase_scale",
    "targeted_dive_mid_shot_phase_scale",
    "targeted_dive_high_shot_phase_scale",
    "targeted_dive_maximum_arm_target_step_rad",
    "targeted_dive_arm_target_filter_fraction",
    "targeted_dive_maximum_lower_body_target_step_rad",
    "targeted_dive_lower_body_target_filter_fraction",
    "combat_teacher_checkout",
    "combat_teacher_checkpoint",
    "maximum_combat_teacher_blend",
    "combat_teacher_intercept_conditioning_enabled",
    "mobility_option_enabled",
    "mobility_lateral_command_limit",
    "mobility_recovery_command_limit",
    "mobility_residual_plasticity_scale",
    "mobility_waist_residual_plasticity_scale",
    "mobility_arm_residual_plasticity_scale",
    "mobility_teacher_lower_body_scale",
    "mobility_teacher_waist_scale",
    "mobility_teacher_arm_scale",
    "mobility_predictive_teacher_gate_floor",
    "mobility_teacher_lower_body_target_step_rad",
    "mobility_teacher_lower_body_target_filter_fraction",
    "mobility_teacher_waist_target_step_rad",
    "mobility_teacher_waist_target_filter_fraction",
    "mobility_teacher_arm_target_step_rad",
    "mobility_teacher_arm_target_filter_fraction",
    "mobility_counter_rotation_enabled",
    "mobility_anticipatory_arm_reach_enabled",
    "mobility_predictive_teacher_warmstart_enabled",
    "mobility_teacher_recovery_latch_enabled",
    "mobility_lateral_velocity_guard_enabled",
    "mobility_substep_upper_body_guard_enabled",
)


@dataclass(frozen=True)
class GoalkeeperPhysicsPPOConfig:
    environments_per_rank: int = 64
    iterations: int = 8
    deterministic_selection_seed_count: int = 1
    deterministic_selection_seed_stride: int = 1009
    rollout_steps: int = 250
    training_episode_duration_sec: float | None = None
    update_epochs: int = 4
    minibatches: int = 8
    hidden_size: int = 128
    learning_rate: float = 2.5e-4
    discount: float = 0.995
    gae_lambda: float = 0.95
    clip_ratio: float = 0.20
    entropy_coefficient: float = 0.001
    value_coefficient: float = 0.50
    policy_anchor_coefficient: float = 0.010
    online_parent_distillation_coefficient: float = 0.0
    successful_trajectory_replay_coefficient: float = 0.0
    successful_trajectory_mirror_augmentation_enabled: bool = False
    successful_trajectory_recovery_only_enabled: bool = False
    symmetry_mirror_loss_coefficient: float = 0.0
    recovery_transition_policy_weight: float = 1.0
    support_landing_causal_recovery_only_enabled: bool = False
    rollback_to_exploration_champion_on_regression_enabled: bool = False
    save_first_exploration_selection_enabled: bool = False
    successful_trajectory_memory_capacity_per_stratum: int = 0
    successful_trajectory_memory_batch_size: int = 256
    successful_trajectory_memory_minimum_episodes_per_stratum: int = 1
    successful_trajectory_memory_full_strength_episodes_per_stratum: int = 0
    successful_trajectory_action_innovation_scale: float = 1.0
    arm_only_online_update: bool = False
    lower_body_and_arms_online_update: bool = False
    lateral_drive_gate_only_online_update: bool = False
    support_landing_online_update: bool = False
    first_save_selection_weight: float = 100.0
    qualified_first_save_selection_weight: float = 0.0
    first_hand_save_selection_weight: float = 40.0
    second_attempt_save_selection_weight: float = 100.0
    second_attempt_hand_save_selection_weight: float = 120.0
    second_save_selection_weight: float = 100.0
    second_hand_save_selection_weight: float = 120.0
    hand_reach_selection_weight: float = 15.0
    hand_target_distance_selection_weight: float = 0.0
    first_save_stratum_balance_selection_weight: float = 0.0
    first_save_side_balance_selection_weight: float = 0.0
    second_release_recenter_selection_weight: float = 0.0
    minimum_first_save_rate_for_selection: float = 0.0
    minimum_first_save_stratum_rate_for_selection: float = 0.0
    minimum_hand_displacement_for_selection_m: float = 0.0
    root_angular_speed_selection_penalty_weight: float = 0.0
    maximum_exploration_root_angular_speed_for_selection: float = 3.50
    teacher_pretraining_enabled: bool = True
    teacher_pretraining_samples: int = 32_768
    teacher_pretraining_epochs: int = 20
    teacher_parent_replay_coefficient: float = 0.50
    task_space_reach_blend: float = 0.0
    task_space_reach_atlas_enabled: bool = False
    runtime_task_space_reach_enabled: bool = False
    runtime_task_space_reach_blend: float = 0.0
    second_shot_reach_multiplier: float = 1.0
    training_second_shot_probability: float = 0.75
    training_first_shot_release_sec: float | None = None
    training_hard_shot_fraction: float = 0.0
    training_hard_shot_height_mode: str = "high"
    training_hard_shot_side_mode: str = "balanced"
    training_hard_shot_flight_time_range_sec: tuple[float, float] | None = None
    training_reach_reward_semantics: str = "STATE_DENSITY"
    training_hard_height_reach_reward_scale: float = 0.0
    training_hard_height_reach_threshold_m: float = 1.10
    training_hard_height_reach_distance_decay: float = 1.25
    training_task_motion_reward_scale: float = 0.0
    training_recovery_progress_reward_scale: float = 0.0
    training_recovery_progress_linear_speed_decay: float = 2.0
    training_recovery_progress_angular_speed_decay: float = 0.50
    training_true_save_bonus: float = 25.0
    training_hand_save_bonus: float | None = None
    training_recovery_event_bonus: float = 15.0
    training_root_angular_speed_penalty_scale: float | None = None
    training_root_angular_speed_soft_limit_rad_s: float = 3.50
    training_root_angular_speed_excess_penalty_scale: float = 0.0
    training_flight_root_angular_penalty_scale: float = 1.0
    training_unsafe_penalty: float = 50.0
    training_save_then_unsafe_penalty: float = 0.0
    shot_difficulty_profile: str = "standard"
    maximum_gradient_norm: float = 0.80
    random_seed: int = 1207
    initialization_checkpoint: str | None = None
    combat_teacher_checkout: str | None = None
    combat_teacher_checkpoint: str | None = None
    targeted_dive_checkpoint: str | None = None
    targeted_dive_option_duration_sec: float = 0.90
    targeted_dive_phase_hold_sec: float = 0.0
    targeted_dive_actor_recovery_plasticity_sec: float = 0.0
    targeted_dive_actor_recovery_residual_authority_scale: float = 0.50
    targeted_dive_post_save_counterstep_enabled: bool = False
    targeted_dive_post_save_counterstep_duration_sec: float = 0.80
    targeted_dive_post_save_counterstep_command_limit: float = 0.55
    targeted_dive_post_save_counterstep_capture_horizon_sec: float = 0.28
    targeted_dive_post_save_counterstep_recenter_weight: float = 1.0
    targeted_dive_post_save_option_release_sec: float = 0.30
    targeted_dive_post_save_fall_recovery_enabled: bool = False
    targeted_dive_post_save_fall_recovery_duration_sec: float = 1.50
    targeted_dive_post_save_fall_minimum_pelvis_height_m: float = 0.12
    targeted_dive_post_save_fall_minimum_upright_projection: float = -0.95
    targeted_dive_post_save_fall_maximum_root_linear_speed_mps: float = 3.50
    targeted_dive_post_save_fall_maximum_root_angular_speed_rad_s: float = 10.0
    targeted_dive_prediction_lead_sec: float = 0.30
    targeted_dive_nominal_shot_flight_time_sec: float = 0.47
    targeted_dive_intercept_phase_at_arrival: float | None = None
    targeted_dive_phase_sync_minimum_target_height_m: float = 0.60
    targeted_dive_posture_exception_duration_sec: float = 1.55
    targeted_dive_root_angular_speed_guard_ceiling_rad_s: float = 8.0
    targeted_dive_decoder_residual_authority: float = 0.10
    targeted_dive_decoder_lower_body_residual_authority: float | None = None
    targeted_dive_decoder_lower_body_command_scale: float | None = None
    targeted_dive_decoder_waist_residual_authority: float | None = None
    targeted_dive_decoder_arm_residual_authority: float | None = None
    targeted_dive_actor_residual_scale: float = 0.70
    targeted_dive_anchor_lower_body_scale: float = 0.25
    targeted_dive_anchor_waist_scale: float = 0.50
    targeted_dive_anchor_arm_scale: float = 1.00
    targeted_dive_minimum_option_gate: float = 0.0
    targeted_dive_runtime_reach_blend: float = 0.0
    targeted_dive_runtime_reach_feedback_blend: float = 0.0
    targeted_dive_runtime_reach_feedback_gain: float = 0.70
    targeted_dive_runtime_reach_feedback_maximum_error_m: float = 0.30
    targeted_dive_runtime_reach_feedback_support_scale: float = 0.0
    targeted_dive_runtime_contact_support_side_enabled: bool = False
    targeted_dive_actor_contact_support_side_enabled: bool = False
    targeted_dive_actor_recovery_context_enabled: bool = False
    targeted_dive_runtime_whole_body_reach_blend: float = 0.0
    targeted_dive_runtime_whole_body_reach_full_below_height_m: float = 0.50
    targeted_dive_runtime_whole_body_reach_maximum_height_m: float = 0.65
    targeted_dive_runtime_whole_body_reach_waist_scale: float = 0.75
    targeted_dive_runtime_whole_body_reach_arm_scale: float = 1.0
    targeted_dive_runtime_whole_body_reach_support_scale: float = 0.65
    targeted_dive_runtime_whole_body_reach_release_sec: float = 0.60
    targeted_dive_runtime_reach_contact_standoff_m: float = 0.0
    targeted_dive_runtime_reach_lateral_lead_m: float = 0.0
    targeted_dive_runtime_reach_vertical_lead_m: float = 0.0
    targeted_dive_runtime_reach_low_vertical_lead_m: float | None = None
    targeted_dive_runtime_reach_mid_vertical_lead_m: float | None = None
    targeted_dive_runtime_reach_high_vertical_lead_m: float | None = None
    targeted_dive_overhead_reach_prior: str | None = None
    targeted_dive_overhead_reach_blend: float = 0.0
    targeted_dive_overhead_reach_minimum_target_height_m: float = 1.10
    targeted_dive_overhead_reach_full_target_height_m: float = 1.25
    targeted_dive_overhead_reach_lower_body_scale: float = 0.0
    targeted_dive_overhead_reach_waist_scale: float = 0.25
    targeted_dive_overhead_reach_arm_scale: float = 1.0
    targeted_dive_mosaic_gmt_model: str | None = None
    targeted_dive_mosaic_gmt_skill: str | None = None
    targeted_dive_mosaic_gmt_blend: float = 0.0
    targeted_dive_mosaic_gmt_stability_floor: float = 0.0
    targeted_dive_mosaic_gmt_minimum_target_height_m: float = 1.10
    targeted_dive_mosaic_gmt_full_target_height_m: float = 1.25
    targeted_dive_mosaic_gmt_lower_body_scale: float = 1.0
    targeted_dive_mosaic_gmt_waist_scale: float = 1.0
    targeted_dive_mosaic_gmt_arm_scale: float = 0.80
    targeted_dive_mosaic_gmt_getup_skill: str | None = None
    targeted_dive_mosaic_gmt_getup_blend: float = 0.0
    targeted_dive_mosaic_gmt_getup_activation_maximum_pelvis_height_m: float = 0.50
    targeted_dive_mosaic_gmt_getup_blend_in_sec: float = 0.35
    targeted_dive_mosaic_gmt_getup_reference_feedforward_blend: float = 0.0
    targeted_dive_mosaic_gmt_getup_lower_body_scale: float = 1.0
    targeted_dive_mosaic_gmt_getup_waist_scale: float = 1.0
    targeted_dive_mosaic_gmt_getup_arm_scale: float = 1.0
    targeted_dive_maximum_arm_target_step_rad: float = 0.10
    targeted_dive_arm_target_filter_fraction: float = 0.50
    targeted_dive_maximum_lower_body_target_step_rad: float = 0.08
    targeted_dive_lower_body_target_filter_fraction: float = 0.35
    targeted_dive_lateral_drive_scale: float = 0.0
    targeted_dive_negative_target_lateral_drive_scale: float = 1.0
    targeted_dive_lateral_drive_full_activation_gate: float = 0.30
    targeted_dive_lateral_drive_capture_enabled: bool = False
    targeted_dive_lateral_drive_capture_horizon_sec: float = 0.35
    targeted_dive_lateral_drive_target_standoff_m: float = 0.32
    targeted_dive_lateral_drive_capture_scale_m: float = 0.45
    targeted_dive_lateral_drive_learned_gate_enabled: bool = False
    targeted_dive_canonical_locomotion_mirror_enabled: bool = False
    targeted_dive_runtime_lateral_lunge_blend: float = 0.0
    targeted_dive_runtime_lateral_lunge_hip_roll_rad: float = 0.18
    targeted_dive_runtime_lateral_lunge_ankle_roll_rad: float = 0.12
    targeted_dive_runtime_lateral_lunge_approach_horizon_sec: float = 0.90
    targeted_dive_substep_upper_body_guard_enabled: bool = False
    targeted_dive_substep_upper_body_guard_onset_rad_s: float = 1.80
    targeted_dive_substep_upper_body_guard_ceiling_rad_s: float = 3.00
    targeted_dive_substep_upper_body_minimum_position_scale: float = 0.05
    targeted_dive_substep_option_lower_body_guard_enabled: bool = False
    targeted_dive_substep_option_lower_body_guard_onset_rad_s: float = 2.40
    targeted_dive_substep_option_lower_body_guard_ceiling_rad_s: float = 3.30
    targeted_dive_substep_option_lower_body_minimum_scale: float = 0.0
    targeted_dive_low_shot_phase_scale: float = 1.0
    targeted_dive_mid_shot_phase_scale: float = 1.0
    targeted_dive_high_shot_phase_scale: float = 1.0
    targeted_dive_initial_gate: float = 0.40
    maximum_combat_teacher_blend: float = 0.35
    combat_teacher_intercept_conditioning_enabled: bool = False
    mobility_option_enabled: bool = False
    mobility_lateral_command_limit: float = 0.75
    mobility_recovery_command_limit: float = 0.55
    mobility_residual_plasticity_scale: float = 0.0
    mobility_waist_residual_plasticity_scale: float | None = None
    mobility_arm_residual_plasticity_scale: float | None = None
    mobility_teacher_lower_body_scale: float = 0.25
    mobility_teacher_waist_scale: float = 0.80
    mobility_teacher_arm_scale: float = 1.00
    mobility_predictive_teacher_gate_floor: float = 0.0
    mobility_teacher_lower_body_target_step_rad: float = 0.08
    mobility_teacher_lower_body_target_filter_fraction: float = 0.35
    mobility_teacher_waist_target_step_rad: float = 0.05
    mobility_teacher_waist_target_filter_fraction: float = 0.25
    mobility_teacher_arm_target_step_rad: float = 0.045
    mobility_teacher_arm_target_filter_fraction: float = 0.15
    mobility_counter_rotation_enabled: bool = False
    mobility_anticipatory_arm_reach_enabled: bool = False
    mobility_predictive_teacher_warmstart_enabled: bool = False
    mobility_teacher_recovery_latch_enabled: bool = False
    mobility_teacher_recovery_hold_sec: float = 0.24
    mobility_teacher_recovery_decay_sec: float = 0.60
    mobility_lateral_velocity_guard_enabled: bool = False
    mobility_substep_upper_body_guard_enabled: bool = False
    mobility_substep_upper_body_guard_onset_rad_s: float = 1.80
    mobility_substep_upper_body_guard_ceiling_rad_s: float = 2.80
    mobility_substep_upper_body_minimum_position_scale: float = 0.05
    shot_intent_cue_enabled: bool = False
    combat_gate_pretraining_batches: int = 0
    combat_gate_pretraining_epochs: int = 8
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.goalkeeper_physics_ppo_config.v103"

    def __post_init__(self) -> None:
        if not 4 <= self.environments_per_rank <= 4096:
            raise ValueError("physics PPO environments_per_rank must be in [4, 4096]")
        if not 1 <= self.iterations <= 20_000 or not 16 <= self.rollout_steps <= 1000:
            raise ValueError("physics PPO rollout dimensions are invalid")
        if self.training_episode_duration_sec is not None and (
            not math.isfinite(self.training_episode_duration_sec)
            or not 5.0 <= self.training_episode_duration_sec <= 10.0
        ):
            raise ValueError("physics PPO episode-duration override is invalid")
        if not 1 <= self.deterministic_selection_seed_count <= 8:
            raise ValueError("physics PPO deterministic selection seed count is invalid")
        if not 1 <= self.deterministic_selection_seed_stride <= 1_000_000:
            raise ValueError("physics PPO deterministic selection seed stride is invalid")
        if not 1 <= self.update_epochs <= 16 or not 1 <= self.minibatches <= 64:
            raise ValueError("physics PPO update dimensions are invalid")
        if not 32 <= self.hidden_size <= 512:
            raise ValueError("physics PPO hidden size must be in [32, 512]")
        if not isinstance(self.teacher_pretraining_enabled, bool):
            raise ValueError("physics PPO teacher pretraining flag must be boolean")
        if not math.isfinite(self.successful_trajectory_replay_coefficient) or not (
            0.0 <= self.successful_trajectory_replay_coefficient <= 10.0
        ):
            raise ValueError("physics PPO successful-trajectory replay is invalid")
        if not math.isfinite(self.symmetry_mirror_loss_coefficient) or not (
            0.0 <= self.symmetry_mirror_loss_coefficient <= 10.0
        ):
            raise ValueError("physics PPO symmetry mirror loss is invalid")
        if not isinstance(self.successful_trajectory_mirror_augmentation_enabled, bool):
            raise ValueError("physics PPO successful-trajectory mirror flag must be boolean")
        if not isinstance(self.successful_trajectory_recovery_only_enabled, bool):
            raise ValueError("physics PPO recovery-only replay flag must be boolean")
        if not isinstance(self.support_landing_causal_recovery_only_enabled, bool):
            raise ValueError("physics PPO causal support-landing flag must be boolean")
        if not isinstance(self.rollback_to_exploration_champion_on_regression_enabled, bool):
            raise ValueError("physics PPO exploration-champion rollback flag must be boolean")
        if not isinstance(self.save_first_exploration_selection_enabled, bool):
            raise ValueError("physics PPO save-first exploration flag must be boolean")
        if not math.isfinite(self.recovery_transition_policy_weight) or not (
            1.0 <= self.recovery_transition_policy_weight <= 20.0
        ):
            raise ValueError("physics PPO recovery-transition policy weight is invalid")
        if self.successful_trajectory_mirror_augmentation_enabled and (
            self.successful_trajectory_memory_capacity_per_stratum <= 0
            or self.successful_trajectory_replay_coefficient <= 0.0
            or self.targeted_dive_checkpoint is None
        ):
            raise ValueError(
                "physics PPO successful-trajectory mirror requires targeted-dive memory replay"
            )
        if self.symmetry_mirror_loss_coefficient > 0.0 and self.targeted_dive_checkpoint is None:
            raise ValueError("physics PPO symmetry mirror requires targeted-dive actor")
        if self.successful_trajectory_recovery_only_enabled and (
            self.successful_trajectory_memory_capacity_per_stratum <= 0
            or self.successful_trajectory_replay_coefficient <= 0.0
            or not self.targeted_dive_actor_recovery_context_enabled
        ):
            raise ValueError("physics PPO recovery-only replay requires causal recovery memory")
        if self.recovery_transition_policy_weight > 1.0 and (
            not self.support_landing_online_update
            or not self.targeted_dive_actor_recovery_context_enabled
        ):
            raise ValueError(
                "physics PPO recovery weighting requires support-landing recovery context"
            )
        if not 0 <= self.successful_trajectory_memory_capacity_per_stratum <= 1_000_000:
            raise ValueError("physics PPO successful-trajectory memory capacity is invalid")
        if not 1 <= self.successful_trajectory_memory_batch_size <= 8192:
            raise ValueError("physics PPO successful-trajectory memory batch size is invalid")
        if not (1 <= self.successful_trajectory_memory_minimum_episodes_per_stratum <= 16_384):
            raise ValueError(
                "physics PPO successful-trajectory memory episode diversity is invalid"
            )
        if not (
            0 <= self.successful_trajectory_memory_full_strength_episodes_per_stratum <= 16_384
        ):
            raise ValueError(
                "physics PPO successful-trajectory memory full-strength episode count is invalid"
            )
        if not math.isfinite(self.successful_trajectory_action_innovation_scale) or not (
            0.0 <= self.successful_trajectory_action_innovation_scale <= 1.0
        ):
            raise ValueError("physics PPO successful-trajectory action innovation scale is invalid")
        if (
            self.successful_trajectory_memory_capacity_per_stratum > 0
            and self.successful_trajectory_replay_coefficient <= 0.0
        ):
            raise ValueError("physics PPO successful-trajectory memory requires replay")
        if not isinstance(self.arm_only_online_update, bool):
            raise ValueError("physics PPO arm-only update flag must be boolean")
        if not isinstance(self.lower_body_and_arms_online_update, bool):
            raise ValueError("physics PPO lower-body-and-arms update flag must be boolean")
        if not isinstance(self.lateral_drive_gate_only_online_update, bool):
            raise ValueError("physics PPO lateral-drive gate-only update flag must be boolean")
        if not isinstance(self.support_landing_online_update, bool):
            raise ValueError("physics PPO support-landing update flag must be boolean")
        if self.support_landing_causal_recovery_only_enabled and not (
            self.support_landing_online_update
        ):
            raise ValueError(
                "physics PPO causal support-landing requires support-landing online update"
            )
        specialist_scopes = (
            self.arm_only_online_update,
            self.lower_body_and_arms_online_update,
            self.lateral_drive_gate_only_online_update,
            self.support_landing_online_update,
        )
        if sum(specialist_scopes) > 1:
            raise ValueError("physics PPO specialist update scopes are mutually exclusive")
        if not isinstance(self.task_space_reach_atlas_enabled, bool):
            raise ValueError("physics PPO task-space reach-atlas flag must be boolean")
        if not isinstance(self.runtime_task_space_reach_enabled, bool):
            raise ValueError("physics PPO runtime reach flag must be boolean")
        if not isinstance(self.mobility_option_enabled, bool):
            raise ValueError("physics PPO mobility-option flag must be boolean")
        if not isinstance(self.targeted_dive_substep_upper_body_guard_enabled, bool):
            raise ValueError("physics PPO targeted-dive substep guard flag must be boolean")
        if not isinstance(self.targeted_dive_runtime_contact_support_side_enabled, bool):
            raise ValueError(
                "physics PPO targeted-dive contact-support proprioception flag must be boolean"
            )
        if not isinstance(self.targeted_dive_actor_contact_support_side_enabled, bool):
            raise ValueError("physics PPO actor contact-support flag must be boolean")
        if not isinstance(self.targeted_dive_actor_recovery_context_enabled, bool):
            raise ValueError("physics PPO actor recovery-context flag must be boolean")
        if not isinstance(self.targeted_dive_canonical_locomotion_mirror_enabled, bool):
            raise ValueError("physics PPO canonical locomotion mirror flag must be boolean")
        if (
            self.targeted_dive_actor_contact_support_side_enabled
            and not self.targeted_dive_runtime_contact_support_side_enabled
        ):
            raise ValueError("physics PPO actor contact support requires runtime contact support")
        if (
            self.targeted_dive_actor_recovery_context_enabled
            and not self.targeted_dive_actor_contact_support_side_enabled
        ):
            raise ValueError("physics PPO actor recovery context requires foot-contact context")
        if not isinstance(self.targeted_dive_substep_option_lower_body_guard_enabled, bool):
            raise ValueError(
                "physics PPO targeted-dive option lower-body guard flag must be boolean"
            )
        if not isinstance(self.combat_teacher_intercept_conditioning_enabled, bool):
            raise ValueError("physics PPO combat-teacher conditioning flag must be boolean")
        if not isinstance(self.mobility_counter_rotation_enabled, bool):
            raise ValueError("physics PPO mobility counter-rotation flag must be boolean")
        if not isinstance(self.mobility_anticipatory_arm_reach_enabled, bool):
            raise ValueError("physics PPO mobility anticipatory-arm flag must be boolean")
        if not isinstance(self.mobility_predictive_teacher_warmstart_enabled, bool):
            raise ValueError("physics PPO mobility predictive-teacher flag must be boolean")
        if not isinstance(self.mobility_teacher_recovery_latch_enabled, bool):
            raise ValueError("physics PPO mobility teacher-recovery latch flag must be boolean")
        if not isinstance(self.mobility_lateral_velocity_guard_enabled, bool):
            raise ValueError("physics PPO mobility lateral-velocity guard flag must be boolean")
        if not isinstance(self.mobility_substep_upper_body_guard_enabled, bool):
            raise ValueError("physics PPO mobility substep upper-body guard flag must be boolean")
        if not isinstance(self.shot_intent_cue_enabled, bool):
            raise ValueError("physics PPO shot-intent cue flag must be boolean")
        if self.training_first_shot_release_sec is not None and (
            not math.isfinite(self.training_first_shot_release_sec)
            or not 0.25 <= self.training_first_shot_release_sec <= 1.50
        ):
            raise ValueError("physics PPO first-shot release override is invalid")
        if self.training_root_angular_speed_penalty_scale is not None and (
            not math.isfinite(self.training_root_angular_speed_penalty_scale)
            or not 0.01 <= self.training_root_angular_speed_penalty_scale <= 1.0
        ):
            raise ValueError("physics PPO root-angular reward penalty is invalid")
        if not (
            math.isfinite(self.training_root_angular_speed_soft_limit_rad_s)
            and 0.50 <= self.training_root_angular_speed_soft_limit_rad_s <= 8.0
            and math.isfinite(self.training_root_angular_speed_excess_penalty_scale)
            and 0.0 <= self.training_root_angular_speed_excess_penalty_scale <= 100.0
            and math.isfinite(self.training_flight_root_angular_penalty_scale)
            and 0.01 <= self.training_flight_root_angular_penalty_scale <= 1.0
        ):
            raise ValueError("physics PPO root-angular tail penalty is invalid")
        if (
            self.targeted_dive_substep_upper_body_guard_enabled
            and self.targeted_dive_checkpoint is None
        ):
            raise ValueError("physics PPO targeted-dive substep guard requires targeted dive")
        if (
            self.targeted_dive_substep_option_lower_body_guard_enabled
            and self.targeted_dive_checkpoint is None
        ):
            raise ValueError(
                "physics PPO targeted-dive option lower-body guard requires targeted dive"
            )
        if not (
            0.50
            <= self.targeted_dive_substep_upper_body_guard_onset_rad_s
            < self.targeted_dive_substep_upper_body_guard_ceiling_rad_s
            <= 8.0
            and 0.0 <= self.targeted_dive_substep_upper_body_minimum_position_scale <= 0.50
        ):
            raise ValueError("physics PPO targeted-dive substep guard settings are invalid")
        if not (
            0.50
            <= self.targeted_dive_substep_option_lower_body_guard_onset_rad_s
            < self.targeted_dive_substep_option_lower_body_guard_ceiling_rad_s
            <= 8.0
            and 0.0 <= self.targeted_dive_substep_option_lower_body_minimum_scale <= 0.50
        ):
            raise ValueError(
                "physics PPO targeted-dive option lower-body guard settings are invalid"
            )
        if self.shot_intent_cue_enabled and not (
            self.mobility_option_enabled or self.targeted_dive_checkpoint is not None
        ):
            raise ValueError(
                "physics PPO shot-intent cue requires mobility or a targeted dive option"
            )
        if (
            not math.isfinite(self.mobility_residual_plasticity_scale)
            or not 0.0 <= self.mobility_residual_plasticity_scale <= 1.0
        ):
            raise ValueError("physics PPO mobility residual plasticity is invalid")
        if not self.mobility_option_enabled and self.mobility_residual_plasticity_scale != 0.0:
            raise ValueError("physics PPO mobility residual plasticity requires mobility option")
        group_plasticity = (
            self.mobility_waist_residual_plasticity_scale,
            self.mobility_arm_residual_plasticity_scale,
        )
        if any(
            value is not None and (not math.isfinite(value) or not 0.0 <= value <= 1.0)
            for value in group_plasticity
        ):
            raise ValueError("physics PPO mobility group residual plasticity is invalid")
        if not self.mobility_option_enabled and any(
            value is not None for value in group_plasticity
        ):
            raise ValueError("physics PPO mobility group plasticity requires mobility option")
        effective_arm_plasticity = (
            self.mobility_residual_plasticity_scale
            if self.mobility_arm_residual_plasticity_scale is None
            else self.mobility_arm_residual_plasticity_scale
        )
        if (
            not math.isfinite(self.mobility_lateral_command_limit)
            or not 0.35 <= self.mobility_lateral_command_limit <= 1.0
            or not math.isfinite(self.mobility_recovery_command_limit)
            or not 0.25
            <= self.mobility_recovery_command_limit
            <= self.mobility_lateral_command_limit
        ):
            raise ValueError("physics PPO mobility command limits are invalid")
        if self.mobility_anticipatory_arm_reach_enabled and (
            not self.mobility_option_enabled
            or not self.shot_intent_cue_enabled
            or effective_arm_plasticity <= 0.0
            or self.task_space_reach_blend <= 0.0
        ):
            raise ValueError(
                "physics PPO anticipatory arms require mobility, intent cue, "
                "task-space reach, and residual plasticity"
            )
        if self.mobility_lateral_velocity_guard_enabled and not self.mobility_option_enabled:
            raise ValueError("physics PPO lateral-velocity guard requires mobility option")
        if self.mobility_substep_upper_body_guard_enabled and not self.mobility_option_enabled:
            raise ValueError("physics PPO substep upper-body guard requires mobility option")
        if not (
            1.0
            <= self.mobility_substep_upper_body_guard_onset_rad_s
            < self.mobility_substep_upper_body_guard_ceiling_rad_s
            <= 3.5
            and 0.02 <= self.mobility_substep_upper_body_minimum_position_scale <= 0.40
        ):
            raise ValueError("physics PPO substep upper-body guard settings are invalid")
        if self.mobility_predictive_teacher_warmstart_enabled and (
            not self.mobility_option_enabled or not self.shot_intent_cue_enabled
        ):
            raise ValueError(
                "physics PPO predictive teacher requires mobility option and shot-intent cue"
            )
        if self.mobility_teacher_recovery_latch_enabled and not self.mobility_option_enabled:
            raise ValueError("physics PPO teacher-recovery latch requires mobility option")
        if (
            not math.isfinite(self.mobility_teacher_recovery_hold_sec)
            or not 0.10 <= self.mobility_teacher_recovery_hold_sec <= 0.60
            or not math.isfinite(self.mobility_teacher_recovery_decay_sec)
            or not 0.20 <= self.mobility_teacher_recovery_decay_sec <= 1.20
        ):
            raise ValueError("physics PPO teacher-recovery timing is invalid")
        mobility_teacher_scales = (
            self.mobility_teacher_lower_body_scale,
            self.mobility_teacher_waist_scale,
            self.mobility_teacher_arm_scale,
        )
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in mobility_teacher_scales
        ):
            raise ValueError("physics PPO mobility teacher group scale is invalid")
        if not (
            0.01 <= self.mobility_teacher_lower_body_target_step_rad <= 0.20
            and 0.10 <= self.mobility_teacher_lower_body_target_filter_fraction <= 1.0
            and 0.01 <= self.mobility_teacher_waist_target_step_rad <= 0.10
            and 0.10 <= self.mobility_teacher_waist_target_filter_fraction <= 1.0
            and 0.02 <= self.mobility_teacher_arm_target_step_rad <= 0.20
            and 0.10 <= self.mobility_teacher_arm_target_filter_fraction <= 1.0
        ):
            raise ValueError("physics PPO mobility teacher target filter is invalid")
        if not math.isfinite(self.mobility_predictive_teacher_gate_floor) or not (
            0.0 <= self.mobility_predictive_teacher_gate_floor <= 0.80
        ):
            raise ValueError("physics PPO predictive teacher-gate floor is invalid")
        if not 4096 <= self.teacher_pretraining_samples <= 1_048_576:
            raise ValueError("physics PPO teacher sample count is invalid")
        if not 1 <= self.teacher_pretraining_epochs <= 200:
            raise ValueError("physics PPO teacher epoch count is invalid")
        values = (
            self.learning_rate,
            self.discount,
            self.gae_lambda,
            self.clip_ratio,
            self.entropy_coefficient,
            self.value_coefficient,
            self.maximum_gradient_norm,
            self.policy_anchor_coefficient,
            self.first_save_selection_weight,
            self.first_hand_save_selection_weight,
            self.second_attempt_save_selection_weight,
            self.second_attempt_hand_save_selection_weight,
            self.second_save_selection_weight,
            self.second_hand_save_selection_weight,
            self.hand_reach_selection_weight,
            self.teacher_parent_replay_coefficient,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("physics PPO settings must be finite and positive")
        if (
            not math.isfinite(self.qualified_first_save_selection_weight)
            or not 0.0 <= self.qualified_first_save_selection_weight <= 10_000.0
        ):
            raise ValueError("physics PPO qualified-save selection weight is invalid")
        if (
            not math.isfinite(self.first_save_side_balance_selection_weight)
            or not 0.0 <= self.first_save_side_balance_selection_weight <= 10_000.0
        ):
            raise ValueError("physics PPO side-balance selection weight is invalid")
        if self.discount > 1.0 or self.gae_lambda > 1.0 or self.clip_ratio > 0.5:
            raise ValueError("physics PPO discount/lambda/clip exceeds its ceiling")
        if not (
            math.isfinite(self.training_second_shot_probability)
            and 0.0 <= self.training_second_shot_probability <= 1.0
        ):
            raise ValueError("physics PPO second-shot curriculum probability is invalid")
        if (
            not math.isfinite(self.training_hard_shot_fraction)
            or not 0.0 <= self.training_hard_shot_fraction <= 1.0
        ):
            raise ValueError("physics PPO hard-shot fraction is invalid")
        if self.training_hard_shot_height_mode not in {"low", "mid", "high", "balanced"}:
            raise ValueError("physics PPO hard-shot height mode is invalid")
        if self.training_hard_shot_side_mode not in {"negative", "positive", "balanced"}:
            raise ValueError("physics PPO hard-shot side mode is invalid")
        if self.training_reach_reward_semantics not in {
            "STATE_DENSITY",
            "POTENTIAL_PROGRESS_ONLY",
        }:
            raise ValueError("physics PPO reach reward semantics are invalid")
        if (
            not math.isfinite(self.training_hard_height_reach_reward_scale)
            or not 0.0 <= self.training_hard_height_reach_reward_scale <= 10.0
            or not math.isfinite(self.training_hard_height_reach_threshold_m)
            or not 0.80 <= self.training_hard_height_reach_threshold_m <= 1.40
            or not math.isfinite(self.training_hard_height_reach_distance_decay)
            or not 0.50 <= self.training_hard_height_reach_distance_decay <= 4.0
        ):
            raise ValueError("physics PPO hard-height reach reward is invalid")
        if (
            not math.isfinite(self.training_unsafe_penalty)
            or not 10.0 <= self.training_unsafe_penalty <= 1000.0
        ):
            raise ValueError("physics PPO training unsafe penalty must be in [10, 1000]")
        if not (
            math.isfinite(self.training_save_then_unsafe_penalty)
            and 0.0 <= self.training_save_then_unsafe_penalty <= 2_000.0
        ):
            raise ValueError("physics PPO save-then-unsafe penalty is invalid")
        if (
            not math.isfinite(self.training_task_motion_reward_scale)
            or not 0.0 <= self.training_task_motion_reward_scale <= 20.0
        ):
            raise ValueError("physics PPO task-motion reward is invalid")
        if (
            not math.isfinite(self.training_recovery_progress_reward_scale)
            or not 0.0 <= self.training_recovery_progress_reward_scale <= 100.0
            or not math.isfinite(self.training_recovery_progress_linear_speed_decay)
            or not 0.10 <= self.training_recovery_progress_linear_speed_decay <= 20.0
            or not math.isfinite(self.training_recovery_progress_angular_speed_decay)
            or not 0.05 <= self.training_recovery_progress_angular_speed_decay <= 10.0
        ):
            raise ValueError("physics PPO recovery-progress reward is invalid")
        if not math.isfinite(self.training_recovery_event_bonus) or not (
            1.0 <= self.training_recovery_event_bonus <= 500.0
        ):
            raise ValueError("physics PPO recovery-event bonus is invalid")
        save_rewards = (self.training_true_save_bonus, self.training_hand_save_bonus)
        if (
            not 1.0 <= self.training_true_save_bonus <= 500.0
            or (
                self.training_hand_save_bonus is not None
                and not 1.0 <= self.training_hand_save_bonus <= 500.0
            )
            or any(value is not None and not math.isfinite(value) for value in save_rewards)
        ):
            raise ValueError("physics PPO save-event bonus is invalid")
        if self.training_hard_shot_flight_time_range_sec is not None:
            hard_flight = self.training_hard_shot_flight_time_range_sec
            if (
                not all(math.isfinite(value) for value in hard_flight)
                or hard_flight[0] >= hard_flight[1]
                or not 0.30 <= hard_flight[0] < hard_flight[1] <= 1.20
                or self.training_hard_shot_fraction <= 0.0
            ):
                raise ValueError("physics PPO hard-shot flight curriculum is invalid")
        if self.shot_difficulty_profile not in {"standard", "match", "advanced", "elite"}:
            raise ValueError("physics PPO shot difficulty profile is invalid")
        if self.shot_difficulty_profile == "match" and not self.shot_intent_cue_enabled:
            raise ValueError("physics PPO match profile requires shot-intent cue")
        if not 0.0 <= self.task_space_reach_blend <= 0.85:
            raise ValueError("physics PPO task-space reach blend is invalid")
        if self.task_space_reach_atlas_enabled and self.task_space_reach_blend <= 0.0:
            raise ValueError("physics PPO reach atlas requires positive task-space blend")
        if (
            not math.isfinite(self.runtime_task_space_reach_blend)
            or not 0.0 <= self.runtime_task_space_reach_blend <= 0.85
        ):
            raise ValueError("physics PPO runtime reach blend is invalid")
        if self.runtime_task_space_reach_enabled and (
            not self.mobility_option_enabled
            or not self.task_space_reach_atlas_enabled
            or self.task_space_reach_blend <= 0.0
            or self.runtime_task_space_reach_blend <= 0.0
        ):
            raise ValueError(
                "physics PPO runtime reach requires mobility and a task-space reach atlas"
            )
        if (
            not math.isfinite(self.second_shot_reach_multiplier)
            or not 1.0 <= self.second_shot_reach_multiplier <= 2.0
            or self.task_space_reach_blend * self.second_shot_reach_multiplier > 0.85
        ):
            raise ValueError("physics PPO second-shot reach multiplier is invalid")
        if (
            not math.isfinite(self.minimum_first_save_rate_for_selection)
            or not 0.0 <= self.minimum_first_save_rate_for_selection <= 1.0
        ):
            raise ValueError("physics PPO first-save selection floor is invalid")
        if (
            not math.isfinite(self.minimum_first_save_stratum_rate_for_selection)
            or not 0.0 <= self.minimum_first_save_stratum_rate_for_selection <= 1.0
        ):
            raise ValueError("physics PPO first-save stratum selection floor is invalid")
        if (
            not math.isfinite(self.minimum_hand_displacement_for_selection_m)
            or not 0.0 <= self.minimum_hand_displacement_for_selection_m <= 0.80
        ):
            raise ValueError("physics PPO hand-displacement selection floor is invalid")
        if (
            not math.isfinite(self.hand_target_distance_selection_weight)
            or not 0.0 <= self.hand_target_distance_selection_weight <= 200.0
        ):
            raise ValueError("physics PPO hand-target selection weight is invalid")
        if (
            not math.isfinite(self.root_angular_speed_selection_penalty_weight)
            or not 0.0 <= self.root_angular_speed_selection_penalty_weight <= 100.0
        ):
            raise ValueError("physics PPO root-angular selection penalty is invalid")
        if (
            not math.isfinite(self.second_release_recenter_selection_weight)
            or not 0.0 <= self.second_release_recenter_selection_weight <= 100.0
        ):
            raise ValueError("physics PPO recenter selection weight is invalid")
        if (
            not math.isfinite(self.first_save_stratum_balance_selection_weight)
            or not 0.0 <= self.first_save_stratum_balance_selection_weight <= 2_000.0
        ):
            raise ValueError("physics PPO stratum-balance selection weight is invalid")
        if (
            not math.isfinite(self.maximum_exploration_root_angular_speed_for_selection)
            or not 3.50 <= self.maximum_exploration_root_angular_speed_for_selection <= 8.0
        ):
            raise ValueError("physics PPO exploration angular-speed ceiling is invalid")
        if (
            not math.isfinite(self.online_parent_distillation_coefficient)
            or not 0.0 <= self.online_parent_distillation_coefficient <= 2.0
        ):
            raise ValueError("physics PPO online parent distillation coefficient is invalid")
        if (
            self.online_parent_distillation_coefficient > 0.0
            and self.initialization_checkpoint is None
        ):
            raise ValueError("physics PPO online parent distillation requires a parent checkpoint")
        if self.activation_ceiling != "SIM_ONLY" or self.hardware_authorized:
            raise ValueError("physics PPO training is SIM_ONLY")
        if self.initialization_checkpoint is not None:
            checkpoint = Path(self.initialization_checkpoint)
            if (
                not checkpoint.is_absolute()
                or not checkpoint.is_file()
                or checkpoint.suffix != ".pt"
            ):
                raise ValueError("physics PPO initialization checkpoint is invalid")
        combat_paths = (self.combat_teacher_checkout, self.combat_teacher_checkpoint)
        if any(value is not None for value in combat_paths) and not all(
            value is not None for value in combat_paths
        ):
            raise ValueError("goalkeeper combat teacher requires checkout and checkpoint together")
        if self.combat_teacher_checkout is not None:
            checkout = Path(self.combat_teacher_checkout)
            checkpoint = Path(str(self.combat_teacher_checkpoint))
            if (
                not checkout.is_absolute()
                or not checkout.is_dir()
                or not checkpoint.is_absolute()
                or not checkpoint.is_file()
            ):
                raise ValueError("goalkeeper combat teacher paths are invalid")
            if (
                self.teacher_pretraining_enabled
                or self.arm_only_online_update
                or self.lower_body_and_arms_online_update
                or self.lateral_drive_gate_only_online_update
                or self.support_landing_online_update
            ):
                raise ValueError(
                    "goalkeeper combat uses its pinned whole-body teacher "
                    "instead of arm pretraining"
                )
        if self.targeted_dive_checkpoint is None and (
            self.targeted_dive_overhead_reach_prior is not None
            or self.targeted_dive_overhead_reach_blend != 0.0
        ):
            raise ValueError("physics PPO overhead reach requires a targeted dive option")
        if self.targeted_dive_checkpoint is not None:
            targeted = Path(self.targeted_dive_checkpoint)
            if not targeted.is_absolute() or not targeted.is_file():
                raise ValueError("goalkeeper targeted dive checkpoint is invalid")
            if (
                self.combat_teacher_checkout is not None
                or self.mobility_option_enabled
                or self.teacher_pretraining_enabled
                or not self.shot_intent_cue_enabled
            ):
                raise ValueError(
                    "targeted dive RL requires intent cues, no legacy teacher, "
                    "and no teacher pretraining"
                )
            dive_values = (
                self.targeted_dive_option_duration_sec,
                self.targeted_dive_posture_exception_duration_sec,
                self.targeted_dive_decoder_residual_authority,
                self.targeted_dive_actor_residual_scale,
                self.targeted_dive_anchor_lower_body_scale,
                self.targeted_dive_anchor_waist_scale,
                self.targeted_dive_anchor_arm_scale,
                self.targeted_dive_low_shot_phase_scale,
                self.targeted_dive_mid_shot_phase_scale,
                self.targeted_dive_high_shot_phase_scale,
            )
            if any(not math.isfinite(value) or value <= 0.0 for value in dive_values):
                raise ValueError("goalkeeper targeted dive RL settings are invalid")
            if (
                self.training_first_shot_release_sec is not None
                and self.targeted_dive_prediction_lead_sec >= self.training_first_shot_release_sec
            ):
                raise ValueError("goalkeeper targeted dive prediction cue precedes the episode")
            if not math.isfinite(self.targeted_dive_phase_hold_sec) or not (
                0.0 <= self.targeted_dive_phase_hold_sec <= 0.40
            ):
                raise ValueError("goalkeeper targeted dive phase hold is invalid")
            if not math.isfinite(self.targeted_dive_lateral_drive_scale) or not (
                0.0 <= self.targeted_dive_lateral_drive_scale <= 1.0
            ):
                raise ValueError("goalkeeper targeted dive lateral drive is invalid")
            from rosclaw_soccer.training.goalkeeper_targeted_dive_mjwarp import (
                GoalkeeperTargetedDiveRLConfig,
            )

            GoalkeeperTargetedDiveRLConfig(
                option_duration_sec=self.targeted_dive_option_duration_sec,
                phase_hold_sec=self.targeted_dive_phase_hold_sec,
                actor_recovery_plasticity_sec=(self.targeted_dive_actor_recovery_plasticity_sec),
                actor_recovery_residual_authority_scale=(
                    self.targeted_dive_actor_recovery_residual_authority_scale
                ),
                post_save_counterstep_enabled=(self.targeted_dive_post_save_counterstep_enabled),
                post_save_counterstep_duration_sec=(
                    self.targeted_dive_post_save_counterstep_duration_sec
                ),
                post_save_counterstep_command_limit=(
                    self.targeted_dive_post_save_counterstep_command_limit
                ),
                post_save_counterstep_capture_horizon_sec=(
                    self.targeted_dive_post_save_counterstep_capture_horizon_sec
                ),
                post_save_counterstep_recenter_weight=(
                    self.targeted_dive_post_save_counterstep_recenter_weight
                ),
                post_save_option_release_sec=(self.targeted_dive_post_save_option_release_sec),
                post_save_fall_recovery_enabled=(
                    self.targeted_dive_post_save_fall_recovery_enabled
                ),
                post_save_fall_recovery_duration_sec=(
                    self.targeted_dive_post_save_fall_recovery_duration_sec
                ),
                post_save_fall_minimum_pelvis_height_m=(
                    self.targeted_dive_post_save_fall_minimum_pelvis_height_m
                ),
                post_save_fall_minimum_upright_projection=(
                    self.targeted_dive_post_save_fall_minimum_upright_projection
                ),
                post_save_fall_maximum_root_linear_speed_mps=(
                    self.targeted_dive_post_save_fall_maximum_root_linear_speed_mps
                ),
                post_save_fall_maximum_root_angular_speed_rad_s=(
                    self.targeted_dive_post_save_fall_maximum_root_angular_speed_rad_s
                ),
                prediction_lead_sec=self.targeted_dive_prediction_lead_sec,
                nominal_shot_flight_time_sec=(self.targeted_dive_nominal_shot_flight_time_sec),
                intercept_phase_at_arrival=(self.targeted_dive_intercept_phase_at_arrival),
                phase_sync_minimum_target_height_m=(
                    self.targeted_dive_phase_sync_minimum_target_height_m
                ),
                posture_exception_duration_sec=(self.targeted_dive_posture_exception_duration_sec),
                dive_maximum_root_angular_speed_rad_s=(
                    self.targeted_dive_root_angular_speed_guard_ceiling_rad_s
                ),
                decoder_residual_authority=self.targeted_dive_decoder_residual_authority,
                decoder_lower_body_residual_authority=(
                    self.targeted_dive_decoder_lower_body_residual_authority
                ),
                decoder_lower_body_command_scale=(
                    self.targeted_dive_decoder_lower_body_command_scale
                ),
                decoder_waist_residual_authority=(
                    self.targeted_dive_decoder_waist_residual_authority
                ),
                decoder_arm_residual_authority=(self.targeted_dive_decoder_arm_residual_authority),
                actor_residual_scale=self.targeted_dive_actor_residual_scale,
                anchor_lower_body_scale=self.targeted_dive_anchor_lower_body_scale,
                anchor_waist_scale=self.targeted_dive_anchor_waist_scale,
                anchor_arm_scale=self.targeted_dive_anchor_arm_scale,
                minimum_option_gate=self.targeted_dive_minimum_option_gate,
                runtime_reach_blend=self.targeted_dive_runtime_reach_blend,
                runtime_reach_feedback_blend=(self.targeted_dive_runtime_reach_feedback_blend),
                runtime_reach_feedback_gain=(self.targeted_dive_runtime_reach_feedback_gain),
                runtime_reach_feedback_maximum_error_m=(
                    self.targeted_dive_runtime_reach_feedback_maximum_error_m
                ),
                runtime_reach_feedback_support_scale=(
                    self.targeted_dive_runtime_reach_feedback_support_scale
                ),
                runtime_contact_support_side_enabled=(
                    self.targeted_dive_runtime_contact_support_side_enabled
                ),
                actor_contact_support_side_enabled=(
                    self.targeted_dive_actor_contact_support_side_enabled
                ),
                actor_recovery_context_enabled=(self.targeted_dive_actor_recovery_context_enabled),
                runtime_whole_body_reach_blend=(self.targeted_dive_runtime_whole_body_reach_blend),
                runtime_whole_body_reach_full_below_height_m=(
                    self.targeted_dive_runtime_whole_body_reach_full_below_height_m
                ),
                runtime_whole_body_reach_maximum_height_m=(
                    self.targeted_dive_runtime_whole_body_reach_maximum_height_m
                ),
                runtime_whole_body_reach_waist_scale=(
                    self.targeted_dive_runtime_whole_body_reach_waist_scale
                ),
                runtime_whole_body_reach_arm_scale=(
                    self.targeted_dive_runtime_whole_body_reach_arm_scale
                ),
                runtime_whole_body_reach_support_scale=(
                    self.targeted_dive_runtime_whole_body_reach_support_scale
                ),
                runtime_whole_body_reach_release_sec=(
                    self.targeted_dive_runtime_whole_body_reach_release_sec
                ),
                runtime_reach_contact_standoff_m=(
                    self.targeted_dive_runtime_reach_contact_standoff_m
                ),
                runtime_reach_lateral_lead_m=(self.targeted_dive_runtime_reach_lateral_lead_m),
                runtime_reach_vertical_lead_m=(self.targeted_dive_runtime_reach_vertical_lead_m),
                runtime_reach_low_vertical_lead_m=(
                    self.targeted_dive_runtime_reach_low_vertical_lead_m
                ),
                runtime_reach_mid_vertical_lead_m=(
                    self.targeted_dive_runtime_reach_mid_vertical_lead_m
                ),
                runtime_reach_high_vertical_lead_m=(
                    self.targeted_dive_runtime_reach_high_vertical_lead_m
                ),
                overhead_reach_prior_path=self.targeted_dive_overhead_reach_prior,
                overhead_reach_blend=self.targeted_dive_overhead_reach_blend,
                overhead_reach_minimum_target_height_m=(
                    self.targeted_dive_overhead_reach_minimum_target_height_m
                ),
                overhead_reach_full_target_height_m=(
                    self.targeted_dive_overhead_reach_full_target_height_m
                ),
                overhead_reach_lower_body_scale=(
                    self.targeted_dive_overhead_reach_lower_body_scale
                ),
                overhead_reach_waist_scale=(self.targeted_dive_overhead_reach_waist_scale),
                overhead_reach_arm_scale=self.targeted_dive_overhead_reach_arm_scale,
                mosaic_gmt_model_path=self.targeted_dive_mosaic_gmt_model,
                mosaic_gmt_skill_path=self.targeted_dive_mosaic_gmt_skill,
                mosaic_gmt_blend=self.targeted_dive_mosaic_gmt_blend,
                mosaic_gmt_stability_floor=(self.targeted_dive_mosaic_gmt_stability_floor),
                mosaic_gmt_minimum_target_height_m=(
                    self.targeted_dive_mosaic_gmt_minimum_target_height_m
                ),
                mosaic_gmt_full_target_height_m=(
                    self.targeted_dive_mosaic_gmt_full_target_height_m
                ),
                mosaic_gmt_lower_body_scale=(self.targeted_dive_mosaic_gmt_lower_body_scale),
                mosaic_gmt_waist_scale=self.targeted_dive_mosaic_gmt_waist_scale,
                mosaic_gmt_arm_scale=self.targeted_dive_mosaic_gmt_arm_scale,
                mosaic_gmt_getup_skill_path=(self.targeted_dive_mosaic_gmt_getup_skill),
                mosaic_gmt_getup_blend=self.targeted_dive_mosaic_gmt_getup_blend,
                mosaic_gmt_getup_activation_maximum_pelvis_height_m=(
                    self.targeted_dive_mosaic_gmt_getup_activation_maximum_pelvis_height_m
                ),
                mosaic_gmt_getup_blend_in_sec=(self.targeted_dive_mosaic_gmt_getup_blend_in_sec),
                mosaic_gmt_getup_reference_feedforward_blend=(
                    self.targeted_dive_mosaic_gmt_getup_reference_feedforward_blend
                ),
                mosaic_gmt_getup_lower_body_scale=(
                    self.targeted_dive_mosaic_gmt_getup_lower_body_scale
                ),
                mosaic_gmt_getup_waist_scale=(self.targeted_dive_mosaic_gmt_getup_waist_scale),
                mosaic_gmt_getup_arm_scale=self.targeted_dive_mosaic_gmt_getup_arm_scale,
                maximum_arm_target_step_rad=(self.targeted_dive_maximum_arm_target_step_rad),
                arm_target_filter_fraction=(self.targeted_dive_arm_target_filter_fraction),
                maximum_lower_body_target_step_rad=(
                    self.targeted_dive_maximum_lower_body_target_step_rad
                ),
                lower_body_target_filter_fraction=(
                    self.targeted_dive_lower_body_target_filter_fraction
                ),
                lateral_drive_scale=self.targeted_dive_lateral_drive_scale,
                negative_target_lateral_drive_scale=(
                    self.targeted_dive_negative_target_lateral_drive_scale
                ),
                lateral_drive_full_activation_gate=(
                    self.targeted_dive_lateral_drive_full_activation_gate
                ),
                lateral_drive_capture_enabled=(self.targeted_dive_lateral_drive_capture_enabled),
                lateral_drive_capture_horizon_sec=(
                    self.targeted_dive_lateral_drive_capture_horizon_sec
                ),
                lateral_drive_target_standoff_m=(
                    self.targeted_dive_lateral_drive_target_standoff_m
                ),
                lateral_drive_capture_scale_m=(self.targeted_dive_lateral_drive_capture_scale_m),
                lateral_drive_learned_gate_enabled=(
                    self.targeted_dive_lateral_drive_learned_gate_enabled
                ),
                runtime_lateral_lunge_blend=(self.targeted_dive_runtime_lateral_lunge_blend),
                runtime_lateral_lunge_hip_roll_rad=(
                    self.targeted_dive_runtime_lateral_lunge_hip_roll_rad
                ),
                runtime_lateral_lunge_ankle_roll_rad=(
                    self.targeted_dive_runtime_lateral_lunge_ankle_roll_rad
                ),
                runtime_lateral_lunge_approach_horizon_sec=(
                    self.targeted_dive_runtime_lateral_lunge_approach_horizon_sec
                ),
                substep_upper_body_guard_enabled=(
                    self.targeted_dive_substep_upper_body_guard_enabled
                ),
                substep_upper_body_guard_onset_rad_s=(
                    self.targeted_dive_substep_upper_body_guard_onset_rad_s
                ),
                substep_upper_body_guard_ceiling_rad_s=(
                    self.targeted_dive_substep_upper_body_guard_ceiling_rad_s
                ),
                substep_upper_body_minimum_position_scale=(
                    self.targeted_dive_substep_upper_body_minimum_position_scale
                ),
                substep_option_lower_body_guard_enabled=(
                    self.targeted_dive_substep_option_lower_body_guard_enabled
                ),
                substep_option_lower_body_guard_onset_rad_s=(
                    self.targeted_dive_substep_option_lower_body_guard_onset_rad_s
                ),
                substep_option_lower_body_guard_ceiling_rad_s=(
                    self.targeted_dive_substep_option_lower_body_guard_ceiling_rad_s
                ),
                substep_option_lower_body_minimum_scale=(
                    self.targeted_dive_substep_option_lower_body_minimum_scale
                ),
                canonical_locomotion_mirror_enabled=(
                    self.targeted_dive_canonical_locomotion_mirror_enabled
                ),
                low_shot_phase_scale=self.targeted_dive_low_shot_phase_scale,
                mid_shot_phase_scale=self.targeted_dive_mid_shot_phase_scale,
                high_shot_phase_scale=self.targeted_dive_high_shot_phase_scale,
            )
            if not math.isfinite(self.targeted_dive_initial_gate) or not (
                0.0 <= self.targeted_dive_initial_gate <= 0.60
            ):
                raise ValueError("goalkeeper targeted dive initial gate is invalid")
        elif self.lower_body_and_arms_online_update:
            raise ValueError(
                "physics PPO lower-body-and-arms specialist requires a targeted dive option"
            )
        elif self.lateral_drive_gate_only_online_update:
            raise ValueError(
                "physics PPO lateral-drive gate specialist requires a targeted dive option"
            )
        elif self.support_landing_online_update:
            raise ValueError(
                "physics PPO support-landing specialist requires a targeted dive option"
            )
        if (
            self.lateral_drive_gate_only_online_update or self.support_landing_online_update
        ) and not self.targeted_dive_lateral_drive_learned_gate_enabled:
            raise ValueError("physics PPO drive-gate specialist requires the learned drive gate")
        maximum_allowed_blend = 1.0 if self.mobility_option_enabled else 0.50
        if (
            not math.isfinite(self.maximum_combat_teacher_blend)
            or not 0.05 <= self.maximum_combat_teacher_blend <= maximum_allowed_blend
        ):
            raise ValueError(
                f"goalkeeper combat teacher blend must be in [0.05, {maximum_allowed_blend:.2f}]"
            )
        if self.mobility_option_enabled and self.combat_teacher_checkout is None:
            raise ValueError("goalkeeper mobility option requires its pinned whole-body teacher")
        if (
            self.combat_teacher_intercept_conditioning_enabled
            and self.combat_teacher_checkout is None
        ):
            raise ValueError("goalkeeper intercept conditioning requires its pinned teacher")
        if not 0 <= self.combat_gate_pretraining_batches <= 64:
            raise ValueError("goalkeeper combat gate batch count is invalid")
        if not 1 <= self.combat_gate_pretraining_epochs <= 100:
            raise ValueError("goalkeeper combat gate epoch count is invalid")
        if self.combat_gate_pretraining_batches > 0 and self.combat_teacher_checkout is None:
            raise ValueError("goalkeeper combat gate curriculum requires its pinned teacher")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


def _controller_semantic_migration(
    *,
    parent_training_config: dict[str, Any],
    current: GoalkeeperPhysicsPPOConfig,
) -> dict[str, Any]:
    """Make checkpoint/control reinterpretation visible in continual learning."""

    defaults = asdict(GoalkeeperPhysicsPPOConfig())
    changes: dict[str, dict[str, Any]] = {}
    for name in _CONTROLLER_SEMANTIC_FIELDS:
        parent_value = parent_training_config.get(name, defaults[name])
        current_value = getattr(current, name)
        if parent_value != current_value:
            changes[name] = {
                "parent": parent_value,
                "current": current_value,
            }
    return {
        "exact": not changes,
        "changed_fields": changes,
        "semantics": "EXPLICIT_CONTROLLER_MIGRATION_NOT_SILENT",
    }


def _build_actor_critic(torch: Any, nn: Any, observation_size: int, action_size: int, hidden: int):
    class ActorCritic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(observation_size, hidden),
                nn.ELU(),
                nn.Linear(hidden, hidden),
                nn.ELU(),
            )
            self.actor = nn.Linear(hidden, action_size)
            adapter_hidden = max(16, hidden // 2)
            self.specialist_adapter_trunk = nn.Sequential(
                nn.Linear(observation_size, adapter_hidden),
                nn.ELU(),
            )
            self.specialist_adapter = nn.Linear(adapter_hidden, action_size)
            self.critic = nn.Linear(hidden, 1)
            # Start exactly at the frozen locomotion champion.  PPO must earn
            # every residual instead of first unlearning a random perturbation.
            nn.init.zeros_(self.actor.weight)
            nn.init.zeros_(self.actor.bias)
            # A legacy checkpoint therefore remains behaviorally exact.  The
            # adapter gains nonlinear representation capacity only after a
            # declared specialist scope receives gradients.
            nn.init.zeros_(self.specialist_adapter.weight)
            nn.init.zeros_(self.specialist_adapter.bias)
            # Teacher pretraining already supplies broad spatial coverage.
            # Lower initial variance avoids learning impulsive one-arm swings
            # while preserving a trainable stochastic PPO policy.
            self.log_std = nn.Parameter(torch.full((action_size,), -2.00))

        def forward(self, observation: Any) -> tuple[Any, Any, Any]:
            latent = self.trunk(observation)
            adapter_latent = self.specialist_adapter_trunk(observation)
            mean = self.actor(latent) + self.specialist_adapter(adapter_latent)
            return mean, self.critic(latent).squeeze(-1), self.log_std

    return ActorCritic()


def _load_actor_critic_state(model: Any, state_dict: dict[str, Any]) -> str:
    """Load a policy while zero-migrating the optional specialist adapter."""

    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    allowed_missing = {
        key
        for key in model.state_dict()
        if key.startswith("specialist_adapter_trunk.") or key.startswith("specialist_adapter.")
    }
    if unexpected or not missing.issubset(allowed_missing):
        raise ValueError("physics PPO actor checkpoint state contract mismatch")
    return "ZERO_OUTPUT_SPECIALIST_ADAPTER" if missing else "EXACT"


def _migrate_initialization_state(
    *,
    torch: Any,
    state_dict: dict[str, Any],
    old_observation_size: int,
    new_observation_size: int,
) -> tuple[dict[str, Any], str]:
    """Expand a parent with explicitly zero-initialized causal features."""

    if old_observation_size == new_observation_size:
        return state_dict, "EXACT"
    migration = (old_observation_size, new_observation_size)
    if migration not in {(74, 77), (89, 90), (89, 92), (90, 92), (92, 96)}:
        raise ValueError("physics PPO initialization observation migration is unsupported")
    key = "trunk.0.weight"
    source = state_dict.get(key)
    if source is None or tuple(source.shape)[1] != old_observation_size:
        raise ValueError("physics PPO initialization first-layer contract changed")
    migrated = dict(state_dict)
    for input_key in (key, "specialist_adapter_trunk.0.weight"):
        input_weight = state_dict.get(input_key)
        if input_weight is None:
            continue
        if tuple(input_weight.shape)[1] != old_observation_size:
            raise ValueError("physics PPO initialization adapter input contract changed")
        expanded = torch.zeros(
            (input_weight.shape[0], new_observation_size),
            dtype=input_weight.dtype,
            device=input_weight.device,
        )
        if migration == (74, 77):
            # The cue is inserted before the final phase triplet and elapsed-time
            # scalar.  All legacy semantics are preserved exactly.
            expanded[:, :70] = input_weight[:, :70]
            expanded[:, 73:] = input_weight[:, 70:]
        elif migration == (89, 90):
            # Foot-support side is inserted before the existing cue, phase and
            # time tail.  The zero column makes an 89-D parent behaviorally exact.
            expanded[:, :82] = input_weight[:, :82]
            expanded[:, 83:] = input_weight[:, 82:]
        elif migration == (89, 92):
            # Insert support side plus independent left/right contact flags.
            expanded[:, :82] = input_weight[:, :82]
            expanded[:, 85:] = input_weight[:, 82:]
        elif migration == (90, 92):
            # A 90-D parent already owns support side at column 82.  Preserve
            # that learned column and insert left/right contact flags after it.
            expanded[:, :83] = input_weight[:, :83]
            expanded[:, 85:] = input_weight[:, 83:]
        else:
            # Preserve contact support at columns 82:85 and insert four
            # zero-initialized causal recovery features before cue/phase/time.
            expanded[:, :85] = input_weight[:, :85]
            expanded[:, 89:] = input_weight[:, 85:]
        migrated[input_key] = expanded
    semantics = {
        (74, 77): "EXPAND_74_TO_77_ZERO_INITIALIZED_CUE",
        (89, 90): "EXPAND_89_TO_90_ZERO_INITIALIZED_CONTACT_SUPPORT",
        (89, 92): "EXPAND_89_TO_92_ZERO_INITIALIZED_FOOT_CONTACT_MODE",
        (90, 92): "EXPAND_90_TO_92_PRESERVE_SUPPORT_ADD_FOOT_CONTACTS",
        (92, 96): "EXPAND_92_TO_96_ZERO_INITIALIZED_CAUSAL_RECOVERY_CONTEXT",
    }[migration]
    return migrated, semantics


def _build_environment(
    *,
    active: GoalkeeperPhysicsPPOConfig,
    asset_root: Path,
    locomotion_policy_path: Path,
    device: Any,
    world_config: Any,
) -> Any:
    if active.targeted_dive_checkpoint is not None:
        from rosclaw_soccer.training.goalkeeper_targeted_dive_mjwarp import (
            GoalkeeperTargetedDiveMJWarpBatch,
            GoalkeeperTargetedDiveRLConfig,
        )

        return GoalkeeperTargetedDiveMJWarpBatch(
            asset_root=asset_root,
            locomotion_policy_path=locomotion_policy_path,
            targeted_dive_checkpoint=Path(active.targeted_dive_checkpoint),
            device=device,
            config=world_config,
            dive_config=GoalkeeperTargetedDiveRLConfig(
                option_duration_sec=active.targeted_dive_option_duration_sec,
                phase_hold_sec=active.targeted_dive_phase_hold_sec,
                actor_recovery_plasticity_sec=(active.targeted_dive_actor_recovery_plasticity_sec),
                actor_recovery_residual_authority_scale=(
                    active.targeted_dive_actor_recovery_residual_authority_scale
                ),
                post_save_counterstep_enabled=(active.targeted_dive_post_save_counterstep_enabled),
                post_save_counterstep_duration_sec=(
                    active.targeted_dive_post_save_counterstep_duration_sec
                ),
                post_save_counterstep_command_limit=(
                    active.targeted_dive_post_save_counterstep_command_limit
                ),
                post_save_counterstep_capture_horizon_sec=(
                    active.targeted_dive_post_save_counterstep_capture_horizon_sec
                ),
                post_save_counterstep_recenter_weight=(
                    active.targeted_dive_post_save_counterstep_recenter_weight
                ),
                post_save_option_release_sec=(active.targeted_dive_post_save_option_release_sec),
                post_save_fall_recovery_enabled=(
                    active.targeted_dive_post_save_fall_recovery_enabled
                ),
                post_save_fall_recovery_duration_sec=(
                    active.targeted_dive_post_save_fall_recovery_duration_sec
                ),
                post_save_fall_minimum_pelvis_height_m=(
                    active.targeted_dive_post_save_fall_minimum_pelvis_height_m
                ),
                post_save_fall_minimum_upright_projection=(
                    active.targeted_dive_post_save_fall_minimum_upright_projection
                ),
                post_save_fall_maximum_root_linear_speed_mps=(
                    active.targeted_dive_post_save_fall_maximum_root_linear_speed_mps
                ),
                post_save_fall_maximum_root_angular_speed_rad_s=(
                    active.targeted_dive_post_save_fall_maximum_root_angular_speed_rad_s
                ),
                prediction_lead_sec=active.targeted_dive_prediction_lead_sec,
                nominal_shot_flight_time_sec=(active.targeted_dive_nominal_shot_flight_time_sec),
                intercept_phase_at_arrival=(active.targeted_dive_intercept_phase_at_arrival),
                phase_sync_minimum_target_height_m=(
                    active.targeted_dive_phase_sync_minimum_target_height_m
                ),
                posture_exception_duration_sec=(
                    active.targeted_dive_posture_exception_duration_sec
                ),
                dive_maximum_root_angular_speed_rad_s=(
                    active.targeted_dive_root_angular_speed_guard_ceiling_rad_s
                ),
                decoder_residual_authority=(active.targeted_dive_decoder_residual_authority),
                decoder_lower_body_residual_authority=(
                    active.targeted_dive_decoder_lower_body_residual_authority
                ),
                decoder_lower_body_command_scale=(
                    active.targeted_dive_decoder_lower_body_command_scale
                ),
                decoder_waist_residual_authority=(
                    active.targeted_dive_decoder_waist_residual_authority
                ),
                decoder_arm_residual_authority=(
                    active.targeted_dive_decoder_arm_residual_authority
                ),
                actor_residual_scale=active.targeted_dive_actor_residual_scale,
                anchor_lower_body_scale=active.targeted_dive_anchor_lower_body_scale,
                anchor_waist_scale=active.targeted_dive_anchor_waist_scale,
                anchor_arm_scale=active.targeted_dive_anchor_arm_scale,
                minimum_option_gate=active.targeted_dive_minimum_option_gate,
                runtime_reach_blend=active.targeted_dive_runtime_reach_blend,
                runtime_reach_feedback_blend=(active.targeted_dive_runtime_reach_feedback_blend),
                runtime_reach_feedback_gain=(active.targeted_dive_runtime_reach_feedback_gain),
                runtime_reach_feedback_maximum_error_m=(
                    active.targeted_dive_runtime_reach_feedback_maximum_error_m
                ),
                runtime_reach_feedback_support_scale=(
                    active.targeted_dive_runtime_reach_feedback_support_scale
                ),
                runtime_contact_support_side_enabled=(
                    active.targeted_dive_runtime_contact_support_side_enabled
                ),
                actor_contact_support_side_enabled=(
                    active.targeted_dive_actor_contact_support_side_enabled
                ),
                actor_recovery_context_enabled=(
                    active.targeted_dive_actor_recovery_context_enabled
                ),
                runtime_whole_body_reach_blend=(
                    active.targeted_dive_runtime_whole_body_reach_blend
                ),
                runtime_whole_body_reach_full_below_height_m=(
                    active.targeted_dive_runtime_whole_body_reach_full_below_height_m
                ),
                runtime_whole_body_reach_maximum_height_m=(
                    active.targeted_dive_runtime_whole_body_reach_maximum_height_m
                ),
                runtime_whole_body_reach_waist_scale=(
                    active.targeted_dive_runtime_whole_body_reach_waist_scale
                ),
                runtime_whole_body_reach_arm_scale=(
                    active.targeted_dive_runtime_whole_body_reach_arm_scale
                ),
                runtime_whole_body_reach_support_scale=(
                    active.targeted_dive_runtime_whole_body_reach_support_scale
                ),
                runtime_whole_body_reach_release_sec=(
                    active.targeted_dive_runtime_whole_body_reach_release_sec
                ),
                runtime_reach_contact_standoff_m=(
                    active.targeted_dive_runtime_reach_contact_standoff_m
                ),
                runtime_reach_lateral_lead_m=(active.targeted_dive_runtime_reach_lateral_lead_m),
                runtime_reach_vertical_lead_m=(active.targeted_dive_runtime_reach_vertical_lead_m),
                runtime_reach_low_vertical_lead_m=(
                    active.targeted_dive_runtime_reach_low_vertical_lead_m
                ),
                runtime_reach_mid_vertical_lead_m=(
                    active.targeted_dive_runtime_reach_mid_vertical_lead_m
                ),
                runtime_reach_high_vertical_lead_m=(
                    active.targeted_dive_runtime_reach_high_vertical_lead_m
                ),
                overhead_reach_prior_path=active.targeted_dive_overhead_reach_prior,
                overhead_reach_blend=active.targeted_dive_overhead_reach_blend,
                overhead_reach_minimum_target_height_m=(
                    active.targeted_dive_overhead_reach_minimum_target_height_m
                ),
                overhead_reach_full_target_height_m=(
                    active.targeted_dive_overhead_reach_full_target_height_m
                ),
                overhead_reach_lower_body_scale=(
                    active.targeted_dive_overhead_reach_lower_body_scale
                ),
                overhead_reach_waist_scale=(active.targeted_dive_overhead_reach_waist_scale),
                overhead_reach_arm_scale=active.targeted_dive_overhead_reach_arm_scale,
                mosaic_gmt_model_path=active.targeted_dive_mosaic_gmt_model,
                mosaic_gmt_skill_path=active.targeted_dive_mosaic_gmt_skill,
                mosaic_gmt_blend=active.targeted_dive_mosaic_gmt_blend,
                mosaic_gmt_stability_floor=(active.targeted_dive_mosaic_gmt_stability_floor),
                mosaic_gmt_minimum_target_height_m=(
                    active.targeted_dive_mosaic_gmt_minimum_target_height_m
                ),
                mosaic_gmt_full_target_height_m=(
                    active.targeted_dive_mosaic_gmt_full_target_height_m
                ),
                mosaic_gmt_lower_body_scale=(active.targeted_dive_mosaic_gmt_lower_body_scale),
                mosaic_gmt_waist_scale=(active.targeted_dive_mosaic_gmt_waist_scale),
                mosaic_gmt_arm_scale=active.targeted_dive_mosaic_gmt_arm_scale,
                mosaic_gmt_getup_skill_path=(active.targeted_dive_mosaic_gmt_getup_skill),
                mosaic_gmt_getup_blend=active.targeted_dive_mosaic_gmt_getup_blend,
                mosaic_gmt_getup_activation_maximum_pelvis_height_m=(
                    active.targeted_dive_mosaic_gmt_getup_activation_maximum_pelvis_height_m
                ),
                mosaic_gmt_getup_blend_in_sec=(active.targeted_dive_mosaic_gmt_getup_blend_in_sec),
                mosaic_gmt_getup_reference_feedforward_blend=(
                    active.targeted_dive_mosaic_gmt_getup_reference_feedforward_blend
                ),
                mosaic_gmt_getup_lower_body_scale=(
                    active.targeted_dive_mosaic_gmt_getup_lower_body_scale
                ),
                mosaic_gmt_getup_waist_scale=(active.targeted_dive_mosaic_gmt_getup_waist_scale),
                mosaic_gmt_getup_arm_scale=(active.targeted_dive_mosaic_gmt_getup_arm_scale),
                maximum_arm_target_step_rad=(active.targeted_dive_maximum_arm_target_step_rad),
                arm_target_filter_fraction=(active.targeted_dive_arm_target_filter_fraction),
                maximum_lower_body_target_step_rad=(
                    active.targeted_dive_maximum_lower_body_target_step_rad
                ),
                lower_body_target_filter_fraction=(
                    active.targeted_dive_lower_body_target_filter_fraction
                ),
                lateral_drive_scale=active.targeted_dive_lateral_drive_scale,
                negative_target_lateral_drive_scale=(
                    active.targeted_dive_negative_target_lateral_drive_scale
                ),
                lateral_drive_full_activation_gate=(
                    active.targeted_dive_lateral_drive_full_activation_gate
                ),
                lateral_drive_capture_enabled=(active.targeted_dive_lateral_drive_capture_enabled),
                lateral_drive_capture_horizon_sec=(
                    active.targeted_dive_lateral_drive_capture_horizon_sec
                ),
                lateral_drive_target_standoff_m=(
                    active.targeted_dive_lateral_drive_target_standoff_m
                ),
                lateral_drive_capture_scale_m=(active.targeted_dive_lateral_drive_capture_scale_m),
                lateral_drive_learned_gate_enabled=(
                    active.targeted_dive_lateral_drive_learned_gate_enabled
                ),
                canonical_locomotion_mirror_enabled=(
                    active.targeted_dive_canonical_locomotion_mirror_enabled
                ),
                runtime_lateral_lunge_blend=(active.targeted_dive_runtime_lateral_lunge_blend),
                runtime_lateral_lunge_hip_roll_rad=(
                    active.targeted_dive_runtime_lateral_lunge_hip_roll_rad
                ),
                runtime_lateral_lunge_ankle_roll_rad=(
                    active.targeted_dive_runtime_lateral_lunge_ankle_roll_rad
                ),
                runtime_lateral_lunge_approach_horizon_sec=(
                    active.targeted_dive_runtime_lateral_lunge_approach_horizon_sec
                ),
                substep_upper_body_guard_enabled=(
                    active.targeted_dive_substep_upper_body_guard_enabled
                ),
                substep_upper_body_guard_onset_rad_s=(
                    active.targeted_dive_substep_upper_body_guard_onset_rad_s
                ),
                substep_upper_body_guard_ceiling_rad_s=(
                    active.targeted_dive_substep_upper_body_guard_ceiling_rad_s
                ),
                substep_upper_body_minimum_position_scale=(
                    active.targeted_dive_substep_upper_body_minimum_position_scale
                ),
                substep_option_lower_body_guard_enabled=(
                    active.targeted_dive_substep_option_lower_body_guard_enabled
                ),
                substep_option_lower_body_guard_onset_rad_s=(
                    active.targeted_dive_substep_option_lower_body_guard_onset_rad_s
                ),
                substep_option_lower_body_guard_ceiling_rad_s=(
                    active.targeted_dive_substep_option_lower_body_guard_ceiling_rad_s
                ),
                substep_option_lower_body_minimum_scale=(
                    active.targeted_dive_substep_option_lower_body_minimum_scale
                ),
                low_shot_phase_scale=active.targeted_dive_low_shot_phase_scale,
                mid_shot_phase_scale=active.targeted_dive_mid_shot_phase_scale,
                high_shot_phase_scale=active.targeted_dive_high_shot_phase_scale,
            ),
        )
    if active.combat_teacher_checkout is None:
        return GoalkeeperMJWarpBatch(
            asset_root=asset_root,
            locomotion_policy_path=locomotion_policy_path,
            device=device,
            config=world_config,
        )
    from rosclaw_soccer.training.goalkeeper_combat_mjwarp import (
        GoalkeeperCombatMJWarpBatch,
    )

    runtime_reach_atlas = None
    if active.runtime_task_space_reach_enabled:
        from rosclaw_soccer.training.goalkeeper_reach import (
            GoalkeeperReachConfig,
            build_g1_task_space_reach_atlas,
        )

        runtime_reach_atlas = build_g1_task_space_reach_atlas(
            asset_root,
            config=GoalkeeperReachConfig(
                damping=0.12,
                reach_gain=0.95,
                maximum_position_error_m=0.75,
                support_arm_scale=0.60,
                central_support_scale=0.95,
                residual_scale=min(
                    1.0,
                    world_config.residual_scale * world_config.arm_residual_scale_multiplier,
                ),
                arm_authority_scale=world_config.agility.arm_authority_scale,
            ),
        )
    return GoalkeeperCombatMJWarpBatch(
        asset_root=asset_root,
        locomotion_policy_path=locomotion_policy_path,
        teacher_checkout=Path(active.combat_teacher_checkout),
        teacher_checkpoint=Path(str(active.combat_teacher_checkpoint)),
        device=device,
        config=world_config,
        maximum_teacher_blend=active.maximum_combat_teacher_blend,
        intercept_conditioning_enabled=(active.combat_teacher_intercept_conditioning_enabled),
        mobility_option_enabled=active.mobility_option_enabled,
        mobility_option_config=GoalkeeperMobilityOptionConfig(
            lateral_command_limit=active.mobility_lateral_command_limit,
            recovery_command_limit=active.mobility_recovery_command_limit,
            residual_plasticity_scale=active.mobility_residual_plasticity_scale,
            waist_residual_plasticity_scale=(active.mobility_waist_residual_plasticity_scale),
            arm_residual_plasticity_scale=(active.mobility_arm_residual_plasticity_scale),
            teacher_lower_body_scale=active.mobility_teacher_lower_body_scale,
            teacher_waist_scale=active.mobility_teacher_waist_scale,
            teacher_arm_scale=active.mobility_teacher_arm_scale,
            predictive_teacher_gate_floor=(active.mobility_predictive_teacher_gate_floor),
            teacher_lower_body_target_step_rad=(active.mobility_teacher_lower_body_target_step_rad),
            teacher_lower_body_target_filter_fraction=(
                active.mobility_teacher_lower_body_target_filter_fraction
            ),
            teacher_waist_target_step_rad=active.mobility_teacher_waist_target_step_rad,
            teacher_waist_target_filter_fraction=(
                active.mobility_teacher_waist_target_filter_fraction
            ),
            teacher_arm_target_step_rad=active.mobility_teacher_arm_target_step_rad,
            teacher_arm_target_filter_fraction=(active.mobility_teacher_arm_target_filter_fraction),
            counter_rotation_enabled=active.mobility_counter_rotation_enabled,
            anticipatory_arm_reach_enabled=(active.mobility_anticipatory_arm_reach_enabled),
            predictive_teacher_warmstart_enabled=(
                active.mobility_predictive_teacher_warmstart_enabled
            ),
            teacher_recovery_latch_enabled=(active.mobility_teacher_recovery_latch_enabled),
            teacher_recovery_hold_sec=active.mobility_teacher_recovery_hold_sec,
            teacher_recovery_decay_sec=active.mobility_teacher_recovery_decay_sec,
            lateral_velocity_guard_enabled=(active.mobility_lateral_velocity_guard_enabled),
            substep_upper_body_guard_enabled=(active.mobility_substep_upper_body_guard_enabled),
            substep_upper_body_guard_onset_rad_s=(
                active.mobility_substep_upper_body_guard_onset_rad_s
            ),
            substep_upper_body_guard_ceiling_rad_s=(
                active.mobility_substep_upper_body_guard_ceiling_rad_s
            ),
            substep_upper_body_minimum_position_scale=(
                active.mobility_substep_upper_body_minimum_position_scale
            ),
        ),
        runtime_reach_atlas=runtime_reach_atlas,
        runtime_reach_blend=active.runtime_task_space_reach_blend,
    )


def _with_first_shot_release_override(
    world_config: Any,
    release_sec: float | None,
) -> Any:
    """Shift the first launch window without changing shot or episode duration."""

    if release_sec is None or release_sec == world_config.first_shot_release_sec:
        return world_config
    window_sec = world_config.first_shot_end_sec - world_config.first_shot_release_sec
    shifted_end = release_sec + window_sec
    if shifted_end >= world_config.second_shot_release_sec:
        raise ValueError("physics PPO first-shot release override overlaps the second shot")
    return replace(
        world_config,
        first_shot_release_sec=release_sec,
        first_shot_end_sec=shifted_end,
    )


def _with_episode_duration_override(
    world_config: Any,
    duration_sec: float | None,
) -> Any:
    """Extend the recovery tail while preserving all causal shot clocks."""

    if duration_sec is None or duration_sec == world_config.episode_duration_sec:
        return world_config
    if duration_sec <= world_config.second_shot_end_sec:
        raise ValueError("physics PPO episode must end after the final shot window")
    return replace(world_config, episode_duration_sec=duration_sec)


def _with_save_event_bonus_override(
    world_config: Any,
    true_save_bonus: float,
    hand_save_bonus: float | None,
) -> Any:
    """Align save and recovery incentives without changing exam physics."""

    resolved_hand = world_config.hand_save_bonus if hand_save_bonus is None else hand_save_bonus
    if (
        true_save_bonus == world_config.true_save_bonus
        and resolved_hand == world_config.hand_save_bonus
    ):
        return world_config
    return replace(
        world_config,
        true_save_bonus=true_save_bonus,
        hand_save_bonus=resolved_hand,
    )


def _with_root_angular_penalty_override(
    world_config: Any,
    penalty_scale: float | None,
    soft_limit_rad_s: float = 3.50,
    excess_penalty_scale: float = 0.0,
    flight_penalty_scale: float = 1.0,
) -> Any:
    """Expose mean and tail-sensitive differentiable stability costs."""

    resolved_penalty = (
        world_config.root_angular_speed_penalty_scale if penalty_scale is None else penalty_scale
    )
    if (
        resolved_penalty == world_config.root_angular_speed_penalty_scale
        and soft_limit_rad_s == world_config.root_angular_speed_soft_limit_rad_s
        and excess_penalty_scale == world_config.root_angular_speed_excess_penalty_scale
        and flight_penalty_scale == world_config.flight_root_angular_penalty_scale
    ):
        return world_config
    return replace(
        world_config,
        root_angular_speed_penalty_scale=resolved_penalty,
        root_angular_speed_soft_limit_rad_s=soft_limit_rad_s,
        root_angular_speed_excess_penalty_scale=excess_penalty_scale,
        flight_root_angular_penalty_scale=flight_penalty_scale,
    )


def _with_recovery_progress_override(
    world_config: Any,
    reward_scale: float,
    linear_speed_decay: float,
    angular_speed_decay: float,
) -> Any:
    """Install backend-neutral potential-based post-save stabilization shaping."""

    if (
        reward_scale == world_config.recovery_progress_reward_scale
        and linear_speed_decay == world_config.recovery_progress_linear_speed_decay
        and angular_speed_decay == world_config.recovery_progress_angular_speed_decay
    ):
        return world_config
    return replace(
        world_config,
        recovery_progress_reward_scale=reward_scale,
        recovery_progress_linear_speed_decay=linear_speed_decay,
        recovery_progress_angular_speed_decay=angular_speed_decay,
    )


def _mask_arm_only_gradients(model: Any, *, arm_action_start_index: int = 4) -> None:
    """Freeze representation/core actor while leaving arms and critic plastic."""

    action_size = int(model.log_std.shape[0])
    _mask_specialist_gradients(
        model,
        plastic_action_indices=tuple(range(arm_action_start_index, action_size)),
    )


def _mask_specialist_gradients(
    model: Any,
    *,
    plastic_action_indices: tuple[int, ...],
) -> None:
    """Freeze the actor trunk and every output outside a declared plastic set."""

    action_size = int(model.log_std.shape[0])
    plastic = set(plastic_action_indices)
    if not plastic or min(plastic) < 0 or max(plastic) >= action_size:
        raise ValueError("physics PPO specialist plastic-action set is invalid")

    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        if name.startswith("trunk."):
            gradient.zero_()
        elif name in {
            "actor.weight",
            "actor.bias",
            "specialist_adapter.weight",
            "specialist_adapter.bias",
            "log_std",
        }:
            frozen = [index for index in range(action_size) if index not in plastic]
            gradient[frozen] = 0.0


def _sample_exploration_action(
    *,
    torch: Any,
    normal_distribution: Any,
    mean: Any,
    log_std: Any,
    arm_only: bool,
    arm_action_start_index: int = 4,
    lower_body_and_arms: bool = False,
    lower_body_action_start_index: int = 1,
    lower_body_action_end_index: int = 13,
    lateral_drive_gate_only: bool = False,
    support_landing: bool = False,
    exploration_active_mask: Any | None = None,
) -> tuple[Any, Any]:
    """Sample only declared plastic channels when training a specialist."""

    if sum((arm_only, lower_body_and_arms, lateral_drive_gate_only, support_landing)) > 1:
        raise ValueError("physics PPO specialist exploration scopes are mutually exclusive")
    if exploration_active_mask is not None and tuple(exploration_active_mask.shape) != (
        mean.shape[0],
    ):
        raise ValueError("physics PPO exploration-active mask is misaligned")
    if arm_only or lower_body_and_arms or lateral_drive_gate_only or support_landing:
        if lateral_drive_gate_only:
            indices = torch.zeros(1, dtype=torch.long, device=mean.device)
        elif support_landing:
            # Gate + twelve lower-body + three waist channels.  The arm rows
            # remain bit-exact to the parent policy, so landing plasticity
            # cannot erase the already learned interception skill.
            indices = torch.arange(0, arm_action_start_index, device=mean.device)
        elif lower_body_and_arms:
            indices = torch.cat(
                (
                    torch.arange(
                        lower_body_action_start_index,
                        lower_body_action_end_index,
                        device=mean.device,
                    ),
                    torch.arange(arm_action_start_index, mean.shape[1], device=mean.device),
                )
            )
        else:
            indices = torch.arange(arm_action_start_index, mean.shape[1], device=mean.device)
        distribution = normal_distribution(
            mean[:, indices],
            torch.exp(log_std[indices]).expand_as(mean[:, indices]),
        )
        raw_action = mean.clone()
        raw_action[:, indices] = distribution.sample()
        if exploration_active_mask is not None:
            raw_action = torch.where(
                exploration_active_mask.to(torch.bool).unsqueeze(1),
                raw_action,
                mean,
            )
        log_probability = distribution.log_prob(raw_action[:, indices]).sum(dim=1)
        return raw_action, log_probability
    distribution = normal_distribution(mean, torch.exp(log_std).expand_as(mean))
    raw_action = distribution.sample()
    return raw_action, distribution.log_prob(raw_action).sum(dim=1)


def _successful_trajectory_replay_mask(
    *,
    torch: Any,
    active_steps: Any,
    first_save: Any,
    quarantined: Any,
    maximum_root_angular_speed_rad_s: Any,
    angular_speed_ceiling_rad_s: float,
) -> tuple[Any, Any]:
    """Select only causally active transitions from safe successful episodes.

    This is self-imitation, not a privileged action oracle: labels are the
    actor's own sampled actions and become eligible only after MuJoCo reports a
    real save.  Unsafe/fallen trajectories are never replayed, which makes the
    stability side of stability-plasticity explicit.
    """

    if active_steps.ndim != 2:
        raise ValueError("successful-trajectory active-step mask must be rank two")
    environment_count = active_steps.shape[1]
    vectors = (first_save, quarantined, maximum_root_angular_speed_rad_s)
    if any(tuple(value.shape) != (environment_count,) for value in vectors):
        raise ValueError("successful-trajectory episode vectors are misaligned")
    eligible = first_save.to(torch.bool) & ~quarantined.to(torch.bool)
    eligible &= torch.isfinite(maximum_root_angular_speed_rad_s)
    eligible &= maximum_root_angular_speed_rad_s <= angular_speed_ceiling_rad_s
    return active_steps.to(torch.bool) & eligible.unsqueeze(0), eligible


def _append_successful_trajectory_memory(
    *,
    torch: Any,
    observations: Any,
    actions: Any,
    replay_mask: Any,
    height_strata: Any,
    memory_observations: Any | None,
    memory_actions: Any | None,
    memory_height_strata: Any | None,
    capacity_per_stratum: int,
    stratum_count: int = 3,
) -> tuple[Any, Any, Any]:
    """Append safe causal rows to a bounded, stratified muscle memory.

    Memory is deliberately stratified instead of being one FIFO.  Otherwise
    abundant mid-height successes can overwrite the rare low/high examples
    that continual learning is supposed to retain.
    """

    if observations.ndim != 3 or actions.ndim != 3 or replay_mask.ndim != 2:
        raise ValueError("successful-trajectory memory rollout tensors are misaligned")
    if observations.shape[:2] != actions.shape[:2] or observations.shape[:2] != replay_mask.shape:
        raise ValueError("successful-trajectory memory rollout shapes are misaligned")
    if tuple(height_strata.shape) != (observations.shape[1],):
        raise ValueError("successful-trajectory memory height strata are misaligned")
    if capacity_per_stratum <= 0 or not 1 <= stratum_count <= 64:
        raise ValueError("successful-trajectory memory capacity must be positive")
    flat_mask = replay_mask.to(torch.bool).flatten()
    expanded_strata = height_strata.unsqueeze(0).expand_as(replay_mask).flatten()
    new_observations = observations.flatten(0, 1)[flat_mask].detach()
    new_actions = actions.flatten(0, 1)[flat_mask].detach()
    new_strata = expanded_strata[flat_mask].to(torch.long).detach()
    return _append_successful_memory_rows(
        torch=torch,
        new_observations=new_observations,
        new_actions=new_actions,
        new_strata=new_strata,
        memory_observations=memory_observations,
        memory_actions=memory_actions,
        memory_height_strata=memory_height_strata,
        capacity_per_stratum=capacity_per_stratum,
        stratum_count=stratum_count,
    )


def _append_successful_memory_rows(
    *,
    torch: Any,
    new_observations: Any,
    new_actions: Any,
    new_strata: Any,
    memory_observations: Any | None,
    memory_actions: Any | None,
    memory_height_strata: Any | None,
    capacity_per_stratum: int,
    stratum_count: int = 3,
) -> tuple[Any, Any, Any]:
    """Append already-qualified local or globally gathered memory rows."""

    if (
        new_observations.ndim != 2
        or new_actions.ndim != 2
        or new_strata.ndim != 1
        or new_observations.shape[0] != new_actions.shape[0]
        or new_observations.shape[0] != new_strata.shape[0]
    ):
        raise ValueError("successful-trajectory new memory rows are invalid")
    if capacity_per_stratum <= 0 or not 1 <= stratum_count <= 64:
        raise ValueError("successful-trajectory memory capacity must be positive")
    supplied = (
        memory_observations is not None,
        memory_actions is not None,
        memory_height_strata is not None,
    )
    if any(supplied) and not all(supplied):
        raise ValueError("successful-trajectory memory tensors must be supplied together")
    if memory_observations is not None:
        if memory_actions is None or memory_height_strata is None:
            raise RuntimeError("successful-trajectory memory validation drifted")
        if (
            memory_observations.ndim != 2
            or memory_actions.ndim != 2
            or memory_height_strata.ndim != 1
            or memory_observations.shape[0] != memory_actions.shape[0]
            or memory_observations.shape[0] != memory_height_strata.shape[0]
            or memory_observations.shape[1] != new_observations.shape[1]
            or memory_actions.shape[1] != new_actions.shape[1]
        ):
            raise ValueError("successful-trajectory memory tensors are invalid")
        new_observations = torch.cat((memory_observations, new_observations), dim=0)
        new_actions = torch.cat((memory_actions, new_actions), dim=0)
        new_strata = torch.cat((memory_height_strata, new_strata), dim=0)
    if int(new_strata.numel()) > 0 and (
        int(new_strata.min()) < 0 or int(new_strata.max()) >= stratum_count
    ):
        raise ValueError("successful-trajectory memory stratum is out of range")

    retained: list[Any] = []
    for stratum in range(stratum_count):
        indices = torch.nonzero(new_strata == stratum, as_tuple=False).flatten()
        retained.append(indices[-capacity_per_stratum:])
    retained_indices = (
        torch.cat(retained)
        if retained
        else torch.empty(0, dtype=torch.long, device=new_strata.device)
    )
    return (
        new_observations[retained_indices],
        new_actions[retained_indices],
        new_strata[retained_indices],
    )


def _all_gather_successful_memory_rows(
    *,
    torch: Any,
    dist: Any,
    observations: Any,
    actions: Any,
    strata: Any,
    world_size: int,
) -> tuple[Any, Any, Any]:
    """Replicate variable-length new safe rows across every DDP rank.

    Only newly qualified rows are exchanged, not the whole long-lived memory.
    Rank-ordered concatenation gives every worker the same bounded replay set
    while avoiding a privileged label or cross-iteration synchronization leak.
    """

    if world_size <= 1:
        return observations, actions, strata
    if (
        observations.ndim != 2
        or actions.ndim != 2
        or strata.ndim != 1
        or observations.shape[0] != actions.shape[0]
        or observations.shape[0] != strata.shape[0]
    ):
        raise ValueError("DDP successful-memory rows are invalid")
    local_count = torch.tensor(observations.shape[0], dtype=torch.long, device=observations.device)
    gathered_counts = [torch.zeros_like(local_count) for _ in range(world_size)]
    dist.all_gather(gathered_counts, local_count)
    counts = [int(value.item()) for value in gathered_counts]
    maximum_count = max(counts)
    if maximum_count == 0:
        return observations, actions, strata

    def gather_padded(rows: Any) -> list[Any]:
        padded = torch.zeros((maximum_count, *rows.shape[1:]), dtype=rows.dtype, device=rows.device)
        padded[: rows.shape[0]] = rows
        gathered = [torch.zeros_like(padded) for _ in range(world_size)]
        dist.all_gather(gathered, padded)
        return gathered

    gathered_observations = gather_padded(observations)
    gathered_actions = gather_padded(actions)
    gathered_strata = gather_padded(strata)
    return (
        torch.cat(
            [rows[:count] for rows, count in zip(gathered_observations, counts, strict=True)]
        ),
        torch.cat([rows[:count] for rows, count in zip(gathered_actions, counts, strict=True)]),
        torch.cat([rows[:count] for rows, count in zip(gathered_strata, counts, strict=True)]),
    )


def _balanced_successful_memory_sample_indices(
    *,
    torch: Any,
    height_strata: Any,
    sample_count: int,
    stratum_count: int = 3,
) -> Any:
    """Sample each represented muscle-memory stratum equally."""

    if height_strata.ndim != 1 or sample_count <= 0 or not 1 <= stratum_count <= 64:
        raise ValueError("successful-trajectory memory sampler inputs are invalid")
    represented = [
        torch.nonzero(height_strata == stratum, as_tuple=False).flatten()
        for stratum in range(stratum_count)
    ]
    represented = [indices for indices in represented if int(indices.numel()) > 0]
    if not represented:
        raise ValueError("successful-trajectory memory sampler requires stored rows")
    base, remainder = divmod(sample_count, len(represented))
    sampled: list[Any] = []
    for index, candidates in enumerate(represented):
        count = base + int(index < remainder)
        if count > 0:
            selected = torch.randint(
                0,
                int(candidates.numel()),
                (count,),
                device=height_strata.device,
            )
            sampled.append(candidates[selected])
    combined = torch.cat(sampled)
    return combined[torch.randperm(combined.shape[0], device=combined.device)]


def _required_successful_memory_strata(
    *,
    hard_shot_fraction: float,
    hard_shot_height_mode: str,
    hard_shot_side_mode: str,
) -> tuple[int, ...]:
    """Return every curriculum stratum that self-imitation must cover.

    PPO may learn from failures immediately, but replaying successful actions
    before the active curriculum is represented on both sides can amplify the
    already-strong side and deepen asymmetry.  Mixed/non-hard curricula remain
    conservative and require all height-by-side strata.
    """

    height_index = {"low": 0, "mid": 1, "high": 2}
    side_index = {"negative": 0, "positive": 1}
    heights: tuple[int, ...]
    if hard_shot_fraction < 1.0 or hard_shot_height_mode == "balanced":
        heights = (0, 1, 2)
    else:
        heights = (height_index[hard_shot_height_mode],)
    sides: tuple[int, ...]
    if hard_shot_fraction < 1.0 or hard_shot_side_mode == "balanced":
        sides = (0, 1)
    else:
        sides = (side_index[hard_shot_side_mode],)
    return tuple(2 * height + side for height in heights for side in sides)


def _successful_memory_covers_strata(
    *, torch: Any, height_strata: Any, required_strata: tuple[int, ...]
) -> bool:
    """Fail closed until every curriculum-required muscle-memory cell exists."""

    if height_strata.ndim != 1 or not required_strata:
        raise ValueError("successful-trajectory memory coverage inputs are invalid")
    return all(bool(torch.any(height_strata == stratum)) for stratum in required_strata)


def _successful_memory_has_episode_diversity(
    *, episode_counts: Any, required_strata: tuple[int, ...], minimum_episodes: int
) -> bool:
    """Require distinct safe episodes, not many frames from one lucky recovery."""

    if (
        episode_counts.ndim != 1
        or not required_strata
        or minimum_episodes <= 0
        or max(required_strata) >= int(episode_counts.shape[0])
    ):
        raise ValueError("successful-trajectory episode diversity inputs are invalid")
    return all(
        int(episode_counts[stratum].item()) >= minimum_episodes for stratum in required_strata
    )


def _successful_memory_replay_strength(
    *,
    episode_counts: Any,
    required_strata: tuple[int, ...],
    minimum_episodes: int,
    full_strength_episodes: int,
) -> float:
    """Ramp self-imitation only as the rarest required stratum matures.

    A binary replay switch made the first handful of lucky recoveries dominate
    an update.  The readiness floor remains fail-closed, while this multiplier
    grows from a conservative fraction to one as independent episode support
    accumulates.  ``full_strength_episodes=0`` preserves the legacy immediate
    full-strength behaviour after the readiness floor.
    """

    if (
        episode_counts.ndim != 1
        or not required_strata
        or minimum_episodes <= 0
        or full_strength_episodes < 0
        or max(required_strata) >= int(episode_counts.shape[0])
    ):
        raise ValueError("successful-trajectory replay-strength inputs are invalid")
    rarest_count = min(int(episode_counts[stratum].item()) for stratum in required_strata)
    if rarest_count < minimum_episodes:
        return 0.0
    resolved_full_strength = max(minimum_episodes, full_strength_episodes)
    return min(1.0, rarest_count / resolved_full_strength)


def _shrink_successful_action_innovation(
    *, torch: Any, policy_mean: Any, sampled_action: Any, innovation_scale: float
) -> Any:
    """Retain a successful exploration direction without memorising all noise."""

    if (
        policy_mean.ndim != 3
        or sampled_action.ndim != 3
        or policy_mean.shape != sampled_action.shape
        or not math.isfinite(innovation_scale)
        or not 0.0 <= innovation_scale <= 1.0
        or not bool(torch.all(torch.isfinite(policy_mean)))
        or not bool(torch.all(torch.isfinite(sampled_action)))
    ):
        raise ValueError("successful-trajectory action innovation inputs are invalid")
    return (
        policy_mean.detach() + innovation_scale * (sampled_action.detach() - policy_mean.detach())
    ).to(dtype=sampled_action.dtype, device=sampled_action.device)


def _mirror_goalkeeper_actor_action(*, torch: Any, action: Any) -> Any:
    """Mirror gate plus 29 G1 residuals about the sagittal plane."""

    if action.ndim != 2 or action.shape[1] != 30:
        raise ValueError("goalkeeper actor mirror expects rank-two 30D actions")
    order = torch.as_tensor(
        (0, *(index + 1 for index in _MIRROR_ORDER)),
        dtype=torch.long,
        device=action.device,
    )
    sign = torch.as_tensor((1.0, *_MIRROR_SIGN), dtype=action.dtype, device=action.device)
    return action[:, order] * sign


def _mirror_goalkeeper_actor_observation(
    *,
    torch: Any,
    observation: Any,
    shot_intent_cue_enabled: bool,
    actor_contact_support_side_enabled: bool,
    actor_recovery_context_enabled: bool = False,
) -> Any:
    """Mirror the causal 29-DoF goalkeeper observation exactly once.

    Polar lateral components change sign, sagittal reflection flips the x/z
    components of root angular velocity, bilateral joints exchange with their
    declared G1 signs, and foot contacts exchange sides.  Applying this
    transform twice must recover the original tensor bit-for-bit.
    """

    expected_size = (
        86
        + 3 * int(shot_intent_cue_enabled)
        + 3 * int(actor_contact_support_side_enabled)
        + 4 * int(actor_recovery_context_enabled)
    )
    if observation.ndim != 2 or observation.shape[1] != expected_size:
        raise ValueError(f"goalkeeper actor mirror expects rank-two {expected_size}D observations")
    mirrored = observation.clone()
    mirrored[:, (1, 4, 7, 10, 13, 15, 17)] *= -1.0
    joint_order = torch.as_tensor(
        tuple(index - 12 for index in _MIRROR_ORDER[12:]),
        dtype=torch.long,
        device=observation.device,
    )
    joint_sign = torch.as_tensor(
        _MIRROR_SIGN[12:], dtype=observation.dtype, device=observation.device
    )
    mirrored[:, 18:35] = observation[:, 18:35][:, joint_order] * joint_sign
    mirrored[:, 35:52] = observation[:, 35:52][:, joint_order] * joint_sign
    mirrored[:, 52:82] = _mirror_goalkeeper_actor_action(torch=torch, action=observation[:, 52:82])
    tail = 82
    if actor_contact_support_side_enabled:
        mirrored[:, tail] = -observation[:, tail]
        mirrored[:, tail + 1] = observation[:, tail + 2]
        mirrored[:, tail + 2] = observation[:, tail + 1]
        tail += 3
    if actor_recovery_context_enabled:
        # Save latch and recovery age are invariant; world-lateral pelvis
        # displacement and capture error reflect about the sagittal plane.
        mirrored[:, tail + 2 : tail + 4] *= -1.0
        tail += 4
    if shot_intent_cue_enabled:
        mirrored[:, tail] = -observation[:, tail]
    return mirrored


def _deterministic_candidate_rollout(
    *,
    torch: Any,
    model: Any,
    environment: Any,
    seed: int,
) -> tuple[Any, Any]:
    """Evaluate the exact mean policy whose weights may become a candidate."""

    observation = environment.reset(seed=seed)
    cumulative_reward = torch.zeros(environment.count, device=environment.device)
    cumulative_task_motion_reward = torch.zeros_like(cumulative_reward)
    cumulative_recovery_progress_reward = torch.zeros_like(cumulative_reward)
    cumulative_reach_reward = torch.zeros_like(cumulative_reward)
    cumulative_bimanual_reach_reward = torch.zeros_like(cumulative_reward)
    cumulative_upright_reward = torch.zeros_like(cumulative_reward)
    cumulative_smoothness_penalty = torch.zeros_like(cumulative_reward)
    cumulative_effort_penalty = torch.zeros_like(cumulative_reward)
    cumulative_event_bonus = torch.zeros_like(cumulative_reward)
    cumulative_safety_penalty = torch.zeros_like(cumulative_reward)
    cumulative_nonfinite_override = torch.zeros_like(cumulative_reward)
    cumulative_action_rate_penalty = torch.zeros_like(cumulative_reward)
    cumulative_joint_acceleration_penalty = torch.zeros_like(cumulative_reward)
    cumulative_root_linear_speed_penalty = torch.zeros_like(cumulative_reward)
    cumulative_root_angular_speed_penalty = torch.zeros_like(cumulative_reward)
    cumulative_root_angular_excess_penalty = torch.zeros_like(cumulative_reward)
    cumulative_action_magnitude_penalty = torch.zeros_like(cumulative_reward)
    alive = torch.ones(environment.count, dtype=torch.bool, device=environment.device)
    for _ in range(environment.config.episode_steps):
        with torch.inference_mode():
            mean, _, _ = model(observation)
            action = torch.tanh(mean)
        observation, reward, done, info = environment.step(action)
        cumulative_reward += torch.where(alive, reward, torch.zeros_like(reward))
        cumulative_task_motion_reward += torch.where(
            alive,
            info["task_motion"],
            torch.zeros_like(info["task_motion"]),
        )
        cumulative_recovery_progress_reward += torch.where(
            alive,
            info["recovery_progress"],
            torch.zeros_like(info["recovery_progress"]),
        )
        for cumulative, key in (
            (cumulative_reach_reward, "reach"),
            (cumulative_bimanual_reach_reward, "bimanual_reach"),
            (cumulative_upright_reward, "upright"),
            (cumulative_smoothness_penalty, "smoothness_penalty"),
            (cumulative_effort_penalty, "effort_penalty"),
            (cumulative_event_bonus, "event_bonus"),
            (cumulative_safety_penalty, "safety_penalty"),
            (cumulative_nonfinite_override, "nonfinite_override"),
            (cumulative_action_rate_penalty, "action_rate_penalty"),
            (cumulative_joint_acceleration_penalty, "joint_acceleration_penalty"),
            (cumulative_root_linear_speed_penalty, "root_linear_speed_penalty"),
            (cumulative_root_angular_speed_penalty, "root_angular_speed_penalty"),
            (cumulative_root_angular_excess_penalty, "root_angular_excess_penalty"),
            (cumulative_action_magnitude_penalty, "action_magnitude_penalty"),
        ):
            cumulative += torch.where(alive, info[key], torch.zeros_like(info[key]))
        alive &= ~done
    if not environment.finite_state():
        raise FloatingPointError("deterministic goalkeeper candidate produced non-finite state")
    bimanual_reach = environment.task._bimanual_reach_steps.sum().to(torch.float32)
    bimanual_reach /= torch.clamp(environment.task._active_flight_steps.sum(), min=1).to(
        torch.float32
    )
    stratum_saves, stratum_counts = _first_save_stratum_counts(torch, environment)
    stratum_failures = _height_stratum_event_counts(
        torch,
        environment,
        environment.task.phase == 7,
    )
    stratum_quarantines = _height_stratum_event_counts(
        torch,
        environment,
        environment._quarantined,
    )
    qualified_save = environment.task.first_save & (environment.task.phase != 7)
    failed = environment.task.phase == 7
    stable_miss = ~environment.task.first_save & ~failed
    side_saves = _side_event_counts(torch, environment, environment.task.first_save)
    side_qualified_saves = _side_event_counts(torch, environment, qualified_save)
    side_counts = _side_event_counts(
        torch,
        environment,
        torch.ones_like(environment.task.first_save),
    )
    side_failures = _side_event_counts(torch, environment, failed)
    metrics = torch.stack(
        (
            cumulative_reward.mean(),
            environment.task.first_save.to(torch.float32).mean(),
            environment.task.first_hand_save.to(torch.float32).mean(),
            environment.task.recovered_after_first.to(torch.float32).mean(),
            environment.task.second_save.to(torch.float32).mean(),
            environment.task.second_hand_save.to(torch.float32).mean(),
            (environment.task.phase == 7).to(torch.float32).mean(),
            environment._maximum_lateral_displacement.mean(),
            environment._maximum_lateral_speed.mean(),
            environment._maximum_hand_displacement.mean(),
            environment._maximum_hand_speed.mean(),
            environment._maximum_root_angular_speed.mean(),
            environment._minimum_upper_body_authority.mean(),
            environment._second_release_lateral_error.mean(),
            bimanual_reach,
            environment.task.second_attempt_save.to(torch.float32).mean(),
            environment.task.second_attempt_hand_save.to(torch.float32).mean(),
            *stratum_saves,
            *stratum_counts,
            environment._quarantined.to(torch.float32).mean(),
            environment._nonfinite_quarantine_latched.to(torch.float32).mean(),
            *stratum_failures,
            *stratum_quarantines,
            environment._minimum_hand_target_distance.mean(),
            cumulative_task_motion_reward.mean(),
            environment._first_decisive_pelvis_lateral_error.mean(),
            environment._first_decisive_hand_intercept_distance.mean(),
            (
                environment._maximum_root_angular_speed
                > environment.config.root_angular_speed_soft_limit_rad_s
            )
            .to(torch.float32)
            .mean(),
            (environment._maximum_root_angular_speed > 3.50).to(torch.float32).mean(),
            cumulative_recovery_progress_reward.mean(),
            (environment.task.first_save & (environment.task.phase != 7)).to(torch.float32).mean(),
            cumulative_reach_reward.mean(),
            cumulative_bimanual_reach_reward.mean(),
            cumulative_upright_reward.mean(),
            cumulative_smoothness_penalty.mean(),
            cumulative_effort_penalty.mean(),
            cumulative_event_bonus.mean(),
            cumulative_safety_penalty.mean(),
            cumulative_nonfinite_override.mean(),
            cumulative_action_rate_penalty.mean(),
            cumulative_joint_acceleration_penalty.mean(),
            cumulative_root_linear_speed_penalty.mean(),
            cumulative_root_angular_speed_penalty.mean(),
            cumulative_root_angular_excess_penalty.mean(),
            cumulative_action_magnitude_penalty.mean(),
            cumulative_reward[qualified_save].sum(),
            qualified_save.to(torch.float32).sum(),
            cumulative_reward[failed].sum(),
            failed.to(torch.float32).sum(),
            cumulative_reward[stable_miss].sum(),
            stable_miss.to(torch.float32).sum(),
            *side_saves,
            *side_qualified_saves,
            *side_counts,
            *side_failures,
        )
    )
    return metrics, environment._maximum_root_angular_speed.max()


def _held_out_selection_seed(*, random_seed: int, rank: int) -> int:
    """Return one stable held-out physics seed per DDP rank.

    Candidate comparisons must use the same exam across iterations.  Varying
    this seed by iteration makes curriculum sampling noise look like policy
    improvement and can select an easier shot batch instead of better weights.
    """

    if random_seed < 0 or rank < 0:
        raise ValueError("held-out selection seed inputs must be non-negative")
    return random_seed + 700_001 + rank


def _reward_accounting_residual(metrics: Mapping[str, Any]) -> float:
    """Return total reward minus the reported signed component ledger."""

    reconstructed = (
        float(metrics["mean_reach_reward"])
        + float(metrics["mean_bimanual_reach_reward"])
        + float(metrics["mean_task_motion_reward"])
        + float(metrics["mean_upright_reward"])
        + float(metrics["mean_recovery_progress_reward"])
        + float(metrics["mean_event_bonus"])
        - float(metrics["mean_smoothness_penalty"])
        - float(metrics["mean_effort_penalty"])
        - float(metrics["mean_safety_penalty"])
        + float(metrics["mean_nonfinite_override"])
    )
    return float(metrics["mean_episode_reward"]) - reconstructed


def _first_save_stratum_counts(
    torch: Any, environment: Any
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Return exact low/mid/high first-save numerators and denominators."""

    return (
        _height_stratum_event_counts(torch, environment, environment.task.first_save),
        _height_stratum_event_counts(
            torch,
            environment,
            torch.ones_like(environment.task.first_save),
        ),
    )


def _height_stratum_event_counts(
    torch: Any,
    environment: Any,
    event: Any,
) -> tuple[Any, ...]:
    """Count a boolean event in the exact low/mid/high first-shot strata."""

    height = environment._target_one[:, 2]
    event_float = event.to(torch.float32)
    masks = (height < 0.60, (height >= 0.60) & (height < 1.10), height >= 1.10)
    return tuple((event_float * mask.to(torch.float32)).sum() for mask in masks)


def _side_event_counts(torch: Any, environment: Any, event: Any) -> tuple[Any, Any]:
    """Count an event for negative and positive first-shot far corners."""

    lateral_target = environment._target_one[:, 1]
    event_float = event.to(torch.float32)
    return (
        (event_float * (lateral_target < 0.0).to(torch.float32)).sum(),
        (event_float * (lateral_target >= 0.0).to(torch.float32)).sum(),
    )


def _side_rates(metrics: Any, *, event_offset: int, count_offset: int) -> tuple[float, float]:
    rates = []
    for index in range(2):
        count = float(metrics[count_offset + index])
        rates.append(0.0 if count <= 0.0 else float(metrics[event_offset + index]) / count)
    return rates[0], rates[1]


def _stratum_rates(
    metrics: Any, *, save_offset: int = 17, count_offset: int = 20
) -> tuple[float, ...]:
    rates = []
    for index in range(3):
        count = float(metrics[count_offset + index])
        # An unobserved stratum carries no evidence.  Treating it as the
        # overall rate could let a high-only specialist masquerade as a
        # balanced low/mid/high policy during candidate selection.
        rates.append(0.0 if count <= 0.0 else float(metrics[save_offset + index]) / count)
    return tuple(rates)


def _stratum_balance_score(rates: tuple[float, ...]) -> float:
    """Prefer the weakest height band while retaining a small mean tie-break."""

    return min(rates) + 0.25 * sum(rates) / len(rates)


def _candidate_truth_key(candidate: dict[str, Any]) -> tuple[float, ...]:
    """Rank task truth before reward or anatomical proxy metrics.

    A later policy must never overwrite a policy that saved more real shots
    merely because it moved farther or collected more shaped reward.  Hand
    saves are ordered after all-contact saves because both are measured
    outcomes; the anatomical score is only the final tie-break.
    """

    return (
        float(
            candidate.get(
                "minimum_seed_qualified_first_save_side_rate",
                candidate.get(
                    "minimum_qualified_first_save_side_rate",
                    candidate.get("qualified_first_save_rate", 0.0),
                ),
            )
        ),
        float(
            candidate.get(
                "minimum_seed_qualified_first_save_rate",
                candidate.get("qualified_first_save_rate", 0.0),
            )
        ),
        float(candidate.get("minimum_seed_first_save_rate", candidate["first_save_rate"])),
        float(candidate["first_hand_save_rate"]),
        float(candidate["second_attempt_save_rate"]),
        float(candidate["second_attempt_hand_save_rate"]),
        float(candidate["second_save_rate"]),
        float(candidate["second_hand_save_rate"]),
        float(candidate["anatomical_selection_score"]),
    )


def _exploration_truth_key(
    candidate: dict[str, Any], *, save_first: bool = False
) -> tuple[float, ...]:
    """Retain the most useful unsafe learning state without calling it a candidate."""

    if save_first:
        # A goalkeeper may deliberately leave its feet.  This development-only
        # ordering keeps the weakest held-out seed, height band, and goal side
        # ahead of posture quality, so a real save cannot be discarded merely
        # because its landing still needs a separate get-up curriculum.
        return (
            float(candidate.get("minimum_seed_first_save_rate", candidate["first_save_rate"])),
            float(candidate.get("minimum_first_save_stratum_rate", 0.0)),
            float(candidate.get("minimum_first_save_side_rate", 0.0)),
            float(candidate["first_save_rate"]),
            float(candidate["first_hand_save_rate"]),
            float(candidate.get("recovery_rate", 0.0)),
            -float(candidate.get("maximum_seed_failed_rate", candidate["failed_rate"])),
            -float(candidate.get("nonfinite_quarantine_rate", 0.0)),
            float(candidate["anatomical_selection_score"]),
        )

    return (
        float(
            candidate.get(
                "minimum_seed_qualified_first_save_side_rate",
                candidate.get(
                    "minimum_qualified_first_save_side_rate",
                    candidate.get("qualified_first_save_rate", 0.0),
                ),
            )
        ),
        float(
            candidate.get(
                "minimum_seed_qualified_first_save_rate",
                candidate.get("qualified_first_save_rate", 0.0),
            )
        ),
        -float(candidate.get("maximum_seed_failed_rate", candidate["failed_rate"])),
        float(candidate.get("minimum_seed_first_save_rate", candidate["first_save_rate"])),
        float(candidate["first_hand_save_rate"]),
        -float(candidate["mean_maximum_root_angular_speed_rad_s"]),
        float(candidate["anatomical_selection_score"]),
    )


def _safe_continuation_truth_key(
    candidate: dict[str, Any], *, maximum_root_angular_speed_rad_s: float
) -> tuple[float, ...] | None:
    """Rank a rollback-safe learning anchor without granting promotion status."""

    if (
        float(candidate.get("maximum_seed_failed_rate", candidate["failed_rate"])) != 0.0
        or float(candidate.get("quarantined_rate", 0.0)) != 0.0
        or float(candidate.get("nonfinite_quarantine_rate", 0.0)) != 0.0
        or float(candidate["maximum_root_angular_speed_rad_s"]) > maximum_root_angular_speed_rad_s
    ):
        return None
    return (
        float(
            candidate.get(
                "minimum_seed_qualified_first_save_side_rate",
                candidate.get(
                    "minimum_qualified_first_save_side_rate",
                    candidate.get("qualified_first_save_rate", 0.0),
                ),
            )
        ),
        float(candidate.get("minimum_first_save_stratum_rate", 0.0)),
        float(candidate["first_save_rate"]),
        float(candidate["first_hand_save_rate"]),
        float(candidate["second_attempt_save_rate"]),
        float(candidate["second_attempt_hand_save_rate"]),
        -float(candidate["maximum_root_angular_speed_rad_s"]),
        float(candidate["anatomical_selection_score"]),
    )


def _candidate_is_selectable(
    candidate: dict[str, Any],
    *,
    config: GoalkeeperPhysicsPPOConfig,
    best_truth_key: tuple[float, ...] | None,
) -> bool:
    """Apply fail-closed safety/effectiveness gates and truth-first ranking."""

    if (
        float(candidate.get("maximum_seed_failed_rate", candidate["failed_rate"])) != 0.0
        or float(candidate.get("quarantined_rate", 0.0)) != 0.0
        or float(candidate.get("nonfinite_quarantine_rate", 0.0)) != 0.0
        or float(candidate["maximum_root_angular_speed_rad_s"])
        > config.maximum_exploration_root_angular_speed_for_selection
        or float(candidate["mean_maximum_hand_displacement_m"])
        < config.minimum_hand_displacement_for_selection_m
        or float(candidate.get("minimum_seed_first_save_rate", candidate["first_save_rate"]))
        < config.minimum_first_save_rate_for_selection
        or float(candidate.get("minimum_first_save_stratum_rate", 0.0))
        < config.minimum_first_save_stratum_rate_for_selection
    ):
        return False
    truth_key = _candidate_truth_key(candidate)
    return best_truth_key is None or truth_key > best_truth_key


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _atomic_torch_save(*, torch: Any, path: Path, payload: dict[str, Any]) -> None:
    """Persist a resumable checkpoint without exposing a partial zip archive."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _selection_outcome(
    *,
    checkpoint: Path | None,
    best_reward: float,
    best_selection_score: float,
    best_rollout_iteration: int | None,
) -> dict[str, Any]:
    """Encode both accepted and fully rejected searches without NaN sentinels."""

    if checkpoint is None:
        return {
            "best_mean_episode_reward": None,
            "best_anatomical_selection_score": None,
            "best_rollout_iteration": None,
            "candidate_checkpoint": None,
            "candidate_checkpoint_bytes": 0,
            "candidate_available": False,
            "rejection_reasons": [
                "no rollout satisfied the declared zero-failure, root-angular-speed, "
                "hand-displacement, overall/stratum first-save, and truth-first selection gate"
            ],
            "promotion_status": "REJECTED_NO_SAFE_CANDIDATE",
        }
    if best_rollout_iteration is None:
        raise ValueError("a goalkeeper candidate requires its rollout iteration")
    return {
        "best_mean_episode_reward": best_reward,
        "best_anatomical_selection_score": best_selection_score,
        "best_rollout_iteration": best_rollout_iteration,
        "candidate_checkpoint": checkpoint.name,
        "candidate_checkpoint_bytes": checkpoint.stat().st_size,
        "candidate_available": True,
        "rejection_reasons": [],
        "promotion_status": "CANDIDATE_PENDING_CPU_MUJOCO_EXAM",
    }


def run_goalkeeper_physics_ppo(
    *,
    asset_root: Path,
    locomotion_policy_path: Path,
    output_dir: Path,
    config: GoalkeeperPhysicsPPOConfig | None = None,
    motion_library_path: Path | None = None,
    motion_dataset_root: Path | None = None,
) -> dict[str, Any] | None:
    """Train one synchronized DDP candidate; only rank zero writes evidence."""

    import torch
    import torch.distributed as dist
    from torch import nn
    from torch.distributions import Normal
    from torch.nn.parallel import DistributedDataParallel

    active = config or GoalkeeperPhysicsPPOConfig()
    if (motion_library_path is None) != (motion_dataset_root is None):
        raise ValueError("physics PPO motion prior requires library and dataset together")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(active.random_seed + rank)
    world_config = goalkeeper_world_config(
        difficulty_profile=active.shot_difficulty_profile,  # type: ignore[arg-type]
        environment_count=active.environments_per_rank,
        second_shot_probability=active.training_second_shot_probability,
        shot_intent_cue_enabled=active.shot_intent_cue_enabled,
        hard_shot_fraction=active.training_hard_shot_fraction,
        hard_shot_height_mode=active.training_hard_shot_height_mode,  # type: ignore[arg-type]
        hard_shot_side_mode=active.training_hard_shot_side_mode,  # type: ignore[arg-type]
        hard_shot_flight_time_range_sec=(active.training_hard_shot_flight_time_range_sec),
        reach_reward_semantics=active.training_reach_reward_semantics,
        hard_height_reach_reward_scale=(active.training_hard_height_reach_reward_scale),
        hard_height_reach_threshold_m=(active.training_hard_height_reach_threshold_m),
        hard_height_reach_distance_decay=(active.training_hard_height_reach_distance_decay),
        task_motion_reward_scale=active.training_task_motion_reward_scale,
        recovery_event_bonus=active.training_recovery_event_bonus,
        unsafe_penalty=active.training_unsafe_penalty,
        save_then_unsafe_penalty=active.training_save_then_unsafe_penalty,
    )
    world_config = _with_first_shot_release_override(
        world_config,
        active.training_first_shot_release_sec,
    )
    world_config = _with_episode_duration_override(
        world_config,
        active.training_episode_duration_sec,
    )
    world_config = _with_save_event_bonus_override(
        world_config,
        active.training_true_save_bonus,
        active.training_hand_save_bonus,
    )
    world_config = _with_root_angular_penalty_override(
        world_config,
        active.training_root_angular_speed_penalty_scale,
        active.training_root_angular_speed_soft_limit_rad_s,
        active.training_root_angular_speed_excess_penalty_scale,
        active.training_flight_root_angular_penalty_scale,
    )
    world_config = _with_recovery_progress_override(
        world_config,
        active.training_recovery_progress_reward_scale,
        active.training_recovery_progress_linear_speed_decay,
        active.training_recovery_progress_angular_speed_decay,
    )
    if active.rollout_steps != world_config.episode_steps:
        raise ValueError("physics PPO rollout must contain one complete multi-shot episode")
    environment = _build_environment(
        active=active,
        asset_root=asset_root,
        locomotion_policy_path=locomotion_policy_path,
        device=device,
        world_config=world_config,
    )
    arm_action_start_index = int(getattr(environment, "arm_action_start_index", 4))
    if not 1 <= arm_action_start_index < environment.action_size:
        raise RuntimeError("goalkeeper arm action boundary is invalid")
    lower_body_action_start_index = int(getattr(environment, "lower_body_action_start_index", 1))
    lower_body_action_end_index = int(
        getattr(environment, "lower_body_action_end_index", arm_action_start_index)
    )
    if (active.lower_body_and_arms_online_update or active.support_landing_online_update) and not (
        0 < lower_body_action_start_index < lower_body_action_end_index <= arm_action_start_index
    ):
        raise RuntimeError("goalkeeper lower-body action boundary is invalid")
    specialist_update = (
        active.arm_only_online_update
        or active.lower_body_and_arms_online_update
        or active.lateral_drive_gate_only_online_update
        or active.support_landing_online_update
    )
    plastic_action_indices = (
        (0,)
        if active.lateral_drive_gate_only_online_update
        else (
            tuple(range(0, arm_action_start_index))
            if active.support_landing_online_update
            else (
                tuple(range(arm_action_start_index, environment.action_size))
                if active.arm_only_online_update
                else (
                    tuple(range(lower_body_action_start_index, lower_body_action_end_index))
                    + tuple(range(arm_action_start_index, environment.action_size))
                    if active.lower_body_and_arms_online_update
                    else tuple(range(environment.action_size))
                )
            )
        )
    )
    plastic_action_tensor = torch.as_tensor(
        plastic_action_indices,
        dtype=torch.long,
        device=device,
    )
    plasticity_scope = (
        "ENVIRONMENT_DECLARED_LATERAL_DRIVE_GATE_ONLY"
        if active.lateral_drive_gate_only_online_update
        else (
            "ENVIRONMENT_DECLARED_SUPPORT_LANDING"
            if active.support_landing_online_update
            else (
                "ENVIRONMENT_DECLARED_ARMS_ONLY"
                if active.arm_only_online_update
                else (
                    "ENVIRONMENT_DECLARED_LOWER_BODY_AND_ARMS"
                    if active.lower_body_and_arms_online_update
                    else "FULL_ACTION"
                )
            )
        )
    )
    model = _build_actor_critic(
        torch,
        nn,
        environment.observation_size,
        environment.action_size,
        active.hidden_size,
    ).to(device)
    if active.targeted_dive_checkpoint is not None and active.initialization_checkpoint is None:
        with torch.no_grad():
            model.actor.bias[0] = math.atanh(active.targeted_dive_initial_gate)
    initialization_report: dict[str, Any] | None = None
    parent_model: Any | None = None
    if active.initialization_checkpoint is not None:
        initialization_path = Path(active.initialization_checkpoint).expanduser().resolve()
        initialization = torch.load(initialization_path, map_location=device, weights_only=True)
        expected = (environment.observation_size, environment.action_size, active.hidden_size)
        actual = (
            int(initialization.get("observation_size", -1)),
            int(initialization.get("action_size", -1)),
            int(initialization.get("hidden_size", -1)),
        )
        observation_migration_compatible = (actual[0], expected[0]) in {
            (74, 77),
            (89, 90),
            (89, 92),
            (90, 92),
            (92, 96),
        }
        compatible = actual == expected or (
            observation_migration_compatible and actual[1:] == expected[1:]
        )
        if not compatible:
            raise ValueError("physics PPO initialization checkpoint contract mismatch")
        initialization_state, observation_migration = _migrate_initialization_state(
            torch=torch,
            state_dict=initialization["state_dict"],
            old_observation_size=actual[0],
            new_observation_size=expected[0],
        )
        adapter_migration = _load_actor_critic_state(model, initialization_state)
        parent_model = _build_actor_critic(
            torch,
            nn,
            environment.observation_size,
            environment.action_size,
            active.hidden_size,
        ).to(device)
        _load_actor_critic_state(parent_model, initialization_state)
        parent_model.eval()
        for parameter in parent_model.parameters():
            parameter.requires_grad_(False)
        parent_training_config = initialization.get("training_config", {})
        if not isinstance(parent_training_config, dict):
            raise ValueError("physics PPO parent training config is invalid")
        initialization_report = {
            "checkpoint_hash": hash_bytes(initialization_path.read_bytes()),
            "role": "PARENT_POLICY_INITIALIZATION_ONLY",
            "online_updates_enabled": True,
            "observation_migration": observation_migration,
            "specialist_adapter_migration": adapter_migration,
            "controller_semantic_migration": _controller_semantic_migration(
                parent_training_config=parent_training_config,
                current=active,
            ),
        }
    combat_curriculum_report: dict[str, Any] | None = None
    # The paired curriculum is deterministic and precedes DDP wrapping.  Run
    # it once on rank zero; DistributedDataParallel will broadcast that exact
    # initialization instead of wasting every GPU on duplicate trajectories.
    if active.combat_gate_pretraining_batches > 0 and rank == 0:
        from rosclaw_soccer.training.goalkeeper_combat_curriculum import (
            pretrain_combat_teacher_gate,
        )

        combat_curriculum_report = pretrain_combat_teacher_gate(
            torch=torch,
            model=model,
            environment=environment,
            device=device,
            batches=active.combat_gate_pretraining_batches,
            epochs=active.combat_gate_pretraining_epochs,
            seed=active.random_seed + 303_007,
            task_space_reach_blend=active.task_space_reach_blend,
            task_space_reach_atlas_enabled=active.task_space_reach_atlas_enabled,
            second_shot_reach_multiplier=active.second_shot_reach_multiplier,
        )
    teacher_report: dict[str, Any] | None = None
    if active.teacher_pretraining_enabled:
        from rosclaw_soccer.providers.g1.asset_qualification import g1_body_hash
        from rosclaw_soccer.training.goalkeeper_reach import (
            GoalkeeperReachConfig,
            build_g1_task_space_reach_atlas,
            build_g1_task_space_reach_model,
        )
        from rosclaw_soccer.training.goalkeeper_teacher import pretrain_goalkeeper_actor

        reach_config = (
            GoalkeeperReachConfig(
                damping=0.12,
                reach_gain=0.95,
                maximum_position_error_m=0.75,
                support_arm_scale=0.60,
                central_support_scale=0.95,
                residual_scale=min(
                    1.0,
                    world_config.residual_scale * world_config.arm_residual_scale_multiplier,
                ),
                arm_authority_scale=world_config.agility.arm_authority_scale,
            )
            if active.shot_difficulty_profile == "elite"
            else GoalkeeperReachConfig(
                residual_scale=min(
                    1.0,
                    world_config.residual_scale * world_config.arm_residual_scale_multiplier,
                ),
                arm_authority_scale=world_config.agility.arm_authority_scale,
            )
        )
        reach_model = (
            (
                build_g1_task_space_reach_atlas(asset_root, config=reach_config)
                if active.task_space_reach_atlas_enabled
                else build_g1_task_space_reach_model(asset_root, config=reach_config)
            )
            if active.task_space_reach_blend > 0.0
            else None
        )
        teacher_report = pretrain_goalkeeper_actor(
            model,
            observation_size=environment.observation_size,
            device=device,
            samples=active.teacher_pretraining_samples,
            epochs=active.teacher_pretraining_epochs,
            parent_replay_coefficient=(
                active.teacher_parent_replay_coefficient
                if active.initialization_checkpoint is not None
                else 0.0
            ),
            motion_library_path=motion_library_path,
            motion_dataset_root=motion_dataset_root,
            expected_body_hash=(
                g1_body_hash(asset_root) if motion_library_path is not None else None
            ),
            reach_model=reach_model,
            task_space_reach_blend=active.task_space_reach_blend,
            arm_only_update=active.arm_only_online_update,
            seed=active.random_seed,
        )
        if not teacher_report["improved"]:
            raise RuntimeError("goalkeeper teacher pretraining failed to improve imitation loss")
    if distributed:
        policy: Any = DistributedDataParallel(model, device_ids=[local_rank])
    else:
        policy = model
    optimizer = torch.optim.Adam(policy.parameters(), lr=active.learning_rate, eps=1.0e-5)
    started = time.perf_counter()
    iteration_metrics: list[dict[str, Any]] = []
    total_world_steps = 0
    deterministic_selection_world_steps = 0
    best_reward = -math.inf
    best_selection_score = -math.inf
    best_truth_key: tuple[float, ...] | None = None
    best_state: dict[str, Any] | None = None
    best_rollout_iteration: int | None = None
    best_safe_truth_key: tuple[float, ...] | None = None
    best_safe_state: dict[str, Any] | None = None
    best_safe_iteration: int | None = None
    best_safe_metrics: dict[str, Any] | None = None
    best_exploration_truth_key: tuple[float, ...] | None = None
    best_exploration_state: dict[str, Any] | None = None
    best_exploration_iteration: int | None = None
    best_exploration_metrics: dict[str, Any] | None = None
    successful_memory_observations: Any | None = None
    successful_memory_actions: Any | None = None
    successful_memory_height_strata: Any | None = None
    successful_memory_episode_counts = torch.zeros(6, dtype=torch.long, device=device)
    required_successful_memory_strata = _required_successful_memory_strata(
        hard_shot_fraction=active.training_hard_shot_fraction,
        hard_shot_height_mode=active.training_hard_shot_height_mode,
        hard_shot_side_mode=active.training_hard_shot_side_mode,
    )
    progress_output = output_dir.expanduser().resolve()
    if rank == 0:
        progress_output.mkdir(parents=True, exist_ok=True)
    exploration_champion_rollback_count = 0
    final_model_is_rolled_back_champion = False

    for iteration in range(active.iterations):
        candidate_regressed_from_exploration_champion = False
        # Bind the later rollout metrics to the exact parameters that produced
        # them.  Saving post-update weights against pre-update metrics creates
        # an off-by-one evidence claim and is intentionally forbidden.
        rollout_state = (
            {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            if rank == 0
            else None
        )
        selection_seed_metrics: list[Any] = []
        selection_seed_maximum_root_angular: list[Any] = []
        for selection_seed_index in range(active.deterministic_selection_seed_count):
            panel_metrics, panel_maximum_root_angular = _deterministic_candidate_rollout(
                torch=torch,
                model=model,
                environment=environment,
                seed=_held_out_selection_seed(
                    random_seed=(
                        active.random_seed
                        + selection_seed_index * active.deterministic_selection_seed_stride
                    ),
                    rank=rank,
                ),
            )
            if distributed:
                dist.all_reduce(panel_metrics, op=dist.ReduceOp.SUM)
                panel_metrics /= world_size
                dist.all_reduce(panel_maximum_root_angular, op=dist.ReduceOp.MAX)
            selection_seed_metrics.append(panel_metrics)
            selection_seed_maximum_root_angular.append(panel_maximum_root_angular)
        selection_metrics = torch.stack(selection_seed_metrics).mean(dim=0)
        selection_maximum_root_angular = torch.stack(selection_seed_maximum_root_angular).max()
        deterministic_selection_world_steps += (
            active.environments_per_rank
            * world_size
            * active.rollout_steps
            * world_config.physics_substeps
            * active.deterministic_selection_seed_count
        )
        deterministic_candidate: dict[str, Any] = {
            "mean_episode_reward": float(selection_metrics[0]),
            "first_save_rate": float(selection_metrics[1]),
            "first_hand_save_rate": float(selection_metrics[2]),
            "recovery_rate": float(selection_metrics[3]),
            "second_save_rate": float(selection_metrics[4]),
            "second_hand_save_rate": float(selection_metrics[5]),
            "failed_rate": float(selection_metrics[6]),
            "mean_maximum_lateral_displacement_m": float(selection_metrics[7]),
            "mean_maximum_lateral_speed_mps": float(selection_metrics[8]),
            "mean_maximum_hand_displacement_m": float(selection_metrics[9]),
            "mean_maximum_hand_speed_mps": float(selection_metrics[10]),
            "mean_maximum_root_angular_speed_rad_s": float(selection_metrics[11]),
            "mean_minimum_upper_body_authority": float(selection_metrics[12]),
            "mean_second_release_lateral_error_m": float(selection_metrics[13]),
            "bimanual_reach_fraction": float(selection_metrics[14]),
            "second_attempt_save_rate": float(selection_metrics[15]),
            "second_attempt_hand_save_rate": float(selection_metrics[16]),
            "maximum_root_angular_speed_rad_s": float(selection_maximum_root_angular),
            "quarantined_rate": float(selection_metrics[23]),
            "nonfinite_quarantine_rate": float(selection_metrics[24]),
            "mean_minimum_hand_target_distance_m": float(selection_metrics[31]),
            "mean_task_motion_reward": float(selection_metrics[32]),
            "mean_first_decisive_pelvis_lateral_error_m": float(selection_metrics[33]),
            "mean_first_decisive_hand_intercept_distance_m": float(selection_metrics[34]),
            "root_angular_speed_soft_limit_rad_s": (
                world_config.root_angular_speed_soft_limit_rad_s
            ),
            "root_angular_speed_soft_limit_exceedance_rate": float(selection_metrics[35]),
            "strict_stability_ceiling_rad_s": 3.50,
            "strict_stability_ceiling_exceedance_rate": float(selection_metrics[36]),
            "mean_recovery_progress_reward": float(selection_metrics[37]),
            "qualified_first_save_rate": float(selection_metrics[38]),
            "mean_reach_reward": float(selection_metrics[39]),
            "mean_bimanual_reach_reward": float(selection_metrics[40]),
            "mean_upright_reward": float(selection_metrics[41]),
            "mean_smoothness_penalty": float(selection_metrics[42]),
            "mean_effort_penalty": float(selection_metrics[43]),
            "mean_event_bonus": float(selection_metrics[44]),
            "mean_safety_penalty": float(selection_metrics[45]),
            "mean_nonfinite_override": float(selection_metrics[46]),
            "mean_action_rate_penalty": float(selection_metrics[47]),
            "mean_joint_acceleration_penalty": float(selection_metrics[48]),
            "mean_root_linear_speed_penalty": float(selection_metrics[49]),
            "mean_root_angular_speed_penalty": float(selection_metrics[50]),
            "mean_root_angular_excess_penalty": float(selection_metrics[51]),
            "mean_action_magnitude_penalty": float(selection_metrics[52]),
            "mean_qualified_save_episode_reward": float(
                selection_metrics[53] / torch.clamp(selection_metrics[54], min=1.0)
            ),
            "mean_failed_episode_reward": float(
                selection_metrics[55] / torch.clamp(selection_metrics[56], min=1.0)
            ),
            "mean_stable_miss_episode_reward": float(
                selection_metrics[57] / torch.clamp(selection_metrics[58], min=1.0)
            ),
            "stochastic_exploration": False,
            "checkpoint_weight_binding": "PRE_UPDATE_ROLLOUT_STATE_EXACT",
        }
        deterministic_strata = _stratum_rates(selection_metrics)
        deterministic_candidate["first_save_rate_by_height"] = {
            "far_corner_low": deterministic_strata[0],
            "far_corner_mid": deterministic_strata[1],
            "far_corner_high": deterministic_strata[2],
        }
        deterministic_candidate["minimum_first_save_stratum_rate"] = min(deterministic_strata)
        deterministic_candidate["first_save_stratum_balance_score"] = _stratum_balance_score(
            deterministic_strata
        )
        deterministic_failed_strata = _stratum_rates(selection_metrics, save_offset=25)
        deterministic_quarantined_strata = _stratum_rates(
            selection_metrics,
            save_offset=28,
        )
        deterministic_candidate["failed_rate_by_height"] = {
            "far_corner_low": deterministic_failed_strata[0],
            "far_corner_mid": deterministic_failed_strata[1],
            "far_corner_high": deterministic_failed_strata[2],
        }
        deterministic_candidate["quarantined_rate_by_height"] = {
            "far_corner_low": deterministic_quarantined_strata[0],
            "far_corner_mid": deterministic_quarantined_strata[1],
            "far_corner_high": deterministic_quarantined_strata[2],
        }
        deterministic_side_saves = _side_rates(selection_metrics, event_offset=59, count_offset=63)
        deterministic_side_qualified = _side_rates(
            selection_metrics, event_offset=61, count_offset=63
        )
        deterministic_side_failures = _side_rates(
            selection_metrics, event_offset=65, count_offset=63
        )
        deterministic_candidate["first_save_rate_by_side"] = {
            "negative_far_corner": deterministic_side_saves[0],
            "positive_far_corner": deterministic_side_saves[1],
        }
        deterministic_candidate["qualified_first_save_rate_by_side"] = {
            "negative_far_corner": deterministic_side_qualified[0],
            "positive_far_corner": deterministic_side_qualified[1],
        }
        deterministic_candidate["failed_rate_by_side"] = {
            "negative_far_corner": deterministic_side_failures[0],
            "positive_far_corner": deterministic_side_failures[1],
        }
        deterministic_candidate["minimum_qualified_first_save_side_rate"] = min(
            deterministic_side_qualified
        )
        deterministic_candidate["minimum_first_save_side_rate"] = min(deterministic_side_saves)
        selection_seed_panel: list[dict[str, Any]] = []
        for panel_index, panel_metrics in enumerate(selection_seed_metrics):
            panel_qualified_side = _side_rates(
                panel_metrics,
                event_offset=61,
                count_offset=63,
            )
            selection_seed_panel.append(
                {
                    "panel_index": panel_index,
                    "first_save_rate": float(panel_metrics[1]),
                    "qualified_first_save_rate": float(panel_metrics[38]),
                    "minimum_qualified_first_save_side_rate": min(panel_qualified_side),
                    "failed_rate": float(panel_metrics[6]),
                    "maximum_root_angular_speed_rad_s": float(
                        selection_seed_maximum_root_angular[panel_index]
                    ),
                }
            )
        deterministic_candidate["selection_seed_panel"] = selection_seed_panel
        deterministic_candidate["minimum_seed_first_save_rate"] = min(
            panel["first_save_rate"] for panel in selection_seed_panel
        )
        deterministic_candidate["minimum_seed_qualified_first_save_rate"] = min(
            panel["qualified_first_save_rate"] for panel in selection_seed_panel
        )
        deterministic_candidate["minimum_seed_qualified_first_save_side_rate"] = min(
            panel["minimum_qualified_first_save_side_rate"] for panel in selection_seed_panel
        )
        deterministic_candidate["maximum_seed_failed_rate"] = max(
            panel["failed_rate"] for panel in selection_seed_panel
        )
        deterministic_candidate["mean_reward_accounting_residual"] = _reward_accounting_residual(
            deterministic_candidate
        )
        deterministic_candidate["anatomical_selection_score"] = (
            deterministic_candidate["mean_episode_reward"]
            + active.first_save_selection_weight * deterministic_candidate["first_save_rate"]
            + active.qualified_first_save_selection_weight
            * deterministic_candidate["qualified_first_save_rate"]
            + active.first_hand_save_selection_weight
            * deterministic_candidate["first_hand_save_rate"]
            + active.second_attempt_save_selection_weight
            * deterministic_candidate["second_attempt_save_rate"]
            + active.second_attempt_hand_save_selection_weight
            * deterministic_candidate["second_attempt_hand_save_rate"]
            + active.second_save_selection_weight * deterministic_candidate["second_save_rate"]
            + active.second_hand_save_selection_weight
            * deterministic_candidate["second_hand_save_rate"]
            + active.hand_reach_selection_weight
            * deterministic_candidate["mean_maximum_hand_displacement_m"]
            + active.first_save_stratum_balance_selection_weight
            * deterministic_candidate["first_save_stratum_balance_score"]
            + active.first_save_side_balance_selection_weight
            * deterministic_candidate["minimum_qualified_first_save_side_rate"]
            - active.root_angular_speed_selection_penalty_weight
            * deterministic_candidate["mean_maximum_root_angular_speed_rad_s"]
            - active.second_release_recenter_selection_weight
            * deterministic_candidate["mean_second_release_lateral_error_m"]
            - active.hand_target_distance_selection_weight
            * deterministic_candidate["mean_minimum_hand_target_distance_m"]
        )
        observation = environment.reset(seed=active.random_seed + 10_007 * iteration + rank)
        observations: list[Any] = []
        actions: list[Any] = []
        action_means: list[Any] = []
        old_log_probabilities: list[Any] = []
        rewards: list[Any] = []
        dones: list[Any] = []
        values: list[Any] = []
        task_motion_rewards: list[Any] = []
        recovery_progress_rewards: list[Any] = []
        cumulative_reach_reward = torch.zeros(active.environments_per_rank, device=device)
        cumulative_bimanual_reach_reward = torch.zeros_like(cumulative_reach_reward)
        cumulative_upright_reward = torch.zeros_like(cumulative_reach_reward)
        cumulative_smoothness_penalty = torch.zeros_like(cumulative_reach_reward)
        cumulative_effort_penalty = torch.zeros_like(cumulative_reach_reward)
        cumulative_event_bonus = torch.zeros_like(cumulative_reach_reward)
        cumulative_safety_penalty = torch.zeros_like(cumulative_reach_reward)
        cumulative_nonfinite_override = torch.zeros_like(cumulative_reach_reward)
        cumulative_action_rate_penalty = torch.zeros_like(cumulative_reach_reward)
        cumulative_joint_acceleration_penalty = torch.zeros_like(cumulative_reach_reward)
        cumulative_root_linear_speed_penalty = torch.zeros_like(cumulative_reach_reward)
        cumulative_root_angular_speed_penalty = torch.zeros_like(cumulative_reach_reward)
        cumulative_root_angular_excess_penalty = torch.zeros_like(cumulative_reach_reward)
        cumulative_action_magnitude_penalty = torch.zeros_like(cumulative_reach_reward)
        replay_active_steps: list[Any] = []
        recovery_transition_steps: list[Any] = []
        alive = torch.ones(active.environments_per_rank, dtype=torch.bool, device=device)
        for _ in range(active.rollout_steps):
            with torch.no_grad():
                mean, value, log_std = policy(observation)
                raw_action, log_probability = _sample_exploration_action(
                    torch=torch,
                    normal_distribution=Normal,
                    mean=mean,
                    log_std=log_std,
                    arm_only=active.arm_only_online_update,
                    arm_action_start_index=arm_action_start_index,
                    lower_body_and_arms=active.lower_body_and_arms_online_update,
                    lower_body_action_start_index=lower_body_action_start_index,
                    lower_body_action_end_index=lower_body_action_end_index,
                    lateral_drive_gate_only=(active.lateral_drive_gate_only_online_update),
                    support_landing=active.support_landing_online_update,
                    exploration_active_mask=(
                        environment.task.first_save
                        if active.support_landing_causal_recovery_only_enabled
                        else None
                    ),
                )
                bounded_action = torch.tanh(raw_action)
            next_observation, reward, done, info = environment.step(bounded_action)
            reward = torch.where(alive, reward, torch.zeros_like(reward))
            task_motion_reward = torch.where(
                alive,
                info["task_motion"],
                torch.zeros_like(info["task_motion"]),
            )
            newly_done = alive & done
            observations.append(observation)
            actions.append(raw_action)
            action_means.append(mean)
            old_log_probabilities.append(log_probability)
            rewards.append(reward)
            dones.append(newly_done | ~alive)
            values.append(value)
            task_motion_rewards.append(task_motion_reward)
            recovery_progress_reward = torch.where(
                alive,
                info["recovery_progress"],
                torch.zeros_like(info["recovery_progress"]),
            )
            recovery_progress_rewards.append(recovery_progress_reward)
            for cumulative, key in (
                (cumulative_reach_reward, "reach"),
                (cumulative_bimanual_reach_reward, "bimanual_reach"),
                (cumulative_upright_reward, "upright"),
                (cumulative_smoothness_penalty, "smoothness_penalty"),
                (cumulative_effort_penalty, "effort_penalty"),
                (cumulative_event_bonus, "event_bonus"),
                (cumulative_safety_penalty, "safety_penalty"),
                (cumulative_nonfinite_override, "nonfinite_override"),
                (cumulative_action_rate_penalty, "action_rate_penalty"),
                (cumulative_joint_acceleration_penalty, "joint_acceleration_penalty"),
                (cumulative_root_linear_speed_penalty, "root_linear_speed_penalty"),
                (cumulative_root_angular_speed_penalty, "root_angular_speed_penalty"),
                (cumulative_root_angular_excess_penalty, "root_angular_excess_penalty"),
                (cumulative_action_magnitude_penalty, "action_magnitude_penalty"),
            ):
                cumulative += torch.where(alive, info[key], torch.zeros_like(info[key]))
            replay_active = getattr(environment, "_actor_plasticity_active", None)
            if replay_active is None:
                replay_active = getattr(
                    environment,
                    "_option_active",
                    info["event_shot_index"] > 0,
                )
            replay_active_steps.append(alive & replay_active)
            recovery_transition_steps.append(alive & replay_active & info["first_save"])
            alive &= ~newly_done
            observation = next_observation
        if not environment.finite_state():
            raise FloatingPointError("MJWarp goalkeeper rollout produced non-finite physics state")

        observation_tensor = torch.stack(observations)
        action_tensor = torch.stack(actions)
        action_mean_tensor = torch.stack(action_means)
        successful_memory_action_tensor = _shrink_successful_action_innovation(
            torch=torch,
            policy_mean=action_mean_tensor,
            sampled_action=action_tensor,
            innovation_scale=active.successful_trajectory_action_innovation_scale,
        )
        old_log_probability_tensor = torch.stack(old_log_probabilities)
        reward_tensor = torch.stack(rewards)
        done_tensor = torch.stack(dones)
        value_tensor = torch.stack(values)
        advantage = torch.zeros_like(reward_tensor)
        last_advantage = torch.zeros(active.environments_per_rank, device=device)
        next_value = torch.zeros_like(last_advantage)
        for step in range(active.rollout_steps - 1, -1, -1):
            nonterminal = (~done_tensor[step]).to(torch.float32)
            delta = reward_tensor[step] + active.discount * next_value * nonterminal
            delta -= value_tensor[step]
            last_advantage = (
                delta + active.discount * active.gae_lambda * nonterminal * last_advantage
            )
            advantage[step] = last_advantage
            next_value = value_tensor[step]
        returns = advantage + value_tensor
        flat_observation = observation_tensor.flatten(0, 1)
        flat_action = action_tensor.flatten(0, 1)
        flat_old_log_probability = old_log_probability_tensor.flatten()
        flat_advantage = advantage.flatten()
        flat_returns = returns.flatten()
        recovery_transition_tensor = torch.stack(recovery_transition_steps)
        replay_eligibility_steps = (
            recovery_transition_tensor
            if active.successful_trajectory_recovery_only_enabled
            else torch.stack(replay_active_steps)
        )
        successful_replay_mask, successful_replay_episodes = _successful_trajectory_replay_mask(
            torch=torch,
            active_steps=replay_eligibility_steps,
            first_save=environment.task.first_save,
            quarantined=environment._quarantined,
            maximum_root_angular_speed_rad_s=(environment._maximum_root_angular_speed),
            angular_speed_ceiling_rad_s=(
                active.maximum_exploration_root_angular_speed_for_selection
            ),
        )
        flat_successful_replay_mask = successful_replay_mask.flatten()
        flat_recovery_transition_mask = recovery_transition_tensor.flatten()
        local_new_successful_memory_rows = 0
        global_new_successful_memory_rows = 0
        local_new_successful_episode_counts = torch.zeros(6, dtype=torch.long, device=device)
        global_new_successful_episode_counts = torch.zeros_like(local_new_successful_episode_counts)
        if active.successful_trajectory_memory_capacity_per_stratum > 0:
            first_target_height = environment._target_one[:, 2]
            first_height_strata = torch.where(
                first_target_height < 0.60,
                torch.zeros_like(first_target_height, dtype=torch.long),
                torch.where(
                    first_target_height < 1.10,
                    torch.ones_like(first_target_height, dtype=torch.long),
                    torch.full_like(first_target_height, 2, dtype=torch.long),
                ),
            )
            first_replay_strata = 2 * first_height_strata + (
                environment._target_one[:, 1] >= 0.0
            ).to(torch.long)
            local_new_successful_episode_counts = torch.bincount(
                first_replay_strata[successful_replay_episodes], minlength=6
            )
            global_new_successful_episode_counts = local_new_successful_episode_counts.clone()
            if distributed:
                dist.all_reduce(
                    global_new_successful_episode_counts,
                    op=dist.ReduceOp.SUM,
                )
            successful_memory_episode_counts += global_new_successful_episode_counts
            expanded_replay_strata = first_replay_strata.unsqueeze(0).expand_as(
                successful_replay_mask
            )
            new_memory_observations = observation_tensor.flatten(0, 1)[
                flat_successful_replay_mask
            ].detach()
            new_memory_actions = successful_memory_action_tensor.flatten(0, 1)[
                flat_successful_replay_mask
            ].detach()
            new_memory_strata = (
                expanded_replay_strata.flatten()[flat_successful_replay_mask]
                .to(torch.long)
                .detach()
            )
            local_new_successful_memory_rows = int(new_memory_strata.shape[0])
            if distributed:
                (
                    new_memory_observations,
                    new_memory_actions,
                    new_memory_strata,
                ) = _all_gather_successful_memory_rows(
                    torch=torch,
                    dist=dist,
                    observations=new_memory_observations,
                    actions=new_memory_actions,
                    strata=new_memory_strata,
                    world_size=world_size,
                )
            global_new_successful_memory_rows = int(new_memory_strata.shape[0])
            (
                successful_memory_observations,
                successful_memory_actions,
                successful_memory_height_strata,
            ) = _append_successful_memory_rows(
                torch=torch,
                new_observations=new_memory_observations,
                new_actions=new_memory_actions,
                new_strata=new_memory_strata,
                memory_observations=successful_memory_observations,
                memory_actions=successful_memory_actions,
                memory_height_strata=successful_memory_height_strata,
                capacity_per_stratum=(active.successful_trajectory_memory_capacity_per_stratum),
                stratum_count=6,
            )
        local_memory_coverage_ready = (
            successful_memory_height_strata is not None
            and _successful_memory_covers_strata(
                torch=torch,
                height_strata=successful_memory_height_strata,
                required_strata=required_successful_memory_strata,
            )
            and _successful_memory_has_episode_diversity(
                episode_counts=successful_memory_episode_counts,
                required_strata=required_successful_memory_strata,
                minimum_episodes=(active.successful_trajectory_memory_minimum_episodes_per_stratum),
            )
        )
        memory_coverage_consensus = torch.tensor(
            int(local_memory_coverage_ready), dtype=torch.int32, device=device
        )
        if distributed:
            dist.all_reduce(memory_coverage_consensus, op=dist.ReduceOp.MIN)
        successful_memory_coverage_ready = bool(memory_coverage_consensus.item())
        successful_memory_replay_strength = (
            _successful_memory_replay_strength(
                episode_counts=successful_memory_episode_counts,
                required_strata=required_successful_memory_strata,
                minimum_episodes=(active.successful_trajectory_memory_minimum_episodes_per_stratum),
                full_strength_episodes=(
                    active.successful_trajectory_memory_full_strength_episodes_per_stratum
                ),
            )
            if active.successful_trajectory_memory_capacity_per_stratum > 0
            else 1.0
        )
        effective_successful_trajectory_replay_coefficient = (
            active.successful_trajectory_replay_coefficient * successful_memory_replay_strength
        )
        flat_advantage = (flat_advantage - flat_advantage.mean()) / (
            flat_advantage.std(unbiased=False) + 1.0e-8
        )
        sample_count = flat_observation.shape[0]
        minibatch_size = max(1, sample_count // active.minibatches)
        policy_loss_value = 0.0
        value_loss_value = 0.0
        entropy_value = 0.0
        parent_distillation_loss_value = 0.0
        successful_trajectory_replay_loss_value = 0.0
        symmetry_mirror_loss_value = 0.0
        successful_trajectory_replay_loss_sum = 0.0
        successful_trajectory_replay_update_count = 0
        for _ in range(active.update_epochs):
            permutation = torch.randperm(sample_count, device=device)
            for start in range(0, sample_count, minibatch_size):
                indices = permutation[start : start + minibatch_size]
                mean, predicted_value, log_std = policy(flat_observation[indices])
                if specialist_update:
                    distribution = Normal(
                        mean[:, plastic_action_tensor],
                        torch.exp(log_std[plastic_action_tensor]).expand_as(
                            mean[:, plastic_action_tensor]
                        ),
                    )
                    new_log_probability = distribution.log_prob(
                        flat_action[indices][:, plastic_action_tensor]
                    ).sum(dim=1)
                else:
                    distribution = Normal(mean, torch.exp(log_std).expand_as(mean))
                    new_log_probability = distribution.log_prob(flat_action[indices]).sum(dim=1)
                ratio = torch.exp(new_log_probability - flat_old_log_probability[indices])
                unclipped = ratio * flat_advantage[indices]
                clipped = (
                    torch.clamp(ratio, 1.0 - active.clip_ratio, 1.0 + active.clip_ratio)
                    * flat_advantage[indices]
                )
                recovery_weight = (
                    flat_recovery_transition_mask[indices].to(torch.float32)
                    if active.support_landing_causal_recovery_only_enabled
                    else 1.0
                    + (active.recovery_transition_policy_weight - 1.0)
                    * flat_recovery_transition_mask[indices].to(torch.float32)
                )
                clipped_surrogate = torch.minimum(unclipped, clipped)
                policy_loss = -torch.sum(clipped_surrogate * recovery_weight) / torch.sum(
                    recovery_weight
                ).clamp(min=1.0)
                value_loss = 0.5 * torch.square(predicted_value - flat_returns[indices]).mean()
                entropy = distribution.entropy().sum(dim=1).mean()
                anchor_target = mean[:, plastic_action_tensor] if specialist_update else mean
                anchor_loss = torch.mean(torch.square(torch.tanh(anchor_target)))
                parent_distillation_loss = torch.zeros((), device=device)
                if parent_model is not None and active.online_parent_distillation_coefficient > 0.0:
                    with torch.no_grad():
                        parent_mean, _, _ = parent_model(flat_observation[indices])
                    parent_distillation_loss = torch.mean(
                        torch.square(torch.tanh(mean) - torch.tanh(parent_mean))
                    )
                symmetry_mirror_loss = torch.zeros((), device=device)
                if active.symmetry_mirror_loss_coefficient > 0.0:
                    mirrored_observation = _mirror_goalkeeper_actor_observation(
                        torch=torch,
                        observation=flat_observation[indices],
                        shot_intent_cue_enabled=active.shot_intent_cue_enabled,
                        actor_contact_support_side_enabled=(
                            active.targeted_dive_actor_contact_support_side_enabled
                        ),
                        actor_recovery_context_enabled=(
                            active.targeted_dive_actor_recovery_context_enabled
                        ),
                    )
                    mirrored_mean, _, _ = policy(mirrored_observation)
                    mirrored_target = _mirror_goalkeeper_actor_action(
                        torch=torch, action=mean.detach()
                    )
                    if specialist_update:
                        mirrored_mean = mirrored_mean[:, plastic_action_tensor]
                        mirrored_target = mirrored_target[:, plastic_action_tensor]
                    symmetry_mirror_loss = torch.mean(
                        torch.square(torch.tanh(mirrored_mean) - torch.tanh(mirrored_target))
                    )
                successful_trajectory_replay_loss = torch.zeros((), device=device)
                if (
                    active.successful_trajectory_memory_capacity_per_stratum > 0
                    and successful_memory_coverage_ready
                    and successful_memory_observations is not None
                    and successful_memory_actions is not None
                    and successful_memory_height_strata is not None
                    and int(successful_memory_observations.shape[0]) > 0
                    and _successful_memory_covers_strata(
                        torch=torch,
                        height_strata=successful_memory_height_strata,
                        required_strata=required_successful_memory_strata,
                    )
                ):
                    memory_indices = _balanced_successful_memory_sample_indices(
                        torch=torch,
                        height_strata=successful_memory_height_strata,
                        sample_count=min(
                            active.successful_trajectory_memory_batch_size,
                            int(successful_memory_observations.shape[0]),
                        ),
                        stratum_count=6,
                    )
                    memory_mean, _, _ = policy(successful_memory_observations[memory_indices])
                    replay_mean = (
                        memory_mean[:, plastic_action_tensor] if specialist_update else memory_mean
                    )
                    replay_action = successful_memory_actions[memory_indices]
                    if specialist_update:
                        replay_action = replay_action[:, plastic_action_tensor]
                    successful_trajectory_replay_loss = torch.mean(
                        torch.square(torch.tanh(replay_mean) - torch.tanh(replay_action))
                    )
                    if active.successful_trajectory_mirror_augmentation_enabled:
                        mirrored_memory_observation = _mirror_goalkeeper_actor_observation(
                            torch=torch,
                            observation=successful_memory_observations[memory_indices],
                            shot_intent_cue_enabled=active.shot_intent_cue_enabled,
                            actor_contact_support_side_enabled=(
                                active.targeted_dive_actor_contact_support_side_enabled
                            ),
                            actor_recovery_context_enabled=(
                                active.targeted_dive_actor_recovery_context_enabled
                            ),
                        )
                        mirrored_memory_action = _mirror_goalkeeper_actor_action(
                            torch=torch,
                            action=successful_memory_actions[memory_indices],
                        )
                        mirrored_memory_mean, _, _ = policy(mirrored_memory_observation)
                        if specialist_update:
                            mirrored_memory_mean = mirrored_memory_mean[:, plastic_action_tensor]
                            mirrored_memory_action = mirrored_memory_action[
                                :, plastic_action_tensor
                            ]
                        mirrored_replay_loss = torch.mean(
                            torch.square(
                                torch.tanh(mirrored_memory_mean)
                                - torch.tanh(mirrored_memory_action)
                            )
                        )
                        successful_trajectory_replay_loss = 0.5 * (
                            successful_trajectory_replay_loss + mirrored_replay_loss
                        )
                elif active.successful_trajectory_memory_capacity_per_stratum == 0:
                    successful_rows = flat_successful_replay_mask[indices]
                    if bool(torch.any(successful_rows)):
                        replay_mean = mean[:, plastic_action_tensor] if specialist_update else mean
                        replay_action = successful_memory_action_tensor.flatten(0, 1)[indices]
                        if specialist_update:
                            replay_action = replay_action[:, plastic_action_tensor]
                        successful_trajectory_replay_loss = torch.mean(
                            torch.square(
                                torch.tanh(replay_mean[successful_rows])
                                - torch.tanh(replay_action[successful_rows])
                            )
                        )
                loss = policy_loss + active.value_coefficient * value_loss
                loss -= active.entropy_coefficient * entropy
                loss += active.policy_anchor_coefficient * anchor_loss
                loss += active.online_parent_distillation_coefficient * parent_distillation_loss
                loss += (
                    effective_successful_trajectory_replay_coefficient
                    * successful_trajectory_replay_loss
                )
                loss += active.symmetry_mirror_loss_coefficient * symmetry_mirror_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if specialist_update:
                    _mask_specialist_gradients(
                        model,
                        plastic_action_indices=plastic_action_indices,
                    )
                nn.utils.clip_grad_norm_(policy.parameters(), active.maximum_gradient_norm)
                optimizer.step()
                policy_loss_value = float(policy_loss.detach())
                value_loss_value = float(value_loss.detach())
                entropy_value = float(entropy.detach())
                parent_distillation_loss_value = float(parent_distillation_loss.detach())
                successful_trajectory_replay_loss_value = float(
                    successful_trajectory_replay_loss.detach()
                )
                symmetry_mirror_loss_value = float(symmetry_mirror_loss.detach())
                if successful_trajectory_replay_loss_value > 0.0:
                    successful_trajectory_replay_loss_sum += successful_trajectory_replay_loss_value
                    successful_trajectory_replay_update_count += 1

        if successful_trajectory_replay_update_count > 0:
            successful_trajectory_replay_loss_value = (
                successful_trajectory_replay_loss_sum / successful_trajectory_replay_update_count
            )

        episode_return = reward_tensor.sum(dim=0)
        local_reward = episode_return.mean()
        local_task_motion_reward = torch.stack(task_motion_rewards).sum(dim=0).mean()
        local_recovery_progress_reward = torch.stack(recovery_progress_rewards).sum(dim=0).mean()
        local_first = environment.task.first_save.to(torch.float32).mean()
        local_first_hand = environment.task.first_hand_save.to(torch.float32).mean()
        local_recovery = environment.task.recovered_after_first.to(torch.float32).mean()
        local_qualified_first_save = (
            (environment.task.first_save & (environment.task.phase != 7)).to(torch.float32).mean()
        )
        qualified_save_mask = environment.task.first_save & (environment.task.phase != 7)
        failed_mask = environment.task.phase == 7
        stable_miss_mask = ~environment.task.first_save & ~failed_mask
        local_second = environment.task.second_save.to(torch.float32).mean()
        local_second_hand = environment.task.second_hand_save.to(torch.float32).mean()
        local_second_attempt = environment.task.second_attempt_save.to(torch.float32).mean()
        local_second_attempt_hand = environment.task.second_attempt_hand_save.to(
            torch.float32
        ).mean()
        local_failed = (environment.task.phase == 7).to(torch.float32).mean()
        local_lateral_displacement = environment._maximum_lateral_displacement.mean()
        local_lateral_speed = environment._maximum_lateral_speed.mean()
        local_hand_displacement = environment._maximum_hand_displacement.mean()
        local_hand_speed = environment._maximum_hand_speed.mean()
        local_root_angular = environment._maximum_root_angular_speed.mean()
        local_upper_authority = environment._minimum_upper_body_authority.mean()
        local_recenter_error = environment._second_release_lateral_error.mean()
        local_bimanual_reach = environment.task._bimanual_reach_steps.sum().to(torch.float32)
        local_bimanual_reach /= torch.clamp(environment.task._active_flight_steps.sum(), min=1).to(
            torch.float32
        )
        local_stratum_saves, local_stratum_counts = _first_save_stratum_counts(torch, environment)
        local_stratum_failures = _height_stratum_event_counts(
            torch,
            environment,
            environment.task.phase == 7,
        )
        local_stratum_quarantines = _height_stratum_event_counts(
            torch,
            environment,
            environment._quarantined,
        )
        local_side_saves = _side_event_counts(torch, environment, environment.task.first_save)
        local_side_qualified_saves = _side_event_counts(torch, environment, qualified_save_mask)
        local_side_counts = _side_event_counts(
            torch,
            environment,
            torch.ones_like(environment.task.first_save),
        )
        local_side_failures = _side_event_counts(torch, environment, failed_mask)
        local_quarantined = environment._quarantined.to(torch.float32).mean()
        local_nonfinite_quarantine = environment._nonfinite_quarantine_latched.to(
            torch.float32
        ).mean()
        maximum_root_angular = environment._maximum_root_angular_speed.max()
        replay_episode_rate = successful_replay_episodes.to(torch.float32).mean()
        replay_transition_fraction = successful_replay_mask.to(torch.float32).mean()
        recovery_transition_fraction = recovery_transition_tensor.to(torch.float32).mean()
        replay_diagnostics = torch.stack(
            (replay_episode_rate, replay_transition_fraction, recovery_transition_fraction)
        )
        if successful_memory_height_strata is None:
            memory_diagnostics = torch.zeros(8, device=device)
        else:
            memory_diagnostics = torch.stack(
                (
                    torch.tensor(
                        successful_memory_height_strata.shape[0],
                        dtype=torch.float32,
                        device=device,
                    ),
                    *(
                        (successful_memory_height_strata == stratum).sum().to(torch.float32)
                        for stratum in range(6)
                    ),
                    torch.tensor(
                        float(local_memory_coverage_ready),
                        dtype=torch.float32,
                        device=device,
                    ),
                )
            )
        if distributed:
            dist.all_reduce(replay_diagnostics, op=dist.ReduceOp.SUM)
            replay_diagnostics /= world_size
            dist.all_reduce(memory_diagnostics, op=dist.ReduceOp.SUM)
            memory_diagnostics /= world_size
        metrics = torch.stack(
            (
                local_reward,
                local_first,
                local_first_hand,
                local_recovery,
                local_second,
                local_second_hand,
                local_failed,
                local_lateral_displacement,
                local_lateral_speed,
                local_hand_displacement,
                local_hand_speed,
                local_root_angular,
                local_upper_authority,
                local_recenter_error,
                local_bimanual_reach,
                local_second_attempt,
                local_second_attempt_hand,
                *local_stratum_saves,
                *local_stratum_counts,
                local_quarantined,
                local_nonfinite_quarantine,
                *local_stratum_failures,
                *local_stratum_quarantines,
                environment._minimum_hand_target_distance.mean(),
                local_task_motion_reward,
                environment._first_decisive_pelvis_lateral_error.mean(),
                environment._first_decisive_hand_intercept_distance.mean(),
                (
                    environment._maximum_root_angular_speed
                    > world_config.root_angular_speed_soft_limit_rad_s
                )
                .to(torch.float32)
                .mean(),
                (environment._maximum_root_angular_speed > 3.50).to(torch.float32).mean(),
                local_recovery_progress_reward,
                local_qualified_first_save,
                cumulative_reach_reward.mean(),
                cumulative_bimanual_reach_reward.mean(),
                cumulative_upright_reward.mean(),
                cumulative_smoothness_penalty.mean(),
                cumulative_effort_penalty.mean(),
                cumulative_event_bonus.mean(),
                cumulative_safety_penalty.mean(),
                cumulative_nonfinite_override.mean(),
                cumulative_action_rate_penalty.mean(),
                cumulative_joint_acceleration_penalty.mean(),
                cumulative_root_linear_speed_penalty.mean(),
                cumulative_root_angular_speed_penalty.mean(),
                cumulative_root_angular_excess_penalty.mean(),
                cumulative_action_magnitude_penalty.mean(),
                episode_return[qualified_save_mask].sum(),
                qualified_save_mask.to(torch.float32).sum(),
                episode_return[failed_mask].sum(),
                failed_mask.to(torch.float32).sum(),
                episode_return[stable_miss_mask].sum(),
                stable_miss_mask.to(torch.float32).sum(),
                *local_side_saves,
                *local_side_qualified_saves,
                *local_side_counts,
                *local_side_failures,
            )
        )
        if distributed:
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            metrics /= world_size
            dist.all_reduce(maximum_root_angular, op=dist.ReduceOp.MAX)
        total_world_steps += (
            active.environments_per_rank
            * world_size
            * active.rollout_steps
            * world_config.physics_substeps
        )
        metric: dict[str, Any] = {
            "iteration": iteration + 1,
            "mean_episode_reward": float(metrics[0]),
            "first_save_rate": float(metrics[1]),
            "first_hand_save_rate": float(metrics[2]),
            "recovery_rate": float(metrics[3]),
            "second_save_rate": float(metrics[4]),
            "second_hand_save_rate": float(metrics[5]),
            "failed_rate": float(metrics[6]),
            "mean_maximum_lateral_displacement_m": float(metrics[7]),
            "mean_maximum_lateral_speed_mps": float(metrics[8]),
            "mean_maximum_hand_displacement_m": float(metrics[9]),
            "mean_maximum_hand_speed_mps": float(metrics[10]),
            "mean_maximum_root_angular_speed_rad_s": float(metrics[11]),
            "mean_minimum_upper_body_authority": float(metrics[12]),
            "mean_second_release_lateral_error_m": float(metrics[13]),
            "bimanual_reach_fraction": float(metrics[14]),
            "second_attempt_save_rate": float(metrics[15]),
            "second_attempt_hand_save_rate": float(metrics[16]),
            "maximum_root_angular_speed_rad_s": float(maximum_root_angular),
            "quarantined_rate": float(metrics[23]),
            "nonfinite_quarantine_rate": float(metrics[24]),
            "mean_minimum_hand_target_distance_m": float(metrics[31]),
            "mean_task_motion_reward": float(metrics[32]),
            "mean_first_decisive_pelvis_lateral_error_m": float(metrics[33]),
            "mean_first_decisive_hand_intercept_distance_m": float(metrics[34]),
            "root_angular_speed_soft_limit_rad_s": (
                world_config.root_angular_speed_soft_limit_rad_s
            ),
            "root_angular_speed_soft_limit_exceedance_rate": float(metrics[35]),
            "strict_stability_ceiling_rad_s": 3.50,
            "strict_stability_ceiling_exceedance_rate": float(metrics[36]),
            "mean_recovery_progress_reward": float(metrics[37]),
            "qualified_first_save_rate": float(metrics[38]),
            "mean_reach_reward": float(metrics[39]),
            "mean_bimanual_reach_reward": float(metrics[40]),
            "mean_upright_reward": float(metrics[41]),
            "mean_smoothness_penalty": float(metrics[42]),
            "mean_effort_penalty": float(metrics[43]),
            "mean_event_bonus": float(metrics[44]),
            "mean_safety_penalty": float(metrics[45]),
            "mean_nonfinite_override": float(metrics[46]),
            "mean_action_rate_penalty": float(metrics[47]),
            "mean_joint_acceleration_penalty": float(metrics[48]),
            "mean_root_linear_speed_penalty": float(metrics[49]),
            "mean_root_angular_speed_penalty": float(metrics[50]),
            "mean_root_angular_excess_penalty": float(metrics[51]),
            "mean_action_magnitude_penalty": float(metrics[52]),
            "mean_qualified_save_episode_reward": float(
                metrics[53] / torch.clamp(metrics[54], min=1.0)
            ),
            "mean_failed_episode_reward": float(metrics[55] / torch.clamp(metrics[56], min=1.0)),
            "mean_stable_miss_episode_reward": float(
                metrics[57] / torch.clamp(metrics[58], min=1.0)
            ),
            "policy_loss": policy_loss_value,
            "value_loss": value_loss_value,
            "entropy": entropy_value,
            "online_parent_distillation_loss": parent_distillation_loss_value,
            "successful_trajectory_replay_loss": (successful_trajectory_replay_loss_value),
            "support_landing_causal_recovery_only": (
                active.support_landing_causal_recovery_only_enabled
            ),
            "successful_trajectory_replay_strength": successful_memory_replay_strength,
            "successful_trajectory_replay_effective_coefficient": (
                effective_successful_trajectory_replay_coefficient
            ),
            "symmetry_mirror_loss": symmetry_mirror_loss_value,
            "successful_trajectory_replay_episode_rate": float(replay_diagnostics[0]),
            "successful_trajectory_replay_transition_fraction": float(replay_diagnostics[1]),
            "causal_recovery_transition_fraction": float(replay_diagnostics[2]),
            "successful_trajectory_memory_mean_rows_per_rank": float(memory_diagnostics[0]),
            "successful_trajectory_memory_mean_rows_by_height_per_rank": {
                "far_corner_low": float(memory_diagnostics[1] + memory_diagnostics[2]),
                "far_corner_mid": float(memory_diagnostics[3] + memory_diagnostics[4]),
                "far_corner_high": float(memory_diagnostics[5] + memory_diagnostics[6]),
            },
            "successful_trajectory_memory_mean_rows_by_side_per_rank": {
                "negative_far_corner": float(
                    memory_diagnostics[1] + memory_diagnostics[3] + memory_diagnostics[5]
                ),
                "positive_far_corner": float(
                    memory_diagnostics[2] + memory_diagnostics[4] + memory_diagnostics[6]
                ),
            },
            "successful_trajectory_memory_required_strata": list(required_successful_memory_strata),
            "successful_trajectory_memory_coverage_ready_rank_fraction": float(
                memory_diagnostics[7]
            ),
            "successful_trajectory_memory_coverage_ready_all_ranks": (
                successful_memory_coverage_ready
            ),
            "successful_trajectory_memory_minimum_episodes_per_stratum": (
                active.successful_trajectory_memory_minimum_episodes_per_stratum
            ),
            "successful_trajectory_memory_full_strength_episodes_per_stratum": (
                active.successful_trajectory_memory_full_strength_episodes_per_stratum
            ),
            "successful_trajectory_memory_episode_counts": {
                "far_corner_low_negative": int(successful_memory_episode_counts[0]),
                "far_corner_low_positive": int(successful_memory_episode_counts[1]),
                "far_corner_mid_negative": int(successful_memory_episode_counts[2]),
                "far_corner_mid_positive": int(successful_memory_episode_counts[3]),
                "far_corner_high_negative": int(successful_memory_episode_counts[4]),
                "far_corner_high_positive": int(successful_memory_episode_counts[5]),
            },
            "successful_trajectory_memory_local_new_episode_counts": (
                local_new_successful_episode_counts.tolist()
            ),
            "successful_trajectory_memory_global_new_episode_counts": (
                global_new_successful_episode_counts.tolist()
            ),
            "successful_trajectory_memory_local_new_rows": (local_new_successful_memory_rows),
            "successful_trajectory_memory_global_new_rows": (global_new_successful_memory_rows),
            "successful_trajectory_memory_ddp_shared": distributed,
            "physics_world_steps": total_world_steps,
        }
        training_strata = _stratum_rates(metrics)
        metric["first_save_rate_by_height"] = {
            "far_corner_low": training_strata[0],
            "far_corner_mid": training_strata[1],
            "far_corner_high": training_strata[2],
        }
        metric["minimum_first_save_stratum_rate"] = min(training_strata)
        metric["first_save_stratum_balance_score"] = _stratum_balance_score(training_strata)
        failed_strata = _stratum_rates(metrics, save_offset=25)
        quarantined_strata = _stratum_rates(metrics, save_offset=28)
        metric["failed_rate_by_height"] = {
            "far_corner_low": failed_strata[0],
            "far_corner_mid": failed_strata[1],
            "far_corner_high": failed_strata[2],
        }
        metric["quarantined_rate_by_height"] = {
            "far_corner_low": quarantined_strata[0],
            "far_corner_mid": quarantined_strata[1],
            "far_corner_high": quarantined_strata[2],
        }
        side_saves = _side_rates(metrics, event_offset=59, count_offset=63)
        side_qualified = _side_rates(metrics, event_offset=61, count_offset=63)
        side_failures = _side_rates(metrics, event_offset=65, count_offset=63)
        metric["first_save_rate_by_side"] = {
            "negative_far_corner": side_saves[0],
            "positive_far_corner": side_saves[1],
        }
        metric["qualified_first_save_rate_by_side"] = {
            "negative_far_corner": side_qualified[0],
            "positive_far_corner": side_qualified[1],
        }
        metric["failed_rate_by_side"] = {
            "negative_far_corner": side_failures[0],
            "positive_far_corner": side_failures[1],
        }
        metric["minimum_qualified_first_save_side_rate"] = min(side_qualified)
        metric["mean_reward_accounting_residual"] = _reward_accounting_residual(metric)
        metric["anatomical_selection_score"] = (
            metric["mean_episode_reward"]
            + active.first_save_selection_weight * metric["first_save_rate"]
            + active.qualified_first_save_selection_weight * metric["qualified_first_save_rate"]
            + active.first_hand_save_selection_weight * metric["first_hand_save_rate"]
            + active.second_attempt_save_selection_weight * metric["second_attempt_save_rate"]
            + active.second_attempt_hand_save_selection_weight
            * metric["second_attempt_hand_save_rate"]
            + active.second_save_selection_weight * metric["second_save_rate"]
            + active.second_hand_save_selection_weight * metric["second_hand_save_rate"]
            + active.hand_reach_selection_weight * metric["mean_maximum_hand_displacement_m"]
            + active.first_save_stratum_balance_selection_weight
            * metric["first_save_stratum_balance_score"]
            + active.first_save_side_balance_selection_weight
            * metric["minimum_qualified_first_save_side_rate"]
            - active.root_angular_speed_selection_penalty_weight
            * metric["mean_maximum_root_angular_speed_rad_s"]
            - active.second_release_recenter_selection_weight
            * metric["mean_second_release_lateral_error_m"]
            - active.hand_target_distance_selection_weight
            * metric["mean_minimum_hand_target_distance_m"]
        )
        metric["deterministic_candidate"] = deterministic_candidate
        if rank == 0:
            iteration_metrics.append(metric)
            safe_truth_key = _safe_continuation_truth_key(
                deterministic_candidate,
                maximum_root_angular_speed_rad_s=(
                    active.maximum_exploration_root_angular_speed_for_selection
                ),
            )
            if safe_truth_key is not None and (
                best_safe_truth_key is None or safe_truth_key > best_safe_truth_key
            ):
                if rollout_state is None:
                    raise RuntimeError("rank zero rollout state is unavailable")
                best_safe_truth_key = safe_truth_key
                best_safe_state = rollout_state
                best_safe_iteration = iteration + 1
                best_safe_metrics = dict(deterministic_candidate)
            exploration_truth_key = _exploration_truth_key(
                deterministic_candidate,
                save_first=active.save_first_exploration_selection_enabled,
            )
            candidate_regressed_from_exploration_champion = (
                best_exploration_truth_key is not None
                and exploration_truth_key < best_exploration_truth_key
            )
            metric["rolled_back_post_update_to_exploration_champion"] = bool(
                active.rollback_to_exploration_champion_on_regression_enabled
                and candidate_regressed_from_exploration_champion
            )
            print(json.dumps(metric, sort_keys=True), flush=True)
            if (
                best_exploration_truth_key is None
                or exploration_truth_key > best_exploration_truth_key
            ):
                if rollout_state is None:
                    raise RuntimeError("rank zero rollout state is unavailable")
                best_exploration_truth_key = exploration_truth_key
                best_exploration_state = rollout_state
                best_exploration_iteration = iteration + 1
                best_exploration_metrics = dict(deterministic_candidate)
                _atomic_torch_save(
                    torch=torch,
                    path=(
                        progress_output / "goalkeeper-physics-ppo-exploration-best-resume-only.pt"
                    ),
                    payload={
                        "state_dict": best_exploration_state,
                        "actor_architecture": "ZERO_OUTPUT_SPECIALIST_ADAPTER_V1",
                        "observation_size": environment.observation_size,
                        "action_size": environment.action_size,
                        "hidden_size": active.hidden_size,
                        "training_config": asdict(active),
                        "world_config": asdict(world_config),
                        "activation_ceiling": "SIM_ONLY",
                        "promotion_status": ("EXPLORATION_BEST_RESUME_ONLY_NOT_A_CANDIDATE"),
                        "promotion_eligible": False,
                        "checkpoint_weight_binding": ("PRE_UPDATE_ROLLOUT_STATE_EXACT"),
                        "selected_rollout_iteration": best_exploration_iteration,
                        "selection_metrics": best_exploration_metrics,
                        "selection_semantics": (
                            "WORST_SEED_QUALIFIED_SAVE_THEN_MAX_FAILURE_THEN_RAW_SAVE"
                        ),
                        "online_plastic_action_start_index": (min(plastic_action_indices)),
                        "online_plastic_action_indices": list(plastic_action_indices),
                        "online_plasticity_scope": plasticity_scope,
                    },
                )
            if _candidate_is_selectable(
                deterministic_candidate,
                config=active,
                best_truth_key=best_truth_key,
            ):
                best_reward = deterministic_candidate["mean_episode_reward"]
                best_selection_score = deterministic_candidate["anatomical_selection_score"]
                best_truth_key = _candidate_truth_key(deterministic_candidate)
                if rollout_state is None:
                    raise RuntimeError("rank zero rollout state is unavailable")
                best_state = rollout_state
                best_rollout_iteration = iteration + 1
            progress_path = progress_output / "training-progress.json"
            progress_temporary = progress_path.with_suffix(".json.tmp")
            progress_temporary.write_text(
                json.dumps(
                    {
                        "schema_version": ("rosclaw_soccer.goalkeeper_physics_ppo_progress.v19"),
                        "training_config_hash": active.config_hash,
                        "completed_iterations": iteration + 1,
                        "planned_iterations": active.iterations,
                        "latest": metric,
                        "best_safe_iteration": best_safe_iteration,
                        "best_safe_metrics": best_safe_metrics,
                        "best_exploration_iteration": best_exploration_iteration,
                        "best_exploration_metrics": best_exploration_metrics,
                        "exploration_champion_rollback_count": (
                            exploration_champion_rollback_count
                            + int(metric["rolled_back_post_update_to_exploration_champion"])
                        ),
                        "promotion_authority": False,
                        "activation_ceiling": "SIM_ONLY",
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            os.replace(progress_temporary, progress_path)

        rollback_flag = torch.tensor(
            int(
                rank == 0
                and active.rollback_to_exploration_champion_on_regression_enabled
                and candidate_regressed_from_exploration_champion
            ),
            dtype=torch.int32,
            device=device,
        )
        if distributed:
            dist.broadcast(rollback_flag, src=0)
        if bool(rollback_flag.item()):
            if rank == 0:
                if best_exploration_state is None:
                    raise RuntimeError("exploration champion state is unavailable for rollback")
                _load_actor_critic_state(model, best_exploration_state)
            if distributed:
                for parameter in model.parameters():
                    dist.broadcast(parameter.data, src=0)
                for buffer in model.buffers():
                    dist.broadcast(buffer.data, src=0)
            optimizer.state.clear()
            exploration_champion_rollback_count += 1
            final_model_is_rolled_back_champion = True
        else:
            final_model_is_rolled_back_champion = False

    parameter_max_difference = 0.0
    if distributed:
        # Every rank must participate in collectives.  Performing this after
        # non-zero ranks exit deadlocks rank zero during evidence finalization.
        for parameter in model.parameters():
            maximum = parameter.detach().clone()
            minimum = parameter.detach().clone()
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
            dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
            parameter_max_difference = max(
                parameter_max_difference, float(torch.max(torch.abs(maximum - minimum)))
            )
        dist.barrier()
    if rank != 0:
        if distributed:
            dist.destroy_process_group()
        return None
    output = progress_output
    # Preserve the final plastic state for an explicitly SIM_ONLY continuation
    # even when every rollout fails promotion.  This is not a candidate: it is
    # deliberately tagged as unevaluated post-update state so a later growth
    # cycle can resume learning without confusing it with accepted evidence.
    resume_checkpoint = output / "goalkeeper-physics-ppo-resume-only.pt"
    resume_temporary = resume_checkpoint.with_suffix(resume_checkpoint.suffix + ".tmp")
    resume_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    torch.save(
        {
            "state_dict": resume_state,
            "actor_architecture": "ZERO_OUTPUT_SPECIALIST_ADAPTER_V1",
            "observation_size": environment.observation_size,
            "action_size": environment.action_size,
            "hidden_size": active.hidden_size,
            "training_config": asdict(active),
            "world_config": asdict(world_config),
            "activation_ceiling": "SIM_ONLY",
            "promotion_status": "TRAINING_RESUME_ONLY_NOT_A_CANDIDATE",
            "promotion_eligible": False,
            "checkpoint_weight_binding": (
                "ROLLED_BACK_EXPLORATION_CHAMPION_PRE_UPDATE_EVALUATED"
                if final_model_is_rolled_back_champion
                else "FINAL_POST_UPDATE_UNEVALUATED"
            ),
            "selected_rollout_iteration": (
                best_exploration_iteration if final_model_is_rolled_back_champion else None
            ),
            "completed_iterations": active.iterations,
            "online_plastic_action_start_index": (min(plastic_action_indices)),
            "online_plastic_action_indices": list(plastic_action_indices),
            "online_plasticity_scope": plasticity_scope,
        },
        resume_temporary,
    )
    os.replace(resume_temporary, resume_checkpoint)
    safe_continuation_checkpoint: Path | None = None
    if best_safe_state is not None:
        safe_continuation_checkpoint = output / "goalkeeper-physics-ppo-safe-continuation.pt"
        torch.save(
            {
                "state_dict": best_safe_state,
                "actor_architecture": "ZERO_OUTPUT_SPECIALIST_ADAPTER_V1",
                "observation_size": environment.observation_size,
                "action_size": environment.action_size,
                "hidden_size": active.hidden_size,
                "training_config": asdict(active),
                "world_config": asdict(world_config),
                "activation_ceiling": "SIM_ONLY",
                "promotion_status": "SAFE_CONTINUATION_ONLY_NOT_A_CANDIDATE",
                "promotion_eligible": False,
                "checkpoint_weight_binding": "PRE_UPDATE_ROLLOUT_STATE_EXACT",
                "selected_rollout_iteration": best_safe_iteration,
                "selection_metrics": best_safe_metrics,
                "online_plastic_action_start_index": (min(plastic_action_indices)),
                "online_plastic_action_indices": list(plastic_action_indices),
                "online_plasticity_scope": plasticity_scope,
            },
            safe_continuation_checkpoint,
        )
    exploration_checkpoint: Path | None = None
    if best_exploration_state is not None:
        exploration_checkpoint = output / "goalkeeper-physics-ppo-exploration-best-resume-only.pt"
        _atomic_torch_save(
            torch=torch,
            path=exploration_checkpoint,
            payload={
                "state_dict": best_exploration_state,
                "actor_architecture": "ZERO_OUTPUT_SPECIALIST_ADAPTER_V1",
                "observation_size": environment.observation_size,
                "action_size": environment.action_size,
                "hidden_size": active.hidden_size,
                "training_config": asdict(active),
                "world_config": asdict(world_config),
                "activation_ceiling": "SIM_ONLY",
                "promotion_status": "EXPLORATION_BEST_RESUME_ONLY_NOT_A_CANDIDATE",
                "promotion_eligible": False,
                "checkpoint_weight_binding": "PRE_UPDATE_ROLLOUT_STATE_EXACT",
                "selected_rollout_iteration": best_exploration_iteration,
                "selection_metrics": best_exploration_metrics,
                "selection_semantics": ("WORST_SEED_QUALIFIED_SAVE_THEN_MAX_FAILURE_THEN_RAW_SAVE"),
                "online_plastic_action_start_index": (min(plastic_action_indices)),
                "online_plastic_action_indices": list(plastic_action_indices),
                "online_plasticity_scope": plasticity_scope,
            },
        )
    checkpoint: Path | None = None
    if best_state is not None:
        _load_actor_critic_state(model, best_state)
        checkpoint = output / "goalkeeper-physics-ppo-candidate.pt"
        torch.save(
            {
                "state_dict": best_state,
                "actor_architecture": "ZERO_OUTPUT_SPECIALIST_ADAPTER_V1",
                "observation_size": environment.observation_size,
                "action_size": environment.action_size,
                "hidden_size": active.hidden_size,
                "training_config": asdict(active),
                "world_config": asdict(world_config),
                "activation_ceiling": "SIM_ONLY",
                "promotion_status": "CANDIDATE_PENDING_CPU_MUJOCO_EXAM",
                "selected_rollout_iteration": best_rollout_iteration,
                "environment_summary": environment.summary(),
                "online_plastic_action_start_index": (min(plastic_action_indices)),
                "online_plastic_action_indices": list(plastic_action_indices),
                "online_plasticity_scope": plasticity_scope,
            },
            checkpoint,
        )
    elapsed = time.perf_counter() - started
    combat_curriculum_world_steps = (
        0
        if combat_curriculum_report is None
        else int(combat_curriculum_report["physics_world_steps"])
    )
    selection_outcome = _selection_outcome(
        checkpoint=checkpoint,
        best_reward=best_reward,
        best_selection_score=best_selection_score,
        best_rollout_iteration=best_rollout_iteration,
    )
    report = {
        "schema_version": "rosclaw_soccer.goalkeeper_physics_ppo_report.v37",
        "training_config": asdict(active),
        "training_config_hash": active.config_hash,
        "teacher_pretraining": teacher_report,
        "combat_gate_curriculum": combat_curriculum_report,
        "initialization": initialization_report,
        "world_config": asdict(world_config),
        "world_config_hash": world_config.config_hash,
        "world_size": world_size,
        "online_plastic_action_start_index": (min(plastic_action_indices)),
        "online_plastic_action_indices": list(plastic_action_indices),
        "online_plasticity_scope": plasticity_scope,
        "successful_trajectory_replay": {
            "coefficient": active.successful_trajectory_replay_coefficient,
            "recovery_only": active.successful_trajectory_recovery_only_enabled,
            "causal_recovery_transition_policy_weight": (active.recovery_transition_policy_weight),
            "causal_support_landing_plasticity": (
                "POST_TRUE_SAVE_EXPLORATION_AND_POLICY_GRADIENT_ONLY"
                if active.support_landing_causal_recovery_only_enabled
                else "DISABLED"
            ),
            "memory_capacity_per_height_stratum": (
                active.successful_trajectory_memory_capacity_per_stratum
            ),
            "memory_batch_size": active.successful_trajectory_memory_batch_size,
            "minimum_distinct_safe_episodes_per_required_stratum": (
                active.successful_trajectory_memory_minimum_episodes_per_stratum
            ),
            "full_strength_distinct_safe_episodes_per_required_stratum": (
                active.successful_trajectory_memory_full_strength_episodes_per_stratum
            ),
            "action_innovation_scale": (active.successful_trajectory_action_innovation_scale),
            "memory_lifetime": "CURRENT_TRAINING_RUN_ACROSS_ITERATIONS",
            "sampling": "HEIGHT_SIDE_STRATUM_BALANCED_WITH_REPLACEMENT",
            "readiness_gate": ("ALL_REQUIRED_STRATA_HAVE_ROWS_AND_MINIMUM_DISTINCT_SAFE_EPISODES"),
            "distributed_memory": (
                "NEW_QUALIFIED_ROWS_ALL_GATHERED_RANK_ORDERED"
                if distributed
                else "SINGLE_RANK_LOCAL"
            ),
            "label_source": (
                "ROLLOUT_POLICY_MEAN_PLUS_SHRUNK_OWN_STOCHASTIC_ACTION_INNOVATION_"
                "AFTER_PHYSICS_TRUE_SAVE"
            ),
            "eligibility": (
                "CAUSAL_RECOVERY_WINDOW_AND_SAVE_AND_NOT_QUARANTINED_AND_ANGULAR_SAFE"
                if active.successful_trajectory_recovery_only_enabled
                else "CAUSAL_OPTION_WINDOW_AND_SAVE_AND_NOT_QUARANTINED_AND_ANGULAR_SAFE"
            ),
            "authority": "TRAINING_LOSS_ONLY_SIM_ONLY",
        },
        "stability_plasticity": {
            "rollback_to_exploration_champion_on_regression": (
                active.rollback_to_exploration_champion_on_regression_enabled
            ),
            "regression_evidence": "THREE_FIXED_SEED_WORST_CASE_TRUTH_KEY",
            "rollback_state": "PRE_UPDATE_CHAMPION_STATE_PLUS_FRESH_OPTIMIZER",
            "rollback_count": exploration_champion_rollback_count,
            "final_model_is_rolled_back_champion": final_model_is_rolled_back_champion,
        },
        "gpu_devices": [torch.cuda.get_device_name(index) for index in range(world_size)],
        "synchronized_ddp": distributed,
        "maximum_cross_rank_parameter_difference": parameter_max_difference,
        "iterations": iteration_metrics,
        "physics_world_steps": (
            total_world_steps + deterministic_selection_world_steps + combat_curriculum_world_steps
        ),
        "stochastic_training_world_steps": total_world_steps,
        "deterministic_selection_world_steps": deterministic_selection_world_steps,
        "deterministic_selection_seed_schedule": "FIXED_HELD_OUT_PANEL_PER_RANK",
        "deterministic_selection_seed_count": (active.deterministic_selection_seed_count),
        "deterministic_selection_seed_stride": (active.deterministic_selection_seed_stride),
        "deterministic_selection_seed_base": _held_out_selection_seed(
            random_seed=active.random_seed,
            rank=0,
        ),
        "combat_curriculum_world_steps": combat_curriculum_world_steps,
        "resume_only_checkpoint": resume_checkpoint.name,
        "resume_only_checkpoint_hash": hash_bytes(resume_checkpoint.read_bytes()),
        "resume_only_checkpoint_promotion_eligible": False,
        "safe_continuation_checkpoint": (
            None if safe_continuation_checkpoint is None else safe_continuation_checkpoint.name
        ),
        "safe_continuation_checkpoint_hash": (
            None
            if safe_continuation_checkpoint is None
            else hash_bytes(safe_continuation_checkpoint.read_bytes())
        ),
        "safe_continuation_checkpoint_promotion_eligible": False,
        "safe_continuation_iteration": best_safe_iteration,
        "safe_continuation_metrics": best_safe_metrics,
        "exploration_best_resume_checkpoint": (
            None if exploration_checkpoint is None else exploration_checkpoint.name
        ),
        "exploration_best_resume_checkpoint_hash": (
            None
            if exploration_checkpoint is None
            else hash_bytes(exploration_checkpoint.read_bytes())
        ),
        "exploration_best_resume_iteration": best_exploration_iteration,
        "exploration_best_resume_metrics": best_exploration_metrics,
        "exploration_best_resume_promotion_eligible": False,
        "policy_samples": active.environments_per_rank
        * world_size
        * active.rollout_steps
        * active.iterations,
        "elapsed_sec": elapsed,
        **selection_outcome,
        "candidate_checkpoint_hash": (
            None if checkpoint is None else hash_bytes(checkpoint.read_bytes())
        ),
        "checkpoint_metric_weight_binding": ("PRE_UPDATE_DETERMINISTIC_ROLLOUT_STATE_EXACT"),
        "physics_backend": "mujoco_warp",
        "lower_body_authority": environment.lower_body_authority,
        "learned_residual_authority": environment.learned_residual_authority,
        "external_combat_teacher": getattr(environment, "teacher_report", None),
        "environment_summary": environment.summary(),
        "multi_step_episode_training": True,
        "continuous_recovery_training": True,
        "strict_cpu_mujoco_evaluation_completed": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    sanitized_report, nonfinite_paths = sanitize_nonfinite_evidence(report)
    if not isinstance(sanitized_report, dict):
        raise RuntimeError("goalkeeper PPO evidence sanitization changed the root type")
    report = sanitized_report
    report["diagnostic_nonfinite_paths"] = nonfinite_paths
    report["diagnostic_complete"] = not nonfinite_paths
    report["report_hash"] = hash_json(report)
    _atomic_json(output / "training-report.json", report)
    if distributed:
        dist.destroy_process_group()
    return report


def evaluate_goalkeeper_physics_candidate(
    *,
    asset_root: Path,
    locomotion_policy_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    seeds: tuple[int, ...] = (81_013, 81_029, 81_043, 81_071),
    environment_count: int = 64,
    difficulty_profile: str = "standard",
) -> dict[str, Any]:
    """Compare a frozen candidate with a zero-residual baseline on held-out shots."""

    import torch
    from torch import nn

    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("physics PPO evaluation seeds must be non-empty and unique")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    checkpoint = torch.load(
        checkpoint_path.expanduser().resolve(), map_location=device, weights_only=True
    )
    training_config_payload = checkpoint.get("training_config", {})
    if not isinstance(training_config_payload, dict):
        raise ValueError("physics PPO checkpoint training config is invalid")
    training_config = GoalkeeperPhysicsPPOConfig(**training_config_payload)
    model = _build_actor_critic(
        torch,
        nn,
        int(checkpoint["observation_size"]),
        int(checkpoint["action_size"]),
        int(checkpoint["hidden_size"]),
    ).to(device)
    _load_actor_critic_state(model, checkpoint["state_dict"])
    model.eval()
    world_config = goalkeeper_world_config(
        difficulty_profile=difficulty_profile,  # type: ignore[arg-type]
        environment_count=environment_count,
        shot_intent_cue_enabled=training_config.shot_intent_cue_enabled,
        unsafe_penalty=training_config.training_unsafe_penalty,
        save_then_unsafe_penalty=(training_config.training_save_then_unsafe_penalty),
        task_motion_reward_scale=training_config.training_task_motion_reward_scale,
        recovery_event_bonus=training_config.training_recovery_event_bonus,
    )
    world_config = _with_first_shot_release_override(
        world_config,
        training_config.training_first_shot_release_sec,
    )
    world_config = _with_episode_duration_override(
        world_config,
        training_config.training_episode_duration_sec,
    )
    world_config = _with_save_event_bonus_override(
        world_config,
        training_config.training_true_save_bonus,
        training_config.training_hand_save_bonus,
    )
    world_config = _with_root_angular_penalty_override(
        world_config,
        training_config.training_root_angular_speed_penalty_scale,
        training_config.training_root_angular_speed_soft_limit_rad_s,
        training_config.training_root_angular_speed_excess_penalty_scale,
        training_config.training_flight_root_angular_penalty_scale,
    )
    world_config = _with_recovery_progress_override(
        world_config,
        training_config.training_recovery_progress_reward_scale,
        training_config.training_recovery_progress_linear_speed_decay,
        training_config.training_recovery_progress_angular_speed_decay,
    )

    def run(policy_name: str) -> dict[str, Any]:
        environment = _build_environment(
            active=training_config,
            asset_root=asset_root,
            locomotion_policy_path=locomotion_policy_path,
            device=device,
            world_config=world_config,
        )
        totals = {
            "first_save": 0.0,
            "first_hand_save": 0.0,
            "recovery": 0.0,
            "second_save": 0.0,
            "second_hand_save": 0.0,
            "second_attempt_save": 0.0,
            "second_attempt_hand_save": 0.0,
            "failed": 0.0,
        }
        mean_episode_rewards: list[float] = []
        peak_requested_action_steps: list[float] = []
        peak_applied_action_steps: list[float] = []
        maximum_lateral_displacements: list[float] = []
        maximum_lateral_speeds: list[float] = []
        maximum_hand_displacements: list[float] = []
        maximum_hand_speeds: list[float] = []
        maximum_root_angular_speeds: list[float] = []
        second_release_lateral_errors: list[float] = []
        minimum_upper_body_authorities: list[float] = []
        bimanual_reach_steps = 0.0
        active_flight_steps = 0.0
        for seed in seeds:
            observation = environment.reset(seed=seed)
            cumulative_reward = torch.zeros(environment_count, device=device)
            previous_requested_action = torch.zeros(
                (environment_count, environment.action_size), device=device
            )
            previous_applied_action = torch.zeros_like(previous_requested_action)
            peak_requested_step = 0.0
            peak_applied_step = 0.0
            for _ in range(world_config.episode_steps):
                if policy_name == "candidate":
                    with torch.no_grad():
                        mean, _, _ = model(observation)
                        action = torch.tanh(mean)
                else:
                    action = torch.zeros_like(previous_requested_action)
                peak_requested_step = max(
                    peak_requested_step,
                    float(torch.max(torch.abs(action - previous_requested_action)).detach()),
                )
                observation, reward, _, info = environment.step(action)
                applied_action = info["applied_action"]
                peak_applied_step = max(
                    peak_applied_step,
                    float(torch.max(torch.abs(applied_action - previous_applied_action)).detach()),
                )
                cumulative_reward += reward
                previous_requested_action = action
                previous_applied_action = applied_action
            totals["first_save"] += float(environment.task.first_save.to(torch.float32).sum())
            totals["first_hand_save"] += float(
                environment.task.first_hand_save.to(torch.float32).sum()
            )
            totals["recovery"] += float(
                environment.task.recovered_after_first.to(torch.float32).sum()
            )
            totals["second_save"] += float(environment.task.second_save.to(torch.float32).sum())
            totals["second_hand_save"] += float(
                environment.task.second_hand_save.to(torch.float32).sum()
            )
            totals["second_attempt_save"] += float(
                environment.task.second_attempt_save.to(torch.float32).sum()
            )
            totals["second_attempt_hand_save"] += float(
                environment.task.second_attempt_hand_save.to(torch.float32).sum()
            )
            totals["failed"] += float((environment.task.phase == 7).to(torch.float32).sum())
            mean_episode_rewards.append(float(cumulative_reward.mean()))
            peak_requested_action_steps.append(peak_requested_step)
            peak_applied_action_steps.append(peak_applied_step)
            maximum_lateral_displacements.extend(
                float(value) for value in environment._maximum_lateral_displacement.detach().cpu()
            )
            maximum_lateral_speeds.extend(
                float(value) for value in environment._maximum_lateral_speed.detach().cpu()
            )
            maximum_hand_displacements.extend(
                float(value) for value in environment._maximum_hand_displacement.detach().cpu()
            )
            maximum_hand_speeds.extend(
                float(value) for value in environment._maximum_hand_speed.detach().cpu()
            )
            maximum_root_angular_speeds.extend(
                float(value) for value in environment._maximum_root_angular_speed.detach().cpu()
            )
            second_release_lateral_errors.extend(
                float(value) for value in environment._second_release_lateral_error.detach().cpu()
            )
            minimum_upper_body_authorities.extend(
                float(value) for value in environment._minimum_upper_body_authority.detach().cpu()
            )
            bimanual_reach_steps += float(environment.task._bimanual_reach_steps.sum())
            active_flight_steps += float(environment.task._active_flight_steps.sum())
            if not environment.finite_state():
                raise FloatingPointError("held-out MJWarp evaluation produced non-finite state")
        episodes = environment_count * len(seeds)
        return {
            "policy": policy_name,
            "episodes": episodes,
            "mean_episode_reward": sum(mean_episode_rewards) / len(mean_episode_rewards),
            "first_save_rate": totals["first_save"] / episodes,
            "first_hand_save_rate": totals["first_hand_save"] / episodes,
            "recovery_rate": totals["recovery"] / episodes,
            "second_save_rate": totals["second_save"] / episodes,
            "second_hand_save_rate": totals["second_hand_save"] / episodes,
            "second_attempt_save_rate": totals["second_attempt_save"] / episodes,
            "second_attempt_hand_save_rate": totals["second_attempt_hand_save"] / episodes,
            "failed_rate": totals["failed"] / episodes,
            "maximum_requested_actor_action_step": max(peak_requested_action_steps),
            "maximum_applied_actor_action_step": max(peak_applied_action_steps),
            "mean_maximum_lateral_displacement_m": sum(maximum_lateral_displacements)
            / len(maximum_lateral_displacements),
            "mean_maximum_lateral_speed_mps": sum(maximum_lateral_speeds)
            / len(maximum_lateral_speeds),
            "mean_maximum_hand_displacement_m": sum(maximum_hand_displacements)
            / len(maximum_hand_displacements),
            "mean_maximum_hand_speed_mps": sum(maximum_hand_speeds) / len(maximum_hand_speeds),
            "p95_maximum_hand_speed_mps": float(
                torch.quantile(torch.tensor(maximum_hand_speeds), 0.95)
            ),
            "maximum_root_angular_speed_rad_s": max(maximum_root_angular_speeds),
            "p95_root_angular_speed_rad_s": float(
                torch.quantile(torch.tensor(maximum_root_angular_speeds), 0.95)
            ),
            "mean_second_release_lateral_error_m": sum(second_release_lateral_errors)
            / len(second_release_lateral_errors),
            "mean_minimum_upper_body_authority": sum(minimum_upper_body_authorities)
            / len(minimum_upper_body_authorities),
            "bimanual_reach_fraction": bimanual_reach_steps / max(1.0, active_flight_steps),
            "finite_state": True,
        }

    baseline = run("frozen_locomotion_baseline")
    candidate = run("candidate")
    passed = bool(
        candidate["failed_rate"] == 0.0
        and candidate["first_save_rate"] >= baseline["first_save_rate"]
        and candidate["first_hand_save_rate"] >= baseline["first_hand_save_rate"]
        and candidate["recovery_rate"] >= baseline["recovery_rate"]
        and candidate["second_attempt_save_rate"] >= baseline["second_attempt_save_rate"]
        and candidate["second_attempt_hand_save_rate"] >= baseline["second_attempt_hand_save_rate"]
        and candidate["second_save_rate"] >= baseline["second_save_rate"]
        and candidate["second_hand_save_rate"] >= baseline["second_hand_save_rate"]
        and candidate["mean_episode_reward"] > baseline["mean_episode_reward"]
        and candidate["maximum_applied_actor_action_step"]
        <= world_config.maximum_applied_actor_action_step + 1.0e-6
        and candidate["maximum_root_angular_speed_rad_s"] <= 3.50
        and candidate["p95_root_angular_speed_rad_s"] <= 3.20
        and candidate["mean_second_release_lateral_error_m"] <= 0.55
        and candidate["mean_maximum_hand_displacement_m"]
        >= baseline["mean_maximum_hand_displacement_m"] + 0.01
        and candidate["p95_maximum_hand_speed_mps"] <= 5.0
    )
    report = {
        "schema_version": "rosclaw_soccer.goalkeeper_physics_holdout.v2",
        "physics_backend": "mujoco_warp_held_out",
        "seeds": list(seeds),
        "world_config_hash": world_config.config_hash,
        "candidate_checkpoint": checkpoint_path.expanduser().resolve().name,
        "baseline": baseline,
        "candidate": candidate,
        "passed": passed,
        "promotion_status": (
            "CANDIDATE_PENDING_CPU_MUJOCO_EXAM" if passed else "REJECTED_BY_GPU_PHYSICS_HOLDOUT"
        ),
        "strict_cpu_mujoco_evaluation_completed": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    _atomic_json(output_path.expanduser().resolve(), report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--locomotion-policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--environments-per-rank", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--deterministic-selection-seed-count", type=int, default=1)
    parser.add_argument("--deterministic-selection-seed-stride", type=int, default=1009)
    parser.add_argument("--rollout-steps", type=int, default=250)
    parser.add_argument("--training-episode-duration-sec", type=float)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatches", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--policy-anchor-coefficient", type=float, default=0.010)
    parser.add_argument("--random-seed", type=int, default=1207)
    parser.add_argument("--first-hand-save-selection-weight", type=float, default=40.0)
    parser.add_argument("--first-save-selection-weight", type=float, default=100.0)
    parser.add_argument("--qualified-first-save-selection-weight", type=float, default=0.0)
    parser.add_argument("--second-attempt-save-selection-weight", type=float, default=100.0)
    parser.add_argument("--second-attempt-hand-save-selection-weight", type=float, default=120.0)
    parser.add_argument("--second-save-selection-weight", type=float, default=100.0)
    parser.add_argument("--second-hand-save-selection-weight", type=float, default=120.0)
    parser.add_argument("--hand-reach-selection-weight", type=float, default=15.0)
    parser.add_argument("--hand-target-distance-selection-weight", type=float, default=0.0)
    parser.add_argument("--first-save-stratum-balance-selection-weight", type=float, default=0.0)
    parser.add_argument("--first-save-side-balance-selection-weight", type=float, default=0.0)
    parser.add_argument("--minimum-first-save-rate-for-selection", type=float, default=0.0)
    parser.add_argument("--minimum-first-save-stratum-rate-for-selection", type=float, default=0.0)
    parser.add_argument("--minimum-hand-displacement-for-selection-m", type=float, default=0.0)
    parser.add_argument("--root-angular-speed-selection-penalty-weight", type=float, default=0.0)
    parser.add_argument("--second-release-recenter-selection-weight", type=float, default=0.0)
    parser.add_argument(
        "--maximum-exploration-root-angular-speed-for-selection", type=float, default=3.50
    )
    parser.add_argument("--training-second-shot-probability", type=float, default=0.75)
    parser.add_argument("--training-first-shot-release-sec", type=float)
    parser.add_argument("--training-hard-shot-fraction", type=float, default=0.0)
    parser.add_argument(
        "--training-hard-shot-height-mode",
        choices=("low", "mid", "high", "balanced"),
        default="high",
    )
    parser.add_argument(
        "--training-hard-shot-side-mode",
        choices=("negative", "positive", "balanced"),
        default="balanced",
    )
    parser.add_argument(
        "--training-reach-reward-semantics",
        choices=("STATE_DENSITY", "POTENTIAL_PROGRESS_ONLY"),
        default="STATE_DENSITY",
    )
    parser.add_argument(
        "--training-hard-shot-flight-time-range-sec",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
    )
    parser.add_argument("--training-hard-height-reach-reward-scale", type=float, default=0.0)
    parser.add_argument("--training-hard-height-reach-threshold-m", type=float, default=1.10)
    parser.add_argument("--training-hard-height-reach-distance-decay", type=float, default=1.25)
    parser.add_argument("--training-task-motion-reward-scale", type=float, default=0.0)
    parser.add_argument("--training-recovery-progress-reward-scale", type=float, default=0.0)
    parser.add_argument("--training-true-save-bonus", type=float, default=25.0)
    parser.add_argument("--training-hand-save-bonus", type=float)
    parser.add_argument("--training-recovery-event-bonus", type=float, default=15.0)
    parser.add_argument("--training-recovery-progress-linear-speed-decay", type=float, default=2.0)
    parser.add_argument(
        "--training-recovery-progress-angular-speed-decay", type=float, default=0.50
    )
    parser.add_argument("--training-root-angular-speed-penalty-scale", type=float)
    parser.add_argument("--training-root-angular-speed-soft-limit-rad-s", type=float, default=3.50)
    parser.add_argument(
        "--training-root-angular-speed-excess-penalty-scale", type=float, default=0.0
    )
    parser.add_argument("--training-flight-root-angular-penalty-scale", type=float, default=1.0)
    parser.add_argument("--training-unsafe-penalty", type=float, default=50.0)
    parser.add_argument("--training-save-then-unsafe-penalty", type=float, default=0.0)
    parser.add_argument("--teacher-pretraining-samples", type=int, default=32_768)
    parser.add_argument("--teacher-pretraining-epochs", type=int, default=20)
    parser.add_argument("--teacher-parent-replay-coefficient", type=float, default=0.50)
    parser.add_argument("--task-space-reach-blend", type=float, default=0.0)
    parser.add_argument("--task-space-reach-atlas", action="store_true")
    parser.add_argument("--runtime-task-space-reach", action="store_true")
    parser.add_argument("--runtime-task-space-reach-blend", type=float, default=0.0)
    parser.add_argument("--second-shot-reach-multiplier", type=float, default=1.0)
    parser.add_argument("--online-parent-distillation-coefficient", type=float, default=0.0)
    parser.add_argument("--successful-trajectory-replay-coefficient", type=float, default=0.0)
    parser.add_argument("--successful-trajectory-mirror-augmentation", action="store_true")
    parser.add_argument("--successful-trajectory-recovery-only", action="store_true")
    parser.add_argument("--symmetry-mirror-loss-coefficient", type=float, default=0.0)
    parser.add_argument("--recovery-transition-policy-weight", type=float, default=1.0)
    parser.add_argument("--support-landing-causal-recovery-only", action="store_true")
    parser.add_argument(
        "--rollback-to-exploration-champion-on-regression",
        action="store_true",
    )
    parser.add_argument(
        "--save-first-exploration-selection",
        action="store_true",
        help=(
            "rank development-only exploration checkpoints by real save rate before "
            "posture quality; never relaxes promotion or finite-state gates"
        ),
    )
    parser.add_argument("--successful-trajectory-memory-capacity-per-stratum", type=int, default=0)
    parser.add_argument("--successful-trajectory-memory-batch-size", type=int, default=256)
    parser.add_argument(
        "--successful-trajectory-memory-minimum-episodes-per-stratum",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--successful-trajectory-memory-full-strength-episodes-per-stratum",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--successful-trajectory-action-innovation-scale",
        type=float,
        default=1.0,
    )
    parser.add_argument("--arm-only-online-update", action="store_true")
    parser.add_argument("--lower-body-and-arms-online-update", action="store_true")
    parser.add_argument("--lateral-drive-gate-only-online-update", action="store_true")
    parser.add_argument("--support-landing-online-update", action="store_true")
    parser.add_argument(
        "--shot-difficulty-profile",
        choices=("standard", "match", "advanced", "elite"),
        default="standard",
    )
    parser.add_argument("--initialization-checkpoint", type=Path)
    parser.add_argument("--combat-teacher-checkout", type=Path)
    parser.add_argument("--combat-teacher-checkpoint", type=Path)
    parser.add_argument("--targeted-dive-checkpoint", type=Path)
    parser.add_argument("--targeted-dive-option-duration-sec", type=float, default=0.90)
    parser.add_argument("--targeted-dive-phase-hold-sec", type=float, default=0.0)
    parser.add_argument("--targeted-dive-actor-recovery-plasticity-sec", type=float, default=0.0)
    parser.add_argument(
        "--targeted-dive-actor-recovery-residual-authority-scale",
        type=float,
        default=0.50,
    )
    parser.add_argument("--targeted-dive-post-save-counterstep", action="store_true")
    parser.add_argument(
        "--targeted-dive-post-save-counterstep-duration-sec", type=float, default=0.80
    )
    parser.add_argument(
        "--targeted-dive-post-save-counterstep-command-limit", type=float, default=0.55
    )
    parser.add_argument(
        "--targeted-dive-post-save-counterstep-capture-horizon-sec",
        type=float,
        default=0.28,
    )
    parser.add_argument(
        "--targeted-dive-post-save-counterstep-recenter-weight", type=float, default=1.0
    )
    parser.add_argument("--targeted-dive-post-save-option-release-sec", type=float, default=0.30)
    parser.add_argument("--targeted-dive-post-save-fall-recovery", action="store_true")
    parser.add_argument(
        "--targeted-dive-post-save-fall-recovery-duration-sec", type=float, default=1.50
    )
    parser.add_argument(
        "--targeted-dive-post-save-fall-minimum-pelvis-height-m", type=float, default=0.12
    )
    parser.add_argument(
        "--targeted-dive-post-save-fall-minimum-upright-projection",
        type=float,
        default=-0.95,
    )
    parser.add_argument(
        "--targeted-dive-post-save-fall-maximum-root-linear-speed-mps",
        type=float,
        default=3.50,
    )
    parser.add_argument(
        "--targeted-dive-post-save-fall-maximum-root-angular-speed-rad-s",
        type=float,
        default=10.0,
    )
    parser.add_argument("--targeted-dive-prediction-lead-sec", type=float, default=0.30)
    parser.add_argument("--targeted-dive-nominal-shot-flight-time-sec", type=float, default=0.47)
    parser.add_argument("--targeted-dive-intercept-phase-at-arrival", type=float)
    parser.add_argument(
        "--targeted-dive-phase-sync-minimum-target-height-m", type=float, default=0.60
    )
    parser.add_argument("--targeted-dive-posture-exception-duration-sec", type=float, default=1.55)
    parser.add_argument(
        "--targeted-dive-root-angular-speed-guard-ceiling-rad-s",
        type=float,
        default=8.0,
    )
    parser.add_argument("--targeted-dive-decoder-residual-authority", type=float, default=0.10)
    parser.add_argument("--targeted-dive-decoder-lower-body-residual-authority", type=float)
    parser.add_argument("--targeted-dive-decoder-lower-body-command-scale", type=float)
    parser.add_argument("--targeted-dive-decoder-waist-residual-authority", type=float)
    parser.add_argument("--targeted-dive-decoder-arm-residual-authority", type=float)
    parser.add_argument("--targeted-dive-actor-residual-scale", type=float, default=0.70)
    parser.add_argument("--targeted-dive-anchor-lower-body-scale", type=float, default=0.25)
    parser.add_argument("--targeted-dive-anchor-waist-scale", type=float, default=0.50)
    parser.add_argument("--targeted-dive-anchor-arm-scale", type=float, default=1.00)
    parser.add_argument("--targeted-dive-minimum-option-gate", type=float, default=0.0)
    parser.add_argument("--targeted-dive-runtime-reach-blend", type=float, default=0.0)
    parser.add_argument("--targeted-dive-runtime-reach-feedback-blend", type=float, default=0.0)
    parser.add_argument("--targeted-dive-runtime-reach-feedback-gain", type=float, default=0.70)
    parser.add_argument(
        "--targeted-dive-runtime-reach-feedback-maximum-error-m",
        type=float,
        default=0.30,
    )
    parser.add_argument(
        "--targeted-dive-runtime-reach-feedback-support-scale",
        type=float,
        default=0.0,
    )
    parser.add_argument("--targeted-dive-runtime-contact-support-side", action="store_true")
    parser.add_argument("--targeted-dive-actor-contact-support-side", action="store_true")
    parser.add_argument("--targeted-dive-actor-recovery-context", action="store_true")
    parser.add_argument("--targeted-dive-runtime-whole-body-reach-blend", type=float, default=0.0)
    parser.add_argument(
        "--targeted-dive-runtime-whole-body-reach-full-below-height-m",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--targeted-dive-runtime-whole-body-reach-maximum-height-m",
        type=float,
        default=0.65,
    )
    parser.add_argument(
        "--targeted-dive-runtime-whole-body-reach-waist-scale", type=float, default=0.75
    )
    parser.add_argument(
        "--targeted-dive-runtime-whole-body-reach-arm-scale", type=float, default=1.0
    )
    parser.add_argument(
        "--targeted-dive-runtime-whole-body-reach-support-scale", type=float, default=0.65
    )
    parser.add_argument(
        "--targeted-dive-runtime-whole-body-reach-release-sec", type=float, default=0.60
    )
    parser.add_argument("--targeted-dive-runtime-reach-contact-standoff-m", type=float, default=0.0)
    parser.add_argument("--targeted-dive-runtime-reach-lateral-lead-m", type=float, default=0.0)
    parser.add_argument("--targeted-dive-runtime-reach-vertical-lead-m", type=float, default=0.0)
    parser.add_argument("--targeted-dive-runtime-reach-low-vertical-lead-m", type=float)
    parser.add_argument("--targeted-dive-runtime-reach-mid-vertical-lead-m", type=float)
    parser.add_argument("--targeted-dive-runtime-reach-high-vertical-lead-m", type=float)
    parser.add_argument("--targeted-dive-overhead-reach-prior", type=Path)
    parser.add_argument("--targeted-dive-overhead-reach-blend", type=float, default=0.0)
    parser.add_argument(
        "--targeted-dive-overhead-reach-minimum-target-height-m",
        type=float,
        default=1.10,
    )
    parser.add_argument(
        "--targeted-dive-overhead-reach-full-target-height-m",
        type=float,
        default=1.25,
    )
    parser.add_argument("--targeted-dive-overhead-reach-lower-body-scale", type=float, default=0.0)
    parser.add_argument("--targeted-dive-overhead-reach-waist-scale", type=float, default=0.25)
    parser.add_argument("--targeted-dive-overhead-reach-arm-scale", type=float, default=1.0)
    parser.add_argument("--targeted-dive-mosaic-gmt-model", type=Path)
    parser.add_argument("--targeted-dive-mosaic-gmt-skill", type=Path)
    parser.add_argument("--targeted-dive-mosaic-gmt-blend", type=float, default=0.0)
    parser.add_argument("--targeted-dive-mosaic-gmt-stability-floor", type=float, default=0.0)
    parser.add_argument(
        "--targeted-dive-mosaic-gmt-minimum-target-height-m",
        type=float,
        default=1.10,
    )
    parser.add_argument(
        "--targeted-dive-mosaic-gmt-full-target-height-m",
        type=float,
        default=1.25,
    )
    parser.add_argument("--targeted-dive-mosaic-gmt-lower-body-scale", type=float, default=1.0)
    parser.add_argument("--targeted-dive-mosaic-gmt-waist-scale", type=float, default=1.0)
    parser.add_argument("--targeted-dive-mosaic-gmt-arm-scale", type=float, default=0.80)
    parser.add_argument("--targeted-dive-mosaic-gmt-getup-skill", type=Path)
    parser.add_argument("--targeted-dive-mosaic-gmt-getup-blend", type=float, default=0.0)
    parser.add_argument(
        "--targeted-dive-mosaic-gmt-getup-activation-maximum-pelvis-height-m",
        type=float,
        default=0.50,
    )
    parser.add_argument("--targeted-dive-mosaic-gmt-getup-blend-in-sec", type=float, default=0.35)
    parser.add_argument(
        "--targeted-dive-mosaic-gmt-getup-reference-feedforward-blend",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--targeted-dive-mosaic-gmt-getup-lower-body-scale", type=float, default=1.0
    )
    parser.add_argument("--targeted-dive-mosaic-gmt-getup-waist-scale", type=float, default=1.0)
    parser.add_argument("--targeted-dive-mosaic-gmt-getup-arm-scale", type=float, default=1.0)
    parser.add_argument("--targeted-dive-maximum-arm-target-step-rad", type=float, default=0.10)
    parser.add_argument("--targeted-dive-arm-target-filter-fraction", type=float, default=0.50)
    parser.add_argument(
        "--targeted-dive-maximum-lower-body-target-step-rad", type=float, default=0.08
    )
    parser.add_argument(
        "--targeted-dive-lower-body-target-filter-fraction", type=float, default=0.35
    )
    parser.add_argument("--targeted-dive-lateral-drive-scale", type=float, default=0.0)
    parser.add_argument(
        "--targeted-dive-negative-target-lateral-drive-scale", type=float, default=1.0
    )
    parser.add_argument(
        "--targeted-dive-lateral-drive-full-activation-gate", type=float, default=0.30
    )
    parser.add_argument("--targeted-dive-lateral-drive-capture", action="store_true")
    parser.add_argument(
        "--targeted-dive-lateral-drive-capture-horizon-sec", type=float, default=0.35
    )
    parser.add_argument("--targeted-dive-lateral-drive-target-standoff-m", type=float, default=0.32)
    parser.add_argument("--targeted-dive-lateral-drive-capture-scale-m", type=float, default=0.45)
    parser.add_argument("--targeted-dive-lateral-drive-learned-gate", action="store_true")
    parser.add_argument("--targeted-dive-canonical-locomotion-mirror", action="store_true")
    parser.add_argument("--targeted-dive-runtime-lateral-lunge-blend", type=float, default=0.0)
    parser.add_argument(
        "--targeted-dive-runtime-lateral-lunge-hip-roll-rad", type=float, default=0.18
    )
    parser.add_argument(
        "--targeted-dive-runtime-lateral-lunge-ankle-roll-rad", type=float, default=0.12
    )
    parser.add_argument(
        "--targeted-dive-runtime-lateral-lunge-approach-horizon-sec",
        type=float,
        default=0.90,
    )
    parser.add_argument("--targeted-dive-substep-upper-body-guard", action="store_true")
    parser.add_argument(
        "--targeted-dive-substep-upper-body-guard-onset-rad-s", type=float, default=1.80
    )
    parser.add_argument(
        "--targeted-dive-substep-upper-body-guard-ceiling-rad-s", type=float, default=3.00
    )
    parser.add_argument(
        "--targeted-dive-substep-upper-body-minimum-position-scale", type=float, default=0.05
    )
    parser.add_argument("--targeted-dive-substep-option-lower-body-guard", action="store_true")
    parser.add_argument(
        "--targeted-dive-substep-option-lower-body-guard-onset-rad-s",
        type=float,
        default=2.40,
    )
    parser.add_argument(
        "--targeted-dive-substep-option-lower-body-guard-ceiling-rad-s",
        type=float,
        default=3.30,
    )
    parser.add_argument(
        "--targeted-dive-substep-option-lower-body-minimum-scale",
        type=float,
        default=0.0,
    )
    parser.add_argument("--targeted-dive-low-shot-phase-scale", type=float, default=1.0)
    parser.add_argument("--targeted-dive-mid-shot-phase-scale", type=float, default=1.0)
    parser.add_argument("--targeted-dive-high-shot-phase-scale", type=float, default=1.0)
    parser.add_argument("--targeted-dive-initial-gate", type=float, default=0.40)
    parser.add_argument("--maximum-combat-teacher-blend", type=float, default=0.35)
    parser.add_argument("--combat-teacher-intercept-conditioning", action="store_true")
    parser.add_argument("--mobility-option", action="store_true")
    parser.add_argument("--mobility-lateral-command-limit", type=float, default=0.75)
    parser.add_argument("--mobility-recovery-command-limit", type=float, default=0.55)
    parser.add_argument("--mobility-residual-plasticity-scale", type=float, default=0.0)
    parser.add_argument("--mobility-waist-residual-plasticity-scale", type=float)
    parser.add_argument("--mobility-arm-residual-plasticity-scale", type=float)
    parser.add_argument("--mobility-teacher-lower-body-scale", type=float, default=0.25)
    parser.add_argument("--mobility-teacher-waist-scale", type=float, default=0.80)
    parser.add_argument("--mobility-teacher-arm-scale", type=float, default=1.00)
    parser.add_argument("--mobility-predictive-teacher-gate-floor", type=float, default=0.0)
    parser.add_argument("--mobility-teacher-lower-body-target-step-rad", type=float, default=0.08)
    parser.add_argument(
        "--mobility-teacher-lower-body-target-filter-fraction", type=float, default=0.35
    )
    parser.add_argument("--mobility-teacher-waist-target-step-rad", type=float, default=0.05)
    parser.add_argument("--mobility-teacher-waist-target-filter-fraction", type=float, default=0.25)
    parser.add_argument("--mobility-teacher-arm-target-step-rad", type=float, default=0.045)
    parser.add_argument("--mobility-teacher-arm-target-filter-fraction", type=float, default=0.15)
    parser.add_argument("--mobility-counter-rotation", action="store_true")
    parser.add_argument("--mobility-anticipatory-arm-reach", action="store_true")
    parser.add_argument("--mobility-predictive-teacher-warmstart", action="store_true")
    parser.add_argument("--mobility-teacher-recovery-latch", action="store_true")
    parser.add_argument("--mobility-teacher-recovery-hold-sec", type=float, default=0.24)
    parser.add_argument("--mobility-teacher-recovery-decay-sec", type=float, default=0.60)
    parser.add_argument("--mobility-lateral-velocity-guard", action="store_true")
    parser.add_argument("--mobility-substep-upper-body-guard", action="store_true")
    parser.add_argument("--mobility-substep-upper-body-guard-onset-rad-s", type=float, default=1.80)
    parser.add_argument(
        "--mobility-substep-upper-body-guard-ceiling-rad-s", type=float, default=2.80
    )
    parser.add_argument(
        "--mobility-substep-upper-body-minimum-position-scale", type=float, default=0.05
    )
    parser.add_argument("--shot-intent-cue", action="store_true")
    parser.add_argument("--combat-gate-pretraining-batches", type=int, default=0)
    parser.add_argument("--combat-gate-pretraining-epochs", type=int, default=8)
    parser.add_argument("--motion-library", type=Path)
    parser.add_argument("--motion-dataset-root", type=Path)
    parser.add_argument(
        "--no-teacher-pretraining",
        action="store_true",
        help="continue online PPO from a qualified parent without teacher re-distillation",
    )
    parser.add_argument("--evaluate-checkpoint", type=Path)
    parser.add_argument("--evaluation-output", type=Path)
    parser.add_argument("--evaluation-environments", type=int, default=64)
    args = parser.parse_args()
    if args.evaluate_checkpoint is not None:
        if args.evaluation_output is None:
            parser.error("--evaluation-output is required with --evaluate-checkpoint")
        evaluate_goalkeeper_physics_candidate(
            asset_root=args.asset_root,
            locomotion_policy_path=args.locomotion_policy,
            checkpoint_path=args.evaluate_checkpoint,
            output_path=args.evaluation_output,
            environment_count=args.evaluation_environments,
            difficulty_profile=args.shot_difficulty_profile,
        )
        return 0
    config = GoalkeeperPhysicsPPOConfig(
        environments_per_rank=args.environments_per_rank,
        iterations=args.iterations,
        deterministic_selection_seed_count=(args.deterministic_selection_seed_count),
        deterministic_selection_seed_stride=(args.deterministic_selection_seed_stride),
        rollout_steps=args.rollout_steps,
        training_episode_duration_sec=args.training_episode_duration_sec,
        update_epochs=args.update_epochs,
        minibatches=args.minibatches,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        policy_anchor_coefficient=args.policy_anchor_coefficient,
        random_seed=args.random_seed,
        first_save_selection_weight=args.first_save_selection_weight,
        qualified_first_save_selection_weight=(args.qualified_first_save_selection_weight),
        first_hand_save_selection_weight=args.first_hand_save_selection_weight,
        second_attempt_save_selection_weight=args.second_attempt_save_selection_weight,
        second_attempt_hand_save_selection_weight=(args.second_attempt_hand_save_selection_weight),
        second_save_selection_weight=args.second_save_selection_weight,
        second_hand_save_selection_weight=args.second_hand_save_selection_weight,
        hand_reach_selection_weight=args.hand_reach_selection_weight,
        hand_target_distance_selection_weight=(args.hand_target_distance_selection_weight),
        first_save_stratum_balance_selection_weight=(
            args.first_save_stratum_balance_selection_weight
        ),
        first_save_side_balance_selection_weight=(args.first_save_side_balance_selection_weight),
        minimum_first_save_rate_for_selection=(args.minimum_first_save_rate_for_selection),
        minimum_first_save_stratum_rate_for_selection=(
            args.minimum_first_save_stratum_rate_for_selection
        ),
        minimum_hand_displacement_for_selection_m=(args.minimum_hand_displacement_for_selection_m),
        root_angular_speed_selection_penalty_weight=(
            args.root_angular_speed_selection_penalty_weight
        ),
        second_release_recenter_selection_weight=(args.second_release_recenter_selection_weight),
        maximum_exploration_root_angular_speed_for_selection=(
            args.maximum_exploration_root_angular_speed_for_selection
        ),
        training_second_shot_probability=args.training_second_shot_probability,
        training_first_shot_release_sec=args.training_first_shot_release_sec,
        training_hard_shot_fraction=args.training_hard_shot_fraction,
        training_hard_shot_height_mode=args.training_hard_shot_height_mode,
        training_hard_shot_side_mode=args.training_hard_shot_side_mode,
        training_reach_reward_semantics=args.training_reach_reward_semantics,
        training_hard_shot_flight_time_range_sec=(
            None
            if args.training_hard_shot_flight_time_range_sec is None
            else tuple(args.training_hard_shot_flight_time_range_sec)
        ),
        training_hard_height_reach_reward_scale=(args.training_hard_height_reach_reward_scale),
        training_hard_height_reach_threshold_m=(args.training_hard_height_reach_threshold_m),
        training_hard_height_reach_distance_decay=(args.training_hard_height_reach_distance_decay),
        training_task_motion_reward_scale=args.training_task_motion_reward_scale,
        training_recovery_progress_reward_scale=(args.training_recovery_progress_reward_scale),
        training_recovery_progress_linear_speed_decay=(
            args.training_recovery_progress_linear_speed_decay
        ),
        training_recovery_progress_angular_speed_decay=(
            args.training_recovery_progress_angular_speed_decay
        ),
        training_true_save_bonus=args.training_true_save_bonus,
        training_hand_save_bonus=args.training_hand_save_bonus,
        training_recovery_event_bonus=args.training_recovery_event_bonus,
        training_root_angular_speed_penalty_scale=(args.training_root_angular_speed_penalty_scale),
        training_root_angular_speed_soft_limit_rad_s=(
            args.training_root_angular_speed_soft_limit_rad_s
        ),
        training_root_angular_speed_excess_penalty_scale=(
            args.training_root_angular_speed_excess_penalty_scale
        ),
        training_flight_root_angular_penalty_scale=(
            args.training_flight_root_angular_penalty_scale
        ),
        training_unsafe_penalty=args.training_unsafe_penalty,
        training_save_then_unsafe_penalty=args.training_save_then_unsafe_penalty,
        teacher_pretraining_samples=args.teacher_pretraining_samples,
        teacher_pretraining_epochs=args.teacher_pretraining_epochs,
        teacher_parent_replay_coefficient=args.teacher_parent_replay_coefficient,
        task_space_reach_blend=args.task_space_reach_blend,
        task_space_reach_atlas_enabled=args.task_space_reach_atlas,
        runtime_task_space_reach_enabled=args.runtime_task_space_reach,
        runtime_task_space_reach_blend=args.runtime_task_space_reach_blend,
        second_shot_reach_multiplier=args.second_shot_reach_multiplier,
        online_parent_distillation_coefficient=args.online_parent_distillation_coefficient,
        successful_trajectory_replay_coefficient=(args.successful_trajectory_replay_coefficient),
        successful_trajectory_mirror_augmentation_enabled=(
            args.successful_trajectory_mirror_augmentation
        ),
        successful_trajectory_recovery_only_enabled=(args.successful_trajectory_recovery_only),
        symmetry_mirror_loss_coefficient=args.symmetry_mirror_loss_coefficient,
        recovery_transition_policy_weight=args.recovery_transition_policy_weight,
        support_landing_causal_recovery_only_enabled=(args.support_landing_causal_recovery_only),
        rollback_to_exploration_champion_on_regression_enabled=(
            args.rollback_to_exploration_champion_on_regression
        ),
        save_first_exploration_selection_enabled=(args.save_first_exploration_selection),
        successful_trajectory_memory_capacity_per_stratum=(
            args.successful_trajectory_memory_capacity_per_stratum
        ),
        successful_trajectory_memory_batch_size=(args.successful_trajectory_memory_batch_size),
        successful_trajectory_memory_minimum_episodes_per_stratum=(
            args.successful_trajectory_memory_minimum_episodes_per_stratum
        ),
        successful_trajectory_memory_full_strength_episodes_per_stratum=(
            args.successful_trajectory_memory_full_strength_episodes_per_stratum
        ),
        successful_trajectory_action_innovation_scale=(
            args.successful_trajectory_action_innovation_scale
        ),
        arm_only_online_update=args.arm_only_online_update,
        lower_body_and_arms_online_update=args.lower_body_and_arms_online_update,
        lateral_drive_gate_only_online_update=(args.lateral_drive_gate_only_online_update),
        support_landing_online_update=args.support_landing_online_update,
        shot_difficulty_profile=args.shot_difficulty_profile,
        teacher_pretraining_enabled=not args.no_teacher_pretraining,
        initialization_checkpoint=(
            None
            if args.initialization_checkpoint is None
            else str(args.initialization_checkpoint.expanduser().resolve())
        ),
        combat_teacher_checkout=(
            None
            if args.combat_teacher_checkout is None
            else str(args.combat_teacher_checkout.expanduser().resolve())
        ),
        combat_teacher_checkpoint=(
            None
            if args.combat_teacher_checkpoint is None
            else str(args.combat_teacher_checkpoint.expanduser().resolve())
        ),
        targeted_dive_checkpoint=(
            None
            if args.targeted_dive_checkpoint is None
            else str(args.targeted_dive_checkpoint.expanduser().resolve())
        ),
        targeted_dive_option_duration_sec=args.targeted_dive_option_duration_sec,
        targeted_dive_phase_hold_sec=args.targeted_dive_phase_hold_sec,
        targeted_dive_actor_recovery_plasticity_sec=(
            args.targeted_dive_actor_recovery_plasticity_sec
        ),
        targeted_dive_actor_recovery_residual_authority_scale=(
            args.targeted_dive_actor_recovery_residual_authority_scale
        ),
        targeted_dive_post_save_counterstep_enabled=(args.targeted_dive_post_save_counterstep),
        targeted_dive_post_save_counterstep_duration_sec=(
            args.targeted_dive_post_save_counterstep_duration_sec
        ),
        targeted_dive_post_save_counterstep_command_limit=(
            args.targeted_dive_post_save_counterstep_command_limit
        ),
        targeted_dive_post_save_counterstep_capture_horizon_sec=(
            args.targeted_dive_post_save_counterstep_capture_horizon_sec
        ),
        targeted_dive_post_save_counterstep_recenter_weight=(
            args.targeted_dive_post_save_counterstep_recenter_weight
        ),
        targeted_dive_post_save_option_release_sec=(
            args.targeted_dive_post_save_option_release_sec
        ),
        targeted_dive_post_save_fall_recovery_enabled=(args.targeted_dive_post_save_fall_recovery),
        targeted_dive_post_save_fall_recovery_duration_sec=(
            args.targeted_dive_post_save_fall_recovery_duration_sec
        ),
        targeted_dive_post_save_fall_minimum_pelvis_height_m=(
            args.targeted_dive_post_save_fall_minimum_pelvis_height_m
        ),
        targeted_dive_post_save_fall_minimum_upright_projection=(
            args.targeted_dive_post_save_fall_minimum_upright_projection
        ),
        targeted_dive_post_save_fall_maximum_root_linear_speed_mps=(
            args.targeted_dive_post_save_fall_maximum_root_linear_speed_mps
        ),
        targeted_dive_post_save_fall_maximum_root_angular_speed_rad_s=(
            args.targeted_dive_post_save_fall_maximum_root_angular_speed_rad_s
        ),
        targeted_dive_prediction_lead_sec=args.targeted_dive_prediction_lead_sec,
        targeted_dive_nominal_shot_flight_time_sec=(
            args.targeted_dive_nominal_shot_flight_time_sec
        ),
        targeted_dive_intercept_phase_at_arrival=(args.targeted_dive_intercept_phase_at_arrival),
        targeted_dive_phase_sync_minimum_target_height_m=(
            args.targeted_dive_phase_sync_minimum_target_height_m
        ),
        targeted_dive_posture_exception_duration_sec=(
            args.targeted_dive_posture_exception_duration_sec
        ),
        targeted_dive_root_angular_speed_guard_ceiling_rad_s=(
            args.targeted_dive_root_angular_speed_guard_ceiling_rad_s
        ),
        targeted_dive_decoder_residual_authority=(args.targeted_dive_decoder_residual_authority),
        targeted_dive_decoder_lower_body_residual_authority=(
            args.targeted_dive_decoder_lower_body_residual_authority
        ),
        targeted_dive_decoder_lower_body_command_scale=(
            args.targeted_dive_decoder_lower_body_command_scale
        ),
        targeted_dive_decoder_waist_residual_authority=(
            args.targeted_dive_decoder_waist_residual_authority
        ),
        targeted_dive_decoder_arm_residual_authority=(
            args.targeted_dive_decoder_arm_residual_authority
        ),
        targeted_dive_actor_residual_scale=args.targeted_dive_actor_residual_scale,
        targeted_dive_anchor_lower_body_scale=args.targeted_dive_anchor_lower_body_scale,
        targeted_dive_anchor_waist_scale=args.targeted_dive_anchor_waist_scale,
        targeted_dive_anchor_arm_scale=args.targeted_dive_anchor_arm_scale,
        targeted_dive_minimum_option_gate=args.targeted_dive_minimum_option_gate,
        targeted_dive_runtime_reach_blend=args.targeted_dive_runtime_reach_blend,
        targeted_dive_runtime_reach_feedback_blend=(
            args.targeted_dive_runtime_reach_feedback_blend
        ),
        targeted_dive_runtime_reach_feedback_gain=(args.targeted_dive_runtime_reach_feedback_gain),
        targeted_dive_runtime_reach_feedback_maximum_error_m=(
            args.targeted_dive_runtime_reach_feedback_maximum_error_m
        ),
        targeted_dive_runtime_reach_feedback_support_scale=(
            args.targeted_dive_runtime_reach_feedback_support_scale
        ),
        targeted_dive_runtime_contact_support_side_enabled=(
            args.targeted_dive_runtime_contact_support_side
        ),
        targeted_dive_actor_contact_support_side_enabled=(
            args.targeted_dive_actor_contact_support_side
        ),
        targeted_dive_actor_recovery_context_enabled=(args.targeted_dive_actor_recovery_context),
        targeted_dive_runtime_whole_body_reach_blend=(
            args.targeted_dive_runtime_whole_body_reach_blend
        ),
        targeted_dive_runtime_whole_body_reach_full_below_height_m=(
            args.targeted_dive_runtime_whole_body_reach_full_below_height_m
        ),
        targeted_dive_runtime_whole_body_reach_maximum_height_m=(
            args.targeted_dive_runtime_whole_body_reach_maximum_height_m
        ),
        targeted_dive_runtime_whole_body_reach_waist_scale=(
            args.targeted_dive_runtime_whole_body_reach_waist_scale
        ),
        targeted_dive_runtime_whole_body_reach_arm_scale=(
            args.targeted_dive_runtime_whole_body_reach_arm_scale
        ),
        targeted_dive_runtime_whole_body_reach_support_scale=(
            args.targeted_dive_runtime_whole_body_reach_support_scale
        ),
        targeted_dive_runtime_whole_body_reach_release_sec=(
            args.targeted_dive_runtime_whole_body_reach_release_sec
        ),
        targeted_dive_runtime_reach_contact_standoff_m=(
            args.targeted_dive_runtime_reach_contact_standoff_m
        ),
        targeted_dive_runtime_reach_lateral_lead_m=(
            args.targeted_dive_runtime_reach_lateral_lead_m
        ),
        targeted_dive_runtime_reach_vertical_lead_m=(
            args.targeted_dive_runtime_reach_vertical_lead_m
        ),
        targeted_dive_runtime_reach_low_vertical_lead_m=(
            args.targeted_dive_runtime_reach_low_vertical_lead_m
        ),
        targeted_dive_runtime_reach_mid_vertical_lead_m=(
            args.targeted_dive_runtime_reach_mid_vertical_lead_m
        ),
        targeted_dive_runtime_reach_high_vertical_lead_m=(
            args.targeted_dive_runtime_reach_high_vertical_lead_m
        ),
        targeted_dive_overhead_reach_prior=(
            None
            if args.targeted_dive_overhead_reach_prior is None
            else str(args.targeted_dive_overhead_reach_prior.expanduser().resolve())
        ),
        targeted_dive_overhead_reach_blend=args.targeted_dive_overhead_reach_blend,
        targeted_dive_overhead_reach_minimum_target_height_m=(
            args.targeted_dive_overhead_reach_minimum_target_height_m
        ),
        targeted_dive_overhead_reach_full_target_height_m=(
            args.targeted_dive_overhead_reach_full_target_height_m
        ),
        targeted_dive_overhead_reach_lower_body_scale=(
            args.targeted_dive_overhead_reach_lower_body_scale
        ),
        targeted_dive_overhead_reach_waist_scale=(args.targeted_dive_overhead_reach_waist_scale),
        targeted_dive_overhead_reach_arm_scale=(args.targeted_dive_overhead_reach_arm_scale),
        targeted_dive_mosaic_gmt_model=(
            None
            if args.targeted_dive_mosaic_gmt_model is None
            else str(args.targeted_dive_mosaic_gmt_model.expanduser().resolve())
        ),
        targeted_dive_mosaic_gmt_skill=(
            None
            if args.targeted_dive_mosaic_gmt_skill is None
            else str(args.targeted_dive_mosaic_gmt_skill.expanduser().resolve())
        ),
        targeted_dive_mosaic_gmt_blend=args.targeted_dive_mosaic_gmt_blend,
        targeted_dive_mosaic_gmt_stability_floor=(args.targeted_dive_mosaic_gmt_stability_floor),
        targeted_dive_mosaic_gmt_minimum_target_height_m=(
            args.targeted_dive_mosaic_gmt_minimum_target_height_m
        ),
        targeted_dive_mosaic_gmt_full_target_height_m=(
            args.targeted_dive_mosaic_gmt_full_target_height_m
        ),
        targeted_dive_mosaic_gmt_lower_body_scale=(args.targeted_dive_mosaic_gmt_lower_body_scale),
        targeted_dive_mosaic_gmt_waist_scale=(args.targeted_dive_mosaic_gmt_waist_scale),
        targeted_dive_mosaic_gmt_arm_scale=args.targeted_dive_mosaic_gmt_arm_scale,
        targeted_dive_mosaic_gmt_getup_skill=(
            None
            if args.targeted_dive_mosaic_gmt_getup_skill is None
            else str(args.targeted_dive_mosaic_gmt_getup_skill.expanduser().resolve())
        ),
        targeted_dive_mosaic_gmt_getup_blend=(args.targeted_dive_mosaic_gmt_getup_blend),
        targeted_dive_mosaic_gmt_getup_activation_maximum_pelvis_height_m=(
            args.targeted_dive_mosaic_gmt_getup_activation_maximum_pelvis_height_m
        ),
        targeted_dive_mosaic_gmt_getup_blend_in_sec=(
            args.targeted_dive_mosaic_gmt_getup_blend_in_sec
        ),
        targeted_dive_mosaic_gmt_getup_reference_feedforward_blend=(
            args.targeted_dive_mosaic_gmt_getup_reference_feedforward_blend
        ),
        targeted_dive_mosaic_gmt_getup_lower_body_scale=(
            args.targeted_dive_mosaic_gmt_getup_lower_body_scale
        ),
        targeted_dive_mosaic_gmt_getup_waist_scale=(
            args.targeted_dive_mosaic_gmt_getup_waist_scale
        ),
        targeted_dive_mosaic_gmt_getup_arm_scale=(args.targeted_dive_mosaic_gmt_getup_arm_scale),
        targeted_dive_maximum_arm_target_step_rad=(args.targeted_dive_maximum_arm_target_step_rad),
        targeted_dive_arm_target_filter_fraction=(args.targeted_dive_arm_target_filter_fraction),
        targeted_dive_maximum_lower_body_target_step_rad=(
            args.targeted_dive_maximum_lower_body_target_step_rad
        ),
        targeted_dive_lower_body_target_filter_fraction=(
            args.targeted_dive_lower_body_target_filter_fraction
        ),
        targeted_dive_lateral_drive_scale=args.targeted_dive_lateral_drive_scale,
        targeted_dive_negative_target_lateral_drive_scale=(
            args.targeted_dive_negative_target_lateral_drive_scale
        ),
        targeted_dive_lateral_drive_full_activation_gate=(
            args.targeted_dive_lateral_drive_full_activation_gate
        ),
        targeted_dive_lateral_drive_capture_enabled=(args.targeted_dive_lateral_drive_capture),
        targeted_dive_lateral_drive_capture_horizon_sec=(
            args.targeted_dive_lateral_drive_capture_horizon_sec
        ),
        targeted_dive_lateral_drive_target_standoff_m=(
            args.targeted_dive_lateral_drive_target_standoff_m
        ),
        targeted_dive_lateral_drive_capture_scale_m=(
            args.targeted_dive_lateral_drive_capture_scale_m
        ),
        targeted_dive_lateral_drive_learned_gate_enabled=(
            args.targeted_dive_lateral_drive_learned_gate
        ),
        targeted_dive_canonical_locomotion_mirror_enabled=(
            args.targeted_dive_canonical_locomotion_mirror
        ),
        targeted_dive_runtime_lateral_lunge_blend=(args.targeted_dive_runtime_lateral_lunge_blend),
        targeted_dive_runtime_lateral_lunge_hip_roll_rad=(
            args.targeted_dive_runtime_lateral_lunge_hip_roll_rad
        ),
        targeted_dive_runtime_lateral_lunge_ankle_roll_rad=(
            args.targeted_dive_runtime_lateral_lunge_ankle_roll_rad
        ),
        targeted_dive_runtime_lateral_lunge_approach_horizon_sec=(
            args.targeted_dive_runtime_lateral_lunge_approach_horizon_sec
        ),
        targeted_dive_substep_upper_body_guard_enabled=(
            args.targeted_dive_substep_upper_body_guard
        ),
        targeted_dive_substep_upper_body_guard_onset_rad_s=(
            args.targeted_dive_substep_upper_body_guard_onset_rad_s
        ),
        targeted_dive_substep_upper_body_guard_ceiling_rad_s=(
            args.targeted_dive_substep_upper_body_guard_ceiling_rad_s
        ),
        targeted_dive_substep_upper_body_minimum_position_scale=(
            args.targeted_dive_substep_upper_body_minimum_position_scale
        ),
        targeted_dive_substep_option_lower_body_guard_enabled=(
            args.targeted_dive_substep_option_lower_body_guard
        ),
        targeted_dive_substep_option_lower_body_guard_onset_rad_s=(
            args.targeted_dive_substep_option_lower_body_guard_onset_rad_s
        ),
        targeted_dive_substep_option_lower_body_guard_ceiling_rad_s=(
            args.targeted_dive_substep_option_lower_body_guard_ceiling_rad_s
        ),
        targeted_dive_substep_option_lower_body_minimum_scale=(
            args.targeted_dive_substep_option_lower_body_minimum_scale
        ),
        targeted_dive_low_shot_phase_scale=args.targeted_dive_low_shot_phase_scale,
        targeted_dive_mid_shot_phase_scale=args.targeted_dive_mid_shot_phase_scale,
        targeted_dive_high_shot_phase_scale=args.targeted_dive_high_shot_phase_scale,
        targeted_dive_initial_gate=args.targeted_dive_initial_gate,
        maximum_combat_teacher_blend=args.maximum_combat_teacher_blend,
        combat_teacher_intercept_conditioning_enabled=(args.combat_teacher_intercept_conditioning),
        mobility_option_enabled=args.mobility_option,
        mobility_lateral_command_limit=args.mobility_lateral_command_limit,
        mobility_recovery_command_limit=args.mobility_recovery_command_limit,
        mobility_residual_plasticity_scale=args.mobility_residual_plasticity_scale,
        mobility_waist_residual_plasticity_scale=(args.mobility_waist_residual_plasticity_scale),
        mobility_arm_residual_plasticity_scale=(args.mobility_arm_residual_plasticity_scale),
        mobility_teacher_lower_body_scale=args.mobility_teacher_lower_body_scale,
        mobility_teacher_waist_scale=args.mobility_teacher_waist_scale,
        mobility_teacher_arm_scale=args.mobility_teacher_arm_scale,
        mobility_predictive_teacher_gate_floor=(args.mobility_predictive_teacher_gate_floor),
        mobility_teacher_lower_body_target_step_rad=(
            args.mobility_teacher_lower_body_target_step_rad
        ),
        mobility_teacher_lower_body_target_filter_fraction=(
            args.mobility_teacher_lower_body_target_filter_fraction
        ),
        mobility_teacher_waist_target_step_rad=args.mobility_teacher_waist_target_step_rad,
        mobility_teacher_waist_target_filter_fraction=(
            args.mobility_teacher_waist_target_filter_fraction
        ),
        mobility_teacher_arm_target_step_rad=args.mobility_teacher_arm_target_step_rad,
        mobility_teacher_arm_target_filter_fraction=(
            args.mobility_teacher_arm_target_filter_fraction
        ),
        mobility_counter_rotation_enabled=args.mobility_counter_rotation,
        mobility_anticipatory_arm_reach_enabled=(args.mobility_anticipatory_arm_reach),
        mobility_predictive_teacher_warmstart_enabled=(args.mobility_predictive_teacher_warmstart),
        mobility_teacher_recovery_latch_enabled=(args.mobility_teacher_recovery_latch),
        mobility_teacher_recovery_hold_sec=args.mobility_teacher_recovery_hold_sec,
        mobility_teacher_recovery_decay_sec=args.mobility_teacher_recovery_decay_sec,
        mobility_lateral_velocity_guard_enabled=(args.mobility_lateral_velocity_guard),
        mobility_substep_upper_body_guard_enabled=(args.mobility_substep_upper_body_guard),
        mobility_substep_upper_body_guard_onset_rad_s=(
            args.mobility_substep_upper_body_guard_onset_rad_s
        ),
        mobility_substep_upper_body_guard_ceiling_rad_s=(
            args.mobility_substep_upper_body_guard_ceiling_rad_s
        ),
        mobility_substep_upper_body_minimum_position_scale=(
            args.mobility_substep_upper_body_minimum_position_scale
        ),
        shot_intent_cue_enabled=args.shot_intent_cue,
        combat_gate_pretraining_batches=args.combat_gate_pretraining_batches,
        combat_gate_pretraining_epochs=args.combat_gate_pretraining_epochs,
    )
    run_goalkeeper_physics_ppo(
        asset_root=args.asset_root,
        locomotion_policy_path=args.locomotion_policy,
        output_dir=args.output_dir,
        config=config,
        motion_library_path=args.motion_library,
        motion_dataset_root=args.motion_dataset_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GoalkeeperPhysicsPPOConfig",
    "evaluate_goalkeeper_physics_candidate",
    "run_goalkeeper_physics_ppo",
]
