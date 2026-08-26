"""MJWarp residual-RL adapter for the target-conditioned G1 dive option.

The imitation decoder supplies a bounded 29-DoF motion manifold.  A learned
actor may only add small joint-position residuals around that manifold; an
environment-owned state machine decides activation, phase and posture
exceptions.  This is candidate generation only and remains ``SIM_ONLY``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.goalkeeper_mjwarp import (
    GoalkeeperMJWarpBatch,
    GoalkeeperMJWarpConfig,
)
from rosclaw_soccer.training.goalkeeper_targeted_dive import (
    _MIRROR_ORDER,
    _MIRROR_SIGN,
    decode_goalkeeper_targeted_dive,
    load_goalkeeper_targeted_dive,
    targeted_dive_features_torch,
)

TARGETED_DIVE_RL_RESIDUAL_LIMITS_RAD = (
    0.10,
    0.08,
    0.10,
    0.12,
    0.08,
    0.06,
    0.10,
    0.08,
    0.10,
    0.12,
    0.08,
    0.06,
    0.08,
    0.08,
    0.08,
    0.22,
    0.22,
    0.18,
    0.18,
    0.10,
    0.08,
    0.08,
    0.22,
    0.22,
    0.18,
    0.18,
    0.10,
    0.08,
    0.08,
)


def _body_frame_waist_counter_rotation(
    *,
    torch: Any,
    root_angular_velocity_body_rad_s: Any,
) -> Any:
    """Map root angular velocity to opposing yaw/roll/pitch waist actions.

    MuJoCo free-joint rotational velocity is already expressed in the body
    frame.  Reorder body ``(roll, pitch, yaw)`` into joint
    ``(yaw, roll, pitch)`` order and negate it for damping.  Applying another
    inverse heading rotation here used to reverse roll/pitch feedback at the
    goalkeeper's pi heading.
    """

    return torch.stack(
        (
            -root_angular_velocity_body_rad_s[:, 2],
            -root_angular_velocity_body_rad_s[:, 0],
            -root_angular_velocity_body_rad_s[:, 1],
        ),
        dim=1,
    )


def _capture_point_lateral_drive(
    *,
    torch: Any,
    target_lateral_m: Any,
    root_lateral_m: Any,
    root_lateral_velocity_mps: Any,
    target_standoff_m: float,
    capture_horizon_sec: float,
    capture_scale_m: float,
) -> Any:
    """Return a bounded local command that accelerates and then brakes.

    The goalkeeper faces approximately pi yaw, so positive local lateral
    command moves toward negative world y.  ``capture - goal`` therefore has
    the required local sign.  Unlike a sign-only command, velocity advances
    the capture point beyond the goal before physical overshoot, producing a
    causal counterstep even when no save event has fired yet.
    """

    lateral_error = target_lateral_m - root_lateral_m
    direction = torch.where(
        lateral_error < 0.0,
        -torch.ones_like(lateral_error),
        torch.ones_like(lateral_error),
    )
    pelvis_goal = target_lateral_m - direction * target_standoff_m
    capture = root_lateral_m + capture_horizon_sec * root_lateral_velocity_mps
    return torch.clamp((capture - pelvis_goal) / capture_scale_m, -1.0, 1.0)


@dataclass(frozen=True)
class GoalkeeperTargetedDiveRLConfig:
    """Safety and authority contract for residual physics learning."""

    option_duration_sec: float = 0.90
    phase_hold_sec: float = 0.0
    actor_recovery_plasticity_sec: float = 0.0
    actor_recovery_residual_authority_scale: float = 0.50
    post_save_counterstep_enabled: bool = False
    post_save_counterstep_duration_sec: float = 0.80
    post_save_counterstep_command_limit: float = 0.55
    post_save_counterstep_capture_horizon_sec: float = 0.28
    post_save_counterstep_recenter_weight: float = 1.0
    post_save_option_release_sec: float = 0.30
    post_save_fall_recovery_enabled: bool = False
    post_save_fall_recovery_duration_sec: float = 1.50
    post_save_fall_minimum_pelvis_height_m: float = 0.12
    post_save_fall_minimum_upright_projection: float = -0.95
    post_save_fall_maximum_root_linear_speed_mps: float = 3.50
    post_save_fall_maximum_root_angular_speed_rad_s: float = 10.0
    prediction_lead_sec: float = 0.30
    nominal_shot_flight_time_sec: float = 0.47
    intercept_phase_at_arrival: float | None = None
    phase_sync_minimum_target_height_m: float = 0.60
    posture_exception_duration_sec: float = 1.55
    decoder_residual_authority: float = 0.10
    decoder_lower_body_residual_authority: float | None = None
    decoder_lower_body_command_scale: float | None = None
    decoder_waist_residual_authority: float | None = None
    decoder_arm_residual_authority: float | None = None
    actor_residual_scale: float = 0.70
    anchor_lower_body_scale: float = 0.25
    anchor_waist_scale: float = 0.50
    anchor_arm_scale: float = 1.00
    minimum_option_gate: float = 0.0
    runtime_reach_blend: float = 0.0
    runtime_reach_feedback_blend: float = 0.0
    runtime_reach_feedback_gain: float = 0.70
    runtime_reach_feedback_maximum_error_m: float = 0.30
    runtime_reach_feedback_support_scale: float = 0.0
    runtime_contact_support_side_enabled: bool = False
    actor_contact_support_side_enabled: bool = False
    actor_recovery_context_enabled: bool = False
    runtime_whole_body_reach_blend: float = 0.0
    runtime_whole_body_reach_full_below_height_m: float = 0.50
    runtime_whole_body_reach_maximum_height_m: float = 0.65
    runtime_whole_body_reach_waist_scale: float = 0.75
    runtime_whole_body_reach_arm_scale: float = 1.0
    runtime_whole_body_reach_support_scale: float = 0.65
    runtime_whole_body_reach_release_sec: float = 0.60
    runtime_reach_contact_standoff_m: float = 0.0
    runtime_reach_lateral_lead_m: float = 0.0
    runtime_reach_vertical_lead_m: float = 0.0
    runtime_reach_low_vertical_lead_m: float | None = None
    runtime_reach_mid_vertical_lead_m: float | None = None
    runtime_reach_high_vertical_lead_m: float | None = None
    overhead_reach_prior_path: str | None = None
    overhead_reach_blend: float = 0.0
    overhead_reach_minimum_target_height_m: float = 1.10
    overhead_reach_full_target_height_m: float = 1.25
    overhead_reach_lower_body_scale: float = 0.0
    overhead_reach_waist_scale: float = 0.25
    overhead_reach_arm_scale: float = 1.0
    mosaic_gmt_model_path: str | None = None
    mosaic_gmt_skill_path: str | None = None
    mosaic_gmt_blend: float = 0.0
    mosaic_gmt_stability_floor: float = 0.0
    mosaic_gmt_minimum_target_height_m: float = 1.10
    mosaic_gmt_full_target_height_m: float = 1.25
    mosaic_gmt_lower_body_scale: float = 1.0
    mosaic_gmt_waist_scale: float = 1.0
    mosaic_gmt_arm_scale: float = 0.80
    mosaic_gmt_getup_skill_path: str | None = None
    mosaic_gmt_getup_blend: float = 0.0
    mosaic_gmt_getup_activation_maximum_pelvis_height_m: float = 0.50
    mosaic_gmt_getup_blend_in_sec: float = 0.35
    mosaic_gmt_getup_reference_feedforward_blend: float = 0.0
    mosaic_gmt_getup_lower_body_scale: float = 1.0
    mosaic_gmt_getup_waist_scale: float = 1.0
    mosaic_gmt_getup_arm_scale: float = 1.0
    runtime_reach_approach_horizon_sec: float = 0.55
    runtime_reach_full_lead_sec: float = 0.15
    runtime_reach_hold_after_arrival_sec: float = 0.18
    runtime_reach_full_activation_gate: float = 0.30
    maximum_arm_target_step_rad: float = 0.10
    arm_target_filter_fraction: float = 0.50
    maximum_lower_body_target_step_rad: float = 0.08
    lower_body_target_filter_fraction: float = 0.35
    maximum_waist_target_step_rad: float = 0.05
    waist_target_filter_fraction: float = 0.25
    lateral_drive_scale: float = 0.0
    negative_target_lateral_drive_scale: float = 1.0
    lateral_drive_full_activation_gate: float = 0.30
    lateral_drive_capture_enabled: bool = False
    lateral_drive_capture_horizon_sec: float = 0.35
    lateral_drive_target_standoff_m: float = 0.32
    lateral_drive_capture_scale_m: float = 0.45
    lateral_drive_learned_gate_enabled: bool = False
    runtime_lateral_lunge_blend: float = 0.0
    runtime_lateral_lunge_hip_roll_rad: float = 0.18
    runtime_lateral_lunge_ankle_roll_rad: float = 0.12
    runtime_lateral_lunge_approach_horizon_sec: float = 0.90
    substep_upper_body_guard_enabled: bool = False
    substep_upper_body_guard_onset_rad_s: float = 1.80
    substep_upper_body_guard_ceiling_rad_s: float = 3.00
    substep_upper_body_minimum_position_scale: float = 0.05
    substep_option_lower_body_guard_enabled: bool = False
    substep_option_lower_body_guard_onset_rad_s: float = 2.40
    substep_option_lower_body_guard_ceiling_rad_s: float = 3.30
    substep_option_lower_body_minimum_scale: float = 0.0
    canonical_locomotion_mirror_enabled: bool = False
    official_goalkeeper_teacher_checkpoint_path: str | None = None
    official_goalkeeper_teacher_blend: float = 1.0
    official_goalkeeper_lower_body_target_step_rad: float = 0.08
    official_goalkeeper_lower_body_filter_fraction: float = 0.35
    official_goalkeeper_waist_target_step_rad: float = 0.05
    official_goalkeeper_waist_filter_fraction: float = 0.25
    official_goalkeeper_arm_target_step_rad: float = 0.08
    official_goalkeeper_arm_filter_fraction: float = 0.35
    low_shot_phase_scale: float = 1.0
    mid_shot_phase_scale: float = 1.0
    high_shot_phase_scale: float = 1.0
    dive_minimum_pelvis_height_m: float = 0.30
    dive_minimum_upright_projection: float = 0.05
    dive_maximum_root_linear_speed_mps: float = 3.0
    dive_maximum_root_angular_speed_rad_s: float = 8.0
    activation_minimum_lateral_error_m: float = 0.70
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.goalkeeper_targeted_dive_rl_config.v51"

    def __post_init__(self) -> None:
        values = (
            self.option_duration_sec,
            self.prediction_lead_sec,
            self.nominal_shot_flight_time_sec,
            self.posture_exception_duration_sec,
            self.decoder_residual_authority,
            self.actor_residual_scale,
            self.anchor_lower_body_scale,
            self.anchor_waist_scale,
            self.anchor_arm_scale,
            self.dive_minimum_pelvis_height_m,
            self.dive_maximum_root_linear_speed_mps,
            self.dive_maximum_root_angular_speed_rad_s,
            self.activation_minimum_lateral_error_m,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("targeted dive RL settings must be finite and positive")
        if not 3.0 <= self.dive_maximum_root_angular_speed_rad_s <= 12.0:
            raise ValueError("targeted dive root angular-speed guard is invalid")
        if not 0.70 <= self.option_duration_sec <= 1.40:
            raise ValueError("targeted dive RL option duration is invalid")
        if not math.isfinite(self.phase_hold_sec) or not 0.0 <= self.phase_hold_sec <= 0.40:
            raise ValueError("targeted dive RL phase hold is invalid")
        if not math.isfinite(self.actor_recovery_plasticity_sec) or not (
            0.0 <= self.actor_recovery_plasticity_sec <= 5.00
        ):
            raise ValueError("targeted dive recovery-plasticity window is invalid")
        if not math.isfinite(self.actor_recovery_residual_authority_scale) or not (
            0.05 <= self.actor_recovery_residual_authority_scale <= 1.0
        ):
            raise ValueError("targeted dive recovery residual authority is invalid")
        if not isinstance(self.post_save_counterstep_enabled, bool):
            raise ValueError("targeted dive post-save counterstep flag is invalid")
        if not isinstance(self.post_save_fall_recovery_enabled, bool):
            raise ValueError("targeted dive post-save fall-recovery flag is invalid")
        if not (
            math.isfinite(self.post_save_counterstep_duration_sec)
            and 0.30 <= self.post_save_counterstep_duration_sec <= 1.50
            and math.isfinite(self.post_save_counterstep_command_limit)
            and 0.20 <= self.post_save_counterstep_command_limit <= 1.0
            and math.isfinite(self.post_save_counterstep_capture_horizon_sec)
            and 0.10 <= self.post_save_counterstep_capture_horizon_sec <= 0.60
            and math.isfinite(self.post_save_counterstep_recenter_weight)
            and 0.0 <= self.post_save_counterstep_recenter_weight <= 1.0
            and math.isfinite(self.post_save_option_release_sec)
            and 0.10 <= self.post_save_option_release_sec <= 0.60
            and self.post_save_option_release_sec <= self.post_save_counterstep_duration_sec
        ):
            raise ValueError("targeted dive post-save counterstep settings are invalid")
        fall_recovery_values = (
            self.post_save_fall_recovery_duration_sec,
            self.post_save_fall_minimum_pelvis_height_m,
            self.post_save_fall_minimum_upright_projection,
            self.post_save_fall_maximum_root_linear_speed_mps,
            self.post_save_fall_maximum_root_angular_speed_rad_s,
        )
        if any(not math.isfinite(value) for value in fall_recovery_values) or not (
            0.50 <= self.post_save_fall_recovery_duration_sec <= 5.00
            and 0.08 <= self.post_save_fall_minimum_pelvis_height_m <= 0.30
            and -1.0 <= self.post_save_fall_minimum_upright_projection <= 0.10
            and 2.0 <= self.post_save_fall_maximum_root_linear_speed_mps <= 5.0
            and 5.0 <= self.post_save_fall_maximum_root_angular_speed_rad_s <= 12.0
        ):
            raise ValueError("targeted dive post-save fall-recovery envelope is invalid")
        if self.post_save_fall_recovery_enabled and not (
            self.post_save_counterstep_enabled
            and self.actor_recovery_plasticity_sec > 0.0
            and self.actor_recovery_context_enabled
        ):
            raise ValueError(
                "targeted dive post-save fall recovery requires counterstep, "
                "recovery plasticity, and causal recovery context"
            )
        if not 0.10 <= self.prediction_lead_sec <= 1.00:
            raise ValueError("targeted dive RL prediction lead is invalid")
        if not 0.30 <= self.nominal_shot_flight_time_sec <= 0.70:
            raise ValueError("targeted dive nominal flight time is invalid")
        if self.intercept_phase_at_arrival is not None and not (
            math.isfinite(self.intercept_phase_at_arrival)
            and 0.45 <= self.intercept_phase_at_arrival <= 0.85
        ):
            raise ValueError("targeted dive intercept phase is invalid")
        if not math.isfinite(self.phase_sync_minimum_target_height_m) or not (
            0.10 <= self.phase_sync_minimum_target_height_m <= 1.20
        ):
            raise ValueError("targeted dive phase-sync height is invalid")
        if not (
            self.option_duration_sec + self.phase_hold_sec
            <= self.posture_exception_duration_sec
            <= 2.20
        ):
            raise ValueError("targeted dive RL posture exception duration is invalid")
        if not 0.0 <= self.dive_minimum_upright_projection <= 0.40:
            raise ValueError("targeted dive RL upright envelope is invalid")
        if not 0.0 < self.decoder_residual_authority <= 0.35:
            raise ValueError("targeted dive decoder authority is invalid")
        group_authorities = (
            self.resolved_decoder_lower_body_residual_authority,
            self.resolved_decoder_waist_residual_authority,
            self.resolved_decoder_arm_residual_authority,
        )
        if any(not math.isfinite(value) or not 0.0 < value <= 1.0 for value in group_authorities):
            raise ValueError("targeted dive decoder group authority is invalid")
        if not (
            self.anchor_lower_body_scale <= self.resolved_decoder_lower_body_command_scale <= 1.0
        ):
            raise ValueError("targeted dive lower-body decoder command scale is invalid")
        if not 0.0 < self.actor_residual_scale <= 1.0:
            raise ValueError("targeted dive actor authority is invalid")
        if not (
            0.05 <= self.anchor_lower_body_scale <= 0.50
            and self.anchor_lower_body_scale <= self.anchor_waist_scale <= 0.80
            and self.anchor_waist_scale <= self.anchor_arm_scale <= 1.0
        ):
            raise ValueError("targeted dive anchor group scales are invalid")
        if not math.isfinite(self.minimum_option_gate) or not (
            0.0 <= self.minimum_option_gate <= 0.80
        ):
            raise ValueError("targeted dive minimum option gate is invalid")
        reach_values = (
            self.runtime_reach_blend,
            self.runtime_reach_feedback_blend,
            self.runtime_reach_feedback_gain,
            self.runtime_reach_feedback_maximum_error_m,
            self.runtime_reach_feedback_support_scale,
            self.runtime_whole_body_reach_blend,
            self.runtime_whole_body_reach_full_below_height_m,
            self.runtime_whole_body_reach_maximum_height_m,
            self.runtime_whole_body_reach_waist_scale,
            self.runtime_whole_body_reach_arm_scale,
            self.runtime_whole_body_reach_support_scale,
            self.runtime_whole_body_reach_release_sec,
            self.runtime_reach_contact_standoff_m,
            self.runtime_reach_lateral_lead_m,
            self.runtime_reach_vertical_lead_m,
            self.runtime_reach_approach_horizon_sec,
            self.runtime_reach_full_lead_sec,
            self.runtime_reach_hold_after_arrival_sec,
            self.runtime_reach_full_activation_gate,
            self.maximum_arm_target_step_rad,
            self.arm_target_filter_fraction,
            self.maximum_waist_target_step_rad,
            self.waist_target_filter_fraction,
            self.lateral_drive_scale,
            self.negative_target_lateral_drive_scale,
            self.lateral_drive_full_activation_gate,
            self.runtime_lateral_lunge_blend,
            self.runtime_lateral_lunge_hip_roll_rad,
            self.runtime_lateral_lunge_ankle_roll_rad,
            self.runtime_lateral_lunge_approach_horizon_sec,
            self.low_shot_phase_scale,
            self.mid_shot_phase_scale,
            self.high_shot_phase_scale,
        )
        if any(not math.isfinite(value) for value in reach_values):
            raise ValueError("targeted dive runtime reach settings must be finite")
        if not 0.0 <= self.runtime_reach_blend <= 1.0:
            raise ValueError("targeted dive runtime reach blend is invalid")
        if not (
            0.0 <= self.runtime_reach_feedback_blend <= 1.0
            and 0.25 <= self.runtime_reach_feedback_gain <= 1.0
            and 0.05 <= self.runtime_reach_feedback_maximum_error_m <= 0.50
            and 0.0 <= self.runtime_reach_feedback_support_scale <= 1.0
        ):
            raise ValueError("targeted dive runtime reach feedback is invalid")
        if self.runtime_reach_feedback_blend > 0.0 and self.runtime_reach_blend <= 0.0:
            raise ValueError("targeted dive reach feedback requires runtime reach")
        if not 0.0 <= self.runtime_whole_body_reach_blend <= 1.0:
            raise ValueError("targeted dive whole-body reach blend is invalid")
        if not (
            0.30
            <= self.runtime_whole_body_reach_full_below_height_m
            < self.runtime_whole_body_reach_maximum_height_m
            <= 0.80
        ):
            raise ValueError("targeted dive whole-body reach height route is invalid")
        if not (
            0.10 <= self.runtime_whole_body_reach_waist_scale <= 1.0
            and 0.10 <= self.runtime_whole_body_reach_arm_scale <= 1.0
        ):
            raise ValueError("targeted dive whole-body reach joint scales are invalid")
        if not 0.0 <= self.runtime_whole_body_reach_support_scale <= 1.0:
            raise ValueError("targeted dive whole-body support scale is invalid")
        if not 0.30 <= self.runtime_whole_body_reach_release_sec <= 1.20:
            raise ValueError("targeted dive whole-body release duration is invalid")
        if not 0.0 <= self.runtime_reach_contact_standoff_m <= 0.22:
            raise ValueError("targeted dive runtime reach contact standoff is invalid")
        if not 0.0 <= self.runtime_reach_lateral_lead_m <= 0.30:
            raise ValueError("targeted dive runtime lateral lead is invalid")
        if not 0.0 <= self.runtime_reach_vertical_lead_m <= 0.40:
            raise ValueError("targeted dive runtime reach vertical lead is invalid")
        height_leads = (
            self.runtime_reach_low_vertical_lead_m,
            self.runtime_reach_mid_vertical_lead_m,
            self.runtime_reach_high_vertical_lead_m,
        )
        if any(value is not None for value in height_leads) and not all(
            value is not None for value in height_leads
        ):
            raise ValueError("targeted dive height-conditioned vertical leads must be complete")
        if any(
            value is not None and (not math.isfinite(value) or not -0.40 <= value <= 0.40)
            for value in height_leads
        ):
            raise ValueError("targeted dive height-conditioned vertical lead is invalid")
        if (
            self.runtime_reach_contact_standoff_m > 0.0
            or self.runtime_reach_lateral_lead_m > 0.0
            or self.runtime_reach_vertical_lead_m > 0.0
            or any(value is not None and value != 0.0 for value in height_leads)
        ) and self.runtime_reach_blend <= 0.0:
            raise ValueError("targeted dive reach compensation requires runtime reach")
        if not 0.0 <= self.lateral_drive_scale <= 1.0:
            raise ValueError("targeted dive lateral drive scale is invalid")
        if not 0.0 <= self.negative_target_lateral_drive_scale <= 1.0:
            raise ValueError("targeted dive negative-target lateral drive scale is invalid")
        if not isinstance(self.lateral_drive_capture_enabled, bool):
            raise ValueError("targeted dive capture-point drive flag must be boolean")
        if not isinstance(self.lateral_drive_learned_gate_enabled, bool):
            raise ValueError("targeted dive learned lateral-drive gate must be boolean")
        if not isinstance(self.runtime_contact_support_side_enabled, bool):
            raise ValueError("targeted dive contact-support proprioception flag must be boolean")
        if not isinstance(self.actor_contact_support_side_enabled, bool):
            raise ValueError("targeted dive actor contact-support flag must be boolean")
        if not isinstance(self.actor_recovery_context_enabled, bool):
            raise ValueError("targeted dive actor recovery-context flag must be boolean")
        if (
            self.actor_contact_support_side_enabled
            and not self.runtime_contact_support_side_enabled
        ):
            raise ValueError("targeted dive actor contact support requires runtime contact support")
        if self.actor_recovery_context_enabled and not self.actor_contact_support_side_enabled:
            raise ValueError("targeted dive actor recovery context requires foot-contact context")
        if not (
            math.isfinite(self.lateral_drive_capture_horizon_sec)
            and 0.10 <= self.lateral_drive_capture_horizon_sec <= 0.80
            and math.isfinite(self.lateral_drive_target_standoff_m)
            and 0.10 <= self.lateral_drive_target_standoff_m <= 0.50
            and math.isfinite(self.lateral_drive_capture_scale_m)
            and 0.15 <= self.lateral_drive_capture_scale_m <= 0.80
        ):
            raise ValueError("targeted dive capture-point drive settings are invalid")
        if not (
            0.0 <= self.runtime_lateral_lunge_blend <= 1.0
            and 0.05 <= self.runtime_lateral_lunge_hip_roll_rad <= 0.30
            and 0.03 <= self.runtime_lateral_lunge_ankle_roll_rad <= 0.25
            and 0.60 <= self.runtime_lateral_lunge_approach_horizon_sec <= 1.20
        ):
            raise ValueError("targeted dive lateral lunge scaffold is invalid")
        if not isinstance(self.substep_upper_body_guard_enabled, bool):
            raise ValueError("targeted dive substep upper-body guard flag must be boolean")
        if not (
            math.isfinite(self.substep_upper_body_guard_onset_rad_s)
            and math.isfinite(self.substep_upper_body_guard_ceiling_rad_s)
            and math.isfinite(self.substep_upper_body_minimum_position_scale)
            and 0.50
            <= self.substep_upper_body_guard_onset_rad_s
            < self.substep_upper_body_guard_ceiling_rad_s
            <= self.dive_maximum_root_angular_speed_rad_s
            and 0.0 <= self.substep_upper_body_minimum_position_scale <= 0.50
        ):
            raise ValueError("targeted dive substep upper-body guard is invalid")
        if not isinstance(self.substep_option_lower_body_guard_enabled, bool):
            raise ValueError("targeted dive substep option lower-body guard flag must be boolean")
        if not isinstance(self.canonical_locomotion_mirror_enabled, bool):
            raise ValueError("targeted dive canonical locomotion mirror flag must be boolean")
        if (
            self.official_goalkeeper_teacher_checkpoint_path is not None
            and self.canonical_locomotion_mirror_enabled
        ):
            raise ValueError("official goalkeeper teacher and locomotion mirror are exclusive")
        if not math.isfinite(self.official_goalkeeper_teacher_blend) or not (
            0.05 <= self.official_goalkeeper_teacher_blend <= 1.0
        ):
            raise ValueError("official goalkeeper teacher blend is invalid")
        official_filter_contracts = (
            (
                self.official_goalkeeper_lower_body_target_step_rad,
                self.official_goalkeeper_lower_body_filter_fraction,
            ),
            (
                self.official_goalkeeper_waist_target_step_rad,
                self.official_goalkeeper_waist_filter_fraction,
            ),
            (
                self.official_goalkeeper_arm_target_step_rad,
                self.official_goalkeeper_arm_filter_fraction,
            ),
        )
        if any(
            not math.isfinite(step)
            or not math.isfinite(fraction)
            or not 0.01 <= step <= 0.20
            or not 0.10 <= fraction <= 1.0
            for step, fraction in official_filter_contracts
        ):
            raise ValueError("official goalkeeper teacher target filter is invalid")
        if not (
            math.isfinite(self.substep_option_lower_body_guard_onset_rad_s)
            and math.isfinite(self.substep_option_lower_body_guard_ceiling_rad_s)
            and math.isfinite(self.substep_option_lower_body_minimum_scale)
            and 0.50
            <= self.substep_option_lower_body_guard_onset_rad_s
            < self.substep_option_lower_body_guard_ceiling_rad_s
            <= self.dive_maximum_root_angular_speed_rad_s
            and 0.0 <= self.substep_option_lower_body_minimum_scale <= 0.50
        ):
            raise ValueError("targeted dive substep option lower-body guard is invalid")
        if not all(
            0.50 <= value <= 1.50
            for value in (
                self.low_shot_phase_scale,
                self.mid_shot_phase_scale,
                self.high_shot_phase_scale,
            )
        ):
            raise ValueError("targeted dive height-conditioned phase scale is invalid")
        if not (
            0.30 <= self.runtime_reach_approach_horizon_sec <= 0.80
            and 0.05 <= self.runtime_reach_full_lead_sec < self.runtime_reach_approach_horizon_sec
            and 0.05 <= self.runtime_reach_hold_after_arrival_sec <= 0.40
            and 0.10 <= self.runtime_reach_full_activation_gate <= 0.60
            and 0.10 <= self.lateral_drive_full_activation_gate <= 1.00
        ):
            raise ValueError("targeted dive runtime reach timing is invalid")
        if not (
            0.02 <= self.maximum_arm_target_step_rad <= 0.20
            and 0.10 <= self.arm_target_filter_fraction <= 1.0
        ):
            raise ValueError("targeted dive arm target filter is invalid")
        if not (
            0.01 <= self.maximum_lower_body_target_step_rad <= 0.20
            and 0.10 <= self.lower_body_target_filter_fraction <= 1.0
        ):
            raise ValueError("targeted dive lower-body target filter is invalid")
        if not (
            0.01 <= self.maximum_waist_target_step_rad <= 0.10
            and 0.10 <= self.waist_target_filter_fraction <= 1.0
        ):
            raise ValueError("targeted dive waist target filter is invalid")
        overhead_scales = (
            self.overhead_reach_lower_body_scale,
            self.overhead_reach_waist_scale,
            self.overhead_reach_arm_scale,
        )
        if (
            not math.isfinite(self.overhead_reach_blend)
            or not 0.0 <= self.overhead_reach_blend <= 1.0
            or any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in overhead_scales)
            or not (
                0.80
                <= self.overhead_reach_minimum_target_height_m
                < self.overhead_reach_full_target_height_m
                <= 1.60
            )
        ):
            raise ValueError("targeted dive overhead reach settings are invalid")
        if self.overhead_reach_blend > 0.0:
            if self.overhead_reach_prior_path is None:
                raise ValueError("targeted dive overhead reach requires a prior")
            prior_path = Path(self.overhead_reach_prior_path)
            if not prior_path.is_absolute() or not prior_path.is_file():
                raise ValueError("targeted dive overhead reach prior path is invalid")
            if self.runtime_reach_blend > 0.0 or self.runtime_whole_body_reach_blend > 0.0:
                raise ValueError("targeted dive cannot combine two runtime reach teachers")
        elif self.overhead_reach_prior_path is not None:
            raise ValueError("targeted dive inactive overhead prior must be omitted")
        gmt_scales = (
            self.mosaic_gmt_lower_body_scale,
            self.mosaic_gmt_waist_scale,
            self.mosaic_gmt_arm_scale,
        )
        if (
            not math.isfinite(self.mosaic_gmt_blend)
            or not 0.0 <= self.mosaic_gmt_blend <= 1.0
            or not math.isfinite(self.mosaic_gmt_stability_floor)
            or not 0.0 <= self.mosaic_gmt_stability_floor <= 0.60
            or any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in gmt_scales)
            or not (
                (0.30 if self.mosaic_gmt_arm_scale == 0.0 else 0.80)
                <= self.mosaic_gmt_minimum_target_height_m
                < self.mosaic_gmt_full_target_height_m
                <= 1.60
            )
        ):
            raise ValueError("targeted dive MOSAIC GMT settings are invalid")
        if self.mosaic_gmt_stability_floor > 0.0 and self.mosaic_gmt_arm_scale > 0.0:
            raise ValueError("targeted dive GMT stability floor cannot own arm joints")
        if self.mosaic_gmt_blend > 0.0:
            paths = (self.mosaic_gmt_model_path, self.mosaic_gmt_skill_path)
            if any(value is None for value in paths):
                raise ValueError("targeted dive MOSAIC GMT requires model and skill paths")
            if any(
                not Path(str(value)).is_absolute() or not Path(str(value)).is_file()
                for value in paths
            ):
                raise ValueError("targeted dive MOSAIC GMT paths are invalid")
            if self.overhead_reach_blend > 0.0:
                raise ValueError("targeted dive permits only one whole-body reach teacher")
            if self.runtime_reach_blend > 0.0 and self.mosaic_gmt_arm_scale > 0.0:
                raise ValueError(
                    "targeted dive GMT and task-space reach require disjoint joint-group authority"
                )
        elif self.mosaic_gmt_skill_path is not None:
            raise ValueError("targeted dive inactive MOSAIC GMT overhead skill must be omitted")
        getup_scales = (
            self.mosaic_gmt_getup_lower_body_scale,
            self.mosaic_gmt_getup_waist_scale,
            self.mosaic_gmt_getup_arm_scale,
        )
        if (
            not math.isfinite(self.mosaic_gmt_getup_blend)
            or not 0.0 <= self.mosaic_gmt_getup_blend <= 1.0
            or not math.isfinite(self.mosaic_gmt_getup_activation_maximum_pelvis_height_m)
            or not 0.30 <= self.mosaic_gmt_getup_activation_maximum_pelvis_height_m <= 0.60
            or not math.isfinite(self.mosaic_gmt_getup_blend_in_sec)
            or not 0.10 <= self.mosaic_gmt_getup_blend_in_sec <= 0.80
            or not math.isfinite(self.mosaic_gmt_getup_reference_feedforward_blend)
            or not 0.0 <= self.mosaic_gmt_getup_reference_feedforward_blend <= 1.0
            or any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in getup_scales)
        ):
            raise ValueError("targeted dive MOSAIC GMT get-up settings are invalid")
        if self.mosaic_gmt_getup_blend > 0.0:
            if (
                self.mosaic_gmt_model_path is None
                or self.mosaic_gmt_getup_skill_path is None
                or not self.post_save_fall_recovery_enabled
            ):
                raise ValueError(
                    "targeted dive MOSAIC GMT get-up requires the model, skill, "
                    "and post-save fall recovery"
                )
            getup_path = Path(self.mosaic_gmt_getup_skill_path)
            model_path = Path(self.mosaic_gmt_model_path)
            if any(
                not path.is_absolute() or not path.is_file() for path in (model_path, getup_path)
            ):
                raise ValueError("targeted dive MOSAIC GMT get-up paths are invalid")
        elif self.mosaic_gmt_getup_skill_path is not None:
            raise ValueError("targeted dive inactive MOSAIC GMT get-up skill must be omitted")
        if self.mosaic_gmt_model_path is not None and not (
            self.mosaic_gmt_blend > 0.0 or self.mosaic_gmt_getup_blend > 0.0
        ):
            raise ValueError("targeted dive inactive MOSAIC GMT model must be omitted")
        if self.runtime_whole_body_reach_blend > 0.0:
            if self.runtime_reach_blend > 0.0:
                raise ValueError("targeted dive permits only one runtime reach teacher")
            if self.mosaic_gmt_blend > 0.0 and (
                self.mosaic_gmt_stability_floor > 0.0
                or self.mosaic_gmt_arm_scale > 0.0
                or self.mosaic_gmt_minimum_target_height_m
                < self.runtime_whole_body_reach_maximum_height_m
            ):
                raise ValueError(
                    "targeted dive low whole-body reach and GMT require disjoint height routing"
                )
        if self.activation_ceiling != "SIM_ONLY" or self.hardware_authorized:
            raise ValueError("targeted dive RL must remain SIM_ONLY")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))

    @property
    def actor_plasticity_duration_sec(self) -> float:
        """Bound actor authority to the declared dive and finite recovery window."""

        return self.option_duration_sec + self.phase_hold_sec + self.actor_recovery_plasticity_sec

    @property
    def resolved_decoder_lower_body_residual_authority(self) -> float:
        return (
            self.decoder_residual_authority
            if self.decoder_lower_body_residual_authority is None
            else self.decoder_lower_body_residual_authority
        )

    @property
    def resolved_decoder_lower_body_command_scale(self) -> float:
        return (
            self.anchor_lower_body_scale
            if self.decoder_lower_body_command_scale is None
            else self.decoder_lower_body_command_scale
        )

    @property
    def resolved_decoder_waist_residual_authority(self) -> float:
        return (
            self.decoder_residual_authority
            if self.decoder_waist_residual_authority is None
            else self.decoder_waist_residual_authority
        )

    @property
    def resolved_decoder_arm_residual_authority(self) -> float:
        return (
            self.decoder_residual_authority
            if self.decoder_arm_residual_authority is None
            else self.decoder_arm_residual_authority
        )


class GoalkeeperTargetedDiveMJWarpBatch(GoalkeeperMJWarpBatch):
    """Fast target-conditioned dive plus bounded full-body residual learning."""

    action_size = 30  # bounded dive gate + one residual per G1 joint
    lower_body_action_start_index = 1
    lower_body_action_end_index = 13
    arm_action_start_index = 16  # gate + 12 lower-body + 3 waist channels
    residual_arm_start_index = 15
    lower_body_authority = "IMITATION_ANCHOR_PLUS_BOUNDED_RL_RESIDUAL"
    learned_residual_authority = "BOUNDED_29_DOF_POSITION_RESIDUAL_SIM_ONLY"

    def __init__(
        self,
        *,
        asset_root: Path,
        locomotion_policy_path: Path,
        targeted_dive_checkpoint: Path,
        device: Any,
        config: GoalkeeperMJWarpConfig,
        dive_config: GoalkeeperTargetedDiveRLConfig | None = None,
    ) -> None:
        super().__init__(
            asset_root=asset_root,
            locomotion_policy_path=locomotion_policy_path,
            device=device,
            config=config,
        )
        torch = self.torch
        self.dive_config = dive_config or GoalkeeperTargetedDiveRLConfig()
        if self.dive_config.prediction_lead_sec >= self.config.first_shot_release_sec:
            raise ValueError("targeted dive RL prediction cue precedes the episode")
        self.targeted_dive_checkpoint_path = targeted_dive_checkpoint.expanduser().resolve()
        self.targeted_dive, self.targeted_dive_checkpoint = load_goalkeeper_targeted_dive(
            checkpoint_path=self.targeted_dive_checkpoint_path,
            device=self.device,
        )
        self.observation_size += self.action_size - 18
        self.observation_size += 3 * int(self.dive_config.actor_contact_support_side_enabled)
        self.observation_size += 4 * int(self.dive_config.actor_recovery_context_enabled)
        self._residual_indices = torch.arange(29, dtype=torch.long, device=self.device)
        self._residual_limits = torch.tensor(
            TARGETED_DIVE_RL_RESIDUAL_LIMITS_RAD,
            dtype=torch.float32,
            device=self.device,
        )
        self._residual_limits *= self.dive_config.actor_residual_scale / self.config.residual_scale
        self._residual_joint_lower = self._joint_ranges[:, 0].clone()
        self._residual_joint_upper = self._joint_ranges[:, 1].clone()
        self._anchor_group_scale = torch.tensor(
            (self.dive_config.anchor_lower_body_scale,) * 12
            + (self.dive_config.anchor_waist_scale,) * 3
            + (self.dive_config.anchor_arm_scale,) * 14,
            dtype=torch.float32,
            device=self.device,
        )
        self._decoder_group_authority = torch.tensor(
            (self.dive_config.resolved_decoder_lower_body_residual_authority,) * 12
            + (self.dive_config.resolved_decoder_waist_residual_authority,) * 3
            + (self.dive_config.resolved_decoder_arm_residual_authority,) * 14,
            dtype=torch.float32,
            device=self.device,
        )
        self._lower_body_decoder_command_boost = (
            self.dive_config.resolved_decoder_lower_body_command_scale
            - self.dive_config.anchor_lower_body_scale
        )
        self._maximum_decoder_lower_body_residual_rad = torch.zeros(self.count, device=self.device)
        self._maximum_applied_decoder_lower_body_correction_rad = torch.zeros(
            self.count, device=self.device
        )
        self._option_started = torch.zeros(self.count, dtype=torch.bool, device=self.device)
        self._option_active = torch.zeros_like(self._option_started)
        self._actor_plasticity_active = torch.zeros_like(self._option_started)
        self._option_age_steps = torch.zeros(self.count, dtype=torch.long, device=self.device)
        self._post_save_recovery_age_steps = torch.full(
            (self.count,), -1, dtype=torch.long, device=self.device
        )
        self._getup_start_recovery_age_steps = torch.full(
            (self.count,), -1, dtype=torch.long, device=self.device
        )
        self._post_save_recovery_anchor_y = torch.zeros(self.count, device=self.device)
        self._maximum_applied_post_save_counterstep = torch.zeros(self.count, device=self.device)
        self._estimated_arrival_time_sec = torch.full(
            (self.count,),
            self.config.first_shot_release_sec + self.dive_config.nominal_shot_flight_time_sec,
            device=self.device,
        )
        self._previous_option_phase = torch.zeros(self.count, device=self.device)
        self._maximum_posture_exception_steps = torch.zeros(
            self.count, dtype=torch.long, device=self.device
        )
        self._applied_option_gate = torch.zeros(self.count, device=self.device)
        self._maximum_applied_option_gate = torch.zeros(self.count, device=self.device)
        self._maximum_applied_runtime_reach_blend = torch.zeros(self.count, device=self.device)
        self._maximum_applied_whole_body_reach_blend = torch.zeros(self.count, device=self.device)
        self._previous_whole_body_reach_gate = torch.zeros(self.count, device=self.device)
        self._latched_whole_body_reach_target = torch.zeros((self.count, 17), device=self.device)
        self._maximum_applied_overhead_reach_blend = torch.zeros(self.count, device=self.device)
        self._maximum_applied_mosaic_gmt_blend = torch.zeros(self.count, device=self.device)
        self._maximum_applied_mosaic_gmt_getup_blend = torch.zeros(self.count, device=self.device)
        self._mosaic_gmt_getup_activation_steps = torch.zeros(
            self.count, dtype=torch.long, device=self.device
        )
        self._maximum_applied_lateral_drive = torch.zeros(self.count, device=self.device)
        self._learned_lateral_drive_gate = torch.zeros(self.count, device=self.device)
        self._maximum_learned_lateral_drive_gate = torch.zeros(self.count, device=self.device)
        self._maximum_applied_lateral_lunge_blend = torch.zeros(self.count, device=self.device)
        self._minimum_substep_arm_authority = torch.ones(self.count, device=self.device)
        self._minimum_substep_option_lower_body_authority = torch.ones(
            self.count, device=self.device
        )
        self._substep_stable_lower_body_target = torch.zeros((self.count, 12), device=self.device)
        motor_mirror_order = torch.as_tensor(_MIRROR_ORDER, dtype=torch.long, device=self.device)
        motor_mirror_sign = torch.as_tensor(_MIRROR_SIGN, dtype=torch.float32, device=self.device)
        motor_to_loco = torch.empty(29, dtype=torch.long, device=self.device)
        motor_to_loco[self._loco_to_motor] = torch.arange(29, device=self.device)
        self._loco_mirror_order = motor_to_loco[motor_mirror_order[self._loco_to_motor]]
        self._loco_mirror_sign = motor_mirror_sign[self._loco_to_motor]
        self._canonical_loco_hidden = torch.zeros_like(self._loco_hidden)
        self._canonical_loco_cell = torch.zeros_like(self._loco_cell)
        self._canonical_loco_action = torch.zeros_like(self._loco_action)
        self._canonical_locomotion_mirror_steps = torch.zeros(
            self.count, dtype=torch.long, device=self.device
        )
        self._contact_support_nonzero_steps = torch.zeros(
            self.count, dtype=torch.long, device=self.device
        )
        self._official_goalkeeper_teacher = None
        self._official_goalkeeper_teacher_contract = None
        self._official_goalkeeper_history = None
        self._official_goalkeeper_action = None
        self._official_goalkeeper_ball_last_local = None
        self._official_goalkeeper_kp = None
        self._official_goalkeeper_kd = None
        self._official_goalkeeper_default = None
        self._official_goalkeeper_initial = None
        self._previous_official_goalkeeper_target_delta = torch.zeros(
            (self.count, 29), device=self.device
        )
        self._maximum_official_goalkeeper_target_step = torch.zeros(
            (self.count, 3), device=self.device
        )
        if self.dive_config.official_goalkeeper_teacher_checkpoint_path is not None:
            from rosclaw_soccer.training.goalkeeper_combat_teacher import (
                OFFICIAL_GOALKEEPER_DEFAULT_QPOS,
                OFFICIAL_GOALKEEPER_INITIAL_QPOS,
                OFFICIAL_GOALKEEPER_KD,
                OFFICIAL_GOALKEEPER_KP,
                load_official_goalkeeper_teacher,
            )

            teacher_path = (
                Path(self.dive_config.official_goalkeeper_teacher_checkpoint_path)
                .expanduser()
                .resolve()
            )
            if len(teacher_path.parents) < 4:
                raise ValueError("official goalkeeper checkpoint layout is invalid")
            (
                self._official_goalkeeper_teacher,
                self._official_goalkeeper_teacher_contract,
            ) = load_official_goalkeeper_teacher(
                checkout=teacher_path.parents[3],
                checkpoint=teacher_path,
                device=self.device,
            )
            self._official_goalkeeper_history = torch.zeros(
                (self.count, 10, 96), device=self.device
            )
            self._official_goalkeeper_action = torch.zeros((self.count, 29), device=self.device)
            self._official_goalkeeper_ball_last_local = torch.zeros(
                (self.count, 3), device=self.device
            )
            self._official_goalkeeper_kp = torch.as_tensor(
                OFFICIAL_GOALKEEPER_KP, dtype=torch.float32, device=self.device
            )
            self._official_goalkeeper_kd = torch.as_tensor(
                OFFICIAL_GOALKEEPER_KD, dtype=torch.float32, device=self.device
            )
            self._official_goalkeeper_default = torch.as_tensor(
                OFFICIAL_GOALKEEPER_DEFAULT_QPOS,
                dtype=torch.float32,
                device=self.device,
            )
            self._official_goalkeeper_initial = torch.as_tensor(
                OFFICIAL_GOALKEEPER_INITIAL_QPOS,
                dtype=torch.float32,
                device=self.device,
            )
        self._maximum_applied_counter_rotation = torch.zeros(self.count, device=self.device)
        self._previous_option_arm_target = torch.zeros((self.count, 14), device=self.device)
        self._previous_option_waist_target = torch.zeros((self.count, 3), device=self.device)
        self._previous_option_lower_body_delta = torch.zeros((self.count, 12), device=self.device)
        self._maximum_applied_lower_body_target_step = torch.zeros(self.count, device=self.device)
        self._maximum_applied_waist_target_step = torch.zeros(self.count, device=self.device)
        self._runtime_reach_atlas = None
        self._runtime_reach_feedback_model = None
        if self.dive_config.runtime_reach_blend > 0.0:
            from rosclaw_soccer.training.goalkeeper_reach import (
                GoalkeeperReachAtlasConfig,
                GoalkeeperReachConfig,
                build_g1_task_space_reach_atlas,
                build_g1_task_space_reach_model,
            )

            reach_config = GoalkeeperReachConfig(
                damping=0.12,
                reach_gain=0.95,
                maximum_position_error_m=0.75,
                support_arm_scale=0.60,
                central_support_scale=0.95,
                residual_scale=1.0,
                arm_authority_scale=0.88,
                workspace_scale=2.50,
            )
            self._runtime_reach_atlas = build_g1_task_space_reach_atlas(
                asset_root,
                config=reach_config,
                atlas_config=GoalkeeperReachAtlasConfig(
                    interpolation_neighbors=42,
                    interpolation_kernel="gaussian",
                    interpolation_temperature=0.75,
                    multistart_count=12,
                ),
            )
            if self.dive_config.runtime_reach_feedback_blend > 0.0:
                self._runtime_reach_feedback_model = build_g1_task_space_reach_model(
                    asset_root,
                    config=reach_config,
                )
        self._whole_body_reach_atlas = None
        if self.dive_config.runtime_whole_body_reach_blend > 0.0:
            from rosclaw_soccer.training.goalkeeper_whole_body_reach import (
                GoalkeeperWholeBodyReachConfig,
                build_g1_whole_body_reach_atlas,
            )

            self._whole_body_reach_atlas = build_g1_whole_body_reach_atlas(
                asset_root,
                config=GoalkeeperWholeBodyReachConfig(
                    support_counterbalance_scale=(
                        self.dive_config.runtime_whole_body_reach_support_scale
                    )
                ),
            )
        self._whole_body_reach_joint_scale = torch.tensor(
            (self.dive_config.runtime_whole_body_reach_waist_scale,) * 3
            + (self.dive_config.runtime_whole_body_reach_arm_scale,) * 14,
            dtype=torch.float32,
            device=self.device,
        )
        self._overhead_reach_prior = None
        self._overhead_reference_times = None
        self._overhead_reference_position = None
        self._overhead_joint_scale = torch.tensor(
            (self.dive_config.overhead_reach_lower_body_scale,) * 12
            + (self.dive_config.overhead_reach_waist_scale,) * 3
            + (self.dive_config.overhead_reach_arm_scale,) * 14,
            dtype=torch.float32,
            device=self.device,
        )
        self._overhead_maximum_correction = torch.tensor(
            (0.35,) * 12 + (0.45,) * 3 + (2.80,) * 14,
            dtype=torch.float32,
            device=self.device,
        )
        if self.dive_config.overhead_reach_prior_path is not None:
            from rosclaw_soccer.growth.mosaic_overhead_reach_prior import (
                load_g1_mosaic_overhead_reach_prior,
            )
            from rosclaw_soccer.providers.g1.asset_qualification import g1_body_hash

            self._overhead_reach_prior = load_g1_mosaic_overhead_reach_prior(
                Path(self.dive_config.overhead_reach_prior_path)
            )
            scene = asset_root / "g1_description" / "scene_with_ball.xml"
            if self._overhead_reach_prior.body_hash != g1_body_hash(
                asset_root
            ) or self._overhead_reach_prior.physics_scene_hash != hash_bytes(scene.read_bytes()):
                raise ValueError("targeted dive overhead prior asset binding is invalid")
            self._overhead_reference_times = torch.as_tensor(
                self._overhead_reach_prior.reference_times_sec,
                dtype=torch.float32,
                device=self.device,
            )
            self._overhead_reference_position = torch.as_tensor(
                self._overhead_reach_prior.whole_body_position_reference_rad,
                dtype=torch.float32,
                device=self.device,
            )
        self._mosaic_gmt_controller = None
        self._mosaic_gmt_getup_controller = None
        self._mosaic_gmt_contract = None
        self._mosaic_gmt_skill = None
        self._mosaic_gmt_getup_skill = None
        self._mosaic_gmt_joint_scale = torch.tensor(
            (self.dive_config.mosaic_gmt_lower_body_scale,) * 12
            + (self.dive_config.mosaic_gmt_waist_scale,) * 3
            + (self.dive_config.mosaic_gmt_arm_scale,) * 14,
            dtype=torch.float32,
            device=self.device,
        )
        self._mosaic_gmt_getup_joint_scale = torch.tensor(
            (self.dive_config.mosaic_gmt_getup_lower_body_scale,) * 12
            + (self.dive_config.mosaic_gmt_getup_waist_scale,) * 3
            + (self.dive_config.mosaic_gmt_getup_arm_scale,) * 14,
            dtype=torch.float32,
            device=self.device,
        )
        self._mosaic_gmt_kp = self._kp.clone()
        self._mosaic_gmt_kd = self._kd.clone()
        self._torso_body = int(self.cpu_model.body("torso_link").id)
        if self.dive_config.mosaic_gmt_model_path is not None:
            from rosclaw_soccer.growth.mosaic_g1_contract import (
                MOSAIC_G1_ISAACLAB_TO_CANONICAL_DOF,
            )
            from rosclaw_soccer.growth.mosaic_gmt import (
                MosaicGMTTorchController,
                load_g1_mosaic_gmt_overhead_skill,
                load_mosaic_gmt_torch,
            )

            model_path = Path(self.dive_config.mosaic_gmt_model_path)
            policy, self._mosaic_gmt_contract = load_mosaic_gmt_torch(
                model_path,
                device=self.device,
            )
            if self.dive_config.mosaic_gmt_skill_path is not None:
                skill_path = Path(self.dive_config.mosaic_gmt_skill_path)
                self._mosaic_gmt_skill = load_g1_mosaic_gmt_overhead_skill(skill_path)
                self._mosaic_gmt_controller = MosaicGMTTorchController(
                    policy=policy,
                    contract=self._mosaic_gmt_contract,
                    skill=self._mosaic_gmt_skill,
                    environment_count=self.count,
                    device=self.device,
                )
            if self.dive_config.mosaic_gmt_getup_skill_path is not None:
                from rosclaw_soccer.growth.mosaic_getup import (
                    load_g1_mosaic_gmt_getup_skill,
                )

                getup_path = Path(self.dive_config.mosaic_gmt_getup_skill_path)
                self._mosaic_gmt_getup_skill = load_g1_mosaic_gmt_getup_skill(getup_path)
                scene = asset_root / "g1_description" / "scene_with_ball.xml"
                from rosclaw_soccer.providers.g1.asset_qualification import g1_body_hash

                if self._mosaic_gmt_getup_skill.body_hash != g1_body_hash(
                    asset_root
                ) or self._mosaic_gmt_getup_skill.physics_scene_hash != hash_bytes(
                    scene.read_bytes()
                ):
                    raise ValueError("targeted dive get-up skill asset binding is invalid")
                self._mosaic_gmt_getup_controller = MosaicGMTTorchController(
                    policy=policy,
                    contract=self._mosaic_gmt_contract,
                    skill=self._mosaic_gmt_getup_skill,
                    environment_count=self.count,
                    device=self.device,
                )
            mapping = torch.as_tensor(
                MOSAIC_G1_ISAACLAB_TO_CANONICAL_DOF,
                dtype=torch.long,
                device=self.device,
            )
            self._mosaic_gmt_kp = torch.as_tensor(
                self._mosaic_gmt_contract.joint_stiffness,
                device=self.device,
            )[mapping]
            self._mosaic_gmt_kd = torch.as_tensor(
                self._mosaic_gmt_contract.joint_damping,
                device=self.device,
            )[mapping]
        ready = torch.zeros(29, dtype=torch.float32, device=self.device)
        ready[self._loco_to_motor] = self._loco_default
        self._runtime_reach_ready = ready
        self._substep_stable_lower_body_target.copy_(ready[:12].unsqueeze(0))
        self._latched_whole_body_reach_target.copy_(ready[12:29].unsqueeze(0))
        self._previous_option_arm_target.copy_(ready[15:29].unsqueeze(0))
        self._previous_option_waist_target.copy_(ready[12:15].unsqueeze(0))
        self._runtime_reach_limits = (
            None
            if self._runtime_reach_atlas is None
            else torch.as_tensor(
                tuple(self._runtime_reach_atlas.effective_arm_limits_rad) * 2,
                dtype=torch.float32,
                device=self.device,
            )
        )
        self._maximum_runtime_reach_feedback_correction = torch.zeros(
            self.count, device=self.device
        )

    def reset(self, *, seed: int) -> Any:
        self._option_started.zero_()
        self._option_active.zero_()
        self._actor_plasticity_active.zero_()
        self._option_age_steps.zero_()
        self._post_save_recovery_age_steps.fill_(-1)
        self._getup_start_recovery_age_steps.fill_(-1)
        self._post_save_recovery_anchor_y.zero_()
        self._maximum_applied_post_save_counterstep.zero_()
        self._estimated_arrival_time_sec.fill_(
            self.config.first_shot_release_sec + self.dive_config.nominal_shot_flight_time_sec
        )
        self._previous_option_phase.zero_()
        self._maximum_posture_exception_steps.zero_()
        self._applied_option_gate.zero_()
        self._maximum_applied_option_gate.zero_()
        self._maximum_applied_runtime_reach_blend.zero_()
        self._maximum_runtime_reach_feedback_correction.zero_()
        self._maximum_applied_whole_body_reach_blend.zero_()
        self._previous_whole_body_reach_gate.zero_()
        self._latched_whole_body_reach_target.copy_(self._runtime_reach_ready[12:29].unsqueeze(0))
        self._maximum_applied_overhead_reach_blend.zero_()
        self._maximum_applied_mosaic_gmt_blend.zero_()
        self._maximum_applied_mosaic_gmt_getup_blend.zero_()
        self._mosaic_gmt_getup_activation_steps.zero_()
        self._maximum_applied_lateral_drive.zero_()
        self._learned_lateral_drive_gate.zero_()
        self._maximum_learned_lateral_drive_gate.zero_()
        self._maximum_applied_lateral_lunge_blend.zero_()
        self._minimum_substep_arm_authority.fill_(1.0)
        self._minimum_substep_option_lower_body_authority.fill_(1.0)
        self._substep_stable_lower_body_target.copy_(self._runtime_reach_ready[:12].unsqueeze(0))
        self._canonical_loco_hidden.zero_()
        self._canonical_loco_cell.zero_()
        self._canonical_loco_action.zero_()
        self._canonical_locomotion_mirror_steps.zero_()
        self._contact_support_nonzero_steps.zero_()
        self._previous_official_goalkeeper_target_delta.zero_()
        self._maximum_official_goalkeeper_target_step.zero_()
        if self._official_goalkeeper_history is not None:
            self._official_goalkeeper_history.zero_()
            assert self._official_goalkeeper_action is not None
            assert self._official_goalkeeper_ball_last_local is not None
            self._official_goalkeeper_action.zero_()
            self._official_goalkeeper_ball_last_local.zero_()
        self._maximum_applied_counter_rotation.zero_()
        self._previous_option_lower_body_delta.zero_()
        self._maximum_applied_lower_body_target_step.zero_()
        self._maximum_applied_waist_target_step.zero_()
        self._previous_option_arm_target.copy_(self._runtime_reach_ready[15:29].unsqueeze(0))
        self._previous_option_waist_target.copy_(self._runtime_reach_ready[12:15].unsqueeze(0))
        if self._mosaic_gmt_controller is not None:
            self._mosaic_gmt_controller.reset()
        if self._mosaic_gmt_getup_controller is not None:
            self._mosaic_gmt_getup_controller.reset()
        observation = super().reset(seed=seed)
        if self._official_goalkeeper_teacher is None:
            return observation
        initial = self._official_goalkeeper_initial
        if initial is None:
            raise RuntimeError("official goalkeeper initial posture is unavailable")
        blend = self.dive_config.official_goalkeeper_teacher_blend
        blended_initial = self._runtime_reach_ready + blend * (initial - self._runtime_reach_ready)
        self.qpos[:, 7:36] = blended_initial
        self._target.copy_(blended_initial.unsqueeze(0))
        self._previous_official_goalkeeper_target_delta.copy_(
            (blend * (initial - self._runtime_reach_ready)).unsqueeze(0)
        )
        self._substep_stable_lower_body_target.copy_(blended_initial[:12].unsqueeze(0))
        self._previous_option_arm_target.copy_(blended_initial[15:29].unsqueeze(0))
        self._previous_option_waist_target.copy_(blended_initial[12:15].unsqueeze(0))
        self.mjw.forward(self.model, self.data)
        left_relative = self.geom_xpos[:, self._left_hand_geom] - self.qpos[:, :3]
        right_relative = self.geom_xpos[:, self._right_hand_geom] - self.qpos[:, :3]
        self._ready_left_hand_relative.copy_(left_relative)
        self._ready_right_hand_relative.copy_(right_relative)
        self._previous_left_hand_relative.copy_(left_relative)
        self._previous_right_hand_relative.copy_(right_relative)
        return self.observation()

    def _target_estimate(self) -> tuple[Any, Any]:
        torch = self.torch
        first_release = int(round(self.config.first_shot_release_sec / self.config.control_dt_sec))
        before_release = self._step_index < first_release
        cue_step = first_release - int(
            round(self.dive_config.prediction_lead_sec / self.config.control_dt_sec)
        )
        causal = self._causal_intercept()
        cue = self._intent_cue_one
        cue_target = torch.stack(
            (
                torch.full_like(cue[:, 0], self.config.keeper_x_m - 0.08),
                cue[:, 0],
                cue[:, 1],
            ),
            dim=1,
        )
        target = cue_target if before_release else causal
        visible = (
            (cue[:, 2] > 0.5) & (self._step_index >= cue_step)
            if before_release
            else self._shot_index == 1
        )
        return target, visible

    def observation(self) -> Any:
        observation = super().observation()
        first_release = int(round(self.config.first_shot_release_sec / self.config.control_dt_sec))
        cue_step = first_release - int(
            round(self.dive_config.prediction_lead_sec / self.config.control_dt_sec)
        )
        if self._step_index < cue_step:
            # cue(3), phase(3), normalized time(1) occupy the observation tail.
            observation[:, self.observation_size - 7 : self.observation_size - 4] = 0.0
        return observation

    def _actor_auxiliary_proprioception(self) -> Any:
        """Expose causal landing contacts and post-save recovery context.

        The recovery actor used to see root velocity but not the causal save
        latch, recovery age, absolute lateral displacement, or capture error.
        Consequently identical-looking flight and landing states required
        opposite actions.  These features contain no future shot truth: every
        value is available after contact from proprioception and the task event
        latch already used by the runtime counterstep controller.
        """

        if not self.dive_config.actor_contact_support_side_enabled:
            return super()._actor_auxiliary_proprioception()
        left_contact, right_contact = self._foot_contact_state()
        support_side = right_contact.to(self.torch.float32) - left_contact.to(self.torch.float32)
        contact_context = self.torch.stack(
            (
                support_side,
                left_contact.to(self.torch.float32),
                right_contact.to(self.torch.float32),
            ),
            dim=1,
        )
        if not self.dive_config.actor_recovery_context_enabled:
            return contact_context
        saved = self.task.first_save.to(self.torch.float32)
        age_steps = self.torch.clamp(self._post_save_recovery_age_steps, min=0)
        recovery_age = self.torch.clamp(
            age_steps.to(self.torch.float32)
            * self.config.control_dt_sec
            / self.dive_config.post_save_counterstep_duration_sec,
            0.0,
            1.0,
        )
        recovery_age *= saved
        recovery_goal = (
            1.0 - self.dive_config.post_save_counterstep_recenter_weight
        ) * self._post_save_recovery_anchor_y
        lateral_position = self.qpos[:, 1]
        capture_error = (
            lateral_position
            + self.dive_config.post_save_counterstep_capture_horizon_sec * self.qvel[:, 1]
            - recovery_goal
        )
        recovery_context = self.torch.stack(
            (
                saved,
                recovery_age,
                0.5 * lateral_position,
                0.5 * capture_error,
            ),
            dim=1,
        )
        return self.torch.cat((contact_context, recovery_context), dim=1)

    def _causal_intercept(self) -> Any:
        intercept = super()._causal_intercept()
        first_release = int(round(self.config.first_shot_release_sec / self.config.control_dt_sec))
        cue_step = first_release - int(
            round(self.dive_config.prediction_lead_sec / self.config.control_dt_sec)
        )
        if self._step_index < cue_step:
            intercept[:, 1] = 0.0
            intercept[:, 2] = 0.82
        return intercept

    def _estimated_time_to_arrival(self) -> Any:
        """Use a public nominal prior pre-kick and measured ball state post-kick."""

        torch = self.torch
        current_time = self._step_index * self.config.control_dt_sec
        first_release = self.config.first_shot_release_sec
        nominal = torch.full(
            (self.count,),
            max(0.0, first_release + self.dive_config.nominal_shot_flight_time_sec - current_time),
            device=self.device,
        )
        ball = self.qpos[:, 36:39]
        velocity = self.qvel[:, 35:38]
        causal = torch.clamp(
            (self.config.keeper_x_m - 0.08 - ball[:, 0]) / torch.clamp(velocity[:, 0], min=0.10),
            0.0,
            1.2,
        )
        release_step = int(round(first_release / self.config.control_dt_sec))
        released = (self._step_index >= release_step) & (self._shot_index == 1)
        if self._step_index == release_step:
            # Latch one causal, measured absolute arrival prediction.  A
            # clamped time-to-arrival becomes zero after a miss and otherwise
            # looks indistinguishable from "arriving now", which used to hold
            # the reach pose indefinitely.  The absolute event clock gives
            # saves and misses the same finite follow-through semantics.
            self._estimated_arrival_time_sec.copy_(current_time + causal)
        return torch.where(released, causal, nominal)

    def _locomotion_target(self, lateral_action: Any) -> Any:
        torch = self.torch
        self._step_kp = self._kp
        self._step_kd = self._kd
        target, visible = self._target_estimate()
        lateral_error = target[:, 1] - self.qpos[:, 1]
        start = (
            ~self._option_started
            & visible
            & (torch.abs(lateral_error) >= self.dive_config.activation_minimum_lateral_error_m)
        )
        self._option_started |= start
        duration_steps = int(
            round(self.dive_config.option_duration_sec / self.config.control_dt_sec)
        )
        hold_steps = int(round(self.dive_config.phase_hold_sec / self.config.control_dt_sec))
        total_steps = hold_steps + duration_steps
        self._option_active.copy_(self._option_started & (self._option_age_steps < total_steps))
        requested_gate = torch.clamp(lateral_action, min=0.0, max=1.0)
        drive_gate = (
            self._learned_lateral_drive_gate
            if self.dive_config.lateral_drive_learned_gate_enabled
            else requested_gate
        )
        drive_activation = torch.clamp(
            drive_gate / self.dive_config.lateral_drive_full_activation_gate,
            0.0,
            1.0,
        )
        if self.dive_config.lateral_drive_capture_enabled:
            lateral_drive = _capture_point_lateral_drive(
                torch=torch,
                target_lateral_m=target[:, 1],
                root_lateral_m=self.qpos[:, 1],
                root_lateral_velocity_mps=self.qvel[:, 1],
                target_standoff_m=self.dive_config.lateral_drive_target_standoff_m,
                capture_horizon_sec=self.dive_config.lateral_drive_capture_horizon_sec,
                capture_scale_m=self.dive_config.lateral_drive_capture_scale_m,
            )
            lateral_drive *= self.dive_config.lateral_drive_scale
        else:
            lateral_drive = -torch.sign(lateral_error) * self.dive_config.lateral_drive_scale
        lateral_drive *= torch.where(
            lateral_error < 0.0,
            torch.full_like(
                lateral_drive,
                self.dive_config.negative_target_lateral_drive_scale,
            ),
            torch.ones_like(lateral_drive),
        )
        lateral_drive *= drive_activation
        lateral_drive = torch.where(
            self._option_active & visible,
            lateral_drive,
            torch.zeros_like(lateral_drive),
        )
        if self.dive_config.post_save_counterstep_enabled:
            newly_saved = self.task.first_save & (self._post_save_recovery_age_steps < 0)
            self._post_save_recovery_anchor_y.copy_(
                torch.where(
                    newly_saved,
                    self.qpos[:, 1],
                    self._post_save_recovery_anchor_y,
                )
            )
            self._post_save_recovery_age_steps.copy_(
                torch.where(
                    newly_saved,
                    torch.zeros_like(self._post_save_recovery_age_steps),
                    torch.where(
                        self._post_save_recovery_age_steps >= 0,
                        self._post_save_recovery_age_steps + 1,
                        self._post_save_recovery_age_steps,
                    ),
                )
            )
            recovery_steps = max(
                1,
                int(
                    round(
                        self.dive_config.post_save_counterstep_duration_sec
                        / self.config.control_dt_sec
                    )
                ),
            )
            recovery_active = (
                (self._post_save_recovery_age_steps >= 0)
                & (self._post_save_recovery_age_steps < recovery_steps)
                & (self.task.phase != 7)
            )
            capture = self.qpos[:, 1] + (
                self.dive_config.post_save_counterstep_capture_horizon_sec * self.qvel[:, 1]
            )
            recovery_goal = (
                self.dive_config.post_save_counterstep_recenter_weight
                * torch.zeros_like(self._post_save_recovery_anchor_y)
                + (1.0 - self.dive_config.post_save_counterstep_recenter_weight)
                * self._post_save_recovery_anchor_y
            )
            # Keeper yaw is pi: a positive local lateral command accelerates
            # toward world -y.  Therefore capture minus recovery goal has the
            # correct braking sign.  A zero recenter weight brakes around the
            # actual save landing instead of dragging the keeper to midfield.
            counterstep = torch.clamp(
                (capture - recovery_goal) / 0.50,
                -self.dive_config.post_save_counterstep_command_limit,
                self.dive_config.post_save_counterstep_command_limit,
            )
            counterstep = torch.where(recovery_active, counterstep, torch.zeros_like(counterstep))
            lateral_drive = torch.where(recovery_active, counterstep, lateral_drive)
            self._maximum_applied_post_save_counterstep.copy_(
                torch.maximum(
                    self._maximum_applied_post_save_counterstep,
                    torch.abs(counterstep),
                )
            )
        self._maximum_applied_lateral_drive.copy_(
            torch.maximum(self._maximum_applied_lateral_drive, torch.abs(lateral_drive))
        )
        stable_target = super()._locomotion_target(lateral_drive)
        if getattr(self, "_official_goalkeeper_teacher", None) is not None:
            stable_target = self._official_goalkeeper_target(stable_target)
        elif self.dive_config.canonical_locomotion_mirror_enabled:
            canonical_observation = self._canonical_locomotion_observation(lateral_drive)
            with torch.inference_mode():
                encoded = self._loco_policy.normalizer.forward(
                    torch.clamp(canonical_observation, -100.0, 100.0)
                )
                sequence, (hidden, cell) = self._loco_rnn(
                    encoded.unsqueeze(0),
                    (self._canonical_loco_hidden, self._canonical_loco_cell),
                )
                self._canonical_loco_hidden.copy_(hidden)
                self._canonical_loco_cell.copy_(cell)
                self._canonical_loco_action.copy_(
                    torch.clamp(
                        self._loco_policy.actor.forward(sequence.squeeze(0)),
                        -100.0,
                        100.0,
                    )
                )
            mirrored_loco_action = (
                self._canonical_loco_action[:, self._loco_mirror_order] * self._loco_mirror_sign
            )
            mirrored_target = torch.zeros_like(stable_target)
            mirrored_target[:, self._loco_to_motor] = (
                0.25 * mirrored_loco_action + self._loco_default
            )
            use_mirror = lateral_drive > 0.0
            stable_target = torch.where(use_mirror.unsqueeze(1), mirrored_target, stable_target)
            self._canonical_locomotion_mirror_steps += use_mirror.to(torch.long)
        self._substep_stable_lower_body_target.copy_(stable_target[:, :12])
        raw_phase = torch.clamp(
            (self._option_age_steps - hold_steps).to(torch.float32) / max(duration_steps - 1, 1),
            0.0,
            1.0,
        )
        time_to_arrival = self._estimated_time_to_arrival()
        arm_phase = raw_phase
        if self.dive_config.intercept_phase_at_arrival is not None:
            arrival_raw_phase = torch.clamp(
                raw_phase
                + time_to_arrival
                / max(self.dive_config.option_duration_sec, self.config.control_dt_sec),
                0.05,
                0.95,
            )
            intercept_phase = self.dive_config.intercept_phase_at_arrival
            arm_phase = torch.where(
                raw_phase <= arrival_raw_phase,
                intercept_phase * raw_phase / arrival_raw_phase,
                intercept_phase
                + (1.0 - intercept_phase)
                * (raw_phase - arrival_raw_phase)
                / (1.0 - arrival_raw_phase),
            )
            arm_phase = torch.where(
                self._option_active,
                torch.maximum(arm_phase, self._previous_option_phase),
                arm_phase,
            )
            arm_phase = torch.where(
                target[:, 2] >= self.dive_config.phase_sync_minimum_target_height_m,
                arm_phase,
                raw_phase,
            )
        phase_scale = torch.where(
            target[:, 2] < 0.60,
            torch.full_like(raw_phase, self.dive_config.low_shot_phase_scale),
            torch.where(
                target[:, 2] < 1.10,
                torch.full_like(raw_phase, self.dive_config.mid_shot_phase_scale),
                torch.full_like(raw_phase, self.dive_config.high_shot_phase_scale),
            ),
        )
        body_phase = torch.clamp(raw_phase * phase_scale, 0.0, 1.0)
        arm_phase = torch.clamp(arm_phase * phase_scale, 0.0, 1.0)
        self._previous_option_phase.copy_(arm_phase)
        current_time = self._step_index * self.config.control_dt_sec
        arrival_time = self._estimated_arrival_time_sec
        qw, qx, qy, qz = (self.qpos[:, index] for index in range(3, 7))
        upright = 2.0 * (qw * qw + qz * qz) - 1.0
        contact_support_side = (
            self._foot_support_side()
            if self.dive_config.runtime_contact_support_side_enabled
            else torch.zeros(self.count, device=self.device)
        )
        self._contact_support_nonzero_steps += (torch.abs(contact_support_side) > 0.5).to(
            torch.long
        )

        def features_for(phase_value: Any) -> Any:
            return targeted_dive_features_torch(
                torch=torch,
                direction=torch.where(lateral_error < 0.0, -1.0, 1.0),
                phase=phase_value,
                target_lateral_m=target[:, 1],
                target_height_m=target[:, 2],
                time_to_arrival_sec=time_to_arrival,
                root_lateral_m=self.qpos[:, 1],
                root_lateral_speed_mps=self.qvel[:, 1],
                pelvis_height_m=self.qpos[:, 2],
                upright_projection=upright,
                support_side=contact_support_side,
                root_angular_speed_rad_s=torch.linalg.vector_norm(self.qvel[:, 3:6], dim=1),
            )

        features = features_for(body_phase)
        option_target = decode_goalkeeper_targeted_dive(
            model=self.targeted_dive,
            checkpoint=self.targeted_dive_checkpoint,
            features=features,
            residual_authority=self._decoder_group_authority,
        )
        if self.dive_config.intercept_phase_at_arrival is not None:
            arm_target = decode_goalkeeper_targeted_dive(
                model=self.targeted_dive,
                checkpoint=self.targeted_dive_checkpoint,
                features=features_for(arm_phase),
                residual_authority=self._decoder_group_authority,
            )
            option_target = option_target.clone()
            option_target[:, 15:29] = arm_target[:, 15:29]
        gate = torch.where(
            self._option_active,
            torch.clamp(lateral_action, min=0.0, max=1.0),
            torch.zeros_like(lateral_action),
        )
        self._applied_option_gate.copy_(gate)
        self._maximum_applied_option_gate.copy_(
            torch.maximum(self._maximum_applied_option_gate, gate)
        )
        # Without a lateral locomotion anchor, avoid warming recurrent memory
        # with off-manifold dive states.  Configured drive keeps that memory
        # because the frozen locomotion expert actively owns the anchor.
        if self.dive_config.lateral_drive_scale == 0.0:
            active_ids = self._option_active & (gate > 0.05)
            self._loco_hidden[:, active_ids] = 0.0
            self._loco_cell[:, active_ids] = 0.0
            self._loco_action[active_ids] = 0.0
        self._option_age_steps[self._option_started] += 1
        joint_gate = gate.unsqueeze(1) * self._anchor_group_scale.unsqueeze(0)
        post_save_release = torch.ones_like(gate)
        if self.dive_config.post_save_counterstep_enabled:
            release_steps = max(
                1,
                int(
                    round(
                        self.dive_config.post_save_option_release_sec / self.config.control_dt_sec
                    )
                ),
            )
            release_phase = torch.clamp(
                self._post_save_recovery_age_steps.to(torch.float32) / release_steps,
                0.0,
                1.0,
            )
            release = 1.0 - release_phase.square() * (3.0 - 2.0 * release_phase)
            recovering = self._post_save_recovery_age_steps >= 0
            post_save_release = torch.where(
                recovering,
                release,
                torch.ones_like(release),
            )
            joint_gate *= post_save_release.unsqueeze(1)
        commanded = stable_target + joint_gate * (option_target - stable_target)
        if self._lower_body_decoder_command_boost > 0.0:
            # Keep the frozen motion anchor conservative, while giving only
            # the learned, target-conditioned posture residual a separately
            # declared command scale.  Previously the anchor scale attenuated
            # a learned crouch a second time before joint control.
            anchor_target = decode_goalkeeper_targeted_dive(
                model=self.targeted_dive,
                checkpoint=self.targeted_dive_checkpoint,
                features=features,
                residual_authority=0.0,
            )
            decoder_delta = option_target[:, :12] - anchor_target[:, :12]
            self._maximum_decoder_lower_body_residual_rad.copy_(
                torch.maximum(
                    self._maximum_decoder_lower_body_residual_rad,
                    torch.max(torch.abs(decoder_delta), dim=1).values,
                )
            )
            applied_decoder_correction = (
                gate.unsqueeze(1)
                * post_save_release.unsqueeze(1)
                * self._lower_body_decoder_command_boost
                * decoder_delta
            )
            self._maximum_applied_decoder_lower_body_correction_rad.copy_(
                torch.maximum(
                    self._maximum_applied_decoder_lower_body_correction_rad,
                    torch.max(torch.abs(applied_decoder_correction), dim=1).values,
                )
            )
            commanded[:, :12] += applied_decoder_correction
        if self._runtime_reach_atlas is not None:
            from rosclaw_soccer.training.goalkeeper_reach import (
                task_space_reach_from_target_torch,
            )

            relative = torch.stack(
                (
                    target[:, 0] - self.qpos[:, 0],
                    target[:, 1] - self.qpos[:, 1],
                    target[:, 2] - self.qpos[:, 2],
                ),
                dim=1,
            )
            relative[:, 1] += (
                torch.sign(relative[:, 1]) * self.dive_config.runtime_reach_lateral_lead_m
            )
            height_lead = torch.full_like(
                relative[:, 2],
                self.dive_config.runtime_reach_vertical_lead_m,
            )
            if self.dive_config.runtime_reach_low_vertical_lead_m is not None:
                if (
                    self.dive_config.runtime_reach_mid_vertical_lead_m is None
                    or self.dive_config.runtime_reach_high_vertical_lead_m is None
                ):
                    raise RuntimeError("targeted dive height lead validation drifted")
                low_lead = float(self.dive_config.runtime_reach_low_vertical_lead_m)
                mid_lead = float(self.dive_config.runtime_reach_mid_vertical_lead_m)
                high_lead = float(self.dive_config.runtime_reach_high_vertical_lead_m)
                height_lead = torch.where(
                    target[:, 2] < 0.60,
                    torch.full_like(height_lead, low_lead),
                    torch.where(
                        target[:, 2] < 1.10,
                        torch.full_like(height_lead, mid_lead),
                        torch.full_like(height_lead, high_lead),
                    ),
                )
            relative[:, 2] += height_lead
            if self.dive_config.runtime_reach_contact_standoff_m > 0.0:
                # The arm controls the hand/glove centre, while success is a
                # physical surface contact with a finite-radius football.  An
                # exact ball-centre IK target overstates required reach by the
                # combined collision envelopes and makes hard corners appear
                # unreachable.  Pull the target toward the pelvis by a bounded
                # declared standoff; MuJoCo contact and post-impact velocity
                # remain the only save authority.
                distance = torch.linalg.vector_norm(relative, dim=1, keepdim=True)
                scale = torch.clamp(
                    (distance - self.dive_config.runtime_reach_contact_standoff_m)
                    / torch.clamp(distance, min=1.0e-6),
                    min=0.0,
                    max=1.0,
                )
                relative = relative * scale
            normalized_reach = task_space_reach_from_target_torch(
                torch=torch,
                target_relative=relative,
                model=self._runtime_reach_atlas,
            )
            if self._runtime_reach_limits is None:
                raise RuntimeError("targeted dive runtime reach limits are unavailable")
            reach_target = self._runtime_reach_ready[15:29].unsqueeze(0)
            reach_target = reach_target + normalized_reach * self._runtime_reach_limits
            if self._runtime_reach_feedback_model is not None:
                feedback = self._runtime_reach_feedback_model
                left_inverse = torch.as_tensor(
                    feedback.left_damped_inverse,
                    dtype=torch.float32,
                    device=self.device,
                )
                right_inverse = torch.as_tensor(
                    feedback.right_damped_inverse,
                    dtype=torch.float32,
                    device=self.device,
                )
                left_relative = self.geom_xpos[:, self._left_hand_geom] - self.qpos[:, :3]
                right_relative = self.geom_xpos[:, self._right_hand_geom] - self.qpos[:, :3]

                def bounded_error(current: Any) -> Any:
                    error = relative - current
                    norm = torch.linalg.vector_norm(error, dim=1, keepdim=True)
                    scale = torch.clamp(
                        self.dive_config.runtime_reach_feedback_maximum_error_m
                        / torch.clamp(norm, min=1.0e-6),
                        max=1.0,
                    )
                    return error * scale

                left_correction = (
                    bounded_error(left_relative) @ left_inverse.transpose(0, 1)
                ) * self.dive_config.runtime_reach_feedback_gain
                right_correction = (
                    bounded_error(right_relative) @ right_inverse.transpose(0, 1)
                ) * self.dive_config.runtime_reach_feedback_gain
                feedback_target = reach_target.clone()
                mirror_sign = torch.as_tensor(
                    (1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0),
                    dtype=torch.float32,
                    device=self.device,
                )
                support_scale = self.dive_config.runtime_reach_feedback_support_scale
                negative_target = (relative[:, 1] < 0.0).unsqueeze(1)
                feedback_target[:, :7] = torch.where(
                    negative_target,
                    self.qpos[:, 22:29] + left_correction,
                    self.qpos[:, 22:29] + support_scale * right_correction * mirror_sign,
                )
                feedback_target[:, 7:14] = torch.where(
                    negative_target,
                    self.qpos[:, 29:36] + support_scale * left_correction * mirror_sign,
                    self.qpos[:, 29:36] + right_correction,
                )
                feedback_blend = self.dive_config.runtime_reach_feedback_blend
                applied = feedback_blend * (feedback_target - reach_target)
                reach_target = reach_target + applied
                self._maximum_runtime_reach_feedback_correction.copy_(
                    torch.maximum(
                        self._maximum_runtime_reach_feedback_correction,
                        torch.max(torch.abs(applied), dim=1).values,
                    )
                )
            cfg = self.dive_config
            reach_progress = torch.clamp(
                (cfg.runtime_reach_approach_horizon_sec - (arrival_time - current_time))
                / (cfg.runtime_reach_approach_horizon_sec - cfg.runtime_reach_full_lead_sec),
                0.0,
                1.0,
            )
            reach_progress = reach_progress * reach_progress * (3.0 - 2.0 * reach_progress)
            post_arrival_hold = torch.clamp(
                (arrival_time - current_time + cfg.runtime_reach_hold_after_arrival_sec)
                / cfg.runtime_reach_hold_after_arrival_sec,
                0.0,
                1.0,
            )
            reach_gate = cfg.runtime_reach_blend * reach_progress * post_arrival_hold
            reach_gate *= torch.clamp(
                gate / cfg.runtime_reach_full_activation_gate,
                0.0,
                1.0,
            )
            reach_gate = torch.where(visible, reach_gate, torch.zeros_like(reach_gate))
            commanded[:, 15:29] += reach_gate.unsqueeze(1) * (reach_target - commanded[:, 15:29])
            self._maximum_applied_runtime_reach_blend.copy_(
                torch.maximum(self._maximum_applied_runtime_reach_blend, reach_gate)
            )
        if self._whole_body_reach_atlas is not None:
            from rosclaw_soccer.training.goalkeeper_whole_body_reach import (
                whole_body_reach_from_target_torch,
            )

            whole_body_relative = torch.stack(
                (
                    target[:, 0] - self.qpos[:, 0],
                    target[:, 1] - self.qpos[:, 1],
                    target[:, 2] - self.qpos[:, 2],
                ),
                dim=1,
            )
            whole_body_delta = whole_body_reach_from_target_torch(
                torch=torch,
                target_relative=whole_body_relative,
                model=self._whole_body_reach_atlas,
            )
            cfg = self.dive_config
            height_alpha = torch.clamp(
                (target[:, 2] - cfg.runtime_whole_body_reach_full_below_height_m)
                / (
                    cfg.runtime_whole_body_reach_maximum_height_m
                    - cfg.runtime_whole_body_reach_full_below_height_m
                ),
                0.0,
                1.0,
            )
            height_gate = 1.0 - height_alpha.square() * (3.0 - 2.0 * height_alpha)
            reach_progress = torch.clamp(
                (cfg.runtime_reach_approach_horizon_sec - time_to_arrival)
                / (cfg.runtime_reach_approach_horizon_sec - cfg.runtime_reach_full_lead_sec),
                0.0,
                1.0,
            )
            reach_progress = reach_progress.square() * (3.0 - 2.0 * reach_progress)
            post_arrival_hold = torch.clamp(
                (time_to_arrival + cfg.runtime_reach_hold_after_arrival_sec)
                / cfg.runtime_reach_hold_after_arrival_sec,
                0.0,
                1.0,
            )
            whole_body_gate = (
                cfg.runtime_whole_body_reach_blend
                * height_gate
                * reach_progress
                * post_arrival_hold
            )
            whole_body_gate *= torch.clamp(
                gate / cfg.runtime_reach_full_activation_gate,
                0.0,
                1.0,
            )
            desired = self._runtime_reach_ready[12:29].unsqueeze(0)
            desired = desired + whole_body_delta * self._whole_body_reach_joint_scale
            live_target = visible & (whole_body_gate > 1.0e-4)
            self._latched_whole_body_reach_target.copy_(
                torch.where(
                    live_target.unsqueeze(1),
                    desired,
                    self._latched_whole_body_reach_target,
                )
            )
            release_step = (
                cfg.runtime_whole_body_reach_blend
                * self.config.control_dt_sec
                / cfg.runtime_whole_body_reach_release_sec
            )
            releasing_gate = torch.clamp(
                self._previous_whole_body_reach_gate - release_step,
                min=0.0,
            )
            whole_body_gate = torch.where(visible, whole_body_gate, releasing_gate)
            desired = torch.where(
                visible.unsqueeze(1),
                desired,
                self._latched_whole_body_reach_target,
            )
            commanded[:, 12:29] += whole_body_gate.unsqueeze(1) * (desired - commanded[:, 12:29])
            self._previous_whole_body_reach_gate.copy_(whole_body_gate)
            self._maximum_applied_whole_body_reach_blend.copy_(
                torch.maximum(
                    self._maximum_applied_whole_body_reach_blend,
                    whole_body_gate,
                )
            )
        if (
            self._overhead_reference_times is not None
            and self._overhead_reference_position is not None
        ):
            cfg = self.dive_config
            times = self._overhead_reference_times
            reference = self._overhead_reference_position
            relative_time = -time_to_arrival
            interval = times[1] - times[0]
            fractional_index = torch.clamp(
                (relative_time - times[0]) / interval,
                0.0,
                float(times.shape[0] - 1),
            )
            lower_index = torch.floor(fractional_index).to(torch.long)
            upper_index = torch.clamp(lower_index + 1, max=times.shape[0] - 1)
            fraction = (fractional_index - lower_index.to(torch.float32)).unsqueeze(1)
            desired = reference[lower_index] * (1.0 - fraction) + reference[upper_index] * fraction

            def smoothstep(value: Any) -> Any:
                clipped = torch.clamp(value, 0.0, 1.0)
                return clipped.square() * (3.0 - 2.0 * clipped)

            height_gate = smoothstep(
                (target[:, 2] - cfg.overhead_reach_minimum_target_height_m)
                / (
                    cfg.overhead_reach_full_target_height_m
                    - cfg.overhead_reach_minimum_target_height_m
                )
            )
            rise = smoothstep((relative_time - times[0]) / -times[0])
            hold_end = min(0.18, float(times[-1]) - 0.05)
            decay = 1.0 - smoothstep((relative_time - hold_end) / (times[-1] - hold_end))
            overhead_gate = cfg.overhead_reach_blend * height_gate * torch.minimum(rise, decay)
            overhead_gate *= torch.clamp(
                gate / cfg.runtime_reach_full_activation_gate,
                0.0,
                1.0,
            )
            inside = (relative_time >= times[0]) & (relative_time <= times[-1])
            overhead_gate = torch.where(
                visible & inside,
                overhead_gate,
                torch.zeros_like(overhead_gate),
            )
            correction = torch.clamp(
                desired - commanded,
                -self._overhead_maximum_correction,
                self._overhead_maximum_correction,
            )
            commanded += (
                overhead_gate.unsqueeze(1) * self._overhead_joint_scale.unsqueeze(0) * correction
            )
            self._maximum_applied_overhead_reach_blend.copy_(
                torch.maximum(self._maximum_applied_overhead_reach_blend, overhead_gate)
            )
        if self._mosaic_gmt_controller is not None:
            cfg = self.dive_config

            def smoothstep(value: Any) -> Any:
                clipped = torch.clamp(value, 0.0, 1.0)
                return clipped.square() * (3.0 - 2.0 * clipped)

            height_gate = smoothstep(
                (target[:, 2] - cfg.mosaic_gmt_minimum_target_height_m)
                / (cfg.mosaic_gmt_full_target_height_m - cfg.mosaic_gmt_minimum_target_height_m)
            )
            # Preserve a bounded low-shot stabilizer only for a deliberately
            # disjoint lower-body/waist composition.  The config contract
            # forbids this floor when GMT owns any arm joint, so an overhead
            # clip cannot become a hidden low-ball reach teacher.
            height_gate = (
                cfg.mosaic_gmt_stability_floor
                + (1.0 - cfg.mosaic_gmt_stability_floor) * height_gate
            )
            gmt_gate = cfg.mosaic_gmt_blend * height_gate
            gmt_gate *= torch.clamp(
                gate / cfg.runtime_reach_full_activation_gate,
                0.0,
                1.0,
            )
            gmt_active = visible & self._option_active & (gmt_gate > 1.0e-4)
            root_quaternion = self.qpos[:, 3:7]
            qw, qx, qy, qz = root_quaternion.unbind(dim=1)
            yaw = torch.atan2(
                2.0 * (qw * qz + qx * qy),
                1.0 - 2.0 * (qy.square() + qz.square()),
            )
            heading = torch.stack(
                (
                    torch.cos(0.5 * yaw),
                    torch.zeros_like(yaw),
                    torch.zeros_like(yaw),
                    torch.sin(0.5 * yaw),
                ),
                dim=1,
            )
            gmt_target, _ = self._mosaic_gmt_controller.target(
                canonical_joint_position=self.qpos[:, 7:36],
                canonical_joint_velocity=self.qvel[:, 6:35],
                torso_quaternion_wxyz=self.xquat[:, self._torso_body],
                base_angular_velocity_body_rad_s=self.qvel[:, 3:6],
                heading_quaternion_wxyz=heading,
                relative_time_sec=-time_to_arrival,
                active=gmt_active,
            )
            joint_gmt_gate = gmt_gate.unsqueeze(1) * self._mosaic_gmt_joint_scale
            commanded += joint_gmt_gate * (gmt_target - commanded)
            gain_gate = gmt_gate.unsqueeze(1)
            self._step_kp = self._kp + gain_gate * (self._mosaic_gmt_kp - self._kp)
            self._step_kd = self._kd + gain_gate * (self._mosaic_gmt_kd - self._kd)
            self._maximum_applied_mosaic_gmt_blend.copy_(
                torch.maximum(self._maximum_applied_mosaic_gmt_blend, gmt_gate)
            )
        lunge_height = torch.clamp((0.65 - target[:, 2]) / 0.25, 0.0, 1.0)
        lunge_height = lunge_height.square() * (3.0 - 2.0 * lunge_height)
        arrival_delta = self._estimated_arrival_time_sec - current_time
        lunge_progress = torch.clamp(
            (self.dive_config.runtime_lateral_lunge_approach_horizon_sec - arrival_delta)
            / (self.dive_config.runtime_lateral_lunge_approach_horizon_sec - 0.15),
            0.0,
            1.0,
        )
        lunge_progress = lunge_progress.square() * (3.0 - 2.0 * lunge_progress)
        lunge_release = torch.clamp((arrival_delta + 0.18) / 0.18, 0.0, 1.0)
        lunge_gate = (
            self.dive_config.runtime_lateral_lunge_blend
            * lunge_height
            * lunge_progress
            * lunge_release
            * gate
        )
        lunge_gate = torch.where(
            visible & self._option_active,
            lunge_gate,
            torch.zeros_like(lunge_gate),
        )
        lunge_direction = torch.sign(lateral_error)
        hip_roll_delta = (
            lunge_gate * lunge_direction * self.dive_config.runtime_lateral_lunge_hip_roll_rad
        )
        ankle_roll_delta = (
            -lunge_gate * lunge_direction * self.dive_config.runtime_lateral_lunge_ankle_roll_rad
        )
        commanded[:, 1] += hip_roll_delta
        commanded[:, 7] += hip_roll_delta
        commanded[:, 5] += ankle_roll_delta
        commanded[:, 11] += ankle_roll_delta
        self._maximum_applied_lateral_lunge_blend.copy_(
            torch.maximum(self._maximum_applied_lateral_lunge_blend, lunge_gate)
        )
        if self._mosaic_gmt_getup_controller is not None:
            cfg = self.dive_config
            if self._mosaic_gmt_getup_skill is None:
                raise RuntimeError("active MOSAIC get-up controller has no bound skill")

            recovery_age = self._post_save_recovery_age_steps
            can_start = (
                self.task.first_save
                & (recovery_age >= 0)
                & (self._getup_start_recovery_age_steps < 0)
                & (self.qpos[:, 2] <= cfg.mosaic_gmt_getup_activation_maximum_pelvis_height_m)
                & (self.task.phase != 7)
            )
            self._getup_start_recovery_age_steps.copy_(
                torch.where(
                    can_start,
                    recovery_age,
                    self._getup_start_recovery_age_steps,
                )
            )
            getup_age = recovery_age - self._getup_start_recovery_age_steps
            getup_time = getup_age.to(torch.float32) * self.config.control_dt_sec
            duration = float(self._mosaic_gmt_getup_skill.duration_sec)
            active = (
                (self._getup_start_recovery_age_steps >= 0)
                & (getup_time >= 0.0)
                & (getup_time <= duration)
                & (self.task.phase != 7)
            )

            def getup_smoothstep(value: Any) -> Any:
                clipped = torch.clamp(value, 0.0, 1.0)
                return clipped.square() * (3.0 - 2.0 * clipped)

            rise = getup_smoothstep(getup_time / cfg.mosaic_gmt_getup_blend_in_sec)
            decay = getup_smoothstep((duration - getup_time) / cfg.mosaic_gmt_getup_blend_in_sec)
            getup_gate = cfg.mosaic_gmt_getup_blend * torch.minimum(rise, decay)
            getup_gate = torch.where(active, getup_gate, torch.zeros_like(getup_gate))
            root_quaternion = self.qpos[:, 3:7]
            qw, qx, qy, qz = root_quaternion.unbind(dim=1)
            yaw = torch.atan2(
                2.0 * (qw * qz + qx * qy),
                1.0 - 2.0 * (qy.square() + qz.square()),
            )
            heading = torch.stack(
                (
                    torch.cos(0.5 * yaw),
                    torch.zeros_like(yaw),
                    torch.zeros_like(yaw),
                    torch.sin(0.5 * yaw),
                ),
                dim=1,
            )
            getup_target, _ = self._mosaic_gmt_getup_controller.target(
                canonical_joint_position=self.qpos[:, 7:36],
                canonical_joint_velocity=self.qvel[:, 6:35],
                torso_quaternion_wxyz=self.xquat[:, self._torso_body],
                base_angular_velocity_body_rad_s=self.qvel[:, 3:6],
                heading_quaternion_wxyz=heading,
                relative_time_sec=getup_time,
                active=active,
            )
            getup_reference = self._mosaic_gmt_getup_controller.reference_target(getup_time)
            feedforward = cfg.mosaic_gmt_getup_reference_feedforward_blend
            getup_target += feedforward * (getup_reference - getup_target)
            joint_getup_gate = getup_gate.unsqueeze(
                1
            ) * self._mosaic_gmt_getup_joint_scale.unsqueeze(0)
            commanded += joint_getup_gate * (getup_target - commanded)
            gain_gate = getup_gate.unsqueeze(1)
            self._step_kp = self._step_kp + gain_gate * (self._mosaic_gmt_kp - self._step_kp)
            self._step_kd = self._step_kd + gain_gate * (self._mosaic_gmt_kd - self._step_kd)
            self._maximum_applied_mosaic_gmt_getup_blend.copy_(
                torch.maximum(self._maximum_applied_mosaic_gmt_getup_blend, getup_gate)
            )
            self._mosaic_gmt_getup_activation_steps += active.to(torch.long)
        lower_body_delta = commanded[:, :12] - stable_target[:, :12]
        lower_body_step = torch.clamp(
            lower_body_delta - self._previous_option_lower_body_delta,
            -self.dive_config.maximum_lower_body_target_step_rad,
            self.dive_config.maximum_lower_body_target_step_rad,
        )
        filtered_lower_body_delta = self._previous_option_lower_body_delta + (
            self.dive_config.lower_body_target_filter_fraction * lower_body_step
        )
        applied_lower_body_step = torch.max(
            torch.abs(filtered_lower_body_delta - self._previous_option_lower_body_delta),
            dim=1,
        ).values
        self._maximum_applied_lower_body_target_step.copy_(
            torch.maximum(
                self._maximum_applied_lower_body_target_step,
                applied_lower_body_step,
            )
        )
        commanded[:, :12] = stable_target[:, :12] + filtered_lower_body_delta
        self._previous_option_lower_body_delta.copy_(filtered_lower_body_delta)
        waist_delta = torch.clamp(
            commanded[:, 12:15] - self._previous_option_waist_target,
            -self.dive_config.maximum_waist_target_step_rad,
            self.dive_config.maximum_waist_target_step_rad,
        )
        filtered_waist = self._previous_option_waist_target + (
            self.dive_config.waist_target_filter_fraction * waist_delta
        )
        applied_waist_step = torch.max(
            torch.abs(filtered_waist - self._previous_option_waist_target),
            dim=1,
        ).values
        self._maximum_applied_waist_target_step.copy_(
            torch.maximum(self._maximum_applied_waist_target_step, applied_waist_step)
        )
        commanded[:, 12:15] = filtered_waist
        self._previous_option_waist_target.copy_(filtered_waist)
        arm_delta = torch.clamp(
            commanded[:, 15:29] - self._previous_option_arm_target,
            -self.dive_config.maximum_arm_target_step_rad,
            self.dive_config.maximum_arm_target_step_rad,
        )
        filtered_arm = self._previous_option_arm_target + (
            self.dive_config.arm_target_filter_fraction * arm_delta
        )
        commanded[:, 15:29] = filtered_arm
        self._previous_option_arm_target.copy_(filtered_arm)
        return commanded

    def _official_goalkeeper_target(self, stable_target: Any) -> Any:
        """Run the content-bound official 10 x 96 G1 teacher in MuJoCo."""

        torch = self.torch
        if (
            self._official_goalkeeper_teacher is None
            or self._official_goalkeeper_history is None
            or self._official_goalkeeper_action is None
            or self._official_goalkeeper_ball_last_local is None
            or self._official_goalkeeper_kp is None
            or self._official_goalkeeper_kd is None
            or self._official_goalkeeper_default is None
            or self._official_goalkeeper_initial is None
        ):
            raise RuntimeError("official goalkeeper teacher state is unavailable")
        from rosclaw_soccer.training.goalkeeper_combat_teacher import rotate_inverse_torch

        quaternion = self.qpos[:, 3:7]
        torso = self.xpos[:, self._torso_body]
        ball_local = rotate_inverse_torch(
            torch=torch,
            quaternion_wxyz=quaternion,
            vector=self.qpos[:, 36:39] - torso,
        )
        active = self._shot_index == 1
        approaching = (ball_local[:, 0] < self._official_goalkeeper_ball_last_local[:, 0]) | (
            self._official_goalkeeper_ball_last_local[:, 0] == 0.0
        )
        visible = (
            active
            & approaching
            & (ball_local[:, 0] > 0.05)
            & (ball_local[:, 0] < 3.40)
            & (torch.abs(ball_local[:, 1]) < 2.0)
            & (ball_local[:, 2] < 1.80)
        )
        self._official_goalkeeper_ball_last_local.copy_(ball_local)
        ball_observation = torch.where(
            visible.unsqueeze(1), ball_local, torch.zeros_like(ball_local)
        )
        angular_velocity_local = rotate_inverse_torch(
            torch=torch,
            quaternion_wxyz=quaternion,
            vector=self.qvel[:, 3:6],
        )
        gravity = torch.zeros((self.count, 3), device=self.device)
        gravity[:, 2] = -1.0
        projected_gravity = rotate_inverse_torch(
            torch=torch,
            quaternion_wxyz=quaternion,
            vector=gravity,
        )
        default = self._official_goalkeeper_default
        current = torch.cat(
            (
                ball_observation,
                0.25 * angular_velocity_local,
                projected_gravity,
                self.qpos[:, 7:36] - default,
                0.05 * self.qvel[:, 6:35],
                self._official_goalkeeper_action,
            ),
            dim=1,
        )
        self._official_goalkeeper_history.copy_(
            torch.cat((self._official_goalkeeper_history[:, 1:], current.unsqueeze(1)), dim=1)
        )
        with torch.inference_mode():
            self._official_goalkeeper_action.copy_(
                torch.clamp(
                    self._official_goalkeeper_teacher(self._official_goalkeeper_history.flatten(1)),
                    -4.0,
                    4.0,
                )
            )
        self._step_kp = self._official_goalkeeper_kp
        self._step_kd = self._official_goalkeeper_kd
        teacher_target = default.unsqueeze(0) + 0.25 * self._official_goalkeeper_action
        release_step = int(round(self.config.first_shot_release_sec / self.config.control_dt_sec))
        if self._step_index < release_step:
            teacher_target = self._official_goalkeeper_initial.unsqueeze(0).expand_as(
                teacher_target
            )
        blend = self.dive_config.official_goalkeeper_teacher_blend
        raw_delta = blend * (teacher_target - stable_target)
        filtered_groups: list[Any] = []
        filter_contracts = (
            (
                0,
                12,
                self.dive_config.official_goalkeeper_lower_body_target_step_rad,
                self.dive_config.official_goalkeeper_lower_body_filter_fraction,
            ),
            (
                12,
                15,
                self.dive_config.official_goalkeeper_waist_target_step_rad,
                self.dive_config.official_goalkeeper_waist_filter_fraction,
            ),
            (
                15,
                29,
                self.dive_config.official_goalkeeper_arm_target_step_rad,
                self.dive_config.official_goalkeeper_arm_filter_fraction,
            ),
        )
        for group_index, (start, end, step_limit, filter_fraction) in enumerate(filter_contracts):
            previous = self._previous_official_goalkeeper_target_delta[:, start:end]
            delta_step = torch.clamp(
                raw_delta[:, start:end] - previous,
                -step_limit,
                step_limit,
            )
            filtered = previous + filter_fraction * delta_step
            applied_step = torch.max(torch.abs(filtered - previous), dim=1).values
            self._maximum_official_goalkeeper_target_step[:, group_index].copy_(
                torch.maximum(
                    self._maximum_official_goalkeeper_target_step[:, group_index],
                    applied_step,
                )
            )
            filtered_groups.append(filtered)
        filtered_delta = torch.cat(filtered_groups, dim=1)
        self._previous_official_goalkeeper_target_delta.copy_(filtered_delta)
        return stable_target + filtered_delta

    def _canonical_locomotion_observation(self, lateral_drive: Any) -> Any:
        """Reflect physical state into the right-dive locomotion half-space."""

        torch = self.torch
        qw, qx, qy, qz = (self.qpos[:, index] for index in range(3, 7))
        gravity = torch.stack(
            (
                2.0 * (-qz * qx + qw * qy),
                -2.0 * (qz * qy + qw * qx),
                1.0 - 2.0 * (qw * qw + qz * qz),
            ),
            dim=1,
        )
        observation = torch.zeros((self.count, 96), device=self.device)
        observation[:, :3] = self.qvel[:, 3:6]
        observation[:, (0, 2)] *= -1.0
        observation[:, 3:6] = gravity
        observation[:, 4] *= -1.0
        observation[:, 7] = -lateral_drive * self.config.maximum_lateral_command_mps
        observation[:, 9:38] = (self.qpos[:, 7:36][:, self._loco_to_motor] - self._loco_default)[
            :, self._loco_mirror_order
        ] * self._loco_mirror_sign
        observation[:, 38:67] = (
            self.qvel[:, 6:35][:, self._loco_to_motor][:, self._loco_mirror_order]
            * self._loco_mirror_sign
        )
        observation[:, 67:96] = self._canonical_loco_action
        return observation

    def _shape_actor_action(self, requested_action: Any) -> tuple[Any, Any]:
        torch = self.torch
        raw_drive_gate = torch.clamp(requested_action[:, 0], min=0.0, max=1.0)
        drive_gate_delta = torch.clamp(
            raw_drive_gate - self._learned_lateral_drive_gate,
            -self.config.maximum_action_step,
            self.config.maximum_action_step,
        )
        self._learned_lateral_drive_gate.add_(self.config.action_filter_fraction * drive_gate_delta)
        self._maximum_learned_lateral_drive_gate.copy_(
            torch.maximum(
                self._maximum_learned_lateral_drive_gate,
                self._learned_lateral_drive_gate,
            )
        )
        angular_speed = torch.linalg.vector_norm(self.qvel[:, 3:6], dim=1)
        onset = self.config.agility.angular_guard_onset_rad_s
        ceiling = self.dive_config.dive_maximum_root_angular_speed_rad_s
        authority = torch.clamp(
            (ceiling - angular_speed) / max(ceiling - onset, 1.0e-6),
            min=0.0,
            max=1.0,
        )
        plasticity_steps = int(
            round(self.dive_config.actor_plasticity_duration_sec / self.config.control_dt_sec)
        )
        plastic = self._option_started & (self._option_age_steps <= plasticity_steps)
        self._actor_plasticity_active.copy_(plastic)
        shaped = requested_action.clone()
        shaped[:, 0] = torch.where(
            plastic,
            torch.maximum(
                torch.clamp(shaped[:, 0], min=0.0, max=1.0),
                torch.full_like(shaped[:, 0], self.dive_config.minimum_option_gate),
            )
            * authority,
            torch.zeros_like(shaped[:, 0]),
        )
        shaped[:, 1:] *= authority.unsqueeze(1)
        shaped[:, 1:] *= shaped[:, :1]
        shaped[:, 1:] = torch.where(
            plastic.unsqueeze(1), shaped[:, 1:], torch.zeros_like(shaped[:, 1:])
        )
        recovery_steps = max(
            1,
            int(round(self.dive_config.actor_recovery_plasticity_sec / self.config.control_dt_sec)),
        )
        option_steps = int(
            round(
                (self.dive_config.option_duration_sec + self.dive_config.phase_hold_sec)
                / self.config.control_dt_sec
            )
        )
        recovery_phase = torch.clamp(
            (self._option_age_steps - option_steps).to(torch.float32) / recovery_steps,
            0.0,
            1.0,
        )
        recovery_ramp = recovery_phase.square() * (3.0 - 2.0 * recovery_phase)
        recovery_scale = self.dive_config.actor_recovery_residual_authority_scale * recovery_ramp
        recovery = plastic & ~self._option_active
        shaped[:, 1:] *= torch.where(
            recovery,
            recovery_scale,
            torch.ones_like(recovery_scale),
        ).unsqueeze(1)
        counter = _body_frame_waist_counter_rotation(
            torch=torch,
            root_angular_velocity_body_rad_s=self.qvel[:, 3:6],
        )
        counter *= (
            self.config.agility.counter_rotation_gain * (1.0 - authority) * plastic.to(shaped.dtype)
        ).unsqueeze(1)
        shaped[:, 13:16] += counter
        shaped = torch.clamp(shaped, -1.0, 1.0)
        self._maximum_applied_counter_rotation.copy_(
            torch.maximum(
                self._maximum_applied_counter_rotation,
                torch.linalg.vector_norm(counter, dim=1),
            )
        )
        return shaped, authority

    def _posture_exception_granted(self, upright_projection: Any) -> Any:
        torch = self.torch
        cfg = self.dive_config
        exception_steps = int(
            round(cfg.posture_exception_duration_sec / self.config.control_dt_sec)
        )
        release_step = int(round(self.config.first_shot_release_sec / self.config.control_dt_sec))
        cue_step = release_step - int(round(cfg.prediction_lead_sec / self.config.control_dt_sec))
        # A signed intent cue is allowed to initiate a controlled crouch before
        # release.  That preparation must be protected by the same strict
        # dynamic envelope, while its finite landing budget must still expire
        # relative to the causal shot-release event.  Starting the exception
        # only at release incorrectly labels a safe 2 mm anticipatory crouch as
        # a fall; ageing it from the cue consumes the recovery budget before
        # the ball arrives.  This interval does neither.
        dynamic_age_steps = self._step_index - release_step
        within_time = (
            self._option_started
            & (self._step_index >= cue_step)
            & (dynamic_age_steps <= exception_steps)
            & (self._maximum_applied_option_gate > 0.05)
        )
        if getattr(self, "_official_goalkeeper_teacher", None) is not None:
            within_time = (self._step_index >= release_step) & (
                dynamic_age_steps <= exception_steps
            )
        envelope = (
            (self.qpos[:, 2] >= cfg.dive_minimum_pelvis_height_m)
            & (upright_projection >= cfg.dive_minimum_upright_projection)
            & (
                torch.linalg.vector_norm(self.qvel[:, :3], dim=1)
                <= cfg.dive_maximum_root_linear_speed_mps
            )
            & (
                torch.linalg.vector_norm(self.qvel[:, 3:6], dim=1)
                <= cfg.dive_maximum_root_angular_speed_rad_s
            )
        )
        granted = within_time & envelope
        if cfg.post_save_fall_recovery_enabled:
            recovery_steps = max(
                1,
                int(round(cfg.post_save_fall_recovery_duration_sec / self.config.control_dt_sec)),
            )
            recovery_age = self._post_save_recovery_age_steps
            recovery_window = (
                self.task.first_save & (recovery_age >= 0) & (recovery_age < recovery_steps)
            )
            # This is deliberately not a general fall exception.  It can only
            # keep a finite, bounded world alive after a simulator-verified
            # save so the causal recovery actor gets a chance to stand up.
            # When the finite window expires, the ordinary posture gate
            # immediately fails any keeper that has not recovered.
            recovery_envelope = (
                (self.qpos[:, 2] >= cfg.post_save_fall_minimum_pelvis_height_m)
                & (upright_projection >= cfg.post_save_fall_minimum_upright_projection)
                & (
                    torch.linalg.vector_norm(self.qvel[:, :3], dim=1)
                    <= cfg.post_save_fall_maximum_root_linear_speed_mps
                )
                & (
                    torch.linalg.vector_norm(self.qvel[:, 3:6], dim=1)
                    <= cfg.post_save_fall_maximum_root_angular_speed_rad_s
                )
            )
            granted |= recovery_window & recovery_envelope
        self._maximum_posture_exception_steps += granted.to(torch.long)
        return granted

    def _substep_upper_body_position_authority(self) -> Any:
        """Brake position torque at 2 ms cadence before a dive destabilizes the root.

        Velocity damping remains active in the base integrator.  This hook only
        scales the upper-body position term, so the guard cannot add energy or
        silently expand the learned actor's command authority.
        """

        torch = self.torch
        cfg = self.dive_config
        if not cfg.substep_upper_body_guard_enabled:
            return super()._substep_upper_body_position_authority()
        angular_speed = torch.linalg.vector_norm(self.qvel[:, 3:6], dim=1)
        fraction = torch.clamp(
            (angular_speed - cfg.substep_upper_body_guard_onset_rad_s)
            / (
                cfg.substep_upper_body_guard_ceiling_rad_s
                - cfg.substep_upper_body_guard_onset_rad_s
            ),
            0.0,
            1.0,
        )
        authority = 1.0 - fraction * (1.0 - cfg.substep_upper_body_minimum_position_scale)
        self._minimum_substep_arm_authority.copy_(
            torch.minimum(self._minimum_substep_arm_authority, authority)
        )
        # Keep all three waist position terms active: the waist is the trunk
        # stabilizer, whereas the fourteen arm terms create the reach impulse.
        return torch.cat(
            (
                torch.ones((self.count, 3), device=self.device),
                authority.unsqueeze(1).expand(-1, 14),
            ),
            dim=1,
        )

    def _substep_position_target(self, target: Any) -> Any:
        """Shed only learned lower-body offset while preserving the locomotion prior."""

        cfg = self.dive_config
        if not cfg.substep_option_lower_body_guard_enabled:
            return super()._substep_position_target(target)
        torch = self.torch
        angular_speed = torch.linalg.vector_norm(self.qvel[:, 3:6], dim=1)
        fraction = torch.clamp(
            (angular_speed - cfg.substep_option_lower_body_guard_onset_rad_s)
            / (
                cfg.substep_option_lower_body_guard_ceiling_rad_s
                - cfg.substep_option_lower_body_guard_onset_rad_s
            ),
            0.0,
            1.0,
        )
        authority = 1.0 - fraction * (1.0 - cfg.substep_option_lower_body_minimum_scale)
        self._minimum_substep_option_lower_body_authority.copy_(
            torch.minimum(self._minimum_substep_option_lower_body_authority, authority)
        )
        guarded = target.clone()
        guarded[:, :12] = self._substep_stable_lower_body_target + authority.unsqueeze(1) * (
            target[:, :12] - self._substep_stable_lower_body_target
        )
        return guarded

    def summary(self) -> dict[str, Any]:
        report = super().summary()
        report.update(
            {
                "schema_version": "rosclaw_soccer.goalkeeper_targeted_dive_mjwarp_summary.v14",
                "targeted_dive_rl_config": asdict(self.dive_config),
                "targeted_dive_rl_config_hash": self.dive_config.config_hash,
                "targeted_dive_checkpoint": self.targeted_dive_checkpoint_path.name,
                "targeted_dive_checkpoint_hash": hash_bytes(
                    self.targeted_dive_checkpoint_path.read_bytes()
                ),
                "targeted_dive_checkpoint_authority": "IMITATION_ANCHOR_ONLY",
                "action_size": self.action_size,
                "posture_exception_owner": "ENVIRONMENT",
                "posture_exception_clock": "INTENT_CUE_START_CAUSAL_RELEASE_EXPIRY",
                "maximum_applied_lower_body_target_step_rad": float(
                    self.torch.max(self._maximum_applied_lower_body_target_step).item()
                ),
                "mean_maximum_applied_lower_body_target_step_rad": float(
                    self.torch.mean(self._maximum_applied_lower_body_target_step).item()
                ),
                "maximum_applied_waist_target_step_rad": float(
                    self.torch.max(self._maximum_applied_waist_target_step).item()
                ),
                "mean_maximum_applied_waist_target_step_rad": float(
                    self.torch.mean(self._maximum_applied_waist_target_step).item()
                ),
                "reach_arrival_clock": "CAUSAL_RELEASE_LATCHED_ABSOLUTE_TIME",
                "actor_residual_plasticity_duration_sec": (
                    self.dive_config.actor_plasticity_duration_sec
                ),
                "zero_action_semantics": "FROZEN_STABLE_LOCOMOTION_PRIOR",
                "official_goalkeeper_teacher": (
                    None
                    if self._official_goalkeeper_teacher_contract is None
                    else dict(self._official_goalkeeper_teacher_contract)
                ),
                "maximum_official_goalkeeper_target_step_rad": {
                    "lower_body": float(self._maximum_official_goalkeeper_target_step[:, 0].max()),
                    "waist": float(self._maximum_official_goalkeeper_target_step[:, 1].max()),
                    "arms": float(self._maximum_official_goalkeeper_target_step[:, 2].max()),
                },
                "maximum_applied_option_gate": float(self._maximum_applied_option_gate.max()),
                "maximum_runtime_reach_feedback_correction_rad": float(
                    self._maximum_runtime_reach_feedback_correction.max()
                ),
                "maximum_applied_lateral_drive": float(self._maximum_applied_lateral_drive.max()),
                "maximum_learned_lateral_drive_gate": float(
                    self._maximum_learned_lateral_drive_gate.max()
                ),
                "mean_maximum_learned_lateral_drive_gate": float(
                    self._maximum_learned_lateral_drive_gate.mean()
                ),
                "maximum_applied_post_save_counterstep": float(
                    self._maximum_applied_post_save_counterstep.max()
                ),
                "maximum_applied_lateral_lunge_blend": float(
                    self._maximum_applied_lateral_lunge_blend.max()
                ),
                "minimum_substep_arm_position_authority": float(
                    self._minimum_substep_arm_authority.min()
                ),
                "minimum_substep_option_lower_body_authority": float(
                    self._minimum_substep_option_lower_body_authority.min()
                ),
                "canonical_locomotion_mirror_fraction": float(
                    self._canonical_locomotion_mirror_steps.to(self.torch.float32).sum()
                    / max(1, self.count * self._step_index)
                ),
                "contact_support_side_enabled": (
                    self.dive_config.runtime_contact_support_side_enabled
                ),
                "actor_contact_support_side_enabled": (
                    self.dive_config.actor_contact_support_side_enabled
                ),
                "actor_foot_contact_observation": (
                    "SUPPORT_SIDE_LEFT_CONTACT_RIGHT_CONTACT"
                    if self.dive_config.actor_contact_support_side_enabled
                    else None
                ),
                "contact_single_support_fraction": float(
                    self._contact_support_nonzero_steps.to(self.torch.float32).sum()
                    / max(1, self.count * self._step_index)
                ),
                "maximum_applied_counter_rotation_action": float(
                    self._maximum_applied_counter_rotation.max()
                ),
                "maximum_decoder_lower_body_residual_rad": float(
                    self._maximum_decoder_lower_body_residual_rad.max()
                ),
                "mean_maximum_decoder_lower_body_residual_rad": float(
                    self._maximum_decoder_lower_body_residual_rad.mean()
                ),
                "maximum_applied_decoder_lower_body_correction_rad": float(
                    self._maximum_applied_decoder_lower_body_correction_rad.max()
                ),
                "mean_maximum_applied_decoder_lower_body_correction_rad": float(
                    self._maximum_applied_decoder_lower_body_correction_rad.mean()
                ),
                "runtime_task_space_reach": (
                    None
                    if self._runtime_reach_atlas is None
                    else {
                        "atlas_hash": self._runtime_reach_atlas.model_hash,
                        "maximum_applied_blend": float(
                            self._maximum_applied_runtime_reach_blend.max()
                        ),
                        "authority": "ENVIRONMENT_OWNED_TARGET_CONDITIONED_ARM_TEACHER",
                    }
                ),
                "runtime_whole_body_reach": (
                    None
                    if self._whole_body_reach_atlas is None
                    else {
                        "atlas_hash": self._whole_body_reach_atlas.model_hash,
                        "maximum_applied_blend": float(
                            self._maximum_applied_whole_body_reach_blend.max()
                        ),
                        "authority": ("HEIGHT_ROUTED_WAIST_ARM_KINEMATIC_TEACHER_SIM_ONLY"),
                    }
                ),
                "mosaic_overhead_reach": (
                    None
                    if self._overhead_reach_prior is None
                    else {
                        "prior_hash": self._overhead_reach_prior.prior_hash,
                        "source_hash": self._overhead_reach_prior.source_hash,
                        "semantic_contract_hash": (
                            self._overhead_reach_prior.semantic_contract_hash
                        ),
                        "maximum_applied_blend": float(
                            self._maximum_applied_overhead_reach_blend.max()
                        ),
                        "authority": ("ENVIRONMENT_OWNED_HEIGHT_CONDITIONED_PD_TARGET_PRIOR"),
                    }
                ),
                "mosaic_gmt_cerebellum": (
                    None
                    if self._mosaic_gmt_contract is None or self._mosaic_gmt_skill is None
                    else {
                        "checkpoint_hash": self._mosaic_gmt_contract.checkpoint_hash,
                        "checkpoint_contract_hash": (self._mosaic_gmt_contract.contract_hash),
                        "skill_hash": self._mosaic_gmt_skill.skill_hash,
                        "source_hash": self._mosaic_gmt_skill.source_hash,
                        "semantic_contract_hash": (self._mosaic_gmt_skill.semantic_contract_hash),
                        "maximum_applied_blend": float(
                            self._maximum_applied_mosaic_gmt_blend.max()
                        ),
                        "authority": ("FROZEN_CLOSED_LOOP_WHOLE_BODY_TRACKER_SIM_ONLY"),
                    }
                ),
                "mosaic_gmt_getup_cerebellum": (
                    None
                    if self._mosaic_gmt_contract is None or self._mosaic_gmt_getup_skill is None
                    else {
                        "checkpoint_hash": self._mosaic_gmt_contract.checkpoint_hash,
                        "checkpoint_contract_hash": self._mosaic_gmt_contract.contract_hash,
                        "skill_hash": self._mosaic_gmt_getup_skill.skill_hash,
                        "source_hash": self._mosaic_gmt_getup_skill.source_hash,
                        "body_hash": self._mosaic_gmt_getup_skill.body_hash,
                        "physics_scene_hash": (self._mosaic_gmt_getup_skill.physics_scene_hash),
                        "semantic_contract_hash": (
                            self._mosaic_gmt_getup_skill.semantic_contract_hash
                        ),
                        "source_duration_sec": self._mosaic_gmt_getup_skill.duration_sec,
                        "maximum_applied_blend": float(
                            self._maximum_applied_mosaic_gmt_getup_blend.max()
                        ),
                        "activation_step_fraction": float(
                            self._mosaic_gmt_getup_activation_steps.to(self.torch.float32).sum()
                            / max(1, self.count * self._step_index)
                        ),
                        "authority": ("POST_SAVE_CAUSAL_FROZEN_CLOSED_LOOP_GETUP_TRACKER_SIM_ONLY"),
                    }
                ),
                "cerebellar_skill_composition": (
                    None
                    if self._mosaic_gmt_controller is None
                    or (self._runtime_reach_atlas is None and self._whole_body_reach_atlas is None)
                    else {
                        "lower_body_owner": "MOSAIC_GMT_CLOSED_LOOP_TRACKER",
                        "waist_and_arm_owner": (
                            "HEIGHT_ROUTED_WHOLE_BODY_REACH_ATLAS"
                            if self._whole_body_reach_atlas is not None
                            else "GMT_WAIST_PLUS_CAUSAL_TASK_SPACE_ARM_ATLAS"
                        ),
                        "overlapping_joint_authority": False,
                        "learned_residual_owner": "BOUNDED_ONLINE_ACTOR",
                    }
                ),
                "activation_ceiling": "SIM_ONLY",
                "hardware_command_sent": False,
            }
        )
        return report


__all__ = [
    "TARGETED_DIVE_RL_RESIDUAL_LIMITS_RAD",
    "GoalkeeperTargetedDiveMJWarpBatch",
    "GoalkeeperTargetedDiveRLConfig",
]
