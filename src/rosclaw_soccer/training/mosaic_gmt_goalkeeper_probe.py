"""Bounded GPU-physics probe for the MOSAIC GMT goalkeeper foundation."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from rosclaw_soccer.sim.contracts import (
    hash_bytes,
    hash_json,
)
from rosclaw_soccer.sim.contracts import (
    sanitize_nonfinite_evidence as _sanitize_nonfinite_evidence,
)
from rosclaw_soccer.training.goalkeeper_mjwarp import goalkeeper_world_config
from rosclaw_soccer.training.goalkeeper_targeted_dive_mjwarp import (
    GoalkeeperTargetedDiveMJWarpBatch,
    GoalkeeperTargetedDiveRLConfig,
)


def _finite_float_or_none(value: Any) -> float | None:
    """Serialize optional diagnostic scalars without weakening strict hashes."""

    scalar = float(value)
    return scalar if math.isfinite(scalar) else None


def _actor_training_exam_contract(
    training_config: Any,
    probe_config: Any,
) -> dict[str, Any]:
    """Audit whether a checkpoint was trained in the strict probe world.

    The probe intentionally accepts out-of-distribution actors for diagnosis,
    but it must make that fact explicit.  Candidate training once used the
    ``standard`` world while this examiner used ``match``; seed statistics
    then looked like learning even though the physics envelopes differed.
    """

    expected = {
        "shot_difficulty_profile": "match",
        "training_first_shot_release_sec": probe_config.first_shot_release_sec,
        "training_hard_shot_fraction": 1.0,
        "training_hard_shot_height_mode": probe_config.hard_shot_height_mode,
        "training_hard_shot_side_mode": "balanced",
        "training_hard_shot_flight_time_range_sec": (0.40, 0.50),
        "targeted_dive_nominal_shot_flight_time_sec": 0.45,
        "targeted_dive_runtime_contact_support_side_enabled": (
            probe_config.contact_support_side_enabled
        ),
        "targeted_dive_actor_contact_support_side_enabled": (
            probe_config.actor_contact_support_side_enabled
        ),
        "targeted_dive_actor_recovery_context_enabled": (
            probe_config.actor_recovery_context_enabled
        ),
        "targeted_dive_canonical_locomotion_mirror_enabled": (
            probe_config.canonical_locomotion_mirror_enabled
        ),
        "targeted_dive_post_save_counterstep_recenter_weight": (
            probe_config.post_save_counterstep_recenter_weight
        ),
    }
    if not isinstance(training_config, dict):
        return {
            "status": "UNKNOWN_MISSING_TRAINING_CONFIG",
            "expected": expected,
            "actual": None,
            "mismatched_fields": sorted(expected),
        }
    boolean_defaults = {
        "targeted_dive_actor_recovery_context_enabled",
        "targeted_dive_canonical_locomotion_mirror_enabled",
    }
    actual = {
        key: (
            training_config.get(key, "balanced")
            if key == "training_hard_shot_side_mode"
            else training_config.get(key, False)
            if key in boolean_defaults
            else training_config.get(key)
        )
        for key in expected
    }
    mismatch = []
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if isinstance(expected_value, tuple) and isinstance(actual_value, list | tuple):
            actual_value = tuple(actual_value)
        if actual_value != expected_value:
            mismatch.append(key)
    return {
        "status": "MATCHED" if not mismatch else "OUT_OF_DISTRIBUTION",
        "expected": expected,
        "actual": actual,
        "mismatched_fields": mismatch,
    }


@dataclass(frozen=True)
class MosaicGMTGoalkeeperProbeConfig:
    environment_count: int = 32
    random_seed: int = 451
    option_gate: float = 1.0
    minimum_option_gate: float = 0.0
    actor_residual_scale: float = 0.70
    lateral_drive_scale: float = 0.0
    negative_target_lateral_drive_scale: float = 1.0
    lateral_drive_full_activation_gate: float = 0.30
    lateral_drive_capture_enabled: bool = False
    lateral_drive_capture_horizon_sec: float = 0.35
    lateral_drive_target_standoff_m: float = 0.32
    lateral_drive_capture_scale_m: float = 0.45
    lateral_drive_learned_gate_enabled: bool = False
    lateral_lunge_blend: float = 0.0
    lateral_lunge_hip_roll_rad: float = 0.18
    lateral_lunge_ankle_roll_rad: float = 0.12
    lateral_lunge_approach_horizon_sec: float = 0.90
    substep_upper_body_guard_enabled: bool = False
    substep_upper_body_guard_onset_rad_s: float = 1.80
    substep_upper_body_guard_ceiling_rad_s: float = 3.00
    substep_upper_body_minimum_position_scale: float = 0.05
    substep_option_lower_body_guard_enabled: bool = False
    substep_option_lower_body_guard_onset_rad_s: float = 2.40
    substep_option_lower_body_guard_ceiling_rad_s: float = 3.30
    substep_option_lower_body_minimum_scale: float = 0.0
    canonical_locomotion_mirror_enabled: bool = False
    official_goalkeeper_teacher_checkpoint: str | None = None
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
    anchor_lower_body_scale: float = 0.25
    anchor_waist_scale: float = 0.50
    decoder_lower_body_residual_authority: float = 0.10
    decoder_lower_body_command_scale: float | None = None
    decoder_waist_residual_authority: float = 0.10
    gmt_blend: float = 1.0
    gmt_stability_floor: float = 0.0
    gmt_lower_body_scale: float = 1.0
    gmt_waist_scale: float = 1.0
    gmt_arm_scale: float = 1.0
    task_space_reach_blend: float = 0.0
    task_space_reach_feedback_blend: float = 0.0
    task_space_reach_feedback_gain: float = 0.70
    task_space_reach_feedback_maximum_error_m: float = 0.30
    task_space_reach_feedback_support_scale: float = 0.0
    contact_support_side_enabled: bool = False
    actor_contact_support_side_enabled: bool = False
    actor_recovery_context_enabled: bool = False
    whole_body_reach_blend: float = 0.0
    whole_body_reach_waist_scale: float = 0.75
    whole_body_reach_arm_scale: float = 1.0
    whole_body_reach_support_scale: float = 0.65
    whole_body_reach_release_sec: float = 0.60
    task_space_contact_standoff_m: float = 0.0
    task_space_lateral_lead_m: float = 0.0
    task_space_vertical_lead_m: float = 0.0
    task_space_low_vertical_lead_m: float | None = None
    task_space_mid_vertical_lead_m: float | None = None
    task_space_high_vertical_lead_m: float | None = None
    first_shot_release_sec: float = 0.70
    prediction_lead_sec: float = 0.30
    option_duration_sec: float = 0.95
    actor_recovery_plasticity_sec: float = 0.0
    actor_recovery_residual_authority_scale: float = 0.50
    post_save_counterstep_enabled: bool = False
    post_save_counterstep_duration_sec: float = 0.80
    post_save_counterstep_command_limit: float = 0.55
    post_save_counterstep_capture_horizon_sec: float = 0.28
    post_save_counterstep_recenter_weight: float = 1.0
    post_save_option_release_sec: float = 0.30
    reach_approach_horizon_sec: float = 0.55
    posture_exception_duration_sec: float = 1.55
    root_angular_speed_guard_ceiling_rad_s: float = 8.0
    dive_minimum_pelvis_height_m: float = 0.30
    dive_minimum_upright_projection: float = 0.05
    dive_maximum_root_linear_speed_mps: float = 3.0
    hard_shot_height_mode: str = "high"
    gmt_minimum_target_height_m: float = 1.10
    gmt_full_target_height_m: float = 1.25
    maximum_arm_target_step_rad: float = 0.20
    arm_target_filter_fraction: float = 1.0
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw.mosaic_gmt_goalkeeper_probe_config.v49"

    def __post_init__(self) -> None:
        height_leads = (
            self.task_space_low_vertical_lead_m,
            self.task_space_mid_vertical_lead_m,
            self.task_space_high_vertical_lead_m,
        )
        values = (
            self.option_gate,
            self.minimum_option_gate,
            self.actor_residual_scale,
            self.lateral_drive_scale,
            self.negative_target_lateral_drive_scale,
            self.lateral_drive_full_activation_gate,
            self.lateral_lunge_blend,
            self.anchor_lower_body_scale,
            self.anchor_waist_scale,
            self.decoder_lower_body_residual_authority,
            self.decoder_waist_residual_authority,
            self.gmt_blend,
            self.gmt_stability_floor,
            self.gmt_lower_body_scale,
            self.gmt_waist_scale,
            self.gmt_arm_scale,
            self.task_space_reach_blend,
            self.task_space_reach_feedback_blend,
            self.task_space_reach_feedback_support_scale,
            self.whole_body_reach_blend,
            self.whole_body_reach_waist_scale,
            self.whole_body_reach_arm_scale,
            self.whole_body_reach_support_scale,
            self.whole_body_reach_release_sec,
            self.task_space_contact_standoff_m,
            self.task_space_lateral_lead_m,
            self.task_space_vertical_lead_m,
            self.arm_target_filter_fraction,
        )
        if (
            not isinstance(self.substep_upper_body_guard_enabled, bool)
            or not isinstance(self.substep_option_lower_body_guard_enabled, bool)
            or not isinstance(self.canonical_locomotion_mirror_enabled, bool)
            or not isinstance(self.post_save_counterstep_enabled, bool)
            or not isinstance(self.lateral_drive_capture_enabled, bool)
            or not isinstance(self.lateral_drive_learned_gate_enabled, bool)
            or not isinstance(self.contact_support_side_enabled, bool)
            or not isinstance(self.actor_contact_support_side_enabled, bool)
            or not isinstance(self.actor_recovery_context_enabled, bool)
            or (self.actor_contact_support_side_enabled and not self.contact_support_side_enabled)
            or (self.actor_recovery_context_enabled and not self.actor_contact_support_side_enabled)
            or not (
                math.isfinite(self.lateral_drive_capture_horizon_sec)
                and 0.10 <= self.lateral_drive_capture_horizon_sec <= 0.80
                and math.isfinite(self.lateral_drive_target_standoff_m)
                and 0.10 <= self.lateral_drive_target_standoff_m <= 0.50
                and math.isfinite(self.lateral_drive_capture_scale_m)
                and 0.15 <= self.lateral_drive_capture_scale_m <= 0.80
            )
            or not (
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
            )
            or (
                self.official_goalkeeper_teacher_checkpoint is not None
                and self.canonical_locomotion_mirror_enabled
            )
            or not 0.05 <= self.official_goalkeeper_teacher_blend <= 1.0
            or any(
                not 0.01 <= step <= 0.20 or not 0.10 <= fraction <= 1.0
                for step, fraction in (
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
            )
            or not 1 <= self.environment_count <= 512
            or any(not 0.0 <= value <= 1.0 for value in values)
            or not 0.10 <= self.lateral_drive_full_activation_gate <= 1.0
            or self.task_space_reach_blend > 0.85
            or not 0.25 <= self.task_space_reach_feedback_gain <= 1.0
            or not 0.05 <= self.task_space_reach_feedback_maximum_error_m <= 0.50
            or (self.task_space_reach_feedback_blend > 0.0 and self.task_space_reach_blend <= 0.0)
            or not 0.05 <= self.anchor_lower_body_scale <= self.anchor_waist_scale <= 0.80
            or not 0.0 < self.decoder_lower_body_residual_authority <= 1.0
            or not 0.0 < self.decoder_waist_residual_authority <= 1.0
            or (
                self.decoder_lower_body_command_scale is not None
                and not self.anchor_lower_body_scale <= self.decoder_lower_body_command_scale <= 1.0
            )
            or self.minimum_option_gate > 0.80
            or self.task_space_contact_standoff_m > 0.22
            or self.task_space_lateral_lead_m > 0.30
            or self.task_space_vertical_lead_m > 0.40
            or (
                any(value is not None for value in height_leads)
                and not all(value is not None for value in height_leads)
            )
            or any(value is not None and (not -0.40 <= value <= 0.40) for value in height_leads)
            or not 0.10 <= self.prediction_lead_sec <= 1.00
            or not 0.70 <= self.first_shot_release_sec <= 1.40
            or self.prediction_lead_sec >= self.first_shot_release_sec
            or not 0.05 <= self.lateral_lunge_hip_roll_rad <= 0.30
            or not 0.03 <= self.lateral_lunge_ankle_roll_rad <= 0.25
            or not 0.60 <= self.lateral_lunge_approach_horizon_sec <= 1.20
            or not (
                0.50
                <= self.substep_upper_body_guard_onset_rad_s
                < self.substep_upper_body_guard_ceiling_rad_s
                <= self.root_angular_speed_guard_ceiling_rad_s
            )
            or not 0.0 <= self.substep_upper_body_minimum_position_scale <= 0.50
            or not (
                0.50
                <= self.substep_option_lower_body_guard_onset_rad_s
                < self.substep_option_lower_body_guard_ceiling_rad_s
                <= self.root_angular_speed_guard_ceiling_rad_s
            )
            or not 0.0 <= self.substep_option_lower_body_minimum_scale <= 0.50
            or any(
                not 0.50 <= value <= 1.50
                for value in (
                    self.low_shot_phase_scale,
                    self.mid_shot_phase_scale,
                    self.high_shot_phase_scale,
                )
            )
            or not 0.70 <= self.option_duration_sec <= 1.40
            or not 0.30 <= self.reach_approach_horizon_sec <= 0.80
            or not 0.95 <= self.posture_exception_duration_sec <= 2.20
            or not 3.0 <= self.root_angular_speed_guard_ceiling_rad_s <= 12.0
            or not 0.10 <= self.dive_minimum_pelvis_height_m <= 0.60
            or not 0.0 <= self.dive_minimum_upright_projection <= 0.40
            or not 1.0 <= self.dive_maximum_root_linear_speed_mps <= 6.0
            or self.hard_shot_height_mode not in {"low", "mid", "high", "balanced"}
            or not 0.30 <= self.gmt_minimum_target_height_m <= 1.30
            or not self.gmt_minimum_target_height_m < self.gmt_full_target_height_m <= 1.60
            or (
                (
                    self.task_space_contact_standoff_m > 0.0
                    or self.task_space_lateral_lead_m > 0.0
                    or self.task_space_vertical_lead_m > 0.0
                    or any(value is not None and value != 0.0 for value in height_leads)
                )
                and self.task_space_reach_blend <= 0.0
            )
            or (self.task_space_reach_blend > 0.0 and self.gmt_arm_scale > 0.0)
            or (self.gmt_stability_floor > 0.0 and self.gmt_arm_scale > 0.0)
            or (
                self.whole_body_reach_blend > 0.0
                and (
                    self.task_space_reach_blend > 0.0
                    or self.gmt_stability_floor > 0.0
                    or self.gmt_arm_scale > 0.0
                    or self.gmt_minimum_target_height_m < 0.65
                )
            )
            or not 0.02 <= self.maximum_arm_target_step_rad <= 0.20
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("MOSAIC GMT goalkeeper probe settings are invalid")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


def run_mosaic_gmt_goalkeeper_probe(
    *,
    asset_root: Path,
    locomotion_policy_path: Path,
    targeted_dive_checkpoint: Path,
    gmt_model_path: Path,
    gmt_skill_path: Path,
    device: Any,
    config: MosaicGMTGoalkeeperProbeConfig,
    actor_checkpoint_path: Path | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Run one hard-high, first-shot-only diagnostic with no learned residual."""

    import torch

    world = goalkeeper_world_config(
        difficulty_profile="match",
        environment_count=config.environment_count,
        second_shot_probability=0.0,
        shot_intent_cue_enabled=True,
        hard_shot_fraction=1.0,
        hard_shot_height_mode=config.hard_shot_height_mode,  # type: ignore[arg-type]
        hard_shot_flight_time_range_sec=(0.40, 0.50),
        hard_height_reach_reward_scale=1.0,
    )
    if config.first_shot_release_sec != world.first_shot_release_sec:
        world = replace(
            world,
            first_shot_release_sec=config.first_shot_release_sec,
            first_shot_end_sec=config.first_shot_release_sec + 1.0,
        )
    dive = GoalkeeperTargetedDiveRLConfig(
        option_duration_sec=config.option_duration_sec,
        actor_recovery_plasticity_sec=config.actor_recovery_plasticity_sec,
        actor_recovery_residual_authority_scale=(config.actor_recovery_residual_authority_scale),
        post_save_counterstep_enabled=config.post_save_counterstep_enabled,
        post_save_counterstep_duration_sec=config.post_save_counterstep_duration_sec,
        post_save_counterstep_command_limit=config.post_save_counterstep_command_limit,
        post_save_counterstep_capture_horizon_sec=(
            config.post_save_counterstep_capture_horizon_sec
        ),
        post_save_counterstep_recenter_weight=(config.post_save_counterstep_recenter_weight),
        post_save_option_release_sec=config.post_save_option_release_sec,
        prediction_lead_sec=config.prediction_lead_sec,
        nominal_shot_flight_time_sec=0.45,
        decoder_residual_authority=0.10,
        decoder_lower_body_residual_authority=(config.decoder_lower_body_residual_authority),
        decoder_lower_body_command_scale=config.decoder_lower_body_command_scale,
        decoder_waist_residual_authority=config.decoder_waist_residual_authority,
        actor_residual_scale=config.actor_residual_scale,
        posture_exception_duration_sec=config.posture_exception_duration_sec,
        dive_maximum_root_angular_speed_rad_s=(config.root_angular_speed_guard_ceiling_rad_s),
        dive_minimum_pelvis_height_m=config.dive_minimum_pelvis_height_m,
        dive_minimum_upright_projection=config.dive_minimum_upright_projection,
        dive_maximum_root_linear_speed_mps=(config.dive_maximum_root_linear_speed_mps),
        anchor_lower_body_scale=config.anchor_lower_body_scale,
        anchor_waist_scale=config.anchor_waist_scale,
        anchor_arm_scale=1.0,
        minimum_option_gate=config.minimum_option_gate,
        lateral_drive_scale=config.lateral_drive_scale,
        negative_target_lateral_drive_scale=(config.negative_target_lateral_drive_scale),
        lateral_drive_full_activation_gate=config.lateral_drive_full_activation_gate,
        lateral_drive_capture_enabled=config.lateral_drive_capture_enabled,
        lateral_drive_capture_horizon_sec=(config.lateral_drive_capture_horizon_sec),
        lateral_drive_target_standoff_m=(config.lateral_drive_target_standoff_m),
        lateral_drive_capture_scale_m=config.lateral_drive_capture_scale_m,
        lateral_drive_learned_gate_enabled=(config.lateral_drive_learned_gate_enabled),
        runtime_lateral_lunge_blend=config.lateral_lunge_blend,
        runtime_lateral_lunge_hip_roll_rad=config.lateral_lunge_hip_roll_rad,
        runtime_lateral_lunge_ankle_roll_rad=config.lateral_lunge_ankle_roll_rad,
        runtime_lateral_lunge_approach_horizon_sec=(config.lateral_lunge_approach_horizon_sec),
        substep_upper_body_guard_enabled=config.substep_upper_body_guard_enabled,
        substep_upper_body_guard_onset_rad_s=config.substep_upper_body_guard_onset_rad_s,
        substep_upper_body_guard_ceiling_rad_s=(config.substep_upper_body_guard_ceiling_rad_s),
        substep_upper_body_minimum_position_scale=(
            config.substep_upper_body_minimum_position_scale
        ),
        substep_option_lower_body_guard_enabled=(config.substep_option_lower_body_guard_enabled),
        substep_option_lower_body_guard_onset_rad_s=(
            config.substep_option_lower_body_guard_onset_rad_s
        ),
        substep_option_lower_body_guard_ceiling_rad_s=(
            config.substep_option_lower_body_guard_ceiling_rad_s
        ),
        substep_option_lower_body_minimum_scale=(config.substep_option_lower_body_minimum_scale),
        canonical_locomotion_mirror_enabled=(config.canonical_locomotion_mirror_enabled),
        official_goalkeeper_teacher_checkpoint_path=(config.official_goalkeeper_teacher_checkpoint),
        official_goalkeeper_teacher_blend=config.official_goalkeeper_teacher_blend,
        official_goalkeeper_lower_body_target_step_rad=(
            config.official_goalkeeper_lower_body_target_step_rad
        ),
        official_goalkeeper_lower_body_filter_fraction=(
            config.official_goalkeeper_lower_body_filter_fraction
        ),
        official_goalkeeper_waist_target_step_rad=(
            config.official_goalkeeper_waist_target_step_rad
        ),
        official_goalkeeper_waist_filter_fraction=(
            config.official_goalkeeper_waist_filter_fraction
        ),
        official_goalkeeper_arm_target_step_rad=(config.official_goalkeeper_arm_target_step_rad),
        official_goalkeeper_arm_filter_fraction=(config.official_goalkeeper_arm_filter_fraction),
        low_shot_phase_scale=config.low_shot_phase_scale,
        mid_shot_phase_scale=config.mid_shot_phase_scale,
        high_shot_phase_scale=config.high_shot_phase_scale,
        runtime_reach_blend=config.task_space_reach_blend,
        runtime_reach_feedback_blend=config.task_space_reach_feedback_blend,
        runtime_reach_feedback_gain=config.task_space_reach_feedback_gain,
        runtime_reach_feedback_maximum_error_m=(config.task_space_reach_feedback_maximum_error_m),
        runtime_reach_feedback_support_scale=(config.task_space_reach_feedback_support_scale),
        runtime_contact_support_side_enabled=config.contact_support_side_enabled,
        actor_contact_support_side_enabled=config.actor_contact_support_side_enabled,
        actor_recovery_context_enabled=config.actor_recovery_context_enabled,
        runtime_whole_body_reach_blend=config.whole_body_reach_blend,
        runtime_whole_body_reach_waist_scale=config.whole_body_reach_waist_scale,
        runtime_whole_body_reach_arm_scale=config.whole_body_reach_arm_scale,
        runtime_whole_body_reach_support_scale=config.whole_body_reach_support_scale,
        runtime_whole_body_reach_release_sec=config.whole_body_reach_release_sec,
        runtime_reach_approach_horizon_sec=config.reach_approach_horizon_sec,
        runtime_reach_contact_standoff_m=config.task_space_contact_standoff_m,
        runtime_reach_lateral_lead_m=config.task_space_lateral_lead_m,
        runtime_reach_vertical_lead_m=config.task_space_vertical_lead_m,
        runtime_reach_low_vertical_lead_m=config.task_space_low_vertical_lead_m,
        runtime_reach_mid_vertical_lead_m=config.task_space_mid_vertical_lead_m,
        runtime_reach_high_vertical_lead_m=config.task_space_high_vertical_lead_m,
        maximum_arm_target_step_rad=config.maximum_arm_target_step_rad,
        arm_target_filter_fraction=config.arm_target_filter_fraction,
        mosaic_gmt_model_path=(
            str(gmt_model_path.expanduser().resolve()) if config.gmt_blend > 0.0 else None
        ),
        mosaic_gmt_skill_path=(
            str(gmt_skill_path.expanduser().resolve()) if config.gmt_blend > 0.0 else None
        ),
        mosaic_gmt_blend=config.gmt_blend,
        mosaic_gmt_stability_floor=config.gmt_stability_floor,
        mosaic_gmt_minimum_target_height_m=config.gmt_minimum_target_height_m,
        mosaic_gmt_full_target_height_m=config.gmt_full_target_height_m,
        mosaic_gmt_lower_body_scale=config.gmt_lower_body_scale,
        mosaic_gmt_waist_scale=config.gmt_waist_scale,
        mosaic_gmt_arm_scale=config.gmt_arm_scale,
    )
    environment = GoalkeeperTargetedDiveMJWarpBatch(
        asset_root=asset_root,
        locomotion_policy_path=locomotion_policy_path,
        targeted_dive_checkpoint=targeted_dive_checkpoint,
        device=device,
        config=world,
        dive_config=dive,
    )
    observation = environment.reset(seed=config.random_seed)
    actor = None
    actor_checkpoint_hash = None
    actor_training_exam_contract = None
    if actor_checkpoint_path is not None:
        from torch import nn

        from rosclaw_soccer.training.goalkeeper_physics_ppo import (
            _build_actor_critic,
            _load_actor_critic_state,
            _migrate_initialization_state,
        )

        actor_file = actor_checkpoint_path.expanduser().resolve()
        payload = torch.load(actor_file, map_location=environment.device, weights_only=True)
        expected = (environment.observation_size, environment.action_size)
        actual = (int(payload["observation_size"]), int(payload["action_size"]))
        observation_compatible = actual[0] == expected[0] or (
            (actual[0], expected[0]) in {(89, 90), (89, 92), (90, 92), (92, 96)}
        )
        if (
            not observation_compatible
            or actual[1] != expected[1]
            or payload.get("activation_ceiling") != "SIM_ONLY"
        ):
            raise ValueError("goalkeeper probe actor checkpoint contract mismatch")
        actor = _build_actor_critic(
            torch,
            nn,
            expected[0],
            expected[1],
            int(payload["hidden_size"]),
        ).to(environment.device)
        actor_state, _ = _migrate_initialization_state(
            torch=torch,
            state_dict=payload["state_dict"],
            old_observation_size=actual[0],
            new_observation_size=expected[0],
        )
        _load_actor_critic_state(actor, actor_state)
        actor.eval()
        actor_checkpoint_hash = hash_bytes(actor_file.read_bytes())
        actor_training_exam_contract = _actor_training_exam_contract(
            payload.get("training_config"), config
        )
    ever_terminated = torch.zeros(
        config.environment_count, dtype=torch.bool, device=environment.device
    )
    ever_saved = torch.zeros_like(ever_terminated)
    ever_contact = torch.zeros_like(ever_terminated)
    ever_hand_contact = torch.zeros_like(ever_terminated)
    arrival_sampled = torch.zeros_like(ever_terminated)
    impact_sampled = torch.zeros_like(ever_terminated)
    peak_hand_height = torch.zeros(config.environment_count, device=environment.device)
    maximum_mean_knee_flexion = torch.full_like(peak_hand_height, -float("inf"))
    minimum_mean_hip_pitch = torch.full_like(peak_hand_height, float("inf"))
    maximum_target_side_waist_roll = torch.full_like(peak_hand_height, -float("inf"))
    minimum_pelvis = torch.full_like(peak_hand_height, float("inf"))
    minimum_upright = torch.ones_like(peak_hand_height)
    minimum_hand_target_distance = torch.full_like(peak_hand_height, float("inf"))
    initial_pelvis_lateral = environment.qpos[:, 1].clone()
    maximum_outward_pelvis_displacement = torch.zeros_like(peak_hand_height)
    closest_pelvis_target_lateral_remaining = torch.full_like(
        peak_hand_height,
        float("nan"),
    )
    closest_hand_outward_extension = torch.full_like(peak_hand_height, float("nan"))
    arrival_pelvis_target_lateral_remaining = torch.full_like(
        peak_hand_height,
        float("nan"),
    )
    arrival_hand_outward_extension = torch.full_like(peak_hand_height, float("nan"))
    closest_hand_target_axis_error = torch.full(
        (config.environment_count, 3),
        float("inf"),
        device=environment.device,
    )
    closest_hand_target_signed_error = torch.zeros_like(closest_hand_target_axis_error)
    arrival_hand_target_axis_error = torch.full_like(
        closest_hand_target_axis_error,
        float("nan"),
    )
    arrival_hand_target_signed_error = torch.full_like(
        closest_hand_target_axis_error,
        float("nan"),
    )
    impact_forward_velocity_retention = torch.full_like(peak_hand_height, float("nan"))
    for _ in range(world.episode_steps):
        pre_step_ball_forward_velocity = environment.qvel[:, 35].clone()
        if actor is None:
            action = torch.zeros(
                (config.environment_count, environment.action_size),
                device=environment.device,
            )
            action[:, 0] = config.option_gate
        else:
            with torch.inference_mode():
                action = torch.tanh(actor(observation)[0])
        observation, _, terminated, info = environment.step(action)
        mean_knee = 0.5 * (environment.qpos[:, 10] + environment.qpos[:, 16])
        mean_hip_pitch = 0.5 * (environment.qpos[:, 7] + environment.qpos[:, 13])
        target_direction = torch.sign(environment._target_one[:, 1])
        target_side_waist_roll = target_direction * environment.qpos[:, 20]
        maximum_mean_knee_flexion.copy_(torch.maximum(maximum_mean_knee_flexion, mean_knee))
        minimum_mean_hip_pitch.copy_(torch.minimum(minimum_mean_hip_pitch, mean_hip_pitch))
        maximum_target_side_waist_roll.copy_(
            torch.maximum(maximum_target_side_waist_roll, target_side_waist_roll)
        )
        ever_terminated |= terminated
        ever_saved |= info["true_save"]
        ever_contact |= info["ball_contact"]
        ever_hand_contact |= info["hand_contact"]
        peak_hand_height = torch.maximum(
            peak_hand_height,
            torch.maximum(
                environment.geom_xpos[:, environment._left_hand_geom, 2],
                environment.geom_xpos[:, environment._right_hand_geom, 2],
            ),
        )
        minimum_pelvis = torch.minimum(minimum_pelvis, environment.qpos[:, 2])
        upright = 2.0 * (environment.qpos[:, 3].square() + environment.qpos[:, 6].square()) - 1.0
        minimum_upright = torch.minimum(minimum_upright, upright)
        left_error = environment.geom_xpos[:, environment._left_hand_geom] - info["target_m"]
        right_error = environment.geom_xpos[:, environment._right_hand_geom] - info["target_m"]
        left_distance = torch.linalg.vector_norm(left_error, dim=1)
        right_distance = torch.linalg.vector_norm(right_error, dim=1)
        use_left = left_distance <= right_distance
        closest_error = torch.where(use_left.unsqueeze(1), left_error, right_error)
        closest_distance = torch.minimum(left_distance, right_distance)
        active_shot = info["event_shot_index"] == 1
        improved = active_shot & (closest_distance < minimum_hand_target_distance)
        closest_hand_target_axis_error = torch.where(
            improved.unsqueeze(1),
            torch.abs(closest_error),
            closest_hand_target_axis_error,
        )
        target_side = torch.where(
            info["target_m"][:, 1] < 0.0,
            -torch.ones_like(info["target_m"][:, 1]),
            torch.ones_like(info["target_m"][:, 1]),
        )
        signed_error = closest_error.clone()
        signed_error[:, 1] *= target_side
        selected_hand = torch.where(
            use_left.unsqueeze(1),
            environment.geom_xpos[:, environment._left_hand_geom],
            environment.geom_xpos[:, environment._right_hand_geom],
        )
        outward_pelvis_displacement = (
            environment.qpos[:, 1] - initial_pelvis_lateral
        ) * target_side
        maximum_outward_pelvis_displacement = torch.maximum(
            maximum_outward_pelvis_displacement,
            outward_pelvis_displacement,
        )
        pelvis_target_lateral_remaining = (
            info["target_m"][:, 1] - environment.qpos[:, 1]
        ) * target_side
        hand_outward_extension = (selected_hand[:, 1] - environment.qpos[:, 1]) * target_side
        closest_hand_target_signed_error = torch.where(
            improved.unsqueeze(1),
            signed_error,
            closest_hand_target_signed_error,
        )
        closest_pelvis_target_lateral_remaining = torch.where(
            improved,
            pelvis_target_lateral_remaining,
            closest_pelvis_target_lateral_remaining,
        )
        closest_hand_outward_extension = torch.where(
            improved,
            hand_outward_extension,
            closest_hand_outward_extension,
        )
        minimum_hand_target_distance = torch.minimum(
            minimum_hand_target_distance,
            torch.where(
                active_shot,
                closest_distance,
                torch.full_like(left_distance, float("inf")),
            ),
        )
        time_to_arrival = environment._estimated_time_to_arrival()
        sample_arrival = (
            active_shot & ~arrival_sampled & (time_to_arrival <= 1.5 * world.control_dt_sec)
        )
        arrival_hand_target_axis_error = torch.where(
            sample_arrival.unsqueeze(1),
            torch.abs(closest_error),
            arrival_hand_target_axis_error,
        )
        arrival_hand_target_signed_error = torch.where(
            sample_arrival.unsqueeze(1),
            signed_error,
            arrival_hand_target_signed_error,
        )
        arrival_pelvis_target_lateral_remaining = torch.where(
            sample_arrival,
            pelvis_target_lateral_remaining,
            arrival_pelvis_target_lateral_remaining,
        )
        arrival_hand_outward_extension = torch.where(
            sample_arrival,
            hand_outward_extension,
            arrival_hand_outward_extension,
        )
        arrival_sampled |= sample_arrival
        first_impact = (
            active_shot
            & info["ball_contact"]
            & ~impact_sampled
            & (pre_step_ball_forward_velocity > 0.25)
        )
        impact_forward_velocity_retention = torch.where(
            first_impact,
            environment.qvel[:, 35] / torch.clamp(pre_step_ball_forward_velocity, min=0.10),
            impact_forward_velocity_retention,
        )
        impact_sampled |= first_impact
    summary = environment.summary()

    def sampled_mean(values: Any, mask: Any) -> float | None:
        if not bool(torch.any(mask)):
            return None
        return float(values[mask].mean())

    report: dict[str, Any] = {
        "schema_version": "rosclaw.mosaic_gmt_goalkeeper_probe.v35",
        "config": asdict(config),
        "config_hash": config.config_hash,
        "environment_summary": summary,
        "actor_checkpoint_hash": actor_checkpoint_hash,
        "actor_training_exam_contract": actor_training_exam_contract,
        "actor_authority": (
            "NONE_OPEN_LOOP_GATE_ONLY"
            if actor is None
            else "FROZEN_DETERMINISTIC_BOUNDED_RESIDUAL_DIAGNOSTIC"
        ),
        "save_rate": float(ever_saved.to(torch.float32).mean()),
        "robot_ball_contact_rate": float(ever_contact.to(torch.float32).mean()),
        "hand_ball_contact_rate": float(ever_hand_contact.to(torch.float32).mean()),
        "failed_rate": float(summary["failed_rate"]),
        "episode_terminated_rate": float(ever_terminated.to(torch.float32).mean()),
        "unsafe_state_rate": float(
            ((minimum_pelvis < 0.30) | (minimum_upright < 0.05)).to(torch.float32).mean()
        ),
        "mean_peak_hand_height_m": float(peak_hand_height.mean()),
        "minimum_peak_hand_height_m": float(peak_hand_height.min()),
        "mean_minimum_pelvis_height_m": float(minimum_pelvis.mean()),
        "mean_maximum_bilateral_knee_flexion_rad": float(maximum_mean_knee_flexion.mean()),
        "mean_minimum_bilateral_hip_pitch_rad": float(minimum_mean_hip_pitch.mean()),
        "mean_maximum_target_side_waist_roll_rad": float(maximum_target_side_waist_roll.mean()),
        "minimum_pelvis_height_m": float(minimum_pelvis.min()),
        "mean_minimum_upright_projection": float(minimum_upright.mean()),
        "minimum_upright_projection": float(minimum_upright.min()),
        "mean_minimum_hand_target_distance_m": float(minimum_hand_target_distance.mean()),
        "p90_minimum_hand_target_distance_m": float(
            torch.quantile(minimum_hand_target_distance, 0.90)
        ),
        "maximum_minimum_hand_target_distance_m": float(minimum_hand_target_distance.max()),
        "mean_maximum_outward_pelvis_displacement_m": float(
            maximum_outward_pelvis_displacement.mean()
        ),
        "mean_closest_pelvis_target_lateral_remaining_m": float(
            closest_pelvis_target_lateral_remaining.mean()
        ),
        "mean_closest_hand_outward_extension_m": float(closest_hand_outward_extension.mean()),
        "mean_closest_hand_target_axis_error_m": {
            "forward_x": float(closest_hand_target_axis_error[:, 0].mean()),
            "lateral_y": float(closest_hand_target_axis_error[:, 1].mean()),
            "vertical_z": float(closest_hand_target_axis_error[:, 2].mean()),
        },
        "mean_closest_hand_target_signed_error_m": {
            "forward_x": float(closest_hand_target_signed_error[:, 0].mean()),
            "lateral_outward_y": float(closest_hand_target_signed_error[:, 1].mean()),
            "vertical_z": float(closest_hand_target_signed_error[:, 2].mean()),
        },
        "arrival_sample_fraction": float(arrival_sampled.to(torch.float32).mean()),
        "mean_arrival_hand_target_axis_error_m": {
            "forward_x": sampled_mean(arrival_hand_target_axis_error[:, 0], arrival_sampled),
            "lateral_y": sampled_mean(arrival_hand_target_axis_error[:, 1], arrival_sampled),
            "vertical_z": sampled_mean(arrival_hand_target_axis_error[:, 2], arrival_sampled),
        },
        "mean_arrival_hand_target_signed_error_m": (
            None
            if not bool(torch.any(arrival_sampled))
            else {
                "forward_x": sampled_mean(arrival_hand_target_signed_error[:, 0], arrival_sampled),
                "lateral_outward_y": sampled_mean(
                    arrival_hand_target_signed_error[:, 1], arrival_sampled
                ),
                "vertical_z": sampled_mean(arrival_hand_target_signed_error[:, 2], arrival_sampled),
            }
        ),
        "mean_arrival_pelvis_target_lateral_remaining_m": sampled_mean(
            arrival_pelvis_target_lateral_remaining,
            arrival_sampled,
        ),
        "mean_arrival_hand_outward_extension_m": sampled_mean(
            arrival_hand_outward_extension,
            arrival_sampled,
        ),
        "impact_sample_fraction": float(impact_sampled.to(torch.float32).mean()),
        "mean_impact_forward_velocity_retention": sampled_mean(
            impact_forward_velocity_retention,
            impact_sampled,
        ),
        "physics_world_steps": (
            config.environment_count * world.episode_steps * world.physics_substeps
        ),
        "finite_state": environment.finite_state(),
        "promotion_status": "DIAGNOSTIC_ONLY_NOT_PROMOTION_AUTHORITY",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    failed = environment.task.phase == 7
    qualified_first_save = ever_saved & ~failed
    report["qualified_first_save_rate"] = float(qualified_first_save.to(torch.float32).mean())
    target_height = environment._target_one[:, 2]
    height_masks = {
        "far_corner_low": target_height < 0.60,
        "far_corner_mid": (target_height >= 0.60) & (target_height < 1.10),
        "far_corner_high": target_height >= 1.10,
    }
    report["failed_rate_by_height"] = {
        name: (None if not bool(torch.any(mask)) else float(failed[mask].to(torch.float32).mean()))
        for name, mask in height_masks.items()
    }
    report["save_rate_by_height"] = {
        name: (
            None if not bool(torch.any(mask)) else float(ever_saved[mask].to(torch.float32).mean())
        )
        for name, mask in height_masks.items()
    }
    report["qualified_first_save_rate_by_height"] = {
        name: (
            None
            if not bool(torch.any(mask))
            else float(qualified_first_save[mask].to(torch.float32).mean())
        )
        for name, mask in height_masks.items()
    }
    report["reach_decomposition_by_height"] = {
        name: (
            None
            if not bool(torch.any(mask))
            else {
                "mean_maximum_outward_pelvis_displacement_m": float(
                    maximum_outward_pelvis_displacement[mask].mean()
                ),
                "mean_closest_pelvis_target_lateral_remaining_m": float(
                    closest_pelvis_target_lateral_remaining[mask].mean()
                ),
                "mean_closest_hand_outward_extension_m": float(
                    closest_hand_outward_extension[mask].mean()
                ),
                "mean_closest_lateral_underreach_m": float(
                    -closest_hand_target_signed_error[mask, 1].mean()
                ),
                "mean_closest_forward_signed_error_m": float(
                    closest_hand_target_signed_error[mask, 0].mean()
                ),
                "mean_closest_vertical_signed_error_m": float(
                    closest_hand_target_signed_error[mask, 2].mean()
                ),
                "mean_minimum_hand_target_distance_m": float(
                    minimum_hand_target_distance[mask].mean()
                ),
                "hand_ball_contact_rate": float(ever_hand_contact[mask].to(torch.float32).mean()),
            }
        )
        for name, mask in height_masks.items()
    }
    side_masks = {
        "left_negative_y": environment._target_one[:, 1] < 0.0,
        "right_positive_y": environment._target_one[:, 1] >= 0.0,
    }

    def side_summary(mask: Any) -> dict[str, Any] | None:
        if not bool(torch.any(mask)):
            return None
        saved_mask = ever_saved & mask
        save_then_failed = ever_saved & failed & mask
        saved_count = int(saved_mask.sum())
        return {
            "episodes": int(mask.sum()),
            "first_save_rate": float(ever_saved[mask].to(torch.float32).mean()),
            "qualified_first_save_rate": float(qualified_first_save[mask].to(torch.float32).mean()),
            "save_then_failed_rate": float(save_then_failed[mask].to(torch.float32).mean()),
            "conditional_failure_after_save_rate": (
                None
                if saved_count == 0
                else float(save_then_failed.sum().to(torch.float32) / saved_count)
            ),
            "failed_rate": float(failed[mask].to(torch.float32).mean()),
            "maximum_root_angular_speed_rad_s": float(
                environment._maximum_root_angular_speed[mask].max()
            ),
            "strict_stability_ceiling_exceedance_rate": float(
                (environment._maximum_root_angular_speed[mask] > 3.50).to(torch.float32).mean()
            ),
            "mean_minimum_hand_target_distance_m": float(minimum_hand_target_distance[mask].mean()),
        }

    report["side_diagnostics"] = {name: side_summary(mask) for name, mask in side_masks.items()}
    failure_cases = []
    for index in torch.nonzero(failed, as_tuple=False).flatten().detach().cpu().tolist():
        failure_context = (
            environment._failure_pelvis_height[index],
            environment._failure_upright_projection[index],
            environment._failure_root_linear_speed[index],
            environment._failure_root_angular_speed[index],
            environment._failure_maximum_applied_option_gate[index],
        )
        failure_pelvis = _finite_float_or_none(environment._failure_pelvis_height[index])
        failure_upright = _finite_float_or_none(environment._failure_upright_projection[index])
        nonfinite_quarantined = bool(environment._nonfinite_quarantine_latched[index])
        pelvis_low = (
            failure_pelvis is not None
            and failure_pelvis < environment.task.config.minimum_pelvis_height_m
        )
        upright_low = (
            failure_upright is not None
            and failure_upright < environment.task.config.minimum_upright_projection
        )
        if nonfinite_quarantined:
            failure_reason = "NONFINITE_STATE"
        elif pelvis_low and upright_low:
            failure_reason = "PELVIS_AND_UPRIGHT_ENVELOPE"
        elif pelvis_low:
            failure_reason = "PELVIS_HEIGHT_ENVELOPE"
        elif upright_low:
            failure_reason = "UPRIGHT_ENVELOPE"
        else:
            failure_reason = "UNCLASSIFIED_SAFETY_STATE"
        failure_cases.append(
            {
                "environment_index": int(index),
                "target_m": [float(value) for value in environment._target_one[index]],
                "flight_time_sec": float(environment._flight_one[index]),
                "minimum_pelvis_height_m": float(minimum_pelvis[index]),
                "minimum_upright_projection": float(minimum_upright[index]),
                "failure_reason": failure_reason,
                "failure_pelvis_height_m": failure_pelvis,
                "failure_upright_projection": failure_upright,
                "failure_root_linear_speed_mps": _finite_float_or_none(
                    environment._failure_root_linear_speed[index]
                ),
                "failure_root_angular_speed_rad_s": _finite_float_or_none(
                    environment._failure_root_angular_speed[index]
                ),
                "failure_step_index": int(environment._failure_step_index[index]),
                "failure_time_sec": float(
                    environment._failure_step_index[index] * environment.config.control_dt_sec
                ),
                "failure_posture_exception_granted": bool(
                    environment._failure_posture_exception_granted[index]
                ),
                "failure_option_age_steps": int(environment._failure_option_age_steps[index]),
                "failure_maximum_applied_option_gate": _finite_float_or_none(
                    environment._failure_maximum_applied_option_gate[index]
                ),
                "failure_context_complete": bool(
                    environment._failure_step_index[index] >= 0
                    and environment._failure_option_age_steps[index] >= 0
                    and all(math.isfinite(float(value)) for value in failure_context)
                ),
                "minimum_hand_target_distance_m": float(minimum_hand_target_distance[index]),
                "first_save": bool(environment.task.first_save[index]),
                "first_hand_save": bool(environment.task.first_hand_save[index]),
                "nonfinite_quarantined": nonfinite_quarantined,
            }
        )
    report["failure_cases"] = failure_cases
    angular_tail_indices = torch.nonzero(
        environment._maximum_root_angular_speed > 3.20,
        as_tuple=False,
    ).flatten()
    if int(angular_tail_indices.numel()) > 0:
        angular_tail_indices = angular_tail_indices[
            torch.argsort(
                environment._maximum_root_angular_speed[angular_tail_indices],
                descending=True,
            )
        ][:16]
    report["root_angular_tail_threshold_rad_s"] = 3.20
    report["root_angular_tail_cases"] = [
        {
            "environment_index": int(index),
            "target_m": [float(value) for value in environment._target_one[index]],
            "intent_cue": [float(value) for value in environment._intent_cue_one[index]],
            "flight_time_sec": float(environment._flight_one[index]),
            "maximum_root_angular_speed_rad_s": float(
                environment._maximum_root_angular_speed[index]
            ),
            "minimum_substep_option_lower_body_authority": float(
                environment._minimum_substep_option_lower_body_authority[index]
            ),
            "maximum_applied_option_gate": float(environment._maximum_applied_option_gate[index]),
            "minimum_hand_target_distance_m": float(minimum_hand_target_distance[index]),
            "minimum_pelvis_height_m": float(minimum_pelvis[index]),
            "minimum_upright_projection": float(minimum_upright[index]),
            "first_save": bool(environment.task.first_save[index]),
            "first_hand_save": bool(environment.task.first_hand_save[index]),
            "failed": bool(failed[index]),
        }
        for index in angular_tail_indices.detach().cpu().tolist()
    ]
    sanitized_report, nonfinite_paths = _sanitize_nonfinite_evidence(report)
    if not isinstance(sanitized_report, dict):
        raise RuntimeError("goalkeeper probe evidence sanitization changed the root type")
    report = sanitized_report
    report["diagnostic_nonfinite_paths"] = nonfinite_paths
    report["diagnostic_complete"] = not nonfinite_paths
    report["report_hash"] = hash_json(report)
    if output_path is not None:
        destination = output_path.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--locomotion-policy", type=Path, required=True)
    parser.add_argument("--targeted-dive-checkpoint", type=Path, required=True)
    parser.add_argument("--gmt-model", type=Path, required=True)
    parser.add_argument("--gmt-skill", type=Path, required=True)
    parser.add_argument("--actor-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--environments", type=int, default=32)
    parser.add_argument("--seed", type=int, default=451)
    parser.add_argument("--option-gate", type=float, default=1.0)
    parser.add_argument("--minimum-option-gate", type=float, default=0.0)
    parser.add_argument("--actor-residual-scale", type=float, default=0.70)
    parser.add_argument("--lateral-drive-scale", type=float, default=0.0)
    parser.add_argument("--negative-target-lateral-drive-scale", type=float, default=1.0)
    parser.add_argument("--lateral-drive-full-activation-gate", type=float, default=0.30)
    parser.add_argument("--lateral-drive-capture", action="store_true")
    parser.add_argument("--lateral-drive-capture-horizon-sec", type=float, default=0.35)
    parser.add_argument("--lateral-drive-target-standoff-m", type=float, default=0.32)
    parser.add_argument("--lateral-drive-capture-scale-m", type=float, default=0.45)
    parser.add_argument("--lateral-drive-learned-gate", action="store_true")
    parser.add_argument("--lateral-lunge-blend", type=float, default=0.0)
    parser.add_argument("--lateral-lunge-hip-roll-rad", type=float, default=0.18)
    parser.add_argument("--lateral-lunge-ankle-roll-rad", type=float, default=0.12)
    parser.add_argument("--lateral-lunge-approach-horizon-sec", type=float, default=0.90)
    parser.add_argument("--substep-upper-body-guard", action="store_true")
    parser.add_argument("--substep-upper-body-guard-onset-rad-s", type=float, default=1.80)
    parser.add_argument("--substep-upper-body-guard-ceiling-rad-s", type=float, default=3.00)
    parser.add_argument("--substep-upper-body-minimum-position-scale", type=float, default=0.05)
    parser.add_argument("--substep-option-lower-body-guard", action="store_true")
    parser.add_argument("--substep-option-lower-body-guard-onset-rad-s", type=float, default=2.40)
    parser.add_argument("--substep-option-lower-body-guard-ceiling-rad-s", type=float, default=3.30)
    parser.add_argument("--substep-option-lower-body-minimum-scale", type=float, default=0.0)
    parser.add_argument("--canonical-locomotion-mirror", action="store_true")
    parser.add_argument("--official-goalkeeper-teacher-checkpoint", type=Path)
    parser.add_argument("--official-goalkeeper-teacher-blend", type=float, default=1.0)
    parser.add_argument(
        "--official-goalkeeper-lower-body-target-step-rad", type=float, default=0.08
    )
    parser.add_argument(
        "--official-goalkeeper-lower-body-filter-fraction", type=float, default=0.35
    )
    parser.add_argument("--official-goalkeeper-waist-target-step-rad", type=float, default=0.05)
    parser.add_argument("--official-goalkeeper-waist-filter-fraction", type=float, default=0.25)
    parser.add_argument("--official-goalkeeper-arm-target-step-rad", type=float, default=0.08)
    parser.add_argument("--official-goalkeeper-arm-filter-fraction", type=float, default=0.35)
    parser.add_argument("--low-shot-phase-scale", type=float, default=1.0)
    parser.add_argument("--mid-shot-phase-scale", type=float, default=1.0)
    parser.add_argument("--high-shot-phase-scale", type=float, default=1.0)
    parser.add_argument("--anchor-lower-body-scale", type=float, default=0.25)
    parser.add_argument("--anchor-waist-scale", type=float, default=0.50)
    parser.add_argument("--decoder-lower-body-residual-authority", type=float, default=0.10)
    parser.add_argument("--decoder-lower-body-command-scale", type=float)
    parser.add_argument("--decoder-waist-residual-authority", type=float, default=0.10)
    parser.add_argument("--gmt-blend", type=float, default=1.0)
    parser.add_argument("--gmt-stability-floor", type=float, default=0.0)
    parser.add_argument("--gmt-lower-body-scale", type=float, default=1.0)
    parser.add_argument("--gmt-waist-scale", type=float, default=1.0)
    parser.add_argument("--gmt-arm-scale", type=float, default=1.0)
    parser.add_argument("--task-space-reach-blend", type=float, default=0.0)
    parser.add_argument("--task-space-reach-feedback-blend", type=float, default=0.0)
    parser.add_argument("--task-space-reach-feedback-gain", type=float, default=0.70)
    parser.add_argument("--task-space-reach-feedback-support-scale", type=float, default=0.0)
    parser.add_argument("--contact-support-side", action="store_true")
    parser.add_argument("--actor-contact-support-side", action="store_true")
    parser.add_argument("--actor-recovery-context", action="store_true")
    parser.add_argument("--task-space-reach-feedback-maximum-error-m", type=float, default=0.30)
    parser.add_argument("--whole-body-reach-blend", type=float, default=0.0)
    parser.add_argument("--whole-body-reach-waist-scale", type=float, default=0.75)
    parser.add_argument("--whole-body-reach-arm-scale", type=float, default=1.0)
    parser.add_argument("--whole-body-reach-support-scale", type=float, default=0.65)
    parser.add_argument("--whole-body-reach-release-sec", type=float, default=0.60)
    parser.add_argument("--task-space-contact-standoff-m", type=float, default=0.0)
    parser.add_argument("--task-space-lateral-lead-m", type=float, default=0.0)
    parser.add_argument("--task-space-vertical-lead-m", type=float, default=0.0)
    parser.add_argument("--task-space-low-vertical-lead-m", type=float)
    parser.add_argument("--task-space-mid-vertical-lead-m", type=float)
    parser.add_argument("--task-space-high-vertical-lead-m", type=float)
    parser.add_argument("--first-shot-release-sec", type=float, default=0.70)
    parser.add_argument("--prediction-lead-sec", type=float, default=0.30)
    parser.add_argument("--option-duration-sec", type=float, default=0.95)
    parser.add_argument("--actor-recovery-plasticity-sec", type=float, default=0.0)
    parser.add_argument("--actor-recovery-residual-authority-scale", type=float, default=0.50)
    parser.add_argument("--post-save-counterstep", action="store_true")
    parser.add_argument("--post-save-counterstep-duration-sec", type=float, default=0.80)
    parser.add_argument("--post-save-counterstep-command-limit", type=float, default=0.55)
    parser.add_argument("--post-save-counterstep-capture-horizon-sec", type=float, default=0.28)
    parser.add_argument("--post-save-counterstep-recenter-weight", type=float, default=1.0)
    parser.add_argument("--post-save-option-release-sec", type=float, default=0.30)
    parser.add_argument("--reach-approach-horizon-sec", type=float, default=0.55)
    parser.add_argument("--posture-exception-duration-sec", type=float, default=1.55)
    parser.add_argument("--root-angular-speed-guard-ceiling-rad-s", type=float, default=8.0)
    parser.add_argument("--dive-minimum-pelvis-height-m", type=float, default=0.30)
    parser.add_argument("--dive-minimum-upright-projection", type=float, default=0.05)
    parser.add_argument("--dive-maximum-root-linear-speed-mps", type=float, default=3.0)
    parser.add_argument(
        "--hard-shot-height-mode",
        choices=("low", "mid", "high", "balanced"),
        default="high",
    )
    parser.add_argument("--gmt-minimum-target-height-m", type=float, default=1.10)
    parser.add_argument("--gmt-full-target-height-m", type=float, default=1.25)
    parser.add_argument("--maximum-arm-target-step-rad", type=float, default=0.20)
    parser.add_argument("--arm-target-filter-fraction", type=float, default=1.0)
    args = parser.parse_args()
    report = run_mosaic_gmt_goalkeeper_probe(
        asset_root=args.asset_root,
        locomotion_policy_path=args.locomotion_policy,
        targeted_dive_checkpoint=args.targeted_dive_checkpoint,
        gmt_model_path=args.gmt_model,
        gmt_skill_path=args.gmt_skill,
        device=args.device,
        config=MosaicGMTGoalkeeperProbeConfig(
            environment_count=args.environments,
            random_seed=args.seed,
            option_gate=args.option_gate,
            minimum_option_gate=args.minimum_option_gate,
            actor_residual_scale=args.actor_residual_scale,
            lateral_drive_scale=args.lateral_drive_scale,
            negative_target_lateral_drive_scale=(args.negative_target_lateral_drive_scale),
            lateral_drive_full_activation_gate=(args.lateral_drive_full_activation_gate),
            lateral_drive_capture_enabled=args.lateral_drive_capture,
            lateral_drive_capture_horizon_sec=(args.lateral_drive_capture_horizon_sec),
            lateral_drive_target_standoff_m=(args.lateral_drive_target_standoff_m),
            lateral_drive_capture_scale_m=args.lateral_drive_capture_scale_m,
            lateral_drive_learned_gate_enabled=args.lateral_drive_learned_gate,
            lateral_lunge_blend=args.lateral_lunge_blend,
            lateral_lunge_hip_roll_rad=args.lateral_lunge_hip_roll_rad,
            lateral_lunge_ankle_roll_rad=args.lateral_lunge_ankle_roll_rad,
            lateral_lunge_approach_horizon_sec=(args.lateral_lunge_approach_horizon_sec),
            substep_upper_body_guard_enabled=args.substep_upper_body_guard,
            substep_upper_body_guard_onset_rad_s=(args.substep_upper_body_guard_onset_rad_s),
            substep_upper_body_guard_ceiling_rad_s=(args.substep_upper_body_guard_ceiling_rad_s),
            substep_upper_body_minimum_position_scale=(
                args.substep_upper_body_minimum_position_scale
            ),
            substep_option_lower_body_guard_enabled=(args.substep_option_lower_body_guard),
            substep_option_lower_body_guard_onset_rad_s=(
                args.substep_option_lower_body_guard_onset_rad_s
            ),
            substep_option_lower_body_guard_ceiling_rad_s=(
                args.substep_option_lower_body_guard_ceiling_rad_s
            ),
            substep_option_lower_body_minimum_scale=(args.substep_option_lower_body_minimum_scale),
            canonical_locomotion_mirror_enabled=(args.canonical_locomotion_mirror),
            official_goalkeeper_teacher_checkpoint=(
                None
                if args.official_goalkeeper_teacher_checkpoint is None
                else str(args.official_goalkeeper_teacher_checkpoint)
            ),
            official_goalkeeper_teacher_blend=args.official_goalkeeper_teacher_blend,
            official_goalkeeper_lower_body_target_step_rad=(
                args.official_goalkeeper_lower_body_target_step_rad
            ),
            official_goalkeeper_lower_body_filter_fraction=(
                args.official_goalkeeper_lower_body_filter_fraction
            ),
            official_goalkeeper_waist_target_step_rad=(
                args.official_goalkeeper_waist_target_step_rad
            ),
            official_goalkeeper_waist_filter_fraction=(
                args.official_goalkeeper_waist_filter_fraction
            ),
            official_goalkeeper_arm_target_step_rad=(args.official_goalkeeper_arm_target_step_rad),
            official_goalkeeper_arm_filter_fraction=(args.official_goalkeeper_arm_filter_fraction),
            low_shot_phase_scale=args.low_shot_phase_scale,
            mid_shot_phase_scale=args.mid_shot_phase_scale,
            high_shot_phase_scale=args.high_shot_phase_scale,
            anchor_lower_body_scale=args.anchor_lower_body_scale,
            anchor_waist_scale=args.anchor_waist_scale,
            decoder_lower_body_residual_authority=(args.decoder_lower_body_residual_authority),
            decoder_lower_body_command_scale=args.decoder_lower_body_command_scale,
            decoder_waist_residual_authority=args.decoder_waist_residual_authority,
            gmt_blend=args.gmt_blend,
            gmt_stability_floor=args.gmt_stability_floor,
            gmt_lower_body_scale=args.gmt_lower_body_scale,
            gmt_waist_scale=args.gmt_waist_scale,
            gmt_arm_scale=args.gmt_arm_scale,
            task_space_reach_blend=args.task_space_reach_blend,
            task_space_reach_feedback_blend=args.task_space_reach_feedback_blend,
            task_space_reach_feedback_gain=args.task_space_reach_feedback_gain,
            task_space_reach_feedback_support_scale=(args.task_space_reach_feedback_support_scale),
            contact_support_side_enabled=args.contact_support_side,
            actor_contact_support_side_enabled=args.actor_contact_support_side,
            actor_recovery_context_enabled=args.actor_recovery_context,
            task_space_reach_feedback_maximum_error_m=(
                args.task_space_reach_feedback_maximum_error_m
            ),
            whole_body_reach_blend=args.whole_body_reach_blend,
            whole_body_reach_waist_scale=args.whole_body_reach_waist_scale,
            whole_body_reach_arm_scale=args.whole_body_reach_arm_scale,
            whole_body_reach_support_scale=args.whole_body_reach_support_scale,
            whole_body_reach_release_sec=args.whole_body_reach_release_sec,
            task_space_contact_standoff_m=args.task_space_contact_standoff_m,
            task_space_lateral_lead_m=args.task_space_lateral_lead_m,
            task_space_vertical_lead_m=args.task_space_vertical_lead_m,
            task_space_low_vertical_lead_m=args.task_space_low_vertical_lead_m,
            task_space_mid_vertical_lead_m=args.task_space_mid_vertical_lead_m,
            task_space_high_vertical_lead_m=args.task_space_high_vertical_lead_m,
            first_shot_release_sec=args.first_shot_release_sec,
            prediction_lead_sec=args.prediction_lead_sec,
            option_duration_sec=args.option_duration_sec,
            actor_recovery_plasticity_sec=args.actor_recovery_plasticity_sec,
            actor_recovery_residual_authority_scale=(args.actor_recovery_residual_authority_scale),
            post_save_counterstep_enabled=args.post_save_counterstep,
            post_save_counterstep_duration_sec=(args.post_save_counterstep_duration_sec),
            post_save_counterstep_command_limit=(args.post_save_counterstep_command_limit),
            post_save_counterstep_capture_horizon_sec=(
                args.post_save_counterstep_capture_horizon_sec
            ),
            post_save_counterstep_recenter_weight=(args.post_save_counterstep_recenter_weight),
            post_save_option_release_sec=args.post_save_option_release_sec,
            reach_approach_horizon_sec=args.reach_approach_horizon_sec,
            posture_exception_duration_sec=args.posture_exception_duration_sec,
            root_angular_speed_guard_ceiling_rad_s=(args.root_angular_speed_guard_ceiling_rad_s),
            dive_minimum_pelvis_height_m=args.dive_minimum_pelvis_height_m,
            dive_minimum_upright_projection=args.dive_minimum_upright_projection,
            dive_maximum_root_linear_speed_mps=args.dive_maximum_root_linear_speed_mps,
            hard_shot_height_mode=args.hard_shot_height_mode,
            gmt_minimum_target_height_m=args.gmt_minimum_target_height_m,
            gmt_full_target_height_m=args.gmt_full_target_height_m,
            maximum_arm_target_step_rad=args.maximum_arm_target_step_rad,
            arm_target_filter_fraction=args.arm_target_filter_fraction,
        ),
        actor_checkpoint_path=args.actor_checkpoint,
        output_path=args.output,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MosaicGMTGoalkeeperProbeConfig",
    "run_mosaic_gmt_goalkeeper_probe",
]
