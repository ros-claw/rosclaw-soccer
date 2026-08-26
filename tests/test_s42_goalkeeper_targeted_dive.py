from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.training.goalkeeper_dive_option import (
    GoalkeeperDiveOptionConfig,
    GoalkeeperDivePhase,
)
from rosclaw_soccer.training.goalkeeper_mjwarp import GoalkeeperMJWarpConfig
from rosclaw_soccer.training.goalkeeper_physics_ppo import (
    GoalkeeperPhysicsPPOConfig,
    _sample_exploration_action,
    _stratum_balance_score,
    _stratum_rates,
    _with_first_shot_release_override,
    _with_root_angular_penalty_override,
    _with_save_event_bonus_override,
)
from rosclaw_soccer.training.goalkeeper_reach import (
    GoalkeeperReachAtlasConfig,
    GoalkeeperReachConfig,
)
from rosclaw_soccer.training.goalkeeper_targeted_dive import (
    TARGETED_DIVE_INPUT_SIZE,
    GoalkeeperTargetedDiveConfig,
    GoalkeeperTorchDiveMonitor,
    _mirror_features_torch,
    _mirror_joints_torch,
    build_targeted_dive_decoder,
    decode_goalkeeper_targeted_dive,
    targeted_dive_features_numpy,
    targeted_dive_features_torch,
)
from rosclaw_soccer.training.goalkeeper_targeted_dive_exam import (
    GoalkeeperTargetedDiveExamConfig,
    _sample_cases,
    _scheduled_phase,
)
from rosclaw_soccer.training.goalkeeper_targeted_dive_mjwarp import (
    TARGETED_DIVE_RL_RESIDUAL_LIMITS_RAD,
    GoalkeeperTargetedDiveMJWarpBatch,
    GoalkeeperTargetedDiveRLConfig,
    _body_frame_waist_counter_rotation,
    _capture_point_lateral_drive,
)


def _features() -> dict[str, np.ndarray]:
    return {
        "direction": np.asarray((-1.0, 1.0)),
        "phase": np.asarray((0.25, 0.75)),
        "target_lateral_m": np.asarray((-0.92, 1.02)),
        "target_height_m": np.asarray((0.44, 1.38)),
        "time_to_arrival_sec": np.asarray((0.48, 0.12)),
        "root_lateral_m": np.asarray((-0.18, 0.27)),
        "root_lateral_speed_mps": np.asarray((-0.32, 0.41)),
        "pelvis_height_m": np.asarray((0.76, 0.61)),
        "upright_projection": np.asarray((0.96, 0.72)),
        "support_side": np.asarray((1.0, -1.0)),
        "root_angular_speed_rad_s": np.asarray((0.2, 1.8)),
    }


def test_targeted_dive_contract_is_bounded_and_sim_only() -> None:
    config = GoalkeeperTargetedDiveConfig(
        epochs=10,
        samples_per_epoch=4_096,
        low_height_crouch_scale=1.20,
        high_height_extension_scale=0.80,
    )
    assert config.config_hash.startswith("sha256:")
    assert config.low_height_crouch_scale == 1.20
    assert config.low_height_max_knee_flexion_delta_rad == 0.70
    assert config.high_height_extension_scale == 0.80
    with pytest.raises(ValueError, match="SIM_ONLY"):
        replace(config, hardware_authorized=True)
    with pytest.raises(ValueError, match="SIM_ONLY"):
        replace(config, commercial_use_allowed=True)
    with pytest.raises(ValueError, match="sample"):
        replace(config, samples_per_epoch=4_095)
    with pytest.raises(ValueError, match="mirrored pairs"):
        replace(config, samples_per_epoch=4_097)
    with pytest.raises(ValueError, match="refresh interval"):
        replace(config, hard_mining_refresh_epochs=0)
    with pytest.raises(ValueError, match="general replay fraction"):
        replace(config, hard_mining_general_replay_fraction=0.91)
    with pytest.raises(ValueError, match="posture scale"):
        replace(config, low_height_crouch_scale=1.51)
    with pytest.raises(ValueError, match="crouch joint delta"):
        replace(config, low_height_max_knee_flexion_delta_rad=0.86)
    with pytest.raises(ValueError, match="torso joint delta"):
        replace(config, low_height_max_waist_roll_delta_rad=0.36)
    with pytest.raises(ValueError, match="crouch lateral progress"):
        replace(
            config,
            low_height_crouch_start_lateral_progress=0.75,
            low_height_crouch_full_lateral_progress=0.70,
        )
    recovery = replace(config, recovery_tail_start_phase=0.72)
    assert recovery.recovery_tail_start_phase == pytest.approx(0.72)
    with pytest.raises(ValueError, match="recovery-tail phase"):
        replace(config, recovery_tail_start_phase=0.54)
    reach = GoalkeeperReachConfig(workspace_scale=2.0)
    assert reach.workspace_scale == 2.0
    with pytest.raises(ValueError, match="workspace scale"):
        replace(reach, workspace_scale=2.51)
    atlas = GoalkeeperReachAtlasConfig(multistart_count=12)
    assert atlas.multistart_count == 12
    with pytest.raises(ValueError, match="multistart"):
        replace(atlas, multistart_count=33)
    smooth_atlas = replace(
        atlas,
        interpolation_neighbors=42,
        interpolation_kernel="gaussian",
    )
    assert smooth_atlas.interpolation_neighbors == 42
    with pytest.raises(ValueError, match="neighbor"):
        replace(atlas, interpolation_neighbors=43)


def test_targeted_dive_rl_contract_bounds_full_body_residuals(tmp_path: Path) -> None:
    config = GoalkeeperTargetedDiveRLConfig()
    assert config.config_hash.startswith("sha256:")
    assert config.actor_plasticity_duration_sec == pytest.approx(config.option_duration_sec)
    extended_recovery = replace(config, posture_exception_duration_sec=2.0)
    assert extended_recovery.actor_plasticity_duration_sec == pytest.approx(
        config.actor_plasticity_duration_sec
    )
    learned_recovery = replace(config, actor_recovery_plasticity_sec=0.60)
    assert learned_recovery.actor_plasticity_duration_sec == pytest.approx(
        config.option_duration_sec + 0.60
    )
    assert learned_recovery.actor_recovery_residual_authority_scale == pytest.approx(0.50)
    long_recovery = replace(config, actor_recovery_plasticity_sec=4.0)
    assert long_recovery.actor_plasticity_duration_sec == pytest.approx(4.9)
    with pytest.raises(ValueError, match="recovery-plasticity"):
        replace(config, actor_recovery_plasticity_sec=5.01)
    with pytest.raises(ValueError, match="recovery residual authority"):
        replace(config, actor_recovery_residual_authority_scale=0.01)
    counterstep = replace(
        config,
        post_save_counterstep_enabled=True,
        post_save_counterstep_command_limit=0.75,
        post_save_counterstep_recenter_weight=0.25,
        post_save_option_release_sec=0.24,
    )
    assert counterstep.post_save_counterstep_enabled
    assert counterstep.post_save_counterstep_command_limit == pytest.approx(0.75)
    assert counterstep.post_save_counterstep_recenter_weight == pytest.approx(0.25)
    with pytest.raises(ValueError, match="counterstep settings"):
        replace(config, post_save_counterstep_command_limit=1.01)
    with pytest.raises(ValueError, match="counterstep settings"):
        replace(config, post_save_counterstep_recenter_weight=1.01)
    with pytest.raises(ValueError, match="counterstep settings"):
        replace(
            config,
            post_save_counterstep_duration_sec=0.30,
            post_save_option_release_sec=0.31,
        )
    with pytest.raises(ValueError, match="requires counterstep"):
        replace(config, post_save_fall_recovery_enabled=True)
    fall_recovery = replace(
        config,
        post_save_counterstep_enabled=True,
        actor_recovery_plasticity_sec=1.0,
        runtime_contact_support_side_enabled=True,
        actor_contact_support_side_enabled=True,
        actor_recovery_context_enabled=True,
        post_save_fall_recovery_enabled=True,
    )
    assert fall_recovery.post_save_fall_recovery_enabled
    long_fall_recovery = replace(fall_recovery, post_save_fall_recovery_duration_sec=4.8)
    assert long_fall_recovery.post_save_fall_recovery_duration_sec == pytest.approx(4.8)
    with pytest.raises(ValueError, match="fall-recovery envelope"):
        replace(fall_recovery, post_save_fall_recovery_duration_sec=5.01)
    model = tmp_path / "gmt.onnx"
    skill = tmp_path / "getup.json"
    model.write_bytes(b"model")
    skill.write_text("{}", encoding="utf-8")
    getup = replace(
        long_fall_recovery,
        mosaic_gmt_model_path=str(model),
        mosaic_gmt_getup_skill_path=str(skill),
        mosaic_gmt_getup_blend=0.75,
    )
    assert getup.mosaic_gmt_blend == 0.0
    assert getup.mosaic_gmt_getup_blend == pytest.approx(0.75)
    with pytest.raises(ValueError, match="get-up settings"):
        replace(getup, mosaic_gmt_getup_blend_in_sec=0.05)
    assert len(TARGETED_DIVE_RL_RESIDUAL_LIMITS_RAD) == 29
    assert GoalkeeperTargetedDiveMJWarpBatch.arm_action_start_index == 16
    assert GoalkeeperTargetedDiveMJWarpBatch.residual_arm_start_index == 15
    assert max(TARGETED_DIVE_RL_RESIDUAL_LIMITS_RAD[:12]) <= 0.12
    with pytest.raises(ValueError, match="SIM_ONLY"):
        replace(config, hardware_authorized=True)
    with pytest.raises(ValueError, match="posture exception"):
        replace(config, posture_exception_duration_sec=0.50)
    guarded = replace(config, dive_maximum_root_angular_speed_rad_s=3.5)
    assert guarded.dive_maximum_root_angular_speed_rad_s == 3.5
    with pytest.raises(ValueError, match="angular-speed guard"):
        replace(config, dive_maximum_root_angular_speed_rad_s=2.99)
    delayed = replace(config, phase_hold_sec=0.25)
    assert delayed.phase_hold_sec == 0.25
    assert delayed.config_hash != config.config_hash
    with pytest.raises(ValueError, match="phase hold"):
        replace(config, phase_hold_sec=0.41)
    with pytest.raises(ValueError, match="nominal flight"):
        replace(config, nominal_shot_flight_time_sec=0.29)
    synchronized = replace(config, intercept_phase_at_arrival=0.70)
    assert synchronized.intercept_phase_at_arrival == 0.70
    with pytest.raises(ValueError, match="intercept phase"):
        replace(config, intercept_phase_at_arrival=0.86)
    with pytest.raises(ValueError, match="phase-sync height"):
        replace(config, phase_sync_minimum_target_height_m=1.21)
    with pytest.raises(ValueError, match="posture exception"):
        replace(config, phase_hold_sec=0.40, posture_exception_duration_sec=1.20)
    reach = replace(config, runtime_reach_blend=0.85, runtime_reach_lateral_lead_m=0.20)
    assert reach.runtime_reach_blend == 0.85
    assert reach.runtime_reach_lateral_lead_m == 0.20
    with pytest.raises(ValueError, match="reach blend"):
        replace(config, runtime_reach_blend=1.01)
    with pytest.raises(ValueError, match="reach compensation"):
        replace(config, runtime_reach_lateral_lead_m=0.20)
    feedback = replace(
        config,
        runtime_reach_blend=0.85,
        runtime_reach_feedback_blend=0.70,
        runtime_reach_feedback_gain=0.80,
    )
    assert feedback.runtime_reach_feedback_blend == 0.70
    with pytest.raises(ValueError, match="feedback requires"):
        replace(config, runtime_reach_feedback_blend=0.50)
    with pytest.raises(ValueError, match="reach timing"):
        replace(config, runtime_reach_full_lead_sec=0.55)
    reflex = replace(config, minimum_option_gate=0.60)
    assert reflex.minimum_option_gate == 0.60
    with pytest.raises(ValueError, match="minimum option gate"):
        replace(config, minimum_option_gate=0.81)
    filtered = replace(
        config,
        maximum_arm_target_step_rad=0.08,
        arm_target_filter_fraction=0.40,
    )
    assert filtered.maximum_arm_target_step_rad == 0.08
    with pytest.raises(ValueError, match="arm target filter"):
        replace(config, maximum_arm_target_step_rad=0.01)
    driven = replace(config, lateral_drive_scale=0.75)
    assert driven.lateral_drive_scale == 0.75
    directional_drive = replace(config, negative_target_lateral_drive_scale=0.60)
    assert directional_drive.negative_target_lateral_drive_scale == 0.60
    with pytest.raises(ValueError, match="lateral drive scale"):
        replace(config, lateral_drive_scale=1.01)
    with pytest.raises(ValueError, match="negative-target lateral drive"):
        replace(config, negative_target_lateral_drive_scale=1.01)
    capture_drive = replace(
        config,
        lateral_drive_capture_enabled=True,
        lateral_drive_capture_horizon_sec=0.40,
        lateral_drive_learned_gate_enabled=True,
    )
    assert capture_drive.lateral_drive_capture_enabled
    assert capture_drive.lateral_drive_learned_gate_enabled
    with pytest.raises(ValueError, match="capture-point drive"):
        replace(config, lateral_drive_capture_scale_m=0.10)
    lunge = replace(config, runtime_lateral_lunge_blend=0.50)
    assert lunge.runtime_lateral_lunge_blend == 0.50
    with pytest.raises(ValueError, match="lateral lunge scaffold"):
        replace(config, runtime_lateral_lunge_hip_roll_rad=0.31)
    substep_guard = replace(
        config,
        substep_upper_body_guard_enabled=True,
        substep_upper_body_guard_onset_rad_s=1.60,
        substep_upper_body_guard_ceiling_rad_s=2.80,
    )
    assert substep_guard.substep_upper_body_minimum_position_scale == 0.05
    with pytest.raises(ValueError, match="substep upper-body guard"):
        replace(config, substep_upper_body_guard_onset_rad_s=3.10)
    lower_body_guard = replace(
        config,
        substep_option_lower_body_guard_enabled=True,
        substep_option_lower_body_guard_onset_rad_s=2.20,
        substep_option_lower_body_guard_ceiling_rad_s=3.20,
        substep_option_lower_body_minimum_scale=0.20,
    )
    assert lower_body_guard.substep_option_lower_body_minimum_scale == 0.20
    with pytest.raises(ValueError, match="substep option lower-body guard"):
        replace(config, substep_option_lower_body_guard_onset_rad_s=3.40)
    mirrored_locomotion = replace(config, canonical_locomotion_mirror_enabled=True)
    assert mirrored_locomotion.canonical_locomotion_mirror_enabled
    recovery_context = replace(
        config,
        runtime_contact_support_side_enabled=True,
        actor_contact_support_side_enabled=True,
        actor_recovery_context_enabled=True,
    )
    assert recovery_context.actor_recovery_context_enabled
    with pytest.raises(ValueError, match="recovery context requires foot-contact"):
        replace(config, actor_recovery_context_enabled=True)
    official = replace(
        config,
        official_goalkeeper_teacher_checkpoint_path="/tmp/goalkeeper.pt",
        official_goalkeeper_teacher_blend=0.75,
    )
    assert official.official_goalkeeper_teacher_blend == 0.75
    with pytest.raises(ValueError, match="exclusive"):
        replace(official, canonical_locomotion_mirror_enabled=True)
    with pytest.raises(ValueError, match="teacher blend"):
        replace(official, official_goalkeeper_teacher_blend=0.0)
    filtered_official = replace(
        official,
        official_goalkeeper_lower_body_target_step_rad=0.05,
        official_goalkeeper_lower_body_filter_fraction=0.20,
    )
    assert filtered_official.official_goalkeeper_lower_body_target_step_rad == 0.05
    with pytest.raises(ValueError, match="target filter"):
        replace(official, official_goalkeeper_arm_filter_fraction=0.09)
    stratified = replace(config, low_shot_phase_scale=1.30, high_shot_phase_scale=0.70)
    assert stratified.low_shot_phase_scale > stratified.high_shot_phase_scale
    with pytest.raises(ValueError, match="height-conditioned phase"):
        replace(config, high_shot_phase_scale=0.49)
    arm_plastic = replace(config, decoder_arm_residual_authority=0.85)
    assert arm_plastic.resolved_decoder_arm_residual_authority == 0.85
    assert arm_plastic.resolved_decoder_waist_residual_authority == 0.10
    with pytest.raises(ValueError, match="group authority"):
        replace(config, decoder_arm_residual_authority=1.01)
    crouch_plastic = replace(
        config,
        decoder_lower_body_residual_authority=1.0,
        decoder_lower_body_command_scale=0.80,
    )
    assert crouch_plastic.resolved_decoder_lower_body_residual_authority == 1.0
    assert crouch_plastic.resolved_decoder_lower_body_command_scale == 0.80
    with pytest.raises(ValueError, match="lower-body decoder command scale"):
        replace(config, decoder_lower_body_command_scale=0.20)
    whole_body = replace(config, runtime_whole_body_reach_blend=0.65)
    assert whole_body.runtime_whole_body_reach_blend == 0.65
    with pytest.raises(ValueError, match="only one runtime reach teacher"):
        replace(
            config,
            runtime_reach_blend=0.40,
            runtime_whole_body_reach_blend=0.65,
        )


def test_targeted_dive_posture_exception_covers_cue_but_expires_from_release() -> None:
    torch = pytest.importorskip("torch")
    environment = object.__new__(GoalkeeperTargetedDiveMJWarpBatch)
    environment.torch = torch
    environment.count = 1
    environment.device = torch.device("cpu")
    environment.config = SimpleNamespace(control_dt_sec=0.02, first_shot_release_sec=1.0)
    environment.dive_config = GoalkeeperTargetedDiveRLConfig(
        posture_exception_duration_sec=1.55,
        prediction_lead_sec=0.50,
        dive_maximum_root_angular_speed_rad_s=3.5,
    )
    environment._option_started = torch.tensor((True,))
    # The predictive cue started 1.74 s ago, but the causal shot was released
    # only 1.00 s ago.  The landing envelope must therefore remain active.
    environment._option_age_steps = torch.tensor((87,), dtype=torch.long)
    environment._maximum_applied_option_gate = torch.tensor((0.70,))
    environment._maximum_posture_exception_steps = torch.zeros(1, dtype=torch.long)
    environment.qpos = torch.zeros((1, 39))
    environment.qpos[:, 2] = 0.595
    environment.qvel = torch.zeros((1, 38))
    environment.qvel[:, 1] = 2.20
    environment.qvel[:, 5] = 2.34
    environment._step_index = 24

    before_cue = environment._posture_exception_granted(torch.tensor((0.86,)))
    assert not bool(before_cue[0])
    environment._step_index = 25
    during_causal_preparation = environment._posture_exception_granted(torch.tensor((0.86,)))
    assert bool(during_causal_preparation[0])
    environment._step_index = 100

    granted = environment._posture_exception_granted(torch.tensor((0.86,)))

    assert bool(granted[0])
    environment._step_index = 129
    expired = environment._posture_exception_granted(torch.tensor((0.86,)))
    assert not bool(expired[0])


def test_targeted_dive_saved_keeper_can_fall_then_must_recover() -> None:
    torch = pytest.importorskip("torch")
    environment = object.__new__(GoalkeeperTargetedDiveMJWarpBatch)
    environment.torch = torch
    environment.count = 1
    environment.device = torch.device("cpu")
    environment.config = SimpleNamespace(control_dt_sec=0.02, first_shot_release_sec=1.0)
    environment.dive_config = GoalkeeperTargetedDiveRLConfig(
        post_save_counterstep_enabled=True,
        actor_recovery_plasticity_sec=1.0,
        runtime_contact_support_side_enabled=True,
        actor_contact_support_side_enabled=True,
        actor_recovery_context_enabled=True,
        post_save_fall_recovery_enabled=True,
        post_save_fall_recovery_duration_sec=1.0,
    )
    environment.task = SimpleNamespace(first_save=torch.tensor((True,)))
    environment._option_started = torch.tensor((True,))
    environment._option_age_steps = torch.tensor((100,), dtype=torch.long)
    environment._maximum_applied_option_gate = torch.tensor((0.70,))
    environment._maximum_posture_exception_steps = torch.zeros(1, dtype=torch.long)
    environment._post_save_recovery_age_steps = torch.tensor((20,), dtype=torch.long)
    environment.qpos = torch.zeros((1, 39))
    environment.qpos[:, 2] = 0.16
    environment.qvel = torch.zeros((1, 38))
    environment._step_index = 200

    allowed_while_getting_up = environment._posture_exception_granted(torch.tensor((-0.80,)))
    assert bool(allowed_while_getting_up[0])

    environment._post_save_recovery_age_steps.fill_(50)
    expired_without_recovery = environment._posture_exception_granted(torch.tensor((-0.80,)))
    assert not bool(expired_without_recovery[0])

    environment._post_save_recovery_age_steps.fill_(20)
    environment.qpos[:, 2] = 0.05
    outside_finite_recovery_envelope = environment._posture_exception_granted(
        torch.tensor((-0.80,))
    )
    assert not bool(outside_finite_recovery_envelope[0])


def test_targeted_dive_substep_guard_brakes_arms_but_preserves_waist() -> None:
    torch = pytest.importorskip("torch")
    environment = object.__new__(GoalkeeperTargetedDiveMJWarpBatch)
    environment.torch = torch
    environment.count = 2
    environment.device = torch.device("cpu")
    environment.dive_config = GoalkeeperTargetedDiveRLConfig(
        substep_upper_body_guard_enabled=True,
        substep_upper_body_guard_onset_rad_s=1.0,
        substep_upper_body_guard_ceiling_rad_s=2.0,
        substep_upper_body_minimum_position_scale=0.10,
    )
    environment.qvel = torch.zeros((2, 38))
    environment.qvel[1, 3] = 2.0
    environment._minimum_substep_arm_authority = torch.ones(2)

    authority = environment._substep_upper_body_position_authority()

    assert authority.shape == (2, 17)
    assert torch.allclose(authority[:, :3], torch.ones((2, 3)))
    assert torch.allclose(authority[0, 3:], torch.ones(14))
    assert torch.allclose(authority[1, 3:], torch.full((14,), 0.10))
    assert float(environment._minimum_substep_arm_authority.min()) == pytest.approx(0.10)


def test_targeted_dive_counter_rotation_preserves_mujoco_body_frame() -> None:
    torch = pytest.importorskip("torch")
    angular_velocity_body = torch.tensor(((1.0, 2.0, 3.0),))

    counter = _body_frame_waist_counter_rotation(
        torch=torch,
        root_angular_velocity_body_rad_s=angular_velocity_body,
    )
    assert torch.allclose(counter, torch.tensor(((-3.0, -1.0, -2.0),)))


def test_targeted_dive_capture_point_drive_brakes_before_overshoot() -> None:
    torch = pytest.importorskip("torch")
    target = torch.tensor((-1.0, 1.0))
    root = torch.tensor((0.0, 0.0))
    stationary = torch.zeros(2)

    accelerate = _capture_point_lateral_drive(
        torch=torch,
        target_lateral_m=target,
        root_lateral_m=root,
        root_lateral_velocity_mps=stationary,
        target_standoff_m=0.32,
        capture_horizon_sec=0.35,
        capture_scale_m=0.45,
    )
    brake = _capture_point_lateral_drive(
        torch=torch,
        target_lateral_m=target,
        root_lateral_m=torch.tensor((-0.50, 0.50)),
        root_lateral_velocity_mps=torch.tensor((-1.0, 1.0)),
        target_standoff_m=0.32,
        capture_horizon_sec=0.35,
        capture_scale_m=0.45,
    )

    assert torch.allclose(accelerate, torch.tensor((1.0, -1.0)))
    assert float(brake[0]) < 0.0
    assert float(brake[1]) > 0.0
    assert torch.allclose(brake[0], -brake[1])


def test_targeted_dive_substep_guard_sheds_only_learned_lower_body_offset() -> None:
    torch = pytest.importorskip("torch")
    environment = object.__new__(GoalkeeperTargetedDiveMJWarpBatch)
    environment.torch = torch
    environment.count = 2
    environment.dive_config = GoalkeeperTargetedDiveRLConfig(
        substep_option_lower_body_guard_enabled=True,
        substep_option_lower_body_guard_onset_rad_s=1.0,
        substep_option_lower_body_guard_ceiling_rad_s=2.0,
        substep_option_lower_body_minimum_scale=0.25,
    )
    environment.qvel = torch.zeros((2, 38))
    environment.qvel[1, 3] = 2.0
    environment._minimum_substep_option_lower_body_authority = torch.ones(2)
    environment._substep_stable_lower_body_target = torch.full((2, 12), 0.20)
    target = torch.full((2, 29), 0.60)
    target[:, 12:] = 0.90

    guarded = environment._substep_position_target(target)

    assert torch.allclose(guarded[0, :12], torch.full((12,), 0.60))
    assert torch.allclose(guarded[1, :12], torch.full((12,), 0.30))
    assert torch.allclose(guarded[:, 12:], target[:, 12:])
    assert float(environment._minimum_substep_option_lower_body_authority.min()) == pytest.approx(
        0.25
    )


def test_targeted_dive_exam_is_hard_stratified_and_sim_only() -> None:
    config = GoalkeeperTargetedDiveExamConfig(shots_per_stratum=8)
    cases = _sample_cases(config)
    assert len(cases) == 24
    assert {case["stratum"] for case in cases} == {
        "far_corner_low",
        "far_corner_mid",
        "far_corner_high",
    }
    assert all(abs(float(case["target_y_m"])) >= 0.74 for case in cases)
    with pytest.raises(ValueError, match="SIM_ONLY"):
        replace(config, hardware_authorized=True)
    with pytest.raises(ValueError, match="0.002 s"):
        replace(config, physics_substeps=8)
    with pytest.raises(ValueError, match="prediction lead"):
        replace(config, prediction_lead_sec=0.51)
    with pytest.raises(ValueError, match="precedes"):
        replace(config, shot_release_sec=0.20, prediction_lead_sec=0.20)


def test_targeted_dive_ppo_binds_full_body_curriculum(tmp_path) -> None:
    checkpoint = tmp_path / "targeted.pt"
    checkpoint.write_bytes(b"candidate")
    config = GoalkeeperPhysicsPPOConfig(
        targeted_dive_checkpoint=str(checkpoint),
        teacher_pretraining_enabled=False,
        shot_intent_cue_enabled=True,
        targeted_dive_anchor_lower_body_scale=0.35,
        targeted_dive_anchor_waist_scale=0.65,
        targeted_dive_decoder_lower_body_residual_authority=1.0,
        targeted_dive_decoder_lower_body_command_scale=0.75,
        targeted_dive_minimum_option_gate=0.60,
        targeted_dive_runtime_reach_blend=0.85,
        targeted_dive_runtime_reach_feedback_blend=0.50,
        targeted_dive_runtime_reach_feedback_gain=0.80,
        targeted_dive_runtime_reach_feedback_support_scale=0.60,
        targeted_dive_runtime_contact_support_side_enabled=True,
        targeted_dive_actor_contact_support_side_enabled=True,
        targeted_dive_runtime_reach_contact_standoff_m=0.20,
        targeted_dive_runtime_reach_vertical_lead_m=0.18,
        targeted_dive_runtime_reach_low_vertical_lead_m=-0.20,
        targeted_dive_runtime_reach_mid_vertical_lead_m=0.0,
        targeted_dive_runtime_reach_high_vertical_lead_m=0.30,
        targeted_dive_prediction_lead_sec=0.40,
        targeted_dive_maximum_arm_target_step_rad=0.08,
        targeted_dive_arm_target_filter_fraction=0.40,
        targeted_dive_lateral_drive_scale=1.0,
        targeted_dive_substep_upper_body_guard_enabled=True,
        targeted_dive_substep_upper_body_guard_onset_rad_s=1.60,
        targeted_dive_substep_upper_body_guard_ceiling_rad_s=2.80,
        targeted_dive_substep_option_lower_body_guard_enabled=True,
        targeted_dive_substep_option_lower_body_guard_onset_rad_s=2.20,
        targeted_dive_substep_option_lower_body_guard_ceiling_rad_s=3.20,
        targeted_dive_substep_option_lower_body_minimum_scale=0.20,
        training_first_shot_release_sec=0.90,
        training_root_angular_speed_penalty_scale=0.30,
        training_root_angular_speed_soft_limit_rad_s=2.50,
        training_root_angular_speed_excess_penalty_scale=10.0,
        arm_only_online_update=True,
    )
    assert config.targeted_dive_minimum_option_gate == 0.60
    assert config.targeted_dive_runtime_reach_blend == 0.85
    assert config.targeted_dive_runtime_reach_feedback_blend == 0.50
    assert config.targeted_dive_runtime_reach_feedback_support_scale == 0.60
    assert config.targeted_dive_runtime_contact_support_side_enabled
    assert config.targeted_dive_actor_contact_support_side_enabled
    assert config.targeted_dive_runtime_reach_contact_standoff_m == 0.20
    assert config.targeted_dive_runtime_reach_vertical_lead_m == 0.18
    assert config.targeted_dive_runtime_reach_low_vertical_lead_m == -0.20
    assert config.targeted_dive_prediction_lead_sec == 0.40
    assert config.targeted_dive_lateral_drive_scale == 1.0
    assert config.targeted_dive_substep_upper_body_guard_enabled
    assert config.targeted_dive_substep_option_lower_body_guard_enabled
    assert config.training_first_shot_release_sec == 0.90
    assert config.training_root_angular_speed_penalty_scale == 0.30
    assert config.training_root_angular_speed_soft_limit_rad_s == 2.50
    assert config.training_root_angular_speed_excess_penalty_scale == 10.0
    assert config.targeted_dive_decoder_lower_body_residual_authority == 1.0
    with pytest.raises(ValueError, match="actor contact support requires"):
        GoalkeeperPhysicsPPOConfig(
            targeted_dive_actor_contact_support_side_enabled=True,
        )
    assert config.targeted_dive_decoder_lower_body_command_scale == 0.75
    assert config.arm_only_online_update
    whole_body_teacher = replace(
        config,
        targeted_dive_runtime_reach_blend=0.0,
        targeted_dive_runtime_reach_feedback_blend=0.0,
        targeted_dive_runtime_reach_contact_standoff_m=0.0,
        targeted_dive_runtime_reach_vertical_lead_m=0.0,
        targeted_dive_runtime_reach_low_vertical_lead_m=0.0,
        targeted_dive_runtime_reach_mid_vertical_lead_m=0.0,
        targeted_dive_runtime_reach_high_vertical_lead_m=0.0,
        targeted_dive_runtime_whole_body_reach_blend=0.35,
        targeted_dive_runtime_whole_body_reach_waist_scale=0.10,
        targeted_dive_runtime_whole_body_reach_support_scale=0.25,
        targeted_dive_runtime_whole_body_reach_release_sec=0.90,
    )
    assert whole_body_teacher.targeted_dive_runtime_whole_body_reach_blend == 0.35
    assert whole_body_teacher.targeted_dive_runtime_whole_body_reach_release_sec == 0.90
    whole_reach_specialist = replace(
        config,
        arm_only_online_update=False,
        lower_body_and_arms_online_update=True,
    )
    assert whole_reach_specialist.lower_body_and_arms_online_update
    with pytest.raises(ValueError, match="anchor group"):
        replace(config, targeted_dive_anchor_waist_scale=0.20)
    with pytest.raises(ValueError, match="targeted-dive substep guard settings"):
        replace(config, targeted_dive_substep_upper_body_guard_onset_rad_s=3.10)
    with pytest.raises(ValueError, match="option lower-body guard settings"):
        replace(config, targeted_dive_substep_option_lower_body_guard_onset_rad_s=3.40)
    with pytest.raises(ValueError, match="prediction cue precedes"):
        replace(config, training_first_shot_release_sec=0.35)
    with pytest.raises(ValueError, match="root-angular reward penalty"):
        replace(config, training_root_angular_speed_penalty_scale=1.01)
    with pytest.raises(ValueError, match="root-angular tail penalty"):
        replace(config, training_root_angular_speed_excess_penalty_scale=101.0)
    assert _stratum_rates([0.0, 0.30] + [0.0] * 15 + [4.0, 3.0, 1.0, 8.0, 6.0, 2.0]) == (
        0.5,
        0.5,
        0.5,
    )
    assert _stratum_rates([0.0, 0.30] + [0.0] * 15 + [3.0, 0.0, 0.0, 6.0, 0.0, 0.0]) == (
        0.5,
        0.0,
        0.0,
    )
    assert _stratum_balance_score((0.9, 0.8, 0.2)) < _stratum_balance_score((0.6, 0.6, 0.6))
    with pytest.raises(ValueError, match="stratum-balance"):
        replace(config, first_save_stratum_balance_selection_weight=2_001.0)


def test_targeted_dive_gate_specialist_explores_only_drive_channel(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "targeted.pt"
    checkpoint.write_bytes(b"candidate")
    config = GoalkeeperPhysicsPPOConfig(
        targeted_dive_checkpoint=str(checkpoint),
        teacher_pretraining_enabled=False,
        shot_intent_cue_enabled=True,
        training_first_shot_release_sec=0.90,
        targeted_dive_lateral_drive_learned_gate_enabled=True,
        lateral_drive_gate_only_online_update=True,
    )
    mean = torch.arange(60, dtype=torch.float32).reshape(2, 30) / 100.0
    log_std = torch.full((30,), -1.0)

    action, log_probability = _sample_exploration_action(
        torch=torch,
        normal_distribution=torch.distributions.Normal,
        mean=mean,
        log_std=log_std,
        arm_only=False,
        lower_body_and_arms=False,
        lateral_drive_gate_only=config.lateral_drive_gate_only_online_update,
    )

    assert action.shape == mean.shape
    assert log_probability.shape == (2,)
    assert torch.allclose(action[:, 1:], mean[:, 1:])
    assert not torch.allclose(action[:, 0], mean[:, 0])
    with pytest.raises(ValueError, match="mutually exclusive"):
        replace(config, arm_only_online_update=True)


def test_targeted_dive_support_landing_specialist_protects_arm_memory(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "targeted.pt"
    checkpoint.write_bytes(b"candidate")
    config = GoalkeeperPhysicsPPOConfig(
        targeted_dive_checkpoint=str(checkpoint),
        teacher_pretraining_enabled=False,
        shot_intent_cue_enabled=True,
        training_first_shot_release_sec=0.90,
        targeted_dive_lateral_drive_learned_gate_enabled=True,
        support_landing_online_update=True,
    )
    mean = torch.arange(60, dtype=torch.float32).reshape(2, 30) / 100.0
    log_std = torch.full((30,), -1.0)

    action, log_probability = _sample_exploration_action(
        torch=torch,
        normal_distribution=torch.distributions.Normal,
        mean=mean,
        log_std=log_std,
        arm_only=False,
        lower_body_and_arms=False,
        lateral_drive_gate_only=False,
        support_landing=config.support_landing_online_update,
        arm_action_start_index=16,
    )

    assert action.shape == mean.shape
    assert log_probability.shape == (2,)
    assert not torch.allclose(action[:, :16], mean[:, :16])
    assert torch.equal(action[:, 16:], mean[:, 16:])
    with pytest.raises(ValueError, match="mutually exclusive"):
        replace(config, lateral_drive_gate_only_online_update=True)


def test_physics_ppo_first_shot_override_preserves_window_and_episode() -> None:
    world = GoalkeeperMJWarpConfig(
        first_shot_release_sec=0.70,
        first_shot_end_sec=1.70,
        second_shot_release_sec=3.05,
    )
    shifted = _with_first_shot_release_override(world, 0.90)
    assert shifted.first_shot_release_sec == 0.90
    assert shifted.first_shot_end_sec == pytest.approx(1.90)
    assert world.first_shot_release_sec == 0.70
    penalized = _with_root_angular_penalty_override(world, 0.30, 2.50, 10.0)
    assert penalized.root_angular_speed_penalty_scale == 0.30
    assert penalized.root_angular_speed_soft_limit_rad_s == 2.50
    assert penalized.root_angular_speed_excess_penalty_scale == 10.0
    save_aligned = _with_save_event_bonus_override(world, 180.0, 120.0)
    assert save_aligned.true_save_bonus == 180.0
    assert save_aligned.hand_save_bonus == 120.0
    assert world.true_save_bonus == 25.0
    assert world.root_angular_speed_penalty_scale != 0.30


def test_targeted_dive_phase_scheduler_preserves_endpoints_and_intercept() -> None:
    assert _scheduled_phase(raw_phase=0.0, arrival_raw_phase=0.45, intercept_phase=0.72) == 0.0
    assert _scheduled_phase(raw_phase=0.45, arrival_raw_phase=0.45, intercept_phase=0.72) == 0.72
    assert _scheduled_phase(raw_phase=1.0, arrival_raw_phase=0.45, intercept_phase=0.72) == 1.0
    values = [
        _scheduled_phase(raw_phase=value, arrival_raw_phase=0.45, intercept_phase=0.72)
        for value in np.linspace(0.0, 1.0, 21)
    ]
    assert np.all(np.diff(values) > 0.0)
    assert _scheduled_phase(raw_phase=0.37, arrival_raw_phase=0.45, intercept_phase=None) == 0.37


def test_targeted_dive_numpy_and_torch_features_match() -> None:
    torch = pytest.importorskip("torch")
    values = _features()
    expected = targeted_dive_features_numpy(**values)
    actual = targeted_dive_features_torch(
        torch=torch,
        **{key: torch.as_tensor(value, dtype=torch.float64) for key, value in values.items()},
    )
    assert expected.shape == (2, TARGETED_DIVE_INPUT_SIZE)
    assert np.allclose(expected, actual.numpy(), atol=1.0e-12, rtol=0.0)
    assert not np.allclose(expected[0], expected[1])


def test_targeted_dive_decoder_preserves_anchor_for_zero_residual() -> None:
    torch = pytest.importorskip("torch")
    from torch import nn

    model = build_targeted_dive_decoder(torch, nn, hidden_size=64)
    for parameter in model.parameters():
        nn.init.zeros_(parameter)
    features = torch.as_tensor(targeted_dive_features_numpy(**_features()), dtype=torch.float32)
    seed = torch.arange(2 * 71 * 29, dtype=torch.float32).reshape(2, 71, 29) / 10_000
    checkpoint = {
        "imitation_seed_joint_position": seed,
        "target_scale": torch.ones(29),
    }
    decoded = decode_goalkeeper_targeted_dive(model=model, checkpoint=checkpoint, features=features)
    expected = torch.stack((seed[0, 17], seed[1, 52]))
    expected += torch.stack(
        (
            (seed[0, 17 + 1] - seed[0, 17]) * 0.5,
            (seed[1, 52 + 1] - seed[1, 52]) * 0.5,
        )
    )
    assert torch.allclose(decoded, expected, atol=1.0e-6, rtol=0.0)


def test_targeted_dive_decoder_applies_per_joint_authority() -> None:
    torch = pytest.importorskip("torch")
    from torch import nn

    model = build_targeted_dive_decoder(torch, nn, hidden_size=64)
    for parameter in model.parameters():
        nn.init.zeros_(parameter)
    nn.init.ones_(model[-1].bias)
    features = torch.as_tensor(targeted_dive_features_numpy(**_features()), dtype=torch.float32)
    checkpoint = {
        "imitation_seed_joint_position": torch.zeros((2, 71, 29)),
        "target_scale": torch.ones(29),
    }
    authority = torch.linspace(0.0, 1.0, 29)
    decoded = decode_goalkeeper_targeted_dive(
        model=model,
        checkpoint=checkpoint,
        features=features,
        residual_authority=authority,
    )
    assert torch.allclose(decoded, authority.expand(2, -1), atol=1.0e-6, rtol=0.0)
    with pytest.raises(ValueError, match="invalid shape"):
        decode_goalkeeper_targeted_dive(
            model=model,
            checkpoint=checkpoint,
            features=features,
            residual_authority=torch.ones(28),
        )


def test_targeted_dive_decoder_enforces_structural_mirror_equivariance() -> None:
    torch = pytest.importorskip("torch")
    from torch import nn

    torch.manual_seed(4201)
    model = build_targeted_dive_decoder(torch, nn, hidden_size=64)
    features = torch.as_tensor(targeted_dive_features_numpy(**_features())[:1], dtype=torch.float32)
    checkpoint = {
        "imitation_seed_joint_position": torch.zeros((2, 71, 29)),
        "target_scale": torch.linspace(0.1, 0.6, 29),
        "decoder_symmetry_enforcement": "MIRROR_CANONICAL_HALF_SPACE_V1",
    }
    # Training makes the paired joint scales equal; reproduce that checkpoint
    # invariant explicitly in this unit-level fixture.
    order = torch.as_tensor(
        (
            6,
            7,
            8,
            9,
            10,
            11,
            0,
            1,
            2,
            3,
            4,
            5,
            12,
            13,
            14,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
        ),
        dtype=torch.long,
    )
    checkpoint["target_scale"] = torch.maximum(
        checkpoint["target_scale"], checkpoint["target_scale"][order]
    )
    decoded = decode_goalkeeper_targeted_dive(
        model=model,
        checkpoint=checkpoint,
        features=features,
    )
    mirrored = decode_goalkeeper_targeted_dive(
        model=model,
        checkpoint=checkpoint,
        features=_mirror_features_torch(features),
    )
    assert torch.allclose(
        mirrored,
        _mirror_joints_torch(torch, decoded),
        atol=2.0e-7,
        rtol=0.0,
    )


def _monitor_step(
    monitor: GoalkeeperTorchDiveMonitor,
    *,
    request: bool,
    threat: int = 1,
    visible: bool = True,
    lateral: float = 0.85,
    pelvis: float = 0.79,
    upright: float = 1.0,
    linear: float = 0.0,
    angular: float = 0.0,
    landing: bool = False,
    forbidden: bool = False,
):
    torch = monitor.torch
    return monitor.step(
        option_request=torch.tensor((request,)),
        threat_id=torch.tensor((threat,), dtype=torch.long),
        threat_visible=torch.tensor((visible,)),
        lateral_intercept_error_m=torch.tensor((lateral,)),
        pelvis_height_m=torch.tensor((pelvis,)),
        upright_projection=torch.tensor((upright,)),
        root_linear_speed_mps=torch.tensor((linear,)),
        root_angular_speed_rad_s=torch.tensor((angular,)),
        permitted_landing_contact=torch.tensor((landing,)),
        forbidden_body_contact=torch.tensor((forbidden,)),
    )


def test_torch_monitor_accepts_predictive_threat_and_requires_recovery() -> None:
    torch = pytest.importorskip("torch")
    config = GoalkeeperDiveOptionConfig(control_dt_sec=0.05, recovery_hold_sec=0.10)
    monitor = GoalkeeperTorchDiveMonitor(
        torch=torch, environment_count=1, device=torch.device("cpu"), config=config
    )
    result = _monitor_step(monitor, request=True)
    assert bool(result["option_started_event"][0])
    assert int(result["phase"][0]) == GoalkeeperDivePhase.TAKEOFF
    flight = _monitor_step(
        monitor,
        request=False,
        pelvis=0.48,
        upright=0.45,
        linear=1.0,
        angular=2.0,
    )
    assert int(flight["phase"][0]) == GoalkeeperDivePhase.FLIGHT
    assert bool(flight["posture_exception_granted"][0])
    _monitor_step(
        monitor,
        request=False,
        pelvis=0.42,
        upright=0.35,
        linear=0.6,
        angular=1.0,
        landing=True,
    )
    _monitor_step(monitor, request=False)
    recovered = _monitor_step(monitor, request=False)
    assert bool(recovered["recovered_event"][0])
    assert int(monitor.phase[0]) == GoalkeeperDivePhase.READY


def test_torch_monitor_fails_closed_on_nonfinite_or_changed_threat() -> None:
    torch = pytest.importorskip("torch")
    monitor = GoalkeeperTorchDiveMonitor(
        torch=torch, environment_count=1, device=torch.device("cpu")
    )
    _monitor_step(monitor, request=True)
    failed = _monitor_step(monitor, request=False, pelvis=float("nan"))
    assert bool(failed["unsafe"][0])
    monitor.reset()
    _monitor_step(monitor, request=True)
    changed = _monitor_step(
        monitor,
        request=False,
        threat=2,
        pelvis=0.48,
        upright=0.45,
        linear=1.0,
        angular=2.0,
    )
    assert bool(changed["unsafe"][0])
