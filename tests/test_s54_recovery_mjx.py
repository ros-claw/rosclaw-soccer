from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.recovery_mjx import (
    RecoveryMJXFailureConstraintConfig,
    RecoveryMJXPPOConfig,
    RecoveryMJXProbeConfig,
    RecoveryMJXTeacherResidualPPOConfig,
    build_recovery_mjx_directional_curriculum,
    build_recovery_mjx_failure_window_plan,
    select_recovery_mjx_failure_constrained_generation,
    select_recovery_mjx_teacher_residual_generation,
    validate_recovery_mjx_action_distribution_audit,
    validate_recovery_mjx_directional_curriculum,
    validate_recovery_mjx_failure_constrained_selection_report,
    validate_recovery_mjx_failure_state_exam_report,
    validate_recovery_mjx_failure_state_manifest,
    validate_recovery_mjx_failure_window_plan,
    validate_recovery_mjx_probe_report,
    validate_recovery_mjx_teacher_residual_report,
)
from rosclaw_soccer.training.recovery_mjx_routes import (
    RecoveryMJXRouteManifestConfig,
    build_recovery_mjx_route_manifest,
    resolve_recovery_mjx_route_group,
    validate_recovery_mjx_route_manifest,
)
from rosclaw_soccer.training.recovery_snapshot import (
    RecoverySnapshot,
    write_recovery_snapshot_corpus,
)
from rosclaw_soccer.training.recovery_teacher_bridge import (
    RecoveryBridgeTrial,
    RecoveryEntryMatch,
    build_recovery_bridge_schedule,
)


def test_mjx_probe_contract_is_modern_parallel_and_sim_only() -> None:
    config = RecoveryMJXProbeConfig(
        batch_size_per_gpu=8,
        benchmark_control_steps=4,
        parity_snapshot_count=2,
    )
    assert config.required_gpu_count == 4
    assert config.substeps == 10
    assert config.snapshot_scene_equivalent is False
    assert config.activation_ceiling == "SIM_ONLY"
    assert config.hardware_authorized is False
    with pytest.raises(ValueError, match="invalid"):
        RecoveryMJXProbeConfig(snapshot_scene_equivalent=True)
    with pytest.raises(ValueError, match="invalid"):
        RecoveryMJXProbeConfig(hardware_authorized=True)
    with pytest.raises(ValueError, match="invalid"):
        RecoveryMJXProbeConfig(control_dt_sec=0.019)


def test_mjx_probe_report_validation_fails_closed(tmp_path: Path) -> None:
    report = {
        "schema_version": "rosclaw_soccer.recovery_mjx_probe_report.v1",
        "rollout_backend": "MUJOCO_MJX",
        "physics_truth_backend": "CPU_MUJOCO",
        "snapshot_scene_equivalent": False,
        "promotion_authority": "NONE",
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "probe_passed": True,
    }
    report["report_hash"] = hash_json(report)
    path = tmp_path / "probe.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    loaded = validate_recovery_mjx_probe_report(path)
    assert loaded["probe_passed"] is True

    tampered = dict(report)
    tampered["promotion_authority"] = "HARDWARE"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid"):
        validate_recovery_mjx_probe_report(path)


def test_mjx_probe_config_hash_binds_every_field() -> None:
    original = RecoveryMJXProbeConfig()
    changed = RecoveryMJXProbeConfig(batch_size_per_gpu=64)
    assert original.config_hash == hash_json(asdict(original))
    assert original.config_hash != changed.config_hash


def test_mjx_ppo_contract_is_four_gpu_residual_and_sim_only() -> None:
    config = RecoveryMJXPPOConfig(
        total_timesteps=65_536,
        num_envs=64,
        batch_size=16,
        num_minibatches=4,
        num_eval_envs=8,
    )
    assert config.num_envs % 4 == 0
    assert config.residual_limits_rad.shape == (29,)
    assert config.residual_limits_rad[:12] == pytest.approx([0.15] * 12)
    assert config.residual_limits_rad[12:15] == pytest.approx([0.12] * 3)
    assert config.residual_limits_rad[15:] == pytest.approx([0.18] * 14)
    assert config.activation_ceiling == "SIM_ONLY"
    with pytest.raises(ValueError, match="invalid"):
        RecoveryMJXPPOConfig(hardware_authorized=True)
    with pytest.raises(ValueError, match="invalid"):
        RecoveryMJXPPOConfig(num_envs=66)


def test_mjx_teacher_residual_contract_requires_momentum_gated_handoff() -> None:
    config = RecoveryMJXTeacherResidualPPOConfig(
        total_timesteps=65_536,
        num_envs=64,
        batch_size=16,
        num_minibatches=4,
        num_eval_envs=8,
    )
    assert config.required_gpu_count == 4
    assert config.handoff_stable_steps == 10
    assert config.success_stable_steps == 100
    assert config.handoff_maximum_linear_speed_mps == pytest.approx(0.5)
    assert config.handoff_maximum_angular_speed_rad_s == pytest.approx(1.5)
    assert config.ready_momentum_penalty_scale == pytest.approx(0.15)
    assert config.directional_momentum_penalty_scale == pytest.approx(0.0)
    assert config.failure_state_directional_penalty_scale == pytest.approx(0.0)
    assert config.failure_state_backward_cost_weight == pytest.approx(3.0)
    assert config.failure_state_lateral_cost_weight == pytest.approx(0.25)
    assert config.failure_state_yaw_cost_weight == pytest.approx(0.10)
    assert config.failure_state_directional_cost_mode == "ALWAYS_ON_PSEUDO_HUBER"
    assert config.failure_state_stable_streak_reward_scale == pytest.approx(0.0)
    assert config.stable_streak_reward_scale == pytest.approx(1.0)
    assert config.handoff_regression_penalty_scale == pytest.approx(1.0)
    assert config.proprioception_history_frames == 4
    assert config.use_pelvis_imu_observation is True
    assert config.use_base_velocity_estimate_observation is False
    assert config.base_velocity_estimate_clip_mps == pytest.approx(2.0)
    assert config.actor_proprioception_frame_dim == 96
    assert config.pelvis_accelerometer_clip_mps2 == pytest.approx(40.0)
    assert config.terminal_balance_reset_fraction == pytest.approx(0.0)
    assert config.failure_state_reset_fraction == pytest.approx(0.0)
    assert config.terminate_failure_state_episode_at_target_horizon is False
    assert config.failure_state_target_horizon_steps == 400
    assert config.use_asymmetric_critic is True
    assert config.failure_state_conditioned_critic is False
    assert config.posture_gated_residual is True
    assert config.residual_limits_rad.shape == (29,)
    assert config.activation_ceiling == "SIM_ONLY"
    assert config.hardware_authorized is False
    with pytest.raises(ValueError, match="invalid"):
        RecoveryMJXTeacherResidualPPOConfig(handoff_stable_steps=1)
    with pytest.raises(ValueError, match="invalid"):
        RecoveryMJXTeacherResidualPPOConfig(hardware_authorized=True)
    with pytest.raises(ValueError, match="invalid"):
        RecoveryMJXTeacherResidualPPOConfig(stable_streak_reward_scale=0.0)
    with pytest.raises(ValueError, match="invalid"):
        RecoveryMJXTeacherResidualPPOConfig(directional_momentum_penalty_scale=2.1)
    with pytest.raises(ValueError, match="invalid"):
        RecoveryMJXTeacherResidualPPOConfig(failure_state_directional_penalty_scale=0.5)
    with pytest.raises(ValueError, match="invalid"):
        RecoveryMJXTeacherResidualPPOConfig(failure_state_backward_cost_weight=0.0)
    with pytest.raises(ValueError, match="invalid"):
        RecoveryMJXTeacherResidualPPOConfig(failure_state_lateral_cost_weight=10.1)
    with pytest.raises(ValueError, match="invalid"):
        RecoveryMJXTeacherResidualPPOConfig(failure_state_directional_cost_mode="CLIPPED")
    with pytest.raises(ValueError, match="invalid"):
        RecoveryMJXTeacherResidualPPOConfig(
            failure_state_reset_fraction=0.25,
            failure_state_directional_penalty_scale=0.5,
        )
    targeted = RecoveryMJXTeacherResidualPPOConfig(
        failure_state_reset_fraction=0.25,
        failure_state_directional_penalty_scale=0.5,
        failure_state_stable_streak_reward_scale=1.0,
        failure_state_conditioned_critic=True,
    )
    assert targeted.failure_state_conditioned_critic is True
    failure_specialist = RecoveryMJXTeacherResidualPPOConfig(
        failure_state_reset_fraction=0.9,
        terminate_failure_state_episode_at_target_horizon=True,
    )
    assert failure_specialist.failure_state_reset_fraction == pytest.approx(0.9)
    assert failure_specialist.expected_failure_target_transition_fraction == pytest.approx(0.9)
    legacy_duration_mixture = RecoveryMJXTeacherResidualPPOConfig(
        failure_state_reset_fraction=0.9,
        terminate_failure_state_episode_at_target_horizon=True,
        schema_version="rosclaw_soccer.recovery_mjx_teacher_residual_ppo_config.v14",
    )
    assert legacy_duration_mixture.expected_failure_target_transition_fraction == pytest.approx(
        0.75
    )
    mixed_long_episode = RecoveryMJXTeacherResidualPPOConfig(
        failure_state_reset_fraction=0.5,
    )
    assert mixed_long_episode.expected_failure_target_transition_fraction == pytest.approx(1 / 6)
    with pytest.raises(ValueError, match="invalid"):
        RecoveryMJXTeacherResidualPPOConfig(
            terminate_failure_state_episode_at_target_horizon=True,
        )
    with pytest.raises(ValueError, match="invalid"):
        RecoveryMJXTeacherResidualPPOConfig(
            total_timesteps=65_536,
            failure_state_reset_fraction=0.9,
            terminate_failure_state_episode_at_target_horizon=True,
        )
    with pytest.raises(ValueError, match="invalid"):
        RecoveryMJXTeacherResidualPPOConfig(failure_state_reset_fraction=0.91)
    with pytest.raises(ValueError, match="invalid"):
        RecoveryMJXTeacherResidualPPOConfig(
            failure_state_reset_fraction=0.25,
            failure_state_conditioned_critic=True,
            use_asymmetric_critic=False,
        )
    with pytest.raises(ValueError, match="invalid"):
        RecoveryMJXTeacherResidualPPOConfig(proprioception_history_frames=17)
    with pytest.raises(ValueError, match="invalid"):
        RecoveryMJXTeacherResidualPPOConfig(pelvis_accelerometer_clip_mps2=9.0)
    with pytest.raises(ValueError, match="invalid"):
        RecoveryMJXTeacherResidualPPOConfig(
            use_pelvis_imu_observation=False,
            use_base_velocity_estimate_observation=True,
        )
    with pytest.raises(ValueError, match="invalid"):
        RecoveryMJXTeacherResidualPPOConfig(base_velocity_estimate_clip_mps=0.1)
    appended_velocity = RecoveryMJXTeacherResidualPPOConfig(
        use_base_velocity_estimate_observation=True,
        preserve_pelvis_accelerometer_observation=True,
        regularize_velocity_adapter_only=True,
    )
    assert appended_velocity.actor_proprioception_frame_dim == 99
    assert appended_velocity.regularize_velocity_adapter_only is True
    with pytest.raises(ValueError, match="invalid"):
        RecoveryMJXTeacherResidualPPOConfig(
            preserve_pelvis_accelerometer_observation=True,
        )
    with pytest.raises(ValueError, match="invalid"):
        RecoveryMJXTeacherResidualPPOConfig(
            use_base_velocity_estimate_observation=True,
            preserve_pelvis_accelerometer_observation=True,
            use_asymmetric_critic=False,
        )
    with pytest.raises(ValueError, match="invalid"):
        RecoveryMJXTeacherResidualPPOConfig(regularize_velocity_adapter_only=True)
    with pytest.raises(ValueError, match="invalid"):
        RecoveryMJXTeacherResidualPPOConfig(failure_state_target_horizon_steps=99)
    with pytest.raises(ValueError, match="invalid"):
        RecoveryMJXTeacherResidualPPOConfig(terminal_balance_reset_fraction=0.95)
    with pytest.raises(ValueError, match="invalid"):
        RecoveryMJXTeacherResidualPPOConfig(
            terminal_balance_reset_fraction=0.1,
            failure_state_reset_fraction=0.1,
        )
    with pytest.raises(ValueError, match="invalid"):
        RecoveryMJXTeacherResidualPPOConfig(required_gpu_count=0)


def test_mjx_checkpoint_velocity_expansion_preserves_parent_behavior() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    running_statistics = pytest.importorskip("brax.training.acme.running_statistics")
    from rosclaw_soccer.training.opentrack_recovery_mjx_ppo import (
        _make_recovery_ppo_networks,
    )
    from rosclaw_soccer.training.opentrack_recovery_mjx_teacher_ppo import (
        _migrate_checkpoint_with_appended_velocity,
    )

    old_actor_dim = 4 * 96
    networks = _make_recovery_ppo_networks(
        {"state": old_actor_dim, "privileged_state": old_actor_dim + 8},
        29,
        running_statistics.normalize,
    )
    policy_key, value_key = jax.random.split(jax.random.PRNGKey(54))
    normalizer = running_statistics.init_state(
        {
            "state": jnp.zeros((old_actor_dim,), dtype=jnp.float32),
            "privileged_state": jnp.zeros((old_actor_dim + 8,), dtype=jnp.float32),
        }
    )
    params = [
        normalizer,
        networks.policy_network.init(policy_key),
        networks.value_network.init(value_key),
    ]
    migrated, contract = _migrate_checkpoint_with_appended_velocity(params, history_frames=4)
    assert migrated[0].mean["state"].shape == (396,)
    assert migrated[0].mean["privileged_state"].shape == (404,)
    migrated_policy_kernel = migrated[1]["params"]["Dense_0"]["kernel"]
    original_policy_kernel = params[1]["params"]["Dense_0"]["kernel"]
    assert migrated_policy_kernel.shape == (384, 256)
    assert migrated[2]["params"]["hidden_0"]["kernel"].shape == (404, 256)
    np.testing.assert_array_equal(migrated_policy_kernel, original_policy_kernel)
    adapter_output = migrated[1]["params"]["velocity_adapter_location"]
    np.testing.assert_array_equal(adapter_output["kernel"], np.zeros((64, 29), dtype=np.float32))
    np.testing.assert_array_equal(adapter_output["bias"], np.zeros((29,), dtype=np.float32))
    assert contract["legacy_accelerometer_slots_preserved"] is True
    assert contract["parent_policy_trunk_frozen"] is True
    assert contract["parent_observation_normalizer_frozen"] is True
    assert contract["parent_normalizer_drift_invariance_max_abs_error"] <= 1.0e-6
    assert contract["new_adapter_output_weights_initialized_to_zero"] is True
    assert contract["parent_policy_gradient_max_abs"] == 0.0
    assert contract["adapter_output_gradient_l2"] > 0.0
    assert contract["behavior_preserved"] is True
    assert contract["policy_output_max_abs_error"] <= 1.0e-6
    assert contract["value_output_max_abs_error"] <= 1.0e-6


def test_mjx_teacher_residual_report_cannot_claim_deployment(tmp_path: Path) -> None:
    report = {
        "schema_version": "rosclaw_soccer.recovery_mjx_teacher_residual_ppo_report.v1",
        "config": {"required_gpu_count": 4, "num_eval_envs": 8},
        "devices": ["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
        "rollout_backend": "MUJOCO_MJX",
        "physics_truth_backend": "CPU_MUJOCO",
        "teacher_frozen": True,
        "actor_observation": "DEPLOYABLE_PROPRIOCEPTION_ONLY",
        "continued_from_parent": False,
        "parent_checkpoint_hash": None,
        "deployment_candidate": False,
        "requires_reference_free_distillation": True,
        "requires_independent_cpu_mujoco_exam": True,
        "promotion_eligible": False,
        "promotion_authority": "NONE",
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    path = tmp_path / "teacher-residual.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert validate_recovery_mjx_teacher_residual_report(path)["teacher_frozen"] is True

    tampered = dict(report)
    tampered["deployment_candidate"] = True
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid"):
        validate_recovery_mjx_teacher_residual_report(path)

    history = dict(report)
    history.update(
        config={
            "required_gpu_count": 4,
            "proprioception_history_frames": 4,
            "use_asymmetric_critic": True,
        },
        actor_observation="DEPLOYABLE_PROPRIOCEPTION_HISTORY_ONLY",
        actor_observation_dim=372,
        critic_observation="SIMULATION_PRIVILEGED_VALUE_FUNCTION_ONLY",
        critic_privileged_auxiliary_dim=8,
        critic_exported_with_actor=False,
    )
    history.pop("report_hash")
    history["report_hash"] = hash_json(history)
    path.write_text(json.dumps(history), encoding="utf-8")
    assert validate_recovery_mjx_teacher_residual_report(path)["actor_observation_dim"] == 372

    history["actor_observation_dim"] = 371
    history.pop("report_hash")
    history["report_hash"] = hash_json(history)
    path.write_text(json.dumps(history), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid"):
        validate_recovery_mjx_teacher_residual_report(path)

    pelvis_imu = dict(report)
    pelvis_imu.update(
        config={
            "required_gpu_count": 4,
            "proprioception_history_frames": 4,
            "use_pelvis_imu_observation": True,
            "pelvis_accelerometer_clip_mps2": 40.0,
            "use_asymmetric_critic": True,
        },
        actor_observation="DEPLOYABLE_PELVIS_IMU_PROPRIOCEPTION_HISTORY_ONLY",
        actor_observation_dim=384,
        actor_proprioception_frame_dim=96,
        actor_pelvis_imu_contract={
            "accelerometer_clip_mps2": 40.0,
            "accelerometer_sensor": "accelerometer_pelvis",
            "accelerometer_scale": "CLIP_THEN_DIVIDE_BY_CLIP",
            "gyroscope_scale": 0.05,
            "gyroscope_sensor": "gyro_pelvis",
        },
        critic_observation="SIMULATION_PRIVILEGED_VALUE_FUNCTION_ONLY",
        critic_privileged_auxiliary_dim=8,
        critic_exported_with_actor=False,
    )
    pelvis_imu.pop("report_hash")
    pelvis_imu["report_hash"] = hash_json(pelvis_imu)
    path.write_text(json.dumps(pelvis_imu), encoding="utf-8")
    assert validate_recovery_mjx_teacher_residual_report(path)["actor_observation_dim"] == 384

    diagnostics = json.loads(json.dumps(pelvis_imu))
    diagnostics["signed_velocity_diagnostics"] = {
        "actor_observation_features": [],
        "angular_frame": "PELVIS_IMU_SENSOR_FRAME",
        "linear_frame": "PELVIS_BODY_FRAME",
        "metrics": [
            "root_body_forward_velocity",
            "root_body_lateral_velocity",
            "root_body_vertical_velocity",
            "root_body_backward_speed",
            "root_body_lateral_speed",
            "pelvis_roll_rate",
            "pelvis_pitch_rate",
            "pelvis_yaw_rate",
            "pelvis_yaw_speed",
        ],
        "temporal_bin_count": 6,
        "temporal_bin_semantics": "EQUAL_WIDTH_BY_WRAPPER_CONTROL_STEP",
        "temporal_metrics": [
            "root_body_backward_speed",
            "root_body_lateral_speed",
            "pelvis_yaw_speed",
        ],
        "use": "EVALUATION_AND_CURRICULUM_DIAGNOSTICS_ONLY",
    }
    diagnostics.pop("report_hash")
    diagnostics["report_hash"] = hash_json(diagnostics)
    path.write_text(json.dumps(diagnostics), encoding="utf-8")
    validated = validate_recovery_mjx_teacher_residual_report(path)
    assert validated["signed_velocity_diagnostics"]["actor_observation_features"] == []

    diagnostics["signed_velocity_diagnostics"]["actor_observation_features"] = [
        "root_body_forward_velocity"
    ]
    diagnostics.pop("report_hash")
    diagnostics["report_hash"] = hash_json(diagnostics)
    path.write_text(json.dumps(diagnostics), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid"):
        validate_recovery_mjx_teacher_residual_report(path)

    curriculum = json.loads(json.dumps(pelvis_imu))
    curriculum["config"].update(
        terminal_balance_reset_fraction=0.5,
        terminal_balance_root_linear_velocity_noise_mps=0.2,
        terminal_balance_root_angular_velocity_noise_rad_s=0.5,
    )
    curriculum["terminal_balance_curriculum"] = {
        "evaluation_reset_fraction": 0.0,
        "reference_frame": 6476,
        "root_angular_velocity_noise_rad_s": 0.5,
        "root_linear_velocity_noise_mps": 0.2,
        "training_reset_fraction": 0.5,
    }
    curriculum.pop("report_hash")
    curriculum["report_hash"] = hash_json(curriculum)
    path.write_text(json.dumps(curriculum), encoding="utf-8")
    assert (
        validate_recovery_mjx_teacher_residual_report(path)["terminal_balance_curriculum"][
            "evaluation_reset_fraction"
        ]
        == 0.0
    )

    curriculum["terminal_balance_curriculum"]["evaluation_reset_fraction"] = 0.5
    curriculum.pop("report_hash")
    curriculum["report_hash"] = hash_json(curriculum)
    path.write_text(json.dumps(curriculum), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid"):
        validate_recovery_mjx_teacher_residual_report(path)

    failure_context = json.loads(json.dumps(pelvis_imu))
    failure_context["config"]["failure_state_reset_fraction"] = 0.1
    failure_context["failure_state_curriculum"] = {
        "evaluation_reset_fraction": 0.0,
        "training_reset_fraction": 0.1,
        "failure_state_manifest_hash": "sha256:" + "1" * 64,
        "failure_state_manifest_file_hash": "sha256:" + "2" * 64,
        "failure_state_archive_hash": "sha256:" + "3" * 64,
        "failure_state_count": 96,
        "context_features_restored": [
            "qpos",
            "qvel",
            "trajectory_step",
            "trajectory_initial_step",
            "handoff_frozen",
            "last_motor_targets",
            "last_teacher_action",
            "last_residual",
            "proprioception_history",
            "phase_repeat",
        ],
    }
    failure_context.pop("report_hash")
    failure_context["report_hash"] = hash_json(failure_context)
    path.write_text(json.dumps(failure_context), encoding="utf-8")
    validated_failure = validate_recovery_mjx_teacher_residual_report(path)
    assert validated_failure["failure_state_curriculum"]["training_reset_fraction"] == 0.1

    targeted_failure = json.loads(json.dumps(failure_context))
    targeted_failure["config"].update(
        failure_state_directional_penalty_scale=0.5,
        failure_state_stable_streak_reward_scale=1.0,
        failure_state_conditioned_critic=True,
    )
    targeted_failure["critic_privileged_features"] = [
        "root_body_linear_velocity",
        "pelvis_angular_velocity",
        "pelvis_height",
        "failure_state_reset_source",
    ]
    targeted_failure["failure_state_targeted_reward"] = {
        "actor_observation_features": [],
        "critic_failure_source_indicator": True,
        "directional_penalty_scale": 0.5,
        "evaluation_active": False,
        "stable_streak_reward_scale": 1.0,
        "training_scope": "FAILURE_STATE_RESET_EPISODES_ONLY",
    }
    targeted_failure.pop("report_hash")
    targeted_failure["report_hash"] = hash_json(targeted_failure)
    path.write_text(json.dumps(targeted_failure), encoding="utf-8")
    assert (
        validate_recovery_mjx_teacher_residual_report(path)["failure_state_targeted_reward"][
            "evaluation_active"
        ]
        is False
    )

    targeted_failure["failure_state_targeted_reward"]["evaluation_active"] = True
    targeted_failure.pop("report_hash")
    targeted_failure["report_hash"] = hash_json(targeted_failure)
    path.write_text(json.dumps(targeted_failure), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid"):
        validate_recovery_mjx_teacher_residual_report(path)

    estimated_velocity = json.loads(json.dumps(targeted_failure))
    estimated_velocity["config"].update(
        schema_version="rosclaw_soccer.recovery_mjx_teacher_residual_ppo_config.v10",
        use_base_velocity_estimate_observation=True,
        base_velocity_estimate_clip_mps=2.0,
        failure_state_target_horizon_steps=400,
    )
    estimated_velocity["actor_observation"] = (
        "DEPLOYABLE_ESTIMATED_BASE_VELOCITY_PROPRIOCEPTION_HISTORY_ONLY"
    )
    estimated_velocity["actor_pelvis_imu_contract"] = {
        "linear_motion_feature": "BASE_VELOCITY_ESTIMATE",
        "linear_motion_scale": "CLIP_THEN_DIVIDE_BY_CLIP",
        "gyroscope_scale": 0.05,
        "gyroscope_sensor": "gyro_pelvis",
    }
    estimated_velocity["actor_base_velocity_estimator_contract"] = {
        "clip_mps": 2.0,
        "deployment_source_required": "ONBOARD_STATE_ESTIMATOR",
        "ground_truth_hardware_velocity_authorized": False,
        "simulation_proxy": "MUJOCO_ROOT_QVEL_ROTATED_TO_PELVIS",
    }
    estimated_velocity["failure_state_targeted_reward"].update(
        directional_cost_weights={"backward": 3.0, "lateral": 0.25, "yaw": 0.10},
        target_horizon_steps=400,
        evaluation_active=False,
    )
    estimated_velocity["failure_state_curriculum"]["observation_context_adapter"] = (
        "BASE_VELOCITY_ESTIMATE_REPLACES_ACCELEROMETER_CHANNELS_6_TO_8"
    )
    estimated_velocity.pop("report_hash")
    estimated_velocity["report_hash"] = hash_json(estimated_velocity)
    path.write_text(json.dumps(estimated_velocity), encoding="utf-8")
    assert (
        validate_recovery_mjx_teacher_residual_report(path)[
            "actor_base_velocity_estimator_contract"
        ]["deployment_source_required"]
        == "ONBOARD_STATE_ESTIMATOR"
    )

    appended_velocity = json.loads(json.dumps(estimated_velocity))
    appended_velocity["config"].update(
        schema_version="rosclaw_soccer.recovery_mjx_teacher_residual_ppo_config.v11",
        preserve_pelvis_accelerometer_observation=True,
    )
    appended_velocity.update(
        actor_observation=(
            "DEPLOYABLE_PELVIS_IMU_AND_ESTIMATED_BASE_VELOCITY_PROPRIOCEPTION_HISTORY_ONLY"
        ),
        actor_observation_dim=396,
        actor_proprioception_frame_dim=99,
        continued_from_parent=True,
        parent_checkpoint_hash="sha256:" + "9" * 64,
    )
    appended_velocity["actor_pelvis_imu_contract"].update(
        accelerometer_clip_mps2=40.0,
        accelerometer_sensor="accelerometer_pelvis",
        accelerometer_scale="CLIP_THEN_DIVIDE_BY_CLIP",
    )
    appended_velocity["actor_observation_migration"] = {
        "strategy": "FROZEN_PARENT_TRUNK_ZERO_OUTPUT_VELOCITY_ADAPTER",
        "source_actor_observation_dim": 384,
        "target_actor_observation_dim": 396,
        "source_frame_dim": 96,
        "target_frame_dim": 99,
        "appended_feature": "ONBOARD_BASE_VELOCITY_ESTIMATE",
        "legacy_accelerometer_slots_preserved": True,
        "parent_policy_trunk_frozen": True,
        "new_adapter_output_weights_initialized_to_zero": True,
        "adapter_location_limit": 0.25,
        "parent_policy_gradient_max_abs": 0.0,
        "adapter_output_gradient_l2": 1.0,
        "new_value_input_weights_initialized_to_zero": True,
        "policy_output_max_abs_error": 1.0e-7,
        "value_output_max_abs_error": 2.0e-7,
        "behavior_preserved": True,
    }
    appended_velocity["failure_state_curriculum"]["observation_context_adapter"] = (
        "BASE_VELOCITY_ESTIMATE_APPENDED_AFTER_PRESERVED_ACCELEROMETER_CHANNELS_6_TO_8"
    )
    appended_velocity.pop("report_hash")
    appended_velocity["report_hash"] = hash_json(appended_velocity)
    path.write_text(json.dumps(appended_velocity), encoding="utf-8")
    assert validate_recovery_mjx_teacher_residual_report(path)["actor_observation_dim"] == 396

    fully_frozen = json.loads(json.dumps(appended_velocity))
    fully_frozen["config"].update(
        schema_version="rosclaw_soccer.recovery_mjx_teacher_residual_ppo_config.v12",
        failure_state_backward_cost_weight=3.0,
        failure_state_lateral_cost_weight=1.0,
        failure_state_yaw_cost_weight=0.5,
    )
    fully_frozen["failure_state_targeted_reward"]["directional_cost_weights"] = {
        "backward": 3.0,
        "lateral": 1.0,
        "yaw": 0.5,
    }
    fully_frozen["actor_observation_migration"].update(
        parent_observation_normalizer_frozen=True,
        parent_normalizer_drift_invariance_max_abs_error=0.0,
    )
    frozen_hash = "sha256:" + "a" * 64
    fully_frozen["candidate_checkpoint_files"] = [
        {"path": "000000000000/params", "size_bytes": 1, "hash": "sha256:" + "b" * 64}
    ]
    fully_frozen["parent_actor_retention"] = {
        "schema_version": "rosclaw_soccer.frozen_parent_actor_retention.v1",
        "source_frozen_state_hash": frozen_hash,
        "frozen_components": [
            "DENSE_TRUNK",
            "ACTION_LOCATION_HEAD",
            "ACTION_SCALE_HEAD",
            "OBSERVATION_NORMALIZER_MEAN",
            "OBSERVATION_NORMALIZER_STD",
        ],
        "checkpoint_results": [
            {
                "step": 0,
                "frozen_state_hash": frozen_hash,
                "exact_equal": True,
                "maximum_absolute_error": 0.0,
            }
        ],
        "all_checkpoints_exact": True,
    }
    fully_frozen.pop("report_hash")
    fully_frozen["report_hash"] = hash_json(fully_frozen)
    path.write_text(json.dumps(fully_frozen), encoding="utf-8")
    assert (
        validate_recovery_mjx_teacher_residual_report(path)["parent_actor_retention"][
            "all_checkpoints_exact"
        ]
        is True
    )

    adapter_only = json.loads(json.dumps(fully_frozen))
    adapter_only["config"].update(
        schema_version="rosclaw_soccer.recovery_mjx_teacher_residual_ppo_config.v13",
        regularize_velocity_adapter_only=True,
    )
    adapter_only["residual_regularization_target"] = "VELOCITY_ADAPTER_INCREMENT_ONLY"
    adapter_only.pop("report_hash")
    adapter_only["report_hash"] = hash_json(adapter_only)
    path.write_text(json.dumps(adapter_only), encoding="utf-8")
    assert (
        validate_recovery_mjx_teacher_residual_report(path)["residual_regularization_target"]
        == "VELOCITY_ADAPTER_INCREMENT_ONLY"
    )

    adapter_only["residual_regularization_target"] = "TOTAL_TEACHER_RESIDUAL"
    adapter_only.pop("report_hash")
    adapter_only["report_hash"] = hash_json(adapter_only)
    path.write_text(json.dumps(adapter_only), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid"):
        validate_recovery_mjx_teacher_residual_report(path)

    fully_frozen["parent_actor_retention"]["checkpoint_results"][0]["exact_equal"] = False
    fully_frozen.pop("report_hash")
    fully_frozen["report_hash"] = hash_json(fully_frozen)
    path.write_text(json.dumps(fully_frozen), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid"):
        validate_recovery_mjx_teacher_residual_report(path)

    appended_velocity["actor_observation_migration"][
        "new_adapter_output_weights_initialized_to_zero"
    ] = False
    appended_velocity.pop("report_hash")
    appended_velocity["report_hash"] = hash_json(appended_velocity)
    path.write_text(json.dumps(appended_velocity), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid"):
        validate_recovery_mjx_teacher_residual_report(path)

    failure_context["failure_state_curriculum"]["context_features_restored"].pop()
    failure_context.pop("report_hash")
    failure_context["report_hash"] = hash_json(failure_context)
    path.write_text(json.dumps(failure_context), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid"):
        validate_recovery_mjx_teacher_residual_report(path)

    pelvis_imu["actor_pelvis_imu_contract"]["accelerometer_sensor"] = "simulated_qvel"
    pelvis_imu.pop("report_hash")
    pelvis_imu["report_hash"] = hash_json(pelvis_imu)
    path.write_text(json.dumps(pelvis_imu), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid"):
        validate_recovery_mjx_teacher_residual_report(path)

    unbound_parent = dict(report)
    unbound_parent.update(
        continued_from_parent=True,
        parent_checkpoint_hash="sha256:" + "1" * 64,
        route_binding_enforced=True,
        route_manifest_hash="sha256:" + "2" * 64,
        route_group_hash="sha256:" + "3" * 64,
    )
    unbound_parent["report_hash"] = hash_json(unbound_parent)
    path.write_text(json.dumps(unbound_parent), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid"):
        validate_recovery_mjx_teacher_residual_report(path)


def test_mjx_directional_curriculum_is_derived_from_bound_signed_failure(
    tmp_path: Path,
) -> None:
    signed_contract = {
        "actor_observation_features": [],
        "angular_frame": "PELVIS_IMU_SENSOR_FRAME",
        "linear_frame": "PELVIS_BODY_FRAME",
        "metrics": [
            "root_body_forward_velocity",
            "root_body_lateral_velocity",
            "root_body_vertical_velocity",
            "root_body_backward_speed",
            "root_body_lateral_speed",
            "pelvis_roll_rate",
            "pelvis_pitch_rate",
            "pelvis_yaw_rate",
            "pelvis_yaw_speed",
        ],
        "temporal_bin_count": 6,
        "temporal_bin_semantics": "EQUAL_WIDTH_BY_WRAPPER_CONTROL_STEP",
        "temporal_metrics": [
            "root_body_backward_speed",
            "root_body_lateral_speed",
            "pelvis_yaw_speed",
        ],
        "use": "EVALUATION_AND_CURRICULUM_DIAGNOSTICS_ONLY",
    }
    source = {
        "schema_version": "rosclaw_soccer.recovery_mjx_teacher_residual_ppo_report.v1",
        "config": {"required_gpu_count": 4, "num_eval_envs": 8},
        "devices": ["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
        "rollout_backend": "MUJOCO_MJX",
        "physics_truth_backend": "CPU_MUJOCO",
        "teacher_frozen": True,
        "actor_observation": "DEPLOYABLE_PROPRIOCEPTION_ONLY",
        "signed_velocity_diagnostics": signed_contract,
        "continued_from_parent": True,
        "parent_checkpoint_hash": "sha256:" + "3" * 64,
        "parent_training_report_hash": "sha256:" + "4" * 64,
        "route_binding_enforced": True,
        "route_manifest_hash": "sha256:" + "1" * 64,
        "route_group_hash": "sha256:" + "2" * 64,
        "progress": [
            {
                "step": 0,
                "metrics": {
                    "eval/avg_episode_length": 1000.0,
                    "eval/episode_root_body_forward_velocity": -250.0,
                    "eval/episode_root_body_lateral_velocity": 80.0,
                    "eval/episode_root_body_vertical_velocity": 5.0,
                    "eval/episode_pelvis_yaw_rate": 900.0,
                    "eval/episode_pelvis_yaw_speed": 950.0,
                },
            }
        ],
        "deployment_candidate": False,
        "requires_reference_free_distillation": True,
        "requires_independent_cpu_mujoco_exam": True,
        "promotion_eligible": False,
        "promotion_authority": "NONE",
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "hardware_command_sent": False,
    }
    for index, (backward, lateral, yaw) in enumerate(
        zip(
            (10.0, 30.0, 50.0, 50.0, 50.0, 50.0),
            (10.0, 20.0, 20.0, 20.0, 20.0, 20.0),
            (20.0, 40.0, 60.0, 60.0, 60.0, 60.0),
            strict=True,
        )
    ):
        metrics = source["progress"][0]["metrics"]
        metrics[f"eval/episode_root_body_backward_speed_phase_{index}"] = backward
        metrics[f"eval/episode_root_body_lateral_speed_phase_{index}"] = lateral
        metrics[f"eval/episode_pelvis_yaw_speed_phase_{index}"] = yaw
    source["report_hash"] = hash_json(source)
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    output_path = tmp_path / "curriculum.json"
    result = build_recovery_mjx_directional_curriculum(
        training_report_path=source_path,
        output_path=output_path,
    )
    assert result["terminal_body_linear_velocity_bias_mps"] == pytest.approx([-0.25, 0.08, 0.0])
    assert result["terminal_pelvis_yaw_rate_bias_rad_s"] == pytest.approx(0.9)
    assert result["source_evaluation_step"] == 0
    assert (
        validate_recovery_mjx_directional_curriculum(output_path)["report_hash"]
        == result["report_hash"]
    )
    with pytest.raises(ValueError, match="overwrite"):
        build_recovery_mjx_directional_curriculum(
            training_report_path=source_path,
            output_path=output_path,
        )

    window_path = tmp_path / "failure-windows.json"
    windows = build_recovery_mjx_failure_window_plan(
        training_report_path=source_path,
        output_path=window_path,
    )
    assert windows["selected_window_indices"] == [1, 2]
    assert windows["requested_collection_steps"] == [166, 249, 332, 333, 416, 499]
    assert (
        validate_recovery_mjx_failure_window_plan(window_path)["report_hash"]
        == windows["report_hash"]
    )

    tampered = json.loads(output_path.read_text(encoding="utf-8"))
    tampered["terminal_body_linear_velocity_bias_mps"][0] = 0.1
    tampered.pop("report_hash")
    tampered["report_hash"] = hash_json(tampered)
    output_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid"):
        validate_recovery_mjx_directional_curriculum(output_path)


def test_mjx_failure_state_manifest_binds_policy_and_proprioceptive_context(
    tmp_path: Path,
) -> None:
    count = 2
    archive_path = tmp_path / "failure-window-states.npz"
    np.savez_compressed(
        archive_path,
        qpos=np.zeros((count, 36), dtype=np.float32),
        qvel=np.zeros((count, 35), dtype=np.float32),
        control_step=np.asarray([200, 300], dtype=np.int32),
        environment_index=np.asarray([0, 1], dtype=np.int32),
        handoff_frozen=np.asarray([False, True], dtype=np.bool_),
        trajectory_step=np.asarray([6431, 6476], dtype=np.int32),
        trajectory_initial_step=np.asarray([6230, 6230], dtype=np.int32),
        root_body_backward_speed_mps=np.asarray([0.0, 0.39], dtype=np.float32),
        root_body_lateral_speed_mps=np.asarray([0.47, 0.06], dtype=np.float32),
        pelvis_yaw_speed_rad_s=np.asarray([0.45, 1.17], dtype=np.float32),
        last_motor_targets=np.zeros((count, 29), dtype=np.float32),
        last_teacher_action=np.zeros((count, 29), dtype=np.float32),
        last_residual=np.zeros((count, 29), dtype=np.float32),
        proprioception_history=np.zeros((count, 4, 96), dtype=np.float32),
        phase_repeat=np.zeros((count,), dtype=np.int32),
    )
    digest = "sha256:" + "a" * 64
    manifest = {
        "schema_version": "rosclaw_soccer.recovery_mjx_failure_state_manifest.v2",
        "config": {"num_environments": 4},
        "source_failure_window_plan_hash": digest,
        "source_failure_window_plan_file_hash": digest,
        "source_training_report_hash": digest,
        "source_actor_checkpoint_hash": digest,
        "source_actor_config_hash": digest,
        "source_route_manifest_hash": digest,
        "source_route_group_hash": digest,
        "teacher_checkpoint_hash": digest,
        "motion_archive_hash": digest,
        "snapshot_manifest_hash": digest,
        "compiled_model_contract": {},
        "rollout_backend": "MUJOCO_MJX",
        "physics_truth_backend": "CPU_MUJOCO_REQUIRED_FOR_PROMOTION",
        "deterministic_actor": True,
        "full_route_reset": True,
        "requested_collection_steps": [200, 300],
        "collected_state_count": count,
        "state_archive": archive_path.name,
        "state_archive_hash": hash_bytes(archive_path.read_bytes()),
        "qpos_shape": [count, 36],
        "qvel_shape": [count, 35],
        "proprioception_history_shape": [count, 4, 96],
        "context_features_collected": [
            "qpos",
            "qvel",
            "trajectory_step",
            "trajectory_initial_step",
            "handoff_frozen",
            "last_motor_targets",
            "last_teacher_action",
            "last_residual",
            "proprioception_history",
            "phase_repeat",
        ],
        "curriculum_use_only": True,
        "deployment_candidate": False,
        "promotion_eligible": False,
        "promotion_authority": "NONE",
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "hardware_command_sent": False,
    }
    manifest["report_hash"] = hash_json(manifest)
    manifest_path = tmp_path / "failure-state-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    validated = validate_recovery_mjx_failure_state_manifest(manifest_path)
    assert validated["proprioception_history_shape"] == [2, 4, 96]

    with np.load(archive_path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    arrays["phase_repeat"][0] = 4
    np.savez_compressed(archive_path, **arrays)
    manifest["state_archive_hash"] = hash_bytes(archive_path.read_bytes())
    manifest.pop("report_hash")
    manifest["report_hash"] = hash_json(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid"):
        validate_recovery_mjx_failure_state_manifest(manifest_path)


def test_mjx_failure_state_exam_recomputes_local_retention_gates(tmp_path: Path) -> None:
    digest = "sha256:" + "a" * 64

    def metrics(*, episode_count: int, candidate: bool) -> dict[str, float | int]:
        return {
            "episode_count": episode_count,
            "mean_episode_length": 400.0,
            "success_rate": 0.1,
            "stable_fraction": 0.21 if candidate else 0.2,
            "ready_fraction": 0.1,
            "mean_maximum_stable_streak": 20.0,
            "root_body_backward_speed_mps": 0.45 if candidate else 0.5,
            "root_body_lateral_speed_mps": 0.202 if candidate else 0.2,
            "pelvis_yaw_speed_rad_s": 1.01 if candidate else 1.0,
            "mean_reward_per_step": 1.0,
            "non_success_termination_rate": 0.1,
        }

    parent = metrics(episode_count=96, candidate=False)
    candidate = metrics(episode_count=96, candidate=True)
    report = {
        "schema_version": "rosclaw_soccer.recovery_mjx_failure_state_exam_report.v2",
        "config": {
            "num_environments": 96,
            "horizon_steps": 400,
            "random_seed": 5480,
            "minimum_state_coverage_fraction": 0.95,
            "minimum_stable_improvement_fraction": 0.02,
            "maximum_streak_regression_fraction": 0.02,
            "minimum_backward_speed_improvement_fraction": 0.02,
            "maximum_lateral_speed_regression_fraction": 0.02,
            "maximum_yaw_speed_regression_fraction": 0.02,
            "activation_ceiling": "SIM_ONLY",
            "hardware_authorized": False,
            "schema_version": "rosclaw_soccer.recovery_mjx_failure_state_exam_config.v1",
        },
        "failure_state_manifest_hash": digest,
        "failure_state_manifest_file_hash": digest,
        "failure_state_archive_hash": digest,
        "parent_training_report_hash": digest,
        "parent_checkpoint_hash": digest,
        "parent_actor_observation": "DEPLOYABLE_PELVIS_IMU_PROPRIOCEPTION_HISTORY_ONLY",
        "parent_actor_observation_dim": 384,
        "parent_training_checkpoint_tree_hash": digest,
        "candidate_training_report_hash": digest,
        "candidate_checkpoint_hash": digest,
        "candidate_actor_observation": "DEPLOYABLE_PELVIS_IMU_PROPRIOCEPTION_HISTORY_ONLY",
        "candidate_actor_observation_dim": 384,
        "candidate_training_checkpoint_tree_hash": digest,
        "route_manifest_hash": digest,
        "route_group_hash": digest,
        "teacher_checkpoint_hash": digest,
        "rollout_backend": "MUJOCO_MJX",
        "physics_truth_backend": "CPU_MUJOCO_REQUIRED_FOR_PROMOTION",
        "paired_identical_reset_keys": True,
        "paired_reset_key_strategy": "DETERMINISTIC_STRATIFIED_FULL_FAILURE_BANK_COVERAGE",
        "diagnostic_failure_state_reset_fraction": 1.0,
        "observed_failure_state_reset_fraction": 1.0,
        "state_coverage": {
            "unique_state_count": 96,
            "total_state_count": 96,
            "coverage_fraction": 1.0,
            "required_control_steps": [200, 300],
            "covered_control_steps": [200, 300],
        },
        "parent_metrics": parent,
        "candidate_metrics": candidate,
        "relative_changes": {
            "stable_fraction": 0.05,
            "maximum_stable_streak": 0.0,
            "root_body_backward_speed": -0.1,
            "root_body_lateral_speed": 0.01,
            "pelvis_yaw_speed": 0.01,
        },
        "per_failure_window": [
            {
                "control_step": step,
                "episode_count": 48,
                "parent_metrics": metrics(episode_count=48, candidate=False),
                "candidate_metrics": metrics(episode_count=48, candidate=True),
            }
            for step in (200, 300)
        ],
        "retention_gates": {
            "coverage_passed": True,
            "success_or_stable_passed": True,
            "maximum_streak_passed": True,
            "backward_speed_passed": True,
            "lateral_speed_passed": True,
            "yaw_speed_passed": True,
            "termination_safety_passed": True,
        },
        "local_retention_passed": True,
        "deployment_candidate": False,
        "promotion_eligible": False,
        "promotion_authority": "NONE",
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    path = tmp_path / "failure-state-exam.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert validate_recovery_mjx_failure_state_exam_report(path)["local_retention_passed"] is True

    migrated_exam = json.loads(json.dumps(report))
    migrated_exam.update(
        schema_version="rosclaw_soccer.recovery_mjx_failure_state_exam_report.v3",
        candidate_actor_observation=(
            "DEPLOYABLE_PELVIS_IMU_AND_ESTIMATED_BASE_VELOCITY_PROPRIOCEPTION_HISTORY_ONLY"
        ),
        candidate_actor_observation_dim=396,
        snapshot_manifest_hash=digest,
        parent_snapshot_manifest_hash=digest,
        candidate_snapshot_manifest_hash=digest,
        failure_state_snapshot_manifest_hash=digest,
        actor_observation_migration={
            "strategy": "FROZEN_PARENT_TRUNK_ZERO_OUTPUT_VELOCITY_ADAPTER",
            "source_actor_observation_dim": 384,
            "target_actor_observation_dim": 396,
            "source_frame_dim": 96,
            "target_frame_dim": 99,
            "appended_feature": "ONBOARD_BASE_VELOCITY_ESTIMATE",
            "legacy_accelerometer_slots_preserved": True,
            "parent_policy_trunk_frozen": True,
            "new_adapter_output_weights_initialized_to_zero": True,
            "adapter_location_limit": 0.25,
            "parent_policy_gradient_max_abs": 0.0,
            "adapter_output_gradient_l2": 1.0,
            "new_value_input_weights_initialized_to_zero": True,
            "policy_output_max_abs_error": 1.0e-7,
            "value_output_max_abs_error": 2.0e-7,
            "behavior_preserved": True,
        },
    )
    migrated_exam.pop("report_hash")
    migrated_exam["report_hash"] = hash_json(migrated_exam)
    path.write_text(json.dumps(migrated_exam), encoding="utf-8")
    assert (
        validate_recovery_mjx_failure_state_exam_report(path)["candidate_actor_observation_dim"]
        == 396
    )

    authority_exam = json.loads(json.dumps(migrated_exam))
    authority_exam.update(
        schema_version="rosclaw_soccer.recovery_mjx_failure_state_exam_report.v4",
        candidate_action_authority={
            "baseline": "FROZEN_PARENT_RESIDUAL_POLICY",
            "normalized_action_space": "CANDIDATE_MINUS_BASELINE_RESIDUAL",
            "motor_target_space": "CLIPPED_CANDIDATE_MINUS_CLIPPED_BASELINE_RAD",
            "frozen_parent_baseline": True,
            "mean_normalized_action_increment_rms": 0.01,
            "maximum_normalized_action_increment_rms": 0.05,
            "mean_motor_target_increment_rms_rad": 0.002,
            "maximum_motor_target_increment_rms_rad": 0.01,
            "residual_active_fraction": 1.0,
        },
    )
    authority_exam.pop("report_hash")
    authority_exam["report_hash"] = hash_json(authority_exam)
    path.write_text(json.dumps(authority_exam), encoding="utf-8")
    validated_authority = validate_recovery_mjx_failure_state_exam_report(path)
    assert validated_authority["candidate_action_authority"][
        "mean_motor_target_increment_rms_rad"
    ] == pytest.approx(0.002)

    authority_exam["candidate_action_authority"][
        "maximum_motor_target_increment_rms_rad"
    ] = 0.001
    authority_exam.pop("report_hash")
    authority_exam["report_hash"] = hash_json(authority_exam)
    path.write_text(json.dumps(authority_exam), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid"):
        validate_recovery_mjx_failure_state_exam_report(path)

    report["retention_gates"]["backward_speed_passed"] = False
    report.pop("report_hash")
    report["report_hash"] = hash_json(report)
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid"):
        validate_recovery_mjx_failure_state_exam_report(path)


def test_mjx_action_distribution_audit_normalizes_episode_sums(tmp_path: Path) -> None:
    digest = "sha256:" + "a" * 64
    normal_normalized = 0.2655 / 1_200.0
    normal_motor = 0.0207 / 1_200.0
    failure_normalized = 0.000245
    failure_motor = 0.0000192
    report = {
        "schema_version": "rosclaw_soccer.recovery_mjx_action_distribution_audit.v1",
        "training_report_hash": digest,
        "failure_state_exam_report_hash": digest,
        "candidate_training_checkpoint_tree_hash": digest,
        "normal_route_action_authority": {
            "evaluation_step": 262_144,
            "average_episode_length": 1_200.0,
            "mean_normalized_action_increment_rms": normal_normalized,
            "mean_motor_target_increment_rms_rad": normal_motor,
        },
        "failure_state_action_authority": {
            "mean_normalized_action_increment_rms": failure_normalized,
            "mean_motor_target_increment_rms_rad": failure_motor,
        },
        "failure_to_normal_authority_ratio": {
            "mean_normalized_action_increment_rms": (
                failure_normalized / normal_normalized
            ),
            "mean_motor_target_increment_rms_rad": failure_motor / normal_motor,
        },
        "minimum_effective_normalized_action_rms": 0.001,
        "diagnosis": "GLOBAL_LOW_ACTION_AUTHORITY",
        "deployment_candidate": False,
        "promotion_eligible": False,
        "promotion_authority": "NONE",
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    path = tmp_path / "action-distribution-audit.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    validated = validate_recovery_mjx_action_distribution_audit(path)
    assert validated["diagnosis"] == "GLOBAL_LOW_ACTION_AUTHORITY"
    assert validated["failure_to_normal_authority_ratio"][
        "mean_normalized_action_increment_rms"
    ] == pytest.approx(1.1073446327683616)

    report["failure_to_normal_authority_ratio"][
        "mean_normalized_action_increment_rms"
    ] = 0.01
    report.pop("report_hash")
    report["report_hash"] = hash_json(report)
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid"):
        validate_recovery_mjx_action_distribution_audit(path)


def test_mjx_generation_selection_keeps_best_non_regressing_step(
    tmp_path: Path,
) -> None:
    def metrics(
        *,
        stable: float,
        linear: float,
        angular: float,
        reward: float,
        residual: float,
        maximum_streak: float,
        episode_length: float = 1200.0,
        success: float = 0.0,
    ) -> dict[str, float]:
        return {
            "eval/avg_episode_length": episode_length,
            "eval/episode_reward": reward,
            "eval/episode_root_angular_speed": angular,
            "eval/episode_root_linear_speed": linear,
            "eval/episode_stable": stable,
            "eval/episode_success": success,
            "eval/episode_residual_rms": residual,
            "eval/episode_maximum_stable_streak": maximum_streak,
            "eval/episode_root_body_backward_speed": linear * 0.6,
            "eval/episode_root_body_lateral_speed": linear * 0.25,
            "eval/episode_pelvis_yaw_rate": angular * 0.7,
        }

    report = {
        "schema_version": "rosclaw_soccer.recovery_mjx_teacher_residual_ppo_report.v1",
        "config": {"required_gpu_count": 4, "num_eval_envs": 8},
        "devices": ["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
        "rollout_backend": "MUJOCO_MJX",
        "physics_truth_backend": "CPU_MUJOCO",
        "teacher_frozen": True,
        "actor_observation": "DEPLOYABLE_PROPRIOCEPTION_ONLY",
        "continued_from_parent": False,
        "parent_checkpoint_hash": None,
        "candidate_checkpoint_hash": "sha256:" + "e" * 64,
        "candidate_checkpoint_files": [
            {"path": "000000131072/weights", "size_bytes": 10, "hash": "sha256:a"},
            {"path": "000000262144/weights", "size_bytes": 10, "hash": "sha256:b"},
            {"path": "000000393216/weights", "size_bytes": 10, "hash": "sha256:c"},
            {"path": "000000524288/weights", "size_bytes": 10, "hash": "sha256:d"},
        ],
        "progress": [
            {
                "step": 0,
                "metrics": metrics(
                    stable=760.0,
                    linear=536.0,
                    angular=1358.0,
                    reward=675.0,
                    residual=0.0,
                    maximum_streak=50.0,
                ),
            },
            {
                "step": 131072,
                "metrics": metrics(
                    stable=797.0,
                    linear=546.0,
                    angular=1296.0,
                    reward=702.0,
                    residual=10.1,
                    maximum_streak=55.0,
                ),
            },
            {
                "step": 262144,
                "metrics": metrics(
                    stable=723.0,
                    linear=562.0,
                    angular=1298.0,
                    reward=646.0,
                    residual=11.6,
                    maximum_streak=45.0,
                ),
            },
            {
                "step": 393216,
                "metrics": metrics(
                    # Only +1.05% normalized stable rate: strict success may
                    # replace the 2% gain floor, but never a stable regression.
                    stable=640.0,
                    linear=430.0,
                    angular=1100.0,
                    reward=600.0,
                    residual=8.0,
                    maximum_streak=50.5,
                    episode_length=1000.0,
                    success=0.125,
                ),
            },
            {
                "step": 524288,
                "metrics": metrics(
                    stable=620.0,
                    linear=430.0,
                    angular=1050.0,
                    reward=650.0,
                    residual=7.0,
                    maximum_streak=60.0,
                    episode_length=1000.0,
                    success=0.25,
                ),
            },
        ],
        "deployment_candidate": False,
        "requires_reference_free_distillation": True,
        "requires_independent_cpu_mujoco_exam": True,
        "promotion_eligible": False,
        "promotion_authority": "NONE",
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    training_path = tmp_path / "training.json"
    training_path.write_text(json.dumps(report), encoding="utf-8")
    selection = select_recovery_mjx_teacher_residual_generation(
        training_report_path=training_path,
        output_path=tmp_path / "selection.json",
    )
    assert selection["selected_step"] == 393216
    assert selection["development_retention_passed"] is True
    assert selection["strict_success_observed"] is True
    assert selection["maximum_stable_streak_enforced"] is True
    assert selection["directional_retention_enforced"] is True
    assert selection["evaluation_environment_count"] == 8
    assert selection["evaluations"][-2]["estimated_successful_evaluation_episodes"] == 1.0
    assert selection["evaluations"][-2]["stable_evidence_passed"] is True
    assert selection["evaluations"][-2]["directional_retention_passed"] is True
    assert selection["evaluations"][-1]["stable_evidence_passed"] is False
    assert selection["evaluations"][-1]["retention_passed"] is False
    assert selection["promotion_eligible"] is False
    assert selection["evaluations"][1]["retention_passed"] is False
    assert selection["selected_checkpoint_files"][0]["path"].startswith("000000393216/")

    digest = "sha256:" + "a" * 64
    parent_metrics = {
        "episode_count": 96,
        "mean_episode_length": 400.0,
        "success_rate": 0.1,
        "stable_fraction": 0.2,
        "ready_fraction": 0.9,
        "mean_maximum_stable_streak": 20.0,
        "root_body_backward_speed_mps": 0.5,
        "root_body_lateral_speed_mps": 0.2,
        "pelvis_yaw_speed_rad_s": 1.0,
        "mean_reward_per_step": 1.0,
        "non_success_termination_rate": 0.1,
    }

    def write_exam(step: int, *, local_passed: bool, suffix: str = "") -> Path:
        candidate_metrics = dict(parent_metrics)
        if local_passed:
            candidate_metrics.update(
                stable_fraction=0.21,
                root_body_backward_speed_mps=0.45,
                root_body_lateral_speed_mps=0.202,
                pelvis_yaw_speed_rad_s=1.01,
            )
        relative = {
            "stable_fraction": 0.05 if local_passed else 0.0,
            "maximum_stable_streak": 0.0,
            "root_body_backward_speed": -0.1 if local_passed else 0.0,
            "root_body_lateral_speed": 0.01 if local_passed else 0.0,
            "pelvis_yaw_speed": 0.01 if local_passed else 0.0,
        }
        source_file = next(
            row
            for row in report["candidate_checkpoint_files"]
            if str(row["path"]).startswith(f"{step:012d}/")
        )
        checkpoint_file = dict(source_file)
        checkpoint_file["path"] = str(source_file["path"]).split("/", maxsplit=1)[1]
        gates = {
            "coverage_passed": True,
            "success_or_stable_passed": local_passed,
            "maximum_streak_passed": True,
            "backward_speed_passed": local_passed,
            "lateral_speed_passed": True,
            "yaw_speed_passed": True,
            "termination_safety_passed": True,
        }
        exam = {
            "schema_version": "rosclaw_soccer.recovery_mjx_failure_state_exam_report.v2",
            "config": {
                "num_environments": 96,
                "horizon_steps": 400,
                "random_seed": 5492,
                "minimum_state_coverage_fraction": 0.95,
                "minimum_stable_improvement_fraction": 0.02,
                "maximum_streak_regression_fraction": 0.02,
                "minimum_backward_speed_improvement_fraction": 0.02,
                "maximum_lateral_speed_regression_fraction": 0.02,
                "maximum_yaw_speed_regression_fraction": 0.02,
                "activation_ceiling": "SIM_ONLY",
                "hardware_authorized": False,
                "schema_version": (
                    "rosclaw_soccer.recovery_mjx_failure_state_exam_config.v1"
                ),
            },
            "failure_state_manifest_hash": digest,
            "failure_state_manifest_file_hash": digest,
            "failure_state_archive_hash": digest,
            "parent_training_report_hash": digest,
            "parent_checkpoint_hash": digest,
            "parent_actor_observation": "DEPLOYABLE_PROPRIOCEPTION_HISTORY_ONLY",
            "parent_actor_observation_dim": 384,
            "parent_checkpoint_files": [],
            "parent_training_checkpoint_tree_hash": digest,
            "candidate_training_report_hash": report["report_hash"],
            "candidate_checkpoint_hash": hash_json([checkpoint_file]),
            "candidate_actor_observation": "DEPLOYABLE_PROPRIOCEPTION_HISTORY_ONLY",
            "candidate_actor_observation_dim": 384,
            "candidate_checkpoint_files": [checkpoint_file],
            "candidate_training_checkpoint_tree_hash": report["candidate_checkpoint_hash"],
            "route_manifest_hash": digest,
            "route_group_hash": digest,
            "teacher_checkpoint_hash": digest,
            "teacher_checkpoint_files": [],
            "motion_archive_hash": digest,
            "rollout_backend": "MUJOCO_MJX",
            "physics_truth_backend": "CPU_MUJOCO_REQUIRED_FOR_PROMOTION",
            "paired_identical_reset_keys": True,
            "paired_reset_key_strategy": (
                "DETERMINISTIC_STRATIFIED_FULL_FAILURE_BANK_COVERAGE"
            ),
            "diagnostic_failure_state_reset_fraction": 1.0,
            "observed_failure_state_reset_fraction": 1.0,
            "state_coverage": {
                "unique_state_count": 96,
                "total_state_count": 96,
                "coverage_fraction": 1.0,
                "required_control_steps": [200, 300],
                "covered_control_steps": [200, 300],
            },
            "parent_metrics": parent_metrics,
            "candidate_metrics": candidate_metrics,
            "relative_changes": relative,
            "per_failure_window": [
                {
                    "control_step": control_step,
                    "episode_count": 48,
                    "parent_metrics": {**parent_metrics, "episode_count": 48},
                    "candidate_metrics": {**candidate_metrics, "episode_count": 48},
                }
                for control_step in (200, 300)
            ],
            "retention_gates": gates,
            "local_retention_passed": local_passed,
            "deployment_candidate": False,
            "promotion_eligible": False,
            "promotion_authority": "NONE",
            "activation_ceiling": "SIM_ONLY",
            "hardware_authorized": False,
            "hardware_command_sent": False,
        }
        exam["report_hash"] = hash_json(exam)
        path = tmp_path / f"exam-{step}{suffix}.json"
        path.write_text(json.dumps(exam), encoding="utf-8")
        return path

    steps = (131072, 262144, 393216, 524288)
    failing_exams = tuple(write_exam(step, local_passed=False) for step in steps)
    constrained = select_recovery_mjx_failure_constrained_generation(
        training_report_path=training_path,
        generation_selection_path=tmp_path / "selection.json",
        failure_state_exam_paths=failing_exams,
        output_path=tmp_path / "constrained-failed.json",
    )
    assert constrained["selected_step"] is None
    assert constrained["development_retention_passed"] is False
    assert constrained["all_persisted_checkpoints_examined"] is True
    assert constrained["updated_effective_multipliers"]["stable_streak"] > 0.0
    assert constrained["updated_effective_multipliers"]["backward"] > 0.1
    assert constrained["promotion_eligible"] is False

    with pytest.raises(ValueError, match="every persisted checkpoint"):
        select_recovery_mjx_failure_constrained_generation(
            training_report_path=training_path,
            generation_selection_path=tmp_path / "selection.json",
            failure_state_exam_paths=failing_exams[:-1],
            output_path=tmp_path / "constrained-missing.json",
        )

    passing_exam = write_exam(393216, local_passed=True, suffix="-passing")
    joint_exams = tuple(
        passing_exam if step == 393216 else path
        for step, path in zip(steps, failing_exams, strict=True)
    )
    jointly_selected = select_recovery_mjx_failure_constrained_generation(
        training_report_path=training_path,
        generation_selection_path=tmp_path / "selection.json",
        failure_state_exam_paths=joint_exams,
        output_path=tmp_path / "constrained-passed.json",
        config=RecoveryMJXFailureConstraintConfig(dual_learning_rate=0.25),
    )
    assert jointly_selected["selected_step"] == 393216
    assert jointly_selected["development_retention_passed"] is True
    validated = validate_recovery_mjx_failure_constrained_selection_report(
        tmp_path / "constrained-passed.json"
    )
    assert validated["selection_rule"] == "NORMAL_ROUTE_AND_EXACT_FAILURE_STATE_CONJUNCTION"

    tampered = dict(jointly_selected)
    tampered["hardware_authorized"] = True
    tampered.pop("report_hash")
    tampered["report_hash"] = hash_json(tampered)
    (tmp_path / "constrained-tampered.json").write_text(
        json.dumps(tampered), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="invalid"):
        validate_recovery_mjx_failure_constrained_selection_report(
            tmp_path / "constrained-tampered.json"
        )


def test_mjx_route_manifest_binds_every_cpu_success_and_time_dilation(
    tmp_path: Path,
) -> None:
    valid_hash = "sha256:" + "1" * 64
    snapshots = []
    for index, posture in enumerate(("PRONE", "LEFT_SIDE")):
        qpos = [0.0] * 36
        qpos[2] = 0.18
        qpos[3] = 1.0
        snapshots.append(
            RecoverySnapshot(
                episode_seed=index,
                environment_index=index,
                control_step=100 + index,
                stage="FAILURE_TERMINAL",
                save_kind="BODY",
                posture_cluster=posture,  # type: ignore[arg-type]
                qpos=qpos,
                qvel=[0.0] * 35,
                applied_action=[0.0] * 29,
                ball_position_m=[0.0, 0.0, 0.11],
                ball_velocity_mps=[0.0, 0.0, 0.0],
                target_position_m=[1.0, 0.0, 1.0],
                left_foot_supported=False,
                right_foot_supported=False,
                failed=True,
                body_hash=valid_hash,
                physics_scene_hash=valid_hash,
                source_policy_hash=valid_hash,
                source_config_hash=valid_hash,
            )
        )
    manifest = write_recovery_snapshot_corpus(
        snapshots=snapshots,
        output_dir=tmp_path,
        corpus_name="routes",
    )
    snapshot_path = tmp_path / "routes.json"
    teacher_policy_path = tmp_path / "policy.onnx"
    teacher_policy_path.write_bytes(b"teacher-policy")
    teacher_config_path = tmp_path / "teacher-config.json"
    teacher_config_path.write_bytes(b'{"teacher": true}')
    teacher_hash = hash_bytes(teacher_policy_path.read_bytes())
    motion_paths = []
    trials = []
    for index, snapshot in enumerate(snapshots):
        motion_path = tmp_path / f"getup_{index}.npz"
        motion_path.write_bytes(f"motion-{index}".encode())
        motion_paths.append(motion_path)
        match = RecoveryEntryMatch(
            motion_id=f"getup_{index}",
            source_hash=hash_bytes(motion_path.read_bytes()),
            entry_frame=100 * index,
            successor_end_frame=100 * index + 200,
            score=0.1,
            joint_rmse_rad=0.1,
            gravity_distance=0.1,
            pelvis_height_error_m=0.1,
            search_config_hash="sha256:" + "5" * 64,
        )
        trials.append(
            RecoveryBridgeTrial(
                snapshot_hash=snapshot.snapshot_hash,
                match=match,
                teacher_policy_hash=teacher_hash,
                time_dilation=index + 1,
                succeeded=True,
                final_stable_sec=2.0,
                executed_sec=10.0,
                peak_root_angular_speed_rad_s=2.0,
                final_pelvis_height_m=0.7,
                finite_state=True,
                ready_handoff_triggered=True,
            )
        )
    schedule = build_recovery_bridge_schedule(trials)
    development = {
        "schema_version": "rosclaw_soccer.opentrack_recovery_bridge_exam.v1",
        "physical_truth": True,
        "physics_backend": "opentrack_mujoco_cpu",
        "snapshot_manifest_hash": hash_bytes(snapshot_path.read_bytes()),
        "reference_library_hash": "sha256:" + "6" * 64,
        "teacher_policy_hash": teacher_hash,
        "teacher_config_hash": hash_bytes(teacher_config_path.read_bytes()),
        "opentrack_commit": "a" * 40,
        "post_skill_transfer": {"development_schedule": schedule},
        "promotion_eligible": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    development["report_hash"] = hash_json(development)
    development_path = tmp_path / "development.json"
    development_path.write_text(json.dumps(development), encoding="utf-8")
    route_path = tmp_path / "route-manifest.json"
    result = build_recovery_mjx_route_manifest(
        snapshot_manifest_path=snapshot_path,
        development_report_path=development_path,
        output_path=route_path,
    )
    assert result["snapshot_corpus_hash"] == manifest["corpus_hash"]
    assert result["route_count"] == 2
    assert result["route_group_count"] == 2
    assert [row["time_dilation"] for row in result["routes"]] == [1, 2]
    assert result["promotion_authority"] == "NONE"
    assert validate_recovery_mjx_route_manifest(route_path)["report_hash"] == result["report_hash"]
    first_group = result["route_groups"][0]
    motion_by_id = {path.stem: path for path in motion_paths}
    job = resolve_recovery_mjx_route_group(
        route_manifest_path=route_path,
        route_group_index=0,
        snapshot_manifest_path=snapshot_path,
        teacher_policy_path=teacher_policy_path,
        teacher_config_path=teacher_config_path,
        motion_archive_path=motion_by_id[first_group["motion_id"]],
    )
    assert job["route_group_hash"] == first_group["route_group_hash"]
    assert job["time_dilation"] in (1, 2)
    assert job["minimum_episode_length"] == 700

    tampered = json.loads(route_path.read_text(encoding="utf-8"))
    tampered["routes"][1]["time_dilation"] = 1
    route_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid|integrity"):
        validate_recovery_mjx_route_manifest(route_path)


def test_mjx_route_manifest_config_is_cpu_truth_and_sim_only() -> None:
    config = RecoveryMJXRouteManifestConfig()
    assert config.required_trial_backend == "mujoco_cpu"
    assert config.control_dt_sec == pytest.approx(0.02)
    assert config.episode_margin_sec == pytest.approx(4.0)
    assert config.activation_ceiling == "SIM_ONLY"
    with pytest.raises(ValueError, match="invalid"):
        RecoveryMJXRouteManifestConfig(hardware_authorized=True)
