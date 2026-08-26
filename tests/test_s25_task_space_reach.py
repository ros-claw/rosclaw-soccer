from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.training.goalkeeper_mjwarp import goalkeeper_world_config
from rosclaw_soccer.training.goalkeeper_physics_ppo import (
    GoalkeeperPhysicsPPOConfig,
    _build_actor_critic,
    _load_actor_critic_state,
    _mask_arm_only_gradients,
    _mask_specialist_gradients,
    _sample_exploration_action,
)
from rosclaw_soccer.training.goalkeeper_reach import (
    GoalkeeperReachAtlasConfig,
    GoalkeeperReachConfig,
    build_g1_task_space_reach_atlas,
    build_g1_task_space_reach_model,
    task_space_reach_from_target_numpy,
    task_space_reach_from_target_torch,
    task_space_reach_teacher_action,
)
from rosclaw_soccer.training.goalkeeper_teacher import pretrain_goalkeeper_actor


def test_advanced_shot_profile_is_faster_wider_and_shortens_recovery() -> None:
    standard = goalkeeper_world_config(difficulty_profile="standard", environment_count=8)
    advanced = goalkeeper_world_config(difficulty_profile="advanced", environment_count=8)

    assert advanced.flight_time_range_sec[1] < standard.flight_time_range_sec[1]
    assert advanced.ball_start_x_range_m[0] < standard.ball_start_x_range_m[0]
    assert advanced.target_y_range_m[1] > standard.target_y_range_m[1]
    assert advanced.target_z_range_m[0] < standard.target_z_range_m[0]
    assert advanced.second_shot_release_sec < standard.second_shot_release_sec
    assert advanced.difficulty_profile == "advanced"


def test_match_profile_has_harder_shots_and_a_causal_windup() -> None:
    standard = goalkeeper_world_config(difficulty_profile="standard", environment_count=8)
    with pytest.raises(ValueError, match="requires causal shot-intent"):
        goalkeeper_world_config(difficulty_profile="match", environment_count=8)
    match = goalkeeper_world_config(
        difficulty_profile="match",
        environment_count=8,
        shot_intent_cue_enabled=True,
    )

    assert match.first_shot_release_sec > standard.first_shot_release_sec
    assert match.flight_time_range_sec[1] < standard.flight_time_range_sec[1]
    assert match.target_y_range_m[1] > standard.target_y_range_m[1]
    assert match.target_z_range_m[1] > standard.target_z_range_m[1]
    assert match.target_z_range_m[0] < standard.target_z_range_m[0]
    assert match.difficulty_profile == "match"


def test_elite_shot_profile_is_faster_and_unlocks_only_arm_agility() -> None:
    advanced = goalkeeper_world_config(difficulty_profile="advanced", environment_count=8)
    elite = goalkeeper_world_config(difficulty_profile="elite", environment_count=8)

    assert elite.flight_time_range_sec[1] < advanced.flight_time_range_sec[1]
    assert elite.ball_start_x_range_m[0] < advanced.ball_start_x_range_m[0]
    assert elite.target_y_range_m[1] > advanced.target_y_range_m[1]
    assert elite.target_z_range_m[1] > advanced.target_z_range_m[1]
    assert elite.second_shot_release_sec < advanced.second_shot_release_sec
    assert elite.action_filter_fraction == advanced.action_filter_fraction
    assert elite.maximum_action_step == advanced.maximum_action_step
    assert elite.arm_action_filter_fraction > elite.action_filter_fraction
    assert elite.maximum_arm_action_step > elite.maximum_action_step
    assert elite.second_shot_arm_authority_scale < advanced.second_shot_arm_authority_scale
    assert elite.maximum_lateral_command_mps < advanced.maximum_lateral_command_mps
    assert elite.arm_residual_scale_multiplier > 1.0
    assert elite.maximum_applied_actor_action_step == pytest.approx(0.072)
    assert elite.reach_reward_scale > advanced.reach_reward_scale
    assert elite.hand_save_bonus > advanced.hand_save_bonus
    assert elite.root_angular_speed_penalty_scale > advanced.root_angular_speed_penalty_scale


def test_reach_config_fails_closed() -> None:
    config = GoalkeeperReachConfig()
    assert config.config_hash.startswith("sha256:")
    with pytest.raises(ValueError, match="SIM_ONLY"):
        replace(config, hardware_authorized=True)
    atlas = GoalkeeperReachAtlasConfig()
    assert atlas.config_hash.startswith("sha256:")
    with pytest.raises(ValueError, match="SIM_ONLY"):
        replace(atlas, hardware_authorized=True)
    with pytest.raises(ValueError, match="grid"):
        replace(atlas, lateral_targets_m=(-1.0, -0.5, 0.0, 0.4, 1.0))


def test_online_parent_distillation_requires_a_bound_parent() -> None:
    with pytest.raises(ValueError, match="requires a parent checkpoint"):
        GoalkeeperPhysicsPPOConfig(online_parent_distillation_coefficient=0.25)


def test_arm_only_update_flag_is_content_bound() -> None:
    config = GoalkeeperPhysicsPPOConfig(arm_only_online_update=True)
    assert config.arm_only_online_update
    assert config.config_hash.startswith("sha256:")
    with pytest.raises(ValueError, match="requires a targeted dive"):
        GoalkeeperPhysicsPPOConfig(lower_body_and_arms_online_update=True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        GoalkeeperPhysicsPPOConfig(
            arm_only_online_update=True,
            lower_body_and_arms_online_update=True,
        )
    assert GoalkeeperPhysicsPPOConfig(training_second_shot_probability=0.0)
    with pytest.raises(ValueError, match="second-shot curriculum"):
        GoalkeeperPhysicsPPOConfig(training_second_shot_probability=-0.01)


def test_arm_only_gradient_mask_keeps_critic_and_arm_actor_plastic() -> None:
    torch = pytest.importorskip("torch")
    from torch import nn

    model = _build_actor_critic(torch, nn, 74, 18, 32)
    sum(parameter.sum() for parameter in model.parameters()).backward()
    _mask_arm_only_gradients(model)

    assert torch.count_nonzero(model.trunk[0].weight.grad) == 0
    assert torch.count_nonzero(model.actor.weight.grad[:4]) == 0
    assert torch.count_nonzero(model.actor.weight.grad[4:]) > 0
    assert torch.count_nonzero(model.specialist_adapter.weight.grad[:4]) == 0
    assert torch.count_nonzero(model.specialist_adapter.weight.grad[4:]) > 0
    assert torch.count_nonzero(model.specialist_adapter_trunk[0].weight.grad) > 0
    assert torch.count_nonzero(model.log_std.grad[:4]) == 0
    assert torch.count_nonzero(model.log_std.grad[4:]) > 0
    assert torch.count_nonzero(model.critic.weight.grad) > 0


def test_legacy_actor_zero_migrates_specialist_adapter_without_output_drift() -> None:
    torch = pytest.importorskip("torch")
    from torch import nn

    source = _build_actor_critic(torch, nn, 74, 18, 32)
    with torch.no_grad():
        source.actor.weight.normal_(0.0, 0.02)
        source.actor.bias.normal_(0.0, 0.02)
    legacy = {
        key: value
        for key, value in source.state_dict().items()
        if not key.startswith("specialist_adapter")
    }
    target = _build_actor_critic(torch, nn, 74, 18, 32)
    observation = torch.randn(5, 74)

    migration = _load_actor_critic_state(target, legacy)

    assert migration == "ZERO_OUTPUT_SPECIALIST_ADAPTER"
    assert torch.equal(source(observation)[0], target(observation)[0])


def test_arm_only_exploration_keeps_core_action_deterministic() -> None:
    torch = pytest.importorskip("torch")
    from torch.distributions import Normal

    torch.manual_seed(31)
    mean = torch.linspace(-0.3, 0.3, 36).reshape(2, 18)
    log_std = torch.full((18,), -1.0)
    action, log_probability = _sample_exploration_action(
        torch=torch,
        normal_distribution=Normal,
        mean=mean,
        log_std=log_std,
        arm_only=True,
    )

    assert torch.equal(action[:, :4], mean[:, :4])
    assert torch.count_nonzero(action[:, 4:] - mean[:, 4:]) > 0
    assert tuple(log_probability.shape) == (2,)


def test_targeted_dive_arm_boundary_freezes_gate_legs_and_waist() -> None:
    torch = pytest.importorskip("torch")
    from torch import nn
    from torch.distributions import Normal

    model = _build_actor_critic(torch, nn, 77, 30, 32)
    sum(parameter.sum() for parameter in model.parameters()).backward()
    _mask_arm_only_gradients(model, arm_action_start_index=16)
    assert torch.count_nonzero(model.actor.weight.grad[:16]) == 0
    assert torch.count_nonzero(model.actor.weight.grad[16:]) > 0

    mean = torch.linspace(-0.3, 0.3, 60).reshape(2, 30)
    log_std = torch.full((30,), -1.0)
    action, _ = _sample_exploration_action(
        torch=torch,
        normal_distribution=Normal,
        mean=mean,
        log_std=log_std,
        arm_only=True,
        arm_action_start_index=16,
    )
    assert torch.equal(action[:, :16], mean[:, :16])
    assert torch.count_nonzero(action[:, 16:] - mean[:, 16:]) > 0


def test_targeted_dive_lower_body_and_arms_scope_freezes_gate_and_waist() -> None:
    torch = pytest.importorskip("torch")
    from torch import nn
    from torch.distributions import Normal

    model = _build_actor_critic(torch, nn, 77, 30, 32)
    sum(parameter.sum() for parameter in model.parameters()).backward()
    plastic = tuple(range(1, 13)) + tuple(range(16, 30))
    _mask_specialist_gradients(model, plastic_action_indices=plastic)
    assert torch.count_nonzero(model.actor.weight.grad[0]) == 0
    assert torch.count_nonzero(model.actor.weight.grad[1:13]) > 0
    assert torch.count_nonzero(model.actor.weight.grad[13:16]) == 0
    assert torch.count_nonzero(model.actor.weight.grad[16:]) > 0

    mean = torch.linspace(-0.3, 0.3, 60).reshape(2, 30)
    log_std = torch.full((30,), -1.0)
    action, _ = _sample_exploration_action(
        torch=torch,
        normal_distribution=Normal,
        mean=mean,
        log_std=log_std,
        arm_only=False,
        arm_action_start_index=16,
        lower_body_and_arms=True,
        lower_body_action_start_index=1,
        lower_body_action_end_index=13,
    )
    assert torch.equal(action[:, 0], mean[:, 0])
    assert torch.count_nonzero(action[:, 1:13] - mean[:, 1:13]) > 0
    assert torch.equal(action[:, 13:16], mean[:, 13:16])
    assert torch.count_nonzero(action[:, 16:] - mean[:, 16:]) > 0


def test_arm_only_teacher_preserves_trunk_and_core_actor_exactly() -> None:
    torch = pytest.importorskip("torch")
    from torch import nn

    torch.manual_seed(29)
    model = _build_actor_critic(torch, nn, 74, 18, 32)
    trunk_before = {key: value.detach().clone() for key, value in model.trunk.state_dict().items()}
    core_weight_before = model.actor.weight[:4].detach().clone()
    core_bias_before = model.actor.bias[:4].detach().clone()

    report = pretrain_goalkeeper_actor(
        model,
        observation_size=74,
        device=torch.device("cpu"),
        samples=4096,
        epochs=2,
        arm_only_update=True,
        seed=29,
    )

    assert report["improved"]
    assert report["arm_only_update"]
    assert report["maximum_frozen_parameter_difference"] == 0.0
    for key, value in model.trunk.state_dict().items():
        assert torch.equal(value, trunk_before[key])
    assert torch.equal(model.actor.weight[:4], core_weight_before)
    assert torch.equal(model.actor.bias[:4], core_bias_before)
    assert torch.count_nonzero(model.actor.weight[4:]) > 0


def test_task_space_reach_is_body_bound_mirrored_and_arm_only() -> None:
    torch = pytest.importorskip("torch")
    asset_value = os.environ.get("ROSCLAW_G1_ASSET_ROOT")
    if asset_value is None:
        pytest.skip("qualified G1 assets are unavailable")
    config = GoalkeeperReachConfig()
    model = build_g1_task_space_reach_model(Path(asset_value), config=config)
    observation = torch.zeros((3, 74))
    observation[:2, 71] = 1.0
    # Intercepts are pelvis-relative and scaled by 0.5 in the public actor
    # observation. Select mirrored high corners plus one inactive ready case.
    observation[0, 6:9] = torch.tensor((-0.08, -0.48, 0.26))
    observation[1, 6:9] = torch.tensor((-0.08, 0.48, 0.26))

    action = task_space_reach_teacher_action(observation, model=model, config=config)

    assert tuple(action.shape) == (3, 18)
    assert torch.count_nonzero(action[:, :4]) == 0
    assert torch.linalg.vector_norm(action[0, 4:11]) > torch.linalg.vector_norm(action[0, 11:18])
    assert torch.linalg.vector_norm(action[1, 11:18]) > torch.linalg.vector_norm(action[1, 4:11])
    assert torch.count_nonzero(action[2]) == 0
    assert torch.max(torch.abs(action)) <= 1.0
    assert model.model_hash.startswith("sha256:")


def test_task_space_reach_pre_shapes_only_from_visible_intent_cue() -> None:
    torch = pytest.importorskip("torch")
    asset_value = os.environ.get("ROSCLAW_G1_ASSET_ROOT")
    if asset_value is None:
        pytest.skip("qualified G1 assets are unavailable")
    model = build_g1_task_space_reach_model(Path(asset_value))
    observation = torch.zeros((2, 77))
    observation[:, 6:9] = torch.tensor((-0.08, -0.35, 0.25))
    observation[:, -4] = 1.0  # ready phase
    observation[0, -5] = 1.0  # visible intent; row 1 represents cue dropout

    action = task_space_reach_teacher_action(
        observation,
        model=model,
        allow_intent_cue=True,
    )

    assert torch.count_nonzero(action[0, 4:]) > 0
    assert torch.count_nonzero(action[1]) == 0


def test_nonlinear_reach_atlas_is_body_bound_mirrored_and_bounded() -> None:
    torch = pytest.importorskip("torch")
    asset_value = os.environ.get("ROSCLAW_G1_ASSET_ROOT")
    if asset_value is None:
        pytest.skip("qualified G1 assets are unavailable")
    config = GoalkeeperReachConfig(reach_gain=0.95)
    atlas = build_g1_task_space_reach_atlas(Path(asset_value), config=config)

    assert len(atlas.target_relative_m) == 42
    assert atlas.model_hash.startswith("sha256:")
    assert atlas.body_hash.startswith("sha256:")
    assert max(abs(value) for row in atlas.left_normalized_action for value in row) <= 1.0
    for index, target in enumerate(atlas.target_relative_m):
        mirror = (target[0], -target[1], target[2])
        mirror_index = atlas.target_relative_m.index(mirror)
        assert atlas.left_terminal_error_m[index] == pytest.approx(
            atlas.right_terminal_error_m[mirror_index], abs=2.0e-5
        )

    observation = torch.zeros((3, 74))
    observation[:2, 71] = 1.0
    observation[0, 6:9] = torch.tensor((-0.04, -0.35, 0.29))
    observation[1, 6:9] = torch.tensor((-0.04, 0.35, 0.29))
    action = task_space_reach_teacher_action(observation, model=atlas, config=config)

    assert tuple(action.shape) == (3, 18)
    assert torch.count_nonzero(action[:, :4]) == 0
    assert torch.linalg.vector_norm(action[0, 4:11]) > torch.linalg.vector_norm(action[0, 11:18])
    assert torch.linalg.vector_norm(action[1, 11:18]) > torch.linalg.vector_norm(action[1, 4:11])
    assert torch.count_nonzero(action[2]) == 0
    assert torch.max(torch.abs(action)) <= 1.0
    targets = np.asarray(((-0.08, -0.72, 0.68), (-0.08, 0.72, -0.42)))
    numpy_reach = task_space_reach_from_target_numpy(
        target_relative=targets,
        model=atlas,
    )
    torch_reach = task_space_reach_from_target_torch(
        torch=torch,
        target_relative=torch.from_numpy(targets),
        model=atlas,
    ).numpy()
    assert numpy_reach.shape == (2, 14)
    assert np.allclose(numpy_reach, torch_reach, atol=1.0e-10, rtol=0.0)
    mirrored_targets = np.asarray(((-0.08, -0.72, 0.54), (-0.08, 0.72, 0.54)))
    mirrored_reach = task_space_reach_from_target_numpy(
        target_relative=mirrored_targets,
        model=atlas,
    )
    arm_mirror_sign = np.asarray((1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0))
    expected_negative = np.concatenate(
        (mirrored_reach[1, 7:] * arm_mirror_sign, mirrored_reach[1, :7] * arm_mirror_sign)
    )
    assert np.allclose(mirrored_reach[0], expected_negative, atol=1.0e-12, rtol=0.0)


def test_reach_pretraining_populates_causal_contact_plane_x() -> None:
    import inspect

    from rosclaw_soccer.training.goalkeeper_teacher import pretrain_goalkeeper_actor

    source = inspect.getsource(pretrain_goalkeeper_actor)
    assert "observation[:, 6] = -0.04" in source
