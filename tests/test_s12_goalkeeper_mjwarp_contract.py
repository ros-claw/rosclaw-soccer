from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from rosclaw_soccer.training.goalkeeper_mjwarp import (
    GoalkeeperMJWarpConfig,
    _root_angular_speed_tail_summary,
    goalkeeper_world_config,
)
from rosclaw_soccer.training.goalkeeper_physics_ppo import (
    GoalkeeperPhysicsPPOConfig,
    _all_gather_successful_memory_rows,
    _append_successful_trajectory_memory,
    _balanced_successful_memory_sample_indices,
    _controller_semantic_migration,
    _mirror_goalkeeper_actor_action,
    _mirror_goalkeeper_actor_observation,
    _required_successful_memory_strata,
    _reward_accounting_residual,
    _sample_exploration_action,
    _shrink_successful_action_innovation,
    _successful_memory_covers_strata,
    _successful_memory_has_episode_diversity,
    _successful_memory_replay_strength,
    _successful_trajectory_replay_mask,
)
from rosclaw_soccer.world.field import build_g1_stadium_model


def test_mjwarp_contract_is_content_bound_and_sim_only(tmp_path: Path) -> None:
    targeted_checkpoint = tmp_path / "targeted.pt"
    targeted_checkpoint.write_bytes(b"targeted-dive-test")
    config = GoalkeeperMJWarpConfig(environment_count=8)
    assert config.episode_steps == 250
    assert config.config_hash.startswith("sha256:")
    assert config.maximum_lateral_command_mps <= 0.40
    assert config.action_filter_fraction < 1.0
    with pytest.raises(ValueError, match="SIM_ONLY"):
        replace(config, hardware_authorized=True)
    cue = goalkeeper_world_config(
        difficulty_profile="standard",
        environment_count=8,
        shot_intent_cue_enabled=True,
    )
    assert cue.shot_intent_cue_enabled
    assert cue.config_hash != config.config_hash
    hard = goalkeeper_world_config(
        difficulty_profile="standard",
        environment_count=8,
        hard_shot_fraction=0.65,
        task_motion_reward_scale=8.0,
    )
    assert hard.hard_shot_fraction == pytest.approx(0.65)
    assert hard.task_motion_reward_scale == pytest.approx(8.0)
    assert hard.config_hash != config.config_hash
    reward_aligned = replace(
        config,
        true_save_bonus=180.0,
        hand_save_bonus=120.0,
        recovery_event_bonus=150.0,
    )
    assert reward_aligned.recovery_event_bonus == pytest.approx(150.0)
    targeted_config = GoalkeeperPhysicsPPOConfig(
        targeted_dive_checkpoint=str(targeted_checkpoint),
        shot_intent_cue_enabled=True,
        teacher_pretraining_enabled=False,
    )
    assert replace(
        targeted_config,
        symmetry_mirror_loss_coefficient=0.25,
    ).symmetry_mirror_loss_coefficient == pytest.approx(0.25)
    with pytest.raises(ValueError, match="symmetry mirror"):
        replace(GoalkeeperPhysicsPPOConfig(), symmetry_mirror_loss_coefficient=-0.01)
    with pytest.raises(ValueError, match="requires targeted-dive actor"):
        replace(GoalkeeperPhysicsPPOConfig(), symmetry_mirror_loss_coefficient=0.25)
    mirrored_replay = GoalkeeperPhysicsPPOConfig(
        targeted_dive_checkpoint=str(targeted_checkpoint),
        shot_intent_cue_enabled=True,
        teacher_pretraining_enabled=False,
        successful_trajectory_replay_coefficient=1.0,
        successful_trajectory_memory_capacity_per_stratum=8,
        successful_trajectory_mirror_augmentation_enabled=True,
    )
    assert mirrored_replay.successful_trajectory_mirror_augmentation_enabled
    recovery_replay = replace(
        mirrored_replay,
        support_landing_online_update=True,
        targeted_dive_runtime_contact_support_side_enabled=True,
        targeted_dive_actor_contact_support_side_enabled=True,
        targeted_dive_actor_recovery_context_enabled=True,
        targeted_dive_lateral_drive_learned_gate_enabled=True,
        successful_trajectory_recovery_only_enabled=True,
        recovery_transition_policy_weight=4.0,
        successful_trajectory_memory_minimum_episodes_per_stratum=8,
        support_landing_causal_recovery_only_enabled=True,
    )
    assert recovery_replay.successful_trajectory_recovery_only_enabled
    assert recovery_replay.recovery_transition_policy_weight == pytest.approx(4.0)
    assert recovery_replay.successful_trajectory_memory_minimum_episodes_per_stratum == 8
    assert recovery_replay.support_landing_causal_recovery_only_enabled
    assert replace(
        recovery_replay,
        rollback_to_exploration_champion_on_regression_enabled=True,
    ).rollback_to_exploration_champion_on_regression_enabled
    with pytest.raises(ValueError, match="causal support-landing requires"):
        replace(targeted_config, support_landing_causal_recovery_only_enabled=True)
    with pytest.raises(ValueError, match="recovery weighting requires"):
        replace(targeted_config, recovery_transition_policy_weight=4.0)
    with pytest.raises(ValueError, match="recovery-only replay requires"):
        replace(mirrored_replay, successful_trajectory_recovery_only_enabled=True)
    with pytest.raises(ValueError, match="mirror requires targeted-dive memory"):
        replace(
            targeted_config,
            successful_trajectory_mirror_augmentation_enabled=True,
        )
    all_hard = replace(config, hard_shot_fraction=1.0, hard_shot_height_mode="balanced")
    assert all_hard.hard_shot_fraction == pytest.approx(1.0)
    assert replace(config, hard_shot_fraction=1.0, hard_shot_height_mode="low")
    assert replace(config, hard_shot_fraction=1.0, hard_shot_side_mode="negative")
    scaffold = replace(
        config,
        hard_shot_fraction=1.0,
        hard_shot_height_mode="balanced",
        hard_shot_flight_time_range_sec=(0.70, 0.90),
    )
    assert scaffold.hard_shot_flight_time_range_sec == (0.70, 0.90)
    with pytest.raises(ValueError, match="hard-shot fraction"):
        replace(config, hard_shot_fraction=1.01)
    with pytest.raises(ValueError, match="height mode"):
        replace(config, hard_shot_height_mode="easy")
    with pytest.raises(ValueError, match="side mode"):
        replace(config, hard_shot_side_mode="center")
    with pytest.raises(ValueError, match="flight curriculum"):
        replace(config, hard_shot_flight_time_range_sec=(0.70, 0.90))
    with pytest.raises(ValueError, match="cue requires mobility"):
        GoalkeeperPhysicsPPOConfig(shot_intent_cue_enabled=True)
    with pytest.raises(ValueError, match="anticipatory arms require"):
        GoalkeeperPhysicsPPOConfig(mobility_anticipatory_arm_reach_enabled=True)
    with pytest.raises(ValueError, match="command limits"):
        GoalkeeperPhysicsPPOConfig(
            mobility_lateral_command_limit=0.5,
            mobility_recovery_command_limit=0.7,
        )
    with pytest.raises(ValueError, match="lateral-velocity guard requires"):
        GoalkeeperPhysicsPPOConfig(mobility_lateral_velocity_guard_enabled=True)
    with pytest.raises(ValueError, match="predictive teacher requires"):
        GoalkeeperPhysicsPPOConfig(mobility_predictive_teacher_warmstart_enabled=True)
    with pytest.raises(ValueError, match="teacher-recovery latch requires"):
        GoalkeeperPhysicsPPOConfig(mobility_teacher_recovery_latch_enabled=True)
    with pytest.raises(ValueError, match="runtime reach requires"):
        GoalkeeperPhysicsPPOConfig(
            runtime_task_space_reach_enabled=True,
            runtime_task_space_reach_blend=0.5,
        )
    assert GoalkeeperPhysicsPPOConfig(training_hard_shot_fraction=1.0)
    assert GoalkeeperPhysicsPPOConfig(
        training_hard_shot_fraction=1.0,
        training_hard_shot_flight_time_range_sec=(0.70, 0.90),
    )
    with pytest.raises(ValueError, match="hard-shot fraction"):
        GoalkeeperPhysicsPPOConfig(training_hard_shot_fraction=1.01)
    with pytest.raises(ValueError, match="flight curriculum"):
        GoalkeeperPhysicsPPOConfig(
            training_hard_shot_flight_time_range_sec=(0.70, 0.90),
        )
    with pytest.raises(ValueError, match="successful-trajectory replay"):
        GoalkeeperPhysicsPPOConfig(successful_trajectory_replay_coefficient=10.1)
    with pytest.raises(ValueError, match="memory requires replay"):
        GoalkeeperPhysicsPPOConfig(
            successful_trajectory_memory_capacity_per_stratum=100,
        )
    with pytest.raises(ValueError, match="episode diversity"):
        GoalkeeperPhysicsPPOConfig(
            successful_trajectory_memory_minimum_episodes_per_stratum=0,
        )
    with pytest.raises(ValueError, match="full-strength"):
        GoalkeeperPhysicsPPOConfig(
            successful_trajectory_memory_full_strength_episodes_per_stratum=-1,
        )
    with pytest.raises(ValueError, match="action innovation"):
        GoalkeeperPhysicsPPOConfig(successful_trajectory_action_innovation_scale=1.01)
    with pytest.raises(ValueError, match="task-motion reward"):
        GoalkeeperPhysicsPPOConfig(training_task_motion_reward_scale=20.1)


def test_root_angular_tail_summary_exposes_rare_instability() -> None:
    torch = pytest.importorskip("torch")

    report = _root_angular_speed_tail_summary(
        torch=torch,
        maximum_speeds=torch.tensor((1.0, 2.0, 3.0, 6.0)),
        soft_limit_rad_s=2.5,
    )
    assert report["maximum_root_angular_speed_rad_s"] == pytest.approx(6.0)
    assert report["p95_maximum_root_angular_speed_rad_s"] > 5.0
    assert (
        report["p99_maximum_root_angular_speed_rad_s"]
        > report["p95_maximum_root_angular_speed_rad_s"]
    )
    assert report["root_angular_speed_soft_limit_exceedance_rate"] == pytest.approx(0.5)
    assert report["strict_stability_ceiling_exceedance_rate"] == pytest.approx(0.25)


def test_reward_component_ledger_reconstructs_episode_return() -> None:
    metrics = {
        "mean_episode_reward": 10.0,
        "mean_reach_reward": 2.0,
        "mean_bimanual_reach_reward": 1.0,
        "mean_task_motion_reward": 3.0,
        "mean_upright_reward": 8.0,
        "mean_recovery_progress_reward": 4.0,
        "mean_event_bonus": 7.0,
        "mean_smoothness_penalty": 5.0,
        "mean_effort_penalty": 2.0,
        "mean_safety_penalty": 6.0,
        "mean_nonfinite_override": -2.0,
    }

    assert _reward_accounting_residual(metrics) == pytest.approx(0.0)


def test_successful_trajectory_replay_excludes_unsafe_and_inactive_steps() -> None:
    import inspect

    torch = pytest.importorskip("torch")
    from rosclaw_soccer.training.goalkeeper_physics_ppo import run_goalkeeper_physics_ppo

    active = torch.tensor(
        (
            (False, False, False),
            (True, True, True),
            (True, False, True),
        )
    )
    mask, eligible = _successful_trajectory_replay_mask(
        torch=torch,
        active_steps=active,
        first_save=torch.tensor((True, True, True)),
        quarantined=torch.tensor((False, True, False)),
        maximum_root_angular_speed_rad_s=torch.tensor((3.0, 2.0, 4.0)),
        angular_speed_ceiling_rad_s=3.5,
    )
    assert eligible.tolist() == [True, False, False]
    assert mask.tolist() == [
        [False, False, False],
        [True, False, False],
        [True, False, False],
    ]
    rollout_source = inspect.getsource(run_goalkeeper_physics_ppo)
    assert '"_option_active"' in rollout_source
    assert '"_option_started"' not in rollout_source
    assert "memory_coverage_consensus" in rollout_source
    assert "dist.ReduceOp.MIN" in rollout_source


def test_successful_trajectory_replay_does_not_count_a_passive_save_episode() -> None:
    torch = pytest.importorskip("torch")

    mask, eligible = _successful_trajectory_replay_mask(
        torch=torch,
        active_steps=torch.tensor(((False, True), (False, False))),
        first_save=torch.tensor((True, True)),
        quarantined=torch.tensor((False, False)),
        maximum_root_angular_speed_rad_s=torch.tensor((1.0, 1.0)),
        angular_speed_ceiling_rad_s=3.5,
    )

    assert eligible.tolist() == [False, True]
    assert mask.tolist() == [[False, True], [False, False]]


def test_successful_trajectory_memory_is_bounded_and_height_balanced() -> None:
    torch = pytest.importorskip("torch")

    observations = torch.arange(6 * 2 * 3, dtype=torch.float32).reshape(6, 2, 3)
    actions = torch.arange(6 * 2 * 2, dtype=torch.float32).reshape(6, 2, 2)
    replay_mask = torch.ones((6, 2), dtype=torch.bool)
    memory = _append_successful_trajectory_memory(
        torch=torch,
        observations=observations,
        actions=actions,
        replay_mask=replay_mask,
        height_strata=torch.tensor((0, 2)),
        memory_observations=None,
        memory_actions=None,
        memory_height_strata=None,
        capacity_per_stratum=3,
    )
    memory_observations, memory_actions, memory_strata = memory
    assert memory_observations.shape == (6, 3)
    assert memory_actions.shape == (6, 2)
    assert torch.bincount(memory_strata, minlength=3).tolist() == [3, 0, 3]
    # FIFO within each stratum preserves the most recent causal rows.
    assert memory_observations[0].tolist() == observations[3, 0].tolist()
    sampled = _balanced_successful_memory_sample_indices(
        torch=torch,
        height_strata=memory_strata,
        sample_count=10,
    )
    sampled_counts = torch.bincount(memory_strata[sampled], minlength=3)
    assert sampled_counts.tolist() == [5, 0, 5]


def test_successful_memory_all_gather_shares_only_new_variable_rows() -> None:
    torch = pytest.importorskip("torch")

    class FakeDist:
        @staticmethod
        def all_gather(outputs, value) -> None:  # type: ignore[no-untyped-def]
            outputs[0].copy_(value)
            if value.ndim == 0:
                outputs[1].fill_(1)
            elif value.ndim == 2 and value.shape[1] == 3:
                outputs[1].zero_()
                outputs[1][0] = torch.tensor((7.0, 8.0, 9.0))
            elif value.ndim == 2:
                outputs[1].zero_()
                outputs[1][0] = torch.tensor((0.7, 0.8))
            else:
                outputs[1].zero_()
                outputs[1][0] = 1

    observations, actions, strata = _all_gather_successful_memory_rows(
        torch=torch,
        dist=FakeDist(),
        observations=torch.tensor(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))),
        actions=torch.tensor(((0.1, 0.2), (0.3, 0.4))),
        strata=torch.tensor((0, 0)),
        world_size=2,
    )

    assert observations.tolist() == [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ]
    assert actions.shape == (3, 2)
    assert strata.tolist() == [0, 0, 1]


def test_successful_memory_balances_height_and_far_corner_side() -> None:
    torch = pytest.importorskip("torch")
    observations = torch.randn(2, 6, 3)
    actions = torch.randn(2, 6, 2)
    replay_mask = torch.ones((2, 6), dtype=torch.bool)
    memory_observations, memory_actions, memory_strata = _append_successful_trajectory_memory(
        torch=torch,
        observations=observations,
        actions=actions,
        replay_mask=replay_mask,
        height_strata=torch.arange(6),
        memory_observations=None,
        memory_actions=None,
        memory_height_strata=None,
        capacity_per_stratum=1,
        stratum_count=6,
    )
    sampled = _balanced_successful_memory_sample_indices(
        torch=torch,
        height_strata=memory_strata,
        sample_count=12,
        stratum_count=6,
    )

    assert memory_observations.shape == (6, 3)
    assert memory_actions.shape == (6, 2)
    assert torch.bincount(memory_strata, minlength=6).tolist() == [1] * 6
    assert torch.bincount(memory_strata[sampled], minlength=6).tolist() == [2] * 6


def test_successful_memory_replay_waits_for_curriculum_coverage() -> None:
    torch = pytest.importorskip("torch")
    required = _required_successful_memory_strata(
        hard_shot_fraction=1.0,
        hard_shot_height_mode="low",
        hard_shot_side_mode="balanced",
    )

    assert required == (0, 1)
    assert not _successful_memory_covers_strata(
        torch=torch,
        height_strata=torch.tensor((1, 1, 1)),
        required_strata=required,
    )
    assert _successful_memory_covers_strata(
        torch=torch,
        height_strata=torch.tensor((0, 1, 1)),
        required_strata=required,
    )
    assert not _successful_memory_has_episode_diversity(
        episode_counts=torch.tensor((2, 8, 0, 0, 0, 0)),
        required_strata=required,
        minimum_episodes=8,
    )
    assert _successful_memory_has_episode_diversity(
        episode_counts=torch.tensor((8, 9, 0, 0, 0, 0)),
        required_strata=required,
        minimum_episodes=8,
    )
    assert _required_successful_memory_strata(
        hard_shot_fraction=1.0,
        hard_shot_height_mode="high",
        hard_shot_side_mode="negative",
    ) == (4,)
    assert _required_successful_memory_strata(
        hard_shot_fraction=0.5,
        hard_shot_height_mode="low",
        hard_shot_side_mode="negative",
    ) == tuple(range(6))


def test_successful_memory_replay_ramps_and_shrinks_lucky_action_noise() -> None:
    torch = pytest.importorskip("torch")
    counts = torch.tensor((8, 16, 0, 0, 0, 0))

    assert _successful_memory_replay_strength(
        episode_counts=counts,
        required_strata=(0, 1),
        minimum_episodes=8,
        full_strength_episodes=32,
    ) == pytest.approx(0.25)
    assert (
        _successful_memory_replay_strength(
            episode_counts=torch.tensor((7, 100, 0, 0, 0, 0)),
            required_strata=(0, 1),
            minimum_episodes=8,
            full_strength_episodes=32,
        )
        == 0.0
    )
    assert (
        _successful_memory_replay_strength(
            episode_counts=torch.tensor((40, 32, 0, 0, 0, 0)),
            required_strata=(0, 1),
            minimum_episodes=8,
            full_strength_episodes=32,
        )
        == 1.0
    )

    mean = torch.zeros((2, 1, 2))
    sampled = torch.tensor((((4.0, -4.0),), ((2.0, -2.0),)))
    target = _shrink_successful_action_innovation(
        torch=torch,
        policy_mean=mean,
        sampled_action=sampled,
        innovation_scale=0.25,
    )
    assert target.tolist() == [[[1.0, -1.0]], [[0.5, -0.5]]]


def test_causal_support_landing_explores_only_after_true_save() -> None:
    torch = pytest.importorskip("torch")
    mean = torch.zeros((3, 6))
    log_std = torch.full((6,), -1.0)
    action, _ = _sample_exploration_action(
        torch=torch,
        normal_distribution=torch.distributions.Normal,
        mean=mean,
        log_std=log_std,
        arm_only=False,
        arm_action_start_index=4,
        support_landing=True,
        exploration_active_mask=torch.tensor((False, True, False)),
    )

    assert torch.equal(action[0], mean[0])
    assert not torch.equal(action[1, :4], mean[1, :4])
    assert torch.equal(action[1, 4:], mean[1, 4:])
    assert torch.equal(action[2], mean[2])


def test_goalkeeper_actor_mirror_is_an_exact_involution() -> None:
    torch = pytest.importorskip("torch")
    observation = torch.arange(2 * 92, dtype=torch.float32).reshape(2, 92)
    action = torch.arange(2 * 30, dtype=torch.float32).reshape(2, 30)

    mirrored_observation = _mirror_goalkeeper_actor_observation(
        torch=torch,
        observation=observation,
        shot_intent_cue_enabled=True,
        actor_contact_support_side_enabled=True,
    )
    mirrored_action = _mirror_goalkeeper_actor_action(torch=torch, action=action)

    assert torch.equal(
        _mirror_goalkeeper_actor_observation(
            torch=torch,
            observation=mirrored_observation,
            shot_intent_cue_enabled=True,
            actor_contact_support_side_enabled=True,
        ),
        observation,
    )
    assert torch.equal(_mirror_goalkeeper_actor_action(torch=torch, action=mirrored_action), action)
    assert mirrored_observation[:, 82].tolist() == (-observation[:, 82]).tolist()
    assert mirrored_observation[:, 83].tolist() == observation[:, 84].tolist()
    assert mirrored_observation[:, 84].tolist() == observation[:, 83].tolist()
    assert mirrored_observation[:, 85].tolist() == (-observation[:, 85]).tolist()

    recovery_observation = torch.arange(2 * 96, dtype=torch.float32).reshape(2, 96)
    mirrored_recovery = _mirror_goalkeeper_actor_observation(
        torch=torch,
        observation=recovery_observation,
        shot_intent_cue_enabled=True,
        actor_contact_support_side_enabled=True,
        actor_recovery_context_enabled=True,
    )
    assert torch.equal(
        _mirror_goalkeeper_actor_observation(
            torch=torch,
            observation=mirrored_recovery,
            shot_intent_cue_enabled=True,
            actor_contact_support_side_enabled=True,
            actor_recovery_context_enabled=True,
        ),
        recovery_observation,
    )
    assert mirrored_recovery[:, 85].tolist() == recovery_observation[:, 85].tolist()
    assert mirrored_recovery[:, 86].tolist() == recovery_observation[:, 86].tolist()
    assert mirrored_recovery[:, 87].tolist() == (-recovery_observation[:, 87]).tolist()
    assert mirrored_recovery[:, 88].tolist() == (-recovery_observation[:, 88]).tolist()
    assert mirrored_recovery[:, 89].tolist() == (-recovery_observation[:, 89]).tolist()


def test_goalkeeper_actor_mirror_rejects_ambiguous_contracts() -> None:
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="30D actions"):
        _mirror_goalkeeper_actor_action(torch=torch, action=torch.zeros((2, 29)))
    with pytest.raises(ValueError, match="92D observations"):
        _mirror_goalkeeper_actor_observation(
            torch=torch,
            observation=torch.zeros((2, 91)),
            shot_intent_cue_enabled=True,
            actor_contact_support_side_enabled=True,
        )


def test_parent_controller_semantic_migration_is_never_silent() -> None:
    config = GoalkeeperPhysicsPPOConfig(
        training_unsafe_penalty=250.0,
        training_save_then_unsafe_penalty=400.0,
    )
    assert config.training_save_then_unsafe_penalty == 400.0
    exact = _controller_semantic_migration(
        parent_training_config={},
        current=config,
    )
    assert exact["exact"]
    assert exact["changed_fields"] == {}

    changed = _controller_semantic_migration(
        parent_training_config={"targeted_dive_actor_residual_scale": 0.40},
        current=config,
    )
    assert not changed["exact"]
    assert changed["changed_fields"] == {
        "targeted_dive_actor_residual_scale": {
            "parent": 0.40,
            "current": config.targeted_dive_actor_residual_scale,
        }
    }


def test_mjwarp_contract_rejects_physics_or_filter_drift() -> None:
    with pytest.raises(ValueError, match="0.002 s"):
        GoalkeeperMJWarpConfig(control_dt_sec=0.025)
    with pytest.raises(ValueError, match="residual/filter"):
        GoalkeeperMJWarpConfig(maximum_action_step=1.1)
    with pytest.raises(ValueError, match="unsafe penalty"):
        GoalkeeperMJWarpConfig(unsafe_penalty=9.0)
    with pytest.raises(ValueError, match="save-then-unsafe"):
        GoalkeeperMJWarpConfig(save_then_unsafe_penalty=-1.0)
    with pytest.raises(ValueError, match="difficulty"):
        goalkeeper_world_config(difficulty_profile="impossible", environment_count=8)  # type: ignore[arg-type]


def test_world_factory_binds_terminal_safety_debt() -> None:
    config = goalkeeper_world_config(
        difficulty_profile="standard",
        environment_count=8,
        unsafe_penalty=300.0,
        save_then_unsafe_penalty=400.0,
    )

    assert config.unsafe_penalty == 300.0
    assert config.save_then_unsafe_penalty == 400.0


def test_physics_ppo_contract_requires_complete_episode_and_sim_only() -> None:
    config = GoalkeeperPhysicsPPOConfig(environments_per_rank=8, iterations=2)
    assert config.config_hash.startswith("sha256:")
    with pytest.raises(ValueError, match="SIM_ONLY"):
        replace(config, hardware_authorized=True)
    with pytest.raises(ValueError, match="rollout dimensions"):
        GoalkeeperPhysicsPPOConfig(rollout_steps=8)


def test_physics_actor_starts_as_zero_residual_champion() -> None:
    torch = pytest.importorskip("torch")
    from torch import nn

    from rosclaw_soccer.training.goalkeeper_physics_ppo import _build_actor_critic

    model = _build_actor_critic(torch, nn, 74, 18, 32)
    mean, _, _ = model(torch.randn(5, 74))
    assert torch.count_nonzero(mean) == 0


def test_legacy_actor_expands_cue_columns_without_changing_outputs() -> None:
    torch = pytest.importorskip("torch")
    from torch import nn

    from rosclaw_soccer.training.goalkeeper_physics_ppo import (
        _build_actor_critic,
        _migrate_initialization_state,
    )

    legacy = _build_actor_critic(torch, nn, 74, 18, 32)
    expanded = _build_actor_critic(torch, nn, 77, 18, 32)
    migrated, method = _migrate_initialization_state(
        torch=torch,
        state_dict=legacy.state_dict(),
        old_observation_size=74,
        new_observation_size=77,
    )
    expanded.load_state_dict(migrated)
    legacy_observation = torch.randn(5, 74)
    cue_observation = torch.cat(
        (legacy_observation[:, :70], torch.zeros((5, 3)), legacy_observation[:, 70:]),
        dim=1,
    )
    legacy_mean, legacy_value, _ = legacy(legacy_observation)
    expanded_mean, expanded_value, _ = expanded(cue_observation)

    assert method == "EXPAND_74_TO_77_ZERO_INITIALIZED_CUE"
    assert torch.allclose(legacy_mean, expanded_mean, atol=1.0e-7, rtol=0.0)
    assert torch.allclose(legacy_value, expanded_value, atol=1.0e-7, rtol=0.0)


def test_targeted_actor_expands_contact_support_without_changing_parent() -> None:
    torch = pytest.importorskip("torch")
    from torch import nn

    from rosclaw_soccer.training.goalkeeper_physics_ppo import (
        _build_actor_critic,
        _migrate_initialization_state,
    )

    parent = _build_actor_critic(torch, nn, 89, 30, 32)
    expanded = _build_actor_critic(torch, nn, 90, 30, 32)
    migrated, method = _migrate_initialization_state(
        torch=torch,
        state_dict=parent.state_dict(),
        old_observation_size=89,
        new_observation_size=90,
    )
    expanded.load_state_dict(migrated)
    parent_observation = torch.randn(5, 89)
    support_observation = torch.cat(
        (parent_observation[:, :82], torch.zeros((5, 1)), parent_observation[:, 82:]),
        dim=1,
    )
    parent_mean, parent_value, _ = parent(parent_observation)
    expanded_mean, expanded_value, _ = expanded(support_observation)

    assert method == "EXPAND_89_TO_90_ZERO_INITIALIZED_CONTACT_SUPPORT"
    assert torch.allclose(parent_mean, expanded_mean, atol=1.0e-7, rtol=0.0)
    assert torch.allclose(parent_value, expanded_value, atol=1.0e-7, rtol=0.0)


def test_support_actor_adds_foot_contacts_without_forgetting_support_side() -> None:
    torch = pytest.importorskip("torch")
    from torch import nn

    from rosclaw_soccer.training.goalkeeper_physics_ppo import (
        _build_actor_critic,
        _migrate_initialization_state,
    )

    parent = _build_actor_critic(torch, nn, 90, 30, 32)
    expanded = _build_actor_critic(torch, nn, 92, 30, 32)
    parent.specialist_adapter_trunk[0].weight.data[:, 82] = 0.25
    migrated, method = _migrate_initialization_state(
        torch=torch,
        state_dict=parent.state_dict(),
        old_observation_size=90,
        new_observation_size=92,
    )
    expanded.load_state_dict(migrated)
    parent_observation = torch.randn(5, 90)
    foot_contact_observation = torch.cat(
        (parent_observation[:, :83], torch.zeros((5, 2)), parent_observation[:, 83:]),
        dim=1,
    )
    parent_mean, parent_value, _ = parent(parent_observation)
    expanded_mean, expanded_value, _ = expanded(foot_contact_observation)

    assert method == "EXPAND_90_TO_92_PRESERVE_SUPPORT_ADD_FOOT_CONTACTS"
    assert torch.allclose(
        migrated["specialist_adapter_trunk.0.weight"][:, 82],
        parent.state_dict()["specialist_adapter_trunk.0.weight"][:, 82],
    )
    assert torch.allclose(parent_mean, expanded_mean, atol=1.0e-7, rtol=0.0)
    assert torch.allclose(parent_value, expanded_value, atol=1.0e-7, rtol=0.0)


def test_contact_actor_adds_recovery_context_without_changing_parent() -> None:
    torch = pytest.importorskip("torch")
    from torch import nn

    from rosclaw_soccer.training.goalkeeper_physics_ppo import (
        _build_actor_critic,
        _migrate_initialization_state,
    )

    parent = _build_actor_critic(torch, nn, 92, 30, 32)
    expanded = _build_actor_critic(torch, nn, 96, 30, 32)
    migrated, method = _migrate_initialization_state(
        torch=torch,
        state_dict=parent.state_dict(),
        old_observation_size=92,
        new_observation_size=96,
    )
    expanded.load_state_dict(migrated)
    parent_observation = torch.randn(5, 92)
    recovery_observation = torch.cat(
        (parent_observation[:, :85], torch.zeros((5, 4)), parent_observation[:, 85:]),
        dim=1,
    )
    parent_mean, parent_value, _ = parent(parent_observation)
    expanded_mean, expanded_value, _ = expanded(recovery_observation)

    assert method == "EXPAND_92_TO_96_ZERO_INITIALIZED_CAUSAL_RECOVERY_CONTEXT"
    assert torch.allclose(parent_mean, expanded_mean, atol=1.0e-7, rtol=0.0)
    assert torch.allclose(parent_value, expanded_value, atol=1.0e-7, rtol=0.0)


def test_multistep_contract_exposes_recovery_as_a_distinct_phase() -> None:
    # Regression guard for the simulator adapters: a save must be credited
    # under shot 1/2, then the following observation switches to shot 0 so the
    # actor can retract instead of holding a reach until a wall-clock timeout.
    from rosclaw_soccer.training.goalkeeper_multistep import GoalkeeperEpisodePhase

    assert int(GoalkeeperEpisodePhase.FIRST_IMPACT) < int(GoalkeeperEpisodePhase.FIRST_RECOVERY)


def test_mjwarp_support_side_uses_causal_ground_contacts() -> None:
    torch = pytest.importorskip("torch")
    from rosclaw_soccer.training.goalkeeper_mjwarp import GoalkeeperMJWarpBatch

    environment = SimpleNamespace(
        torch=torch,
        contact_geom=torch.tensor(((0, 1), (0, 2), (0, 2), (1, 0))),
        contact_world=torch.tensor((0, 0, 1, 2)),
        contact_distance=torch.tensor((0.0, 0.0, 0.0, 0.0)),
        cpu_model=SimpleNamespace(ngeom=4),
        _geom_body=torch.tensor((0, 10, 20, 30)),
        _left_foot_body=10,
        _right_foot_body=20,
        count=3,
        device=torch.device("cpu"),
    )
    environment._foot_contact_state = lambda: GoalkeeperMJWarpBatch._foot_contact_state(environment)

    support = GoalkeeperMJWarpBatch._foot_support_side(environment)

    assert torch.equal(support, torch.tensor((0.0, 1.0, -1.0)))

    from rosclaw_soccer.training.goalkeeper_targeted_dive_mjwarp import (
        GoalkeeperTargetedDiveMJWarpBatch,
    )

    environment.dive_config = SimpleNamespace(
        actor_contact_support_side_enabled=True,
        actor_recovery_context_enabled=False,
    )
    actor_support = GoalkeeperTargetedDiveMJWarpBatch._actor_auxiliary_proprioception(environment)
    assert actor_support.shape == (3, 3)
    assert torch.equal(
        actor_support,
        torch.tensor(((0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (-1.0, 1.0, 0.0))),
    )


def test_mjwarp_returns_residual_authority_to_frozen_cerebellum_during_recovery() -> None:
    import inspect

    from rosclaw_soccer.training.goalkeeper_mjwarp import GoalkeeperMJWarpBatch

    source = inspect.getsource(GoalkeeperMJWarpBatch.step)
    shaping_source = inspect.getsource(GoalkeeperMJWarpBatch._shape_actor_action)
    assert "self._shape_actor_action" in source
    assert "shape_goalkeeper_action_torch" in shaping_source
    assert "shot_active=self._shot_index > 0" in shaping_source
    assert source.index("self._apply_timeline_releases()") < source.index(
        "requested_action = torch.clamp"
    )


def test_failed_batched_worlds_are_quarantined_without_erasing_failure() -> None:
    import inspect

    from rosclaw_soccer.training.goalkeeper_mjwarp import GoalkeeperMJWarpBatch

    step_source = inspect.getsource(GoalkeeperMJWarpBatch.step)
    restore_source = inspect.getsource(GoalkeeperMJWarpBatch._restore_quarantined_worlds)
    assert "self.task.phase[new_nonfinite]" in step_source
    assert "self._quarantined |= self.task.phase == 7" in step_source
    assert 'result["terminated"] |= new_nonfinite' in step_source
    assert "self.task.reset" not in restore_source
    assert "self.mjw.forward" in restore_source


def test_goalkeeper_glove_uses_conservative_visible_dimensions() -> None:
    asset_value = os.environ.get("ROSCLAW_G1_ASSET_ROOT")
    if asset_value is None:
        pytest.skip("qualified G1 assets are unavailable")
    assets = Path(asset_value)
    model = build_g1_stadium_model(assets)
    for name in ("left_goalkeeper_glove", "right_goalkeeper_glove"):
        glove = model.geom(name)
        assert tuple(glove.size) == pytest.approx((0.095, 0.050, 0.0325))
        assert tuple(2.0 * glove.size) == pytest.approx((0.19, 0.10, 0.065))
