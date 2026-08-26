from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest

from rosclaw_soccer.training.goalkeeper_combat_teacher import (
    OFFICIAL_GOALKEEPER_BLEND_GROUP_SCALE,
    OFFICIAL_GOALKEEPER_DEFAULT_QPOS,
    OFFICIAL_GOALKEEPER_INITIAL_QPOS,
    OFFICIAL_GOALKEEPER_KD,
    OFFICIAL_GOALKEEPER_KP,
    rotate_inverse_torch,
)
from rosclaw_soccer.training.goalkeeper_physics_ppo import GoalkeeperPhysicsPPOConfig


def test_combat_teacher_contract_is_complete_and_disables_legacy_pretraining(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "teacher"
    checkout.mkdir()
    checkpoint = checkout / "goalkeeper.pt"
    checkpoint.write_bytes(b"not-the-pinned-checkpoint")

    with pytest.raises(ValueError, match="checkout and checkpoint together"):
        GoalkeeperPhysicsPPOConfig(
            combat_teacher_checkout=str(checkout.resolve()),
            teacher_pretraining_enabled=False,
        )
    with pytest.raises(ValueError, match="instead of arm pretraining"):
        GoalkeeperPhysicsPPOConfig(
            combat_teacher_checkout=str(checkout.resolve()),
            combat_teacher_checkpoint=str(checkpoint.resolve()),
        )
    mobile = GoalkeeperPhysicsPPOConfig(
        combat_teacher_checkout=str(checkout.resolve()),
        combat_teacher_checkpoint=str(checkpoint.resolve()),
        teacher_pretraining_enabled=False,
        mobility_option_enabled=True,
        maximum_combat_teacher_blend=0.75,
    )
    assert mobile.mobility_option_enabled
    assert mobile.maximum_combat_teacher_blend == pytest.approx(0.75)


def test_combat_teacher_blend_is_hard_bounded() -> None:
    import inspect

    from rosclaw_soccer.training.goalkeeper_combat_mjwarp import (
        GoalkeeperCombatMJWarpBatch,
    )

    config = GoalkeeperPhysicsPPOConfig()
    with pytest.raises(ValueError, match=r"\[0.05, 0.50\]"):
        replace(config, maximum_combat_teacher_blend=0.51)
    with pytest.raises(ValueError, match=r"\[0.05, 0.50\]"):
        replace(config, maximum_combat_teacher_blend=float("nan"))
    target_source = inspect.getsource(GoalkeeperCombatMJWarpBatch._locomotion_target)
    assert "blend = blend * self.maximum_teacher_blend" in target_source
    assert "blend *= self.maximum_teacher_blend" not in target_source
    with pytest.raises(ValueError, match="requires its pinned teacher"):
        replace(config, combat_gate_pretraining_batches=1)
    with pytest.raises(ValueError, match="batch count"):
        replace(config, combat_gate_pretraining_batches=65)
    with pytest.raises(ValueError, match="recenter selection"):
        replace(config, second_release_recenter_selection_weight=float("nan"))
    with pytest.raises(ValueError, match="second-shot reach"):
        replace(
            config,
            task_space_reach_blend=0.50,
            second_shot_reach_multiplier=2.0,
        )
    with pytest.raises(ValueError, match="intercept conditioning"):
        replace(config, combat_teacher_intercept_conditioning_enabled=True)
    with pytest.raises(ValueError, match="reach atlas"):
        replace(config, task_space_reach_atlas_enabled=True)
    with pytest.raises(ValueError, match="substep upper-body guard requires"):
        replace(config, mobility_substep_upper_body_guard_enabled=True)


def test_combat_teacher_can_bind_causal_intercept_semantics() -> None:
    import inspect

    from rosclaw_soccer.training.goalkeeper_combat_mjwarp import (
        GoalkeeperCombatMJWarpBatch,
    )
    from rosclaw_soccer.training.goalkeeper_cpu_exam import _run_episode

    gpu_source = inspect.getsource(GoalkeeperCombatMJWarpBatch._locomotion_target)
    cpu_source = inspect.getsource(_run_episode)
    assert "intercept = self._causal_intercept()" in gpu_source
    assert "teacher_target_visible = (self._shot_index > 0) | predictive" in gpu_source
    assert (
        "intercept\n                    if combat_teacher.intercept_conditioning_enabled"
        in cpu_source
    )
    assert "else np.asarray(data.qpos[36:39]" in cpu_source


def test_substep_upper_body_guard_preserves_damping_and_matches_torch() -> None:
    import inspect

    import numpy as np

    torch = pytest.importorskip("torch")
    from rosclaw_soccer.training.goalkeeper_combat_mjwarp import (
        GoalkeeperCombatMJWarpBatch,
    )
    from rosclaw_soccer.training.goalkeeper_cpu_exam import _run_episode
    from rosclaw_soccer.training.goalkeeper_mjwarp import GoalkeeperMJWarpBatch
    from rosclaw_soccer.training.goalkeeper_mobility_option import (
        GoalkeeperMobilityOptionConfig,
        substep_upper_body_authority_numpy,
        substep_upper_body_authority_torch,
    )

    config = GoalkeeperMobilityOptionConfig(
        substep_upper_body_guard_enabled=True,
        substep_upper_body_guard_onset_rad_s=1.5,
        substep_upper_body_guard_ceiling_rad_s=2.5,
        substep_upper_body_minimum_position_scale=0.10,
    )
    angular = np.asarray(((0.0, 0.0, 1.0), (0.0, 0.0, 2.0), (0.0, 0.0, 3.0)))
    numpy_values = np.asarray(
        [
            substep_upper_body_authority_numpy(
                root_angular_velocity_rad_s=row,
                config=config,
            )
            for row in angular
        ]
    )
    torch_values = substep_upper_body_authority_torch(
        root_angular_velocity_rad_s=torch.from_numpy(angular),
        config=config,
    )

    assert numpy_values.tolist() == pytest.approx((1.0, 0.55, 0.10))
    assert torch_values.tolist() == pytest.approx(numpy_values.tolist())
    gpu_source = inspect.getsource(GoalkeeperMJWarpBatch.step)
    cpu_source = inspect.getsource(_run_episode)
    assert "substep_authority = self._substep_upper_body_position_authority()" in gpu_source
    assert "position_torque[:, 12:] *= substep_authority" in gpu_source
    assert "position_torque[12:] *= substep_authority" in cpu_source
    assert "torque = position_torque -" in gpu_source
    assert "torque = position_torque -" in cpu_source
    assert GoalkeeperCombatMJWarpBatch.lower_body_authority.startswith("BOUNDED")


def test_official_goalkeeper_pose_matches_g1_action_contract() -> None:
    assert len(OFFICIAL_GOALKEEPER_DEFAULT_QPOS) == 29
    assert len(OFFICIAL_GOALKEEPER_INITIAL_QPOS) == 29
    assert len(OFFICIAL_GOALKEEPER_KP) == 29
    assert len(OFFICIAL_GOALKEEPER_KD) == 29
    assert all(math.isfinite(value) for value in OFFICIAL_GOALKEEPER_DEFAULT_QPOS)
    assert all(math.isfinite(value) for value in OFFICIAL_GOALKEEPER_INITIAL_QPOS)
    assert len(OFFICIAL_GOALKEEPER_BLEND_GROUP_SCALE) == 29
    assert max(OFFICIAL_GOALKEEPER_BLEND_GROUP_SCALE[:12]) < min(
        OFFICIAL_GOALKEEPER_BLEND_GROUP_SCALE[15:]
    )


def test_official_goalkeeper_rotation_uses_wxyz_inverse() -> None:
    torch = pytest.importorskip("torch")
    vectors = torch.tensor(((0.4, -0.2, 1.1), (1.0, 2.0, 3.0)))
    quaternions = torch.tensor(((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)))

    rotated = rotate_inverse_torch(
        torch=torch,
        quaternion_wxyz=quaternions,
        vector=vectors,
    )

    assert rotated[0].tolist() == pytest.approx(vectors[0].tolist())
    assert rotated[1].tolist() == pytest.approx((-1.0, -2.0, 3.0))


def test_combat_teacher_world_to_body_rotation_uses_wxyz_quaternion() -> None:
    torch = pytest.importorskip("torch")
    from rosclaw_soccer.training.goalkeeper_combat_mjwarp import (
        GoalkeeperCombatMJWarpBatch,
    )

    adapter = object.__new__(GoalkeeperCombatMJWarpBatch)
    adapter.torch = torch
    vectors = torch.tensor(((0.4, -0.2, 1.1), (1.0, 2.0, 3.0)))
    quaternions = torch.tensor(
        (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    )

    rotated = adapter._rotate_inverse(quaternions, vectors)

    assert rotated[0].tolist() == pytest.approx(vectors[0].tolist())
    assert rotated[1].tolist() == pytest.approx((-1.0, -2.0, 3.0))

    from rosclaw_soccer.training.goalkeeper_cpu_exam import _rotate_inverse

    assert _rotate_inverse(quaternions[0].numpy(), vectors[0].numpy()).tolist() == pytest.approx(
        vectors[0].tolist()
    )
    assert _rotate_inverse(quaternions[1].numpy(), vectors[1].numpy()).tolist() == pytest.approx(
        (-1.0, -2.0, 3.0)
    )


def test_combat_adapter_declares_safe_body_authority_split() -> None:
    from rosclaw_soccer.training.goalkeeper_combat_mjwarp import (
        COMBAT_ARM_RESIDUAL_LIMIT,
        COMBAT_SIGNED_LATERAL_GATE_LIMIT,
        COMBAT_WAIST_RESIDUAL_LIMIT,
        GoalkeeperCombatMJWarpBatch,
    )

    assert GoalkeeperCombatMJWarpBatch.lower_body_authority == (
        "BOUNDED_FROZEN_GOALKEEPER_TEACHER_BLEND"
    )
    assert GoalkeeperCombatMJWarpBatch.learned_residual_authority == (
        "SIGNED_STABLE_LATERAL_TEACHER_GATE_AND_BOUNDED_UPPER_BODY_RESIDUAL"
    )
    assert COMBAT_SIGNED_LATERAL_GATE_LIMIT <= 0.35
    assert COMBAT_WAIST_RESIDUAL_LIMIT < COMBAT_ARM_RESIDUAL_LIMIT <= 0.25


def test_combat_curriculum_rejects_excessive_task_space_authority() -> None:
    from rosclaw_soccer.training.goalkeeper_combat_curriculum import (
        pretrain_combat_teacher_gate,
    )

    with pytest.raises(ValueError, match="dimensions"):
        pretrain_combat_teacher_gate(
            torch=None,
            model=None,
            environment=None,
            device=None,
            batches=1,
            epochs=1,
            seed=1,
            task_space_reach_blend=0.86,
        )


def test_candidate_selection_uses_deterministic_mean_policy() -> None:
    import inspect

    from rosclaw_soccer.training.goalkeeper_physics_ppo import (
        _deterministic_candidate_rollout,
    )

    source = inspect.getsource(_deterministic_candidate_rollout)
    assert "torch.tanh(mean)" in source
    assert "distribution" not in source
    assert "inference_mode" in source


def test_causal_lateral_teacher_accelerates_then_brakes_without_future_state() -> None:
    torch = pytest.importorskip("torch")
    from rosclaw_soccer.training.goalkeeper_combat_curriculum import (
        causal_lateral_teacher_action,
    )

    observations = torch.zeros((3, 74), dtype=torch.float32)
    observations[0, 0] = -0.8  # ball is 2 m from the keeper
    observations[0, 3] = 1.0  # incoming at 5 m/s
    observations[0, 7] = 0.5  # intercept is 1 m toward world +y
    observations[0, -3] = 1.0  # first shot
    observations[1] = observations[0]
    observations[1, 7] = 0.01  # only 2 cm remain
    observations[1, 13] = 0.20  # but root still moves at 0.4 m/s
    observations[2, 7] = -0.20  # recovery: root is 0.4 m off centre
    observations[2, -4] = 1.0

    action = causal_lateral_teacher_action(
        torch=torch,
        observation=observations,
        gate_ceiling=0.35,
        maximum_lateral_command_mps=0.40,
        allow_recovery=True,
    )

    assert action[0].item() == pytest.approx(-0.35)
    assert action[1].item() > 0.0  # velocity-aware braking reverses before crossing
    assert action[2].item() == pytest.approx(0.25)  # recovery has a stricter cap
    assert bool(torch.all(torch.isfinite(action)))

    cue_observation = torch.zeros((1, 77), dtype=torch.float32)
    cue_observation[0, 7] = 0.5
    cue_observation[0, -5] = 1.0
    cue_observation[0, -4] = 1.0
    predictive = causal_lateral_teacher_action(
        torch=torch,
        observation=cue_observation,
        gate_ceiling=0.75,
        maximum_lateral_command_mps=0.40,
        allow_recovery=True,
        allow_full_predictive_cue=True,
    )
    assert predictive.item() == pytest.approx(-0.75)

    scheduled = torch.zeros((2, 77), dtype=torch.float32)
    scheduled[:, 7] = 0.05
    scheduled[:, -5] = 1.0
    scheduled[:, -4] = 1.0
    scheduled[1, -1] = 0.60 / 5.0
    scheduled_action = causal_lateral_teacher_action(
        torch=torch,
        observation=scheduled,
        gate_ceiling=0.75,
        maximum_lateral_command_mps=0.40,
        allow_recovery=True,
        allow_full_predictive_cue=True,
        release_schedule_sec=(0.70, 1.70, 3.05, 4.35),
        episode_duration_sec=5.0,
    )
    assert abs(scheduled_action[0].item()) < abs(scheduled_action[1].item())


def test_combat_recovery_guard_recenters_and_removes_upper_body_authority() -> None:
    torch = pytest.importorskip("torch")
    from rosclaw_soccer.training.goalkeeper_combat_mjwarp import (
        COMBAT_RECOVERY_LATERAL_GATE_LIMIT,
        COMBAT_SECOND_SHOT_ARM_RESIDUAL_LIMIT,
        GoalkeeperCombatMJWarpBatch,
    )
    from rosclaw_soccer.training.goalkeeper_mjwarp import GoalkeeperMJWarpConfig

    adapter = object.__new__(GoalkeeperCombatMJWarpBatch)
    adapter.torch = torch
    adapter.config = GoalkeeperMJWarpConfig(environment_count=2)
    adapter.qpos = torch.zeros((2, 43), dtype=torch.float32)
    adapter.qvel = torch.zeros((2, 41), dtype=torch.float32)
    adapter.qpos[0, 1] = 0.40
    adapter._shot_index = torch.zeros(2, dtype=torch.long)
    adapter._step_index = 20
    requested = torch.ones((2, 18), dtype=torch.float32)
    requested[0, 0] = -1.0  # unsafe outward sign must not pass through

    shaped, authority = adapter._shape_actor_action(requested)

    assert shaped[0, 0].item() == pytest.approx(COMBAT_RECOVERY_LATERAL_GATE_LIMIT)
    assert shaped[1, 0].item() == pytest.approx(0.0)
    assert torch.count_nonzero(shaped[:, 1:]).item() == 0
    assert authority.tolist() == pytest.approx([1.0, 1.0])
    assert pytest.approx(0.30) == COMBAT_SECOND_SHOT_ARM_RESIDUAL_LIMIT

    adapter._shot_index.fill_(2)
    second_shot, _ = adapter._shape_actor_action(requested)
    assert torch.max(second_shot[:, 4:]).item() == pytest.approx(
        COMBAT_SECOND_SHOT_ARM_RESIDUAL_LIMIT
    )


def test_mobile_option_separates_lateral_teacher_gate_and_recovery_position() -> None:
    torch = pytest.importorskip("torch")
    from rosclaw_soccer.training.goalkeeper_combat_mjwarp import (
        GoalkeeperCombatMJWarpBatch,
    )
    from rosclaw_soccer.training.goalkeeper_mjwarp import GoalkeeperMJWarpConfig
    from rosclaw_soccer.training.goalkeeper_mobility_option import (
        MOBILE_TEACHER_GROUP_SCALE,
        GoalkeeperMobilityOptionConfig,
        guard_lateral_velocity_numpy,
        project_recovery_command_numpy,
    )

    mobility = GoalkeeperMobilityOptionConfig()
    assert mobility.residual_plasticity_scale == 0.0
    assert mobility.effective_waist_plasticity_scale == 0.0
    assert mobility.effective_arm_plasticity_scale == 0.0
    assert mobility.teacher_group_scale == MOBILE_TEACHER_GROUP_SCALE
    assert len(mobility.teacher_group_scale) == 29
    assert not mobility.counter_rotation_enabled
    adapter = object.__new__(GoalkeeperCombatMJWarpBatch)
    adapter.torch = torch
    adapter.config = GoalkeeperMJWarpConfig(environment_count=1)
    adapter.mobility_option_enabled = True
    adapter.mobility_option_config = mobility
    adapter.qpos = torch.zeros((1, 43), dtype=torch.float32)
    adapter.qvel = torch.zeros((1, 41), dtype=torch.float32)
    adapter._shot_index = torch.ones(1, dtype=torch.long)
    adapter._step_index = 20
    adapter._mobility_teacher_gate = torch.zeros(1)
    requested = torch.ones((1, 18), dtype=torch.float32)

    shaped, _ = adapter._shape_actor_action(requested)

    assert shaped[0, 0].item() == pytest.approx(mobility.lateral_command_limit)
    assert shaped[0, 1].item() == pytest.approx(0.0)
    assert adapter._mobility_teacher_gate.item() == pytest.approx(0.12)
    assert torch.allclose(shaped[0, 2:], torch.zeros_like(shaped[0, 2:]))

    decoupled = replace(
        mobility,
        waist_residual_plasticity_scale=0.10,
        arm_residual_plasticity_scale=0.60,
    )
    adapter.mobility_option_config = decoupled
    adapter._mobility_teacher_gate.zero_()
    shaped, _ = adapter._shape_actor_action(requested)
    assert torch.allclose(shaped[0, 2:4], torch.full((2,), 0.10))
    assert torch.allclose(shaped[0, 4:], torch.full((14,), 0.35))
    assert project_recovery_command_numpy(
        requested=-0.5,
        root_lateral_position_m=1.0,
        root_lateral_velocity_mps=0.0,
        config=mobility,
    ) == pytest.approx(0.5)
    assert project_recovery_command_numpy(
        requested=-0.5,
        root_lateral_position_m=0.3,
        root_lateral_velocity_mps=0.0,
        config=mobility,
    ) == pytest.approx(-0.5)
    assert project_recovery_command_numpy(
        requested=-0.7,
        root_lateral_position_m=0.3,
        root_lateral_velocity_mps=0.0,
        config=mobility,
        predictive_threat=True,
    ) == pytest.approx(-0.7)
    guarded = GoalkeeperMobilityOptionConfig(
        lateral_command_limit=1.0,
        lateral_velocity_guard_enabled=True,
    )
    assert guard_lateral_velocity_numpy(
        requested=-0.8,
        root_lateral_velocity_mps=0.60,
        config=guarded,
    ) == pytest.approx(-0.8 * (0.85 - 0.60) / (0.85 - 0.55))
    assert guard_lateral_velocity_numpy(
        requested=-0.8,
        root_lateral_velocity_mps=0.90,
        config=guarded,
    ) == pytest.approx(0.35)
    assert guard_lateral_velocity_numpy(
        requested=0.8,
        root_lateral_velocity_mps=0.90,
        config=guarded,
    ) == pytest.approx(0.8)


def test_visible_intent_unlocks_only_bounded_predictive_arms() -> None:
    torch = pytest.importorskip("torch")
    from rosclaw_soccer.training.goalkeeper_combat_mjwarp import (
        GoalkeeperCombatMJWarpBatch,
    )
    from rosclaw_soccer.training.goalkeeper_mjwarp import GoalkeeperMJWarpConfig
    from rosclaw_soccer.training.goalkeeper_mobility_option import (
        GoalkeeperMobilityOptionConfig,
    )

    adapter = object.__new__(GoalkeeperCombatMJWarpBatch)
    adapter.torch = torch
    adapter.config = GoalkeeperMJWarpConfig(
        environment_count=1,
        shot_intent_cue_enabled=True,
    )
    adapter.mobility_option_enabled = True
    adapter.mobility_option_config = GoalkeeperMobilityOptionConfig(
        residual_plasticity_scale=0.35,
        anticipatory_arm_reach_enabled=True,
    )
    adapter.qpos = torch.zeros((1, 43), dtype=torch.float32)
    adapter.qvel = torch.zeros((1, 41), dtype=torch.float32)
    adapter._shot_index = torch.zeros(1, dtype=torch.long)
    adapter._step_index = 2
    adapter._intent_cue_one = torch.tensor([[0.4, 1.2, 1.0]])
    adapter._intent_cue_two = torch.zeros((1, 3))
    adapter._mobility_teacher_gate = torch.zeros(1)

    shaped, _ = adapter._shape_actor_action(torch.ones((1, 18)))

    assert shaped[0, 0].item() == pytest.approx(0.75)
    assert torch.count_nonzero(shaped[0, 1:4]) == 0
    assert torch.count_nonzero(shaped[0, 4:]) == 14
    assert torch.max(torch.abs(shaped[0, 4:])).item() <= 0.35

    adapter.mobility_option_config = replace(
        adapter.mobility_option_config,
        predictive_teacher_warmstart_enabled=True,
    )
    adapter._mobility_teacher_gate.zero_()
    gate_request = torch.zeros((1, 18))
    gate_request[:, 1] = 1.0
    shaped, _ = adapter._shape_actor_action(gate_request)
    assert adapter._mobility_teacher_gate.item() == pytest.approx(0.12)
    assert shaped[0, 1].item() == 0.0


def test_predictive_teacher_floor_is_bounded_and_requires_visible_threat() -> None:
    torch = pytest.importorskip("torch")
    from rosclaw_soccer.training.goalkeeper_combat_mjwarp import (
        GoalkeeperCombatMJWarpBatch,
    )
    from rosclaw_soccer.training.goalkeeper_mjwarp import GoalkeeperMJWarpConfig
    from rosclaw_soccer.training.goalkeeper_mobility_option import (
        GoalkeeperMobilityOptionConfig,
    )

    adapter = object.__new__(GoalkeeperCombatMJWarpBatch)
    adapter.torch = torch
    adapter.config = GoalkeeperMJWarpConfig(
        environment_count=1,
        shot_intent_cue_enabled=True,
    )
    adapter.mobility_option_enabled = True
    adapter.mobility_option_config = GoalkeeperMobilityOptionConfig(
        residual_plasticity_scale=0.10,
        anticipatory_arm_reach_enabled=True,
        predictive_teacher_warmstart_enabled=True,
        predictive_teacher_gate_floor=0.60,
    )
    adapter.qpos = torch.zeros((1, 43), dtype=torch.float32)
    adapter.qvel = torch.zeros((1, 41), dtype=torch.float32)
    adapter._shot_index = torch.zeros(1, dtype=torch.long)
    adapter._step_index = 2
    adapter._intent_cue_one = torch.tensor([[0.9, 0.3, 1.0]])
    adapter._intent_cue_two = torch.zeros((1, 3))
    adapter._mobility_teacher_gate = torch.zeros(1)
    requested = torch.zeros((1, 18), dtype=torch.float32)

    for _ in range(6):
        adapter._shape_actor_action(requested)
    assert 0.55 < adapter._mobility_teacher_gate.item() <= 0.60

    adapter._intent_cue_one.zero_()
    for _ in range(12):
        adapter._shape_actor_action(requested)
    assert adapter._mobility_teacher_gate.item() < 1.0e-3

    with pytest.raises(ValueError, match="predictive teacher-gate floor"):
        GoalkeeperMobilityOptionConfig(predictive_teacher_gate_floor=0.81)


def test_teacher_recovery_latch_finishes_option_instead_of_abrupt_cutoff() -> None:
    torch = pytest.importorskip("torch")
    from rosclaw_soccer.training.goalkeeper_combat_mjwarp import (
        GoalkeeperCombatMJWarpBatch,
    )
    from rosclaw_soccer.training.goalkeeper_mjwarp import GoalkeeperMJWarpConfig
    from rosclaw_soccer.training.goalkeeper_mobility_option import (
        GoalkeeperMobilityOptionConfig,
    )

    adapter = object.__new__(GoalkeeperCombatMJWarpBatch)
    adapter.torch = torch
    adapter.config = GoalkeeperMJWarpConfig(environment_count=1)
    adapter.mobility_option_enabled = True
    adapter.mobility_option_config = GoalkeeperMobilityOptionConfig(
        teacher_recovery_latch_enabled=True,
    )
    adapter.qpos = torch.zeros((1, 43), dtype=torch.float32)
    adapter.qvel = torch.zeros((1, 41), dtype=torch.float32)
    adapter._shot_index = torch.ones(1, dtype=torch.long)
    adapter._step_index = 20
    adapter._mobility_teacher_gate = torch.zeros(1)
    adapter._teacher_recovery_gate = torch.zeros(1)
    adapter._teacher_recovery_age_steps = torch.full((1,), -1, dtype=torch.long)
    adapter._previous_teacher_shot_active = torch.zeros(1, dtype=torch.bool)
    adapter._teacher_recovery_active = torch.zeros(1, dtype=torch.bool)
    requested = torch.zeros((1, 18), dtype=torch.float32)
    requested[:, 1] = 1.0

    adapter._shape_actor_action(requested)
    active_gate = float(adapter._mobility_teacher_gate.item())
    assert active_gate > 0.0
    assert bool(adapter._previous_teacher_shot_active[0])

    adapter._shot_index.zero_()
    adapter._shape_actor_action(torch.zeros_like(requested))

    assert bool(adapter._teacher_recovery_active[0])
    assert adapter._teacher_recovery_age_steps.item() == 1
    assert adapter._mobility_teacher_gate.item() == pytest.approx(active_gate)


def test_cpu_replay_blends_predictive_and_recovery_teacher_states() -> None:
    import inspect

    from rosclaw_soccer.training.goalkeeper_cpu_exam import _run_episode

    source = inspect.getsource(_run_episode)
    assert "shot_index > 0 or predictive_ready or teacher_recovery_active" in source
