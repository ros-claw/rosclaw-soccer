from __future__ import annotations

import inspect

import numpy as np
import pytest

from rosclaw_soccer.training.opentrack_recovery_residual_ppo import (
    _RecoveryResidualPhysics,
)
from rosclaw_soccer.training.recovery_residual_ppo import (
    FailurePrioritizedRecoveryCurriculum,
    RecoveryCurriculumState,
    RecoveryResidualObservationSpec,
    RecoveryResidualPPOConfig,
    build_recovery_residual_actor_observation,
    compute_recovery_residual_reward,
    generalized_advantage_estimate,
    recovery_successor_potential,
)

_HASHES = tuple("sha256:" + character * 64 for character in "abcdef")


def test_residual_actor_observation_contains_internal_error_not_reference() -> None:
    spec = RecoveryResidualObservationSpec()
    result = build_recovery_residual_actor_observation(
        proprioception=np.zeros(93),
        internal_memory_target_rad=np.full(29, 0.30),
        joint_position_rad=np.full(29, 0.10),
        spec=spec,
    )
    assert result.shape == (122,)
    assert result[-29:] == pytest.approx(np.full(29, 0.20))
    assert not set(spec.actor_features) & set(spec.forbidden_actor_features)
    assert "external_reference_phase" in spec.forbidden_actor_features
    assert "teacher_identity" in spec.forbidden_actor_features


def test_residual_config_has_grouped_authority_and_rejects_hardware() -> None:
    config = RecoveryResidualPPOConfig(iterations=2, rollout_steps=128)
    limits = config.residual_limits_rad
    assert limits.shape == (29,)
    assert limits[:12] == pytest.approx(np.full(12, 0.15))
    assert limits[12:15] == pytest.approx(np.full(3, 0.15))
    assert limits[15:] == pytest.approx(np.full(14, 0.18))
    with pytest.raises(ValueError, match="invalid"):
        RecoveryResidualPPOConfig(hardware_authorized=True)
    with pytest.raises(ValueError, match="invalid"):
        RecoveryResidualPPOConfig(residual_limit_lower_body_rad=0.30)


def test_failure_curriculum_covers_every_source_and_records_results() -> None:
    sources = (
        "CAPABILITY_FRONTIER",
        "RECENT_FAILURE",
        "HISTORICAL_ANCHOR",
        "NIGHTMARE",
        "SOCIAL_TEACHER",
    )
    states = tuple(
        RecoveryCurriculumState(
            state_hash=_HASHES[index],
            base_snapshot_hash=_HASHES[-1],
            source=source,
            difficulty=index / 5.0,
        )
        for index, source in enumerate(sources)
    )
    curriculum = FailurePrioritizedRecoveryCurriculum(states, seed=7)
    assert sum(curriculum.source_weights.values()) == pytest.approx(1.0)
    sampled = curriculum.sample()
    assert curriculum.sample(source="SOCIAL_TEACHER").source == "SOCIAL_TEACHER"
    curriculum.record(sampled.state_hash, succeeded=False)
    metrics = curriculum.metrics()
    assert sum(row["attempts"] for row in metrics["by_source"].values()) == 1
    assert sum(row["successes"] for row in metrics["by_source"].values()) == 0


def test_successor_reward_prefers_progress_and_penalizes_terminal_failure() -> None:
    fallen = recovery_successor_potential(
        pelvis_height_m=0.10,
        upright_projection=-0.50,
        root_linear_speed_mps=0.20,
        root_angular_speed_rad_s=0.30,
    )
    ready = recovery_successor_potential(
        pelvis_height_m=0.72,
        upright_projection=0.95,
        root_linear_speed_mps=0.10,
        root_angular_speed_rad_s=0.20,
    )
    progress = compute_recovery_residual_reward(
        previous_potential=fallen,
        current_potential=ready,
        nominal_tracking_rmse_rad=0.05,
        normalized_residual_rms=0.10,
        normalized_residual_delta_rms=0.05,
        torque_saturation_fraction=0.01,
        stable=True,
        succeeded=True,
        failed=False,
    )
    failure = compute_recovery_residual_reward(
        previous_potential=fallen,
        current_potential=fallen,
        nominal_tracking_rmse_rad=0.50,
        normalized_residual_rms=1.0,
        normalized_residual_delta_rms=1.0,
        torque_saturation_fraction=0.50,
        stable=False,
        succeeded=False,
        failed=True,
    )
    assert ready > fallen
    assert progress > 30.0
    assert failure < -10.0


def test_generalized_advantage_estimate_respects_terminal_boundary() -> None:
    advantage, returns = generalized_advantage_estimate(
        rewards=np.asarray((1.0, 2.0), dtype=np.float32),
        values=np.asarray((0.5, 0.25), dtype=np.float32),
        dones=np.asarray((False, True)),
        bootstrap_value=100.0,
        discount=1.0,
        gae_lambda=1.0,
    )
    assert advantage == pytest.approx((2.5, 1.75))
    assert returns == pytest.approx((3.0, 2.0))


def test_residual_physics_control_path_has_no_reference_or_env_step() -> None:
    source = inspect.getsource(_RecoveryResidualPhysics)
    for forbidden in (
        "environment.step(",
        "trajectory_handler",
        "reference_phase",
        "teacher_identity",
        "ref_mj_data",
    ):
        assert forbidden not in source
