from __future__ import annotations

import pytest

from rosclaw_soccer.training import goalkeeper_ppo
from rosclaw_soccer.training.goalkeeper_ppo import (
    GoalkeeperPPOConfig,
    GoalkeeperPPOTrainingReport,
    GoalkeeperResetCurriculum,
)


def test_ppo_config_and_continuous_reset_are_content_bound() -> None:
    config = GoalkeeperPPOConfig(environments_per_rank=64, iterations=2)
    curriculum = GoalkeeperResetCurriculum()

    assert config.config_hash.startswith("sha256:")
    assert config.maximum_joint_residual_scale == 0.25
    assert config.paired_mirror_curriculum
    assert config.hard_negative_fraction == 0.35
    assert curriculum.curriculum_hash.startswith("sha256:")
    assert set(curriculum.phase_names) == {"ready", "flight", "landing", "recovery"}
    assert not curriculum.terminates_on_save
    assert not curriculum.terminates_on_ball_contact
    assert curriculum.continuous_episode_sec >= 3.0


def test_continuous_reset_rejects_contact_termination_or_bad_probabilities() -> None:
    with pytest.raises(ValueError, match="cannot terminate"):
        GoalkeeperResetCurriculum(terminates_on_save=True)
    with pytest.raises(ValueError, match="sum to one"):
        GoalkeeperResetCurriculum(phase_probabilities=(0.1, 0.1, 0.1, 0.1))
    with pytest.raises(ValueError, match="exceeds"):
        GoalkeeperPPOConfig(maximum_joint_residual_scale=1.01)


def test_training_report_does_not_claim_multistep_episode_training() -> None:
    assert not GoalkeeperPPOTrainingReport.__dataclass_fields__[
        "multi_step_episode_training"
    ].default
    assert not GoalkeeperPPOTrainingReport.__dataclass_fields__["continuous_reset_enabled"].default


def test_export_contract_keeps_neural_legs_below_a_small_safety_ceiling() -> None:
    assert len(goalkeeper_ppo._JOINT_LIMITS) == 29
    config = GoalkeeperPPOConfig()
    assert 0.0 < config.maximum_leg_residual_scale <= 0.30
    assert config.maximum_leg_residual_scale < config.maximum_joint_residual_scale + 0.10


def test_mirror_consistency_loss_accepts_reflected_pair() -> None:
    torch = pytest.importorskip("torch")
    left = torch.zeros((1, 30))
    left[0, 0] = 0.6
    left[0, 1 + 12] = 0.2
    left[0, 1 + 13] = -0.3
    left[0, 1 + 14] = 0.1
    left[0, 1 + 15 : 1 + 22] = torch.arange(7) * 0.1
    right = torch.zeros((1, 30))
    right[0, 0] = -0.6
    right[0, 1 + 12] = -0.2
    right[0, 1 + 13] = 0.3
    right[0, 1 + 14] = 0.1
    right[0, 1 + 22 : 1 + 29] = torch.arange(7) * 0.1

    loss = goalkeeper_ppo._paired_mirror_consistency_loss(
        torch=torch,
        action=torch.cat((left, right), dim=0),
        phases=torch.ones(2, dtype=torch.long),
    )

    assert float(loss) == pytest.approx(0.0)
