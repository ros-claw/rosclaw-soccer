from __future__ import annotations

import pytest

from rosclaw_soccer.training.goalkeeper_teacher import (
    _motiondecode_teacher_action,
    goalkeeper_teacher_action,
)


def test_teacher_is_causal_mirrored_and_returns_to_ready() -> None:
    torch = pytest.importorskip("torch")
    observation = torch.zeros((3, 74))
    observation[0, 7:9] = torch.tensor((-0.40, 0.60))
    observation[1, 7:9] = torch.tensor((0.40, 0.60))
    observation[:2, 71] = 1.0
    action = goalkeeper_teacher_action(observation)

    assert action.shape == (3, 18)
    assert action[0, 0] == pytest.approx(-action[1, 0])
    assert torch.count_nonzero(action[0, 4:11]) > 0
    assert torch.count_nonzero(action[1, 11:18]) > 0
    assert float(action[0, 5]) < 0.0
    assert float(action[1, 12]) > 0.0
    assert torch.count_nonzero(action[2]) == 6
    assert action[2, 5] == pytest.approx(-action[2, 12])


def test_teacher_rejects_invalid_observation() -> None:
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="74-D"):
        goalkeeper_teacher_action(torch.zeros((2, 73)))


def test_teacher_distinguishes_world_high_and_low_from_relative_height() -> None:
    torch = pytest.importorskip("torch")
    observation = torch.zeros((2, 74))
    # World heights 1.30 and 0.35 m around the 0.77 m qualified pelvis.
    observation[:, 7] = -0.35
    observation[:, 8] = torch.tensor(((1.30 - 0.77) * 0.5, (0.35 - 0.77) * 0.5))
    observation[:, 71] = 1.0
    action = goalkeeper_teacher_action(observation)
    assert abs(float(action[0, 5])) > abs(float(action[1, 5]))
    assert action[0, 7] < action[1, 7]


def test_teacher_uses_both_arms_for_central_high_save() -> None:
    torch = pytest.importorskip("torch")
    observation = torch.zeros((1, 74))
    observation[0, 7] = 0.0
    observation[0, 8] = (1.30 - 0.77) * 0.5
    observation[0, 71] = 1.0

    action = goalkeeper_teacher_action(observation)

    assert torch.linalg.vector_norm(action[0, 4:11]) > 0.50
    assert torch.linalg.vector_norm(action[0, 11:18]) > 0.50
    assert float(action[0, 7]) < 0.0
    assert float(action[0, 14]) < 0.0
    assert torch.count_nonzero(action[0, (8, 9, 10, 15, 16, 17)]) == 6


def test_motiondecode_proxy_selects_mirrored_upper_body_families() -> None:
    torch = pytest.importorskip("torch")
    names = (
        "ready",
        "split_step",
        "shuffle_left",
        "shuffle_right",
        "low_save_left",
        "low_save_right",
        "high_reach_left",
        "high_reach_right",
        "center_block",
        "recovery",
    )
    table = {name: (float(index),) * 17 for index, name in enumerate(names)}
    observation = torch.zeros((4, 74))
    observation[:, 71] = 1.0
    observation[:, 7] = torch.tensor((-0.4, 0.4, -0.4, 0.0))
    observation[:, 8] = torch.tensor((0.30, 0.30, -0.20, 0.30))

    action = _motiondecode_teacher_action(observation, motion_table=table)

    assert action[:, 0].count_nonzero() == 0
    assert action[0, 1] == names.index("high_reach_left")
    assert action[1, 1] == names.index("high_reach_right")
    assert action[2, 1] == names.index("low_save_left")
    assert action[3, 1] == names.index("center_block")
