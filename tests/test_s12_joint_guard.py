from __future__ import annotations

import pytest

from rosclaw_soccer.training.joint_guard import (
    project_joint_safe_torque_numpy,
    project_joint_safe_torque_torch,
)


def test_numpy_joint_guard_brakes_only_threatened_direction() -> None:
    import numpy as np

    position = np.zeros(29)
    velocity = np.zeros(29)
    torque = np.zeros(29)
    ranges = np.asarray([[-1.0, 1.0]] * 29)
    limited = np.ones(29, dtype=np.bool_)
    position[3] = 0.97
    velocity[3] = 1.0
    torque[3] = 50.0
    projected, active = project_joint_safe_torque_numpy(
        joint_position=position,
        joint_velocity=velocity,
        commanded_torque=torque,
        joint_ranges=ranges,
        limited=limited,
    )
    assert projected[3] < torque[3]
    assert active


def test_torch_joint_guard_brakes_only_threatened_direction() -> None:
    torch = pytest.importorskip("torch")
    position = torch.zeros((2, 29))
    velocity = torch.zeros_like(position)
    torque = torch.zeros_like(position)
    ranges = torch.tensor([[-1.0, 1.0]] * 29)
    limited = torch.ones(29, dtype=torch.bool)
    position[0, 3] = 0.97
    velocity[0, 3] = 1.0
    torque[0, 3] = 50.0
    position[1, 4] = -0.97
    velocity[1, 4] = -1.0
    torque[1, 4] = -50.0

    projected, active = project_joint_safe_torque_torch(
        joint_position=position,
        joint_velocity=velocity,
        commanded_torque=torque,
        joint_ranges=ranges,
        limited=limited,
    )
    assert projected[0, 3] < torque[0, 3]
    assert projected[1, 4] > torque[1, 4]
    assert bool(active[0, 3]) and bool(active[1, 4])
    assert torch.count_nonzero(projected[:, :3]) == 0


def test_torch_joint_guard_validates_contract() -> None:
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="29-DoF"):
        project_joint_safe_torque_torch(
            joint_position=torch.zeros((2, 28)),
            joint_velocity=torch.zeros((2, 28)),
            commanded_torque=torch.zeros((2, 28)),
            joint_ranges=torch.zeros((29, 2)),
            limited=torch.ones(29, dtype=torch.bool),
        )
