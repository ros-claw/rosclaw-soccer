import numpy as np
import pytest
import torch

from rosclaw_soccer.rsi.receiving_command_actor import ReceivingCommandActor
from rosclaw_soccer.rsi.receiving_task_command_actor import (
    ReceivingTaskCommandActor,
    receiving_task_context6,
)


@pytest.mark.parametrize("rows", [1, 32, 2048])
def test_zero_projection_exact_parent_function_not_new_random_policy(rows):
    torch.manual_seed(123)
    parent = ReceivingCommandActor(torch.zeros(112), torch.ones(112)).eval().requires_grad_(False)
    child = ReceivingTaskCommandActor(parent, torch.zeros(6), torch.ones(6))
    x = torch.randn(rows, 118)
    with torch.no_grad():
        assert torch.equal(child(x), parent(x[:, :112]))
        x[:, 112:] *= 100
        assert torch.equal(child(x), parent(x[:, :112]))
    assert len(tuple(child.parameters())) == 7
    assert all(p.requires_grad for p in child.parameters())
    assert all(not p.requires_grad for p in parent.parameters())
    child(x).square().mean().backward()
    assert all(p.grad is not None and p.grad.norm() > 0 for p in child.parameters())


def test_task_context_is_bodyframe_and_distinguishes_aliased_task_messages():
    root = np.array([0.0, 0.0, 0.78, 1.0, 0.0, 0.0, 0.0])
    anchor = np.array([0.25, -0.15, 0.11])
    features = []
    for offset in (0.0, 0.5):
        a = -np.pi / 4 + offset
        f = np.array([np.cos(a), np.sin(a)])
        target = anchor.copy()
        target[:2] -= 0.25 * f + 0.12 * np.array([-f[1], f[0]])
        features.append(receiving_task_context6(root, target, f))
    assert not np.array_equal(features[0], features[1])
    assert features[0].shape == (6,) and features[0].dtype == np.float32
    rotated_root = np.array([0.0, 0.0, 0.78, np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
    task = receiving_task_context6(rotated_root, np.array([1.0, 0.0, 0.11]), np.array([1.0, 0.0]))
    np.testing.assert_allclose(task, [0.0, -1.0, -0.67, 0.0, -1.0, 0.0], atol=1e-6)


@pytest.mark.parametrize("fault", ["root", "goal", "direction", "quaternion"])
def test_invalid_task_conditions_are_rejected_not_sanitized(fault):
    root = np.array([0.0, 0.0, 0.78, 1.0, 0.0, 0.0, 0.0])
    goal, forward = np.array([1.0, 0.0, 0.11]), np.array([1.0, 0.0])
    if fault == "root":
        root[0] = float("nan")
    elif fault == "goal":
        goal = goal.astype(np.float32)
    elif fault == "direction":
        forward *= 2
    else:
        root[3] = 0
    with pytest.raises(ValueError):
        receiving_task_context6(root, goal, forward)
