import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.bounded_residual_policy_gradient import optimize_residual


def inputs():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(120, 3))
    action = rng.normal(scale=0.1, size=(120, 2))
    advantage = action[:, 0] / 0.1
    layers = [dict(weight=np.zeros((2, 3)).tolist(), bias=[0.0, 0.0])]
    return layers, x, np.zeros((120, 2)), np.ones(120), action, np.full(120, 0.1), advantage


def test_numeric_optimizer_keeps_bounded_monotonic_learning_without_mutating_inputs():
    args = inputs()
    before = copy.deepcopy(args)
    result = optimize_residual(*args, learning_rate=4e-4)
    assert result["completed_optimizer_steps"] > 0
    assert 0 <= result["torch_mean_latent_kl"] <= 0.0049
    assert all(
        a > b
        for a, b in zip(
            result["full_batch_loss_history"], result["full_batch_loss_history"][1:], strict=False
        )
    )
    assert result["promotion_authorized"] is False
    assert result["hardware_authorized"] is False
    assert args[0] == before[0]
    assert all(np.array_equal(a, b) for a, b in zip(args[1:], before[1:], strict=True))


@pytest.mark.parametrize(
    "key,value",
    [
        ("learning_rate", True),
        ("learning_rate", 0.01),
        ("learning_rate", np.nan),
        ("steps", True),
        ("steps", 161),
        ("raw_cap", np.inf),
        ("seed", False),
    ],
)
def test_unsafe_or_unbounded_numeric_configuration_is_rejected(key, value):
    with pytest.raises(ValueError):
        optimize_residual(*inputs(), **{key: value})


def test_closed_gate_cannot_be_bypassed_to_create_learning():
    args = list(inputs())
    args[3] = np.zeros(120)
    with pytest.raises(ValueError, match="no actual"):
        optimize_residual(*args)


def test_nonfinite_data_is_rejected_before_torch_optimization():
    args = list(inputs())
    args[4][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite bounded"):
        optimize_residual(*args)
