import numpy as np
import pytest

from rosclaw_soccer.rsi.motion_value_calibration import (
    calibration_split,
    task_returns,
    value_metrics,
)


def test_full_terminal_return_no_critic_bootstrap_and_prefix_not_in_metrics():
    r = np.zeros((2, 300), dtype=np.float32)
    r[:, 299] = 1
    y = task_returns(r)
    assert y[0, 299] == 1
    assert y[0, 298] == pytest.approx(0.99)
    assert y[0, 20] == pytest.approx(0.99**279)
    assert value_metrics(y, y)["explained_variance"] == 1
    p = y.copy()
    p[:, :20] = 123
    assert value_metrics(p, y)["mse"] == 0
    assert value_metrics(np.zeros_like(y), y)["explained_variance"] == pytest.approx(0)
    assert value_metrics(np.zeros_like(y), np.zeros_like(y))["explained_variance"] is None


def test_split_is_context_level_and_independent_of_rewards():
    ids = sorted(set(range(40)) - {7, 13, 20, 23, 25, 28, 30, 38})
    fit, hold = calibration_split(ids)
    assert len(fit) == 24 and len(hold) == 8
    assert not set(fit) & set(hold)
    assert sorted(fit + hold) == list(range(32))
    with pytest.raises(ValueError):
        calibration_split(ids[::-1])


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((0, 300), dtype=np.float32),
        np.zeros((2, 299), dtype=np.float32),
        np.zeros((2, 300), dtype=np.float64),
        np.full((2, 300), np.nan, dtype=np.float32),
    ],
)
def test_bad_targets_never_enter_value_learning(bad):
    with pytest.raises(ValueError):
        task_returns(bad)
