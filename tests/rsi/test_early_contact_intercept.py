import numpy as np
import pytest

from rosclaw_soccer.rsi.early_contact_intercept import early_contact_intercept


def v(x):
    return np.asarray(x, dtype=np.float64)


def test_incoming_ball_has_causal_closest_approach():
    result = early_contact_intercept(v([1.5, 0.2, 0.0]), v([-3.0, 0.0, 0.0]))
    assert result["eligible"]
    assert result["eta_s"] == pytest.approx(0.5)
    assert result["miss_m"] == pytest.approx(0.2)


@pytest.mark.parametrize(
    "position,velocity",
    [
        ([1, 0, 0], [1, 0, 0]),
        ([1, 1, 0], [-1, 0, 0]),
        ([2, 0, 0], [-1, 0, 0]),
        ([1, 0, 0], [0, 0, 0]),
    ],
)
def test_moving_away_late_wide_or_stationary_never_activates(position, velocity):
    assert not early_contact_intercept(v(position), v(velocity))["eligible"]


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        early_contact_intercept(v([float("nan"), 0, 0]), v([0, 0, 0]))
    with pytest.raises(ValueError):
        early_contact_intercept(np.zeros(3, np.float32), v([0, 0, 0]))
