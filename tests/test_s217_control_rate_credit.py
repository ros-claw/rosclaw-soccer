import numpy as np
import pytest

from rosclaw_soccer.training.near_ball_residual_ppo import episodic_gae


def test_delayed_pass_credit_at_measured_50hz_gap() -> None:
    reward = np.zeros((89, 8))
    reward[-1, :2] = 1
    short, _ = episodic_gae(reward, np.zeros_like(reward))
    long, _ = episodic_gae(reward, np.zeros_like(reward), gamma=0.997, trace_decay=0.997)
    assert short[0, 0] == pytest.approx((0.99 * 0.95) ** 88)
    assert long[0, 0] == pytest.approx((0.997 * 0.997) ** 88)
    assert long[0, 0] > 100 * short[0, 0]
    np.testing.assert_array_equal(long[:, 2:], np.zeros((89, 6)))
    np.testing.assert_array_equal(long[-1], short[-1])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.5, 1.0])
def test_invalid_credit_parameters_rejected(bad: float) -> None:
    with pytest.raises(ValueError):
        episodic_gae(np.zeros((2, 8)), np.zeros((2, 8)), gamma=bad)
