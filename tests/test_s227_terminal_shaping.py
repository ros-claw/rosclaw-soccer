import numpy as np
import pytest

from rosclaw_soccer.training.football_reward_shaping import terminal_approach_shaping


@pytest.mark.parametrize("gamma", [0.9, 0.99, 0.997])
@pytest.mark.parametrize("length", [1, 3, 600])
def test_discounted_shaping_has_no_terminal_ball_hoarding_bonus(gamma, length):
    rng = np.random.default_rng(227)
    distance = rng.uniform(0, 5, (length, 8))
    rewards = terminal_approach_shaping(distance, gamma=gamma)
    discounted = (rewards * gamma ** np.arange(length)[:, None]).sum(axis=0)
    np.testing.assert_allclose(discounted, -2 * np.exp(-4 * distance[0]), atol=1e-12)
    # Change the final ball position, keeping the starting state fixed.
    if length > 1:
        distance[-1] = 0
        alternative = terminal_approach_shaping(distance, gamma=gamma)
        np.testing.assert_allclose(
            (alternative * gamma ** np.arange(length)[:, None]).sum(axis=0), discounted, atol=1e-12
        )


@pytest.mark.parametrize("gamma", [True, 1.0, 0.89, float("nan")])
def test_bad_discount_fails(gamma):
    with pytest.raises(ValueError):
        terminal_approach_shaping(np.ones((3, 8)), gamma=gamma)


@pytest.mark.parametrize(
    "distance", [np.zeros((0, 8)), np.zeros((3, 7)), -np.ones((3, 8)), np.full((3, 8), np.nan)]
)
def test_bad_potential_fails(distance):
    with pytest.raises(ValueError):
        terminal_approach_shaping(distance, gamma=0.997)
