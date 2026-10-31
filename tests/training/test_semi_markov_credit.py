import numpy as np
import pytest

from rosclaw_soccer.training.semi_markov_credit import duration_aware_gae


def test_terminal_reward_crosses_inactive_gap_with_real_elapsed_discount():
    advantage, target = duration_aware_gae([0, 0, 0, -2], [0], [0], terminated=True, gamma=0.9)
    assert np.allclose(advantage, [-2 * 0.9**3])
    assert np.allclose(target, advantage)


def test_intermediate_reward_and_next_value_across_variable_durations():
    advantage, target = duration_aware_gae(
        [1, 2, 3, 4], [0, 2], [0.5, 1], terminated=True, gamma=0.5, gae_lambda=0
    )
    assert np.allclose(advantage, [1 + 0.5 * 2 + 0.5**2 * 1 - 0.5, 3 + 0.5 * 4 - 1])
    assert np.allclose(target, advantage + [0.5, 1])


def test_unit_lambda_matches_discounted_monte_carlo_at_each_decision():
    reward = np.array([0.0, 1.0, -0.5, 0.0, -2.0])
    gamma = 0.9
    _, targets = duration_aware_gae(
        reward, [0, 2, 4], [0.2, 0.1, 0.3], terminated=True, gamma=gamma, gae_lambda=1
    )
    expected = [sum(gamma ** (i - t) * reward[i] for i in range(t, len(reward))) for t in (0, 2, 4)]
    assert np.allclose(targets, expected)


@pytest.mark.parametrize("decisions", [[1, 1], [2, 1], [-1], [4], [1.0], []])
def test_bad_decisions_fail_closed(decisions):
    with pytest.raises(ValueError):
        duration_aware_gae([0, 0, 0, 0], decisions, np.zeros(len(decisions)), terminated=True)


@pytest.mark.parametrize("gamma,lam", [(1.1, 0.5), (0.9, -0.1), (float("nan"), 1)])
def test_bad_discounts_fail_closed(gamma, lam):
    with pytest.raises(ValueError):
        duration_aware_gae([0], [0], [0], terminated=True, gamma=gamma, gae_lambda=lam)


def test_truncated_episode_is_rejected():
    with pytest.raises(ValueError):
        duration_aware_gae([0, -2], [0], [0], terminated=False)
