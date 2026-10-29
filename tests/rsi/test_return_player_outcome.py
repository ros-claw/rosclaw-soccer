import numpy as np
import pytest

from rosclaw_soccer.rsi.return_player_outcome import (
    discounted_sample_returns,
    return_player_outcome,
)


def test_only_contacts_after_sampling_and_full_late_failure_count():
    forces = np.zeros((8, 10, 2, 3))
    failed = np.zeros((8, 2), bool)
    forces[1, 0, 0, 0] = 20
    outcome = return_player_outcome(forces, failed, sample_start=3)
    assert not outcome["a_foot_contact"] and outcome["terminal_proxy_reward"] == 0
    forces[4, 0, 0, 0] = 20
    assert return_player_outcome(forces, failed, sample_start=3)["terminal_proxy_reward"] == 1
    failed[7, 1] = True
    outcome = return_player_outcome(forces, failed, sample_start=3)
    assert outcome["b_body_failed"] and outcome["terminal_proxy_reward"] == -1
    np.testing.assert_allclose(
        discounted_sample_returns(outcome, sample_end=5, gamma=0.5),
        [-0.0625, -0.125, -0.25],
    )
    assert not outcome["possession_proven"]


def test_nonfoot_and_invalid_boundary():
    forces = np.zeros((2, 10, 2, 3))
    forces[1, 0, 0, 2] = 2
    failures = np.zeros((2, 2), bool)
    assert return_player_outcome(forces, failures, sample_start=1)["terminal_proxy_reward"] == -1
    with pytest.raises(ValueError):
        return_player_outcome(forces, failures, sample_start=2)
