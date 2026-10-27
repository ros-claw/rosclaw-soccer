import numpy as np
import pytest

from rosclaw_soccer.rsi.relay_balance_selector import decision_rewards, select_with_parent_fallback


def test_rewards_keep_failure_and_precision_distinct():
    success = np.array([True, False, False, True])
    failure = np.array([False, True, False, False])
    kick = np.array([True, True, False, True])
    assert np.allclose(decision_rewards(success, failure, kick), [2.25, -1.75, 0, 2.25])
    with pytest.raises(ValueError):
        decision_rewards(success, success, kick)


def test_unseen_and_small_predicted_gains_use_unchanged_parent():
    assert select_with_parent_fallback(np.array([0, 1, 0, 0]), 2.1)[0] == 0
    assert select_with_parent_fallback(np.array([0, 0.24, 0, 0]), 1.0)[0] == 0
    assert select_with_parent_fallback(np.array([0, 1, 0, 0]), 1.0) == (
        1,
        "RESEARCH_TEACHER_CHOICE",
    )
