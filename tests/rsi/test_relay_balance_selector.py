import numpy as np
import pytest

from rosclaw_soccer.rsi.relay_balance_selector import (
    RelayBalanceResearchSelector,
    decision_rewards,
    select_with_parent_fallback,
)


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


def test_selector_is_stateless_and_never_authorizes_hardware():
    arrays = dict(
        center=np.zeros(17),
        scale=np.ones(17),
        support=np.zeros((8, 17)),
        weight1=np.zeros((32, 17)),
        bias1=np.zeros(32),
        weight2=np.zeros((4, 32)),
        bias2=np.array([0, 1, 0, 0]),
    )
    selector = RelayBalanceResearchSelector(arrays)
    q, v = np.zeros(43), np.zeros(41)
    q[2], q[3], q[38] = 0.78, 1, 0.11
    result = selector.decide(q, v, np.array([0, 0, 0.11]))
    assert result["teacher_index"] == 0
    assert result["hardware_authorized"] is False
    assert result["promotion_authorized"] is False
    arrays["bias2"][:] = 0
    assert selector.decide(q, v, np.array([0, 0, 0.11]))["teacher_index"] == 0
