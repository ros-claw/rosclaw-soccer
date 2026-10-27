import numpy as np
import pytest

from rosclaw_soccer.rsi.sim_contact_bandit import (
    action_delta,
    contact_features,
    offline_contact_reward,
    select_contact_action,
)


def test_fixed_action_table_and_state_validation():
    assert action_delta(0).sum() == 0
    assert action_delta(1)[0] == -0.08
    assert action_delta(8)[3] == 0.08
    with pytest.raises(ValueError):
        action_delta(True)
    with pytest.raises(ValueError):
        contact_features(float("nan"), 0, 0)


def test_reward_preserves_unsafe_and_absent_observation():
    assert offline_contact_reward({"safe": False, "task_pass": False}) == -2
    assert offline_contact_reward({"safe": True, "task_pass": False, "forward_60_m": None}) == -0.2
    assert offline_contact_reward(
        {"safe": True, "task_pass": True, "forward_60_m": 1.1}
    ) == pytest.approx(1.1)
    with pytest.raises(ValueError):
        offline_contact_reward({"safe": True, "task_pass": False, "forward_60_m": float("nan")})


def test_actor_selects_only_fixed_discrete_options():
    w = np.zeros((9, 2))
    b = np.arange(9, dtype=float)
    assert select_contact_action(np.zeros(2), w, b) == 8
    with pytest.raises(ValueError):
        select_contact_action(np.zeros(2), w[:8], b)
