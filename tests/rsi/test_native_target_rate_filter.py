from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.velocity_policy import target29
from rosclaw_soccer.rsi.native_target_rate_filter import (
    ANKLE_JOINT_INDICES,
    limit_ankle_target_rate,
    limit_target_rate,
)


def test_exact_noop_and_bounded_multi_axis_changes():
    params = SimpleNamespace(action_scale=(0.3,) * 29, action_offset=(0.4,) * 29)
    previous = np.zeros(29, np.float32)
    near = np.full(29, 0.1, np.float32)
    filtered, difference = limit_target_rate(action=near, previous_action=previous, params=params)
    np.testing.assert_array_equal(filtered, near)
    np.testing.assert_array_equal(difference, np.zeros(29))
    action = np.linspace(-2, 2, 29, dtype=np.float32)
    filtered, difference = limit_target_rate(action=action, previous_action=previous, params=params)
    previous_target = target29(previous, params)
    actual = target29(filtered, params)
    assert np.max(np.abs(actual - previous_target)) <= 0.080001
    np.testing.assert_array_equal(difference, actual - target29(action, params))


def test_repeated_quantized_rate_limits_and_invalid_inputs():
    rng = np.random.default_rng(811)
    params = SimpleNamespace(
        action_scale=tuple(rng.uniform(0.1, 1, 29)), action_offset=tuple(rng.uniform(-2, 2, 29))
    )
    previous = rng.uniform(-1, 1, 29).astype(np.float32)
    for _ in range(100):
        action = rng.uniform(-2, 2, 29).astype(np.float32)
        actual, _ = limit_target_rate(action=action, previous_action=previous, params=params)
        assert np.max(np.abs(target29(actual, params) - target29(previous, params))) <= 0.080001
        previous = actual
    with pytest.raises(ValueError):
        limit_target_rate(
            action=previous, previous_action=previous, params=params, limit_rad=float("nan")
        )


def test_ankle_only_preserves_every_other_policy_command():
    params = SimpleNamespace(action_scale=(0.3,) * 29, action_offset=(0.4,) * 29)
    previous = np.zeros(29, np.float32)
    action = np.linspace(-2, 2, 29, dtype=np.float32)
    filtered, delta = limit_ankle_target_rate(
        action=action, previous_action=previous, params=params
    )
    ankles = list(ANKLE_JOINT_INDICES)
    other = [i for i in range(29) if i not in ankles]
    np.testing.assert_array_equal(filtered[other], action[other])
    np.testing.assert_array_equal(delta[other], np.zeros(len(other)))
    assert (
        np.max(np.abs((target29(filtered, params) - target29(previous, params))[ankles]))
        <= 0.080001
    )
    assert np.max(np.abs((target29(filtered, params) - target29(previous, params))[other])) > 0.08
