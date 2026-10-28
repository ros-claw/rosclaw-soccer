from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.velocity_policy import target29
from rosclaw_soccer.rsi.native_teacher_action import require_teacher_slew, teacher_target_to_action


def test_teacher_conversion_keeps_other_axes_bit_exact_and_memory_actual():
    params = SimpleNamespace(
        action_scale=tuple(np.linspace(0.2, 0.8, 29)),
        action_offset=tuple(np.linspace(-0.4, 0.4, 29)),
    )
    action = np.linspace(-0.9, 0.8, 29, dtype=np.float32)
    nominal = target29(action, params)
    delta = np.zeros(29)
    delta[3], delta[5] = 0.16, -0.07
    changed, actual = teacher_target_to_action(action, nominal + delta, delta, params)
    mask = delta != 0
    np.testing.assert_array_equal(changed[~mask], action[~mask])
    np.testing.assert_array_equal(actual, target29(changed, params) - nominal)
    assert np.max(np.abs(actual)) <= 0.16 + 1e-12
    np.testing.assert_allclose(actual, delta, atol=1e-6, rtol=0)


def test_teacher_conversion_rejects_unbound_and_overlimit_target():
    params = SimpleNamespace(action_scale=(0.2,) * 29, action_offset=(0.0,) * 29)
    action = np.zeros(29, dtype=np.float32)
    with pytest.raises(ValueError, match="bound"):
        teacher_target_to_action(action, np.ones(29), np.ones(29), params)
    with pytest.raises(ValueError, match="bind"):
        teacher_target_to_action(action, np.ones(29), np.zeros(29), params)


def test_quantized_target_bound_holds_across_parent_magnitudes():
    rng = np.random.default_rng(719)
    for _ in range(300):
        params = SimpleNamespace(
            action_scale=tuple(rng.uniform(0.1, 1.0, 29)),
            action_offset=tuple(rng.uniform(-2.0, 2.0, 29)),
        )
        action = rng.uniform(-2, 2, 29).astype(np.float32)
        nominal = target29(action, params)
        delta = np.zeros(29)
        delta[2] = rng.choice([-0.16, 0.16])
        changed, actual = teacher_target_to_action(action, nominal + delta, delta, params)
        assert abs(actual[2]) <= 0.16 + 1e-12
        np.testing.assert_array_equal(changed[np.arange(29) != 2], action[np.arange(29) != 2])


def test_slew_conflict_is_rejected_instead_of_snapping_to_parent():
    previous = np.zeros(29)
    previous[9] = -0.16
    with pytest.raises(ValueError, match="slew infeasible"):
        require_teacher_slew(previous, np.zeros(29))
    next_delta = previous.copy()
    next_delta[9] += 0.025
    require_teacher_slew(previous, next_delta)
