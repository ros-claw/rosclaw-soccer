import numpy as np
import pytest

from rosclaw_soccer.rsi.receiving_early_arrival import approaching_contact_window


def state(distance, ball_speed, receiver_speed=0.0):
    q, v = np.zeros(43), np.zeros(41)
    q[36] = distance
    v[35], v[0] = ball_speed, receiver_speed
    return q, v


def test_actual_relative_approach_not_distance_or_world_speed_alone():
    assert approaching_contact_window(*state(0.8, -2.0))
    assert not approaching_contact_window(*state(0.8, 2.0))
    assert not approaching_contact_window(*state(0.8, -2.0, -1.9))
    assert not approaching_contact_window(*state(0.8, -1.0))
    assert not approaching_contact_window(*state(1.3, -4.0))
    assert not approaching_contact_window(*state(0.5, -4.0))


def test_invariant_under_joint_translation_and_velocity_boost():
    q, v = state(0.8, -2.0)
    translated, boosted = q.copy(), v.copy()
    translated[[0, 36]] += 12
    boosted[[0, 35]] += 3
    assert approaching_contact_window(q, v) == approaching_contact_window(translated, boosted)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_state_never_triggers(bad):
    q, v = state(0.8, -2.0)
    q[36] = bad
    with pytest.raises(ValueError):
        approaching_contact_window(q, v)
