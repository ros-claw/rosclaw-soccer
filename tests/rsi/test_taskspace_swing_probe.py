"""Task-space probe moves one airborne foot within joint safety bounds."""

import numpy as np
import pytest

from rosclaw_soccer.rsi.taskspace_swing_probe import (
    choose_swing_side,
    release_joint_delta,
    swing_joint_delta,
)


def test_taskspace_selects_airborne_leg_and_bounds_ik() -> None:
    feet = np.array([[0.0, 0.1, 0.14], [0.0, -0.1, 0.04]])
    ball = np.array([0.6, 0.1, 0.11])
    assert choose_swing_side(feet, ball, -1) == 0
    assert choose_swing_side(feet, ball, 1) == 1
    feet[0, 2] = 0.05
    assert choose_swing_side(feet, ball, -1) == -1
    jac = np.zeros((3, 6))
    jac[0, 0] = 0.5
    jac[1, 1] = 0.5
    limits = np.tile([-1.0, 1.0], (6, 1))
    delta = swing_joint_delta(
        feet[0],
        ball,
        jac,
        np.zeros(6),
        limits,
        forward_cap_m=0.16,
    )
    assert 0 < delta[0] <= 0.35
    np.testing.assert_array_equal(delta[2:], 0)
    limits[0] = [-0.1, 0.1]
    assert swing_joint_delta(feet[0], ball, jac, np.zeros(6), limits, forward_cap_m=0.16)[0] == 0
    with pytest.raises(ValueError):
        swing_joint_delta(feet[0], ball, jac, np.zeros(6), limits, forward_cap_m=0.2)


def test_taskspace_release_is_causal_and_monotone() -> None:
    contact_delta = np.full(6, 0.2)
    assert release_joint_delta(contact_delta, 1)[0] == pytest.approx(0.19)
    assert release_joint_delta(contact_delta, 20)[0] == 0
    with pytest.raises(ValueError):
        release_joint_delta(contact_delta, 0)
