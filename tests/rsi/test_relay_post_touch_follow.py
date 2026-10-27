import numpy as np
import pytest

from rosclaw_soccer.rsi.relay_post_touch_follow import post_touch_follow


def run(*, contact=True, gain=0.5, ball=(0.2, 0), velocity=(1, 0)):
    return post_touch_follow(
        np.array([0, 0, 0.2], np.float32),
        root_xy=np.zeros(2),
        yaw=0,
        ball_xy=np.array(ball, dtype=float),
        ball_velocity_xy=np.array(velocity, dtype=float),
        pass_direction=np.array([1.0, 0]),
        contact_seen=contact,
        gain=gain,
    )


def test_no_interference_before_contact_or_outside_support():
    expected = np.array([0, 0, 0.2], np.float32)
    assert np.array_equal(run(contact=False), expected)
    assert np.array_equal(run(gain=0.0), expected)
    assert np.array_equal(run(ball=(0.7, 0)), expected)


def test_follow_is_bounded_and_does_not_override_yaw():
    action = run(velocity=(30, -30), gain=0.75)
    assert -0.5 <= action[0] <= 1
    assert -0.5 <= action[1] <= 0.5
    assert action[2] == np.float32(0.2)
    assert action[0] > 0 and action[1] < 0
    with pytest.raises(ValueError):
        run(gain=0.4)
