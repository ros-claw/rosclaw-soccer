import numpy as np
import pytest

from rosclaw_soccer.rsi.native_receive_credit import NativeReceiveCredit, planar_ball_root_distance


def step(credit, t, f, *, active=None, distance=None, speed=None, healthy=None):
    n = credit.worlds
    return credit.step(
        step=t,
        forces=f,
        active=np.ones(n, bool) if active is None else active,
        ball_root_planar_distance_m=np.full(n, 0.4) if distance is None else distance,
        relative_speed_mps=np.full(n, 0.2) if speed is None else speed,
        healthy=np.ones(n, bool) if healthy is None else healthy,
    )


def test_true_foot_then_ten_controls_once_not_proximity_only():
    c = NativeReceiveCredit(2)
    f = np.zeros((2, 10, 3))
    for t in range(13):
        if t == 2:
            f[0, 0, 0] = 11
        r = step(c, t, f)
        assert r["sustained_capture"][1] == 0
        assert r["sustained_capture"][0] == (1 if t == 12 else 0)
        f[:] = 0


def test_standing_pelvis_height_is_not_added_to_planar_capture_distance():
    ball = np.array([[0.3, 0.4, 0.11]])
    root = np.array([[0, 0, 0.78]])
    assert np.linalg.norm(ball - root) > 0.65
    assert planar_ball_root_distance(ball, root)[0] == pytest.approx(0.5)
    ball[0, 2] = np.nan
    with pytest.raises(ValueError, match="finite measured"):
        planar_ball_root_distance(ball, root)


def test_nonfoot_before_or_simultaneous_invalidates_and_inactive_freezes():
    c = NativeReceiveCredit(3)
    f = np.zeros((3, 10, 3))
    f[0, 1, 2], f[0, 2, 0] = 2, 11
    f[1, 1, 2], f[1, 1, 0] = 2, 11
    f[2, 0, 0] = 11
    r = step(c, 0, f, active=np.array([True, True, False]))
    np.testing.assert_array_equal(r["first_clean_foot"], [0, 0, 0])
    np.testing.assert_array_equal(r["contaminated"], [True, True, False])
    with pytest.raises(ValueError):
        step(c, 0, f)


def test_fast_ball_breaks_consecutive_control_and_credit_is_not_repeated():
    c = NativeReceiveCredit(1)
    f = np.zeros((1, 10, 3))
    f[0, 0, 0] = 11
    step(c, 0, f)
    f[:] = 0
    assert step(c, 1, f, speed=np.array([1.0]))["stable_controls"][0] == 0
    for t in range(2, 12):
        r = step(c, t, f)
    assert r["sustained_capture"][0] == 1
    assert step(c, 12, f)["sustained_capture"][0] == 0
