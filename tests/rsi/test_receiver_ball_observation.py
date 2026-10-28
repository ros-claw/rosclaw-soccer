import numpy as np
import pytest

from rosclaw_soccer.rsi.receiver_ball_observation import receiver_ball_observation


def test_rotating_frame_derivative_and_heading():
    q = np.zeros(43, dtype=np.float64)
    v = np.zeros(41, dtype=np.float64)
    q[3:7] = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
    q[36:39] = [0, 2, 1]
    v[35:38] = [0, 3, 0]
    v[3:6] = [0, 0, 1]
    np.testing.assert_allclose(receiver_ball_observation(q, v), [2, 0, 1, 3, -2, 0], atol=1e-12)


def test_common_translation_and_velocity_are_invariant():
    q = np.zeros(43, dtype=np.float64)
    v = np.zeros(41, dtype=np.float64)
    q[3] = 1
    q[36:39] = [1, 2, 3]
    v[35:38] = [4, 5, 6]
    before = receiver_ball_observation(q, v)
    q[:3] += [7, 8, 9]
    q[36:39] += [7, 8, 9]
    v[:3] += [3, -2, 1]
    v[35:38] += [3, -2, 1]
    np.testing.assert_allclose(receiver_ball_observation(q, v), before)


@pytest.mark.parametrize("kind", ["shape", "quaternion", "nan", "integer"])
def test_invalid_inputs_fail_closed(kind):
    q = np.zeros(43, dtype=np.float64)
    v = np.zeros(41, dtype=np.float64)
    q[3] = 1
    if kind == "shape":
        q = q[:-1]
    elif kind == "quaternion":
        q[3] = 0
    elif kind == "nan":
        v[37] = np.nan
    else:
        q = q.astype(np.int64)
    with pytest.raises(ValueError):
        receiver_ball_observation(q, v)
