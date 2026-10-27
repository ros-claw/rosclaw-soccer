import numpy as np
import pytest

from rosclaw_soccer.rsi.relay_balance_context import relay_balance_context


def test_context_invariant_to_world_translation_and_yaw():
    q, v = np.zeros(43), np.zeros(41)
    q[2], q[3], q[36:39] = 0.78, 1, [1.0, 0.2, 0.11]
    goal = np.array([3.0, -1.0, 0.11])
    v[:3], v[35:38], v[3:6] = [0.2, -0.1, 0], [0.5, 0.3, 0], [0.1, 0.2, 0.3]
    expected = relay_balance_context(q, v, goal)
    angle = np.pi / 2
    rotate = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    shift = np.array([4.0, -2.0, 0])
    q2, v2 = q.copy(), v.copy()
    q2[:3], q2[36:39] = rotate @ q[:3] + shift, rotate @ q[36:39] + shift
    q2[3:7] = [np.cos(angle / 2), 0, 0, np.sin(angle / 2)]
    v2[:3], v2[35:38] = rotate @ v[:3], rotate @ v[35:38]
    assert np.allclose(relay_balance_context(q2, v2, rotate @ goal + shift), expected, atol=1e-12)


def test_rejects_nonfinite_or_invalid_pose():
    q, v = np.zeros(43), np.zeros(41)
    with pytest.raises(ValueError):
        relay_balance_context(q, v, np.zeros(3))
