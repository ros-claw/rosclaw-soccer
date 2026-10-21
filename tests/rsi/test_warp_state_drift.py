import numpy as np
import pytest

from scripts.rsi_probe_warp_option_drift import state_drift


def test_metre_and_rotation_metrics_are_separate_and_quaternion_sign_invariant():
    a = np.zeros((1, 1, 43))
    a[..., 3] = a[..., 10] = 1.0
    b = a.copy()
    b[..., 3] = -1.0
    b[..., 0:2] = [0.3, 0.4]
    b[..., 14] = 0.2
    result = state_drift(a, b, robot_qpos=7, ball_qpos=0, joints=np.arange(14, 43))
    assert result["ball_position_max_distance_m"] == pytest.approx(0.5)
    assert result["ball_rotation_max_geodesic_rad"] == 0
    assert result["root_position_max_distance_m"] == 0
    assert result["hinge_position_max_absolute_rad"] == pytest.approx(0.2)
    assert np.array_equal(a[..., 3], [[1.0]])
    assert np.array_equal(b[..., 3], [[-1.0]])


def test_refuses_zero_quaternion_instead_of_repairing_state():
    with pytest.raises(ValueError, match="nonzero"):
        state_drift(
            np.zeros((1, 1, 43)),
            np.zeros((1, 1, 43)),
            robot_qpos=7,
            ball_qpos=0,
            joints=np.arange(14, 43),
        )
