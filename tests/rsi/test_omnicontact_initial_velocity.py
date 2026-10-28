import numpy as np
import pytest

from rosclaw_soccer.rsi.omnicontact_initial_velocity import reference_initial_velocities


def test_world_angular_velocity_converts_to_local_frame():
    quat = np.array([2**-0.5, 0, 0, 2**-0.5])
    motion = dict(
        body_quat_w=quat.reshape(1, 1, 4),
        object_quat_w=quat.reshape(1, 4),
        body_lin_vel_w=np.zeros((1, 1, 3)),
        body_ang_vel_w=np.array([[[1.0, 0, 0]]]),
        joint_vel=np.arange(29, dtype=float).reshape(1, 29),
        object_lin_vel_w=np.zeros((1, 3)),
        object_ang_vel_w=np.array([[1.0, 0, 0]]),
    )
    result = reference_initial_velocities(motion, np.arange(29))
    np.testing.assert_allclose(result["root_angular_local"], [0, -1, 0], atol=1e-12)
    np.testing.assert_allclose(result["ball_angular_local"], [0, -1, 0], atol=1e-12)
    motion["joint_vel"][0, 0] = 50
    with pytest.raises(ValueError, match="bounded"):
        reference_initial_velocities(motion, np.arange(29))
