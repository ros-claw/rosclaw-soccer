from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi.motion_reference_tempo import sample_reference


def test_interpolation_and_physical_velocity_scaling():
    clip = SimpleNamespace(
        length=2,
        joint_pos=np.array([[0, 0], [2, 4]], np.float32),
        joint_vel=np.array([[2, 4], [4, 8]], np.float32),
        body_ang_vel_w=np.array([[[2, 4, 6]], [[4, 8, 12]]], np.float32),
    )
    pos, vel, ang = sample_reference(clip, 0, 0.5, 0.75)
    np.testing.assert_array_equal(pos, [1, 2])
    np.testing.assert_array_equal(vel, [2.25, 4.5])
    np.testing.assert_array_equal(ang, [2.25, 4.5, 6.75])
    np.testing.assert_array_equal(sample_reference(clip, 0, 1.5, 0.5)[0], [2, 4])


@pytest.mark.parametrize("position,rate", [(-1, 0.8), (2, 0.8), (0, 0.4), (0, float("nan"))])
def test_invalid_reference_tempo_fails_closed(position, rate):
    with pytest.raises(ValueError):
        sample_reference(SimpleNamespace(length=2), 0, position, rate)
