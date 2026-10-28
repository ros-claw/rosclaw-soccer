from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi.restored_motion_reference import restored_motion_reference


def test_motion_reference_becomes_standing_at_real_clip_boundary():
    clip = SimpleNamespace(
        length=3,
        joint_pos=np.arange(87, dtype=np.float32).reshape(3, 29),
        joint_vel=np.ones((3, 29), np.float32),
        body_ang_vel_w=np.ones((3, 1, 3), np.float32),
    )
    pos, vel, ang = restored_motion_reference(clip, 0, 2)
    np.testing.assert_array_equal(pos, clip.joint_pos[1])
    assert np.all(vel == 1) and np.all(ang == 1)
    pos, vel, ang = restored_motion_reference(clip, 0, 3)
    np.testing.assert_array_equal(pos, clip.joint_pos[2])
    assert np.all(vel == 0) and np.all(ang == 0)
    pos, _, _ = restored_motion_reference(clip, 0, 4)
    np.testing.assert_array_equal(pos, clip.joint_pos[2])
    pos, _, _ = restored_motion_reference(clip, 0, 599)
    np.testing.assert_array_equal(pos, clip.joint_pos[2])
    with pytest.raises(ValueError):
        restored_motion_reference(clip, 0, 0)
    with pytest.raises(ValueError):
        restored_motion_reference(clip, 0, 600)
