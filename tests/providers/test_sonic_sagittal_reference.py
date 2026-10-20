import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.providers.g1.sonic_sagittal_reference import mirror_sagittal_reference


def poses():
    x = np.random.default_rng(47).normal(0, 0.2, (17, 36))
    x[:, 2] = 0.8
    x[:, 3:7] = Rotation.from_rotvec(x[:, :3]).as_quat()[:, [3, 0, 1, 2]]
    return x


def test_involution_private_and_named_axes():
    x = poses()
    saved = x.copy()
    y = mirror_sagittal_reference(x)
    np.testing.assert_array_equal(x, saved)
    np.testing.assert_array_equal(mirror_sagittal_reference(y), x)
    assert not np.shares_memory(x, y)
    for left, right, sign in (
        ("left_knee_joint", "right_knee_joint", 1),
        ("left_hip_yaw_joint", "right_hip_yaw_joint", -1),
        ("left_shoulder_roll_joint", "right_shoulder_roll_joint", -1),
    ):
        np.testing.assert_array_equal(
            y[:, 7 + G1_DDS_JOINT_NAMES.index(left)],
            sign * x[:, 7 + G1_DDS_JOINT_NAMES.index(right)],
        )


def test_quaternion_reflects_polar_vectors_correctly():
    x = poses()
    y = mirror_sagittal_reference(x)
    r = Rotation.from_quat(x[:, [4, 5, 6, 3]]).as_matrix()
    reflected = Rotation.from_quat(y[:, [4, 5, 6, 3]]).as_matrix()
    s = np.diag([1, -1, 1])
    np.testing.assert_allclose(reflected, s @ r @ s, atol=1e-15)


@pytest.mark.parametrize("bad", [np.zeros((3, 35)), np.full((3, 36), np.nan)])
def test_invalid_not_clipped(bad):
    with pytest.raises(ValueError):
        mirror_sagittal_reference(bad)
