import numpy as np
import pytest

from rosclaw_soccer.rsi.candidate_motion23_mapping import candidate_map_motion23


def test_candidate_order_and_wrist_insertion_are_explicit():
    dof = np.tile(np.arange(23, dtype=np.float64), (2, 1))
    quat = np.array([[0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.float64)
    result = candidate_map_motion23(dof, quat)
    np.testing.assert_array_equal(result["dof29"][:, :19], dof[:, :19])
    np.testing.assert_array_equal(result["dof29"][:, 22:26], dof[:, 19:23])
    np.testing.assert_array_equal(result["dof29"][:, [19, 20, 21, 26, 27, 28]], 0)
    np.testing.assert_array_equal(result["root_rot_wxyz"][0], [1, 0, 0, 0])
    np.testing.assert_array_equal(result["root_rot_wxyz"][1], [0, 0, 0, 1])
    assert result["physical_execution_qualified"] is False
    assert result["input_joint_order_is_assumed"] is True


def test_invalid_rotation_and_nonfinite_reference_are_rejected():
    dof = np.zeros((2, 23), np.float64)
    quat = np.zeros((2, 4), np.float64)
    with pytest.raises(ValueError):
        candidate_map_motion23(dof, quat)
    quat[:, 3] = 1
    dof[0, 0] = np.nan
    with pytest.raises(ValueError):
        candidate_map_motion23(dof, quat)
