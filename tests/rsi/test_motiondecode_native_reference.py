import numpy as np
import pytest

from rosclaw_soccer.rsi.motiondecode_native_reference import _slerp, native_motion_reference


def test_slerp_uses_short_path_and_antipodal_quaternions():
    a = np.array([1.0, 0, 0, 0])
    b = np.array([2**-0.5, 0, 0, 2**-0.5])
    expected = np.array([np.cos(np.pi / 8), 0, 0, np.sin(np.pi / 8)])
    np.testing.assert_allclose(_slerp(a, b, 0.5), expected)
    np.testing.assert_allclose(_slerp(a, -b, 0.5), expected)
    np.testing.assert_allclose(_slerp(a, -a, 0.5), a)


def test_reject_undeclared_or_bad_source_before_native_model_access():
    values = np.zeros((4, 36), dtype=np.float64)
    values[:, 3] = 1
    with pytest.raises(ValueError, match="explicit"):
        native_motion_reference(model=None, values=values, source_fps=120)
    values[2, 4] = 0.1
    with pytest.raises(ValueError, match="unit source quaternion"):
        native_motion_reference(model=None, values=values, source_fps=120.0)
