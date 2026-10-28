import numpy as np
import pytest

from rosclaw_soccer.rsi.receive_reference_matching import match_error, yaw_local


def test_yaw_invariant_reference_comparison():
    quat = np.array([2**-0.5, 0.0, 0.0, 2**-0.5])
    np.testing.assert_allclose(yaw_local(quat, np.array([0.0, 1.0, 2.0])), [1, 0, 2], atol=1e-12)
    item = dict(
        ball_relative=np.array([1.0, 0, 0]),
        velocity_relative=np.zeros(3),
        joints=np.zeros(29),
        root_height=0.75,
    )
    result = match_error(item, item)
    assert result["geometry_state_compatible"] is True
    assert result["dynamic_feasibility_verified"] is False
    assert result["score"] == 0


def test_wrong_shape_and_nonfinite_rejected():
    item = dict(
        ball_relative=np.zeros(3),
        velocity_relative=np.zeros(3),
        joints=np.zeros(29),
        root_height=0.75,
    )
    bad = dict(item, joints=np.full(29, np.nan))
    with pytest.raises(ValueError):
        match_error(item, bad)
