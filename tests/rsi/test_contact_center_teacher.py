import numpy as np
import pytest

from rosclaw_soccer.rsi.contact_center_teacher import current_body_offset, measured_centering


def event():
    return dict(
        phase="kick",
        role=0,
        robot_body="left_ankle_roll_link",
        force_norm_N=20.0,
        ball_solver_center_world=[0.0, 0.0, 0.11],
        contact_position_world=[-0.11, 0.0, 0.08],
        reference_frame=104,
    )


def test_low_contact_requires_upward_not_downward_correction():
    v, center = measured_centering([event()], np.array([5.0, 0.0, 1.0]))
    np.testing.assert_allclose(v, [0, 0, 0.03], atol=1e-12)
    assert center == 104


def test_actual_body_rotation_roundtrip_and_finite_support():
    r = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    v = np.array([0.01, 0.02, 0.03])
    local = current_body_offset(v, r, 104, 104)
    np.testing.assert_allclose(r @ local[:3], v)
    assert not local[3:].any()
    for t in (0, 80, 128, 300):
        assert not current_body_offset(v, r, t, 104).any()


def test_reject_unknown_geometry_and_degenerate_frame():
    with pytest.raises(ValueError):
        measured_centering([], np.array([5.0, 0.0, 0.0]))
    e = event()
    e["contact_position_world"] = [float("nan"), 0.0, 0.08]
    with pytest.raises(ValueError):
        measured_centering([e], np.array([5.0, 0.0, 0.0]))
    with pytest.raises(ValueError):
        current_body_offset(np.zeros(3), -np.eye(3), 100, 104)
