import numpy as np
import pytest

from rosclaw_soccer.rsi.contact_normal_response import (
    contact_normal_proposals,
    measured_side_offset,
)


def test_geometry_moves_foot_toward_missing_goal_side_contact():
    q = np.zeros((10, 43))
    q[:, 3] = 1
    events = [
        dict(
            role=0,
            solver_time_s=0.004,
            ball_solver_center_world=[0.0, 0.0, 0.11],
            contact_position_world=[-0.1, -0.02, 0.09],
            force_norm_N=20.0,
        )
    ]
    offset = measured_side_offset(events, np.array([6.0, 0.0, 0.11]), q, initial_time=0.0, role=0)
    assert np.allclose(offset, [0.0, 0.02])
    p, r = contact_normal_proposals(
        np.zeros((2, 6)), center=100, contact_control=100, local_xy_m=offset
    )
    assert (p[8, :, 1] > 0).all()
    assert (p[-1, :, 1] < 0).all()
    assert not p[:, :, 2:].any()
    assert r["physical_success_predicted"] is False


def test_body_orientation_not_world_axis_is_used():
    q = np.zeros((10, 43))
    q[:, 3] = q[:, 6] = np.sqrt(0.5)
    events = [
        dict(
            role=1,
            solver_time_s=0.004,
            ball_solver_center_world=[0.0, 0.0, 0.11],
            contact_position_world=[-0.1, -0.02, 0.09],
            force_norm_N=20.0,
        )
    ]
    offset = measured_side_offset(events, np.array([6.0, 0.0, 0.11]), q, initial_time=0.0, role=1)
    assert np.allclose(offset, [0.02, 0.0], atol=1e-15)


def test_contact_clock_cannot_interpolate_body_pose():
    q = np.zeros((10, 43))
    q[:, 3] = 1
    events = [dict(role=0, solver_time_s=0.0045)]
    with pytest.raises(ValueError):
        measured_side_offset(events, np.array([6.0, 0.0, 0.11]), q, initial_time=0.0, role=0)


def test_empty_contact_is_not_a_zero_offset_measurement():
    with pytest.raises(ValueError):
        measured_side_offset(
            [], np.array([6.0, 0.0, 0.11]), np.zeros((10, 43)), initial_time=0.0, role=0
        )
