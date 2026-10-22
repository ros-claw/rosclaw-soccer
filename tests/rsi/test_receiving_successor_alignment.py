import numpy as np
import pytest

from rosclaw_soccer.rsi.receiving_successor_alignment import (
    assigned_receiver_pass_goal,
    successor_alignment,
)


def state():
    q, v = np.zeros(43), np.zeros(41)
    q[:7] = [0, 0, 0.78, 1, 0, 0, 0]
    q[36:43] = [0.12, 0.03, 0.11, 1, 0, 0, 0]
    return q, v


def test_current_reachable_ball_is_ready_without_any_future_or_case_identifier():
    q, v = state()
    before = q.copy(), v.copy()
    result = successor_alignment(q, v, np.array([4.0, 0.0, 0.11]))
    assert result.geometry_ready and np.linalg.norm(result.command_body) < 1e-10
    np.testing.assert_array_equal(q, before[0])
    np.testing.assert_array_equal(v, before[1])


def test_context20_measured_wrong_side_is_not_ready_and_requests_bounded_reposition():
    q, v = state()
    q[36:38] = [0.364, -0.178]
    result = successor_alignment(q, v, np.array([4.0, 0.0, 0.11]))
    assert not result.geometry_ready
    assert result.command_body[0] > 0 and result.command_body[1] < 0
    assert np.linalg.norm(result.command_body[:2]) <= 0.35 + 1e-12


def test_coordinate_rotation_and_ball_velocity_following_are_body_consistent():
    q, v = state()
    q[3:7] = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
    q[36:38] = [-0.03, 0.12]
    v[35:37] = [0.0, 0.2]
    v[:2] = v[35:37]
    result = successor_alignment(q, v, np.array([0.0, 4.0, 0.11]))
    assert result.geometry_ready
    np.testing.assert_allclose(result.command_body, [0.2, 0, 0], atol=1e-14)


def test_ball_behind_or_far_and_wrong_heading_do_not_trigger_kick():
    q, v = state()
    for ball in ([-0.1, 0], [1.0, 0], [0.12, 0.3]):
        q[36:38] = ball
        assert not successor_alignment(q, v, np.array([4.0, 0.0, 0.11])).geometry_ready
    q[36:38] = [0.12, 0.03]
    assert not successor_alignment(q, v, np.array([-4.0, 0.0, 0.11])).geometry_ready


def test_assigned_pass_target_matches_receivers_declared_forefoot_point_not_old6m():
    target = assigned_receiver_pass_goal(np.array([4.0, 0.12]), np.array([1.0, 0.0]))
    np.testing.assert_allclose(target, [3.75, 0, 0.11])
    with pytest.raises(ValueError):
        assigned_receiver_pass_goal(np.array([4.0, 0.12]), np.array([2.0, 0.0]))
