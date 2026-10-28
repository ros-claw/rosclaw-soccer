import numpy as np
import pytest

from rosclaw_soccer.rsi.receiving_progress_reward import (
    decompose_proximity_change,
    receiver_motion_progress_rewards,
    receiving_progress_rewards,
)


def test_potential_and_one_shot_collision_and_fall_events():
    receiver = np.zeros((4, 2))
    ball = np.array([[3.0, 0], [2.0, 0], [1.0, 0], [1.0, 0]])
    forces = np.zeros((4, 10, 3))
    forces[1:, 0, 0] = 11
    forces[2:, 0, 2] = 2
    failed = np.array([False, False, True, True])
    reward = receiving_progress_rewards(receiver, ball, forces, failed, discount=1.0)
    np.testing.assert_allclose(reward, [0, 2, -5, 0])


def test_thresholds_are_strict_and_single_control_has_no_distance_reward():
    xy = np.zeros((1, 2))
    forces = np.zeros((1, 10, 3))
    forces[0, 0] = [10, 0, 1]
    result = receiving_progress_rewards(xy, xy, forces, np.zeros(1, dtype=bool))
    np.testing.assert_array_equal(result, [0])


def test_invalid_physics_fails_closed():
    xy = np.zeros((2, 2))
    bad_forces = np.zeros((2, 10, 3))
    bad_forces[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        receiving_progress_rewards(xy, xy, bad_forces, np.zeros(2, dtype=bool))
    with pytest.raises(ValueError):
        receiving_progress_rewards(xy, xy, np.zeros((2, 10, 3)), np.zeros(2))


def test_decomposition_separates_ball_motion_and_telescopes_with_cap():
    receiver = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [8.0, 0.0]])
    ball = np.array([[4.0, 0.0], [2.0, 0.0], [2.0, 0.0], [8.0, 0.0]])
    result = decompose_proximity_change(receiver, ball)
    distance = np.minimum(np.linalg.norm(receiver - ball, axis=1), 4.0)
    np.testing.assert_allclose(
        result["receiver_motion"] + result["ball_motion"], np.r_[0.0, distance[:-1] - distance[1:]]
    )
    assert result["receiver_motion"][1] == 0.0
    assert result["ball_motion"][2] == 0.0


def test_decomposition_rejects_nonfinite_or_misaligned_history():
    with pytest.raises(ValueError):
        decompose_proximity_change(np.zeros((2, 2)), np.zeros((1, 2)))
    with pytest.raises(ValueError):
        decompose_proximity_change(np.full((2, 2), np.nan), np.zeros((2, 2)))


def test_ball_approaching_stationary_receiver_does_not_create_motion_reward():
    receiver = np.zeros((3, 2))
    ball = np.array([[3.0, 0.0], [2.0, 0.0], [1.0, 0.0]])
    forces = np.zeros((3, 10, 3))
    failed = np.zeros(3, dtype=bool)
    np.testing.assert_allclose(receiver_motion_progress_rewards(receiver, ball, forces, failed), 0)
    forces[2, 0, 0] = 11
    forces[2, 1, 2] = 2
    failed[2] = True
    np.testing.assert_allclose(
        receiver_motion_progress_rewards(receiver, ball, forces, failed), [0, 0, -5]
    )
