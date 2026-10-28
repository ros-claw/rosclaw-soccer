import numpy as np
import pytest

from rosclaw_soccer.rsi.receiving_progress_reward import receiving_progress_rewards


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
