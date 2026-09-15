import numpy as np
import pytest

from rosclaw_soccer.training.motor_task_rewards import motor_task_targets


def test_shot_credit_does_not_reverse_when_planner_returns_to_chase():
    planner = np.array([[7.25, 0, 0], [3.06, 0.32, 0], [7.25, 0, 0]])
    option = np.tile([7.5, 1.0, 0.115], (3, 1))
    codes = np.array([6, 6, 0])
    bound = motor_task_targets(planner, option, codes, agent_code=6)
    ball = np.array([3.21, 0.30, 0.14])
    velocity = np.array([8.34, 0.026, 2.30])
    assert np.dot(planner[1, :2] - ball[:2], velocity[:2]) < 0
    assert np.dot(bound[1, :2] - ball[:2], velocity[:2]) > 0
    np.testing.assert_array_equal(bound[2], planner[2])
    np.testing.assert_array_equal(motor_task_targets(planner, option, codes, agent_code=2), planner)
    np.testing.assert_array_equal(planner[1], [3.06, 0.32, 0])


def test_mirrored_shot_uses_recorded_left_goal_not_a_red_team_constant():
    planner = np.array([[3.1, 0.0, 0.0]])
    option = np.array([[-1.5, -1.0, 0.115]])
    result = motor_task_targets(planner, option, np.array([2]), agent_code=2)
    np.testing.assert_array_equal(result, option)


@pytest.mark.parametrize("codes", [np.array([9]), np.array([-1]), np.array([2.0])])
def test_invalid_controller_code_is_rejected(codes):
    with pytest.raises(ValueError, match="bound motor-task"):
        motor_task_targets(np.zeros((1, 3)), np.zeros((1, 3)), codes, agent_code=2)


def test_missing_or_nonfinite_target_cannot_silently_become_planner_credit():
    for target in (np.zeros((1, 2)), np.full((1, 3), np.nan)):
        with pytest.raises(ValueError, match="bound motor-task"):
            motor_task_targets(np.zeros((1, 3)), target, np.array([2]), agent_code=2)
