import numpy as np
import pytest

from rosclaw_soccer.rsi.receiver_return_transition_diagnostics import return_transition_diagnostics


def test_handoff_jump_body_failure_and_true_ball_error():
    targets = np.zeros((25, 29), np.float64)
    targets[5:, 2] = 0.8
    targets[7:, 3] = 0.5
    q = np.zeros((25, 10, 79), np.float64)
    q[9, 2, :3] = [1.0, 2.0, 0.1]
    failed = np.zeros((25, 2), np.bool_)
    failed[10, 1] = True
    result = return_transition_diagnostics(
        targets, q, failed, np.array([1.0, 2.0, 0.1]), handoff_control=5, ball_q=0
    )
    assert result["handoff_target_jump_rad"] == 0.8
    assert result["following_window_max_target_jump_rad"] == 0.8
    assert result["first_body_failure_control"] == 10
    assert result["minimum_return_goal_error_m"] == 0.0
    assert result["proof_of_causation"] is False


def test_rejects_nonfinite_and_missing_previous_control():
    targets, q, fail = np.zeros((2, 29)), np.zeros((2, 10, 79)), np.zeros((2, 2), np.bool_)
    with pytest.raises(ValueError):
        return_transition_diagnostics(targets, q, fail, np.zeros(3), handoff_control=0, ball_q=0)
    q[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        return_transition_diagnostics(targets, q, fail, np.zeros(3), handoff_control=1, ball_q=0)
