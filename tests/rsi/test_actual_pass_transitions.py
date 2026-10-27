import numpy as np
import pytest

from rosclaw_soccer.rsi.actual_pass_transitions import actual_pass_transitions


def sample():
    n = 3
    return {
        "initial_qpos": np.full(79, -1.0),
        "initial_qvel": np.full(76, -2.0),
        "qpos": np.arange(n * 10 * 79, dtype=float).reshape(n, 10, 79),
        "qvel": np.arange(n * 10 * 76, dtype=float).reshape(n, 10, 76),
        "ball_forces": np.ones((n, 10, 2, 3)),
        "time_before": np.arange(n, dtype=float) * 0.02,
        "A_NN_action": np.zeros((n, 29)),
        "B_V_obs": np.zeros((n, 98)),
        "B_V_action": np.ones((n, 29)),
        "B_V_command": np.zeros((n, 3)),
        "body_failed": np.zeros((n, 2), dtype=bool),
        "B_CAP_teacher_applied_action": np.full((n, 29), 2.0),
        "B_CAP_teacher_active": np.ones(n, dtype=bool),
    }


def test_true_causal_alignment_and_teacher_action():
    trace = sample()
    result = actual_pass_transitions(trace, teacher=True)
    assert np.array_equal(result["qpos_before"][0], trace["initial_qpos"])
    assert np.array_equal(result["qpos_before"][1], trace["qpos"][0, -1])
    assert np.array_equal(result["qvel_before"][2], trace["qvel"][1, -1])
    assert np.array_equal(result["receiver_action_applied"], trace["B_CAP_teacher_applied_action"])
    assert not np.array_equal(result["receiver_action_applied"], result["receiver_base_action"])
    result["qpos_before"][0, 0] = 999
    assert trace["initial_qpos"][0] == -1


def test_missing_teacher_or_invalid_time_fails_closed():
    trace = sample()
    del trace["B_CAP_teacher_applied_action"]
    with pytest.raises(ValueError, match="teacher"):
        actual_pass_transitions(trace, teacher=True)
    trace["time_before"][2] = trace["time_before"][1]
    with pytest.raises(ValueError, match="monotone"):
        actual_pass_transitions(trace, teacher=False)
