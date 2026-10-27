import numpy as np
import pytest

from rosclaw_soccer.rsi.relay_balance_teacher import RelayBalanceTeacher


def test_zero_is_identity_and_executed_feedback_is_recorded():
    order = np.arange(29)
    teacher = RelayBalanceTeacher(np.zeros((9, 4)), scale=np.ones(29, np.float32), order=order)
    q, v = np.zeros(43), np.zeros(41)
    q[2], q[3] = 0.78, 1
    action = np.zeros(29, np.float32)
    out, info = teacher.apply(action, action, q, v)
    assert np.array_equal(out, action)
    assert np.array_equal(info["BAL_actual_HG"], np.zeros(29))
    assert info["BAL_is_learned_actor"] is False


def test_bounded_slew_and_total_authority():
    order = np.arange(29)
    scale = np.ones(29, np.float32)
    teacher = RelayBalanceTeacher(np.ones((9, 4)), scale=scale, order=order)
    q, v = np.zeros(43), np.zeros(41)
    q[2], q[3], v[4] = 0.78, 1, 2
    action = np.full(29, 0.19, np.float32)
    ref = np.zeros(29, np.float32)
    out, info = teacher.apply(action, ref, q, v)
    assert np.max(np.abs((out - ref) * scale)) <= 0.20002
    assert np.max(np.abs(info["BAL_actual_HG"])) <= 0.020002
    assert np.array_equal(teacher.previous, info["BAL_actual_HG"])
    with pytest.raises(ValueError):
        RelayBalanceTeacher(np.ones((9, 4)) * 2, scale=scale, order=order)
