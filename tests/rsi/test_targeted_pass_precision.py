"""Synthetic objective boundaries only, not physical pass qualification."""

import numpy as np

from rosclaw_soccer.rsi.targeted_pass_precision import precision_rank, targeted_pass_precision


def trajectory(n=400):
    q, v, f = np.zeros((n, 43)), np.zeros((n, 41)), np.zeros((n, 3))
    q[:, 36] = np.linspace(0, 4, n)
    q[:, 38] = 0.11
    v[:, 35] = 2.0
    f[10, 0] = 20
    return q, v, f, np.array([3.75, 0, 0.11])


def test_precision_is_actual_contact_followed_by_arrival_and_full_recovery():
    q, v, f, g = trajectory()
    result = targeted_pass_precision(q, v, f, g, body_healthy=True, recovery_observed=True)
    assert result["precision_0p1m_moderate_speed_success"]
    assert not result["received_by_teammate_verified"]
    assert not result["old_first_touch_exam_regraded"]
    no_recovery = targeted_pass_precision(q, v, f, g, body_healthy=True, recovery_observed=False)
    assert not no_recovery["precision_0p1m_moderate_speed_success"]
    assert precision_rank(no_recovery) < precision_rank(result)


def test_diagnostic_recovery_tail_cannot_rescue_late_arrival():
    q, v, f, g = trajectory(4000)
    q[:3000, 36] = 1.0
    q[3000:, 36] = g[0]
    result = targeted_pass_precision(q, v, f, g, body_healthy=True, recovery_observed=True)
    assert result["formal_native_steps"] == 3000
    assert result["minimum_goal_error_3d_m"] == 2.75
    assert not result["precision_0p1m_moderate_speed_success"]


def test_no_foot_contact_is_missing_measurement_not_zero_error():
    q, v, f, g = trajectory()
    f[:] = 0
    result = targeted_pass_precision(q, v, f, g, body_healthy=True, recovery_observed=True)
    assert result["minimum_goal_error_3d_m"] is None
    assert result["arrival_speed_mps"] is None
    assert not result["precision_0p1m_moderate_speed_success"]


def test_uncontrolled_fast_delivery_and_nonfoot_touch_are_not_positive_teachers():
    q, v, f, g = trajectory()
    v[:, 35] = 5.0
    result = targeted_pass_precision(q, v, f, g, body_healthy=True, recovery_observed=True)
    assert not result["precision_0p1m_moderate_speed_success"]
    v[:, 35] = 2.0
    f[20, 2] = 2
    result = targeted_pass_precision(q, v, f, g, body_healthy=True, recovery_observed=True)
    assert not result["clean_foot"] and not result["precision_0p1m_moderate_speed_success"]


def test_contexts_with_right_kick_move_only_declared_right_leg_not_legacy_left_default():
    from types import SimpleNamespace

    from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
    from scripts.rsi_stroke_search_common import stroke_action

    ep = SimpleNamespace(frame=110, order=np.arange(29), scale=np.ones(29, np.float32))
    mean = np.zeros(29, np.float32)
    action, delta, _ = stroke_action(ep, mean, mean, 110, np.full((2, 6), 0.5), foot="right")
    active = np.flatnonzero(delta)
    assert len(active) == 6 and all(G1_DDS_JOINT_NAMES[i].startswith("right_") for i in active)
    np.testing.assert_allclose(action, delta, atol=1e-8)
