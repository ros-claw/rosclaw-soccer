"""Task-space probe moves one airborne foot within joint safety bounds."""

import numpy as np
import pytest

from rosclaw_soccer.rsi.taskspace_swing_probe import (
    choose_swing_side,
    guard_swing_joint_delta,
    recover_swing_joint_boundary,
    release_joint_delta,
    swing_joint_delta,
)


def test_taskspace_selects_airborne_leg_and_bounds_ik() -> None:
    feet = np.array([[0.0, 0.1, 0.14], [0.0, -0.1, 0.04]])
    ball = np.array([0.6, 0.1, 0.11])
    assert choose_swing_side(feet, ball, -1) == 0
    assert choose_swing_side(feet, ball, 1) == 1
    assert choose_swing_side(feet, ball, -1, acquisition_max_gap_m=0.55) == -1
    feet[0, 2] = 0.05
    assert choose_swing_side(feet, ball, -1) == -1
    jac = np.zeros((3, 6))
    jac[0, 0] = 0.5
    jac[1, 1] = 0.5
    limits = np.tile([-1.0, 1.0], (6, 1))
    delta = swing_joint_delta(
        feet[0],
        ball,
        jac,
        np.zeros(6),
        limits,
        forward_cap_m=0.16,
    )
    assert 0 < delta[0] <= 0.35
    np.testing.assert_array_equal(delta[2:], 0)
    limits[0] = [-0.1, 0.1]
    assert swing_joint_delta(feet[0], ball, jac, np.zeros(6), limits, forward_cap_m=0.16)[0] == 0
    with pytest.raises(ValueError):
        swing_joint_delta(feet[0], ball, jac, np.zeros(6), limits, forward_cap_m=0.2)


def test_lateral_acquisition_waits_for_reachable_swing_foot() -> None:
    feet = np.array([[0.0, -0.52, 0.05], [0.0, -0.80, 0.15]])
    ball = np.array([0.48, -0.43, 0.115])
    assert choose_swing_side(feet, ball, -1, acquisition_max_gap_m=0.55) == 1
    assert (
        choose_swing_side(
            feet,
            ball,
            -1,
            acquisition_max_gap_m=0.55,
            acquisition_max_lateral_gap_m=0.22,
        )
        == -1
    )
    feet[:, 2] = [0.15, 0.05]
    assert (
        choose_swing_side(
            feet,
            ball,
            -1,
            acquisition_max_gap_m=0.55,
            acquisition_max_lateral_gap_m=0.22,
        )
        == 0
    )
    assert (
        choose_swing_side(
            feet,
            ball,
            1,
            acquisition_max_gap_m=0.55,
            acquisition_max_lateral_gap_m=0.22,
        )
        == 1
    )
    with pytest.raises(ValueError):
        choose_swing_side(feet, ball, -1, acquisition_max_lateral_gap_m=0.5)


def test_revalidated_swing_never_keeps_a_now_lower_foot() -> None:
    feet = np.array([[0.0, -0.1, 0.14], [0.0, 0.1, 0.04]])
    ball = np.array([0.45, 0.0, 0.115])
    assert choose_swing_side(feet, ball, 0, revalidate_swing_side=True) == 0
    feet[:, 2] = [0.04, 0.14]
    assert choose_swing_side(feet, ball, 0) == 0
    assert choose_swing_side(feet, ball, 0, revalidate_swing_side=True) == 1
    feet[:, 2] = [0.04, 0.05]
    assert choose_swing_side(feet, ball, 1, revalidate_swing_side=True) == -1
    with pytest.raises(ValueError):
        choose_swing_side(feet, ball, 1, revalidate_swing_side=1)


def test_boundary_recovery_only_moves_measured_near_limit_joint_inward() -> None:
    limits = np.tile([-0.26, 0.26], (6, 1))
    position = np.zeros(6)
    position[5] = -0.255
    delta = recover_swing_joint_boundary(position, np.zeros(6), limits, cap_rad=0.04)
    assert 0 < delta[5] <= 0.04
    np.testing.assert_array_equal(delta[:5], 0.0)
    position[5] = 0.255
    assert recover_swing_joint_boundary(position, np.zeros(6), limits, cap_rad=0.04)[5] < 0
    with pytest.raises(ValueError):
        recover_swing_joint_boundary(position, np.zeros(6), limits, cap_rad=0.4)


def test_taskspace_release_is_causal_and_monotone() -> None:
    contact_delta = np.full(6, 0.2)
    assert release_joint_delta(contact_delta, 1)[0] == pytest.approx(0.19)
    assert release_joint_delta(contact_delta, 20)[0] == 0
    with pytest.raises(ValueError):
        release_joint_delta(contact_delta, 0)
    rounded = np.full(6, 0.35000002)
    assert release_joint_delta(rounded, 1)[0] > 0
    with pytest.raises(ValueError):
        release_joint_delta(np.full(6, 0.3501), 1)


def test_taskspace_lateral_vertical_profiles_are_bounded() -> None:
    foot = np.array([0.0, 0.0, 0.12])
    ball = np.array([0.6, 0.12, 0.11])
    jac = np.zeros((3, 6))
    jac[:, :3] = np.eye(3)
    limits = np.tile([-1.0, 1.0], (6, 1))
    common = (foot, ball, jac, np.zeros(6), limits)
    baseline = swing_joint_delta(*common, forward_cap_m=0.08)
    widened = swing_joint_delta(*common, forward_cap_m=0.08, lateral_cap_m=0.10)
    raised = swing_joint_delta(*common, forward_cap_m=0.08, vertical_offset_m=0.04)
    lowered = swing_joint_delta(*common, forward_cap_m=0.08, vertical_offset_m=-0.04)
    assert 0 < baseline[1] < widened[1] <= 0.35
    assert raised[2] > 0 > lowered[2]
    for option in ({"lateral_cap_m": 0.2}, {"vertical_offset_m": 0.05}):
        with pytest.raises(ValueError):
            swing_joint_delta(*common, forward_cap_m=0.08, **option)


def test_strike_through_keeps_forward_target_active_at_ball_contact() -> None:
    foot = np.array([0.0, 0.0, 0.11])
    ball = np.array([0.22, 0.0, 0.11])
    jac = np.zeros((3, 6))
    jac[:, :3] = np.eye(3)
    limits = np.tile([-1.0, 1.0], (6, 1))
    common = (foot, ball, jac, np.zeros(6), limits)
    baseline = swing_joint_delta(*common, forward_cap_m=0.16)
    through = swing_joint_delta(*common, forward_cap_m=0.16, strike_through_m=0.08)
    assert 0 < baseline[0] < through[0] <= 0.35
    with pytest.raises(ValueError):
        swing_joint_delta(*common, forward_cap_m=0.16, strike_through_m=0.2)


def test_joint_risk_guard_only_tapers_outward_residual_near_measured_limit() -> None:
    limits = np.tile([-1.0, 1.0], (6, 1))
    q = np.zeros(6)
    dq = np.zeros(6)
    delta = np.array([0.2, -0.2, 0.2, -0.2, 0.2, -0.2])
    q[0] = 0.9
    q[1] = -0.9
    q[2] = 0.9
    q[3] = -0.9
    dq[2] = -1.0
    dq[3] = 1.0
    result = guard_swing_joint_delta(q, dq, delta, limits, margin_rad=0.12)
    np.testing.assert_allclose(result[:2], 0.0)
    assert 0.0 < result[2] < delta[2]
    assert delta[3] < result[3] < 0.0
    np.testing.assert_allclose(result[4:], delta[4:])
    with pytest.raises(ValueError):
        guard_swing_joint_delta(q, dq, delta, limits, margin_rad=0.30)
