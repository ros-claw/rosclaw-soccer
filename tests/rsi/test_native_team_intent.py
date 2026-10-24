import numpy as np
import pytest

from rosclaw_soccer.rsi.native_team_intent import (
    ROLES,
    clearance_command,
    command_toward,
    finisher_intercept_target,
    foot_tracking_command,
    role_target,
)


@pytest.mark.parametrize("role", ROLES)
def test_team_role_law_is_half_turn_symmetric(role):
    ball = np.array([1.0, 0.7])
    np.testing.assert_array_equal(role_target("red", role, ball), -role_target("blue", role, -ball))


def test_commands_use_actual_body_frame_and_qualified_domain():
    red = command_toward(np.zeros(2), 0.0, np.array([2.0, 1.0]), np.array([3.0, 1.0]))
    blue = command_toward(np.zeros(2), np.pi, np.array([-2.0, -1.0]), np.array([-3.0, -1.0]))
    np.testing.assert_allclose(red, blue, atol=1e-6)
    assert -0.5 <= red[0] <= 1 and -0.5 <= red[1] <= 0.5 and -1 <= red[2] <= 1
    assert command_toward(np.zeros(2), 0.0, np.zeros(2), np.array([1.0, 1.0]))[0] == 0


def test_unknown_role_or_nonfinite_state_is_not_silently_coached():
    with pytest.raises(ValueError):
        role_target("red", "nobody", np.zeros(2))
    with pytest.raises(ValueError):
        command_toward(np.zeros(2), float("nan"), np.ones(2), np.ones(2))


def test_interception_leads_measured_motion_and_preserves_team_symmetry():
    root, ball, velocity = np.array([-1.5, 0.0]), np.zeros(2), np.array([0.0, 0.8])
    target, horizon, reachable = finisher_intercept_target("red", root, ball, velocity)
    opposite, oh, oreach = finisher_intercept_target("blue", -root, -ball, -velocity)
    np.testing.assert_allclose(target, -opposite, atol=1e-12)
    assert horizon == oh and reachable == oreach
    assert 0 < horizon <= 2.0 and target[1] > 0.8


def test_fast_ball_is_not_clipped_into_slow_model_domain():
    target, horizon, reachable = finisher_intercept_target(
        "red", np.zeros(2), np.array([1.0, 0.0]), np.array([2.0, 0.0])
    )
    assert horizon == 0.2 and not reachable
    assert abs(target[0] - (1.0 + 0.4 - 0.25)) < 1e-12
    with pytest.raises(ValueError):
        command_toward(np.zeros(2), 0.0, np.ones(2), np.ones(2), maximum_forward_speed=1.01)


def test_measured_foot_tracking_has_no_quarter_meter_root_dead_zone():
    root = np.array([-0.25, 0.0])
    feet = np.array([[-0.2, 0.1], [-0.2, -0.1]])
    saved = feet.copy()
    command, world = foot_tracking_command(root, 0.0, feet, np.zeros(2), np.zeros(2), np.zeros(2))
    assert command[0] > 0
    assert np.linalg.norm(world) <= 0.08 + 1e-6
    np.testing.assert_array_equal(feet, saved)


def test_foot_tracking_follows_ball_speed_even_without_position_error():
    command, world = foot_tracking_command(
        np.array([-0.1, 0.0]),
        0.0,
        np.array([[0.0, 0.0], [0.0, 0.1]]),
        np.zeros(2),
        np.array([0.8, 0.0]),
        np.array([0.8, 0.0]),
    )
    np.testing.assert_allclose(command, [0.8, 0, 0], atol=1e-6)
    np.testing.assert_allclose(world, [0.8, 0], atol=1e-6)


def test_foot_feedback_uses_body_frame_and_half_turn_symmetry():
    args = [
        np.array([-0.4, 0.0]),
        np.array([[-0.3, 0.1], [-0.3, -0.1]]),
        np.array([0.0, 0.1]),
        np.array([0.0, 0.8]),
        np.zeros(2),
    ]
    red, rw = foot_tracking_command(args[0], 0.0, *args[1:])
    blue, bw = foot_tracking_command(-args[0], np.pi, *[-x for x in args[1:]])
    np.testing.assert_allclose(red, blue, atol=1e-6)
    np.testing.assert_allclose(rw, -bw, atol=1e-6)


def test_foot_tracking_refuses_invalid_state_instead_of_sanitizing_it():
    with pytest.raises(ValueError):
        foot_tracking_command(
            np.zeros(2), 0.0, np.zeros((3, 2)), np.zeros(2), np.zeros(2), np.zeros(2)
        )
    with pytest.raises(ValueError):
        foot_tracking_command(
            np.zeros(2), 0.0, np.zeros((2, 2)), np.zeros(2), np.zeros(2), np.ones(2)
        )


def test_clearance_reuses_world_projection_and_preserves_tangential_run():
    command, feasible, constrained = clearance_command(
        np.array([0.8, 0.3, 0.4]), 0.0, np.array([[0.5, 0.0]]), maximum_speed_mps=1.0
    )
    assert feasible and constrained
    assert command[0] <= -0.06 + 1e-6
    assert abs(command[1] - 0.3) < 1e-6
    assert abs(command[2] - 0.4) < 1e-6


def test_clearance_halfplanes_keep_body_domain_without_postprojection_clipping():
    for yaw in (0.0, 0.7, np.pi):
        cmd, feasible, _ = clearance_command(
            np.array([1.0, 0.5, 1.0]), yaw, np.array([[0.4, 0.0]]), maximum_speed_mps=1.0
        )
        assert feasible
        assert -0.5 - 1e-6 <= cmd[0] <= 1.0 + 1e-6
        assert abs(cmd[1]) <= 0.5 + 1e-6
        assert np.linalg.norm(cmd[:2]) <= 1.0 + 1e-6


def test_clearance_infeasible_symmetric_squeeze_is_explicit_zero_not_fake_safe():
    cmd, feasible, constrained = clearance_command(
        np.array([0.8, 0.0, 0.8]), 0.0, np.array([[-0.3, 0.0], [0.3, 0.0]]), maximum_speed_mps=1.0
    )
    assert not feasible and constrained
    np.testing.assert_array_equal(cmd, np.zeros(3))
