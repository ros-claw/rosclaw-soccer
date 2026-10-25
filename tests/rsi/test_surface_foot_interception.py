import numpy as np
import pytest

from rosclaw_soccer.rsi.surface_foot_interception import surface_foot_interception_command


def context():
    return dict(
        yaw=0.0,
        surfaces=np.array([[-0.11, 0, 0.11], [-0.5, 0.1, 0.05]]),
        ball_position=np.array([0, 0, 0.11]),
        ball_velocity_xy=np.array([0.8, 0.0]),
        previous_nominal_world_command=np.array([0.8, 0.0]),
    )


def test_at_sphere_surface_match_ball_velocity_without_chasing_center():
    args = context()
    command, trace = surface_foot_interception_command(**args)
    np.testing.assert_allclose(command, [0.8, 0, 0], atol=1e-7)
    np.testing.assert_allclose(trace["target"], args["surfaces"][0], atol=1e-12)
    assert trace["selected_foot"] == 0


def test_target_is_actual_three_dimensional_sphere_surface_not_flat_radius_padding():
    args = context()
    args["surfaces"][0] = [-0.2, 0, 0.03]
    _, trace = surface_foot_interception_command(**args)
    assert np.linalg.norm(trace["target"] - args["ball_position"]) == pytest.approx(0.11)
    assert abs(trace["target"][0]) < 0.11


def test_world_half_turn_preserves_body_command_and_does_not_mutate_observations():
    args = context()
    expected, trace = surface_foot_interception_command(**args)
    saved = args["surfaces"].copy()
    turn = np.array([-1, -1, 1])
    args["surfaces"] = args["surfaces"] * turn
    args["ball_position"] = args["ball_position"] * turn
    args["ball_velocity_xy"] *= -1
    args["previous_nominal_world_command"] *= -1
    args["yaw"] = np.pi
    command, _ = surface_foot_interception_command(**args)
    np.testing.assert_allclose(command, expected, atol=1e-7)
    np.testing.assert_allclose(saved * turn, args["surfaces"])


def test_zero_length_ray_has_explicit_diagnostic_not_nan_or_fake_contact():
    args = context()
    args["surfaces"][0] = args["ball_position"]
    command, trace = surface_foot_interception_command(**args)
    assert trace["degenerate_ray"]
    assert np.isfinite(command).all()


def test_invalid_history_not_silently_clipped_into_parent_domain():
    args = context()
    args["previous_nominal_world_command"][0] = 2
    with pytest.raises(ValueError):
        surface_foot_interception_command(**args)
