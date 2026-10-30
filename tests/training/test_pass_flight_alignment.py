import math

import pytest

from rosclaw_soccer.training.pass_flight_alignment import inspect_pass_flight_alignment


def inspect(**changes):
    values = dict(
        ball_xy_m=(0.0, 0.0),
        ball_velocity_xy_mps=(2.0, 0.0),
        receiver_xy_m=(2.0, 0.0),
        capture_radius_m=0.3,
        horizon_sec=2.0,
    )
    return inspect_pass_flight_alignment(**(values | changes))


def test_on_target_and_short_horizon():
    result = inspect()
    assert result.within_capture_corridor
    assert result.closest_time_sec == pytest.approx(1.0)
    assert result.closest_distance_m == pytest.approx(0.0)
    assert not result.promotion_authorized
    assert not inspect(horizon_sec=0.25).within_capture_corridor


def test_positive_projection_is_not_enough_for_a_pass():
    result = inspect(ball_velocity_xy_mps=(2.0, -1.5))
    assert result.forward_speed_mps > 0.4
    assert result.closest_distance_m == pytest.approx(1.2)
    assert not result.within_capture_corridor
    assert not inspect(ball_velocity_xy_mps=(-1.0, 0.0)).within_capture_corridor


def test_rotation_translation_equivariance():
    angle = 1.1
    c, s = math.cos(angle), math.sin(angle)

    def rot(x, y):
        return (c * x - s * y + 3.0, s * x + c * y - 1.0)

    result = inspect(
        ball_xy_m=rot(0.0, 0.0),
        receiver_xy_m=rot(2.0, 0.0),
        ball_velocity_xy_mps=(2 * c + 1.5 * s, 2 * s - 1.5 * c),
    )
    assert result.closest_distance_m == pytest.approx(1.2)
    assert result.lateral_speed_mps == pytest.approx(-1.5)


@pytest.mark.parametrize(
    "changes",
    [
        {"receiver_xy_m": (0.0, 0.0)},
        {"ball_velocity_xy_mps": (0.0, 0.0)},
        {"horizon_sec": -1.0},
        {"capture_radius_m": float("nan")},
        {"ball_xy_m": (float("inf"), 0.0)},
        {"horizon_sec": True},
    ],
)
def test_rejects_invalid_geometry(changes):
    with pytest.raises(ValueError):
        inspect(**changes)
