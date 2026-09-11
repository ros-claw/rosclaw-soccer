import math
from dataclasses import FrozenInstanceError

import pytest

from rosclaw_soccer.world.ground_intercept import propose_ground_intercept


def proposal(**kwargs):
    values = dict(root_xy=(0.0, 0.0), ball_xy=(0.7, 0.0), ball_velocity_xy=(0.2, 0.0))
    return propose_ground_intercept(**(values | kwargs))


def test_recending_ball_lead_is_bounded_not_claimed_reachable():
    p = proposal()
    assert p.horizon_sec == 0.8 and not p.kinematically_reachable
    assert p.predicted_ball_xy == pytest.approx((0.86, 0))
    assert p.root_target_xy == pytest.approx((0.58, 0))
    assert p.activation_ceiling == "SIM_ONLY"
    with pytest.raises(FrozenInstanceError):
        p.horizon_sec = 2


@pytest.mark.parametrize("velocity", [-0.4, 0.0, 0.2])
def test_earliest_contact_region_solution(velocity):
    p = proposal(ball_velocity_xy=(velocity, 0.0), maximum_horizon_sec=2.0)
    assert p.kinematically_reachable
    assert p.horizon_sec == pytest.approx((0.7 - 0.28) / (0.7 - velocity), abs=1e-12)
    assert math.dist(p.root_target_xy, p.predicted_ball_xy) == pytest.approx(0.28)
    assert math.hypot(*p.root_target_xy) == pytest.approx(0.7 * p.horizon_sec)


@pytest.mark.parametrize("ball", [(0.0, 0.0), (0.1, 0.1), (0.28, 0.0)])
def test_inside_contact_region_does_not_command_root_reposition(ball):
    p = proposal(ball_xy=ball)
    assert p.root_target_xy == (0.0, 0.0) and p.horizon_sec == 0
    assert p.kinematically_reachable


def test_planar_rotation_translation_equivariance():
    p = proposal(ball_xy=(0.5, 0.2), ball_velocity_xy=(0.1, -0.2))
    q = proposal(root_xy=(3.0, -2.0), ball_xy=(2.8, -1.5), ball_velocity_xy=(0.2, 0.1))
    assert q.horizon_sec == pytest.approx(p.horizon_sec)
    assert q.root_target_xy == pytest.approx((3 - p.root_target_xy[1], -2 + p.root_target_xy[0]))
    assert q.predicted_ball_xy == pytest.approx(
        (3 - p.predicted_ball_xy[1], -2 + p.predicted_ball_xy[0])
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("root_xy", [0.0, 0.0]),
        ("ball_xy", (0.0,)),
        ("ball_xy", (float("nan"), 0.0)),
        ("ball_velocity_xy", (0.7, 0.0)),
        ("ball_velocity_xy", (float("inf"), 0.0)),
        ("root_xy", (True, 0.0)),
        ("root_xy", (1001.0, 0.0)),
        ("root_speed_mps", True),
        ("root_speed_mps", 3.0),
        ("contact_standoff_m", 0.01),
        ("maximum_horizon_sec", 3.0),
    ],
)
def test_invalid_or_unmodelled_inputs_fail_closed(key, value):
    with pytest.raises(ValueError):
        proposal(**{key: value})
