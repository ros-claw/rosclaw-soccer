import math
from dataclasses import replace

import numpy as np
import pytest

from rosclaw_soccer.growth.locomotion_contact_teacher import G1RollingOptionBridgeConfig
from rosclaw_soccer.training.rolling_strike_entry import inspect_rolling_strike_entry


def evaluate(**changes):
    args = dict(
        pelvis_xy=(0.0, 0.0),
        ball_xy=(0.5, 0.0),
        target_xy=(5.0, 0.0),
        yaw_rad=0.0,
        config=G1RollingOptionBridgeConfig(),
    )
    return inspect_rolling_strike_entry(**(args | changes))


def test_geometry_never_grants_activation():
    result = evaluate()
    assert result.geometry_admissible
    assert result.activation_authorized is False
    with pytest.raises(ValueError):
        replace(result, activation_authorized=True)


@pytest.mark.parametrize("depth", [0.45, 1.2])
def test_original_inclusive_depth_limits(depth):
    assert evaluate(ball_xy=(depth, 0.0)).geometry_admissible


@pytest.mark.parametrize(
    "changes, reason",
    [
        ({"ball_xy": (0.44, 0.0)}, "STANCE_DEPTH"),
        ({"pelvis_xy": (0.0, 0.51)}, "LATERAL_ALIGNMENT"),
        ({"yaw_rad": 0.36}, "HEADING_ALIGNMENT"),
        ({"target_xy": (0.5, 0.0)}, "STANCE_DEPTH"),
    ],
)
def test_original_rejection_reasons(changes, reason):
    assert reason in evaluate(**changes).rejection_reasons


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, 10**1000])
def test_invalid_geometry_rejected_without_overflow(value):
    with pytest.raises(ValueError):
        evaluate(ball_xy=(value, 0.0))
    with pytest.raises(ValueError):
        evaluate(yaw_rad=value)


def test_original_arithmetic_exact_over_seeded_geometry():
    rng = np.random.default_rng(92021)
    config = G1RollingOptionBridgeConfig()
    for _ in range(1000):
        pelvis, ball, target = rng.uniform(-10, 10, size=(3, 2))
        yaw = float(rng.uniform(-math.pi, math.pi))
        ray = target - ball
        ray /= max(float(np.linalg.norm(ray)), 1e-9)
        target_yaw = math.atan2(float(ray[1]), float(ray[0]))
        error = abs(math.atan2(math.sin(target_yaw - yaw), math.cos(target_yaw - yaw)))
        depth = float(np.dot(ball - pelvis, ray))
        side = abs(float(np.dot(ball - pelvis, (-ray[1], ray[0]))))
        result = evaluate(
            pelvis_xy=tuple(map(float, pelvis)),
            ball_xy=tuple(map(float, ball)),
            target_xy=tuple(map(float, target)),
            yaw_rad=yaw,
        )
        assert result.stance_depth_m == depth
        assert result.lateral_error_m == side
        assert result.yaw_error_rad == error
        assert result.direction_xy == tuple(ray)
        assert result.geometry_admissible == (
            config.minimum_strike_stance_depth_m <= depth <= config.maximum_strike_stance_depth_m
            and side <= config.maximum_strike_lateral_error_m
            and error <= config.maximum_strike_yaw_error_rad
        )
