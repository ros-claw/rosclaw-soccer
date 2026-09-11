import math

import pytest

from rosclaw_soccer.training.handoff_policy import (
    HandoffPolicyConfig,
    build_handoff_features,
    shape_handoff_action,
)


def test_features_are_bounded_and_target_conditioned() -> None:
    f = build_handoff_features(
        target_dx=2.0, target_dy=1.0, body_yaw=0.2, ball_x=0.1, ball_y=-0.2, residual_norm=4.0
    )
    assert len(f) == 7
    assert all(-1.0 <= x <= 1.0 for x in f)
    assert f != build_handoff_features(
        target_dx=2.0, target_dy=-1.0, body_yaw=0.2, ball_x=0.1, ball_y=-0.2, residual_norm=4.0
    )


def test_action_is_bounded() -> None:
    out = shape_handoff_action(yaw_error=9.0, ball_x=9.0, ball_y=-9.0, config=HandoffPolicyConfig())
    assert out == (1.2, 0.45, 0.35)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_rejected(value: float) -> None:
    with pytest.raises(ValueError):
        build_handoff_features(
            target_dx=value, target_dy=0, body_yaw=0, ball_x=0, ball_y=0, residual_norm=0
        )


@pytest.mark.parametrize("limit", [math.nan, math.inf, -math.inf, 0, -1])
@pytest.mark.parametrize("field", ["max_yaw_rate", "max_lateral_rate", "max_forward_rate"])
def test_invalid_limits_rejected(field: str, limit: float) -> None:
    with pytest.raises(ValueError):
        HandoffPolicyConfig(**{field: limit})


def test_features_are_invariant_to_world_rotation() -> None:
    args = dict(ball_x=0.2, ball_y=-0.1, residual_norm=0.4)
    first = build_handoff_features(target_dx=2, target_dy=1, body_yaw=0.3, **args)
    second = build_handoff_features(target_dx=-1, target_dy=2, body_yaw=0.3 + math.pi / 2, **args)
    assert second == pytest.approx(first)


def test_undefined_target_heading_rejected() -> None:
    with pytest.raises(ValueError):
        build_handoff_features(
            target_dx=0, target_dy=0, body_yaw=1, ball_x=0, ball_y=0, residual_norm=0
        )
