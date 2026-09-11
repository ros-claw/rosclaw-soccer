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
