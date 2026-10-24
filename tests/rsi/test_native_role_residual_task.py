import numpy as np
import pytest

from rosclaw_soccer.rsi.native_role_residual_task import (
    native_role_residual_reward,
    residual_reward_hash,
)


def state():
    return dict(
        before_roots=np.zeros((8, 2)),
        after_roots=np.zeros((8, 2)),
        before_ball=np.array([1.0, 0.0]),
        after_ball=np.array([1.0, 0.0]),
        foot_gap=np.full(8, 0.2),
        foot_relative_speed=np.ones(8),
        contact=np.zeros(8, np.bool_),
        contact_age=np.full(8, 999, np.int32),
        collision=np.zeros(8, np.bool_),
        body_failed=np.zeros(8, np.bool_),
        raw_action=np.zeros((8, 3)),
    )


def test_reward_values_actual_close_relative_motion_not_absolute_ball_speed():
    data = state()
    data["contact"][3] = True
    data["contact_age"][3] = 0
    fast, _ = native_role_residual_reward(**data)
    data["foot_relative_speed"][3] = 0.1
    near, parts = native_role_residual_reward(**data)
    assert near[3] > fast[3] + 0.5
    assert parts[3, 3] == 0.5
    data["contact_age"][3] = 26
    data["contact"][3] = False
    after, _ = native_role_residual_reward(**data)
    assert near[3] > after[3] + 0.49
    assert residual_reward_hash() == residual_reward_hash()


def test_collision_and_body_failures_are_costs_not_dropped_samples():
    data = state()
    normal, _ = native_role_residual_reward(**data)
    data["collision"][2] = True
    data["body_failed"][3] = True
    failed, _ = native_role_residual_reward(**data)
    assert failed[2] == pytest.approx(normal[2] - 0.2)
    assert failed[3] == pytest.approx(normal[3] - 5.0)


def test_nonfinite_or_projected_actions_cannot_silently_enter_raw_reward():
    data = state()
    data["raw_action"][0, 0] = 1.1
    with pytest.raises(ValueError):
        native_role_residual_reward(**data)
    data = state()
    data["foot_gap"][0] = np.nan
    with pytest.raises(ValueError):
        native_role_residual_reward(**data)
