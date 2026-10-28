import numpy as np
import pytest

from rosclaw_soccer.rsi.relay_restored_receiving_role import restored_receiving_role


def state():
    q = np.zeros(43, dtype=np.float64)
    q[0], q[3], q[36] = 4.0, 1.0, 1.0
    v = np.zeros(41, dtype=np.float64)
    return dict(
        initial_ball=np.zeros(2),
        initial_home=np.array([4.0, 0.0]),
        pass_goal=np.array([5.0, 0.0, 0.11]),
        current_q=q,
        current_v=v,
        past_ball_velocities=np.zeros((140, 2)),
    )


def test_past_actual_launch_is_remembered_but_not_fabricated():
    args = state()
    assert not restored_receiving_role(**args).launch_seen
    args["past_ball_velocities"][120, 0] = 0.41
    assert restored_receiving_role(**args).launch_seen


def test_invalid_history_and_goal_fail_closed():
    args = state()
    args["past_ball_velocities"][0, 0] = np.nan
    with pytest.raises(ValueError):
        restored_receiving_role(**args)
    args = state()
    args["pass_goal"][:2] = 0
    with pytest.raises(ValueError):
        restored_receiving_role(**args)


def test_nondefault_role_offset_is_explicit_but_initial_anchor_stays_fixed():
    args = state()
    fixed = restored_receiving_role(**args, lateral_offset_m=-0.35)
    assert fixed.target_xy == (4.0, 0.12)
    args["past_ball_velocities"][0, 0] = 1.0
    left = restored_receiving_role(**args, lateral_offset_m=-0.35)
    right = restored_receiving_role(**args, lateral_offset_m=0.35)
    assert left.target_xy[1] == -0.35 and right.target_xy[1] == 0.35
    with pytest.raises(ValueError):
        restored_receiving_role(**args, lateral_offset_m=0.41)
