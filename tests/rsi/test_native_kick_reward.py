import numpy as np
import pytest

from rosclaw_soccer.rsi.native_kick_reward import finite_returns, native_kick_rewards


def signals(ball):
    ball = np.asarray(ball, dtype=float)
    n = len(ball) - 1
    pelvis = np.zeros((n, 7))
    pelvis[:, 2] = 0.8
    pelvis[:, 3] = 1
    return dict(
        ball_positions=ball,
        pelvis_after=pelvis,
        joint_velocity_after=np.zeros((n, 2)),
        applied_torque=np.zeros((n, 2)),
        hard_limits=np.ones(2),
        foot_contact=np.ones(n, bool),
        illegal_contact=np.zeros(n, bool),
        used_parent=np.zeros(n, bool),
    )


def test_real_crossing_gives_precision_reward_only_at_crossing():
    data = signals([[4, 0, 0.11], [4.5, 0, 0.11], [5.5, 0, 0.11]])
    result = native_kick_rewards(**data)
    assert result["crossing_index"] == 1
    assert result["crossing_error_m"] == 0
    assert result["rewards"][1] == 2
    miss = data["ball_positions"].copy()
    miss[:, 1] = 1.0
    assert (
        native_kick_rewards(**dict(data, ball_positions=miss))["rewards"][1] < result["rewards"][1]
    )


def test_future_trajectory_cannot_change_past_rewards():
    a = signals([[1, 0, 0.11], [2, 0, 0.11], [3, 0, 0.11], [6, 0, 0.11]])
    b = dict(a, ball_positions=a["ball_positions"].copy())
    b["ball_positions"][-1, 1] = 5
    np.testing.assert_array_equal(
        native_kick_rewards(**a)["rewards"][:2], native_kick_rewards(**b)["rewards"][:2]
    )


def test_no_contact_has_no_football_reward():
    a = signals([[4, 0, 0.11], [6, 0, 0.11]])
    a["foot_contact"][:] = False
    assert native_kick_rewards(**a)["rewards"].tolist() == [0.0]


def test_returns_discount_in_native_time():
    assert finite_returns([0, 1, 2], gamma=0.5).tolist() == [1.0, 2.0, 2.0]
    with pytest.raises(ValueError):
        finite_returns([np.nan], gamma=0.9995)
