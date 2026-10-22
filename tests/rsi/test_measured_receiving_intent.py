import numpy as np
import pytest

from rosclaw_soccer.rsi.measured_receiving_intent import receiving_intent


def state():
    return dict(
        root_xy=np.array([4.0, 0.5]),
        yaw=float(np.pi),
        ball_xy=np.zeros(2),
        ball_velocity_xy=np.zeros(2),
        home_xy=np.array([4.0, 0.5]),
        pass_direction=np.array([1.0, 0.0]),
        launch_seen=False,
    )


def test_receiver_waits_for_actual_pass_not_frame_index():
    s = state()
    before = {k: v.copy() for k, v in s.items() if isinstance(v, np.ndarray)}
    intent = receiving_intent(**s)
    assert intent.role_state == "WAIT_PASS" and not intent.launch_seen
    assert intent.command_body[:2] == (0.0, 0.0)
    for k, value in before.items():
        np.testing.assert_array_equal(s[k], value)


def test_incoming_measured_velocity_changes_intercept_and_history_survives_stop():
    s = state()
    s["ball_velocity_xy"] = np.array([5.0, 0.2])
    s["ball_xy"] = np.array([2.0, 0.0])
    intent = receiving_intent(**s)
    assert intent.role_state == "INTERCEPT" and intent.launch_seen
    assert intent.target_xy[0] > s["ball_xy"][0]
    assert -0.5 <= intent.command_body[0] <= 1
    assert abs(intent.command_body[1]) <= 0.5 and abs(intent.command_body[2]) <= 1
    s["launch_seen"] = True
    s["ball_velocity_xy"][:] = 0
    assert receiving_intent(**s).role_state == "INTERCEPT"


def test_backward_ball_does_not_fabricate_pass_and_nonfinite_rejected():
    s = state()
    s["ball_velocity_xy"] = np.array([-5.0, 0.0])
    assert not receiving_intent(**s).launch_seen
    s["ball_xy"][0] = np.nan
    with pytest.raises(ValueError):
        receiving_intent(**s)


def test_communicated_anchor_activates_support_without_fabricating_ball_launch():
    s = state()
    anchor = np.array([4.0, 0.12])
    intent = receiving_intent(**s, ready_anchor_xy=anchor)
    assert intent.role_state == "MOVE_SUPPORT" and not intent.launch_seen
    assert intent.target_xy == (4.0, 0.12)
    assert abs(intent.command_body[1]) > 0.1
    s["root_xy"] = anchor.copy()
    ready = receiving_intent(**s, ready_anchor_xy=anchor)
    assert ready.role_state == "WAIT_PASS" and ready.command_body[:2] == (0.0, 0.0)
    with pytest.raises(ValueError):
        receiving_intent(**s, ready_anchor_xy=np.array([float("nan"), 0.0]))
