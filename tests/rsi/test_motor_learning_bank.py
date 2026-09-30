import numpy as np
import pytest

from rosclaw_soccer.rsi.motor_learning_bank import causal_context


def traces():
    body = {
        "root_pose_xyzw_m": np.zeros((300, 1, 7)),
        "root_velocity_world": np.zeros((300, 1, 6)),
        "ball_position_before_step_m": np.ones((300, 1, 3)),
        "ball_linear_velocity_before_step_m_s": np.zeros((300, 1, 3)),
        "foot_geometry_position_before_step_m": np.zeros((300, 1, 4, 3)),
    }
    motor = {"applied_joint_delta_rad": np.zeros((300, 1, 12))}
    return body, motor


def test_context_does_not_read_future_or_motor_parameters():
    body, motor = traces()
    expected = causal_context(body, motor, 60)
    assert len(expected) == 13
    for value in body.values():
        value[31:] = 9999
    motor["applied_joint_delta_rad"][31:] = 0.1
    assert causal_context(body, motor, 90) == expected


@pytest.mark.parametrize("frame", [0, 20, 30])
def test_context_must_precede_motor_and_ball_contact(frame):
    body, motor = traces()
    with pytest.raises(ValueError, match="contact occurred"):
        causal_context(body, motor, frame)
    motor["applied_joint_delta_rad"][frame, 0, 0] = 0.000001
    with pytest.raises(ValueError, match="affected"):
        causal_context(body, motor, 60)


def test_no_nonfinite_action_can_hide_in_the_context_prefix():
    body, motor = traces()
    motor["applied_joint_delta_rad"][5, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        causal_context(body, motor, 60)
