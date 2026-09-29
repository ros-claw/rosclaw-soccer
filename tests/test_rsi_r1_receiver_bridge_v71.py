from types import SimpleNamespace

import numpy as np
import pytest
from rsi_r1_receiver_bridge_v71 import (
    RetiringReceiverMotor,
    rolling_receive_joint_delta,
    select_grounded_receiver_foot,
)

from rosclaw_soccer.skills.team.motor_retirement import validate_motor_retirement


def test_own_foot_contact_is_required_for_same_frame_idle_retirement():
    motor = RetiringReceiverMotor("red.finisher", False, {})
    assert motor.retirement_request(frame=0, time_sec=0.0) is None
    motor.first_contact_frame = 3
    motor.next_frame = 4
    assert motor.retirement_request(frame=4, time_sec=0.08) is None
    assert motor.propose(SimpleNamespace(agent_id="red.finisher", frame=4)) is None
    request = motor.retirement_request(frame=4, time_sec=0.08)
    assert request is not None
    validate_motor_retirement(
        request,
        agent_id="red.finisher",
        frame=4,
        time_sec=0.08,
        contract_hash=motor.contract_hash,
        proposed_target=False,
    )
    assert motor.retirement_request(frame=5, time_sec=0.10) is None


def test_grounded_selector_is_bounded_and_fails_closed():
    feet = np.array([[1.0, -0.2, 0.05], [1.0, 0.2, 0.05]])
    assert select_grounded_receiver_foot(feet, np.array([1.3, 0.1, 0.115])) == 1
    assert select_grounded_receiver_foot(feet, np.array([1.7, 0.1, 0.115])) == -1
    assert select_grounded_receiver_foot(feet, np.array([1.3, 0.9, 0.115])) == -1
    assert select_grounded_receiver_foot(feet, np.array([0.7, 0.1, 0.115])) == 1
    with pytest.raises(ValueError):
        select_grounded_receiver_foot(feet, np.array([float("nan"), 0.1, 0.115]))


def test_rolling_receive_delta_is_direction_equivariant_and_bounded():
    jacobian = np.eye(3, 6)
    baseline = np.zeros(6)
    limits = np.tile(np.array([-1.0, 1.0]), (6, 1))
    left = rolling_receive_joint_delta(
        np.zeros(3), np.array([-0.3, 0.0, 0.115]), jacobian, baseline, limits
    )
    right = rolling_receive_joint_delta(
        np.zeros(3), np.array([0.3, 0.0, 0.115]), jacobian, baseline, limits
    )
    assert left[0] < 0 < right[0]
    assert np.max(np.abs(left)) <= 0.08
    assert np.max(np.abs(right)) <= 0.08
