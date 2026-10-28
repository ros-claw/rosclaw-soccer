import numpy as np
import pytest

from rosclaw_soccer.rsi.receive_teacher_evidence import audit_teacher_torque


def data():
    parent = np.zeros((3, 10, 58))
    delta = np.zeros_like(parent)
    delta[1, 4, 29] = 3.0
    return parent + delta, parent, delta, np.arange(29, 58)


def test_teacher_delta_binds_actual_torque_and_owned_player():
    torque, parent, delta, receiver = data()
    result = audit_teacher_torque(
        torque, parent, delta, receiver_actuators=receiver, commands=20, active_commands=1
    )
    assert result["native_changed_commands"] == 1
    assert result["maximum_abs_applied_delta_nm"] == 3.0


def test_untracked_torque_and_counterfeit_teammate_change_fail_closed():
    torque, parent, delta, receiver = data()
    torque[2, 2, 30] = 1
    with pytest.raises(ValueError, match="composition"):
        audit_teacher_torque(
            torque, parent, delta, receiver_actuators=receiver, commands=20, active_commands=2
        )
    torque, parent, delta, receiver = data()
    torque[2, 2, 0] = delta[2, 2, 0] = 1
    with pytest.raises(ValueError, match="another player"):
        audit_teacher_torque(
            torque, parent, delta, receiver_actuators=receiver, commands=20, active_commands=2
        )


def test_consecutive_native_count_and_activation_are_mandatory():
    torque, parent, delta, receiver = data()
    with pytest.raises(ValueError, match="consecutive"):
        audit_teacher_torque(
            torque, parent, delta, receiver_actuators=receiver, commands=19, active_commands=1
        )
    with pytest.raises(ValueError, match="exceed"):
        audit_teacher_torque(
            torque, parent, delta, receiver_actuators=receiver, commands=20, active_commands=0
        )
