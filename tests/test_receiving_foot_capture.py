"""SIM_ONLY foot-capture teacher contract and inert-state checks."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from rosclaw_soccer.providers.g1.receiving_foot_capture import ReceivingFootCaptureTeacher


def test_foot_capture_teacher_is_bounded_and_hash_bound() -> None:
    teacher = ReceivingFootCaptureTeacher(0.3, 0.3)
    assert teacher.contract_hash.startswith("sha256:")
    assert teacher.activation_ceiling == "SIM_ONLY"
    assert (
        teacher.contract_hash
        != ReceivingFootCaptureTeacher(0.3, 0.3, offset_y_m=0.04).contract_hash
    )
    with pytest.raises(ValueError):
        ReceivingFootCaptureTeacher(0.0, 0.0)
    with pytest.raises(ValueError):
        ReceivingFootCaptureTeacher(0.3, 0.3, offset_x_m=-0.31)
    with pytest.raises(ValueError):
        ReceivingFootCaptureTeacher(0.3, float("nan"))
    with pytest.raises(ValueError):
        ReceivingFootCaptureTeacher(0.3, 0.3, activation_ceiling="REAL")


def test_foot_capture_teacher_inert_after_contact_window() -> None:
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body name='foot'/></worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    teacher = ReceivingFootCaptureTeacher(0.3, 0.3)
    target = np.zeros(29, dtype=np.float64)
    result = teacher.motor_target(
        model=model,
        data=data,
        left_ankle_body=1,
        left_joint_dofs=np.zeros(6, dtype=np.int64),
        left_q=np.zeros(6),
        foundation_target=target,
        current_target=target,
        left_joint_ranges=np.tile(np.asarray((-1.0, 1.0)), (6, 1)),
        ball_position=np.asarray((0.2, 0.0, 0.115)),
        has_foot_contact=True,
        elapsed_sec=0.51,
    )
    assert result is target
