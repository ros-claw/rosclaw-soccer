import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.reference_feedback_divergence import CHANNELS, reference_feedback_divergence


def traces():
    trace = {key: np.zeros((300, 1, *tail)) for _, key, tail, _ in CHANNELS}
    trace.update(canonical_qpos=np.zeros((300, 1, 43)), canonical_qvel=np.zeros((300, 1, 41)))
    return trace, copy.deepcopy(trace)


def test_different_channel_clocks_and_units_do_not_claim_causality_or_task_pass():
    z, s = traces()
    s["reference_student_requested_reference_increment_rad"][0, 0, 0] = 0.1
    s["reference_student_applied_reference_increment_rad"][30, 0, 0] = 0.002
    s["pre_motor_joint_target_rad"][30, 0, 0] = 0.002
    s["torque_nm"][30, 0, 9, 0] = 1.5
    s["joint_position_rad"][31, 0, 0] = 0.0001
    s["ball_position_after_step_m"][90, 0, 0] = 0.03
    result = reference_feedback_divergence(z, s)
    ch = result["channels"]
    assert ch["reference_request"]["first_frame_above"]["1e-08"] == 0
    assert ch["reference_applied"]["first_frame_above"]["0.001"] == 30
    assert ch["pd_torque"]["first_frame_above"]["0.1"] == 30
    assert ch["joint_position_before"]["first_frame_above"]["1e-05"] == 31
    assert ch["joint_position_before"]["first_frame_above"]["0.001"] is None
    assert ch["ball_after"]["first_frame_above"]["0.01"] == 90
    assert result["first_difference_is_not_proven_causal_order"]
    assert not result["physical_source_authentication_performed_here"]
    assert "task_pass" not in result


@pytest.mark.parametrize(
    "tamper", ["initial", "prefix", "nav_nonzero", "early_student", "nan", "partial", "dtype"]
)
def test_no_cross_course_or_incomplete_physics_repair(tamper):
    z, s = traces()
    if tamper == "initial":
        s["canonical_qvel"][0, 0, 0] = 1e-12
    elif tamper == "prefix":
        s["joint_target_rad"][29, 0, 0] = -0.0
    elif tamper == "nav_nonzero":
        z["reference_student_applied_reference_increment_rad"][35, 0, 0] = 0.01
    elif tamper == "early_student":
        s["reference_student_applied_reference_increment_rad"][29, 0, 0] = 0.01
    elif tamper == "nan":
        s["torque_nm"][99, 0, 0, 0] = np.nan
    elif tamper == "partial":
        s["ball_position_after_step_m"] = s["ball_position_after_step_m"][:-1]
    else:
        s["force_n"] = s["force_n"].astype(np.float32)
    with pytest.raises(ValueError):
        reference_feedback_divergence(z, s)
