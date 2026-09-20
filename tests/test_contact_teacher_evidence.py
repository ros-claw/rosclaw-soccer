import numpy as np
import pytest

from rosclaw_soccer.training.contact_teacher_ablation import ContactTeacherSuppression
from rosclaw_soccer.training.contact_teacher_evidence import inspect_teacher_suppression

ROSTER = ("blue.playmaker", "red.playmaker")
CONTRACT = ContactTeacherSuppression("blue.playmaker", 2)


def trace():
    return {
        "time": np.arange(5) * 0.02,
        "contact_teacher_suppression_contract": np.asarray([CONTRACT.contract_hash] * 5),
        "contact_teacher_suppressed_agent": np.asarray(["", ""] + [CONTRACT.agent_id] * 3),
        "contact_teacher_agent_code": np.ones(5, dtype=np.int64),
        "contact_teacher_active": np.asarray([True, True, False, False, False]),
        "contact_teacher_last_substep_valid": np.asarray([True, True, False, False, False]),
        "contact_teacher_peak_torque_nm": np.asarray([3.0, 3.0, 0.0, 0.0, 0.0]),
        "contact_teacher_residual_torque_nm": np.zeros((5, 29)),
        "contact_teacher_tracking_adjustment_nm": np.zeros((5, 29)),
    }


def inspect(value):
    return inspect_teacher_suppression(value, contract=CONTRACT, agent_ids=ROSTER)


def test_eligibility_is_not_execution():
    result = inspect(trace())
    assert result.suppressed_frames == result.eligible_but_unexecuted_frames == 3


def test_other_teacher_remains_allowed():
    value = trace()
    value["contact_teacher_agent_code"][3] = 2
    value["contact_teacher_active"][3] = True
    value["contact_teacher_last_substep_valid"][3] = True
    value["contact_teacher_peak_torque_nm"][3] = 10
    value["contact_teacher_residual_torque_nm"][3] = 1
    assert inspect(value).eligible_but_unexecuted_frames == 2


@pytest.mark.parametrize(
    "field,new_value",
    [
        ("contact_teacher_active", True),
        ("contact_teacher_last_substep_valid", True),
        ("contact_teacher_peak_torque_nm", 0.001),
        ("contact_teacher_residual_torque_nm", 0.001),
        ("contact_teacher_tracking_adjustment_nm", 0.001),
        ("contact_teacher_suppressed_agent", ""),
        ("contact_teacher_suppression_contract", "sha256:" + "0" * 64),
        ("contact_teacher_agent_code", 9),
        ("contact_teacher_peak_torque_nm", np.nan),
        ("time", 0.123),
    ],
)
def test_reject_bad_or_contradictory_evidence(field, new_value):
    value = trace()
    value[field][3] = new_value
    with pytest.raises(ValueError):
        inspect(value)


def test_requires_actual_post_boundary_measurements():
    value = {key: array[:2] for key, array in trace().items()}
    with pytest.raises(ValueError):
        inspect(value)


def test_cannot_hide_execution_under_no_candidate():
    value = trace()
    value["contact_teacher_agent_code"][3] = 0
    value["contact_teacher_active"][3] = True
    with pytest.raises(ValueError, match="no candidate"):
        inspect(value)
