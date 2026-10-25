import copy

import pytest

from scripts.rsi_diagnose_teacher_contacts import diagnostic_records


def fixture(pilot=True):
    indices = [0, 2, 8, 22, 27, 30] if pilot else list(range(40))
    audit = dict(
        status="FULL_NN_AIM_COMMAND_RAW_AND_PHYSICAL_EXAM_VERIFIED"
        if pilot
        else "MOTION_SYNERGY_TEACHER_RAW_CEM_FUTURES_AND_FULL_EXAM_VERIFIED",
        actual_worlds=2 * len(indices),
        new_physical_steps=0,
    )
    producer = dict(results=[dict(course_index=i, arms=dict(B={}, C={})) for i in indices])
    return audit, producer


def test_paired_diagnosis_keeps_noncontiguous_identity_and_both_arms():
    audit, producer = fixture()
    before = copy.deepcopy(producer)
    rows = diagnostic_records(audit, producer, paired_aim_pilot=True)
    assert [(i, arm) for i, arm, _ in rows] == [
        (i, arm) for i in [0, 2, 8, 22, 27, 30] for arm in ("B", "C")
    ]
    assert producer == before


def test_original_diagnostic_bank_stays_c_only():
    audit, producer = fixture(False)
    assert [(i, arm) for i, arm, _ in diagnostic_records(audit, producer)] == [
        (i, "C") for i in range(40)
    ]


@pytest.mark.parametrize("defect", ["status", "worlds", "offline", "index", "arms"])
def test_wrong_or_incomplete_pilot_evidence_rejected(defect):
    audit, producer = fixture()
    if defect == "status":
        audit["status"] = "FULL_NN_AIM_COMMAND_PILOT_COMPLETE"
    elif defect == "worlds":
        audit["actual_worlds"] = 11
    elif defect == "offline":
        audit["new_physical_steps"] = 1
    elif defect == "index":
        producer["results"][2]["course_index"] = 2
    else:
        producer["results"][0]["arms"].pop("B")
    with pytest.raises(ValueError, match="audited"):
        diagnostic_records(audit, producer, paired_aim_pilot=True)
