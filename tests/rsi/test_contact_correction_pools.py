import pytest

from rosclaw_soccer.rsi.contact_consolidation_data import (
    CORRECTION_BANK_STATUS,
    consolidation_pools,
    context_witness_hash,
)


def bank(correction=True):
    roles = [
        ("retained_W6_skill", 14, True),
        ("source_stroke_teacher", 3, True),
        ("robust_stroke_teacher", 45, True),
        ("failed_teacher_negative", 3, False),
    ]
    if correction:
        roles += [
            ("retention_own_state_correction", 13, True),
            ("retention_own_state_correction", 5, False),
            ("new_skill_teacher_continuation", 18, True),
            ("retained_new_neural_skill", 9, True),
        ]
    rows = []
    for role, count, good in roles:
        for _ in range(count):
            rows.append(dict(index=len(rows), role=role, bc_weight=float(good), task_success=good))
    return rows


def test_all_failed_corrections_are_retained_but_cannot_enter_imitation():
    rows = bank()
    anchors, teachers, repairs = consolidation_pools(rows, correction=True)
    assert (len(anchors), len(teachers), len(repairs)) == (23, 66, 13)
    positives = set(anchors + teachers + repairs)
    assert len(positives) == 102 and len(rows) == 110
    assert all(r["index"] not in positives for r in rows if not r["task_success"])
    assert len(consolidation_pools(bank(False))[0]) == 14


@pytest.mark.parametrize(
    "fault", ["dropped_failure", "label_forged", "missing_new_anchor", "unknown_positive"]
)
def test_bank_cannot_omit_failures_or_erase_new_neural_skills(fault):
    rows = bank()
    if fault == "dropped_failure":
        rows.pop(63)
    elif fault == "label_forged":
        rows[63]["bc_weight"] = 1
    elif fault == "missing_new_anchor":
        rows[-1]["role"] = "new_skill_teacher_continuation"
    else:
        rows[0]["role"] = "arbitrary"
    with pytest.raises(ValueError):
        consolidation_pools(rows, correction=True)


def test_new_data_must_use_new_typed_checkpoint_and_original_context_witness():
    data = dict(status=CORRECTION_BANK_STATUS, context_witness_hash="old-exam-not-new-data")
    assert context_witness_hash(data, "contact-consolidated-v2") == "old-exam-not-new-data"
    for wrong in ("supervised", "ar1-actor-critic", "contact-consolidated"):
        with pytest.raises(ValueError):
            context_witness_hash(data, wrong)
