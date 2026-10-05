import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.recurrent_teacher_curriculum import sequence_teacher_weights


def rows():
    return [
        dict(
            group=i,
            seed=i // 2,
            lane=0,
            first_contact_frame=70,
            high_quality=i < 4,
            safety_passed=True,
        )
        for i in range(6)
    ]


def test_whole_safe_teachers_and_equal_context_weight_not_failure_gradient():
    records = rows()
    records[3]["seed"] = 8
    before = copy.deepcopy(records)
    w = sequence_teacher_weights(records, profile="balanced_success")
    assert w.shape == (6, 270)
    np.testing.assert_array_equal(w[4:], np.zeros((2, 270)))
    assert w[0].sum() + w[1].sum() == w[2].sum() == w[3].sum()
    assert w[w > 0].mean() == 1
    assert records == before


def test_offline_event_focus_is_exactly_eight_before_twenty_after():
    w = sequence_teacher_weights(rows(), profile="balanced_contact8_recovery20")
    for frame in range(30, 300):
        assert w[0, frame - 30] / w[0, 0] == (8 if 62 <= frame < 90 else 1)
    np.testing.assert_array_equal(w[4:], np.zeros((2, 270)))


def test_clipped_contact_windows_still_balance_whole_context_mass():
    records = rows()
    records[0]["first_contact_frame"] = 0
    records[1]["first_contact_frame"] = 299
    records[2]["first_contact_frame"] = 30
    records[3]["seed"] = 8
    w = sequence_teacher_weights(records, profile="balanced_contact8_recovery20")
    np.testing.assert_allclose(w[0].sum() + w[1].sum(), w[2].sum(), rtol=1e-14)
    np.testing.assert_allclose(w[2].sum(), w[3].sum(), rtol=1e-14)
    np.testing.assert_allclose(w[0], np.full(270, w[0, 0]), rtol=0, atol=0)
    assert w[1, -1] / w[1, 0] == 8


def test_returned_weights_owned_and_failed_missing_contact_allowed():
    records = rows()
    records[-1]["first_contact_frame"] = None
    first = sequence_teacher_weights(records, profile="balanced_success")
    first[:] = 100
    second = sequence_teacher_weights(records, profile="balanced_success")
    assert second[0, 0] != 100
    np.testing.assert_array_equal(second[4:], np.zeros((2, 270)))


@pytest.mark.parametrize(
    "field,value",
    [
        ("group", True),
        ("group", 2),
        ("seed", -1),
        ("lane", 16),
        ("first_contact_frame", True),
        ("first_contact_frame", 300),
        ("first_contact_frame", None),
        ("high_quality", 1),
        ("safety_passed", 0),
    ],
)
def test_malformed_labels_rejected(field, value):
    records = rows()
    records[0][field] = value
    with pytest.raises(ValueError):
        sequence_teacher_weights(records, profile="balanced_success")


def test_insufficient_or_unsafe_teachers_rejected():
    records = rows()
    records[0]["safety_passed"] = False
    with pytest.raises(ValueError, match="at least four"):
        sequence_teacher_weights(records, profile="balanced_success")
    with pytest.raises(ValueError):
        sequence_teacher_weights(rows(), profile="unknown")
