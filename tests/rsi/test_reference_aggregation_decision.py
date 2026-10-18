"""A scoped research branch is neither a training permit nor a policy promotion."""

import copy

import pytest

from rosclaw_soccer.rsi.reference_aggregation_decision import recommend_aggregation


def pair(n, base, evaluated, *, unsafe=0):
    return dict(
        pairs=n,
        baseline_passes=base,
        evaluated_passes=evaluated,
        rescued_pairs=max(0, evaluated - base),
        regressed_pairs=max(0, base - evaluated),
        newly_unsafe_pairs=unsafe,
        net_success_gain=evaluated - base,
    )


def results():
    return dict(
        positive_teacher_vs_student=pair(8, 4, 4),
        retention_zero_vs_student=pair(2, 1, 1),
        retention_legacy_vs_student=pair(2, 2, 2),
    )


@pytest.mark.parametrize(
    "condition", ["contact_failure", "new_unsafe", "retention", "gain", "equal"]
)
def test_branching_is_not_blindly_more_imitation(condition):
    p = results()
    if condition == "contact_failure":
        p["positive_teacher_vs_student"] = pair(8, 4, 0)
    elif condition == "new_unsafe":
        p["positive_teacher_vs_student"] = pair(8, 4, 4, unsafe=1)
    elif condition == "retention":
        p["retention_zero_vs_student"] = pair(2, 1, 0)
    elif condition == "gain":
        p["positive_teacher_vs_student"] = pair(8, 4, 5)
    before = copy.deepcopy(p)
    decision = recommend_aggregation(p, positive_dense_nonfoot=0)
    if condition in ("contact_failure", "new_unsafe"):
        assert decision["finite_numeric_fit_recommended"]
    elif condition == "retention":
        assert decision["action"].startswith("REVIEW_APPROACH")
    else:
        assert decision["action"].startswith("MATCHED_ZERO")
    assert p == before
    assert decision["maximum_new_numeric_candidates"] == 1
    assert not decision["neural_training_authorized"] and not decision["promotion_authorized"]


@pytest.mark.parametrize("tamper", ["partial", "bool", "nan", "count", "gain", "teacher"])
def test_rejects_partial_or_changed_pairing_contracts(tamper):
    p = results()
    if tamper == "partial":
        p["retention_zero_vs_student"]["pairs"] = 1
    elif tamper == "bool":
        p["positive_teacher_vs_student"]["evaluated_passes"] = True
    elif tamper == "nan":
        p["positive_teacher_vs_student"]["rescued_pairs"] = float("nan")
    elif tamper == "count":
        p["positive_teacher_vs_student"]["evaluated_passes"] = 7
    elif tamper == "gain":
        p["positive_teacher_vs_student"]["net_success_gain"] = 1
    else:
        p["positive_teacher_vs_student"] = pair(8, 5, 5)
    with pytest.raises(ValueError):
        recommend_aggregation(p, positive_dense_nonfoot=0)


@pytest.mark.parametrize("count", [True, -1, 9, float("nan")])
def test_dense_contact_evidence_cannot_be_missing_or_malformed(count):
    with pytest.raises(ValueError):
        recommend_aggregation(results(), positive_dense_nonfoot=count)


def test_primitive_success_cannot_hide_dense_nonfoot_contact():
    decision = recommend_aggregation(results(), positive_dense_nonfoot=1)
    assert decision["action"].startswith("REVIEW_DENSE_CONTACT")
    assert not decision["finite_numeric_fit_recommended"]
