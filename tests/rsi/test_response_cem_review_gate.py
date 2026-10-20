"""Spend-only metadata gate; deliberately never a physical/policy approval."""

import copy

import pytest

from scripts.rsi_response_cem_review_gate import review_decision


def rows(qualified):
    return [
        dict(
            course=dict(parent_index=c, variant_index=v),
            score=dict(qualified=4 * c + v < qualified, physical_prefix_and_suffix_safe=True),
            old_fixed_six_teacher_score_comparison_only=dict(
                qualified=4 * c + v < 6, physical_prefix_and_suffix_safe=True
            ),
        )
        for c in range(4)
        for v in range(4)
    ]


@pytest.mark.parametrize("count", [5, 6, 7, 16])
def test_fixed_gain_threshold_never_approves_physical_candidate(count):
    data = rows(count)
    old = copy.deepcopy(data)
    result = review_decision(data)
    assert result["status"] == (
        "REVIEW_CPU_AUTHORIZED" if count >= 7 else "CEM_FAMILY_REJECTED_STOP"
    )
    assert not result["metadata_is_physical_qualification"]
    assert not result["promotion_authorized"] and not result["hardware_authorized"]
    assert data == old


@pytest.mark.parametrize("kind", ["retention", "safety"])
def test_total_gain_cannot_hide_regression_or_new_unsafe(kind):
    data = rows(8)
    if kind == "retention":
        data[0]["score"]["qualified"] = False
    else:
        data[-1]["score"]["physical_prefix_and_suffix_safe"] = False
    assert review_decision(data)["status"] == "CEM_FAMILY_REJECTED_STOP"


@pytest.mark.parametrize("kind", ["missing", "duplicate", "foreign", "wrong_old", "untyped"])
def test_incomplete_or_foreign_metadata_never_queues(kind):
    data = rows(8)
    if kind == "missing":
        data.pop()
    elif kind == "duplicate":
        data[-1] = copy.deepcopy(data[0])
    elif kind == "foreign":
        data[-1]["course"]["parent_index"] = 4
    elif kind == "wrong_old":
        data[6]["old_fixed_six_teacher_score_comparison_only"]["qualified"] = True
    else:
        data[0]["score"]["qualified"] = 1
    with pytest.raises(ValueError):
        review_decision(data)
