import copy

import pytest

from scripts.rsi_review_paid_unseen_growth import validate_rows


def pair():
    rows = [
        dict(case=i, repeat=j, quality=i == 0, healthy_success=i == 0)
        for i in range(32)
        for j in range(2)
    ]
    return dict(records=rows, repeats_exact=True), dict(
        outcomes=copy.deepcopy(rows), repeats_exact=True
    )


def test_all_contexts_not_repeat_inflated():
    producer, audit = pair()
    rows = validate_rows(producer, audit)
    assert len(rows) == 32 and sum(r["healthy_success"] for r in rows) == 1


@pytest.mark.parametrize("mutation", ["missing", "reorder", "mismatch", "repeat", "nonbool"])
def test_fail_closed_on_incomplete_or_contradictory_evidence(mutation):
    producer, audit = pair()
    if mutation == "missing":
        producer["records"].pop()
    elif mutation == "reorder":
        audit["outcomes"].reverse()
    elif mutation == "mismatch":
        audit["outcomes"][0]["healthy_success"] = False
    elif mutation == "repeat":
        producer["repeats_exact"] = False
    else:
        producer["records"][0]["quality"] = audit["outcomes"][0]["quality"] = 1
    with pytest.raises(ValueError):
        validate_rows(producer, audit)
