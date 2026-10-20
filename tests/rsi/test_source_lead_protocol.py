"""Synthetic six-option baseline matching, not physics evidence."""

import copy

import pytest

from rosclaw_soccer.rsi.source_lead_protocol import (
    choose_zero_c_six,
    matched_old_six_selection,
    zero_c_six_options,
)


def six():
    return [
        dict(
            index=i,
            law_index=law,
            offset_root_m=list(offset),
            contact_reference_foot=side,
            score=dict(qualified=False, cost=None),
        )
        for i, (law, offset, side) in enumerate(zero_c_six_options())
    ]


def test_six_fallback_and_stable_cost_order():
    rows = six()
    assert choose_zero_c_six(rows) == 0
    for i in (1, 4):
        rows[i]["score"] = dict(qualified=True, cost=0.1)
    assert choose_zero_c_six(rows) == 1
    rows[4]["score"]["cost"] = 0.05
    assert choose_zero_c_six(rows) == 4


@pytest.mark.parametrize(
    "kind", ["short", "law", "side", "offset", "nan", "qualified_int", "index_bool"]
)
def test_unqualified_six_inputs_rejected(kind):
    rows = six()
    if kind == "short":
        rows.pop()
    elif kind == "law":
        rows[0]["law_index"] = 1
    elif kind == "side":
        rows[3]["contact_reference_foot"] = "right"
    elif kind == "offset":
        rows[1]["offset_root_m"] = [0.0, -0.025, -0.025]
    elif kind == "nan":
        rows[1]["score"] = dict(qualified=True, cost=float("nan"))
    elif kind == "qualified_int":
        rows[1]["score"]["qualified"] = 1
    else:
        rows[1]["index"] = True
    with pytest.raises(ValueError):
        choose_zero_c_six(rows)


def test_matching_old6_cannot_reuse_different_old9_selected_endpoint():
    candidates = []
    for i, (law, offset, side) in enumerate(
        [
            (law, off, "right")
            for law in (0, 3, 4)
            for off in ((0.0, 0.0, 0.0), (0.0, -0.025, -0.025))
        ]
        + [(law, (0.0, 0.0, 0.0), "left") for law in (0, 3, 4)]
    ):
        candidates.append(
            dict(
                index=i,
                law_index=law,
                offset_root_m=list(offset),
                contact_reference_foot=side,
                score=dict(qualified=i == 7, cost=0.1 if i == 7 else None),
            )
        )
    row = dict(status="STUDENT30_PRIVILEGED_RECOVERY_EXECUTED", candidates=candidates, selected=7)
    assert matched_old_six_selection(row) == dict(six_index=4, original_nine_index=7)
    changed = copy.deepcopy(row)
    changed["selected"] = 1
    changed["candidates"][1]["score"] = dict(qualified=True, cost=0.05)
    with pytest.raises(ValueError, match="differs from recorded"):
        matched_old_six_selection(changed)
