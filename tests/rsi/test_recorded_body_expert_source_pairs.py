"""Controls have no teacher candidate; counterfactual queries must bind their pair."""

from dataclasses import asdict

import pytest

from rosclaw_soccer.rsi.bounded_first_touch_control import BoundedFirstTouchControl
from scripts.rsi_verify_recorded_body_expert import query_controls


@pytest.mark.parametrize("tamper", [None, "partial", "pair", "role", "control", "swing"])
def test_none_body_is_queried_by_original_paired_expert_only(tamper):
    rows = []
    for i in range(8):
        rows.extend(
            [
                dict(group=i, role="control", candidate=None),
                dict(
                    group=i,
                    role="teacher",
                    candidate=dict(
                        swing="bounded_contact_diagnostic",
                        parameters=asdict(BoundedFirstTouchControl()),
                    ),
                ),
            ]
        )
    if tamper == "partial":
        rows.pop()
    elif tamper == "pair":
        rows[0]["group"] = 9
    elif tamper == "role":
        rows[0]["role"] = "teacher"
    elif tamper == "control":
        rows[0]["candidate"] = rows[1]["candidate"]
    elif tamper == "swing":
        rows[1]["candidate"]["swing"] = "mesh"
    if tamper:
        with pytest.raises(ValueError):
            query_controls(rows)
    else:
        result = query_controls(rows)
        assert len(result) == 16 and result[0] is result[1]
