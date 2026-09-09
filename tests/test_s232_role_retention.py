from copy import deepcopy

import pytest

from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.role_retention import assess_retention


def evidence(old=(1, 0), new=(1, 1)):
    def rows(counts):
        return [
            dict(
                role="playmaker",
                blue=False,
                offset=offset,
                safe=True,
                exact_replay=True,
                foot_only_ball_control=True,
                qualified_passes=count,
                full_match_passed=False,
            )
            for offset, count in zip((-0.1, 0.1), counts, strict=True)
        ]

    value = dict(
        evaluation=dict(parent=rows(old), candidate=rows(new)),
        activation_ceiling="SIM_ONLY",
        promotion_eligible=False,
    )
    value["audit_hash"] = hash_json(value)
    return value


def resign(value):
    value.pop("audit_hash", None)
    value["audit_hash"] = hash_json(value)
    return value


def test_more_passes_cannot_hide_forgetting():
    result = assess_retention(evidence(new=(0, 2)))
    assert not result["research_continuation_accepted"]
    assert result["regressed_courses"][0]["regressions"] == ["qualified_passes"]


def test_retained_gain_is_research_only():
    result = assess_retention(evidence())
    assert result["research_continuation_accepted"]
    assert result["promotion_eligible"] is False
    assert not assess_retention(evidence(new=(1, 0)))["research_continuation_accepted"]


@pytest.mark.parametrize("field", ["safe", "foot_only_ball_control", "full_match_passed"])
def test_other_course_regression_cannot_be_compensated(field):
    value = evidence(old=(0, 0), new=(0, 1))
    value["evaluation"]["parent"][0][field] = True
    value["evaluation"]["candidate"][0][field] = False
    assert not assess_retention(resign(value))["research_continuation_accepted"]


@pytest.mark.parametrize(
    "mutation", ["tamper", "duplicate", "missing", "nan", "no_replay", "unsafe_positive"]
)
def test_invalid_evidence_fails_closed(mutation):
    value = deepcopy(evidence())
    rows = value["evaluation"]["candidate"]
    if mutation == "tamper":
        value["activation_ceiling"] = "REAL"
    else:
        if mutation == "duplicate":
            rows.append(dict(rows[0]))
        elif mutation == "missing":
            rows.pop()
        elif mutation == "nan":
            rows[0]["offset"] = float("nan")
        elif mutation == "no_replay":
            rows[0]["exact_replay"] = False
        else:
            rows[0]["safe"] = False
        if mutation == "nan":
            # Hashing itself must reject non-finite evidence.
            with pytest.raises(ValueError):
                resign(value)
            return
        resign(value)
    with pytest.raises(ValueError):
        assess_retention(value)
