import pytest

from rosclaw_soccer.training.receiving_research_branch import receiving_research_branch
from rosclaw_soccer.training.role_receiving_courses import ROSTER

REFERENCE, CANDIDATE, EVIDENCE = ("sha256:" + c * 64 for c in "abc")


def rows():
    return [
        dict(
            course_id=a,
            agent_id=a,
            reference_policy_hash=REFERENCE,
            candidate_policy_hash=CANDIDATE,
            reference_safe=True,
            candidate_safe=True,
            reference_capture=False,
            candidate_capture=False,
        )
        for a in ROSTER
    ]


def decide(data):
    return receiving_research_branch(
        data,
        expected_course_ids=ROSTER,
        reference_policy_hash=REFERENCE,
        candidate_policy_hash=CANDIDATE,
        evidence_hash=EVIDENCE,
    )


def test_tie_can_continue_research_but_not_claim_improvement_or_promotion():
    result = decide(rows())
    assert result["next_parent_hash"] == CANDIDATE
    assert result["action"] == "RETAIN_RESEARCH_CANDIDATE"
    assert not result["skill_improved_on_public_exam"]
    assert not result["promoted"] and not result["match_qualified"]


@pytest.mark.parametrize("failure", ["safety", "regression"])
def test_failed_candidate_routes_back_to_reference(failure):
    data = rows()
    if failure == "safety":
        data[0]["candidate_safe"] = False
    else:
        data[0]["reference_capture"] = True
    result = decide(data)
    assert result["next_parent_hash"] == REFERENCE
    assert result["action"] == "ROLLBACK_AND_RETRAIN"
    assert result["remediation_course_roles"] == [ROSTER[0]]


def test_gain_does_not_cancel_regression_or_collision():
    data = rows()
    data[0]["candidate_capture"] = True
    data[1]["candidate_safe"] = False
    assert decide(data)["next_parent_hash"] == REFERENCE


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "policy", "bool", "reference", "role"]
)
def test_corrupt_or_incomplete_evidence_stops_not_retries(mutation):
    data = rows()
    if mutation == "missing":
        data.pop()
    elif mutation == "duplicate":
        data.append(data[0])
    elif mutation == "policy":
        data[0]["candidate_policy_hash"] = REFERENCE
    elif mutation == "bool":
        data[0]["candidate_safe"] = 1
    elif mutation == "reference":
        data[0]["reference_safe"] = False
    else:
        data[0]["agent_id"] = ROSTER[1]
    with pytest.raises(ValueError):
        decide(data)


def test_gain_is_public_exam_only():
    data = rows()
    data[0]["candidate_capture"] = True
    result = decide(data)
    assert result["skill_improved_on_public_exam"]
    assert not result["evidence_authenticated_by_this_function"]
    assert result["activation_ceiling"] == "SIM_ONLY"


@pytest.mark.parametrize("expected", [(), ("a", "a"), ("a", None), ["a"]])
def test_bad_expected_courses(expected):
    with pytest.raises(ValueError):
        receiving_research_branch(
            rows(),
            expected_course_ids=expected,
            reference_policy_hash=REFERENCE,
            candidate_policy_hash=CANDIDATE,
            evidence_hash=EVIDENCE,
        )
