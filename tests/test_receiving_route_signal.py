import pytest

from rosclaw_soccer.training.receiving_route_signal import receiving_development_route_signal
from rosclaw_soccer.training.role_receiving_courses import ROSTER

REFERENCE = "sha256:" + "1" * 64
CANDIDATE = "sha256:" + "2" * 64
EVIDENCE = "sha256:" + "3" * 64
IDS = tuple(f"course-{i:03d}" for i in range(128))


def outcomes(successes=25):
    return [
        dict(
            course_id=IDS[i],
            agent_id=ROSTER[i % 8],
            reference_execution_hash=REFERENCE,
            candidate_execution_hash=CANDIDATE,
            reference_safe=True,
            candidate_safe=True,
            reference_capture=i < 17,
            candidate_capture=i < successes,
            intervention_applied=True,
            strict_replay=True,
        )
        for i in range(128)
    ]


def evaluate(rows, ids=IDS):
    return receiving_development_route_signal(
        rows,
        expected_course_ids=ids,
        reference_execution_hash=REFERENCE,
        candidate_execution_hash=CANDIDATE,
        evidence_hash=EVIDENCE,
    )


def test_exact_milestone_has_no_training_promotion_or_blind_authority():
    result = evaluate(outcomes())
    assert result["status"] == "ROUTE_VALIDATED"
    assert result["candidate_safe_successes"] == 25
    assert result["new_safe_successes"] == 8
    assert result["evidence_domain"] == "SEEN_DEVELOPMENT_ONLY"
    for key in (
        "training_authorized",
        "promotion_authorized",
        "teacher_qualified",
        "blind_exam_evaluated",
        "evidence_authenticated_by_this_function",
    ):
        assert result[key] is False


def test_below_milestone_is_not_a_route_breakthrough():
    assert evaluate(outcomes(24))["status"] == "ROUTE_NOT_VALIDATED"
    assert evaluate(outcomes(17))["status"] == "ROUTE_NOT_VALIDATED"


@pytest.mark.parametrize("kind", ["loss", "unsafe"])
def test_extra_gains_never_compensate_for_regression_or_unsafe_course(kind):
    rows = outcomes(30)
    rows[0]["candidate_capture" if kind == "loss" else "candidate_safe"] = False
    assert evaluate(rows)["status"] == "ROUTE_NOT_VALIDATED"


def test_fallback_is_counted_separately_and_cannot_gain():
    rows = outcomes()
    for row in rows[:17] + rows[25:]:
        row["intervention_applied"] = False
    result = evaluate(rows)
    assert result["intervention_courses"] == 8
    assert result["unchanged_fallback_courses"] == 120
    rows[17]["intervention_applied"] = False
    with pytest.raises(ValueError, match="fallback"):
        evaluate(rows)


@pytest.mark.parametrize(
    "change",
    [
        {"strict_replay": False},
        {"strict_replay": 1},
        {"intervention_applied": 1},
        {"reference_capture": "true"},
        {"candidate_safe": 1},
        {"reference_safe": False},
        {"candidate_execution_hash": EVIDENCE},
        {"reference_execution_hash": EVIDENCE},
        {"course_id": "unknown"},
        {"agent_id": "red.unknown"},
    ],
)
def test_malformed_or_mixed_lineage_rejected(change):
    rows = outcomes()
    rows[0].update(change)
    with pytest.raises(ValueError):
        evaluate(rows)


def test_repeats_missing_cases_unbalanced_roles_and_changed_reference_rejected():
    for rows in (outcomes()[:-1], outcomes() + [outcomes()[0]], [outcomes()[0]] * 128):
        with pytest.raises(ValueError):
            evaluate(rows)
    rows = outcomes()
    rows[0]["agent_id"] = ROSTER[1]
    with pytest.raises(ValueError, match="sixteen"):
        evaluate(rows)
    rows = outcomes()
    rows[17]["reference_capture"] = True
    with pytest.raises(ValueError, match="17-success"):
        evaluate(rows)
    with pytest.raises(ValueError):
        evaluate(outcomes(), IDS[:-1])
