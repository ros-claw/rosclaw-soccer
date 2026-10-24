"""Boundary proximity is not evidence of an actual B contact."""

import pytest

from scripts.rsi_close_two_body_response_refinement import summarize


def outcome(contact=True, strict=True):
    return dict(
        whole_body_healthy=True,
        passer_full100NNrecovery=True,
        passer_clean_foot=True,
        B_excluded_contact=False,
        B_legal_functional_receive_before_deadline=True,
        combined_precision_and_functional_receive=strict,
        incoming=dict(
            A_first_foot_strike_substep=900, B_first_ball_contact_substep=1200 if contact else None
        ),
    )


def test_strict_boundary_without_contact_not_full_chain():
    r = summarize([dict(outcome=outcome(contact=False))])
    assert r == dict(
        controlled=0, strict_composite=1, actual_contact_and_strict=0, all_healthy_clean=True
    )


def test_contact_and_precision_reported_separately():
    r = summarize([dict(outcome=outcome()), dict(outcome=outcome(strict=False))])
    assert r["controlled"] == 2
    assert r["actual_contact_and_strict"] == r["strict_composite"] == 1


def test_missing_actual_contact_evidence_rejected():
    o = outcome()
    del o["incoming"]["B_first_ball_contact_substep"]
    with pytest.raises(KeyError):
        summarize([dict(outcome=o)])
