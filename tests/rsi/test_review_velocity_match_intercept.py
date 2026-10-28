import sys

import pytest

from scripts import rsi_review_velocity_match_intercept as review


def test_unrelated_worker_is_rejected_before_wait(monkeypatch, tmp_path):
    monkeypatch.setattr(review, "clean", lambda _: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "review",
            "--source",
            str(tmp_path / "source"),
            "--output",
            str(tmp_path / "output"),
            "--producer-head",
            "producer",
            "--expected-head",
            "review",
            "--wait-unit",
            "unrelated.service",
        ],
    )
    monkeypatch.setattr(review, "wait_unit", lambda *_: pytest.fail("must not wait"))
    with pytest.raises(ValueError, match="exact bounded worker"):
        review.main()


def test_summary_reports_health_and_nonfoot_in_correct_direction():
    def outcome(handoff, nonfoot):
        return dict(
            strict_both_clean_handoff=handoff,
            strict_receiver_clean_handoff=handoff,
            passer_foot_precedes_receiver=handoff,
            original_handoff=handoff,
            receiver_nonfoot_substep=10 if nonfoot else None,
        )

    rows = [
        dict(
            baseline=outcome(True, False),
            candidate=outcome(False, True),
            health_common=dict(
                raw_healthy_common=True, executed_healthy_common=False, equal_exposure=False
            ),
            baseline_precision=True,
            candidate_precision=False,
            maximum_command_delta=0.2,
        )
    ]
    summary = review.summarize_pairs(rows)
    assert summary["health_common"]["regressions"] == 1
    assert summary["unequal_exposure_pairs"] == 1
    assert summary["receiver_nonfoot"]["new_nonfoot"] == 1
    assert summary["precision"]["regressions"] == 1
    assert summary["changed_command_pairs"] == 1
