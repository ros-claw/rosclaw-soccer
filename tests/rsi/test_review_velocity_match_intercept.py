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
