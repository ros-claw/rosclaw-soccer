"""Admission tests, not physical qualification or NN evaluation."""

import sys

import pytest

from scripts import rsi_probe_close_contact_robustness as runner


def arguments(tmp_path):
    result = ["runner"]
    for key in ("prior", "prior-review", "world", "motion-csv", "model-root", "core-source"):
        result.extend(["--" + key, str(tmp_path / key)])
    return result + [
        "--output",
        str(tmp_path / "out"),
        "--expected-head",
        "expected",
        "--expected-prior-review-hash",
        "sha256:" + "1" * 64,
    ]


@pytest.mark.parametrize("reason", ["source", "output"])
def test_source_and_unused_output_required_before_evidence(tmp_path, monkeypatch, reason):
    monkeypatch.setattr(sys, "argv", arguments(tmp_path))
    monkeypatch.setattr(
        runner.subprocess,
        "check_output",
        lambda *a, **k: "wrong\n" if reason == "source" else "expected\n",
    )
    if reason == "output":
        (tmp_path / "out").mkdir()
    with pytest.raises(ValueError, match="sealed source and unused output"):
        runner.main()


@pytest.mark.parametrize(
    "flags,reason",
    [
        ([], "explicit shard"),
        (["--cluster", "0"], "explicit shard"),
        (["--preflight-only", "--cluster", "0"], "cannot allocate a shard"),
        (["--preflight-only", "--bank-preflight", "unused"], "cannot allocate a shard"),
    ],
)
def test_incomplete_or_mixed_stage_rejected_without_output(tmp_path, monkeypatch, flags, reason):
    monkeypatch.setattr(sys, "argv", arguments(tmp_path) + flags)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **k: "expected\n")
    with pytest.raises(ValueError, match=reason):
        runner.main()
    assert not (tmp_path / "out").exists()


def test_review_digest_checked_before_physics_imports(tmp_path, monkeypatch):
    review = tmp_path / "prior-review"
    review.mkdir()
    (review / "complete.json").write_text("{}")
    monkeypatch.setattr(sys, "argv", arguments(tmp_path) + ["--preflight-only"])
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **k: "expected\n")
    with pytest.raises(ValueError, match="independent proof differs"):
        runner.main()
    assert not (tmp_path / "out").exists()
