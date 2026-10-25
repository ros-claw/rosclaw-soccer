"""Fail closed before loading models or allocating physical simulations."""

import json

import pytest

from scripts import rsi_review_actuated_axis_qualification as reader
from scripts.rsi_native_context_recovery_common import digest


@pytest.mark.parametrize("violation", ["hash", "status", "output", "symlink"])
def test_reject_unqualified_or_reused_readback(monkeypatch, tmp_path, violation):
    root = tmp_path / "qualification"
    root.mkdir()
    report = root / "complete.json"
    report.write_text(json.dumps({"status": "NOT_A_QUALIFIED_ACTUAL_TARGET"}))
    expected = digest(report)
    output = tmp_path / "readback"
    if violation == "output":
        output.mkdir()
    if violation == "symlink":
        original = root / "saved.json"
        report.rename(original)
        report.symlink_to(original)
    monkeypatch.setattr("scripts.rsi_native_fresh_common.clean", lambda head: None)
    monkeypatch.setattr(
        "sys.argv",
        [
            "readback",
            "--qualification",
            str(root),
            "--qualification-hash",
            "sha256:not-matching" if violation == "hash" else expected,
            "--expected-head",
            "fixture",
            "--output",
            str(output),
        ],
    )
    with pytest.raises(ValueError):
        reader.main()
    assert not (output / "complete.json").exists()
