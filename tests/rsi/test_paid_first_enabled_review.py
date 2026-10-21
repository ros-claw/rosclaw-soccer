import sys

import pytest

from scripts import rsi_review_paid_first_enabled as runner


def test_dirty_diagnostic_source_cannot_read_physical_evidence(tmp_path, monkeypatch):
    args = ["review", "--qualification-hash", "hash", "--expected-head", "frozen"]
    for name in (
        "qualification-root",
        "parent-root",
        "parent-audit-root",
        "router-root",
        "router-audit-root",
        "output",
    ):
        args.extend(["--" + name, str(tmp_path / name)])
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **k: b"dirty")
    with pytest.raises(ValueError, match="frozen"):
        runner.main()
    assert not (tmp_path / "output").exists()
