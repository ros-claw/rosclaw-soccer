import sys

import pytest

from scripts import rsi_curate_paid_full_entry as runner


def test_dirty_full_entry_curation_cannot_read_teacher_or_model(tmp_path, monkeypatch):
    args = [
        "curate",
        "--qualification-root",
        str(tmp_path / "review"),
        "--qualification-hash",
        "hash",
        "--expected-head",
        "frozen",
        "--output",
        str(tmp_path / "out"),
        "--teacher-roots",
    ]
    args += [str(tmp_path / f"teacher{i}") for i in range(3)]
    args += ["--teacher-audit-roots"] + [str(tmp_path / f"audit{i}") for i in range(3)]
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **k: b"dirty")
    with pytest.raises(ValueError, match="frozen"):
        runner.main()
    assert not (tmp_path / "out").exists()
