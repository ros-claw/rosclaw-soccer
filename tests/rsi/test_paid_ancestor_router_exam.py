"""Runner admission tests allocate neither CUDA nor native physics."""

import sys

import pytest

from scripts import rsi_audit_paid_ancestor_router as audit
from scripts import rsi_exam_paid_ancestor_router as exam


@pytest.mark.parametrize("runner", [exam, audit])
def test_dirty_source_rejected_before_output_or_physics(runner, tmp_path, monkeypatch):
    args = ["run", "--expected-head", "frozen", "--output", str(tmp_path / "out")]
    if runner is exam:
        args += [
            "--plan-root",
            str(tmp_path / "plan"),
            "--plan-hash",
            "hash",
            "--student-root",
            str(tmp_path / "student"),
        ]
    else:
        args += ["--root", str(tmp_path / "source"), "--producer-head", "producer"]
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **k: b"dirty")
    with pytest.raises(ValueError, match="frozen"):
        runner.main()
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("runner", [exam, audit])
def test_failed_upstream_is_terminal_not_an_implicit_retry(runner, tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "failed.json").write_text("{}")
    args = ["run", "--expected-head", "frozen", "--output", str(tmp_path / "out")]
    if runner is exam:
        args += [
            "--plan-root",
            str(tmp_path / "plan"),
            "--plan-hash",
            "hash",
            "--student-root",
            str(source),
            "--wait-for-student",
        ]
    else:
        args += ["--root", str(source), "--producer-head", "producer", "--wait-for-completion"]
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(
        runner.subprocess, "check_output", lambda cmd, **k: b"" if "status" in cmd else "frozen\n"
    )
    with pytest.raises(ValueError, match="failed"):
        runner.main()
    assert not (tmp_path / "out").exists()
