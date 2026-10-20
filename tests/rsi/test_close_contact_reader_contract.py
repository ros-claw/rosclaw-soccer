"""Admission-only fixtures: never actual physics success evidence."""

import sys

import pytest

from scripts import rsi_verify_close_contact_robustness as reader


def arguments(tmp_path):
    result = ["reader"]
    for name in ("producer", "world", "motion-csv", "core-source"):
        result += ["--" + name, str(tmp_path / name)]
    return result + [
        "--output",
        str(tmp_path / "out"),
        "--expected-head",
        "expected",
        "--expected-producer-head",
        "producer-head",
        "--expected-producer-hash",
        "sha256:" + "1" * 64,
        "--producer-unit",
        "fixture.service",
    ]


@pytest.mark.parametrize("reason", ["source", "output"])
def test_source_or_existing_output_refused_before_evidence(tmp_path, monkeypatch, reason):
    monkeypatch.setattr(sys, "argv", arguments(tmp_path))
    monkeypatch.setattr(
        reader.subprocess,
        "check_output",
        lambda *a, **k: "wrong\n" if reason == "source" else "expected\n",
    )
    if reason == "output":
        (tmp_path / "out").mkdir()
    with pytest.raises(ValueError, match="sealed reader source"):
        reader.main()


@pytest.mark.parametrize(
    "terminal",
    [
        "MainPID=12\nResult=success\nSubState=start\nExecMainStatus=0\n",
        "MainPID=0\nResult=exit-code\nSubState=failed\nExecMainStatus=1\n",
        "MainPID=0\nResult=success\nSubState=dead\nExecMainStatus=0\n",
    ],
)
def test_nonterminal_or_failed_producer_refused(tmp_path, monkeypatch, terminal):
    monkeypatch.setattr(sys, "argv", arguments(tmp_path))
    monkeypatch.setattr(
        reader.subprocess,
        "check_output",
        lambda cmd, **k: "expected\n" if cmd[0] == "git" else terminal,
    )
    with pytest.raises(ValueError, match="closed successful durable shard"):
        reader.main()
    assert not (tmp_path / "out").exists()


def test_completion_digest_checked_before_project_imports(tmp_path, monkeypatch):
    (tmp_path / "producer").mkdir()
    (tmp_path / "producer" / "complete.json").write_text("{}")
    monkeypatch.setattr(sys, "argv", arguments(tmp_path))
    terminal = "MainPID=0\nResult=success\nSubState=exited\nExecMainStatus=0\n"
    monkeypatch.setattr(
        reader.subprocess,
        "check_output",
        lambda cmd, **k: "expected\n" if cmd[0] == "git" else terminal,
    )
    with pytest.raises(ValueError, match="completion identity differs"):
        reader.main()
    assert not (tmp_path / "out").exists()
