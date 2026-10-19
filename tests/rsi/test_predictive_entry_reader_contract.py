"""Reader admission checks only; not scientific physics/NN qualification."""

import sys

import pytest

from scripts import rsi_verify_predictive_ball_entry as reader


def arguments(tmp_path):
    return [
        "reader",
        "--producer",
        str(tmp_path / "producer"),
        "--world",
        "unused",
        "--core-source",
        "unused",
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
def test_source_or_used_output_rejected_before_reading_physics(tmp_path, monkeypatch, reason):
    monkeypatch.setattr(sys, "argv", arguments(tmp_path))
    monkeypatch.setattr(reader.subprocess, "check_output", lambda *a, **k: "wrong\n")
    if reason == "output":
        (tmp_path / "out").mkdir()
    with pytest.raises(ValueError, match="sealed reader"):
        reader.main()


@pytest.mark.parametrize(
    "terminal",
    [
        "MainPID=12\nResult=success\nExecMainStatus=0\nSubState=start\n",
        "MainPID=0\nResult=exit-code\nExecMainStatus=1\nSubState=failed\n",
        "MainPID=0\nResult=success\nExecMainStatus=0\nSubState=dead\n",
    ],
)
def test_reader_refuses_unfinished_or_failed_producer(tmp_path, monkeypatch, terminal):
    monkeypatch.setattr(sys, "argv", arguments(tmp_path))
    monkeypatch.setattr(
        reader.subprocess,
        "check_output",
        lambda cmd, **k: "expected\n" if cmd[0] == "git" else terminal,
    )
    with pytest.raises(ValueError, match="successfully closed"):
        reader.main()
    assert not (tmp_path / "out").exists()


def test_completion_hash_checked_before_project_imports(tmp_path, monkeypatch):
    producer = tmp_path / "producer"
    producer.mkdir()
    (producer / "complete.json").write_text("{}")
    terminal = "MainPID=0\nResult=success\nExecMainStatus=0\nSubState=exited\n"
    monkeypatch.setattr(sys, "argv", arguments(tmp_path))
    monkeypatch.setattr(
        reader.subprocess,
        "check_output",
        lambda cmd, **k: "expected\n" if cmd[0] == "git" else terminal,
    )
    with pytest.raises(ValueError, match="completion identity"):
        reader.main()
    assert not (tmp_path / "out").exists()
