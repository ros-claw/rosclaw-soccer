"""Reader preimport guards; no native or neural workload."""

import sys
from pathlib import Path

import pytest

from scripts import rsi_verify_response_cem as reader


def args(tmp_path):
    result = ["review"]
    for k in ("producer", "world", "motion-csv", "core-source", "output", "previous-producer"):
        result += ["--" + k, str(tmp_path / k)]
    for k in ("expected-head", "expected-producer-head", "expected-producer-hash", "producer-unit"):
        result += ["--" + k, "expected"]
    return result


def test_source_refused_before_project_imports(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", args(tmp_path))
    monkeypatch.setattr(reader.subprocess, "check_output", lambda *a, **k: "wrong\n")
    with pytest.raises(ValueError, match="sealed reader"):
        reader.main()


def test_dirty_bootstrap_refused_before_terminal_or_physics(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", args(tmp_path))
    monkeypatch.setattr(
        reader.subprocess,
        "check_output",
        lambda cmd, **k: "expected\n" if cmd[1] == "rev-parse" else b"dirty",
    )
    with pytest.raises(ValueError, match="bootstrap differs"):
        reader.main()


@pytest.mark.parametrize(
    "terminal",
    [
        "MainPID=123\nResult=success\nSubState=start\nExecMainStatus=0\n",
        "MainPID=0\nResult=exit-code\nSubState=failed\nExecMainStatus=1\n",
    ],
)
def test_never_replay_live_or_failed_producer(tmp_path, monkeypatch, terminal):
    monkeypatch.setattr(sys, "argv", args(tmp_path))
    monkeypatch.setattr(reader.subprocess, "run", lambda *a, **k: None)

    def output(cmd, **kwargs):
        if cmd[0] == "systemctl":
            return terminal
        if cmd[1] == "rev-parse":
            return "expected\n"
        if cmd[1] == "show":
            return Path(cmd[-1].split(":", 1)[1]).read_bytes()
        return ""

    monkeypatch.setattr(reader.subprocess, "check_output", output)
    with pytest.raises(ValueError, match="closed and successful"):
        reader.main()
