"""Admission-only fixtures, not physics or ONNX replay evidence."""

import sys

import pytest

from scripts import rsi_verify_measured_planner_context as reader


def arguments(tmp_path):
    result = ["reader"]
    for name in ("producer", "world", "model-root", "core-source", "output"):
        result += ["--" + name, str(tmp_path / name)]
    for name, value in (
        ("expected-head", "expected"),
        ("expected-producer-head", "producer-head"),
        ("expected-producer-hash", "sha256:" + "1" * 64),
        ("producer-unit", "fixture.service"),
    ):
        result += ["--" + name, value]
    for i in range(4):
        result += ["--previous-producer", str(tmp_path / f"p{i}")]
    return result


@pytest.mark.parametrize("kind", ["source", "output", "missing_cluster"])
def test_sealed_unused_complete_admission(tmp_path, monkeypatch, kind):
    args = arguments(tmp_path)
    if kind == "missing_cluster":
        args = args[:-2]
    if kind == "output":
        (tmp_path / "output").mkdir()
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(
        reader.subprocess,
        "check_output",
        lambda *a, **k: "wrong\n" if kind == "source" else "expected\n",
    )
    with pytest.raises(ValueError, match="sealed reader"):
        reader.main()


@pytest.mark.parametrize(
    "terminal",
    [
        "MainPID=12\nResult=success\nSubState=start\nExecMainStatus=0\n",
        "MainPID=0\nResult=exit-code\nSubState=failed\nExecMainStatus=1\n",
        "MainPID=0\nResult=success\nSubState=dead\nExecMainStatus=0\n",
    ],
)
def test_nonterminal_or_failed_audit_refused(tmp_path, monkeypatch, terminal):
    monkeypatch.setattr(sys, "argv", arguments(tmp_path))
    monkeypatch.setattr(
        reader.subprocess,
        "check_output",
        lambda cmd, **k: "expected\n" if cmd[0] == "git" else terminal,
    )
    with pytest.raises(ValueError, match="successfully closed"):
        reader.main()
    assert not (tmp_path / "output").exists()


def test_completed_audit_hash_required_before_imports(tmp_path, monkeypatch):
    (tmp_path / "producer").mkdir()
    (tmp_path / "producer" / "complete.json").write_text("{}")
    monkeypatch.setattr(sys, "argv", arguments(tmp_path))
    terminal = "MainPID=0\nResult=success\nSubState=exited\nExecMainStatus=0\n"
    monkeypatch.setattr(
        reader.subprocess,
        "check_output",
        lambda cmd, **k: "expected\n" if cmd[0] == "git" else terminal,
    )
    with pytest.raises(ValueError, match="commitment differs"):
        reader.main()
    assert not (tmp_path / "output").exists()
