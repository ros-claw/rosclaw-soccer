import sys

import pytest

from scripts import rsi_verify_motor_student as verify


def test_live_evaluation_is_not_restarted_or_read_as_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(
        verify.subprocess,
        "check_output",
        lambda *a, **kw: "MainPID=42\nSubState=start\nResult=success\nExecMainStatus=0\n",
    )
    with pytest.raises(ValueError, match="actually be terminal"):
        verify.admit("unit", tmp_path / "missing", "head")
    assert not (tmp_path / "missing").exists()


def test_verifier_wrong_head_before_models_or_outputs(tmp_path, monkeypatch):
    args = [
        "verify",
        "--mode",
        "replay",
        "--producer",
        "unit",
        str(tmp_path / "producer"),
        "--expected-head",
        "expected",
        "--expected-producer-head",
        "producer-head",
    ]
    for key in ("world", "core-source", "output"):
        args += ["--" + key, str(tmp_path / key)]
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(verify.subprocess, "check_output", lambda *a, **kw: "wrong\n")
    with pytest.raises(ValueError, match="clean sealed"):
        verify.main()
    assert not (tmp_path / "output").exists()
