from types import SimpleNamespace

import pytest

from scripts import rsi_watch_relay_balance_selector_audit as watch


def test_failed_training_does_not_trigger_audit(monkeypatch, tmp_path):
    monkeypatch.setattr(watch, "clean", lambda _: None)
    monkeypatch.setattr(
        watch.sys,
        "argv",
        [
            "watch",
            "--root",
            str(tmp_path / "root"),
            "--output",
            str(tmp_path / "audit.json"),
            "--unit",
            "rosclaw-team-learning.service",
            "--training-head",
            "train",
            "--expected-head",
            "audit",
        ],
    )
    calls = []

    def run(cmd, **_kwargs):
        calls.append(cmd)
        return SimpleNamespace(
            returncode=0, stdout="ActiveState=failed\nResult=exit-code\nExecMainStatus=1\n"
        )

    monkeypatch.setattr(watch.subprocess, "run", run)
    with pytest.raises(ValueError, match="training failed"):
        watch.main()
    assert len(calls) == 1
