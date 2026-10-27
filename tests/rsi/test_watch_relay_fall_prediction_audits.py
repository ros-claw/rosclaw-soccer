from types import SimpleNamespace

import pytest

from scripts import rsi_watch_relay_fall_prediction_audits as watch


def test_failed_training_never_audits(monkeypatch, tmp_path):
    monkeypatch.setattr(watch, "clean", lambda _: None)
    monkeypatch.setattr(
        watch.sys,
        "argv",
        [
            "watch",
            "--root",
            str(tmp_path / "source"),
            "--output",
            str(tmp_path / "audit"),
            "--unit",
            "rosclaw-team-learning.service",
            "--training-head",
            "source",
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
    with pytest.raises(ValueError, match="failed training"):
        watch.main()
    assert len(calls) == 1
