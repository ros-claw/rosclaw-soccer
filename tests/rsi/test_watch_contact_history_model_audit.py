from types import SimpleNamespace

import pytest

from scripts import rsi_watch_contact_history_model_audit as watch


def test_failed_training_does_not_launch_audit(monkeypatch, tmp_path):
    monkeypatch.setattr(watch, "clean", lambda _: None)
    args = ["watch"]
    for name in ("source", "history", "continuous", "parent", "parent-audit", "output"):
        args.extend(("--" + name, str(tmp_path / name)))
    for name in (
        "history-hash",
        "continuous-hash",
        "parent-hash",
        "parent-audit-hash",
        "producer-head",
        "expected-head",
    ):
        args.extend(("--" + name, "x"))
    args.extend(("--unit", "rosclaw-team-training.service"))
    monkeypatch.setattr(watch.sys, "argv", args)
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
