from types import SimpleNamespace

import pytest

from scripts import rsi_watch_relay_balance_selector_admission as watch


def test_failed_audit_cannot_trigger_admission(monkeypatch, tmp_path):
    monkeypatch.setattr(watch, "clean", lambda _: None)
    monkeypatch.setattr(
        watch.sys,
        "argv",
        [
            "watch",
            "--source",
            str(tmp_path / "source"),
            "--audit",
            str(tmp_path / "audit"),
            "--output",
            str(tmp_path / "output.json"),
            "--unit",
            "rosclaw-team-audit.service",
            "--training-head",
            "training",
            "--audit-head",
            "audit",
            "--expected-head",
            "gate",
        ],
    )
    calls = []

    def run(cmd, **_kwargs):
        calls.append(cmd)
        return SimpleNamespace(
            returncode=0, stdout="ActiveState=failed\nResult=exit-code\nExecMainStatus=1\n"
        )

    monkeypatch.setattr(watch.subprocess, "run", run)
    with pytest.raises(ValueError, match="replay failed"):
        watch.main()
    assert len(calls) == 1
    assert not (tmp_path / "output.json").exists()
