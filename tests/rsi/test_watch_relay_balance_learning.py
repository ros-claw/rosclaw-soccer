from types import SimpleNamespace

import pytest

from scripts import rsi_watch_relay_balance_learning as watch


def test_failed_expanded_worlds_never_train(monkeypatch, tmp_path):
    monkeypatch.setattr(watch, "clean", lambda _: None)
    argv = [
        "watch",
        "--output",
        str(tmp_path / "output"),
        "--expected-head",
        "source",
        "--expanded-head",
        "expanded",
        "--pilot-head",
        "pilot",
    ]
    for i in range(3):
        argv.extend(
            [
                "--group",
                f"rosclaw-team-{i}.service",
                str(tmp_path / f"e{i}"),
                str(tmp_path / f"p{i}"),
                str(tmp_path / f"a{i}"),
            ]
        )
    monkeypatch.setattr(watch.sys, "argv", argv)
    calls = []

    def run(cmd, **_kwargs):
        calls.append(cmd)
        return SimpleNamespace(
            returncode=0, stdout="ActiveState=failed\nResult=exit-code\nExecMainStatus=1\n"
        )

    monkeypatch.setattr(watch.subprocess, "run", run)
    with pytest.raises(ValueError, match="no training"):
        watch.main()
    assert len(calls) == 1
    assert not (tmp_path / "output").exists()
