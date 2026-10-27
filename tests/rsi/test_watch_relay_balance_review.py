from types import SimpleNamespace

import pytest

from scripts import rsi_watch_relay_balance_review as watch


def test_failed_audit_prevents_review(monkeypatch, tmp_path):
    monkeypatch.setattr(watch, "clean", lambda _: None)
    argv = [
        "watch",
        "--output",
        str(tmp_path / "review.json"),
        "--expected-head",
        "current",
        "--source-head",
        "physics",
        "--prior-head",
        "prior",
    ]
    for i in range(3):
        argv.extend(
            [
                "--group",
                str(tmp_path / f"s{i}"),
                str(tmp_path / f"a{i}"),
                f"rosclaw-team-{i}.service",
                str(tmp_path / f"p{i}"),
                str(tmp_path / f"pa{i}"),
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
    with pytest.raises(ValueError, match="failed replay"):
        watch.main()
    assert len(calls) == 1
