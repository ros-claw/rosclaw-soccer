from types import SimpleNamespace

import pytest

from scripts import rsi_watch_continuous_contact_audit as watch


def test_failure_does_not_run_audit(monkeypatch, tmp_path):
    monkeypatch.setattr(watch, "clean", lambda _: None)
    pairs = []
    for i in range(6):
        pairs.extend(("--pair", str(tmp_path / f"r{i}"), str(tmp_path / f"a{i}")))
    monkeypatch.setattr(
        watch.sys,
        "argv",
        [
            "watch",
            *pairs,
            "--dataset",
            str(tmp_path / "dataset"),
            "--features",
            str(tmp_path / "features"),
            "--output",
            str(tmp_path / "out"),
            "--dataset-hash",
            "x",
            "--unit",
            "rosclaw-team-data.service",
            "--producer-head",
            "p",
            "--expected-head",
            "a",
        ],
    )
    calls = []

    def run(cmd, **_kwargs):
        calls.append(cmd)
        return SimpleNamespace(
            returncode=0, stdout="ActiveState=failed\nResult=exit-code\nExecMainStatus=1\n"
        )

    monkeypatch.setattr(watch.subprocess, "run", run)
    with pytest.raises(ValueError, match="producer failed"):
        watch.main()
    assert len(calls) == 1
