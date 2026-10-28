from types import SimpleNamespace

import pytest

from scripts import rsi_watch_receiver_center_data_audit as watch


def test_failed_export_cannot_start_data_audit(monkeypatch, tmp_path):
    monkeypatch.setattr(watch, "clean", lambda _: None)
    groups = []
    for i in range(3):
        groups.extend(("--pair", str(tmp_path / f"source{i}"), str(tmp_path / f"audit{i}")))
    monkeypatch.setattr(
        watch.sys,
        "argv",
        [
            "watch",
            *groups,
            "--review",
            str(tmp_path / "review"),
            "--dataset",
            str(tmp_path / "data"),
            "--output",
            str(tmp_path / "out"),
            "--export-unit",
            "rosclaw-receiver-center-strict-export-test.service",
            "--producer-head",
            "p",
            "--expected-head",
            "h",
        ],
    )
    calls = []

    def run(command, **_kwargs):
        calls.append(command)
        return SimpleNamespace(
            stdout="LoadState=loaded\nActiveState=failed\nResult=exit-code\nExecMainStatus=1\n"
        )

    monkeypatch.setattr(watch.subprocess, "run", run)
    with pytest.raises(ValueError, match="export failed"):
        watch.main()
    assert len(calls) == 1
    assert not (tmp_path / "out").exists()
