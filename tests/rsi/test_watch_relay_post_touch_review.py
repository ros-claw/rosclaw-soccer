from types import SimpleNamespace

import pytest

from scripts import rsi_watch_relay_post_touch_review as watch


def test_failed_worker_blocks_review(monkeypatch, tmp_path):
    monkeypatch.setattr(watch, "clean", lambda _: None)
    groups = []
    for i in range(3):
        groups.extend(
            [
                "--group",
                str(tmp_path / str(i)),
                f"rosclaw-team-{i}.service",
                str(tmp_path / f"old{i}"),
                str(tmp_path / f"audit{i}"),
            ]
        )
    monkeypatch.setattr(
        watch.sys,
        "argv",
        [
            "watch",
            *groups,
            "--dataset",
            str(tmp_path / "data"),
            "--dataset-hash",
            "d",
            "--producer-head",
            "p",
            "--expected-head",
            "r",
            "--output",
            str(tmp_path / "out"),
        ],
    )
    calls = []

    def run(cmd, **_kwargs):
        calls.append(cmd)
        return SimpleNamespace(
            returncode=0, stdout="ActiveState=failed\nResult=exit-code\nExecMainStatus=1\n"
        )

    monkeypatch.setattr(watch.subprocess, "run", run)
    with pytest.raises(ValueError, match="failed"):
        watch.main()
    assert len(calls) == 1
