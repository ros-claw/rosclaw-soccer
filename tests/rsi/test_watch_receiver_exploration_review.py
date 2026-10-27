from types import SimpleNamespace

import pytest

from scripts import rsi_watch_receiver_exploration_review as watch


def test_failed_sample_replay_cannot_launch_review(monkeypatch, tmp_path):
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
            "--identity-review",
            str(tmp_path / "review"),
            "--identity-review-hash",
            "r",
            "--producer-head",
            "p",
            "--identity-producer-head",
            "i",
            "--expected-head",
            "h",
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
