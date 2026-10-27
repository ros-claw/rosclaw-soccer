from types import SimpleNamespace

import pytest

from scripts import rsi_watch_receiver_onpolicy_export as watch


def test_failed_physics_review_blocks_dataset(monkeypatch, tmp_path):
    monkeypatch.setattr(watch, "clean", lambda _: None)
    pairs = []
    for i in range(3):
        pairs.extend(("--pair", str(tmp_path / f"s{i}"), str(tmp_path / f"a{i}")))
    monkeypatch.setattr(
        watch.sys,
        "argv",
        [
            "watch",
            *pairs,
            "--review",
            str(tmp_path / "review"),
            "--output",
            str(tmp_path / "output"),
            "--review-unit",
            "rosclaw-team-review.service",
            "--review-head",
            "r",
            "--producer-head",
            "p",
            "--expected-head",
            "h",
        ],
    )
    calls = []

    def run(cmd, **_kwargs):
        calls.append(cmd)
        return SimpleNamespace(
            returncode=0, stdout="ActiveState=failed\nResult=exit-code\nExecMainStatus=1\n"
        )

    monkeypatch.setattr(watch.subprocess, "run", run)
    with pytest.raises(ValueError, match="review failed"):
        watch.main()
    assert len(calls) == 1
    assert not (tmp_path / "output").exists()
