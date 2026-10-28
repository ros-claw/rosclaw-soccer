import sys

import pytest

from scripts import rsi_watch_multiblock_risk as watcher


def test_rejects_wrong_block_count_before_launch(monkeypatch, tmp_path):
    monkeypatch.setattr(watcher, "clean", lambda _: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "watch",
            "--block",
            "r",
            "pending",
            "f",
            "h",
            "complete",
            "--expected-head",
            "head",
            "--output",
            str(tmp_path / "out.json"),
        ],
    )
    with pytest.raises(ValueError, match="three explicit blocks"):
        watcher.main()
    assert not (tmp_path / "out.json").exists()
