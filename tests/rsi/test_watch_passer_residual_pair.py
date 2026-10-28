import pytest

from scripts.rsi_watch_passer_residual_pair import worker_ready


def test_live_terminal_and_collected_workers(tmp_path):
    live = {
        "LoadState": "loaded",
        "ActiveState": "active",
        "Result": "success",
        "ExecMainStatus": "0",
    }
    assert worker_ready(tmp_path, live) is False
    assert worker_ready(tmp_path, {**live, "ActiveState": "inactive"}) is True
    with pytest.raises(ValueError, match="failed"):
        worker_ready(tmp_path, {**live, "ActiveState": "failed", "Result": "exit-code"})
    missing = {"LoadState": "not-found", "ActiveState": "inactive"}
    with pytest.raises(ValueError, match="receipts"):
        worker_ready(tmp_path, missing)
    for name in ("source", "audit"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "complete.json").write_text("{}")
    assert worker_ready(tmp_path, missing) is True
    (tmp_path / "audit/failed.json").write_text("{}")
    with pytest.raises(ValueError, match="receipts"):
        worker_ready(tmp_path, missing)
