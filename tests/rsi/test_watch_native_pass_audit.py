import json
from types import SimpleNamespace

import pytest

from scripts import rsi_watch_native_pass_audit as watch


def setup(monkeypatch, tmp_path, *, state, complete):
    source = tmp_path / "source"
    source.mkdir()
    if complete:
        (source / "complete.json").write_text(
            json.dumps(
                dict(
                    status="CAPTURE_GEOMETRY_SWEEP_DEVELOPMENT32_COLLECTED_PENDING_OWN_REPLAY",
                    source_head="train",
                    records=[{}] * 32,
                    promotion_authorized=False,
                )
            )
        )
    monkeypatch.setattr(watch, "clean", lambda _: None)
    monkeypatch.setattr(
        watch.sys,
        "argv",
        [
            "watch",
            "--source",
            str(source),
            "--output",
            str(tmp_path / "output"),
            "--unit",
            "rosclaw-team-sweep.service",
            "--source-head",
            "train",
            "--expected-head",
            "audit",
        ],
    )
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(stdout=state, returncode=0)

    monkeypatch.setattr(watch.subprocess, "run", run)
    return calls


def test_complete_sweep_replays(monkeypatch, tmp_path):
    calls = setup(
        monkeypatch,
        tmp_path,
        state="ActiveState=inactive\nResult=success\nExecMainStatus=0\n",
        complete=True,
    )
    watch.main()
    assert len(calls) == 2
    assert calls[1][1] == "scripts/rsi_audit_native_receiving_roles.py"


def test_failed_sweep_does_not_retry(monkeypatch, tmp_path):
    calls = setup(
        monkeypatch,
        tmp_path,
        state="ActiveState=failed\nResult=exit-code\nExecMainStatus=1\n",
        complete=False,
    )
    with pytest.raises(ValueError, match="without completion"):
        watch.main()
    assert len(calls) == 1
