import json
from types import SimpleNamespace

import pytest

from scripts import rsi_watch_relay_learning as watch


def arguments(monkeypatch, tmp_path):
    monkeypatch.setattr(watch, "clean", lambda _: None)
    args = [
        "watch",
        "--output",
        str(tmp_path / "out"),
        "--expected-head",
        "new",
        "--source-head",
        "old",
    ]
    for i in range(3):
        source, audit = tmp_path / f"s{i}", tmp_path / f"a{i}"
        source.mkdir()
        audit.mkdir()
        src = source / "complete.json"
        src.write_text(
            json.dumps(
                {
                    "status": "NEURAL_RELAY_DEVELOPMENT32_COLLECTED_PENDING_OWN_REPLAY",
                    "source_head": "old",
                }
            )
        )
        (audit / "complete.json").write_text(
            json.dumps(
                {
                    "status": "ALL_NEURAL_RELAY_OWN_HISTORY_NATIVE_REPLAY_MATCH",
                    "producer_hash": watch.digest(src),
                    "worlds": 32,
                }
            )
        )
        args += ["--pair", str(source), str(audit), f"rosclaw-team-{i}.service"]
    monkeypatch.setattr(watch.sys, "argv", args)


def test_closed_replays_chain_full_data_audit_and_three_seeds(monkeypatch, tmp_path):
    arguments(monkeypatch, tmp_path)
    calls = []

    def run(cmd, **_kwargs):
        calls.append(cmd)
        if cmd[0] == "systemctl":
            return SimpleNamespace(
                returncode=0, stdout="ActiveState=inactive\nResult=success\nExecMainStatus=0\n"
            )
        if cmd[1] == "scripts/rsi_export_relay_fall_dataset.py":
            out = tmp_path / "out" / "data"
            out.mkdir()
            (out / "complete.json").write_text("{}")
        if cmd[1] == "scripts/rsi_audit_relay_fall_dataset.py":
            out = tmp_path / "out" / "data-audit"
            out.mkdir()
            (out / "complete.json").write_text("{}")
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(watch.subprocess, "run", run)
    watch.main()
    assert len(calls) == 8
    assert [c[1] for c in calls[3:]] == [
        "scripts/rsi_export_relay_fall_dataset.py",
        "scripts/rsi_audit_relay_fall_dataset.py",
        *["scripts/rsi_train_relay_fall_predictor.py"] * 3,
    ]


def test_failure_blocks_training(monkeypatch, tmp_path):
    arguments(monkeypatch, tmp_path)
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
