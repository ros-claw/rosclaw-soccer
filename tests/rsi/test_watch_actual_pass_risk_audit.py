import json
from types import SimpleNamespace

import pytest

from scripts import rsi_watch_actual_pass_risk_audit as watch


def setup(monkeypatch, tmp_path, *, state, complete):
    source = tmp_path / "source"
    source.mkdir()
    if complete:
        (source / "complete.json").write_text(
            json.dumps(
                dict(
                    status="ACTUAL_PASS_CONTACT_RISK_SUPERVISED_CONTEXT_HOLDOUT_NOT_CONTROL",
                    source_head="train",
                    dataset_hash="data",
                    shard_index=1,
                    shard_count=4,
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
            "--dataset",
            str(tmp_path / "dataset"),
            "--output",
            str(tmp_path / "output"),
            "--unit",
            "rosclaw-team-risk.service",
            "--dataset-hash",
            "data",
            "--training-head",
            "train",
            "--expected-head",
            "audit",
            "--shard-index",
            "1",
        ],
    )
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(stdout=state, returncode=0)

    monkeypatch.setattr(watch.subprocess, "run", run)
    return calls


def test_complete_shard_is_audited(monkeypatch, tmp_path):
    calls = setup(
        monkeypatch,
        tmp_path,
        state="ActiveState=inactive\nResult=success\nExecMainStatus=0\n",
        complete=True,
    )
    watch.main()
    assert len(calls) == 2
    assert calls[1][1] == "scripts/rsi_audit_actual_pass_contact_risk.py"


def test_failed_shard_is_not_retried(monkeypatch, tmp_path):
    calls = setup(
        monkeypatch,
        tmp_path,
        state="ActiveState=failed\nResult=exit-code\nExecMainStatus=1\n",
        complete=False,
    )
    with pytest.raises(ValueError, match="without completion"):
        watch.main()
    assert len(calls) == 1
