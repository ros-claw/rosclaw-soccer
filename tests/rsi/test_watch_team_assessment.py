import json
from types import SimpleNamespace

import pytest

from scripts import rsi_watch_team_assessment as watch


def setup(monkeypatch, tmp_path, state, *, complete):
    source, audit = tmp_path / "source.json", tmp_path / "audit.json"
    source.write_text(json.dumps({"source_head": "simulation"}))
    if complete:
        audit.write_text(
            json.dumps(
                {
                    "source_head": "audit-head",
                    "source_hash": watch.digest(source),
                    "models_hash": "models-hash",
                }
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
            "--audit",
            str(audit),
            "--output",
            str(tmp_path / "output"),
            "--unit",
            "rosclaw-team-test.service",
            "--simulation-head",
            "simulation",
            "--audit-head",
            "audit-head",
            "--models-hash",
            "models-hash",
            "--expected-head",
            "head",
        ],
    )
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(stdout=state, returncode=0)

    monkeypatch.setattr(watch.subprocess, "run", run)
    return calls


def test_watcher_assesses_bound_complete_audit(monkeypatch, tmp_path):
    calls = setup(
        monkeypatch,
        tmp_path,
        "ActiveState=inactive\nResult=success\nExecMainStatus=0\n",
        complete=True,
    )
    watch.main()
    assert len(calls) == 2
    assert calls[1][1] == "scripts/rsi_assess_team_coverage_retention.py"


def test_watcher_blocks_failed_audit(monkeypatch, tmp_path):
    calls = setup(
        monkeypatch,
        tmp_path,
        "ActiveState=failed\nResult=exit-code\nExecMainStatus=1\n",
        complete=False,
    )
    with pytest.raises(ValueError, match="without completion"):
        watch.main()
    assert len(calls) == 1
