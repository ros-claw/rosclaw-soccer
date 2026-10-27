import json
from types import SimpleNamespace

import pytest

from scripts import rsi_watch_team_retention as watch


def setup(monkeypatch, tmp_path, state, *, complete):
    source = tmp_path / "source"
    source.mkdir()
    if complete:
        (source / "complete.json").write_text(
            json.dumps(
                dict(
                    status="TEAM_RECEIVING_COVERAGE_LEARNED_FULL_RETENTION_NOT_PROMOTED",
                    source_head="abc",
                    physical_runs=220,
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
            "--retention-proof",
            str(tmp_path / "retention"),
            "--models",
            str(tmp_path / "models"),
            "--output",
            str(tmp_path / "output"),
            "--unit",
            "rosclaw-team-test.service",
            "--source-head",
            "abc",
            "--retention-hash",
            "retention",
            "--models-hash",
            "models",
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


def test_watcher_only_audits_successful_complete_job(monkeypatch, tmp_path):
    calls = setup(
        monkeypatch,
        tmp_path,
        "ActiveState=inactive\nResult=success\nExecMainStatus=0\n",
        complete=True,
    )
    watch.main()
    assert len(calls) == 2
    assert calls[1][1] == "scripts/rsi_audit_team_coverage_retention.py"


def test_watcher_does_not_retry_failed_job(monkeypatch, tmp_path):
    calls = setup(
        monkeypatch,
        tmp_path,
        "ActiveState=failed\nResult=exit-code\nExecMainStatus=1\n",
        complete=False,
    )
    with pytest.raises(ValueError, match="without completion"):
        watch.main()
    assert len(calls) == 1
