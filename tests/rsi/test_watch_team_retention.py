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


def test_watcher_audits_neighbor_without_retention_proof(monkeypatch, tmp_path):
    calls = setup(
        monkeypatch,
        tmp_path,
        "ActiveState=inactive\nResult=success\nExecMainStatus=0\n",
        complete=True,
    )
    path = tmp_path / "source" / "complete.json"
    path.write_text(
        json.dumps(
            dict(
                status="TEAM_RECEIVING_NEIGHBOR_DEVELOPMENT_NOT_PROMOTED",
                source_head="abc",
                physical_runs=48,
                promotion_authorized=False,
            )
        )
    )
    args = watch.sys.argv
    for option in ("--retention-proof", "--retention-hash"):
        i = args.index(option)
        del args[i : i + 2]
    args.append("--neighbor")
    watch.main()
    assert len(calls) == 2
    assert calls[1][1] == "scripts/rsi_audit_team_coverage_neighbor.py"


def test_neighbor_rejection_is_intentionally_skipped(monkeypatch, tmp_path):
    calls = setup(
        monkeypatch,
        tmp_path,
        "ActiveState=inactive\nResult=success\nExecMainStatus=0\n",
        complete=False,
    )
    proof = tmp_path / "admission.json"
    proof.write_text(
        json.dumps(
            dict(
                status="TEAM_COVERAGE_RETENTION_ASSESSED_NOT_PROMOTED",
                result=dict(development_generalization_admitted=False),
            )
        )
    )
    args = watch.sys.argv
    for option in ("--retention-proof", "--retention-hash"):
        i = args.index(option)
        del args[i : i + 2]
    args.extend(["--neighbor", "--admission-proof", str(proof)])
    watch.main()
    assert len(calls) == 1


def test_watcher_audits_failure_derived_guard(monkeypatch, tmp_path):
    calls = setup(
        monkeypatch,
        tmp_path,
        "ActiveState=inactive\nResult=success\nExecMainStatus=0\n",
        complete=True,
    )
    path = tmp_path / "source" / "complete.json"
    path.write_text(
        json.dumps(
            dict(
                status="TEAM_RECEIVING_NEIGHBOR_RADIUS_GUARD_NOT_PROMOTED",
                source_head="abc",
                physical_runs=48,
                promotion_authorized=False,
            )
        )
    )
    args = watch.sys.argv
    for option in ("--retention-proof", "--retention-hash"):
        i = args.index(option)
        del args[i : i + 2]
    args.extend(["--neighbor", "--radius-guard"])
    watch.main()
    assert len(calls) == 2
    assert calls[1][1] == "scripts/rsi_audit_team_coverage_neighbor.py"


def test_watcher_audits_guarded_full_retention(monkeypatch, tmp_path):
    calls = setup(
        monkeypatch,
        tmp_path,
        "ActiveState=inactive\nResult=success\nExecMainStatus=0\n",
        complete=True,
    )
    path = tmp_path / "source" / "complete.json"
    path.write_text(
        json.dumps(
            dict(
                status="TEAM_RECEIVING_COVERAGE_RADIUS_FULL_RETENTION_NOT_PROMOTED",
                source_head="abc",
                physical_runs=220,
                promotion_authorized=False,
            )
        )
    )
    watch.sys.argv.append("--radius-guard")
    watch.main()
    assert calls[1][1] == "scripts/rsi_audit_team_coverage_retention.py"
