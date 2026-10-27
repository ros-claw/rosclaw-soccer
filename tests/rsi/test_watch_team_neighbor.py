import json
from types import SimpleNamespace

import pytest

from scripts import rsi_watch_team_neighbor as watch


def setup(
    monkeypatch,
    tmp_path,
    *,
    admitted,
    state="ActiveState=inactive\nResult=success\nExecMainStatus=0\n",
):
    proof = tmp_path / "assessment.json"
    proof.write_text(
        json.dumps(
            {
                "status": "TEAM_COVERAGE_RETENTION_ASSESSED_NOT_PROMOTED",
                "hardware_authorized": False,
                "result": {"development_generalization_admitted": admitted},
            }
        )
    )
    monkeypatch.setattr(watch, "clean", lambda _: None)
    monkeypatch.setattr(
        watch.sys,
        "argv",
        [
            "watch",
            "--assessment",
            str(proof),
            "--unit",
            "rosclaw-team-test.service",
            "--expected-head",
            "head",
            "--",
            "--neighbor-bank",
            "bank.json",
            "--admission-proof",
            str(proof),
        ],
    )
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(stdout=state, returncode=0)

    monkeypatch.setattr(watch.subprocess, "run", run)
    return calls


def test_neighbor_starts_only_after_admission(monkeypatch, tmp_path):
    calls = setup(monkeypatch, tmp_path, admitted=True)
    watch.main()
    assert len(calls) == 2
    assert calls[1][1] == "scripts/rsi_team_selector_smoke.py"
    assert "--admission-hash" in calls[1]


def test_neighbor_rejection_does_not_launch(monkeypatch, tmp_path):
    calls = setup(monkeypatch, tmp_path, admitted=False)
    watch.main()
    assert len(calls) == 1


def test_neighbor_blocks_failed_assessment(monkeypatch, tmp_path):
    calls = setup(
        monkeypatch,
        tmp_path,
        admitted=True,
        state="ActiveState=failed\nResult=exit-code\nExecMainStatus=1\n",
    )
    with pytest.raises(ValueError, match="assessment failed"):
        watch.main()
    assert len(calls) == 1
