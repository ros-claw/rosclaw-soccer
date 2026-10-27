import json
from types import SimpleNamespace

import pytest

from scripts import rsi_watch_relay_balance_expansion as watch


def test_failed_pilot_does_not_start_physics(monkeypatch, tmp_path):
    monkeypatch.setattr(watch, "clean", lambda _: None)
    argv = [
        "watch",
        "--frontier-root",
        str(tmp_path / "frontier"),
        "--frontier-head",
        "frontier",
        "--frontier-hash",
        "hash",
        "--pilot-review",
        str(tmp_path / "pilot.json"),
        "--pilot-unit",
        "rosclaw-team-review.service",
        "--pilot-review-head",
        "review",
        "--new-context-block",
        "0",
        "--output",
        str(tmp_path / "output"),
        "--expected-head",
        "source",
    ]
    for i in range(3):
        argv.extend(["--balance-teacher-pair", f"s{i}", f"a{i}", "sh", "ah"])
    monkeypatch.setattr(watch.sys, "argv", argv)
    calls = []

    def run(cmd, **_kwargs):
        calls.append(cmd)
        return SimpleNamespace(
            returncode=0, stdout="ActiveState=failed\nResult=exit-code\nExecMainStatus=1\n"
        )

    monkeypatch.setattr(watch.subprocess, "run", run)
    with pytest.raises(ValueError, match="no physics"):
        watch.main()
    assert len(calls) == 1
    assert not (tmp_path / "output").exists()


def test_verified_pilot_starts_collection_then_replay(monkeypatch, tmp_path):
    monkeypatch.setattr(watch, "clean", lambda _: None)
    pilot = tmp_path / "pilot.json"
    pilot.write_text(
        json.dumps(
            dict(
                status="RELAY_BALANCE_PAIRED_DEVELOPMENT_NOT_PROMOTED",
                source_head="review",
                paired_physical_and_private_handoff_prefix_exact=True,
                original_baselines_raw_byte_identical=True,
                records=[{}] * 72,
                summaries={"0": {"body_failure_rescues": 1}},
            )
        )
    )
    argv = [
        "watch",
        "--frontier-root",
        str(tmp_path / "frontier"),
        "--frontier-head",
        "frontier",
        "--frontier-hash",
        "hash",
        "--pilot-review",
        str(pilot),
        "--pilot-unit",
        "rosclaw-team-review.service",
        "--pilot-review-head",
        "review",
        "--new-context-block",
        "0",
        "--output",
        str(tmp_path / "output"),
        "--expected-head",
        "source",
    ]
    for i in range(3):
        argv.extend(["--balance-teacher-pair", f"s{i}", f"a{i}", "sh", "ah"])
    monkeypatch.setattr(watch.sys, "argv", argv)
    calls = []

    def run(cmd, **_kwargs):
        calls.append(cmd)
        if cmd[0] == "systemctl":
            return SimpleNamespace(
                returncode=0, stdout="ActiveState=inactive\nResult=success\nExecMainStatus=0\n"
            )
        if cmd[1] == "scripts/rsi_collect_neural_relay_chain.py":
            source = tmp_path / "output" / "source"
            source.mkdir()
            (source / "complete.json").write_text("{}")
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(watch.subprocess, "run", run)
    watch.main()
    assert [c[1] for c in calls[1:]] == [
        "scripts/rsi_collect_neural_relay_chain.py",
        "scripts/rsi_audit_neural_relay_chain.py",
    ]
