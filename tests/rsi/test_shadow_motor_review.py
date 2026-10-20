import json
import sys

import pytest

from scripts import rsi_collect_shadow_motor as runner
from scripts.rsi_native_context_recovery_common import digest


def arguments(tmp_path, monkeypatch, *, live):
    bank = tmp_path / "bank"
    bank.mkdir()
    (bank / "complete.json").write_text("{}")
    (bank / "declaration.json").write_text(json.dumps(dict(input_file_hashes={}, rows=[])))
    args = [
        "review",
        "--mode",
        "verify",
        "--expected-head",
        "sealed",
        "--expected-bank-hash",
        digest(bank / "complete.json"),
        "--collection-unit",
        "unit",
        "--expected-collection-source",
        "producer",
    ]
    for key in ("bank", "world", "model-root", "motion-csv", "core-source", "output", "collection"):
        args += ["--" + key, str(tmp_path / key)]
    monkeypatch.setattr(sys, "argv", args)

    def command(cmd, **kwargs):
        if cmd[:2] == ["git", "rev-parse"]:
            return "sealed\n"
        if cmd[0] == "git":
            return b""
        return (
            f"MainPID={'42' if live else '0'}\nSubState={'running' if live else 'exited'}\n"
            "Result=success\nExecMainStatus=0\n"
        )

    monkeypatch.setattr(runner.subprocess, "check_output", command)


def test_live_collection_not_restarted_or_used_for_learning(tmp_path, monkeypatch):
    arguments(tmp_path, monkeypatch, live=True)
    with pytest.raises(ValueError, match="actually have closed"):
        runner.main()
    assert not (tmp_path / "output").exists()


def test_partial_collection_rejected_before_neural_or_physical_allocation(tmp_path, monkeypatch):
    arguments(tmp_path, monkeypatch, live=False)
    collection = tmp_path / "collection"
    collection.mkdir()
    (collection / "complete.json").write_text(
        json.dumps(dict(status="ASSISTED_DAGGER_COLLECTION_NOT_STUDENT_QUALIFICATION", results=[]))
    )
    (collection / "declaration.json").write_text("{}")
    with pytest.raises(ValueError, match="complete original fixed-beta"):
        runner.main()
    assert not (tmp_path / "output").exists()
