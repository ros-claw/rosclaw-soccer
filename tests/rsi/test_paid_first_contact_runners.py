import sys

import pytest

from scripts import rsi_audit_paid_first_contact as audit
from scripts import rsi_search_paid_first_contact as search


@pytest.mark.parametrize("runner", [search, audit])
def test_dirty_first_contact_source_cannot_allocate_physics(tmp_path, monkeypatch, runner):
    args = ["run", "--output", str(tmp_path / "out"), "--expected-head", "frozen"]
    if runner is search:
        args += [
            "--qualification-root",
            str(tmp_path / "qualification"),
            "--qualification-hash",
            "hash",
            "--seed",
            "0",
        ]
        # Qualification hash is checked before source, but no physical allocation is allowed.
        monkeypatch.setattr("scripts.rsi_native_context_recovery_common.digest", lambda _: "hash")
    else:
        args += ["--root", str(tmp_path / "source"), "--producer-head", "producer"]
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **k: b"dirty")
    with pytest.raises(ValueError, match="frozen"):
        runner.main()
    assert not (tmp_path / "out").exists()
