"""Preimport source and stage checks; no model or physics execution."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import rsi_probe_response_cem as runner
from scripts import rsi_response_cem_admission as admission


def arguments(tmp_path):
    result = ["response-cem"]
    for key in (
        "bank",
        "world",
        "motion-csv",
        "model-root",
        "core-source",
        "output",
        "previous-producer",
    ):
        result += ["--" + key, str(tmp_path / key)]
    for key in ("expected-head", "expected-bank-hash", "expected-rejected-head"):
        result += ["--" + key, "expected"]
    return result + ["--rejected-producer", "unit", str(tmp_path / "rejected"), "sha256"]


def test_wrong_source_before_any_project_or_physics(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", arguments(tmp_path))
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **k: "wrong\n")
    with pytest.raises(ValueError, match="sealed response CEM source"):
        runner.main()


def test_dirty_bootstrap_before_admission_helper(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", arguments(tmp_path))
    monkeypatch.setattr(
        runner.subprocess,
        "check_output",
        lambda cmd, **k: "expected\n" if cmd[1] == "rev-parse" else b"dirty",
    )
    with pytest.raises(ValueError, match="bootstrap differs"):
        runner.main()


@pytest.mark.parametrize("extra", [[], ["--preflight-only", "--cluster", "0"]])
def test_no_worker_without_same_source_full_bank_proof(tmp_path, monkeypatch, extra):
    monkeypatch.setattr(sys, "argv", arguments(tmp_path) + extra)
    monkeypatch.setattr(
        runner.subprocess,
        "check_output",
        lambda cmd, **k: (
            "expected\n" if cmd[1] == "rev-parse" else Path(cmd[-1].split(":", 1)[1]).read_bytes()
        ),
    )
    with pytest.raises(ValueError, match="cannot allocate|requires same-source"):
        runner.main()


def test_existing_output_never_overwritten(tmp_path):
    output = tmp_path / "old"
    output.mkdir()
    with pytest.raises(ValueError, match="unused output"):
        admission.admit(SimpleNamespace(output=output))


def test_untracked_source_rejected_before_old_bank_or_project_imports(tmp_path, monkeypatch):
    monkeypatch.setattr(admission.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(
        admission.subprocess,
        "check_output",
        lambda cmd, **k: "expected\n" if cmd[1] == "rev-parse" else "scripts/unsealed.py\n",
    )
    with pytest.raises(ValueError, match="unsealed Python"):
        admission.admit(SimpleNamespace(output=tmp_path / "unused", expected_head="expected"))
