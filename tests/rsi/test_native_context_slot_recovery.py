"""Recovery admission and trace rejection fixtures; no simulation evidence."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import rsi_recover_native_context_slots as runner
from scripts.rsi_native_context_failure_trace import validate_failure_trace

REASON = "proposed measured-joint context outside native joint limits; no clipping"


def arguments(tmp_path):
    result = ["recover"]
    for name in (
        "inventory",
        "bank",
        "actor-source",
        "world",
        "motion-csv",
        "model-root",
        "core-source",
        "fault-review",
        "output",
        "previous-producer",
    ):
        result += ["--" + name, str(tmp_path / name)]
    for name in (
        "expected-head",
        "expected-actor-head",
        "expected-inventory-hash",
        "expected-fault-review-hash",
        "fault-review-unit",
    ):
        result += ["--" + name, "expected"]
    return result


def test_source_guard_before_imports(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", arguments(tmp_path))
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **k: "wrong\n")
    with pytest.raises(ValueError, match="sealed recovery source"):
        runner.main()


def test_bootstrap_bytes_before_physics(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", arguments(tmp_path))
    monkeypatch.setattr(
        runner.subprocess,
        "check_output",
        lambda cmd, **k: "expected\n" if cmd[1] == "rev-parse" else b"dirty",
    )
    with pytest.raises(ValueError, match="sealed Git blob"):
        runner.main()


@pytest.mark.parametrize("extra", [[], ["--preflight-only", "--cluster", "1"]])
def test_stage_mixing_rejected_before_inventory(tmp_path, monkeypatch, extra):
    from pathlib import Path

    monkeypatch.setattr(sys, "argv", arguments(tmp_path) + extra)
    monkeypatch.setattr(
        runner.subprocess,
        "check_output",
        lambda cmd, **k: (
            "expected\n" if cmd[1] == "rev-parse" else Path(cmd[-1].split(":", 1)[1]).read_bytes()
        ),
    )
    with pytest.raises(ValueError, match="explicit cluster|cannot allocate"):
        runner.main()


@pytest.mark.parametrize("change", ["partial", "model_calls", "persistence", "ticks_bool"])
def test_incomplete_tick_evidence_is_fatal(tmp_path, change):
    data = dict(
        completed_control_ticks=40,
        attempted_native_steps=400,
        integrated_native_steps=400,
        trace_persistence_error=None,
        attempted_source_planner_calls=2,
        returned_source_planner_calls=2,
    )
    key, value = {
        "partial": ("integrated_native_steps", 399),
        "model_calls": ("returned_source_planner_calls", 1),
        "persistence": ("trace_persistence_error", "disk error"),
        "ticks_bool": ("completed_control_ticks", True),
    }[change]
    data[key] = value
    (tmp_path / "failure.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="complete finite control ticks"):
        validate_failure_trace(tmp_path, None, REASON)


def test_unknown_reason_is_not_resumable(tmp_path):
    with pytest.raises(ValueError, match="unknown exception"):
        validate_failure_trace(tmp_path, None, "NaN physics")


def test_nonfinite_or_incomplete_arrays_are_fatal(tmp_path):
    (tmp_path / "failure.json").write_text(
        json.dumps(
            dict(
                completed_control_ticks=40,
                attempted_native_steps=400,
                integrated_native_steps=400,
                trace_persistence_error=None,
                attempted_source_planner_calls=2,
                returned_source_planner_calls=2,
            )
        )
    )
    for name in ("observed-control-ticks", "last-native-state", "planner-calls"):
        np.savez(tmp_path / (name + ".npz"), bad=np.array([np.nan]))
    with pytest.raises(ValueError, match="nonfinite or incomplete"):
        validate_failure_trace(tmp_path, SimpleNamespace(nq=43, nv=41, nu=29, nbody=46), REASON)
