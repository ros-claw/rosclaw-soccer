"""No physical trials: admission and observer isolation only."""

import sys

import numpy as np
import pytest

from rosclaw_soccer.rsi.full_entry_forecast import _record_step
from scripts import rsi_probe_measured_context_native as runner


def arguments(tmp_path):
    result = ["runner"]
    for name in (
        "audit",
        "audit-review",
        "world",
        "motion-csv",
        "model-root",
        "core-source",
        "output",
    ):
        result += ["--" + name, str(tmp_path / name)]
    result += ["--expected-head", "expected", "--expected-audit-review-hash", "sha256:" + "1" * 64]
    for i in range(4):
        result += ["--previous-producer", str(tmp_path / f"p{i}")]
    return result


@pytest.mark.parametrize("kind", ["source", "output", "missing_cluster"])
def test_sealed_complete_unused_before_imports(tmp_path, monkeypatch, kind):
    args = arguments(tmp_path)
    if kind == "missing_cluster":
        args = args[:-2]
    if kind == "output":
        (tmp_path / "output").mkdir()
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(
        runner.subprocess,
        "check_output",
        lambda *a, **k: "wrong\n" if kind == "source" else "expected\n",
    )
    with pytest.raises(ValueError, match="sealed source"):
        runner.main()


@pytest.mark.parametrize(
    "flags",
    [
        [],
        ["--cluster", "0"],
        ["--preflight-only", "--cluster", "0"],
        ["--preflight-only", "--bank-preflight", "unused"],
    ],
)
def test_master_worker_stage_mixing_or_incomplete_worker_refused(tmp_path, monkeypatch, flags):
    monkeypatch.setattr(sys, "argv", arguments(tmp_path) + flags)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **k: "expected\n")
    with pytest.raises(ValueError, match="zero-actor stage|worker requires"):
        runner.main()
    assert not (tmp_path / "output").exists()


def test_independent_nn_proof_commitment_before_allocation(tmp_path, monkeypatch):
    (tmp_path / "audit-review").mkdir()
    (tmp_path / "audit-review" / "complete.json").write_text("{}")
    monkeypatch.setattr(sys, "argv", arguments(tmp_path) + ["--preflight-only"])
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **k: "expected\n")
    with pytest.raises(ValueError, match="audit proof identity differs"):
        runner.main()
    assert not (tmp_path / "output").exists()


def test_observer_can_mutate_its_copy_but_not_execution_row():
    row = {"actual": np.array([[1.0, 2.0]], dtype=np.float64)}

    def observer(owned):
        assert not owned["actual"].flags.writeable
        assert not np.shares_memory(owned["actual"], row["actual"])
        owned["actual"].setflags(write=True)
        owned["actual"][:] = 99
        owned.clear()

    _record_step(observer, row)
    np.testing.assert_array_equal(row["actual"], [[1.0, 2.0]])
    _record_step(None, row)


def test_observer_fault_is_not_silently_ignored():
    def broken(row):
        raise ValueError("fixture recorder failure")

    with pytest.raises(ValueError, match="recorder failure"):
        _record_step(broken, {"q": np.zeros((1, 43))})
