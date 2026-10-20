"""Independent-process comparison contract with synthetic artifacts only."""

import json

import numpy as np
import pytest

from scripts.rsi_probe_student_intervention import compare_case


def fixture(tmp_path):
    a, b = tmp_path / "actual", tmp_path / "expected"
    a.mkdir()
    b.mkdir()
    row = dict(
        status="PREINTERVENTION_HISTORY_VETO",
        new_native_steps=300,
        causal_qualified=False,
        artifact_hash="different compressed file identities",
    )
    for path in (a, b):
        (path / "complete.json").write_text(json.dumps(row))
        for name in ("actual-student30.npz", "shadow-labels30.npz"):
            np.savez_compressed(path / name, value=np.zeros((30, 1), np.float32))
    return a, b, row


def test_artifact_container_hash_not_numerical_equivalence(tmp_path):
    a, b, row = fixture(tmp_path)
    row["artifact_hash"] = "new compressed container hash"
    compare_case(a, b, row)


def test_signed_zero_changes_cannot_hide_in_numerical_equality(tmp_path):
    a, b, row = fixture(tmp_path)
    values = np.zeros((30, 1), np.float32)
    values[20, 0] = -0.0
    np.savez_compressed(a / "actual-student30.npz", value=values)
    with pytest.raises(AssertionError, match="bytes differ"):
        compare_case(a, b, row)


def test_task_gate_or_charge_changes_cannot_hide_in_array_equivalence(tmp_path):
    a, b, row = fixture(tmp_path)
    row["new_native_steps"] = 27300
    with pytest.raises(ValueError, match="charges differ"):
        compare_case(a, b, row)
