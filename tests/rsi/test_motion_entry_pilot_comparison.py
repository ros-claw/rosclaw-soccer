"""Original RHS full300 parity mapping fixtures, not physical qualification."""

import json

import numpy as np
import pytest

from scripts.rsi_pilot_motion_entry import compare_episode, compare_original_right


def fixture(tmp_path):
    actual, old = tmp_path / "actual", tmp_path / "old"
    actual.mkdir()
    old.mkdir()
    report = dict(fixture=[1, 2])
    score = dict(qualified=True, cost=0.1)
    oldrow = dict(candidates=[dict(report=report, score=score) for _ in range(9)])
    (old / "complete.json").write_text(json.dumps(oldrow))
    np.savez_compressed(old / "own-prefix20.npz", fixture=np.zeros((20, 1), np.float32))
    np.savez_compressed(old / "candidate2.npz", fixture=np.ones((280, 1), np.float32))
    np.savez_compressed(
        actual / "native300.npz",
        fixture=np.concatenate((np.zeros((20, 1), np.float32), np.ones((280, 1), np.float32))),
    )
    np.savez_compressed(actual / "own-prefix20.npz", fixture=np.zeros((20, 1), np.float32))
    row = dict(
        reference_side="right",
        law_index=3,
        source_window=[286, 767],
        report=dict(fixture=(1, 2)),
        score=score,
        new_native_steps=3000,
    )
    (actual / "complete.json").write_text(json.dumps(row))
    return actual, old, row


def test_right_law3_maps_old_zero_c_candidate2_and_keeps_prefix20(tmp_path):
    a, b, row = fixture(tmp_path)
    compare_original_right(a, b, row)
    compare_episode(a, a, row)


def test_prefix_neural_signed_zero_change_rejected(tmp_path):
    a, b, row = fixture(tmp_path)
    x = np.concatenate((np.zeros((20, 1), np.float32), np.ones((280, 1), np.float32)))
    x[0, 0] = -0.0
    np.savez_compressed(a / "native300.npz", fixture=x)
    with pytest.raises(AssertionError, match="bytes differ"):
        compare_original_right(a, b, row)


def test_new_left_history_not_mislabeled_as_old_right_parity(tmp_path):
    a, b, row = fixture(tmp_path)
    row["reference_side"] = "left"
    with pytest.raises(ValueError, match="RHS neural/native"):
        compare_original_right(a, b, row)
