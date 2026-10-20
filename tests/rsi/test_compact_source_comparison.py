"""Numerical mapping fixtures only, not six-option native parity evidence."""

import copy
import json

import numpy as np
import pytest

from rosclaw_soccer.rsi.source_lead_protocol import ORIGINAL_NINE_ZERO_C_INDICES
from scripts.rsi_probe_compact_source import compare_old_six, compare_six_case


def fixture(tmp_path):
    actual, old = tmp_path / "actual", tmp_path / "old"
    actual.mkdir()
    old.mkdir()
    candidates = []
    options = [
        (law, off, "right") for law in (0, 3, 4) for off in ((0.0, 0.0, 0.0), (0.0, -0.025, -0.025))
    ]
    options += [(law, (0.0, 0.0, 0.0), "left") for law in (0, 3, 4)]
    for i, (law, offset, side) in enumerate(options):
        candidates.append(
            dict(
                index=i,
                law_index=law,
                offset_root_m=list(offset),
                contact_reference_foot=side,
                report=dict(fixture=[1, 2]),
                score=dict(qualified=i == 4, cost=0.1 if i == 4 else None),
                artifact=f"candidate{i}.npz",
                artifact_hash="fixture",
            )
        )
        np.savez_compressed(old / f"candidate{i}.npz", fixture=np.full((3, 1), float(i)))
    selected_score = dict(qualified=True, cost=0.1)
    oldrow = dict(
        status="STUDENT30_PRIVILEGED_RECOVERY_EXECUTED",
        candidates=candidates,
        selected=4,
        selected_score=selected_score,
        causal_qualified=True,
    )
    (old / "complete.json").write_text(json.dumps(oldrow))
    rows = []
    for i, j in enumerate(ORIGINAL_NINE_ZERO_C_INDICES):
        rows.append(dict(copy.deepcopy(candidates[j]), index=i, artifact=f"candidate{i}.npz"))
        np.savez_compressed(actual / f"candidate{i}.npz", fixture=np.full((3, 1), float(j)))
    row = dict(
        status="STUDENT30_PRIVILEGED_RECOVERY_EXECUTED",
        option_protocol="zero_c_six",
        source_window=[286, 767],
        selected=2,
        candidates=rows,
        new_native_steps=19200,
        forecast_native_steps=16200,
        executed_native_steps=3000,
        selected_score=selected_score,
        causal_qualified=True,
        selected_report=dict(fixture=(1, 2)),
    )
    for name in (
        "actual-student30.npz",
        "shadow-labels30.npz",
        "selected-actual270.npz",
        "selected-physical300.npz",
    ):
        for root in (actual, old):
            np.savez_compressed(root / name, fixture=np.zeros((3, 1), np.float32))
    (actual / "complete.json").write_text(json.dumps(row))
    return actual, old, row


def test_old_six_mapping_and_json_tuple_normalization(tmp_path):
    a, b, row = fixture(tmp_path)
    compare_old_six(a, b, row)
    compare_six_case(a, a, row)


def test_unselected_future_corruption_still_fails_entire_zero_parity(tmp_path):
    a, b, row = fixture(tmp_path)
    np.savez_compressed(a / "candidate5.npz", fixture=np.zeros((3, 1)))
    with pytest.raises(AssertionError, match="bytes differ"):
        compare_old_six(a, b, row)


def test_compact_window_cannot_be_mislabeled_old_source_qualification(tmp_path):
    a, b, row = fixture(tmp_path)
    row["source_window"] = [466, 767]
    with pytest.raises(ValueError, match="baseline/choice/count/report differs"):
        compare_old_six(a, b, row)


def test_six_repeat_does_not_ignore_last_candidate(tmp_path):
    a, _, row = fixture(tmp_path)
    data = np.zeros((3, 1))
    data[0, 0] = -0.0
    repeat = tmp_path / "repeat"
    repeat.mkdir()
    for f in a.iterdir():
        (repeat / f.name).write_bytes(f.read_bytes())
    np.savez_compressed(repeat / "candidate5.npz", fixture=data)
    with pytest.raises(AssertionError, match="bytes differ"):
        compare_six_case(repeat, a, row)
