"""Selected nonzero diagnostic admission and post-forecast comparison only."""

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi.feedback_branch_control import BODY_SHAPES
from scripts.rsi_early_feedback_calibration import preserve_forecast
from scripts.rsi_owned_selected_parity_inputs import (
    bind_case,
    compare_arrays,
    preserve_and_compare_selected,
    require_scope,
)
from tests.rsi.test_early_feedback_calibration import raw
from tests.rsi.test_owned_feedback_parity_scope import scope as zero_scope


def scope():
    cfg = zero_scope()
    cfg.pop("zero_branches_per_case")
    cfg.pop("contact_frame_lookback")
    cfg.update(
        profile="six-selected-owned-continuation-parity-v1",
        frame_selection="CLOSED_H300_SELECTED_ACTION_COMPARISON_ONLY",
        selected_branches_per_case=2,
    )
    return cfg


def test_scope():
    require_scope(scope())


@pytest.mark.parametrize("key", list(scope()))
def test_missing_or_widened_scope_fails(key):
    cfg = scope()
    del cfg[key]
    with pytest.raises(ValueError):
        require_scope(cfg)
    cfg = scope()
    v = cfg[key]
    cfg[key] = int(v) if type(v) is bool else float(v) if type(v) is int else "REAL"
    with pytest.raises(ValueError):
        require_scope(cfg)


@pytest.mark.parametrize("key", ["zero_branches_per_case", "contact_frame_lookback"])
def test_cannot_mislabel_selected_action_as_zero_or_contact_minus_two(key):
    cfg = scope()
    cfg[key] = 2
    with pytest.raises(ValueError):
        require_scope(cfg)


def fixture():
    trace = {k: np.zeros((300, 1, *shape)) for k, shape in BODY_SHAPES.items()}
    for key, size in (
        ("canonical_qpos", 43),
        ("canonical_qvel", 41),
        ("joint_target_rad", 29),
        ("motor_delta_rad", 12),
        ("recurrent_residual_hidden_state", 4),
        ("foundation_navigation_joint_target_rad", 29),
        ("taskspace_joint_increment_rad", 29),
    ):
        trace[key] = np.zeros((300, 1, size))
    trace["foundation_neural_a"] = np.zeros((300, 1, 2), dtype=np.float32)
    actual = {k: a.copy() for k, a in trace.items()}
    offset = np.zeros(29)
    offset[2] = -12
    requested = np.zeros((40, 29))
    requested[:30] = offset
    case = dict(
        job={"role": "heading_on"}, receipt={"receipt_hash": "receipt"}, frame=148, trace=trace
    )
    row = dict(job=case["job"], source_receipt_hash="receipt")
    alignment = dict(
        job=case["job"],
        selected_window_alignment=[
            dict(
                full80ms_execution_alignment_verified=True,
                actual_safety_cancel_frame=None,
                forecast_start_frame=145,
            )
        ],
    )
    decision = dict(
        frame=145,
        persisted_before_current_native_substeps=True,
        future_source_frame_or_target_used=False,
        requested_offset_nm=offset.tolist(),
    )
    gold = dict(offset_nm=offset, requested_offset_nm=requested)
    return case, row, alignment, decision, gold, actual


def test_selected_start_is_not_assumed_to_be_original_contact_minus_two():
    args = fixture()
    bind_case(*args)
    assert args[0]["frame"] == 145
    assert args[0]["original_contact_frame"] == 148
    assert args[0]["selected_offset_nm"][2] == -12


@pytest.mark.parametrize(
    "key,index",
    [
        ("canonical_qvel", 144),
        ("canonical_qvel", 145),
        ("foundation_neural_a", 145),
        ("motor_delta_rad", 144),
    ],
)
def test_any_prior_or_current_before_step_memory_mismatch_rejected(key, index):
    args = fixture()
    args[-1][key][index].flat[0] = 0.01
    with pytest.raises(ValueError, match="exact arrays differ"):
        bind_case(*args)


def test_future_body_not_consumed_by_case_binding():
    args = fixture()
    args[-1]["canonical_qvel"][146:] = 99
    bind_case(*args)


@pytest.mark.parametrize("mutate", ["cancel", "frame", "source", "lease", "offset"])
def test_selected_lease_identity_and_full_window_required(mutate):
    args = fixture()
    if mutate == "cancel":
        args[2]["selected_window_alignment"][0]["actual_safety_cancel_frame"] = 146
    elif mutate == "frame":
        args[3]["frame"] = 146
    elif mutate == "source":
        args[1]["source_receipt_hash"] = "other"
    elif mutate == "lease":
        args[4]["requested_offset_nm"][30, 2] = 1
    else:
        args[3]["requested_offset_nm"][2] = -11
    with pytest.raises(ValueError):
        bind_case(*args)


def test_all_nonzero_forecast_fields_saved_then_compared(tmp_path):
    out, _, _ = raw()
    offset = np.zeros(29)
    offset[2] = -12
    out.arrays["requested_offset_nm"][:30] = offset
    goldrow = preserve_forecast(tmp_path, 9, 0, out)
    with np.load(goldrow["path"], allow_pickle=False) as a:
        arrays = dict(a)
    goldpath = tmp_path / "selected-gold.npz"
    with goldpath.open("xb") as stream:
        np.savez_compressed(stream, **arrays, offset_nm=offset)
    from scripts.rsi_stage_release_contact_replay import digest

    gold = dict(
        path=str(goldpath), sha256=digest(goldpath), donor_hash=out.donor_integration_state_hash
    )
    case = dict(selected_gold_witness=gold, selected_offset_nm=offset)
    pins = {str(goldpath): gold["sha256"]}
    row = preserve_and_compare_selected(tmp_path, 0, 0, copy.deepcopy(out), case, pins)
    assert row["all_selected_gold_arrays_bit_exact"] is True
    out.next_neural_records[1]["a"][0, 0] = -0.0
    with pytest.raises(ValueError, match="exact arrays differ"):
        preserve_and_compare_selected(tmp_path, 0, 1, out, case, pins)
    assert (tmp_path / "case-0-branch-1.npz").is_file()


@pytest.mark.parametrize("mutation", ["fields", "dtype", "shape", "nan", "signedzero"])
def test_comparison_checks_more_than_numeric_equality(mutation):
    a = dict(x=np.zeros(2))
    b = copy.deepcopy(a)
    if mutation == "fields":
        b["y"] = np.zeros(2)
    elif mutation == "dtype":
        b["x"] = b["x"].astype(np.float32)
    elif mutation == "shape":
        b["x"] = b["x"].reshape(1, 2)
    elif mutation == "nan":
        b["x"][0] = np.nan
    else:
        b["x"][0] = -0.0
    with pytest.raises(ValueError):
        compare_arrays(a, b)


def test_preflight_dispatch_never_allocates(monkeypatch, capsys):
    from scripts import rsi_calibrate_feedback_branches as runner
    from scripts import rsi_owned_selected_parity_inputs as inputs

    monkeypatch.setattr(
        inputs,
        "prepare",
        lambda path: SimpleNamespace(source=Path(runner.__file__).resolve().parent.parent),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["calibrate", "--config", "fixture.json", "--owned-selected-parity", "--preflight"],
    )
    runner.main()
    assert "SIX_OWNED_SELECTED_PARITY_READY_NO_FULL_MODELS_OR_PHYSICS" in capsys.readouterr().out
