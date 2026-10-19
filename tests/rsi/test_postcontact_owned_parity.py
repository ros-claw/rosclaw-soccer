"""Postcontact parity admission only, not proof of faster G1 execution."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import rsi_postcontact_owned_parity_inputs as mod
from tests.rsi.test_owned_selected_parity import fixture as old_fixture
from tests.rsi.test_owned_selected_parity import scope as old_scope


def scope():
    return dict(
        old_scope(),
        profile="six-postcontact-owned-selected-parity-v1",
        frame_selection="CLOSED_POSTCONTACT_SELECTED_ACTION_COMPARISON_ONLY",
        selected_source_indices=list(range(12, 18)),
        new_task_gain_claimed=False,
    )


def test_scope():
    mod.require_scope(scope())


@pytest.mark.parametrize("key", list(scope()))
def test_changed_or_missing_scope_fails(key):
    cfg = scope()
    del cfg[key]
    with pytest.raises(ValueError):
        mod.require_scope(cfg)
    cfg = scope()
    cfg[key] = "changed"
    with pytest.raises(ValueError):
        mod.require_scope(cfg)


def fixture():
    case, _, alignment, decision, gold, actual = old_fixture()
    f = decision["frame"]
    case["frame"] = f - 1
    actual["force_n"] = np.zeros((300, 1, 6))
    case["trace"]["force_n"] = np.zeros((300, 1, 6))
    actual["force_n"][f - 1, 0, 0] = 2.0
    case["trace"]["force_n"][f - 1, 0, 0] = 2.0
    actual["postcontact_exit_requested_offset_nm"] = np.zeros((300, 1, 29))
    actual["postcontact_exit_applied_offset_nm"] = np.zeros((300, 1, 10, 29))
    actual["postcontact_exit_forecast_steps"] = np.zeros((300, 1, 1), dtype=np.int64)
    actual["postcontact_exit_requested_offset_nm"][f : f + 3, 0] = gold["offset_nm"]
    row = dict(
        index=12,
        source_index=0,
        partition="CONSUMED_ORIGINAL_RETENTION",
        source_receipt_hash="receipt",
        report_hash="case12",
    )
    checked = dict(
        index=12,
        source_case_hash="case12",
        selected_execution_alignment=alignment["selected_window_alignment"],
    )
    decision["schema"] = "soccer.rsi.preaction_postcontact80ms_exit_decision.v1"
    return case, row, checked, decision, gold, actual


def test_binds_fixed_original_posttouch_window_without_trace_namespace_alias():
    args = fixture()
    names = set(args[-1])
    mod.bind_postcontact_case(*args, source_index=0)
    assert args[0]["frame"] == 145 and set(args[-1]) == names
    assert not any(k.startswith("early_impact_feedback_") for k in args[-1])


@pytest.mark.parametrize(
    "bad",
    [
        "index",
        "source",
        "partition",
        "receipt",
        "hash",
        "no_past_touch",
        "early_namespace",
        "cancel",
    ],
)
def test_mismatched_or_early_source_rejected(bad):
    case, row, checked, decision, gold, actual = fixture()
    if bad == "index":
        row["index"] = 0
    elif bad == "source":
        row["source_index"] = 1
    elif bad == "partition":
        row["partition"] = "CONSUMED_LOCAL_DEVELOPMENT"
    elif bad == "receipt":
        row["source_receipt_hash"] = "other"
    elif bad == "hash":
        checked["source_case_hash"] = "other"
    elif bad == "no_past_touch":
        actual["force_n"][:] = 0
    elif bad == "early_namespace":
        actual["early_impact_feedback_requested_offset_nm"] = actual.pop(
            "postcontact_exit_requested_offset_nm"
        )
    else:
        checked["selected_execution_alignment"][0]["actual_safety_cancel_frame"] = 146
    with pytest.raises(ValueError):
        mod.bind_postcontact_case(case, row, checked, decision, gold, actual, source_index=0)


def test_new_cli_preflight_dispatch_does_not_allocate(monkeypatch, capsys):
    from scripts import rsi_calibrate_feedback_branches as runner

    calls = []
    monkeypatch.setattr(
        mod,
        "prepare",
        lambda path: (
            calls.append(path)
            or SimpleNamespace(source=Path(runner.__file__).resolve().parent.parent)
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "calibrate",
            "--config",
            "fixture.json",
            "--postcontact-owned-selected-parity",
            "--preflight",
        ],
    )
    runner.main()
    assert calls == [Path("fixture.json")]
    assert (
        "SIX_POSTCONTACT_SELECTED_OWNED_PARITY_READY_NO_FULL_MODELS_OR_PHYSICS"
        in capsys.readouterr().out
    )
