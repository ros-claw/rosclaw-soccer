"""Synthetic raw-bank guards; not full-case physics or a trained policy."""

from pathlib import Path

import numpy as np
import pytest

from scripts import rsi_read_postcontact_full_bank as mod
from tests.rsi.test_postcontact_full_bank import scope
from tests.rsi.test_postcontact_release_readback import row_trace


def complete():
    d = dict(
        schema="soccer.rsi.postcontact_full_h300_bank_declaration.v1",
        report_hash="decl",
        config=dict(scope(), output_root="/synthetic"),
        jobs=[{}] * 18,
    )
    c = dict(
        schema="soccer.rsi.postcontact_full_h300_bank_complete.v1",
        declaration_hash="decl",
        contexts=[dict(context_index=i, rows=[{}] * 14) for i in range(18)],
        comparison_contexts=18,
        comparison_rollouts=252,
        new_rollouts=238,
        reused_execution_count=14,
        new_native_steps=714000,
        executed_control_foundation_pairs=71400,
        forecast_steps=0,
        new_optimizer_updates=0,
        selection_uses_full_simulated_future_outcomes=True,
        runtime_future_outcome_input_authorized=False,
        teacher_qualified=False,
        training_authorized=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    return c, d


def test_whole252_scope_never_grants_training():
    c, d = complete()
    mod.check_complete(c, d, Path("/synthetic"))


@pytest.mark.parametrize(
    "bad",
    [
        "drop",
        "new",
        "reuse",
        "native",
        "nn",
        "bool_counter",
        "forecasts",
        "train",
        "runtime_future",
    ],
)
def test_scope_counter_or_authorization_mutations_rejected(bad):
    c, d = complete()
    if bad == "drop":
        c["contexts"].pop()
    elif bad == "new":
        c["new_rollouts"] = 252
    elif bad == "reuse":
        c["reused_execution_count"] = 0
    elif bad == "native":
        c["new_native_steps"] = 713999
    elif bad == "nn":
        c["executed_control_foundation_pairs"] = 75600
    elif bad == "bool_counter":
        c["forecast_steps"] = False
    elif bad == "forecasts":
        c["forecast_steps"] = 40
    elif bad == "train":
        c["training_authorized"] = True
    else:
        c["runtime_future_outcome_input_authorized"] = True
    with pytest.raises(ValueError):
        mod.check_complete(c, d, Path("/synthetic"))


def fixture(monkeypatch):
    old, t, baseline = row_trace(monkeypatch)
    row = dict(old, context_index=0, action_index=3, course=[2.675, 0.132, 0.319])
    row["causal_review"].update(
        native_attempted_steps=3000,
        forecast_attempted_steps=0,
        current_body_policy_recomputed=True,
        source_future_trace_or_targets_consumed=False,
        independently_implemented_neural_policy=False,
    )
    row["training_authorized"] = False
    row["normalized_action_norm_squared"] = float(
        np.max(np.sum((t["postcontact_release_requested_offset_nm"][:, 0] / mod.CAPS) ** 2, axis=1))
    )
    expected_outcome = dict(row["outcome"])
    monkeypatch.setattr(
        mod,
        "label_first_touch",
        lambda trace: type("Label", (), {"to_dict": lambda self: expected_outcome})(),
    )
    args = dict(
        job=dict(course=row["course"]),
        parent=dict(receipt_hash="receipt"),
        commitment=dict(motor_policy_hash="policy"),
        context_index=0,
        action_index=3,
        action_name="lateral_plus",
    )
    return row, t, baseline, args


def test_raw_whole_body_fields_actuator_attribution_and_prefix_checked(monkeypatch):
    r, t, b, kw = fixture(monkeypatch)
    assert mod.check_row(r, t, b, **kw) == r["control_review"]


@pytest.mark.parametrize(
    "bad",
    [
        "policy",
        "course",
        "source",
        "index",
        "action",
        "future",
        "counter",
        "norm",
        "prefix",
        "mixed",
        "missing",
        "actual_force",
        "outcome",
        "bool_index",
        "bool_counter",
    ],
)
def test_source_control_or_provenance_mismatch_rejected(monkeypatch, bad):
    r, t, b, kw = fixture(monkeypatch)
    if bad == "policy":
        r["original_policy_hash"] = "other"
    elif bad == "course":
        r["course"] = [2.6, 0.0, 0.3]
    elif bad == "source":
        r["source_receipt_hash"] = "other"
    elif bad == "index":
        r["context_index"] = 1
    elif bad == "action":
        r["action_name"] = "yaw_plus"
    elif bad == "future":
        r["causal_review"]["source_future_trace_or_targets_consumed"] = True
    elif bad == "counter":
        r["causal_review"]["native_attempted_steps"] = 2999
    elif bad == "norm":
        r["normalized_action_norm_squared"] += 0.1
    elif bad == "prefix":
        t["foundation_neural_latent_token"][0, 0, 0] = -0.0
    elif bad == "mixed":
        t["postcontact_exit_requested_offset_nm"] = np.zeros((300, 1, 29))
    elif bad == "missing":
        t.pop("canonical_qvel")
    elif bad == "outcome":
        r["outcome"] = dict(task_pass=False, safe=True)
    elif bad == "bool_index":
        r["context_index"] = False
    elif bad == "bool_counter":
        r["causal_review"]["forecast_integrated_steps"] = False
    else:
        t["actual_actuator_force_nm"][70, 0, 0, 0] = 0.1
    with pytest.raises(ValueError):
        mod.check_row(r, t, b, **kw)
