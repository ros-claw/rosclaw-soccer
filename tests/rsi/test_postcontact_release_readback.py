"""Readback scope/corruption tests with synthetic data, not G1 evidence."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.rsi.foot_heading_parity import SHAPES
from scripts import rsi_replay_postcontact_authority as mod
from tests.rsi.test_postcontact_release_protocol import arrays, scope


def complete():
    d = dict(
        schema="soccer.rsi.postcontact_release_authority_declaration.v1",
        report_hash="declaration",
        config=dict(scope(), output_root="/unused"),
    )
    c = dict(
        schema="soccer.rsi.postcontact_release_authority_complete.v1",
        declaration_hash="declaration",
        new_native_steps=42000,
        executed_control_foundation_pairs=4200,
        forecast_steps=0,
        new_optimizer_updates=0,
        physical_contexts=1,
        teacher_qualified=False,
        training_authorized=False,
        hardware_authorized=False,
        promotion_authorized=False,
        both_original_zero_controls_matched=True,
        rows=[dict(index=i, action_name=n) for i, n in enumerate(d["config"]["action_names"])],
    )
    return c, d


def test_all_fourteen_single_context_required():
    c, d = complete()
    mod.check_complete(c, d, Path("/unused"))


@pytest.mark.parametrize(
    "bad", ["drop", "order", "index", "counter", "bool_counter", "teacher", "scope", "contexts"]
)
def test_missing_or_promoted_scope_rejected(bad):
    c, d = complete()
    if bad == "drop":
        c["rows"].pop()
    elif bad == "order":
        c["rows"][2]["action_name"] = "lateral_plus"
    elif bad == "index":
        c["rows"][-1]["index"] = 0
    elif bad == "counter":
        c["new_native_steps"] -= 1
    elif bad == "bool_counter":
        c["forecast_steps"] = False
    elif bad == "teacher":
        c["teacher_qualified"] = True
    elif bad == "contexts":
        c["physical_contexts"] = 14
    else:
        d["config"]["maximum_release_duration_frames"] = 4
    with pytest.raises(ValueError):
        mod.check_complete(c, d, Path("/unused"))


def row_trace(monkeypatch):
    trace, review = arrays("lateral_plus")
    # Complete original ABI, kept explicit as a synthetic contract fixture.
    for key, tail in SHAPES.items():
        if key not in trace:
            dtype = np.bool_ if key == "recurrent_sampling_performed" else np.float64
            if key.startswith("foundation_neural_"):
                dtype = np.float32
            trace[key] = np.zeros((300, 1, *tail), dtype=dtype)
    original = {k: v.copy() for k, v in trace.items() if not k.startswith("postcontact_release_")}
    original["torque_nm"][:] = 0
    original["actual_actuator_force_nm"][:] = 0
    outcome = dict(task_pass=True, safe=True)
    monkeypatch.setattr(
        mod,
        "label_first_touch",
        lambda trace: type("Label", (), {"to_dict": lambda self: outcome})(),
    )
    review.update(native_integrated_steps=3000, forecast_integrated_steps=0)
    row = dict(
        action_name="lateral_plus",
        original_policy_hash="policy",
        source_receipt_hash="receipt",
        original_prefix_and_baseline_neural_bits_matched=True,
        causal_review=review,
        outcome=outcome,
        new_optimizer_updates=0,
        teacher_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    row["control_review"] = mod.check_release_trace(trace, review, action_name="lateral_plus")
    return row, trace, original


def test_whole_raw_row_is_rederived(monkeypatch):
    row, trace, original = row_trace(monkeypatch)
    mod.check_row(row, trace, original, policy_hash="policy", receipt_hash="receipt")


@pytest.mark.parametrize(
    "bad", ["force", "offset", "pd", "outcome", "policy", "prefix", "hybrid", "source_field"]
)
def test_raw_row_mutations_rejected(monkeypatch, bad):
    row, trace, original = row_trace(monkeypatch)
    row = deepcopy(row)
    if bad == "force":
        trace["force_n"][:] = 0
    elif bad == "offset":
        trace["postcontact_release_requested_offset_nm"][200, 0, 0] = 0.1
    elif bad == "pd":
        trace["actual_actuator_force_nm"][250, 0, 0, 0] += 0.1
    elif bad == "outcome":
        row["outcome"] = dict(task_pass=False, safe=True)
    elif bad == "policy":
        row["original_policy_hash"] = "other"
    elif bad == "prefix":
        trace["canonical_qpos"][0, 0, 0] += 0.1
    elif bad == "source_field":
        trace.pop("foundation_neural_encoder_features")
    else:
        trace["early_impact_feedback_requested_offset_nm"] = np.zeros((300, 1, 29))
    with pytest.raises(ValueError):
        mod.check_row(row, trace, original, policy_hash="policy", receipt_hash="receipt")
