"""Declared full pilot bounds and causal action interval, no physical task claims."""

import copy

import numpy as np
import pytest

from scripts.rsi_early_impact_h300 import intervention_frames
from scripts.rsi_early_impact_pilot_inputs import require_scope, validate_screen


def scope():
    return dict(
        mode="SIM_ONLY",
        profile="six-prospective-early-impact-h300-diagnostic-v1",
        source_cases=6,
        new_causal_rollouts=6,
        reused_original_controls=6,
        native_step_budget=18000,
        forecast_step_budget=10800,
        combined_step_budget=28800,
        official_horizon_frames=300,
        post_contact_frames=60,
        new_optimizer_updates=0,
        hardware_authorized=False,
        promotion_authorized=False,
        teacher_qualified=False,
        training_authorized=False,
        private_fresh_accessed=False,
        actor_future_event_input_allowed=False,
        independent_qualification_claimed=False,
        new_neural_policy_trained=False,
    )


def test_scope():
    require_scope(scope())


@pytest.mark.parametrize("key", list(scope()))
def test_no_missing_or_widened_scope(key):
    cfg = scope()
    del cfg[key]
    with pytest.raises(ValueError):
        require_scope(cfg)
    cfg = scope()
    value = cfg[key]
    cfg[key] = int(value) if type(value) is bool else float(value) if type(value) is int else "REAL"
    with pytest.raises(ValueError):
        require_scope(cfg)


def screen():
    cases = [
        dict(job={"case": i}, receipt=dict(receipt_hash=f"s{i}", outcome={"task_pass": i >= 2}))
        for i in range(6)
    ]
    d = dict(
        schema="soccer.rsi.early_feedback_impact_screen_readback.v1",
        all_original_six_including_failures_retained=True,
        candidate_jacobian_bank_independently_regenerated=True,
        score_formula_shared_with_producer=True,
        candidate_geometry_and_bank_formula_shared_with_producer=True,
        raw_branch_count=84,
        nonzero_candidates=72,
        local_proxy_improved_source_cases=1,
        new_physical_steps=0,
        new_optimizer_updates=0,
        nonzero_branch_independent_physics_replayed=False,
        prospective_causal_trigger_qualified=False,
        full_m0_gain_proven=False,
        teacher_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
        rows=[
            dict(
                index=i,
                job=c["job"],
                source_receipt_hash=c["receipt"]["receipt_hash"],
                original_outcome=c["receipt"]["outcome"],
                all_raw_branch_scores_offsets_and_feedback_targets_checked=True,
                zero_branches_match_original_dense_and_neural_bits=True,
                local_proxy_improved=i == 0,
            )
            for i, c in enumerate(cases)
        ],
    )
    return d, cases


def test_full_screen_readback_not_teacher_claim_is_accepted_for_diagnostic_only():
    validate_screen(*screen())


@pytest.mark.parametrize(
    "bad", ["no_failed_gain", "subset", "40ms", "permission", "count", "index"]
)
def test_no_subset_no_blind_pilot_no_qualification_upgrade(bad):
    d, cases = copy.deepcopy(screen())
    if bad == "no_failed_gain":
        d["rows"][0]["local_proxy_improved"] = False
        d["rows"][2]["local_proxy_improved"] = True
    elif bad == "subset":
        d["rows"].pop(0)
    elif bad == "40ms":
        d["schema"] = "soccer.rsi.feedback_impact_screen_readback.v1"
    elif bad == "permission":
        d["teacher_qualified"] = True
    elif bad == "index":
        d["rows"][0]["index"] = False
    else:
        d["raw_branch_count"] = 83
    with pytest.raises(ValueError):
        validate_screen(d, cases)


def trace():
    return dict(
        early_impact_feedback_requested_offset_nm=np.zeros((300, 1, 29)),
        early_impact_feedback_applied_offset_nm=np.zeros((300, 1, 10, 29)),
    )


def test_requested_action_and_first_physical_effect_are_distinct_when_clipped():
    a = trace()
    assert intervention_frames(a) == (None, None)
    a["early_impact_feedback_requested_offset_nm"][30:33, 0, 0] = 1
    a["early_impact_feedback_applied_offset_nm"][31:33, 0, :, 0] = 1
    assert intervention_frames(a) == (30, 31)


@pytest.mark.parametrize("bad", ["long", "gap", "outside", "nonfinite", "short"])
def test_action_schedule_is_strictly_bounded_and_whole(bad):
    a = trace()
    if bad == "long":
        a["early_impact_feedback_requested_offset_nm"][30:34] = 1
    elif bad == "gap":
        a["early_impact_feedback_requested_offset_nm"][[30, 32]] = 1
    elif bad == "outside":
        a["early_impact_feedback_applied_offset_nm"][30] = 1
    elif bad == "nonfinite":
        a["early_impact_feedback_requested_offset_nm"][0, 0, 0] = np.nan
    else:
        a["early_impact_feedback_requested_offset_nm"] = a[
            "early_impact_feedback_requested_offset_nm"
        ][:299]
    with pytest.raises(ValueError):
        intervention_frames(a)
