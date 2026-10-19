"""Screen admission cannot substitute40ms proof or omit failed original cases."""

import copy

import pytest

from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_early_feedback_screen_inputs import require_scope, validate_readback
from tests.rsi.test_early_feedback_calibration import scope as zero_scope


def scope():
    cfg = zero_scope()
    cfg.pop("branch_encoder_decoder_pairs")
    cfg.pop("candidate_actions_optimized")
    cfg.update(
        profile="six-early-feedback-impact-screen-v1",
        maximum_candidate_actions_per_case=13,
        maximum_nonzero_candidate_actions=72,
        repeat_zero_after_candidates=True,
        feedback_forecast_substep_budget=3360,
        combined_substep_budget=21360,
        maximum_branch_encoder_decoder_pairs=252,
    )
    return cfg


def test_fixed_screen_bounds():
    require_scope(scope())


@pytest.mark.parametrize("key", list(scope()))
def test_no_missing_or_widened_scope(key):
    cfg = scope()
    del cfg[key]
    with pytest.raises(ValueError):
        require_scope(cfg)
    cfg = scope()
    v = cfg[key]
    cfg[key] = int(v) if type(v) is bool else float(v) if type(v) is int else "REAL"
    with pytest.raises(ValueError):
        require_scope(cfg)


def proof():
    cases = [
        dict(job={"case": i}, receipt=dict(receipt_hash=f"source-{i}", outcome={"pass": i >= 2}))
        for i in range(6)
    ]
    d = dict(
        schema="soccer.rsi.zero_early_feedback_calibration_readback.v1",
        all_six_no_intervention_feedback_branches_qualified=True,
        new_physical_steps=0,
        new_optimizer_updates=0,
        nonzero_interventions_qualified=False,
        full_h300_teacher_qualified=False,
        teacher_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
        rows=[
            dict(
                index=i,
                job=c["job"],
                source_receipt_hash=c["receipt"]["receipt_hash"],
                original_outcome=c["receipt"]["outcome"],
                all40_substeps_match_original_dense=True,
                all_three_neural_and_target_updates_verified=True,
                all_neural_next_capture_bits_match=True,
                next_target_decomposition_matches=True,
                repeated_whole_branch_bytes_equal=True,
            )
            for i, c in enumerate(cases)
        ],
    )
    d["report_hash"] = hash_json(d)
    return d, dict(early_zero_readback_hash=d["report_hash"]), cases


def reseal(d, cfg):
    d["report_hash"] = hash_json({k: v for k, v in d.items() if k != "report_hash"})
    cfg["early_zero_readback_hash"] = d["report_hash"]


def test_all_six_original_cases_and_three_neural_updates_required():
    validate_readback(*proof())


@pytest.mark.parametrize("bad", ["40ms", "missing_failure", "reorder", "outcome", "hash"])
def test_cannot_replace_missing_proof_by_summary_or_easy_subset(bad):
    d, cfg, cases = proof()
    if bad == "40ms":
        d["schema"] = "soccer.rsi.zero_feedback_calibration_readback.v1"
    elif bad == "missing_failure":
        d["rows"] = d["rows"][1:]
    elif bad == "reorder":
        d["rows"] = list(reversed(d["rows"]))
    elif bad == "outcome":
        d["rows"][0]["original_outcome"] = {"pass": True}
    else:
        cfg["early_zero_readback_hash"] = "wrong"
    if bad != "hash":
        reseal(d, cfg)
    with pytest.raises(ValueError):
        validate_readback(d, cfg, cases)


@pytest.mark.parametrize(
    "key",
    [
        "all40_substeps_match_original_dense",
        "all_three_neural_and_target_updates_verified",
        "all_neural_next_capture_bits_match",
        "next_target_decomposition_matches",
        "repeated_whole_branch_bytes_equal",
    ],
)
def test_each_intermediate_physical_and_nn_proof_required_for_all_cases(key):
    d, cfg, cases = proof()
    d["rows"][5].pop(key)
    reseal(d, cfg)
    with pytest.raises(ValueError):
        validate_readback(d, cfg, cases)


@pytest.mark.parametrize(
    "key",
    [
        "nonzero_interventions_qualified",
        "full_h300_teacher_qualified",
        "teacher_qualified",
        "hardware_authorized",
        "promotion_authorized",
    ],
)
def test_zero_proof_must_not_claim_unproved_authority(key):
    d, cfg, cases = proof()
    tampered = copy.deepcopy(d)
    tampered[key] = True
    reseal(tampered, cfg)
    with pytest.raises(ValueError):
        validate_readback(tampered, cfg, cases)
