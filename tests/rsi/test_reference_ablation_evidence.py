"""Synthetic provenance mutations; no trained or physically qualified policy."""

import copy

import numpy as np
import pytest
from rosclaw.growth.causal_residual_memory import initial_parameters

from rosclaw_soccer.rsi.pre_parent_reference_features import FEATURE_NAMES
from rosclaw_soccer.rsi.reference_ablation_evidence import require_matched_fit
from rosclaw_soccer.rsi.reference_student_execution import make_model
from rosclaw_soccer.sim.contracts import hash_json


def evidence():
    initial = initial_parameters(135, 12, hidden_dimension=64, seed=2026101902)
    ignored = [
        i
        for i, n in enumerate(FEATURE_NAMES)
        if n.startswith("previous_applied_reference_increment:")
    ]
    kw = dict(
        mean=np.zeros(135),
        scale=np.ones(135),
        world_hash="sha256:" + "a" * 64,
        parent_model_hash="sha256:" + "b" * 64,
    )
    original_model = make_model(initial, **kw)
    w = np.asarray(initial["weight_ih"])
    w[:, ignored] = 0
    initial["weight_ih"] = w.tolist()
    initial_hash = hash_json(initial)
    initial["head_bias"][0] = 0.1  # Synthetic nonzero head, not an actual fit.
    model = make_model(initial, **kw)
    lr = dict(
        config=dict(seed=2026101902, steps=128),
        parameters=model["parameters"],
        fitted_parameters_hash=hash_json(model["parameters"]),
        completed_optimizer_steps=128,
    )
    shared = dict(
        normalization_hash="norm",
        whole_observations_hash="x",
        whole_reference_labels_hash="y",
        train_source_seeds=[11, 12],
        validation_source_seeds=[13],
        input_episode_count=16,
        optimized_episode_count=3,
        original_ineligible_episode_count=12,
    )
    oldfit = dict(shared, numeric_model=original_model, learning_receipt=dict(config=lr["config"]))
    fit = dict(
        shared,
        report_hash="newfit",
        numeric_model=model,
        learning_receipt=lr,
        past_reference_input_ablation=True,
        ignored_input_weights_remain_exactly_zero=True,
        previous_reference_inputs_are_teacher_forced=False,
        ignored_input_feature_names=[FEATURE_NAMES[i] for i in ignored],
        initial_parameters_hash=initial_hash,
        validation_excluded_from_gradient_and_normalization=True,
    )
    oldstage = dict(
        report_hash="original",
        source_receipt_hashes=[str(i) for i in range(16)],
        eligibility=[False, True] * 8,
    )
    configuration = dict(ignore_past_reference=True, fit_seed=2026101902, fit_steps=128)
    rows = [
        dict(
            numeric_model_hash=model["model_hash"],
            student_body_rollout=False,
            errors={
                k: dict(predictions_hash="exact", mean_squared_reference_error_rad2=0.001)
                for k in (
                    "teacher_forced_raw",
                    "self_fed_raw",
                    "teacher_forced_projected",
                    "self_fed_projected",
                )
            },
        )
        for _ in range(16)
    ]
    stage = dict(
        schema="soccer.rsi.reference_feedback_ablation_fit_complete.v1",
        status="MATCHED_ABLATION_FIT_COMPLETE_REQUIRES_OWN_BODY_EXAM",
        configuration=configuration,
        previous_fit_complete_hash="original",
        numeric_fit_hash="newfit",
        numeric_model_hash=model["model_hash"],
        source_receipt_hashes=oldstage["source_receipt_hashes"],
        eligibility=oldstage["eligibility"],
        all_original_failures_retained=True,
        optimizer_updates=128,
        same_recorded_body_feedback_diagnostics=rows,
        **{
            k: False
            for k in (
                "own_body_student_evaluated",
                "hardware_authorized",
                "promotion_authorized",
                "dataset_admission_authorized",
                "teacher_qualified",
                "online_rl_claimed",
            )
        },
    )
    return (
        stage,
        fit,
        dict(
            configuration=copy.deepcopy(configuration), original_stage=oldstage, original_fit=oldfit
        ),
    )


@pytest.mark.parametrize(
    "tamper",
    [
        None,
        "oldhash",
        "split",
        "input",
        "normalization",
        "updates",
        "config",
        "initial",
        "ignored",
        "teacherforced",
        "gradient",
        "feedback",
        "partial",
        "ownbody",
        "grant",
    ],
)
def test_exact_ablation_lineage_or_fail_closed(tamper):
    stage, fit, kw = evidence()
    if tamper == "oldhash":
        stage["previous_fit_complete_hash"] = "other"
    elif tamper == "split":
        fit["train_source_seeds"] = [13]
    elif tamper == "input":
        fit["whole_observations_hash"] = "newstates"
    elif tamper == "normalization":
        fit["normalization_hash"] = "other"
    elif tamper == "updates":
        stage["optimizer_updates"] = True
    elif tamper == "config":
        stage["configuration"]["fit_steps"] = 129
    elif tamper == "initial":
        fit["initial_parameters_hash"] = "original-unmasked-init"
    elif tamper == "ignored":
        fit["ignored_input_feature_names"].pop()
    elif tamper == "teacherforced":
        fit["previous_reference_inputs_are_teacher_forced"] = True
    elif tamper == "gradient":
        fit["learning_receipt"]["parameters"] = {}
    elif tamper == "feedback":
        stage["same_recorded_body_feedback_diagnostics"][0]["errors"]["self_fed_raw"][
            "predictions_hash"
        ] = "other"
    elif tamper == "partial":
        stage["same_recorded_body_feedback_diagnostics"].pop()
    elif tamper == "ownbody":
        stage["own_body_student_evaluated"] = True
    elif tamper == "grant":
        stage["promotion_authorized"] = True
    if tamper:
        with pytest.raises(ValueError):
            require_matched_fit(stage, fit, **kw)
    else:
        result = require_matched_fit(stage, fit, **kw)
        assert result == fit["numeric_model"] and not result["physics_qualified"]
