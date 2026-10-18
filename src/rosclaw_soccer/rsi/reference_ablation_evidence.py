"""Authenticate matched input-ablation provenance; never a promotion gate."""

import numpy as np
from rosclaw.growth.causal_residual_memory import initial_parameters

from rosclaw_soccer.rsi.pre_parent_reference_features import FEATURE_NAMES
from rosclaw_soccer.rsi.reference_student_execution import own_model
from rosclaw_soccer.sim.contracts import hash_json


def require_matched_fit(stage, fit, *, configuration, original_stage, original_fit):
    """Numeric receipt identities after caller authenticates the original files."""
    ignored = [
        i
        for i, n in enumerate(FEATURE_NAMES)
        if n.startswith("previous_applied_reference_increment:")
    ]
    initial = initial_parameters(135, 12, hidden_dimension=64, seed=2026101902)
    weight = np.asarray(initial["weight_ih"])
    weight[:, ignored] = 0
    initial["weight_ih"] = weight.tolist()
    model = own_model(fit["numeric_model"])
    original = own_model(original_fit["numeric_model"])
    lr = fit["learning_receipt"]
    if (
        stage.get("schema") != "soccer.rsi.reference_feedback_ablation_fit_complete.v1"
        or stage.get("status") != "MATCHED_ABLATION_FIT_COMPLETE_REQUIRES_OWN_BODY_EXAM"
        or stage.get("configuration") != configuration
        or configuration.get("ignore_past_reference") is not True
        or configuration.get("fit_seed") != 2026101902
        or configuration.get("fit_steps") != 128
        or stage.get("previous_fit_complete_hash") != original_stage["report_hash"]
        or stage.get("numeric_fit_hash") != fit["report_hash"]
        or stage.get("numeric_model_hash") != model["model_hash"]
        or stage.get("source_receipt_hashes") != original_stage["source_receipt_hashes"]
        or stage.get("eligibility") != original_stage["eligibility"]
        or stage.get("all_original_failures_retained") is not True
        or type(stage.get("optimizer_updates")) is not int
        or stage["optimizer_updates"] != 128
        or any(
            stage.get(k) is not False
            for k in (
                "own_body_student_evaluated",
                "hardware_authorized",
                "promotion_authorized",
                "dataset_admission_authorized",
                "teacher_qualified",
                "online_rl_claimed",
            )
        )
        or fit.get("past_reference_input_ablation") is not True
        or fit.get("ignored_input_weights_remain_exactly_zero") is not True
        or fit.get("previous_reference_inputs_are_teacher_forced") is not False
        or fit.get("ignored_input_feature_names") != [FEATURE_NAMES[i] for i in ignored]
        or fit.get("initial_parameters_hash") != hash_json(initial)
        or np.any(np.asarray(model["parameters"]["weight_ih"])[:, ignored] != 0)
        or lr.get("parameters") != model["parameters"]
        or lr.get("fitted_parameters_hash") != hash_json(model["parameters"])
        or lr.get("config") != original_fit["learning_receipt"]["config"]
        or type(lr.get("completed_optimizer_steps")) is not int
        or lr["completed_optimizer_steps"] != 128
        or any(
            model[k] != original[k] for k in ("mean", "scale", "world_hash", "parent_model_hash")
        )
        or any(
            fit.get(k) != original_fit[k]
            for k in (
                "normalization_hash",
                "whole_observations_hash",
                "whole_reference_labels_hash",
                "train_source_seeds",
                "validation_source_seeds",
                "input_episode_count",
                "optimized_episode_count",
                "original_ineligible_episode_count",
            )
        )
        or fit.get("validation_excluded_from_gradient_and_normalization") is not True
    ):
        raise ValueError("exact matched ablation, original split and optimizer lineage required")
    rows = stage.get("same_recorded_body_feedback_diagnostics", [])
    if len(rows) != 16 or any(
        r.get("numeric_model_hash") != model["model_hash"]
        or r.get("student_body_rollout") is not False
        or any(
            r["errors"]["teacher_forced_" + k] != r["errors"]["self_fed_" + k]
            for k in ("raw", "projected")
        )
        for r in rows
    ):
        raise ValueError("whole exported-model past-reference invariance diagnostic required")
    return model
