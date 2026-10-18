"""Separate numerical replay qualification from a task's success label.

The historical CPU quality flag excludes the separately reported pelvis gate.
Do not modify that historical flag, or treat a legal unsafe outcome as an
infrastructure failure. Callers still authenticate/recompute both reviews.
"""

import math
import re
from typing import Any

from rosclaw_soccer.rsi.first_touch_outcome import FirstTouchOutcome, outcome_contract


def require_first_touch_qualification(
    review: dict[str, Any],
    foundation: dict[str, Any],
    outcome: FirstTouchOutcome,
    *,
    cerebellar_model_hash: str | None = None,
    cerebellar_sampling_draws: int | None = None,
    reference_student_model_hash: str | None = None,
) -> dict[str, Any]:
    """Validate counts and label composition, never authorize execution."""
    if (
        type(review) is not dict
        or type(foundation) is not dict
        or type(outcome) is not FirstTouchOutcome
    ):
        raise ValueError("typed replay reviews and primitive M0 outcome required")
    reference_review = review.get("reference_student_review")
    if reference_student_model_hash is None:
        if "reference_student_review" in review:
            raise ValueError("unexpected reference student requires explicit qualification intent")
    elif (
        type(reference_student_model_hash) is not str
        or re.fullmatch(r"sha256:[0-9a-f]{64}", reference_student_model_hash) is None
        or type(reference_review) is not dict
        or reference_review.get("schema")
        != "soccer.rsi.pre_parent_reference_student_reconstruction.v1"
        or reference_review.get("model_hash") != reference_student_model_hash
        or type(reference_review.get("frames_reconstructed")) is not int
        or reference_review["frames_reconstructed"] != 300
        or any(
            reference_review.get(k) is not True
            for k in (
                "own_hidden_and_applied_reference_reconstructed",
                "raw_foundation_navigation_targets_bit_exact",
                "reconstruction_uses_same_pinned_decoder",
                "parent_target_composition_reconstructed_separately",
            )
        )
        or any(
            reference_review.get(k) is not False
            for k in (
                "independent_neural_implementation",
                "training_provenance_authenticated",
                "teacher_qualified",
                "hardware_authorized",
                "promotion_authorized",
            )
        )
        or any(
            review.get(k) is not True
            for k in (
                "actual_mujoco_dynamics_replayed",
                "actual_pd_torque_reconstructed",
                "neural_target_reconstructed",
            )
        )
    ):
        raise ValueError(
            "whole own-state reference student and parent motor reconstruction required"
        )
    if cerebellar_model_hash is None:
        if cerebellar_sampling_draws is not None or any(
            key.startswith("cerebellar_") or key.startswith("actual_cerebellar_") for key in review
        ):
            raise ValueError(
                "unexpected neural torque layer requires explicit qualification intent"
            )
    elif (
        type(cerebellar_model_hash) is not str
        or re.fullmatch(r"sha256:[0-9a-f]{64}", cerebellar_model_hash) is None
        or type(cerebellar_sampling_draws) is not int
        or cerebellar_sampling_draws not in (0, 3000)
        or review.get("cerebellar_model_hash") != cerebellar_model_hash
        or type(review.get("actual_cerebellar_inferences_reconstructed")) is not int
        or review["actual_cerebellar_inferences_reconstructed"] != 3000
        or type(review.get("actual_cerebellar_sampling_draws_reconstructed")) is not int
        or review["actual_cerebellar_sampling_draws_reconstructed"] != cerebellar_sampling_draws
        or review.get("cerebellar_actual_pd_plus_residual_torque_reconstructed") is not True
    ):
        raise ValueError(
            "whole actual neural torque inference, sampling and dynamics qualification required"
        )
    for document, key, expected in (
        (review, "physical_substeps", 3000),
        (review, "actual_recurrent_sampling_draws_reconstructed", 270),
        (foundation, "recomputed_executed_calls", 300),
    ):
        if type(document.get(key)) is not int or document[key] != expected:
            raise ValueError("whole independent physical and Foundation qualification required")
    if any(
        type(review.get(key)) is not bool
        for key in ("high_quality", "safety_passed", "clean_foot_only")
    ):
        raise ValueError("explicit independent CPU quality and safety flags required")
    minimum = review.get("minimum_pelvis_z_m")
    if type(minimum) is not float or not math.isfinite(minimum):
        raise ValueError("finite measured CPU pelvis minimum required")
    threshold = outcome_contract()["minimum_safe_pelvis_m"]
    if review["safety_passed"] is not (minimum >= threshold):
        raise ValueError("independent CPU safety flag differs from frozen pelvis gate")
    if not math.isclose(minimum, outcome.minimum_pelvis_z_m, abs_tol=1e-8, rel_tol=0):
        raise ValueError("independent and recorded pelvis minima differ")
    if review["high_quality"] and not review["clean_foot_only"]:
        raise ValueError("legacy ball quality cannot pass without clean foot contact")
    combined = review["high_quality"] and review["safety_passed"]
    if combined is not outcome.task_pass:
        raise ValueError("independent CPU quality plus safety differs from primitive M0 outcome")
    result = dict(
        schema="soccer.rsi.first_touch_qualification_alignment.v1",
        legacy_cpu_ball_quality=review["high_quality"],
        independent_cpu_pelvis_safety=review["safety_passed"],
        m0_task_pass=combined,
        outcome_contract_hash=outcome_contract()["contract_hash"],
        legacy_quality_flag_alone_is_not_task_admission=True,
        input_replays_authenticated_here=False,
        task_or_safety_contract_changed=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    if cerebellar_model_hash is not None:
        result["cerebellar_model_hash"] = cerebellar_model_hash
        result["cerebellar_sampling_draws"] = cerebellar_sampling_draws
    if reference_student_model_hash is not None:
        result["reference_student_model_hash"] = reference_student_model_hash
    return result
