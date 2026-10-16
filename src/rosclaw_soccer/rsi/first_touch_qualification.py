"""Separate numerical replay qualification from a task's success label.

The historical CPU quality flag excludes the separately reported pelvis gate.
Do not modify that historical flag, or treat a legal unsafe outcome as an
infrastructure failure. Callers still authenticate/recompute both reviews.
"""

import math
from typing import Any

from rosclaw_soccer.rsi.first_touch_outcome import FirstTouchOutcome, outcome_contract


def require_first_touch_qualification(
    review: dict[str, Any], foundation: dict[str, Any], outcome: FirstTouchOutcome
) -> dict[str, Any]:
    """Validate counts and label composition, never authorize execution."""
    if (
        type(review) is not dict
        or type(foundation) is not dict
        or type(outcome) is not FirstTouchOutcome
    ):
        raise ValueError("typed replay reviews and primitive M0 outcome required")
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
    return dict(
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
