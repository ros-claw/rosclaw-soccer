"""Separate acquisition, observation coverage and M0 without changing its gate.

Callers authenticate the complete declared cohort and physical evidence. Late
contact is acquisition with unobserved ball quality, never fabricated zero travel.
This readout does not qualify a teacher, readiness, receiving or a whole match.
"""

import math
from collections.abc import Sequence
from typing import Any, cast

from rosclaw_soccer.rsi.first_touch_outcome import outcome_contract
from rosclaw_soccer.rsi.paired_learning_assessment import assess_pairs
from rosclaw_soccer.sim.contracts import hash_json


def _stage(outcome: dict[str, Any]) -> dict[str, bool]:
    if type(outcome) is not dict:
        raise ValueError("explicit original first-touch outcome required")
    first = outcome.get("first_foot_contact_frame")
    if first is not None and (type(first) is not int or not 0 <= first < 300):
        raise ValueError("original0..299 or no foot contact required")
    acquired = first is not None
    observed = first is not None and first + 60 < 300
    safe = outcome.get("safe")
    if (
        "first_foot_contact_frame" not in outcome
        or type(safe) is not bool
        or type(outcome.get("task_pass")) is not bool
        or type(outcome.get("quality_observed")) is not bool
        or outcome["quality_observed"] != observed
        or outcome.get("outcome_contract_hash") != outcome_contract()["contract_hash"]
        or outcome.get("measurement_status")
        != ("OBSERVED" if observed else "RIGHT_CENSORED" if acquired else "NOT_APPLICABLE")
        or outcome.get("terminated") is not True
        or outcome.get("truncated") is not False
        or any(outcome.get(k) is not False for k in ("hardware_authorized", "promotion_authorized"))
    ):
        raise ValueError("original fixed-horizon primitive labels and observation status required")
    for key in ("minimum_pelvis_z_m", "maximum_lateral_excursion_m"):
        if type(outcome.get(key)) not in (float, int) or not math.isfinite(outcome[key]):
            raise ValueError("finite actual safety and excursion measurements required")
    if outcome["maximum_lateral_excursion_m"] < 0 or (
        safe and outcome["minimum_pelvis_z_m"] < 0.65
    ):
        raise ValueError("original safety and nonnegative excursion must be preserved")
    travel = [outcome.get(k) for k in ("forward_60_m", "lateral_60_m")]
    if observed:
        if any(type(x) not in (float, int) or not math.isfinite(cast(float, x)) for x in travel):
            raise ValueError("observed quality requires both finite measured travel components")
    elif any(x is not None for x in travel):
        raise ValueError("censored or absent contact must not fabricate travel")
    forward, lateral = (cast(float, x) for x in travel)
    passed = bool(
        safe
        and observed
        and forward >= 1.0
        and abs(lateral) / max(forward, 0.01) <= 0.3
        and outcome["maximum_lateral_excursion_m"] <= 4.0
    )
    status = (
        "UNSAFE_CONTACT"
        if not safe
        else "NO_FOOT_CONTACT"
        if not acquired
        else "POST_CONTACT_CENSORED"
        if not observed
        else "CONTACT_OBSERVED"
    )
    reason = (
        "SAFETY_VETO"
        if not safe
        else "NO_FOOT_CONTACT"
        if not acquired
        else "POST_CONTACT_WINDOW_INCOMPLETE"
        if not observed
        else None
        if passed
        else "BALL_OUTCOME_FAILED"
    )
    if (
        outcome.get("task_pass") != passed
        or outcome.get("event_status") != status
        or outcome.get("failure_reason") != reason
    ):
        raise ValueError("original M0 score, safety precedence and failure reason must not change")
    return dict(
        acquired=acquired,
        observed=observed,
        censored=acquired and not observed,
        safe=safe,
        safe_acquired=safe and acquired,
        safe_observed=safe and observed,
        task_pass=passed,
    )


def assess_first_touch_stages(
    controls: dict[str, dict[str, Any]],
    evaluations: dict[str, dict[str, Any]],
    *,
    declared_pair_ids: Sequence[str],
) -> dict[str, Any]:
    """Conditional rates keep their denominators; empty strata remain unknown."""
    if isinstance(declared_pair_ids, (str, bytes)):
        raise ValueError("explicit declared pair sequence, not a string, required")
    ids = list(declared_pair_ids)
    if (
        not ids
        or any(type(k) is not str or not k for k in ids)
        or len(set(ids)) != len(ids)
        or set(controls) != set(ids)
        or set(evaluations) != set(ids)
    ):
        raise ValueError("all distinct declared pairs including failures required")
    base = {k: _stage(controls[k]) for k in ids}
    evaluated = {k: _stage(evaluations[k]) for k in ids}

    def summary(stages: dict[str, dict[str, bool]]) -> dict[str, Any]:
        counts = {k: sum(row[k] for row in stages.values()) for k in next(iter(stages.values()))}
        return dict(
            endpoints=len(ids),
            **counts,
            contact_rate=counts["acquired"] / len(ids),
            safe_contact_rate=counts["safe_acquired"] / len(ids),
            m0_task_pass_rate=counts["task_pass"] / len(ids),
            measured_quality_coverage_given_contact=(
                counts["observed"] / counts["acquired"] if counts["acquired"] else None
            ),
            m0_pass_given_safe_observed_contact=(
                counts["task_pass"] / counts["safe_observed"] if counts["safe_observed"] else None
            ),
            censored_quality_not_imputed=True,
        )

    transitions: list[dict[str, Any]] = [
        dict(
            pair_id=k,
            baseline=base[k],
            evaluated=evaluated[k],
            contact_rescued=not base[k]["acquired"] and evaluated[k]["acquired"],
            contact_regressed=base[k]["acquired"] and not evaluated[k]["acquired"],
            quality_observation_gained=not base[k]["observed"] and evaluated[k]["observed"],
            quality_observation_lost=base[k]["observed"] and not evaluated[k]["observed"],
        )
        for k in ids
    ]
    result = dict(
        schema="soccer.rsi.first_touch_stage_diagnostics.v1",
        outcome_contract_hash=outcome_contract()["contract_hash"],
        original_m0_pairs=assess_pairs(
            {k: controls[k] for k in ids}, {k: evaluations[k] for k in ids}
        ),
        baseline=summary(base),
        evaluated=summary(evaluated),
        contact_rescues=sum(r["contact_rescued"] for r in transitions),
        contact_regressions=sum(r["contact_regressed"] for r in transitions),
        quality_observation_gains=sum(r["quality_observation_gained"] for r in transitions),
        quality_observation_losses=sum(r["quality_observation_lost"] for r in transitions),
        transitions=transitions,
        missing_quality_preserved=True,
        empirical_fractions_not_calibrated_capability_probabilities=True,
        conditional_rate_alone_must_not_select_a_winner=True,
        original_exam_unchanged=True,
        source_and_raw_trace_authentication_performed_here=False,
        successor_readiness_qualified=False,
        receiving_skill_qualified=False,
        teacher_qualified=False,
        dataset_admission_authorized=False,
        new_physical_steps=0,
        new_neural_inferences=0,
        new_optimizer_updates=0,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    return dict(result, report_hash=hash_json(result))
