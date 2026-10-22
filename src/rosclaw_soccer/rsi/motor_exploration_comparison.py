"""Paired TRAIN intervention readout, not policy improvement or promotion."""

import math
from statistics import median
from typing import Any

from rosclaw_soccer.rsi.motor_exploration_curriculum import motor_curriculum


def physical_readout(records: list[dict[str, Any]]) -> dict[str, Any]:
    windows = [r["result"]["postcontact"] for r in records]
    measured = [p for p in windows if p["post_contact_quality_observed"]]
    values = {}
    for field in (
        "goal_progress_1p2s_m",
        "goal_lateral_1p2s_m",
        "maximum_pelvis_retreat_after_contact_m",
    ):
        raw = [p[field] for p in measured]
        if any(type(v) not in (float, int) or not math.isfinite(v) for v in raw):
            raise ValueError("observed windows require finite actual measurements")
        values[field + "_median"] = median(raw) if raw else None
    return dict(
        observed_windows=len(measured),
        correct_foot_first=sum(p["correct_foot_first"] for p in windows),
        healthy_neural_successors=sum(
            r.get("stand") is not None and r["stand"]["healthy"] for r in records
        ),
        **values,
    )


def compare_motor_profiles(
    control: list[dict[str, Any]], treatment: list[dict[str, Any]], *, generation: int
) -> dict[str, Any]:
    curriculum = motor_curriculum("retention-run64")
    pairs = curriculum.shard_pairs(generation, 0, 1)
    for records in (control, treatment):
        if [(r["context"], r["seed"]) for r in records] != pairs:
            raise ValueError("complete identical ordered paired worlds required")
        for row in records:
            if (
                type(row["healthy_quality"]) is not bool
                or type(row["result"]["body_failed"]) is not bool
                or type(row["result"]["forbidden"]) is not bool
                or row["assisted"] is not False
                or row["rejection"] is not None
            ):
                raise ValueError("unassisted actual bounded outcomes required")
    by_context = {}
    for context in curriculum.context_ids:
        a = [r for r in control if r["context"] == context]
        b = [r for r in treatment if r["context"] == context]
        by_context[str(context)] = dict(
            pairs=len(a),
            control_physical=physical_readout(a),
            treatment_physical=physical_readout(b),
            control_quality=sum(r["healthy_quality"] for r in a),
            treatment_quality=sum(r["healthy_quality"] for r in b),
            rescues=sum(
                not x["healthy_quality"] and y["healthy_quality"] for x, y in zip(a, b, strict=True)
            ),
            regressions=sum(
                x["healthy_quality"] and not y["healthy_quality"] for x, y in zip(a, b, strict=True)
            ),
            control_body_failures=sum(r["result"]["body_failed"] for r in a),
            treatment_body_failures=sum(r["result"]["body_failed"] for r in b),
            control_nonfoot_worlds=sum(r["result"]["forbidden"] for r in a),
            treatment_nonfoot_worlds=sum(r["result"]["forbidden"] for r in b),
        )
    hard_gain = sum(
        by_context[str(c)]["treatment_quality"] - by_context[str(c)]["control_quality"]
        for c in (4, 20)
    )
    no_new_falls = all(
        r["treatment_body_failures"] <= r["control_body_failures"] for r in by_context.values()
    )
    return dict(
        contexts=by_context,
        paired_worlds=curriculum.worlds,
        total_executions=2 * curriculum.worlds,
        distinct_task_contexts=len(curriculum.context_ids),
        difficult_context_net_quality=hard_gain,
        next_research_action=(
            "ONE_BOUNDED_WHOLE_BANK_LEARNING_TRIAL_NOT_POLICY_GAIN"
            if hard_gain > 0 and no_new_falls
            else "PIVOT_TO_TEACHER_STUDENT_DISTRIBUTION_AND_CONTROL_INTERFACE"
        ),
        new_policy_learning_proven=False,
        fresh_generalization_proven=False,
        promotion_authorized=False,
        goal_complete=False,
    )
