"""Describe local-model prediction errors without claiming policy improvement."""

import math
from typing import Any


def vector(value: Any) -> tuple[float, ...]:
    if (
        type(value) is not list
        or len(value) != 4
        or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)
    ):
        raise ValueError("four finite actual/predicted response scalars required")
    return tuple(float(v) for v in value)


def cost(v: tuple[float, ...]) -> float:
    """Original .1m/3mps delivery limits; velocity coordinate is speed minus2.8."""
    return math.hypot(*v[:3]) / 0.1 + max(v[3] - 0.2, 0.0) / 0.2


def response_validation(*, anchor: list[float], proposals: list[dict[str, Any]]) -> dict[str, Any]:
    base = vector(anchor)
    if len(proposals) != 6 or [p["model_candidate_index"] for p in proposals] != list(range(6)):
        raise ValueError("all six ordered physical model proposals required")
    rows = []
    observed_errors = []
    for p in proposals:
        predicted = vector(p["predicted_response"])
        actual = None if p["actual_response"] is None else vector(p["actual_response"])
        error = None if actual is None else [a - b for a, b in zip(actual, predicted, strict=True)]
        if error is not None:
            observed_errors.append(error)
        rows.append(
            dict(
                candidate_index=p["model_candidate_index"],
                predicted_response=list(predicted),
                actual_response=None if actual is None else list(actual),
                actual_minus_predicted=error,
                predicted_cost=cost(predicted),
                actual_cost=None if actual is None else cost(actual),
                predicted_improvement=cost(predicted) < cost(base),
                actual_improvement=None if actual is None else cost(actual) < cost(base),
            )
        )
    return dict(
        anchor_response=list(base),
        anchor_cost=cost(base),
        rows=rows,
        proposals=6,
        observed=len(observed_errors),
        missing=6 - len(observed_errors),
        coordinate_rmse=None
        if not observed_errors
        else [
            math.sqrt(sum(v[k] ** 2 for v in observed_errors) / len(observed_errors))
            for k in range(4)
        ],
        predicted_improvements=sum(r["predicted_improvement"] for r in rows),
        actual_improvements=sum(r["actual_improvement"] is True for r in rows),
        predicted_improvement_but_actual_not=sum(
            r["predicted_improvement"] and r["actual_improvement"] is False for r in rows
        ),
        missing_response_is_failure_not_zero=True,
        cost_is_diagnostic_not_receive_label=True,
        student_improvement_verified=False,
        promotion_authorized=False,
    )
