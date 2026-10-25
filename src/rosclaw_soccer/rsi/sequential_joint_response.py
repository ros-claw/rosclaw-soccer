"""Finite teacher planning on full NN forecasts, not a new actuator or actor."""

from collections.abc import Callable
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.joint_feasibility_restoration import (
    restore_joint_feasibility,
    task_violation,
)
from rosclaw_soccer.rsi.joint_task_response import response_features

Forecast = tuple[dict[str, Any], dict[str, Any]]


def search_sequential_joint_response(
    baseline_prediction: dict[str, Any],
    baseline_window: dict[str, Any],
    evaluate: Callable[[str, np.ndarray], Forecast],
) -> dict[str, Any]:
    """Four rounds at most; improvement requires newly observed NN forecast evidence.

    All calls must fork the SAME real donor state. Intermediate dirty trajectories
    can inform this offline teacher's search but NEVER qualify for actual adoption.
    """
    features = response_features(baseline_prediction, baseline_window)
    report: dict[str, Any] = dict(
        schema="soccer.sequential_full_nn_joint_response.v1",
        maximum_rounds=4,
        rounds=[],
        newly_requested_forecasts=0,
        actual_execution_success_proven=False,
        neural_optimizer_updates=0,
        hardware_authorized=False,
    )
    if baseline_window["window_pass"]:
        return dict(report, status="PARENT_ALREADY_PREDICTED_FEASIBLE")
    if features is None:
        return dict(report, status="ABSTAIN_BASELINE_CENSORED_OR_BODY_FAILED")
    scale = max(float(features[2]), 1.0)
    report["fixed_initial_force_scale"] = scale
    center, center_name = np.zeros(29, dtype=np.float64), "parent"
    for iteration in range(4):
        current = task_violation(features, force_scale=scale)
        best_features, best_coeff, best_name, best_score = features, center, center_name, current
        probe_results: list[tuple[np.ndarray, np.ndarray | None]] = []
        predicted_pass = False

        def observe(name: str, vector: np.ndarray) -> np.ndarray | None:
            nonlocal best_features, best_coeff, best_name, best_score, predicted_pass
            if vector.shape != (29,) or vector.dtype != np.float64 or abs(vector).max() > 0.06:
                raise ValueError("bounded whole-joint reference before any forecast required")
            coeff = np.zeros((2, 29), dtype=np.float64)
            coeff[0] = vector
            prediction, window = evaluate(name, coeff)
            report["newly_requested_forecasts"] += 1
            values = response_features(prediction, window)
            predicted_pass |= bool(window["window_pass"])
            if values is not None:
                score = task_violation(values, force_scale=scale)
                if score < best_score - 1e-6:
                    best_features, best_coeff, best_name, best_score = (
                        values.copy(),
                        vector.copy(),
                        name,
                        score,
                    )
            return values

        for axis in range(29):
            for sign in (-1, 1):
                vector = center.copy()
                vector[axis] = float(np.clip(center[axis] + sign * 0.03, -0.06, 0.06))
                name = f"joint-r{iteration}-fd-{axis:02d}-{'minus' if sign < 0 else 'plus'}"
                values = observe(name, vector)
                probe_results.append((vector, values))
        jacobian, valid = np.zeros((3, 29), dtype=np.float64), []
        for axis in range(29):
            minus, plus = probe_results[2 * axis : 2 * axis + 2]
            if minus[1] is None or plus[1] is None:
                continue
            distance = plus[0][axis] - minus[0][axis]
            if distance <= 0:
                raise ValueError("nondegenerate declared bounded signed probe interval required")
            jacobian[:, axis] = (plus[1] - minus[1]) / distance
            valid.append(axis)
        candidate, fit = None, None
        if not predicted_pass:
            candidate, fit = restore_joint_feasibility(
                features, jacobian, center, valid, force_scale=scale
            )
            if candidate is not None:
                observe(f"joint-r{iteration}-restore", candidate)
        accepted = best_score < current - 1e-6
        report["rounds"].append(
            dict(
                iteration=iteration,
                center_name=center_name,
                center=center.tolist(),
                observed_center_features=features.tolist(),
                observed_center_violation=current,
                fit=fit,
                observed_predicted_pass=predicted_pass,
                accepted_new_center=accepted,
                accepted_name=best_name if accepted else None,
                accepted_coefficients=best_coeff.tolist() if accepted else None,
                accepted_observed_features=best_features.tolist() if accepted else None,
                accepted_observed_violation=best_score if accepted else None,
            )
        )
        if report["newly_requested_forecasts"] > 4 * 59:
            raise ValueError("finite four-round full NN prediction budget exceeded")
        if predicted_pass:
            return dict(report, status="PREDICTED_TASK_FEASIBLE_REQUIRES_ACTUAL_EXAM")
        if not accepted:
            return dict(report, status="ABSTAIN_NO_NEW_NN_PHYSICAL_PROGRESS")
        center, center_name, features = best_coeff.copy(), best_name, best_features.copy()
    return dict(report, status="FOUR_ROUND_LIMIT_REQUIRES_ACTUAL_EXAM")
