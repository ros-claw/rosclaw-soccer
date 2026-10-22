"""Learn a small local TRAIN action-response model for a SIM teacher.

Inputs are12 bounded joint-stroke parameters, outputs measured signed pass
error(lateral/longitudinal/vertical) and speed-minus2.8m/s. Predictions NEVER
qualify a motor or football skill. Every candidate needs new native physics.
No student neural weights, PPO likelihood or hardware command lives here.
"""

from typing import Any

import numpy as np


def feasible_first_rank(result: dict[str, Any]) -> tuple[int, int, int, int, float]:
    error, speed = result["minimum_goal_error_3d_m"], result["arrival_speed_mps"]
    # v1 CEM's softspeed penalty could prefer an infeasible point over an
    # already feasible witness. Preserve v1; this NEW selector puts the exact
    # existing physical success predicate ahead of any continuous cost.
    score = -1e6 if error is None else -error / 0.1 - max(0.0, speed - 3.0) / 0.2
    return (
        int(result["whole_body_healthy"] and result["recovery_observed"]),
        int(result["clean_foot"]),
        int(result["first_foot_contact_substep"] is not None),
        int(result["precision_0p1m_moderate_speed_success"]),
        score,
    )


def response_proposals(
    latents: np.ndarray, responses: np.ndarray, *, start: np.ndarray, start_response: np.ndarray
) -> dict[str, Any]:
    n = len(latents)
    if (
        latents.shape != (n, 2, 6)
        or responses.shape != (n, 4)
        or start.shape != (2, 6)
        or start_response.shape != (4,)
        or any(not np.isfinite(a).all() for a in (latents, responses, start, start_response))
        or (abs(latents) > 1).any()
        or (abs(start) > 1).any()
        or len(np.unique(latents.reshape(n, 12), axis=0)) != n
    ):
        raise ValueError("finite unique bounded actual parameter/response rows required")
    if n < 14:
        return dict(status="INSUFFICIENT_SAFE_UNIQUE_DATA", unique_rows=n, candidates=[])
    x = np.column_stack((np.ones(n), (latents - start).reshape(n, 12)))
    rank = int(np.linalg.matrix_rank(x))
    if rank < 13:
        return dict(status="RANK_DEFICIENT_SAFE_DATA", unique_rows=n, rank=rank, candidates=[])
    weights = np.exp(-np.square(x[:, 1:]).sum(axis=1) / (2 * 0.8**2))
    penalty = np.diag([1e-8] + [1.0] * 12)
    models, candidates = [], []
    scales = np.array([1.0, 1.0, 1.0, 0.2])
    residual = start_response.copy()
    # 2.8 is an upper margin, not a demand to accelerate an already slow pass.
    residual[3] = max(0.0, residual[3])
    for ridge in (0.001, 0.01, 0.1):
        normal = x.T @ (weights[:, None] * x) + ridge * penalty
        coefficient = np.linalg.solve(normal, x.T @ (weights[:, None] * responses))
        predicted = x @ coefficient
        loo = []
        for i in range(n):
            keep = np.arange(n) != i
            xx, yy, ww = x[keep], responses[keep], weights[keep]
            fitted = np.linalg.solve(
                xx.T @ (ww[:, None] * xx) + ridge * penalty, xx.T @ (ww[:, None] * yy)
            )
            loo.append(x[i] @ fitted - responses[i])
        jacobian = coefficient[1:].T * scales[:, None]
        delta = -jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + ridge * np.eye(4), residual * scales
        )
        magnitude = float(np.max(abs(delta)))
        if magnitude > 0.2:
            delta *= 0.2 / magnitude
        models.append(
            dict(
                ridge=ridge,
                coefficients=coefficient.tolist(),
                normal_condition=float(np.linalg.cond(normal)),
                train_rmse=np.sqrt(np.mean(np.square(predicted - responses), axis=0)).tolist(),
                leave_one_out_rmse=np.sqrt(np.mean(np.square(loo), axis=0)).tolist(),
                fit_start_response=coefficient[0].tolist(),
            )
        )
        for step_scale in (0.5, 1.0):
            requested = start.reshape(12) + step_scale * delta
            point = np.clip(requested, -1.0, 1.0).reshape(2, 6)
            response = np.r_[1.0, (point - start).reshape(12)] @ coefficient
            candidates.append(
                dict(
                    ridge=ridge,
                    step_scale=step_scale,
                    latent=point.tolist(),
                    projected_coordinates=int(np.count_nonzero(point.reshape(12) != requested)),
                    predicted_response=response.tolist(),
                )
            )
    return dict(
        status="LOCAL_RESPONSE_MODELS_FIT_NOT_PHYSICALLY_VALIDATED",
        unique_rows=n,
        rank=rank,
        local_weight_sigma=0.8,
        trust_region_latent_infinity=0.2,
        target_arrival_speed_mps=2.8,
        safe_speed_is_not_accelerated_to_margin=True,
        response_scales=scales.tolist(),
        actual_start_response=start_response.tolist(),
        models=models,
        candidates=candidates,
        predictions_are_physical_evidence=False,
        student_optimizer_updates=0,
    )
