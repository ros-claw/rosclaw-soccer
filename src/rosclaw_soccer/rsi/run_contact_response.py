"""Local TRAIN intervention response, not a motor policy or qualification.

Physical outcomes are labels. Ridge slopes propose bounded *experiments*;
they never count as predicted successful football controls.
"""

import numpy as np


def fit_response(x: np.ndarray, y: np.ndarray) -> dict[str, object]:
    if (
        x.ndim != 2
        or x.shape[1] != 12
        or y.shape != (len(x), 2)
        or not 14 <= len(x) <= 256
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or (np.abs(x) > 1).any()
    ):
        raise ValueError("finite sufficient TRAIN12 intent/physical2 displacement rows required")
    center = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 0.1)
    z = (x - center) / scale
    design = np.column_stack([np.ones(len(x)), z])
    penalty = np.diag([0.0] + [0.1] * 12)
    inverse = np.linalg.inv(design.T @ design + penalty)
    beta = inverse @ design.T @ y
    fitted = design @ beta
    leverage = np.einsum("ij,jk,ik->i", design, inverse, design)
    if (leverage >= 1 - 1e-8).any():
        raise ValueError("singular leave-one-out response diagnostic")
    loo_error = (y - fitted) / (1 - leverage[:, None])
    return dict(
        rows=len(x),
        center=center.tolist(),
        scale=scale.tolist(),
        intercept=beta[0].tolist(),
        latent_gradient=(beta[1:] / scale[:, None]).tolist(),
        train_rmse=np.sqrt(np.mean((y - fitted) ** 2, axis=0)).tolist(),
        leave_one_out_rmse=np.sqrt(np.mean(loo_error**2, axis=0)).tolist(),
        ridge=0.1,
        labels=["actual_lateral_signed_1p2s_m", "actual_progress_1p2s_m"],
        runtime_motor_policy=False,
        predicted_success_is_evidence=False,
    )


def bounded_proposals(base: np.ndarray, lateral: float, gradient: np.ndarray) -> list[list[float]]:
    if (
        base.shape != (12,)
        or gradient.shape != (12,)
        or not np.isfinite(base).all()
        or not np.isfinite(gradient).all()
        or not np.isfinite(lateral)
        or (np.abs(base) > 1).any()
    ):
        raise ValueError("finite actual base and response direction required")
    error = lateral - float(np.clip(lateral, -0.1, 0.1))
    direction = -gradient * error / (0.01 + gradient @ gradient)
    norm = np.linalg.norm(direction)
    if norm > 0.5:
        direction *= 0.5 / norm
    proposals = [base.copy()]
    for alpha in (0.25, 0.5, 0.75, 1.0):
        proposals.append(np.clip(base + alpha * direction, -1, 1))
    # Directional interventions test slope validity instead of trusting it.
    for axis in np.argsort(-np.abs(gradient), kind="stable")[:3]:
        for sign in (-1.0, 1.0):
            probe = base.copy()
            probe[axis] = np.clip(probe[axis] + sign * 0.15, -1, 1)
            proposals.append(probe)
    return [v.tolist() for v in proposals]
