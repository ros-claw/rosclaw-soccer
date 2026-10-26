"""Bounded SIM-only constant-velocity reference for a measured rolling ball."""

from typing import Any

import numpy as np


def predict_ball_reference(
    measured_position: np.ndarray, measured_velocity: np.ndarray, *, horizon_s: float
) -> tuple[np.ndarray, dict[str, Any]]:
    """Predict a planning point, not a measurement or physical ball-state update."""
    if (
        any(
            type(x) is not np.ndarray
            or x.shape != (3,)
            or x.dtype.kind != "f"
            or not np.isfinite(x).all()
            for x in (measured_position, measured_velocity)
        )
        or type(horizon_s) is not float
        or not np.isfinite(horizon_s)
        or not 1.0 <= horizon_s <= 2.0
        or np.linalg.norm(measured_velocity[:2]) > 0.2
        or abs(measured_velocity[2]) > 1e-12
    ):
        raise ValueError("finite measured ground ball and bounded SIM reference horizon required")
    prediction = measured_position.astype(np.float64, copy=True) + measured_velocity * horizon_s
    if np.linalg.norm(prediction - measured_position) > 0.3:
        raise ValueError("predicted ball displacement exceeds reference bound")
    return prediction, dict(
        measured_ball_position=measured_position.tolist(),
        measured_ball_velocity=measured_velocity.tolist(),
        predicted_ball_position=prediction.tolist(),
        horizon_s=horizon_s,
        prediction_is_not_future_observation=True,
        privileged_sim_velocity_input=True,
        physical_state_modified=False,
        hardware_authorized=False,
    )
