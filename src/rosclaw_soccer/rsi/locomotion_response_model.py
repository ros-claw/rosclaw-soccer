"""Small offline delayed velocity model; free rollout, never teacher-forced admission.

No actor, simulator, weights, or robot interface. Training delay selection sees
only nine training interventions; all six validation commands stay excluded.
"""

from typing import Any

import numpy as np


def _episode(command: np.ndarray, velocity: np.ndarray) -> None:
    for x in (command, velocity):
        if (
            type(x) is not np.ndarray
            or x.shape != (300, 2)
            or x.dtype != np.float64
            or not np.isfinite(x).all()
        ):
            raise ValueError("finite H300 float64 XY observations required")


def predict(
    weights: np.ndarray, delay: int, command: np.ndarray, initial_velocity: np.ndarray
) -> np.ndarray:
    if (
        type(weights) is not np.ndarray
        or weights.shape != (5, 2)
        or weights.dtype != np.float64
        or not np.isfinite(weights).all()
        or type(delay) is not int
        or not 0 <= delay <= 20
        or type(initial_velocity) is not np.ndarray
        or initial_velocity.shape != (2,)
        or initial_velocity.dtype != np.float64
        or not np.isfinite(initial_velocity).all()
    ):
        raise ValueError("bounded finite delayed model and measured initial velocity required")
    _episode(command, np.zeros((300, 2)))
    v = np.zeros((300, 2))
    v[0] = initial_velocity
    for f in range(299):
        u = command[f - delay] if f >= delay else np.zeros(2)
        v[f + 1] = np.concatenate((v[f], u, [1.0])) @ weights
        if not np.isfinite(v[f + 1]).all() or np.linalg.norm(v[f + 1]) > 5:
            raise ValueError("unstable unforced model rollout")
    return v


def fit_training(episodes: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, Any]:
    if type(episodes) is not list or len(episodes) != 9:
        raise ValueError("exact nine training interventions, never validation rows, required")
    for u, v in episodes:
        _episode(u, v)
    candidates = []
    for delay in range(21):
        xs, ys = [], []
        for u, v in episodes:
            history = np.zeros((299, 2))
            history[delay:] = u[: 299 - delay]
            xs.append(np.column_stack((v[:-1], history, np.ones(299))))
            ys.append(v[1:])
        x, y = np.concatenate(xs), np.concatenate(ys)
        # Fixed ridge, no validation-dependent hyperparameter selection.
        weights = np.linalg.solve(x.T @ x + 1e-4 * np.eye(5), x.T @ y)
        try:
            forecasts = [predict(weights, delay, u, v[0]) for u, v in episodes]
            rmse = float(
                np.sqrt(
                    np.mean(
                        np.concatenate(
                            [p - v for p, (_, v) in zip(forecasts, episodes, strict=True)]
                        )
                        ** 2
                    )
                )
            )
            if not np.isfinite(rmse):
                raise ValueError("nonfinite training loss")
        except ValueError:
            rmse = None
        candidates.append(
            dict(
                delay_frames=delay,
                training_free_rollout_velocity_rmse_mps=rmse,
                weights=weights.tolist(),
            )
        )
    valid = [c for c in candidates if c["training_free_rollout_velocity_rmse_mps"] is not None]
    if not valid:
        raise ValueError("all delayed model fits failed free rollout; no predictor admitted")
    selected = min(
        valid, key=lambda c: (c["training_free_rollout_velocity_rmse_mps"], c["delay_frames"])
    )
    return dict(
        schema="soccer.rsi.delayed_root_velocity_model.v1",
        selected=selected,
        candidates=candidates,
        ridge=1e-4,
        training_interventions=9,
        validation_used_for_fit=False,
        model_fit_coefficients=10,
        delay_candidates=21,
        neural_optimizer_updates=0,
        model_is_not_motor_policy=True,
        promotion_authorized=False,
        hardware_authorized=False,
    )


def validate_prediction(
    weights: np.ndarray,
    delay: int,
    command: np.ndarray,
    velocity: np.ndarray,
    position: np.ndarray,
    *,
    onset_frame: int,
) -> dict[str, Any]:
    _episode(command, velocity)
    if (
        type(position) is not np.ndarray
        or position.shape != (300, 2)
        or position.dtype != np.float64
        or not np.isfinite(position).all()
        or type(onset_frame) is not int
        or onset_frame != 75
    ):
        raise ValueError("all fixed validation intervention positions and onset75 required")
    forecast = predict(weights, delay, command, velocity[0])
    p = np.zeros((300, 2))
    p[0] = position[0]
    # Trapezoidal integration exposes model position drift instead of resetting
    # to measured states at each frame or one-step teacher forcing.
    p[1:] = p[0] + np.cumsum(0.01 * (forecast[:-1] + forecast[1:]), axis=0)
    error = np.linalg.norm(p - position, axis=1)
    rmse = float(np.sqrt(np.mean((forecast - velocity) ** 2)))
    zero_rmse = float(np.sqrt(np.mean(velocity**2)))
    return dict(
        velocity_rmse_mps=rmse,
        zero_velocity_baseline_rmse_mps=zero_rmse,
        one_second_after_onset_position_error_m=float(error[onset_frame + 50]),
        maximum_position_error_m=float(error.max()),
        predictor_research_gate=bool(
            rmse <= 0.1
            and rmse <= 0.8 * zero_rmse
            and error[onset_frame + 50] <= 0.04
            and error.max() <= 0.08
        ),
        actual_motor_control_validated=False,
    )
