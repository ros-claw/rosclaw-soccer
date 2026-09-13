"""Explicit moving-contact residual learning math; no simulator or activation.

This 136-feature schema is NOT checkpoint-compatible with other 136-input
policies. Foundation histories, measured coordinate transforms, contact truth,
Core learning leases and independent exams remain caller-owned.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

SCHEMA = "soccer.moving_contact_residual.136x29.v1"
LEARNING_FEATURE = 106


def _vector(value: np.ndarray, size: int, bound: float) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.shape != (size,)
        or value.dtype.kind != "f"
        or not np.isfinite(value).all()
        or np.any(np.abs(value) > bound)
    ):
        raise ValueError("finite bounded floating moving-contact vector required")
    return value.astype(np.float32, copy=True)


def moving_contact_observation(
    *,
    locomotion_observation: np.ndarray,
    ball_relative_position: np.ndarray,
    ball_relative_velocity: np.ndarray,
    target_relative_position: np.ndarray,
    phase: float,
    learning_active: bool,
    previous_residual: np.ndarray,
) -> np.ndarray:
    """96 causal Loco inputs + 9 local task fields + phase/bit + 29 residuals.

    The caller must transform all three task vectors to the same current body
    yaw frame. Scaling is fixed in code, never fitted to evaluation samples.
    Explicit clipping is feature normalization, not a physical safety claim.
    """
    if (
        type(learning_active) is not bool
        or type(phase) not in (int, float)
        or not math.isfinite(phase)
        or not 0 <= phase <= 1
    ):
        raise ValueError("bounded phase and explicit learning-window bit required")
    parts = (
        np.clip(_vector(locomotion_observation, 96, 10000) / 10, -1, 1),
        np.clip(_vector(ball_relative_position, 3, 100) / 2, -1, 1),
        np.clip(_vector(ball_relative_velocity, 3, 100) / 5, -1, 1),
        np.clip(_vector(target_relative_position, 3, 100) / 5, -1, 1),
        np.array([2 * phase - 1, float(learning_active)], dtype=np.float32),
        _vector(previous_residual, 29, 0.250001) / np.float32(0.25),
    )
    result: np.ndarray = np.concatenate(parts).astype(np.float32)
    return result


def advance_moving_contact_residual(
    raw: np.ndarray,
    previous: np.ndarray,
    *,
    learning_active: bool,
) -> np.ndarray:
    """At most .25 rad residual and .025 rad/decision change, proposals only.

    Outside the declared learning window it ramps to zero, never resets the
    physical robot or jumps the residual to zero. Base target/gain and torque
    constraints remain the execution owner's responsibility.
    """
    if type(learning_active) is not bool:
        raise ValueError("explicit learning-window bit required")
    value = _vector(raw, 29, 100)
    last = _vector(previous, 29, 0.250001)
    wanted = np.tanh(value) * np.float32(0.25) if learning_active else np.zeros(29, np.float32)
    result: np.ndarray = np.clip(last + np.clip(wanted - last, -0.025, 0.025), -0.25, 0.25)
    return result.astype(np.float32)


def moving_contact_potential(
    *, progress: float, lateral_error: float, height: float, failed: bool
) -> float:
    """Dense contact-learning signal, not a pass-success classifier."""
    if type(failed) is not bool or any(
        type(x) not in (int, float) or not math.isfinite(x) or abs(x) > 100
        for x in (progress, lateral_error, height)
    ):
        raise ValueError("finite bounded contact diagnostics required")
    if failed:
        return 0.0
    return float(
        3 * np.clip(progress, 0, 1)
        - 6 * min(abs(lateral_error), 1)
        - 3 * np.clip(height - 0.2, 0, 1)
    )


def build_moving_contact_actor_critic() -> Any:
    """Fresh zero-mean 136/29 residual actor; no foundation weights owned."""
    import torch

    class MovingContactActorCritic(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()

            def network(width: int) -> Any:
                model = torch.nn.Sequential(
                    torch.nn.Linear(136, 128),
                    torch.nn.Tanh(),
                    torch.nn.Linear(128, 128),
                    torch.nn.Tanh(),
                    torch.nn.Linear(128, width),
                )
                output: Any = model[-1]
                torch.nn.init.zeros_(output.weight)
                torch.nn.init.zeros_(output.bias)
                return model

            self.actor = network(29)
            self.critic = network(1)
            self.logstd = torch.nn.Parameter(torch.full((29,), -2.5))

        def forward(self, observation: Any) -> tuple[Any, Any]:
            if (
                observation.ndim != 2
                or observation.shape[1] != 136
                or not bool(torch.isfinite(observation).all())
            ):
                raise ValueError("complete finite moving-contact observation required")
            return self.actor(observation), self.critic(observation).squeeze(-1)

    return MovingContactActorCritic()
