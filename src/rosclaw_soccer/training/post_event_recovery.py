"""Pure offline recovery cost after a measured event, never motion authority.

Inputs are task-independent scalar observations in a declared episode-fixed
forward frame. This cost does not require a G1, a ball, a simulator, Torch or a
runtime. A caller must separately preserve task outcomes and safety gates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PostEventRecoveryConfig:
    delay_sec: float = 0.2
    minimum_clearance_m: float = 0.35
    backward_tolerance_mps: float = 0.1
    angular_tolerance_radps: float = 1.0
    tilt_tolerance_rad: float = 0.15
    backward_weight: float = 20.0
    angular_weight: float = 2.0
    tilt_weight: float = 4.0
    dt_sec: float = 0.02
    per_step_cap: float = 2.0

    def __post_init__(self) -> None:
        if any(
            type(value) not in (int, float) or not math.isfinite(value) or value < 0
            for value in vars(self).values()
        ):
            raise ValueError("finite nonnegative recovery configuration required")
        if (
            self.delay_sec > 60
            or self.minimum_clearance_m > 100
            or self.backward_tolerance_mps > 100
            or self.angular_tolerance_radps > 100
            or self.tilt_tolerance_rad > math.pi
            or max(self.backward_weight, self.angular_weight, self.tilt_weight) > 1000
            or not 0 < self.dt_sec <= 1
            or not 0 < self.per_step_cap <= 1000
        ):
            raise ValueError("bounded recovery configuration required")


@dataclass(frozen=True)
class PostEventRecoveryCost:
    cost: NDArray[np.float64]
    enabled: NDArray[np.bool_]


def post_event_recovery_cost(
    *,
    event_observed: NDArray[np.bool_],
    event_age_sec: NDArray[np.floating],
    clearance_m: NDArray[np.floating],
    forward_velocity_mps: NDArray[np.floating],
    angular_speed_radps: NDArray[np.floating],
    upright_cosine: NDArray[np.floating],
    config: PostEventRecoveryConfig | None = None,
) -> PostEventRecoveryCost:
    """Penalize excess backward motion, angular speed and tilt after clearance.

    Forward/lateral motion is not directly penalized. Event age must be finite
    even before an event (use zero with event_observed=False). The caller owns
    the episode-fixed forward frame; do not flip it when an object passes its
    target. Invalid observations raise before any cost is returned. This is
    an offline training diagnostic, not a safety controller or success gate.
    """
    if config is not None and not isinstance(config, PostEventRecoveryConfig):
        raise ValueError("declared recovery configuration required")
    active = config or PostEventRecoveryConfig()
    if (
        not isinstance(event_observed, np.ndarray)
        or event_observed.dtype != np.bool_
        or event_observed.ndim != 1
        or not 1 <= len(event_observed) <= 65536
    ):
        raise ValueError("bounded one-dimensional boolean event mask required")
    values = (
        event_age_sec,
        clearance_m,
        forward_velocity_mps,
        angular_speed_radps,
        upright_cosine,
    )
    if any(
        not isinstance(value, np.ndarray)
        or value.shape != event_observed.shape
        or value.dtype not in (np.dtype("float32"), np.dtype("float64"))
        or not np.isfinite(value).all()
        or (np.abs(value) > 1e6).any()
        for value in values
    ):
        raise ValueError("matching bounded finite scalar recovery observations required")
    age, clearance, forward, angular, upright = (value.astype(np.float64) for value in values)
    if (
        (age < 0).any()
        or (clearance < 0).any()
        or (angular < 0).any()
        or (np.abs(upright) > 1 + 1e-5).any()
    ):
        raise ValueError("physical recovery magnitudes and upright cosine required")
    enabled = (
        event_observed & (age > active.delay_sec + 1e-8) & (clearance > active.minimum_clearance_m)
    )
    backward_excess = np.maximum(0.0, -forward - active.backward_tolerance_mps)
    angular_excess = np.maximum(0.0, angular - active.angular_tolerance_radps)
    tilt_excess = np.maximum(
        0.0, np.arccos(np.clip(upright, -1.0, 1.0)) - active.tilt_tolerance_rad
    )
    cost = np.where(
        enabled,
        np.minimum(
            active.per_step_cap,
            active.dt_sec
            * (
                active.backward_weight * backward_excess**2
                + active.angular_weight * angular_excess**2
                + active.tilt_weight * tilt_excess**2
            ),
        ),
        0.0,
    )
    cost.flags.writeable = False
    enabled.flags.writeable = False
    return PostEventRecoveryCost(cost=cost, enabled=enabled)
