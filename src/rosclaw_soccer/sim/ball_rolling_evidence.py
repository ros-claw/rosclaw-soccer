"""Auditable pre-impact ball rolling diagnostic for Isaac contact courses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class RollingEvidence:
    grounded_frames: int
    mean_slip_m_s: float | None
    maximum_slip_m_s: float | None


def precontact_rolling_evidence(
    position_m: NDArray[np.float64],
    linear_velocity_m_s: NDArray[np.float64],
    angular_velocity_rad_s: NDArray[np.float64],
    *,
    first_body_impact_microstep: int | None,
    physics_steps_per_frame: int = 10,
    radius_m: float = 0.11,
) -> RollingEvidence:
    """Measure ground-contact slip without treating post-kick flight as rolling.

    A ball's bottom point has planar velocity ``v - r*(omega_y, -omega_x)``.
    Only frames before the first measured robot contact and with the center
    near ground height contribute; no such frames yields no rolling claim.
    """
    position = np.asarray(position_m, dtype=np.float64)
    velocity = np.asarray(linear_velocity_m_s, dtype=np.float64)
    angular = np.asarray(angular_velocity_rad_s, dtype=np.float64)
    if (
        position.ndim != 2
        or position.shape[1] != 3
        or len(position) < 2
        or velocity.shape != position.shape
        or angular.shape != position.shape
        or not all(np.isfinite(a).all() for a in (position, velocity, angular))
        or not np.isfinite(radius_m)
        or not 0.05 <= radius_m <= 0.2
        or not 1 <= physics_steps_per_frame <= 100
        or (first_body_impact_microstep is not None and first_body_impact_microstep < 0)
    ):
        raise ValueError("finite aligned physical ball observations required")
    cutoff = (
        len(position)
        if first_body_impact_microstep is None
        else min(len(position), first_body_impact_microstep // physics_steps_per_frame)
    )
    grounded = (position[:cutoff, 2] >= radius_m - 0.005) & (
        position[:cutoff, 2] <= radius_m + 0.01
    )
    if not np.any(grounded):
        return RollingEvidence(0, None, None)
    bottom_velocity = velocity[:cutoff, :2] - radius_m * np.stack(
        (angular[:cutoff, 1], -angular[:cutoff, 0]), axis=1
    )
    slip = np.linalg.norm(bottom_velocity[grounded], axis=1)
    return RollingEvidence(int(np.count_nonzero(grounded)), float(slip.mean()), float(slip.max()))
