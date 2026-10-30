"""Measured native football contact summaries; not force attribution or safety."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray


def first_foot_impact_observation(
    ball_linear_velocity: NDArray[Any],
    grouped_force_magnitude: NDArray[Any],
    *,
    timestep_sec: float,
    ball_mass_kg: float,
    contact_threshold_n: float = 1.0,
) -> dict[str, Any]:
    """Summarize the first contiguous above-threshold foot-force interval.

    Each row is a post-physics-step ball velocity and the matching native
    substep's measured force channels [left foot, right foot, non-foot].
    The runtime stores the largest individual contact-force norm in each
    channel, NOT a vector sum. Its scalar time integral is therefore a useful
    descriptor but is not the ball impulse, not a certified force bound, and
    cannot attribute ball momentum to any particular contact. Ball momentum
    change also includes ground and any other concurrent forces. A sample at
    contact onset requires the immediately preceding saved substep; terminal
    contact without a following non-contact substep is right-censored.
    """
    velocity = np.asarray(ball_linear_velocity)
    force = np.asarray(grouped_force_magnitude)
    scalars = (timestep_sec, ball_mass_kg, contact_threshold_n)
    if (
        velocity.ndim != 2
        or velocity.shape[1:] != (3,)
        or force.shape != (len(velocity), 3)
        or len(velocity) < 1
        or velocity.dtype.kind != "f"
        or force.dtype.kind != "f"
        or not np.isfinite(velocity).all()
        or not np.isfinite(force).all()
        or np.any(force < 0)
        or any(
            isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v)
            for v in scalars
        )
        or not 0 < timestep_sec <= 0.02
        or not 0 < ball_mass_kg <= 10
        or not 0 < contact_threshold_n <= 100
    ):
        raise ValueError("finite aligned native observations and physical units required")
    mask = np.max(force[:, :2], axis=1) > contact_threshold_n
    hits = np.flatnonzero(mask)
    common: dict[str, Any] = {
        "schema": "soccer.first_foot_impact_observation.v1",
        "contact_threshold_n": contact_threshold_n,
        "timestep_sec": timestep_sec,
        "ball_mass_kg": ball_mass_kg,
        "force_channels": "per-part maximum individual contact force norm",
        "force_time_integral_is_ball_impulse": False,
        "ball_momentum_change_attributed_to_foot": False,
        "rotational_energy_included": False,
        "physical_safety_proven": False,
        "hardware_authorized": False,
        "promotion_authorized": False,
    }
    if not len(hits):
        return {**common, "status": "NO_OBSERVED_FOOT_CONTACT", "first_substep": None}
    first = int(hits[0])
    if first == 0:
        return {**common, "status": "PRECONTACT_CENSORED", "first_substep": first}
    end = first
    while end + 1 < len(mask) and mask[end + 1]:
        end += 1
    segment = force[first : end + 1]
    v0, v1 = velocity[first - 1], velocity[end]
    before = 0.5 * ball_mass_kg * float(np.dot(v0, v0))
    after = 0.5 * ball_mass_kg * float(np.dot(v1, v1))
    return {
        **common,
        "status": "POSTCONTACT_CENSORED" if end == len(mask) - 1 else "OBSERVED",
        "first_substep": first,
        "last_substep": end,
        "duration_sec": (end - first + 1) * timestep_sec,
        "peak_foot_force_n": float(np.max(segment[:, :2])),
        "max_channel_force_time_integral_ns": float(
            np.sum(np.max(segment[:, :2], axis=1)) * timestep_sec
        ),
        "concurrent_nonfoot_contact": bool(np.any(segment[:, 2] > contact_threshold_n)),
        "ball_velocity_before_mps": v0.tolist(),
        "ball_velocity_after_mps": v1.tolist(),
        "ball_linear_momentum_change_kg_mps": (ball_mass_kg * (v1 - v0)).tolist(),
        "ball_translational_energy_change_j": after - before,
    }
