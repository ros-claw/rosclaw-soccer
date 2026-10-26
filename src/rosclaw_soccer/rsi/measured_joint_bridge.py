"""Bounded C1 sampled reference between a measured joint state and a stance."""

from typing import Any

import numpy as np


def sampled_hermite_joint_bridge(
    measured_position: np.ndarray,
    measured_velocity: np.ndarray,
    terminal_position: np.ndarray,
    limits: np.ndarray,
    *,
    controls: int,
    control_interval_s: float,
    maximum_sampled_speed_rad_s: float,
) -> dict[str, Any]:
    """Build an unexecuted C1 joint reference, rejecting sampled bound violations.

    The first reference has the measured position and velocity; the last has
    the declared terminal position and zero velocity. This is not a contact,
    whole-body dynamics, torque or physical stability certificate. Sampled
    checks alone do not prove the continuous curve is within limits.
    """
    vectors = (measured_position, measured_velocity, terminal_position)
    if (
        any(
            type(x) is not np.ndarray
            or x.shape != (29,)
            or x.dtype.kind != "f"
            or not np.isfinite(x).all()
            for x in vectors
        )
        or type(limits) is not np.ndarray
        or limits.shape != (29, 2)
        or limits.dtype.kind != "f"
        or not np.isfinite(limits).all()
        or np.any(limits[:, 0] >= limits[:, 1])
        or type(controls) is not int
        or not 2 <= controls <= 100
        or type(control_interval_s) is not float
        or not np.isfinite(control_interval_s)
        or not 0.001 <= control_interval_s <= 0.1
        or type(maximum_sampled_speed_rad_s) is not float
        or not np.isfinite(maximum_sampled_speed_rad_s)
        or not 0 < maximum_sampled_speed_rad_s <= 20
    ):
        raise ValueError(
            "finite HG29 boundary states, bounds and explicit finite schedule required"
        )
    duration = controls * control_interval_s
    u = np.linspace(0.0, 1.0, controls + 1, dtype=np.float64)[:, None]
    h00, h10, h01 = 2 * u**3 - 3 * u**2 + 1, u**3 - 2 * u**2 + u, -2 * u**3 + 3 * u**2
    dh00, dh10, dh01 = 6 * u**2 - 6 * u, 3 * u**2 - 4 * u + 1, -6 * u**2 + 6 * u
    q0, dq0, q1 = [x.astype(np.float64) for x in vectors]
    positions = h00 * q0 + h10 * duration * dq0 + h01 * q1
    velocities = (dh00 * q0 + dh10 * duration * dq0 + dh01 * q1) / duration
    if (
        not np.isfinite(positions).all()
        or not np.isfinite(velocities).all()
        or np.any(positions < limits[:, 0])
        or np.any(positions > limits[:, 1])
        or np.max(np.abs(velocities)) > maximum_sampled_speed_rad_s
    ):
        raise ValueError("sampled joint position or speed bounds exceeded; no clipping")
    return dict(
        position=positions,
        velocity=velocities,
        controls=controls,
        duration_s=duration,
        max_sampled_speed_rad_s=float(np.max(np.abs(velocities))),
        minimum_sampled_joint_margin_rad=float(
            np.min(np.minimum(positions - limits[:, 0], limits[:, 1] - positions))
        ),
        continuous_joint_bound_not_proven=True,
        contact_or_dynamics_not_qualified=True,
        hardware_authorized=False,
    )
