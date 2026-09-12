"""Read-only ideal flat-ground sphere estimates, never a motion authority."""

from __future__ import annotations

import math


def _vector(value: tuple[float, ...], size: int, bound: float) -> None:
    if (
        type(value) is not tuple
        or len(value) != size
        or any(type(x) not in (int, float) or not math.isfinite(x) or abs(x) > bound for x in value)
    ):
        raise ValueError("finite bounded vector required")


def sphere_angular_velocity_world(
    quaternion_wxyz: tuple[float, float, float, float],
    angular_velocity_local: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Rotate a free sphere's body-local angular velocity into world coordinates.

    MuJoCo free-joint rotational qvel is body-local, unlike its translational
    qvel. This helper accepts a unit quaternion; it does not repair corrupt
    observations or read/modify a simulator.
    """
    _vector(quaternion_wxyz, 4, 1.000001)
    _vector(angular_velocity_local, 3, 1000)
    if abs(sum(x * x for x in quaternion_wxyz) - 1) > 1e-6:
        raise ValueError("unit quaternion required")
    w, x, y, z = quaternion_wxyz
    a, b, c = angular_velocity_local
    # v' = v + 2 w (u cross v) + 2 u cross (u cross v).
    tx, ty, tz = 2 * (y * c - z * b), 2 * (z * a - x * c), 2 * (x * b - y * a)
    return a + w * tx + y * tz - z * ty, b + w * ty + z * tx - x * tz, c + w * tz + x * ty - y * tx


def ideal_planar_rolling_velocity(
    *,
    linear_velocity_world: tuple[float, float, float],
    angular_velocity_world: tuple[float, float, float],
    radius_m: float,
    inertia_ratio: float,
) -> tuple[float, float]:
    """Horizontal sliding-to-rolling limit for an isotropic sphere.

    Uses (v + kappa R (omega cross upward_normal)) / (1 + kappa), where
    kappa = I/(mass R**2). Assumes sustained flat-ground contact, positive
    sliding friction and no later external impulse. It ignores flight,
    bounce, damping, rolling resistance and obstacles. It predicts neither
    contact time nor an actual goal, and must not override observed physics.
    Vertical components are accepted for coordinate consistency but unused.
    """
    _vector(linear_velocity_world, 3, 100)
    _vector(angular_velocity_world, 3, 1000)
    if any(
        type(x) not in (int, float) or not math.isfinite(x) for x in (radius_m, inertia_ratio)
    ) or not (0.001 <= radius_m <= 1 and 0.01 <= inertia_ratio <= 1):
        raise ValueError("bounded positive sphere radius and inertia ratio required")
    vx, vy, _ = linear_velocity_world
    wx, wy, _ = angular_velocity_world
    return (
        (vx + inertia_ratio * radius_m * wy) / (1 + inertia_ratio),
        (vy - inertia_ratio * radius_m * wx) / (1 + inertia_ratio),
    )
