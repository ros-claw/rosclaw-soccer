"""Signed sphere/capsule surface separation using unchanged compiled geometry."""

import numpy as np


def sphere_capsule_clearance(
    sphere: np.ndarray,
    radius: float,
    center: np.ndarray,
    axis: np.ndarray,
    half_length: float,
    capsule_radius: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    if (
        any(v.shape != (3,) or not np.isfinite(v).all() for v in (sphere, center, axis))
        or not np.isfinite([radius, half_length, capsule_radius]).all()
        or min(radius, half_length, capsule_radius) <= 0
        or not np.isclose(np.linalg.norm(axis), 1, atol=1e-8)
    ):
        raise ValueError("finite compiled sphere/capsule and unit world axis required")
    point = center + axis * np.clip((sphere - center) @ axis, -half_length, half_length)
    outward = point - sphere
    distance = np.linalg.norm(outward)
    if distance < 1e-10:
        raise ValueError("degenerate centerline overlap needs explicit planner failure")
    return (
        float(distance - radius - capsule_radius),
        np.asarray(point),
        np.asarray(outward / distance),
    )
