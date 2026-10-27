"""Conservatively derive a SIM-only neighborhood radius from failure data."""

import math


def radius_below_failure(distance: float, *, prior_radius: float) -> float:
    """Use a 50-micro-unit grid strictly below the nearest measured failure.

    This is a development heuristic, not a generalization or safety guarantee.
    """
    if (
        type(distance) is not float
        or type(prior_radius) is not float
        or not math.isfinite(distance)
        or not math.isfinite(prior_radius)
        or not 0 < distance <= prior_radius <= 0.004
    ):
        raise ValueError("finite in-support failure and bounded previous radius required")
    step = 0.00005
    units = math.floor(distance / step)
    radius = units * step
    if radius >= distance:
        radius = (units - 1) * step
    if radius <= 0 or radius >= prior_radius:
        raise ValueError("no positive strictly smaller grid radius available")
    return float(radius)
