"""Football-prior reference lookup only; never a simulator or actuator clock.

Physical integration remains 500Hz, observation/action history remains50Hz.
Changing the reference rate defines a new parent controller, not faster replay
of saved actions and not permission to rewrite the robot's state.
"""

import math
from fractions import Fraction

import numpy as np


def reference_policy_index(frame: int, *, warmup: int, rate: float) -> int:
    if (
        type(frame) is not int
        or not 0 <= frame <= 100000
        or type(warmup) is not int
        or not 0 <= warmup <= 1000
        or not math.isfinite(rate)
        or not 1 <= rate <= 1.2
    ):
        raise ValueError(
            "bounded consecutive physical frame and mild reference acceleration required"
        )
    if frame < warmup:
        return frame
    ratio = Fraction(str(rate))
    return warmup + (frame - warmup) * ratio.numerator // ratio.denominator


def retimed_reference_velocity(original: np.ndarray, *, rate: float) -> np.ndarray:
    value = np.asarray(original)
    if (
        value.ndim != 2
        or value.shape[1] != 29
        or not 1 <= len(value) <= 100000
        or value.dtype.kind != "f"
        or not np.isfinite(value).all()
        or not math.isfinite(rate)
        or not 1 <= rate <= 1.2
    ):
        raise ValueError("finite HG-independent original29-joint reference velocities required")
    result = value.copy() if rate == 1 else value * np.asarray(rate, dtype=value.dtype)
    if not np.isfinite(result).all():
        raise ValueError("retimed reference overflow")
    return result


def declared_clock_starts(plan: dict) -> list[tuple[float, float]]:
    """Four legacy anchors first; expanded public development domain is explicit."""
    value = np.asarray(plan["starts_m"], dtype=np.float64)
    original = np.array([[x, y] for x in (0.97, 1.03) for y in (-0.03, 0.03)])
    if (
        value.shape not in ((4, 2), (16, 2))
        or not np.isfinite(value).all()
        or not np.array_equal(value[:4], original)
        or len(np.unique(value, axis=0)) != len(value)
        or np.any((value[:, 0] < 0.94) | (value[:, 0] > 1.06))
        or np.any(np.abs(value[:, 1]) > 0.06)
    ):
        raise ValueError(
            "exact four anchors plus bounded unique public development starts required"
        )
    return [(float(x), float(y)) for x, y in value]
