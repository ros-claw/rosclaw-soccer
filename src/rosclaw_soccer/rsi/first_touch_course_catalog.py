"""Disjoint SIM_ONLY first-touch course families; never unlocks held-out evaluation."""

from __future__ import annotations

import math

import numpy as np


def static_development_courses(count: int) -> tuple[tuple[float, float, float], ...]:
    """Preserve the historical eight lanes; extend to sixteen unique lanes."""
    if type(count) is not int or count not in {2, 4, 8, 16}:
        raise ValueError("supported independent lane count must be 2, 4, 8 or 16")
    x_values = (2.4, 2.6, 2.3, 2.7)
    return tuple(
        (
            x_values[i // 4],
            -0.1 if (i // 2) % 2 == 0 else 0.1,
            -0.5 if i % 2 == 0 else 0.5,
        )
        for i in range(count)
    )


def sample_training_courses(seed: int, count: int = 16) -> tuple[tuple[float, float, float], ...]:
    """Stratified train-only moving balls, separated from the sealed x=2.45/2.55 tests.

    Every draw uses its own x stratum. The x gap avoids the sealed evaluation
    locations even if y and velocity coincide. This is course generation, not
    evidence that a policy has learned or generalized.
    """
    if type(seed) is not int or not 0 <= seed < 2**32 or type(count) is not int or count != 16:
        raise ValueError("bounded 16-lane training seed required")
    rng = np.random.default_rng(seed)
    x_strata = np.linspace(2.2, 2.4, 8, endpoint=False)
    x_strata = np.concatenate((x_strata, np.linspace(2.6, 2.8, 8, endpoint=False)))
    jitter = rng.uniform(0.0, 0.025, count)
    x = x_strata + jitter
    y = (np.arange(count) + rng.uniform(0.0, 1.0, count)) / count * 0.32 - 0.16
    rng.shuffle(y)
    vx_magnitude = rng.uniform(0.25, 0.7, count)
    vx = vx_magnitude * np.where(np.arange(count) % 2 == 0, -1.0, 1.0)
    courses = tuple((float(a), float(b), float(c)) for a, b, c in zip(x, y, vx, strict=True))
    if (
        len(set(courses)) != count
        or any(not all(math.isfinite(value) for value in course) for course in courses)
        or any(2.4 < course[0] < 2.6 for course in courses)
    ):
        raise RuntimeError("training split generator violated disjointness")
    return courses
