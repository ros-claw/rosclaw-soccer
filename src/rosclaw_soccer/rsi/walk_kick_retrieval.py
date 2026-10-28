"""Explicitly experimental retrieval baseline for public first-touch teachers.

This is an offline supervised lookup, not a qualified policy or a private exam.
Only initial observable course features are accepted at inference.
"""

import numpy as np


def prospective_courses(*, extended=False):
    """Fixed public diagnostic banks balanced across the original four classes."""
    from rosclaw_soccer.training.receiving_course_preflight import FirstTouchCourse

    if type(extended) is not bool:
        raise ValueError("explicit diagnostic bank required")
    seed = 2026102930 if extended else 2026102929
    rng = np.random.default_rng(seed)
    courses = []
    for bucket in range(4):
        for _ in range(10 if extended else 2):
            x = float(rng.uniform(2.2 if bucket < 2 else 2.6, 2.4 if bucket < 2 else 2.8))
            y = float(rng.uniform(-0.16, 0.16))
            vx = float(rng.uniform(0.25, 0.7)) * (1 if bucket % 2 else -1)
            courses.append(FirstTouchCourse("striker", seed + len(courses), (x, y, vx)))
    return tuple(courses)


def nearest_teacher(query, courses, eligible):
    query = np.asarray(query, dtype=np.float64)
    courses = np.asarray(courses, dtype=np.float64)
    eligible = np.asarray(eligible)
    if (
        query.shape != (3,)
        or courses.ndim != 2
        or courses.shape[1] != 3
        or eligible.shape != (len(courses),)
        or eligible.dtype != np.bool_
        or not np.isfinite(query).all()
        or not np.isfinite(courses).all()
        or not eligible.any()
    ):
        raise ValueError("finite course and nonempty explicitly eligible teachers required")
    # Fixed physical scales; no outcome-dependent normalization or tuning.
    scale = np.array([0.5, 0.3, 0.8])
    distances = np.sum(((courses - query) / scale) ** 2, axis=1)
    distances[~eligible] = np.inf
    return int(np.argmin(distances))
