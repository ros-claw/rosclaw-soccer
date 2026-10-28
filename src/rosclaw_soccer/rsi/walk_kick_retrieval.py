"""Explicitly experimental retrieval baseline for public first-touch teachers.

This is an offline supervised lookup, not a qualified policy or a private exam.
Only initial observable course features are accepted at inference.
"""

import numpy as np


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
