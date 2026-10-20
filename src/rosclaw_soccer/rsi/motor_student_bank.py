"""Label-independent course split and training-only observation statistics."""

from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import hash_json


def validation_course_indices(courses: list[dict[str, Any]]) -> set[int]:
    if len(courses) != 40 or sorted(c["course_index"] for c in courses) != list(range(40)):
        raise ValueError("all40 unique course identities required")
    validation: set[int] = set()
    for speed, count, total in ((0.0, 4, 20), (-0.15, 2, 10), (0.15, 2, 10)):
        ids = [c["course_index"] for c in courses if c["course"][2] == speed]
        if len(ids) != total:
            raise ValueError("original20/10/10 speed strata required")
        ranked = sorted(
            ids,
            key=lambda i: hash_json(
                dict(operation="motor_student_course_split_v1", seed=20261020, course_index=i)
            ),
        )
        validation.update(ranked[:count])
    return validation


def training_observation_statistics(
    features: np.ndarray, train: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if (
        features.shape != (40, 300, 108)
        or features.dtype != np.float32
        or not np.isfinite(features).all()
        or train.shape != (40,)
        or train.dtype != np.bool_
        or int(train.sum()) != 32
    ):
        raise ValueError("finite40 observations and32 training course mask required")
    mean = features[train].mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = np.maximum(features[train].std(axis=(0, 1), dtype=np.float64), 0.05).astype(np.float32)
    return mean, std
