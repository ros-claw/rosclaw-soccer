"""Prepare causal, grouped actual-pass contact prediction data for SIM research."""

from collections.abc import Mapping

import numpy as np


def context_shard(contexts: tuple[int, ...], *, index: int, count: int) -> tuple[int, ...]:
    """Partition all context folds deterministically without seed selection."""
    if (
        type(contexts) is not tuple
        or len(contexts) < 4
        or any(type(v) is not int or v < 0 for v in contexts)
        or tuple(sorted(set(contexts))) != contexts
        or type(index) is not int
        or type(count) is not int
        or count not in (1, 4)
        or not 0 <= index < count
        or len(contexts) < count
    ):
        raise ValueError("sorted unique context groups and bounded shard identity required")
    return contexts[index::count]


def training_recall_threshold(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Select a <=0.5 threshold from training positives, never holdout labels."""
    p, y = np.asarray(probabilities), np.asarray(labels)
    if (
        p.ndim != 1
        or y.shape != p.shape
        or y.dtype != np.bool_
        or not np.isfinite(p).all()
        or np.any((p < 0) | (p > 1))
    ):
        raise ValueError("finite probabilities and explicit boolean labels required")
    positives = np.sort(p[y])
    if not len(positives):
        raise ValueError("training positives required")
    index = int(np.floor(0.05 * len(positives)))
    return float(min(0.5, positives[index]))


def contact_risk_data(
    data: Mapping[str, np.ndarray], courses: list[dict]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict measured receiver non-foot contact in the *next* 20 ms only.

    Context groups are held out together because different arms share starts.
    This is not a certified safety detector or an action/control interface.
    """
    required = (
        "qpos_before",
        "qvel_before",
        "receiver_observation",
        "receiver_action_applied",
        "ball_force_max_after",
        "course_index",
    )
    if any(k not in data for k in required):
        raise ValueError("complete causal transition arrays required")
    values = {k: np.asarray(data[k]) for k in required}
    n = len(values["course_index"])
    shapes = {
        "qpos_before": (n, 79),
        "qvel_before": (n, 76),
        "receiver_observation": (n, 98),
        "receiver_action_applied": (n, 29),
        "ball_force_max_after": (n, 2, 3),
        "course_index": (n,),
    }
    if not 1 <= n <= 1000000 or any(
        values[k].shape != s
        or values[k].dtype.kind not in "fiu"
        or not np.isfinite(values[k]).all()
        for k, s in shapes.items()
    ):
        raise ValueError("finite causal feature, action, outcome and course arrays required")
    if values["course_index"].dtype.kind not in "iu" or not 4 <= len(courses) <= 4096:
        raise ValueError("integer course indices and bounded declared course bank required")
    context_by_course = {entry["index"]: entry["course"]["context"] for entry in courses}
    if (
        len(context_by_course) != len(courses)
        or set(context_by_course) != set(range(len(courses)))
        or not 4 <= len(set(context_by_course.values())) <= 64
        or any(type(value) is not int or value < 0 for value in context_by_course.values())
    ):
        raise ValueError("unique courses across four recorded context groups required")
    if not np.isin(values["course_index"], list(context_by_course)).all():
        raise ValueError("only declared course identities may appear in transitions")
    if set(values["course_index"]) != set(context_by_course):
        raise ValueError("every declared course needs actual transition samples")
    groups = np.asarray([context_by_course[int(i)] for i in values["course_index"]], dtype=np.int64)
    if set(groups) != set(context_by_course.values()):
        raise ValueError("each declared context must have actual transition samples")
    x = np.concatenate(
        [
            values[k]
            for k in (
                "qpos_before",
                "qvel_before",
                "receiver_observation",
                "receiver_action_applied",
            )
        ],
        axis=1,
    ).astype(np.float32)
    y = (values["ball_force_max_after"][:, 1, 2] > 1.0).astype(np.float32)
    return x, y, groups
