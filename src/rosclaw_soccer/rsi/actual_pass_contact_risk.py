"""Prepare causal, grouped actual-pass contact prediction data for SIM research."""

from collections.abc import Mapping

import numpy as np


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
    if values["course_index"].dtype.kind not in "iu" or len(courses) not in (32, 72):
        raise ValueError("integer course indices and a declared 32- or 72-course bank required")
    context_by_course = {entry["index"]: entry["course"]["context"] for entry in courses}
    if (
        len(context_by_course) != len(courses)
        or set(context_by_course) != set(range(len(courses)))
        or set(context_by_course.values()) != {1, 4, 20, 21}
    ):
        raise ValueError("unique courses across four recorded context groups required")
    if not np.isin(values["course_index"], list(context_by_course)).all():
        raise ValueError("only declared course identities may appear in transitions")
    groups = np.asarray([context_by_course[int(i)] for i in values["course_index"]], dtype=np.int64)
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
