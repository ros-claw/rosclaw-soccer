"""Read-only same-observation action drift, not a task-success estimator."""

from typing import Any

import numpy as np


def token_drift(before: np.ndarray, after: np.ndarray) -> dict[str, Any]:
    for x in (before, after):
        if (
            x.shape != (300, 64)
            or not np.isfinite(x).all()
            or not np.array_equal(x * 16, np.round(x * 16))
            or (x < -1).any()
            or (x > 0.9375).any()
        ):
            raise ValueError("complete finite original FSQ32 H300x64 tokens required")
    changed = before[20:] != after[20:]
    frames = np.flatnonzero(changed.any(axis=1))
    return dict(
        suffix_frames=280,
        changed_frames=int(changed.any(axis=1).sum()),
        changed_dimensions=int(changed.sum()),
        first_changed_control_frame=int(frames[0] + 20) if len(frames) else None,
        maximum_token_delta=float(np.abs(before[20:] - after[20:]).max()),
        diagnostic_only=True,
        task_improvement_inferred=False,
    )


def first_physical_divergence(before: np.ndarray, after: np.ndarray) -> int | None:
    if (
        before.shape != after.shape
        or before.ndim < 2
        or before.shape[0] != 300
        or not np.isfinite(before).all()
        or not np.isfinite(after).all()
    ):
        raise ValueError("complete paired finite native H300 arrays required")
    changed = np.flatnonzero((before != after).reshape(300, -1).any(axis=1))
    return int(changed[0]) if len(changed) else None
