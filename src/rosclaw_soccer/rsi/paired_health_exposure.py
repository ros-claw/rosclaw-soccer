"""Compare actual body failures over the same observed native horizon."""

import numpy as np


def paired_health_at_common_horizon(raw_failed: np.ndarray, executed_failed: np.ndarray) -> dict:
    for history in (raw_failed, executed_failed):
        if (
            not isinstance(history, np.ndarray)
            or history.ndim != 2
            or history.shape[1] != 2
            or len(history) == 0
            or history.dtype != np.bool_
        ):
            raise ValueError("two nonempty actual boolean two-body histories required")
    horizon = min(len(raw_failed), len(executed_failed))
    return dict(
        common_controls=horizon,
        raw_controls=len(raw_failed),
        executed_controls=len(executed_failed),
        equal_exposure=len(raw_failed) == len(executed_failed),
        raw_healthy_common=not bool(raw_failed[:horizon].any()),
        executed_healthy_common=not bool(executed_failed[:horizon].any()),
        raw_healthy_full=not bool(raw_failed.any()),
        executed_healthy_full=not bool(executed_failed.any()),
        raw_failure_by_body_common=raw_failed[:horizon].any(axis=0).tolist(),
        executed_failure_by_body_common=executed_failed[:horizon].any(axis=0).tolist(),
    )
