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


def paired_health_from_first_failures(first: dict, second: dict) -> dict:
    """Use reviewed first-failure timestamps without inventing unseen states."""
    for arm in (first, second):
        n, failures = arm.get("observed_controls"), arm.get("first_body_failure_controls")
        if (
            type(n) is not int
            or n < 1
            or not isinstance(failures, list)
            or len(failures) != 2
            or any(x is not None and (type(x) is not int or not 0 <= x < n) for x in failures)
            or type(arm.get("outcome", {}).get("both_bodies_healthy")) is not bool
            or arm["outcome"]["both_bodies_healthy"] != all(x is None for x in failures)
        ):
            raise ValueError("complete consistent native first-failure timeline required")
    horizon = min(first["observed_controls"], second["observed_controls"])

    def healthy(arm):
        return all(x is None or x >= horizon for x in arm["first_body_failure_controls"])

    return dict(
        common_controls=horizon,
        equal_exposure=first["observed_controls"] == second["observed_controls"],
        first_healthy_common=healthy(first),
        second_healthy_common=healthy(second),
    )
