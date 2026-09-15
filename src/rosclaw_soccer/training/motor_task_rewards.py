"""Bind contact credit to the admitted motor task, not a changing planner target."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def motor_task_targets(
    planner_targets: NDArray[np.float64],
    option_targets: NDArray[np.float64],
    option_agent_codes: NDArray[np.int64],
    *,
    agent_code: int,
) -> NDArray[np.float64]:
    """Use recorded option targets only for their actual physical controller.

    No inferred future goal, mirrored team-name heuristic, or episode-success
    label is accepted. Missing option evidence must fail closed in the caller.
    """
    count = len(planner_targets)
    if (
        type(agent_code) is not int
        or not 1 <= agent_code <= 8
        or planner_targets.shape != (count, 3)
        or option_targets.shape != (count, 3)
        or option_agent_codes.shape != (count,)
        or not np.issubdtype(option_agent_codes.dtype, np.integer)
        or np.any((option_agent_codes < 0) | (option_agent_codes > 8))
        or not np.isfinite(planner_targets).all()
        or not np.isfinite(option_targets).all()
    ):
        raise ValueError("finite bound motor-task target evidence required")
    return np.where((option_agent_codes == agent_code)[:, None], option_targets, planner_targets)
