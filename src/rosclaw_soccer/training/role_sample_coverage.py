"""Measure actor-eligible samples for bounded on-policy collection.

Counts are physical activation intersected with every explicit learning and
exploration mask. They do not assert task success or make a promotion decision.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def role_sample_coverage(
    traces: Sequence[Mapping[str, Any]], *, agent_ids: tuple[str, ...], required: tuple[str, ...]
) -> dict[str, int]:
    if (
        not traces
        or not agent_ids
        or tuple(sorted(set(agent_ids))) != agent_ids
        or not required
        or tuple(sorted(set(required))) != required
        or not set(required).issubset(agent_ids)
    ):
        raise ValueError("nonempty explicit canonical roster, roles and rollouts required")
    if any("residual_learning_mask" in trace for trace in traces) and not all(
        "residual_learning_mask" in trace for trace in traces
    ):
        raise ValueError("learning eligibility must be explicit on every rollout")
    totals = {agent: 0 for agent in required}
    for trace in traces:
        active = np.asarray(trace["residual_active"])
        if active.ndim != 2 or active.shape[1] != len(agent_ids) or active.dtype != np.bool_:
            raise ValueError("complete boolean physical activation required")
        mask = active.copy()
        for key in ("residual_exploration_mask", "residual_learning_mask"):
            if key in trace:
                value = np.asarray(trace[key])
                if value.shape != active.shape or value.dtype != np.bool_:
                    raise ValueError("aligned explicit boolean masks required")
                mask &= value
        for agent in required:
            totals[agent] += int(np.count_nonzero(mask[:, agent_ids.index(agent)]))
    return totals
