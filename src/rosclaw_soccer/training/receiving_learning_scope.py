"""Causal, opt-in tactical eligibility for sampled on-policy receiving actions.

This is a gradient mask, not a contact, capture, safety, or promotion label.
Returns and advantages must still be computed from the complete episode.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np

from rosclaw_soccer.growth.role_self_model import TacticalIntent


def receiving_learning_scope(
    trace: Mapping[str, Any], *, agent_ids: tuple[str, ...]
) -> dict[str, Any]:
    """Mask gradient rows to the current recorded receive/press/intercept intent.

    Exploration and physical activation remain independently enforced by PPO;
    no future contact, reward, success or outcome enters the mask. The input
    trajectory is never changed and existing learning masks cannot be replaced.
    """
    if "residual_learning_mask" in trace:
        raise ValueError("refusing to replace a recorded learning mask")
    time = np.asarray(trace["time"])
    if (
        time.ndim != 1
        or not len(time)
        or not agent_ids
        or tuple(sorted(set(agent_ids))) != agent_ids
    ):
        raise ValueError("canonical roster and timeline required")
    allowed = [
        list(TacticalIntent).index(intent)
        for intent in (TacticalIntent.RECEIVE, TacticalIntent.PRESS, TacticalIntent.INTERCEPT)
    ]
    columns = []
    for agent in agent_ids:
        intent = np.asarray(trace[agent.replace(".", "_") + "_intent_code"])
        if (
            intent.shape != time.shape
            or intent.dtype.kind not in "iu"
            or not np.isin(intent, range(len(TacticalIntent))).all()
        ):
            raise ValueError("aligned, valid recorded intent codes required")
        columns.append(np.isin(intent, allowed))
    return {**trace, "residual_learning_mask": np.stack(columns, axis=1)}
