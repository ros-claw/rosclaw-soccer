"""Decode explicit per-body motor ownership without a scalar winner fiction."""

from __future__ import annotations

from typing import Any

import numpy as np


def per_player_motor_columns(
    trace: dict[str, Any], agent_id: str, *, agent_code: int, frames: int
) -> tuple[np.ndarray, np.ndarray] | None:
    if "per_player_motor_contract" not in trace:
        return None
    contract = np.asarray(trace["per_player_motor_contract"])
    key = agent_id.replace(".", "_")
    required = (key + "_motor_option_active", key + "_motor_option_target_m")
    if not all(name in trace for name in required):
        raise ValueError("incomplete per-player motor ownership trace")
    active, target = (np.asarray(trace[name]) for name in required)
    if (
        type(agent_code) is not int
        or agent_code <= 0
        or contract.shape != (frames,)
        or contract.dtype != np.bool_
        or not contract.all()
        or active.shape != (frames,)
        or active.dtype != np.bool_
        or target.shape != (frames, 3)
        or not np.isfinite(target).all()
        or np.any(target[~active] != 0.0)
    ):
        raise ValueError("finite explicit per-player motor ownership required")
    return np.where(active, agent_code, 0), target
