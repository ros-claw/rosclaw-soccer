"""Training-only support envelope for cross-world first-touch shadow probes."""

from __future__ import annotations

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.contact_time_phase_features import ACTION_FEATURE_NAMES
from rosclaw_soccer.rsi.local_phase_memory import DISTANCE_COLUMNS


def leave_one_out_support_radius(memories: np.ndarray[Any, Any]) -> float:
    """Conservative radius defined only by frozen training memories, not the query."""
    if (
        memories.ndim != 2
        or memories.shape[1] != len(ACTION_FEATURE_NAMES)
        or len(memories) < 3
        or not np.isfinite(memories).all()
    ):
        raise ValueError("finite frozen training memories required")
    selected = memories[:, np.asarray(DISTANCE_COLUMNS)]
    distances = np.linalg.norm(selected[:, None] - selected[None, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    return float(np.max(np.min(distances, axis=1)))


def query_support_distance(memories: np.ndarray[Any, Any], query: np.ndarray[Any, Any]) -> float:
    if (
        memories.ndim != 2
        or memories.shape[1] != len(ACTION_FEATURE_NAMES)
        or query.shape != (len(ACTION_FEATURE_NAMES),)
        or not np.isfinite(memories).all()
        or not np.isfinite(query).all()
    ):
        raise ValueError("finite memory and one query required")
    columns = np.asarray(DISTANCE_COLUMNS)
    return float(np.min(np.linalg.norm(memories[:, columns] - query[columns], axis=1)))
