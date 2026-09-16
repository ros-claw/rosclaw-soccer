"""Explicit training-assist lifecycle segmentation, never reward coach resets."""

from collections.abc import Mapping
from typing import Any

import numpy as np


def _codes(trace: Mapping[str, Any]) -> np.ndarray:
    time = np.asarray(trace["time"])
    codes = np.asarray(trace["training_return_event_code"])
    if (
        time.ndim != 1
        or len(time) == 0
        or time.dtype.kind not in "fiu"
        or not np.all(np.isfinite(time))
        or time[0] < 0
        or np.any(np.diff(time.astype(float)) <= 0)
        or codes.shape != time.shape
        or codes.dtype.kind not in "iu"
        or not np.all(np.isin(codes, [0, 1, 2, 3]))
    ):
        raise ValueError("finite monotonic clock and aligned integer return events required")
    return codes


def require_returned_live_segment(trace: Mapping[str, Any]) -> None:
    """Reject reward calculation across an external ball release or dead-ball gap."""
    if "training_return_event_code" not in trace:
        return  # Legacy, unassisted rollouts retain their original semantics.
    codes = _codes(trace)
    if codes[0] != 3 or np.any(codes[1:] != 0):
        raise ValueError("reward requires one observed returned-live segment without reset")


def returned_ball_segments(trace: Mapping[str, Any]) -> list[dict[str, np.ndarray]]:
    """Copy measured re-entry→next exit epochs for episodic advantage calculation.

    The first attack and all pending/airborne external throws are excluded.
    A failed throw may exit again without re-entry. Lifecycle errors fail closed;
    this validates recorded events, not the underlying physical boundary geometry.
    Every column must have the same leading frame dimension.
    """
    codes = _codes(trace)
    arrays = {key: np.asarray(value) for key, value in trace.items()}
    if any(v.ndim == 0 or len(v) != len(codes) for v in arrays.values()):
        raise ValueError("all learning columns must be frame-aligned")
    phase = "initial"
    start: int | None = None
    windows = []
    for frame, code in enumerate(codes):
        if code == 0:
            continue
        if code == 1 and phase in {"initial", "live", "released"}:
            if start is not None:
                windows.append((start, frame))
                start = None
            phase = "pending"
        elif code == 2 and phase == "pending":
            phase = "released"
        elif code == 3 and phase == "released":
            phase, start = "live", frame
        else:
            raise ValueError("invalid observed training-return lifecycle")
    if start is not None:
        windows.append((start, len(codes)))
    return [{k: v[a:b].copy() for k, v in arrays.items()} for a, b in windows]
