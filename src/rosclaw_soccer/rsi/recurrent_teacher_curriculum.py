"""Offline successful-sequence weights; labels never become actor inputs.

Callers authenticate physical eligibility. This module does not verify physics
or teach avoidance from failed episodes; their weights remain exactly zero.
"""

from collections import Counter
from typing import Any

import numpy as np


def sequence_teacher_weights(
    records: list[dict[str, Any]], *, profile: str
) -> np.ndarray[Any, Any]:
    if (
        type(records) is not list
        or not 4 <= len(records) <= 740
        or profile not in ("balanced_success", "balanced_contact8_recovery20")
    ):
        raise ValueError("bounded complete offline sequence declaration required")
    eligible = []
    contexts = []
    for index, row in enumerate(records):
        if (
            type(row) is not dict
            or type(row.get("group")) is not int
            or row["group"] != index
            or type(row.get("seed")) is not int
            or not 0 <= row["seed"] < 2**32
            or type(row.get("lane")) is not int
            or not 0 <= row["lane"] < 16
            or any(type(row.get(k)) is not bool for k in ("high_quality", "safety_passed"))
        ):
            raise ValueError("ordered complete physical-label declaration required")
        keep = row["high_quality"] and row["safety_passed"]
        frame = row.get("first_contact_frame")
        if (frame is not None and (type(frame) is not int or not 0 <= frame < 300)) or (
            keep and frame is None
        ):
            raise ValueError("bounded actual first-contact label required")
        eligible.append(keep)
        contexts.append((row["seed"], row["lane"]))
    if sum(eligible) < 4:
        raise ValueError("at least four complete safe successful teachers required")
    counts = Counter(context for context, keep in zip(contexts, eligible, strict=True) if keep)
    weights = np.zeros((len(records), 270), dtype=np.float64)
    frames = np.arange(30, 300)
    for index, (row, keep, context) in enumerate(zip(records, eligible, contexts, strict=True)):
        if not keep:
            continue
        weights[index] = 1.0
        if profile == "balanced_contact8_recovery20":
            first = row["first_contact_frame"]
            window = (frames >= first - 8) & (frames < first + 20)
            weights[index, window] *= 8.0
        # A clipped event window at either episode boundary must not silently
        # change the total mass assigned to that teacher's context.
        weights[index] /= weights[index].mean() * counts[context]
    # Core normalizes positive weights again. Keep a canonical local scale for
    # inspection: equal-context teachers, not extra independent trajectories.
    weights /= weights[weights > 0].mean()
    return weights
