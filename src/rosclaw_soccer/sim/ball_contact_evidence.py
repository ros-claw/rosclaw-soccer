"""Microstep body-contact credit for simulated football learning."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class BallContactSummary:
    first_foot_microstep: int | None
    first_nonfoot_microstep: int | None
    foot_first: bool
    clean_foot_only: bool
    foot_contact_microsteps: int
    nonfoot_contact_microsteps: int


def classify_ball_body_contacts(
    force_micro_n: NDArray[np.float64], *, threshold_n: float = 1.0
) -> BallContactSummary:
    """Classify 500 Hz physics contacts without inferring causality from pixels.

    The first two columns are anatomical feet. Remaining columns are other
    named body links. A foot-first impact can still be contaminated by a
    subsequent knee/shin impact; only ``clean_foot_only`` excludes that.
    """
    force = np.asarray(force_micro_n)
    if (
        force.ndim != 3
        or force.shape[0] < 1
        or force.shape[1] < 1
        or force.shape[2] < 3
        or force.dtype.kind not in "fiu"
        or not np.isfinite(force).all()
        or np.any(force < 0)
        or not math.isfinite(threshold_n)
        or threshold_n <= 0
    ):
        raise ValueError("finite nonnegative microstep force array and threshold required")
    flattened = force.reshape(-1, force.shape[2])
    foot = np.max(flattened[:, :2], axis=1) > threshold_n
    nonfoot = np.max(flattened[:, 2:], axis=1) > threshold_n
    foot_indices = np.flatnonzero(foot)
    nonfoot_indices = np.flatnonzero(nonfoot)
    first_foot = int(foot_indices[0]) if foot_indices.size else None
    first_nonfoot = int(nonfoot_indices[0]) if nonfoot_indices.size else None
    return BallContactSummary(
        first_foot_microstep=first_foot,
        first_nonfoot_microstep=first_nonfoot,
        foot_first=bool(
            first_foot is not None and (first_nonfoot is None or first_foot < first_nonfoot)
        ),
        clean_foot_only=bool(first_foot is not None and first_nonfoot is None),
        foot_contact_microsteps=int(np.count_nonzero(foot)),
        nonfoot_contact_microsteps=int(np.count_nonzero(nonfoot)),
    )
