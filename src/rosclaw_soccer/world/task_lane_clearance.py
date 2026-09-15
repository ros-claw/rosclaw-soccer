"""A teammate's task-space reservation, subordinate to all physical clearances."""

from __future__ import annotations

import math

import numpy as np


def stance_reservation_allowed(*, intent: str, post_receive_hold: bool) -> bool:
    """A soft support objective cannot preempt receiving, recovery or active skills."""
    if not isinstance(intent, str) or type(post_receive_hold) is not bool:
        raise ValueError("explicit tactical intent and receive protection required")
    return not post_receive_hold and intent in {"support", "run_in_behind", "cover"}


def stance_clearance_halfplane(
    player_xy: np.ndarray,
    reserved_stance_xy: np.ndarray,
    fallback_outward: np.ndarray,
    *,
    clearance_m: float,
) -> np.ndarray | None:
    """Propose an outward velocity halfplane; never move a body or relax a guard.

    The caller applies this only to the preparing player's own teammates.
    A downstream simultaneous clearance solver may reject the intersection.
    """
    if (
        any(
            v.shape != (2,) or not np.isfinite(v).all()
            for v in (player_xy, reserved_stance_xy, fallback_outward)
        )
        or type(clearance_m) not in (float, int)
        or not math.isfinite(clearance_m)
        or not 0.8 <= clearance_m <= 1.5
        or np.linalg.norm(fallback_outward) < 1e-9
    ):
        raise ValueError("finite bounded task-space reservation required")
    away = np.asarray(player_xy - reserved_stance_xy, dtype=np.float64)
    distance = float(np.linalg.norm(away))
    if distance >= clearance_m:
        return None
    if distance < 1e-9:
        away = np.asarray(fallback_outward, dtype=np.float64).copy()
    away /= float(np.linalg.norm(away))
    lower_speed = min(0.4, 0.8 * (clearance_m - distance))
    return np.array([[away[0], away[1], lower_speed]], dtype=np.float64)
