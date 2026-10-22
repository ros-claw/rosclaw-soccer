"""Declared coarse contact-skill reward and credit across REAL neural recovery.

This is not goal-plane precision, a goal score, or team-play performance.
"""

import math
from typing import Any

import numpy as np


def contact_skill_return(result: dict[str, Any]) -> float:
    if type(result["body_failed"]) is not bool or type(result["forbidden"]) is not bool:
        raise ValueError("actual physical outcome flags required")
    impulse = result["nonfoot_impulse_Ns"]
    if not math.isfinite(impulse) or impulse < 0:
        raise ValueError("finite nonnegative measured impulse required")
    if result["body_failed"]:
        return -30.0
    p = result["postcontact"]
    if not (p["correct_foot_first"] and p["post_contact_quality_observed"]):
        return -20.0
    forward, lateral, retreat = [
        p[k]
        for k in (
            "goal_progress_1p2s_m",
            "goal_lateral_1p2s_m",
            "maximum_pelvis_retreat_after_contact_m",
        )
    ]
    if any(
        type(x) not in (int, float) or not math.isfinite(x) for x in (forward, lateral, retreat)
    ):
        raise ValueError("finite observed contact window required")
    return float(
        min(4.0, max(0.0, forward))
        - 2 * abs(lateral)
        - 8 * result["forbidden"]
        - min(3.0, 10 * impulse)
        - 10 * max(0.0, retreat)
        + 6 * (not result["forbidden"])
    )


def terminal_option_returns(
    rows: int, standing_controls: int, terminal_reward: float
) -> np.ndarray:
    if (
        type(rows) is not int
        or not 1 <= rows <= 300
        or type(standing_controls) is not int
        or not 0 <= standing_controls <= 100
        or not math.isfinite(terminal_reward)
    ):
        raise ValueError("actual finite reference and observed neural-standing lengths required")
    # Every standing control really occurred. No synthetic held-target suffix,
    # nonexistent value bootstrap or artificial time-limit terminal.
    exponents = np.arange(rows - 1, -1, -1) + standing_controls
    return np.asarray(terminal_reward * 0.995**exponents, dtype=np.float32)


def observed_option_tail(
    reference_failed: np.ndarray,
    standing_failed: np.ndarray | None,
    *,
    body_failed: bool,
    standing_healthy: bool | None,
) -> int:
    """Keep failures, never invent their unexecuted recovery or a healthy terminal."""
    streams = [reference_failed] + ([standing_failed] if standing_failed is not None else [])
    if type(body_failed) is not bool or any(
        x.ndim != 1 or x.dtype != np.bool_ or not 1 <= len(x) <= (300 if i == 0 else 100)
        for i, x in enumerate(streams)
    ):
        raise ValueError("actual bounded boolean control-failure streams required")
    if reference_failed.any():
        if (
            not body_failed
            or not reference_failed[-1]
            or reference_failed[:-1].any()
            or standing_failed is not None
            or standing_healthy is not None
        ):
            raise ValueError("reference failure must be the actual first terminal boundary")
        return 0
    if standing_failed is None:
        raise ValueError("healthy reference cannot omit unobserved neural recovery")
    if standing_failed.any():
        if (
            not body_failed
            or not standing_failed[-1]
            or standing_failed[:-1].any()
            or standing_healthy is not False
        ):
            raise ValueError("standing failure must be the actual first terminal boundary")
        return len(standing_failed)
    if body_failed or len(standing_failed) != 100 or standing_healthy is not True:
        raise ValueError("healthy terminal requires the whole observed recovery horizon")
    return 100
