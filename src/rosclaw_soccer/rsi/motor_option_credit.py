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
