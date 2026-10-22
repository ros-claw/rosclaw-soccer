"""TRAIN expert ranking: precision is meaningless without a useful kick."""

import math
from typing import Any


def run_teacher_rank(outcome: dict[str, Any]) -> tuple[int, int, int, int, int, float, float]:
    result = outcome["result"]
    post = result["postcontact"]
    stand = outcome["stand"]
    healthy = not result["body_failed"] and bool(
        stand and stand["healthy"] and stand["stability_observed"]
    )
    observed = bool(post["post_contact_quality_observed"])
    progress, lateral = post["goal_progress_1p2s_m"], post["goal_lateral_1p2s_m"]
    if observed and any(
        type(x) not in (float, int) or not math.isfinite(x) for x in (progress, lateral)
    ):
        raise ValueError("observed finite physical kick displacement required")
    useful = bool(
        healthy
        and observed
        and post["correct_foot_first"]
        and not post["other_foot_contact_in_window"]
        and progress >= 1
    )
    # Keep success first; within failures, never prefer a stationary ball for
    # its deceptively small lateral error. Stability is part of usefulness.
    return (
        int(bool(outcome["healthy_quality"])),
        int(useful),
        int(healthy),
        int(bool(post["correct_foot_first"])),
        int(not result["forbidden"]),
        -abs(float(lateral)) if useful else -100.0,
        min(6.0, max(-6.0, float(progress))) if observed else -6.0,
    )
