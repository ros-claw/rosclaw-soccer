"""Explain the unchanged near first-touch gate, never relax or promote it."""

import math
from typing import Any


def first_touch_failure_causes(score: dict[str, Any]) -> list[str]:
    for key in ("qualified", "common_prefix_physical_safe", "complete_post60_observable"):
        if type(score[key]) is not bool:
            raise ValueError("typed native gate flags required")
    for key in (
        "minimum_pelvis_z_m",
        "minimum_upright",
        "maximum_nonfoot_force_n",
        "forward_to_task_end_m",
        "lateral_to_task_end_m",
        "maximum_ball_vx_mps",
    ):
        if type(score[key]) not in (float, int) or not math.isfinite(score[key]):
            raise ValueError("finite native task metric required")
    touch = score["first_foot_contact_global_frame"]
    if touch is not None and (type(touch) is not int or not 20 <= touch < 300):
        raise ValueError("suffix-native first foot contact or explicit absence required")
    if score["complete_post60_observable"] != (touch is not None and touch <= 239):
        raise ValueError("original H300/post60 contact observation differs")
    causes = []
    if not score["common_prefix_physical_safe"]:
        causes.append("prefix_unsafe")
    if score["minimum_pelvis_z_m"] < 0.65:
        causes.append("body_height")
    if score["minimum_upright"] < 0.9:
        causes.append("body_upright")
    if score["maximum_nonfoot_force_n"] > 1:
        causes.append("nonfoot_ball_contact")
    if touch is None:
        causes.append("no_foot_touch")
    elif touch > 239:
        causes.append("late_touch_incomplete_post60")
    if score["forward_to_task_end_m"] < 0.5:
        causes.append("insufficient_forward_travel")
    if abs(score["lateral_to_task_end_m"]) > 0.3:
        causes.append("lateral_miss")
    if score["maximum_ball_vx_mps"] < 0.5:
        causes.append("insufficient_forward_speed")
    if score["qualified"] != (not causes):
        raise ValueError("explanation disagrees with original strict task qualification")
    return causes
