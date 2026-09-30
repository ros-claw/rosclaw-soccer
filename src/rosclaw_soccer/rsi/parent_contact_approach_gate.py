"""SIM_ONLY shadow-contact veto for a candidate G1 approach option.

This is a simulator teacher, not a deployable real-world risk estimator. It
chooses from an independently audited candidate *parent* rollout, never from
the candidate actor's outcome.
"""

from __future__ import annotations

import math
from typing import Any

from rosclaw_soccer.sim.contracts import hash_json


def choose_shadow_approach_gain(parent: dict[str, Any]) -> tuple[float, str]:
    """Veto gain 1.2 when its parent predicts right support-knee contact."""
    if (
        parent.get("report_hash")
        != hash_json({key: value for key, value in parent.items() if key != "report_hash"})
        or parent.get("activation_ceiling") != "SIM_ONLY"
        or parent.get("navigation_lateral_ball_gain") != 1.2
        or parent.get("navigation_lateral_negative_only") is not True
        or parent.get("frames") != 300
        or not isinstance(parent.get("environments"), list)
        or len(parent["environments"]) != 1
    ):
        raise ValueError("sealed SIM_ONLY candidate parent required")
    row = parent["environments"][0]
    indices = row.get("contact_body_indices")
    if (
        not isinstance(indices, list)
        or any(type(index) is not int or index not in range(6) for index in indices)
        or type(row.get("minimum_pelvis_z_m")) not in (float, int)
        or not math.isfinite(row["minimum_pelvis_z_m"])
    ):
        raise ValueError("validated contact and pelvis evidence required")
    if row["minimum_pelvis_z_m"] < 0.65:
        return 0.8, "low_pelvis"
    if not indices:
        return 0.8, "no_parent_contact"
    if 5 in indices:
        return 0.8, "right_support_knee_contact"
    return 1.2, "candidate_parent_no_right_knee_contact"
