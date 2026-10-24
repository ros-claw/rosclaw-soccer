"""Predeclare nearby TRAIN proposals from audited safe delivery attempts.

A safe source can still be a failed pass. That distinction stays explicit.
This privileged teacher planner is never a student observation or success gate.
"""

import copy
import math
import re
from typing import Any

PILOT_STATES = (20, 23, 26, 29)
EXPANDED_STATES = (21, 22, 24, 25, 27, 28)


def _numbers(value: Any, size: int) -> list[float]:
    if (
        type(value) is not list
        or len(value) != size
        or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)
    ):
        raise ValueError("finite explicit physical coordinates required")
    return value


def _course(c: dict[str, Any], states: tuple[int, ...]) -> None:
    if (
        type(c["state_id"]) is not int
        or c["state_id"] not in states
        or type(c["task"]) is not int
        or c["task"] not in (0, 1)
        or c["context"] != 20
        or c["assisted"] is not True
        or c["side"] not in (-0.55, 0.55)
    ):
        raise ValueError("declared public context20 factorial TRAIN course required")
    _numbers(c["ball_planar_offset_world"], 2)
    _numbers(c["target"], 3)


def transfer_plans(
    courses: list[dict[str, Any]], donors: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if len(courses) != 12 or len(donors) != 8:
        raise ValueError("all twelve expansion tasks and eight pilot proposals required")
    for c in courses:
        _course(c, EXPANDED_STATES)
    for d in donors:
        _course(d["course"], PILOT_STATES)
        if (
            type(d["controlled"]) is not bool
            or d["healthy_clean"] is not True
            or type(d["evidence_hash"]) is not str
            or re.fullmatch(r"sha256:[0-9a-f]{64}", d["evidence_hash"]) is None
            or type(d["course"]["latent"]) is not list
            or len(d["course"]["latent"]) != 2
            or any(abs(v) > 1 for row in d["course"]["latent"] for v in _numbers(row, 6))
        ):
            raise ValueError("bounded explicitly safe source with honest functional label required")
    if {(c["state_id"], c["task"]) for c in courses} != {
        (s, t) for s in EXPANDED_STATES for t in (0, 1)
    } or {(d["course"]["state_id"], d["course"]["task"]) for d in donors} != {
        (s, t) for s in PILOT_STATES for t in (0, 1)
    }:
        raise ValueError("complete unique expansion and disjoint source grids required")
    plans = []
    for c in sorted(courses, key=lambda c: (c["state_id"], c["task"])):
        eligible = [
            d
            for d in donors
            if d["course"]["task"] == c["task"] and d["course"]["side"] == c["side"]
        ]
        if not eligible:
            raise ValueError("no extrapolation across receiving-side domain")
        donor = min(
            eligible,
            key=lambda d: (
                math.dist(d["course"]["ball_planar_offset_world"], c["ball_planar_offset_world"]),
                d["course"]["state_id"],
            ),
        )
        plans.append(
            dict(
                state_id=c["state_id"],
                task=c["task"],
                donor_state=donor["course"]["state_id"],
                donor_evidence_hash=donor["evidence_hash"],
                donor_controlled=donor["controlled"],
                course=copy.deepcopy(c | dict(latent=donor["course"]["latent"])),
                rule="same_goal_and_Bside_then_nearest_balloffset_then_lowest_state_id",
                privileged_TRAIN_geometry_only=True,
                student_input=False,
                source_failure_does_not_become_success=True,
                transferred_skill_physically_qualified=False,
            )
        )
    return plans
