"""Bounded geometric waypoints around a contact region, not a safe path proof."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class AnnularWaypoint:
    target_xy: tuple[float, float]
    phase: str
    angular_error_rad: float


def annular_entry_waypoint(
    *,
    root_xy: tuple[float, float],
    contact_xy: tuple[float, float],
    entry_offset_xy: tuple[float, float],
    final_approach_allowed: bool,
    approach_committed: bool = False,
    radius_m: float = 0.5,
) -> AnnularWaypoint:
    """Retreat radially, orbit, then propose an entry-side approach.

    All coordinates share a caller-owned world frame. The final boolean is a
    geometric preparation cue, never a possession or motor permission. Fixed
    0.3-radian orbit steps and a 0.15-radian entry sector only bound waypoints:
    they do not model feet, support, moving contacts, obstacles or tracking.
    Callers must independently retain all actual execution/admission guards.
    A caller-owned previous approach phase may use a 0.35-radian exit sector
    to avoid boundary chatter. That cue carries no execution authority.
    """
    for vector in (root_xy, contact_xy, entry_offset_xy):
        if (
            type(vector) is not tuple
            or len(vector) != 2
            or any(
                type(x) not in (int, float) or not math.isfinite(x) or abs(x) > 1000 for x in vector
            )
        ):
            raise ValueError("finite immutable planar coordinates required")
    if (
        type(final_approach_allowed) is not bool
        or type(approach_committed) is not bool
        or type(radius_m) not in (int, float)
        or not math.isfinite(radius_m)
        or not 0.2 <= radius_m <= 2.0
        or not 0.05 <= math.hypot(*entry_offset_xy) <= radius_m - 0.05
    ):
        raise ValueError("bounded nonzero inner entry and explicit preparation cue required")
    dx, dy = root_xy[0] - contact_xy[0], root_xy[1] - contact_xy[1]
    distance = math.hypot(dx, dy)
    if distance < 1e-8:
        raise ValueError("coincident root/contact has no measured retreat direction")
    angle = math.atan2(dy, dx)
    desired = math.atan2(entry_offset_xy[1], entry_offset_xy[0])
    error = math.atan2(math.sin(desired - angle), math.cos(desired - angle))
    entry_sector = 0.35 if approach_committed else 0.15
    if abs(error) <= entry_sector and final_approach_allowed:
        return AnnularWaypoint(
            (contact_xy[0] + entry_offset_xy[0], contact_xy[1] + entry_offset_xy[1]),
            "approach",
            error,
        )
    if distance < radius_m - 0.03:
        target_angle, phase = angle, "outward"
    else:
        target_angle, phase = angle + max(-0.3, min(0.3, error)), "orbit"
    return AnnularWaypoint(
        (
            contact_xy[0] + radius_m * math.cos(target_angle),
            contact_xy[1] + radius_m * math.sin(target_angle),
        ),
        phase,
        error,
    )
