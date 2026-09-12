"""Convert a skill's body-relative contact geometry into a navigation target.

Geometry only: not an intercept predictor, policy, admission gate or motion
authority. Callers must retain obstacle, body, ownership and skill checks.
"""

from __future__ import annotations

import math


def planar_entry_root_target(
    *,
    contact_xy: tuple[float, float],
    root_to_contact_body_xy: tuple[float, float],
    heading_rad: float,
) -> tuple[float, float]:
    """Place a *proposed* root target so its rotated contact offset hits a point.

    Inputs are metres and radians in one caller-owned planar frame. This
    performs no extrapolation, pose writes or physical reachability claim.
    The body-relative offset belongs to a separately qualified skill contract.
    """
    for vector, bound in ((contact_xy, 1000.0), (root_to_contact_body_xy, 2.0)):
        if (
            type(vector) is not tuple
            or len(vector) != 2
            or any(
                type(value) not in (int, float) or not math.isfinite(value) or abs(value) > bound
                for value in vector
            )
        ):
            raise ValueError("finite bounded immutable planar coordinates required")
    if (
        type(heading_rad) not in (int, float)
        or not math.isfinite(heading_rad)
        or abs(heading_rad) > math.pi
    ):
        raise ValueError("finite heading in [-pi, pi] required")
    c, s = math.cos(heading_rad), math.sin(heading_rad)
    x, y = root_to_contact_body_xy
    return contact_xy[0] - c * x + s * y, contact_xy[1] - s * x - c * y
