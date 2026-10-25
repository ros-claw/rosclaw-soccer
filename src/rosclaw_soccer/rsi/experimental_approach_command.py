"""Separate SIM approach authority, not the old residual or a motion grant.

Acceleration precedes a simultaneous experimental command-domain, slew and
player-clearance projection. No state write, simulator step, or hardware.
"""

import math
from typing import Any

import numpy as np

from rosclaw_soccer.world.player_clearance import propose_clearance_velocity


def experimental_approach_command(
    parent: np.ndarray,
    previous: np.ndarray,
    *,
    root_xy: np.ndarray,
    ball_xy: np.ndarray,
    yaw: float,
    neighbor_offsets: np.ndarray,
) -> dict[str, Any]:
    if (
        any(
            not isinstance(a, np.ndarray) or a.shape != shape or not np.isfinite(a).all()
            for a, shape in ((parent, (3,)), (previous, (3,)), (root_xy, (2,)), (ball_xy, (2,)))
        )
        or parent.dtype != np.float32
        or previous.dtype != np.float32
        or not -0.5 <= parent[0] <= 1
        or abs(parent[1]) > 0.5
        or abs(parent[2]) > 1
        or not -1 <= previous[0] <= 2
        or abs(previous[1]) > 1
        or abs(previous[2]) > 1
        or not math.isfinite(yaw)
    ):
        raise ValueError("finite qualified parent and own experimental command history required")
    scale = 1 + float(np.clip((np.linalg.norm(ball_xy - root_xy) - 0.5), 0, 1))
    raw = parent.astype(np.float64)
    raw[:2] *= scale
    slewed = previous.astype(np.float64) + np.clip(raw - previous, -0.08, 0.08)
    c, s = math.cos(yaw), math.sin(yaw)
    f, lateral = np.array([c, s]), np.array([-s, c])
    low = np.maximum([-1.0, -1.0], previous[:2].astype(np.float64) - 0.08)
    high = np.minimum([2.0, 1.0], previous[:2].astype(np.float64) + 0.08)
    # Exactly eight halfplanes: four experimental-domain and four own-slew.
    # No parent-deviation claim is inherited from the old5% residual interface.
    planes = np.array(
        [
            [*f, -1.0],
            [*(-f), -2.0],
            [*lateral, -1.0],
            [*(-lateral), -1.0],
            [*f, low[0]],
            [*(-f), -high[0]],
            [*lateral, low[1]],
            [*(-lateral), -high[1]],
        ],
        np.float64,
    )
    result = propose_clearance_velocity(
        f * slewed[0] + lateral * slewed[1],
        neighbor_offsets,
        maximum_speed_mps=2.0,
        clearance_m=0.6,
        influence_m=1.2,
        approach_gain=0.6,
        additional_halfplanes=planes,
        experimental_velocity_envelope=True,
    )
    if result.feasible:
        world = np.array(result.velocity_mps)
        command = np.array([world @ f, world @ lateral, np.clip(slewed[2], -1, 1)], np.float32)
    else:
        # May reset slew. Fallback is explicit and never called a guarantee of
        # the candidate's slew constraints, just like the earlier guard.
        command = parent.copy()
    return dict(
        command=command,
        raw_command=raw,
        slewed_command=slewed,
        scale=scale,
        feasible=result.feasible,
        projection_changed=result.constrained,
        parent_fallback=not result.feasible,
    )
