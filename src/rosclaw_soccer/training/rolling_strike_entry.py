"""Explain the existing PASS/SHOOT stance gate without granting motor authority.

Geometry alone does not establish contact ownership, a team handshake, measured
history, role permission, or readiness. Callers retain all of those checks.
"""

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from rosclaw_soccer.growth.locomotion_contact_teacher import G1RollingOptionBridgeConfig


@dataclass(frozen=True)
class RollingStrikeEntryGeometry:
    direction_xy: tuple[float, float]
    stance_depth_m: float
    lateral_error_m: float
    yaw_error_rad: float
    rejection_reasons: tuple[str, ...]
    activation_authorized: Literal[False] = field(default=False, init=False)

    @property
    def geometry_admissible(self) -> bool:
        return not self.rejection_reasons


def inspect_rolling_strike_entry(
    *,
    pelvis_xy: tuple[float, float],
    ball_xy: tuple[float, float],
    target_xy: tuple[float, float],
    yaw_rad: float,
    config: G1RollingOptionBridgeConfig,
) -> RollingStrikeEntryGeometry:
    """Use the original gate's arithmetic and inclusive thresholds, without state."""
    if not isinstance(config, G1RollingOptionBridgeConfig):
        raise ValueError("typed rolling option configuration required")
    config.__post_init__()
    for point in (pelvis_xy, ball_xy, target_xy):
        if (
            type(point) is not tuple
            or len(point) != 2
            or any(type(v) not in (float, int) or abs(v) > 1e4 for v in point)
            or not all(math.isfinite(v) for v in point)
        ):
            raise ValueError("finite immutable planar geometry required")
    if type(yaw_rad) not in (float, int) or abs(yaw_rad) > 1e4 or not math.isfinite(yaw_rad):
        raise ValueError("finite measured yaw required")
    ball = np.asarray(ball_xy, dtype=np.float64)
    direction = np.asarray(target_xy, dtype=np.float64) - ball
    direction /= max(float(np.linalg.norm(direction)), 1e-9)
    target_yaw = math.atan2(float(direction[1]), float(direction[0]))
    yaw_error = abs(math.atan2(math.sin(target_yaw - yaw_rad), math.cos(target_yaw - yaw_rad)))
    lateral = np.asarray((-direction[1], direction[0]), dtype=np.float64)
    offset = ball - np.asarray(pelvis_xy, dtype=np.float64)
    depth = float(np.dot(offset, direction))
    lateral_error = abs(float(np.dot(offset, lateral)))
    reasons = []
    if not config.minimum_strike_stance_depth_m <= depth <= config.maximum_strike_stance_depth_m:
        reasons.append("STANCE_DEPTH")
    if lateral_error > config.maximum_strike_lateral_error_m:
        reasons.append("LATERAL_ALIGNMENT")
    if yaw_error > config.maximum_strike_yaw_error_rad:
        reasons.append("HEADING_ALIGNMENT")
    return RollingStrikeEntryGeometry(
        (float(direction[0]), float(direction[1])), depth, lateral_error, yaw_error, tuple(reasons)
    )
