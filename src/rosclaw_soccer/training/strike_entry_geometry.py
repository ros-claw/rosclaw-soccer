"""Physical initiation geometry for the existing mirrored G1 strike reference.

This is a training objective, not shoot intent, possession, motor readiness,
reference-policy coverage, permission, or evidence of a completed strike.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class StrikeEntryGeometry:
    depth_m: NDArray[np.float64]
    lateral_m: NDArray[np.float64]
    forward_speed_mps: NDArray[np.float64]
    upright: NDArray[np.float64]
    eligible: NDArray[np.bool_]


def evaluate_strike_entry_geometry(
    qpos: NDArray[np.floating], qvel: NDArray[np.floating], *, attack_sign: int
) -> StrikeEntryGeometry:
    """Match the existing .50..68/.30/>.1/>.65/>.9 physical entry predicate.

    Inputs are native isolated G1+ball observations. Blue attacks negative X;
    red attacks positive X. No yaw-support or rolling-ball claim is implied:
    these are not conditions of the legacy strike geometry being reproduced.
    Returned arrays are independent, read-only observations, not authority.
    """
    if (
        type(attack_sign) is not int
        or attack_sign not in (-1, 1)
        or not isinstance(qpos, np.ndarray)
        or not isinstance(qvel, np.ndarray)
        or qpos.ndim != 2
        or qpos.shape[1] != 43
        or not 1 <= len(qpos) <= 4096
        or qvel.shape != (len(qpos), 41)
        or qpos.dtype not in (np.dtype("float32"), np.dtype("float64"))
        or qvel.dtype not in (np.dtype("float32"), np.dtype("float64"))
        or not np.isfinite(qpos).all()
        or not np.isfinite(qvel).all()
        or (np.abs(qpos) > 1e6).any()
        or (np.abs(qvel) > 1e6).any()
    ):
        raise ValueError(
            "bounded finite native G1/ball states and signed attack direction required"
        )
    p = qpos.astype(np.float64, copy=True)
    v = qvel.astype(np.float64, copy=True)
    if np.any(np.abs(np.linalg.norm(p[:, 3:7], axis=1) - 1) > 1e-5):
        raise ValueError("normalized physical root quaternion required")
    depth = attack_sign * (p[:, 36] - p[:, 0])
    lateral = attack_sign * (p[:, 37] - p[:, 1])
    forward = attack_sign * v[:, 0]
    upright = 1 - 2 * (p[:, 4] ** 2 + p[:, 5] ** 2)
    eligible = (
        (depth >= 0.50)
        & (depth <= 0.68)
        & (np.abs(lateral) <= 0.30)
        & (forward > 0.1)
        & (p[:, 2] > 0.65)
        & (upright > 0.9)
    )
    for value in (depth, lateral, forward, upright, eligible):
        value.flags.writeable = False
    return StrikeEntryGeometry(depth, lateral, forward, upright, eligible)


def evaluate_short_range_moving_strike_entry(
    qpos: NDArray[np.floating],
    qvel: NDArray[np.floating],
    *,
    attack_sign: int,
    reference_yaw_rad: float,
    allow_planted: bool = False,
) -> StrikeEntryGeometry:
    """Experimental S653-derived proposal domain, NOT the legacy entry gate.

    A closer .18..32 m ball moving goalward .5..1.5 m/s, rolling height,
    and reference-facing body are needed in addition to the old body checks.
    This does not qualify reference transfer, relax physical safety guards,
    assign possession or authorize a motor. Shared-world validation is separate.
    Explicit planted mode additionally admits near-zero forward speed (within
    .05 m/s) and planar speed <=.2 m/s. That is NOT a running-strike claim.
    """
    if (
        type(allow_planted) is not bool
        or type(reference_yaw_rad) not in (int, float)
        or not math.isfinite(reference_yaw_rad)
        or abs(reference_yaw_rad) > math.pi
    ):
        raise ValueError("finite signed reference yaw required")
    geometry = evaluate_strike_entry_geometry(qpos, qvel, attack_sign=attack_sign)
    p = qpos.astype(np.float64)
    w, x, y, z = p[:, 3:7].T
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    error = np.arctan2(np.sin(yaw - reference_yaw_rad), np.cos(yaw - reference_yaw_rad))
    ball_forward = attack_sign * qvel[:, 35].astype(np.float64)
    eligible = (
        (geometry.depth_m >= 0.18)
        & (geometry.depth_m <= 0.32)
        & (np.abs(geometry.lateral_m) <= 0.30)
        & (
            (geometry.forward_speed_mps > 0.1)
            | (
                allow_planted
                & (np.abs(geometry.forward_speed_mps) <= 0.05)
                & (np.linalg.norm(qvel[:, :2].astype(np.float64), axis=1) <= 0.2)
            )
        )
        & (p[:, 2] > 0.65)
        & (geometry.upright > 0.9)
        & (ball_forward > 0.5)
        & (np.linalg.norm(qvel[:, 35:38].astype(np.float64), axis=1) <= 1.5)
        & (p[:, 38] >= 0)
        & (p[:, 38] <= 0.3)
        & (np.abs(error) <= 0.35)
    )
    eligible.flags.writeable = False
    return StrikeEntryGeometry(
        geometry.depth_m,
        geometry.lateral_m,
        geometry.forward_speed_mps,
        geometry.upright,
        eligible,
    )
