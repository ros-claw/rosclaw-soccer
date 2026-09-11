"""Physical initiation geometry for the existing mirrored G1 strike reference.

This is a training objective, not shoot intent, possession, motor readiness,
reference-policy coverage, permission, or evidence of a completed strike.
"""

from __future__ import annotations

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
