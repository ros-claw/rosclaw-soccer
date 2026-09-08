"""Bounded 2-D player-velocity proposals, not a physical collision guarantee.

Project only closing motion; preserve useful tangential runs. The simultaneous
half-plane intersection avoids order-dependent sequential repulsion. Physics
must still verify gait tracking, reach, contact and balance downstream.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass(frozen=True)
class ClearanceProposal:
    velocity_mps: tuple[float, float]
    feasible: bool
    constrained: bool


def propose_clearance_velocity(
    nominal: np.ndarray,
    neighbor_offsets: np.ndarray,
    *,
    maximum_speed_mps: float,
    clearance_m: float = 0.8,
    influence_m: float = 1.2,
    approach_gain: float = 0.6,
) -> ClearanceProposal:
    if (
        nominal.shape != (2,)
        or neighbor_offsets.ndim != 2
        or neighbor_offsets.shape[1] != 2
        or len(neighbor_offsets) > 16
        or not np.all(np.isfinite(nominal))
        or not np.all(np.isfinite(neighbor_offsets))
        or not 0.1 <= maximum_speed_mps <= 1.0
        or not 0.5 <= clearance_m <= 1.0
        or not clearance_m < influence_m <= 2.0
        or not 0.1 <= approach_gain <= 2.0
    ):
        raise ValueError("invalid bounded football clearance proposal")
    normals, bounds = [], []
    for offset in neighbor_offsets:
        distance = float(np.linalg.norm(offset))
        if distance < 1e-9:
            return ClearanceProposal((0.0, 0.0), False, True)
        if distance < influence_m:
            normals.append(-offset / distance)
            bounds.append(approach_gain * (clearance_m - distance))
    normal = np.asarray(normals).reshape((-1, 2))
    bound = np.asarray(bounds)

    def feasible(v: np.ndarray) -> bool:
        return bool(
            np.linalg.norm(v) <= maximum_speed_mps + 1e-10 and np.all(normal @ v >= bound - 1e-10)
        )

    candidates = [
        nominal,
        np.zeros(2),
        nominal * min(1.0, maximum_speed_mps / max(float(np.linalg.norm(nominal)), 1e-12)),
    ]
    for n, b in zip(normal, bound, strict=True):
        candidates.append(nominal + (b - float(n @ nominal)) * n)
        if abs(b) <= maximum_speed_mps:
            tangent = np.array([-n[1], n[0]])
            radius = math.sqrt(max(0.0, maximum_speed_mps**2 - float(b) ** 2))
            candidates.extend((b * n + radius * tangent, b * n - radius * tangent))
    for i, j in combinations(range(len(normal)), 2):
        matrix = normal[[i, j]]
        if abs(float(np.linalg.det(matrix))) > 1e-10:
            candidates.append(np.linalg.solve(matrix, bound[[i, j]]))
    valid = [v for v in candidates if feasible(v)]
    if not valid:
        return ClearanceProposal((0.0, 0.0), False, True)
    selected = min(
        valid, key=lambda v: (float(np.square(v - nominal).sum()), float(v[0]), float(v[1]))
    )
    return ClearanceProposal(
        (float(selected[0]), float(selected[1])),
        True,
        bool(np.linalg.norm(selected - nominal) > 1e-10),
    )
