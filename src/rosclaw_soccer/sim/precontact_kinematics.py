"""SIM-only separated-contact geometry and bounded shaping observations.

These quantities are not a contact force, success criterion, or safety gate.
"""

from dataclasses import dataclass

import mujoco
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PrecontactKinematics:
    status: str
    distance_m: float
    radial_separation_rate_mps: float | None
    nearest_geom: int | None
    shaping_only: bool = True
    hardware_authorized: bool = False


@dataclass(frozen=True)
class PrecontactPair:
    """Two separated witnesses at one instant; never a collision prediction."""

    status: str
    gap_difference_m: float | None
    radial_difference_mps: float | None
    linear_arrival_difference_s: float | None
    hardware_authorized: bool = False


def compare_separated_witnesses(
    preferred: PrecontactKinematics, other: PrecontactKinematics
) -> PrecontactPair:
    """Compare other-minus-preferred gap/radial speed with explicit unknowns.

    The optional arrival difference extrapolates each instantaneous radial
    velocity as constant. Changing geometry, rotation, acceleration, contact,
    and other objects invalidate that extrapolation; it is not a prediction,
    evidence of force, safety decision, task success, or execution authority.
    """
    if not isinstance(preferred, PrecontactKinematics) or not isinstance(
        other, PrecontactKinematics
    ):
        raise ValueError("two explicit geometric observations required")
    for witness in (preferred, other):
        if witness.status not in ("SEPARATED", "CONTACT_OR_OVERLAP", "DISTANCE_CENSORED"):
            raise ValueError("unknown geometric observation status")
        if not np.isfinite(witness.distance_m):
            raise ValueError("finite geometric distance required")
        if witness.status == "SEPARATED" and (
            witness.distance_m <= 0
            or witness.radial_separation_rate_mps is None
            or not np.isfinite(witness.radial_separation_rate_mps)
            or type(witness.nearest_geom) is not int
        ):
            raise ValueError("separated geometry requires a finite witness")
    if preferred.status != "SEPARATED" or other.status != "SEPARATED":
        return PrecontactPair("UNOBSERVED_OR_CONTACT", None, None, None)
    p_rate = preferred.radial_separation_rate_mps
    o_rate = other.radial_separation_rate_mps
    assert p_rate is not None and o_rate is not None
    # A near-zero or separating radial velocity has no finite arrival estimate.
    arrival = (
        float(other.distance_m / -o_rate - preferred.distance_m / -p_rate)
        if p_rate < -1e-6 and o_rate < -1e-6
        else None
    )
    gap = float(other.distance_m - preferred.distance_m)
    radial = float(o_rate - p_rate)
    if (
        not np.isfinite(gap)
        or not np.isfinite(radial)
        or (arrival is not None and not np.isfinite(arrival))
    ):
        raise ValueError("finite comparison required")
    return PrecontactPair("SEPARATED", gap, radial, arrival)


def observe_separated_geometry(
    model: mujoco.MjModel,
    qpos: NDArray[np.float64],
    qvel: NDArray[np.float64],
    *,
    object_geom: int,
    target_geoms: tuple[int, ...],
    distance_cap_m: float = 2.0,
) -> PrecontactKinematics:
    """Read nearest separated surfaces from a fresh scratch state and Jacobians.

    Negative radial rate means the observed witness points are approaching.
    The current source state is never mutated. Contact and distant geometry
    have explicit censored states rather than fabricated zero velocity.
    """
    if (
        not isinstance(model, mujoco.MjModel)
        or not isinstance(qpos, np.ndarray)
        or qpos.shape != (model.nq,)
        or qpos.dtype != np.float64
        or not isinstance(qvel, np.ndarray)
        or qvel.shape != (model.nv,)
        or qvel.dtype != np.float64
        or not np.isfinite(qpos).all()
        or not np.isfinite(qvel).all()
        or type(object_geom) is not int
        or not 0 <= object_geom < model.ngeom
        or type(target_geoms) is not tuple
        or not target_geoms
        or any(
            type(x) is not int or not 0 <= x < model.ngeom or x == object_geom for x in target_geoms
        )
        or len(set(target_geoms)) != len(target_geoms)
        or not np.isfinite(distance_cap_m)
        or not 0.01 <= distance_cap_m <= 10.0
    ):
        raise ValueError("finite typed simulation state and distinct geometry IDs required")
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = qpos
    scratch.qvel[:] = qvel
    mujoco.mj_forward(model, scratch)
    witnesses = []
    for geom in target_geoms:
        points = np.zeros(6, dtype=np.float64)
        distance = float(
            mujoco.mj_geomDistance(model, scratch, object_geom, geom, distance_cap_m, points)
        )
        if not np.isfinite(distance) or not np.isfinite(points).all():
            raise ValueError("finite geometry witness required")
        witnesses.append((distance, geom, points))
    distance, geom, points = min(witnesses, key=lambda x: (x[0], x[1]))
    if distance >= distance_cap_m:
        return PrecontactKinematics("DISTANCE_CENSORED", float(distance_cap_m), None, None)
    if distance <= 1e-8:
        return PrecontactKinematics("CONTACT_OR_OVERLAP", distance, None, geom)
    separation = points[3:] - points[:3]
    norm = float(np.linalg.norm(separation))
    if norm <= 1e-10 or abs(norm - distance) > 1e-7:
        raise ValueError("separated witness points inconsistent with distance")
    first = np.zeros((3, model.nv), np.float64)
    second = np.zeros((3, model.nv), np.float64)
    mujoco.mj_jac(model, scratch, first, None, points[:3], int(model.geom_bodyid[object_geom]))
    mujoco.mj_jac(model, scratch, second, None, points[3:], int(model.geom_bodyid[geom]))
    radial = float(((second - first) @ scratch.qvel) @ (separation / norm))
    if not np.isfinite(radial):
        raise ValueError("finite measured separation rate required")
    return PrecontactKinematics("SEPARATED", distance, radial, geom)


def cushion_shaping(observation: PrecontactKinematics) -> float | None:
    """Bounded shaping only: reward proximity while penalizing closing speed.

    None denotes unobserved or contact/overlap; callers must not turn it into
    a success label, a safety pass, or a score for unknown data.
    """
    if not isinstance(observation, PrecontactKinematics):
        raise ValueError("explicit precontact observation required")
    if observation.status != "SEPARATED":
        return None
    distance, rate = observation.distance_m, observation.radial_separation_rate_mps
    if not np.isfinite(distance) or distance <= 0 or rate is None or not np.isfinite(rate):
        raise ValueError("finite separated distance and relative speed required")
    return float(np.exp(-distance / 0.25) * np.exp(-max(0.0, -rate) / 2.0))
