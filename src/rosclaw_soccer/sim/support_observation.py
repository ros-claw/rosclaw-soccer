"""Read-only entry-state measurements on a separate CPU MuJoCo data object.

Foot support is a measured ground normal-force threshold, not a gait-phase
guess. Refreshing derived quantities never mutates the live solver warm start.
These measurements do not certify balance, readiness, or task success.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class SupportObservation:
    pelvis_position_m: tuple[float, float, float]
    pelvis_velocity_mps: tuple[float, float, float]
    subtree_com_m: tuple[float, float, float]
    left_normal_force_n: float
    right_normal_force_n: float
    support_code: int  # 0 airborne, 1 left, 2 right, 3 double
    force_threshold_n: float


def observe_support(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    pelvis_body: int,
    left_foot_geoms: frozenset[int],
    right_foot_geoms: frozenset[int],
    ground_geoms: frozenset[int],
    force_threshold_n: float = 1.0,
) -> SupportObservation:
    """Measure an instantaneous support state without changing live physics.

    Ground identifiers must be explicit static-world geometries. A ball or
    opponent touching a foot must never be counted as ground support.
    """
    if data.model is not model:
        raise ValueError("data must belong to model")
    if type(pelvis_body) is not int or not 0 < pelvis_body < model.nbody:
        raise ValueError("invalid pelvis body")
    groups = (left_foot_geoms, right_foot_geoms, ground_geoms)
    if any(
        not isinstance(group, frozenset)
        or not group
        or any(type(i) is not int or not 0 <= i < model.ngeom for i in group)
        for group in groups
    ):
        raise ValueError("invalid or empty geometry binding")
    if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("geometry groups must be disjoint")
    if any(model.geom_bodyid[i] != 0 for i in ground_geoms):
        raise ValueError("ground must belong to static world")
    for geom in left_foot_geoms | right_foot_geoms:
        body = int(model.geom_bodyid[geom])
        while body not in (0, pelvis_body):
            body = int(model.body_parentid[body])
        if body != pelvis_body:
            raise ValueError("foot geometry outside pelvis subtree")
    if (
        type(force_threshold_n) not in (int, float)
        or not np.isfinite(force_threshold_n)
        or force_threshold_n <= 0
    ):
        raise ValueError("invalid force threshold")
    # This is a synchronous, same-model observation, not an external checkpoint
    # restore. Copy the identical integration specification into fresh data;
    # do not serialize/hash the large compiled mesh model twice per player.
    # Durable PhysicalCheckpoint capture/restore still verifies model identity.
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(model, spec), dtype=np.float64)
    mujoco.mj_getState(model, data, state, spec)
    if not np.isfinite(state).all():
        raise ValueError("nonfinite integration state")
    sample = mujoco.MjData(model)
    mujoco.mj_setState(model, sample, state, spec)
    mujoco.mj_forward(model, sample)
    forces = [0.0, 0.0]
    wrench = np.zeros(6)
    for index in range(sample.ncon):
        contact = sample.contact[index]
        pair = {int(contact.geom1), int(contact.geom2)}
        if not pair & ground_geoms:
            continue
        for side, feet in enumerate((left_foot_geoms, right_foot_geoms)):
            if pair & feet:
                mujoco.mj_contactForce(model, sample, index, wrench)
                forces[side] += max(0.0, float(wrench[0]))
    spatial = np.zeros(6)
    mujoco.mj_objectVelocity(model, sample, mujoco.mjtObj.mjOBJ_BODY, pelvis_body, spatial, 0)
    values = np.r_[sample.xpos[pelvis_body], spatial[3:], sample.subtree_com[pelvis_body], forces]
    if not np.isfinite(values).all():
        raise ValueError("nonfinite support measurement")
    return SupportObservation(
        (float(values[0]), float(values[1]), float(values[2])),
        (float(values[3]), float(values[4]), float(values[5])),
        (float(values[6]), float(values[7]), float(values[8])),
        forces[0],
        forces[1],
        int(forces[0] >= force_threshold_n) + 2 * int(forces[1] >= force_threshold_n),
        float(force_threshold_n),
    )
