"""Inspect the compiled ball rather than infer dimensions from a goal label."""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class NativeBallDimensions:
    geom_id: int
    body_id: int
    radius_m: float
    circumference_m: float
    body_mass_kg: float
    principal_inertia_kg_m2: tuple[float, ...]
    circumference_in_ifab_range: bool
    mass_in_ifab_range: bool

    @property
    def size_and_mass_in_ifab_range(self) -> bool:
        """This says nothing about pressure, material, rolling or game skill."""
        return self.circumference_in_ifab_range and self.mass_in_ifab_range


def inspect_native_ball_dimensions(model: mujoco.MjModel, *, geom_id: int) -> NativeBallDimensions:
    """Read one spherical geom on a single-free-joint body without mutation.

    Non-regulation size/mass is recorded, not rejected: deliberate simulation
    experiments may use it, but must not claim adult regulation dimensions.
    The body mass includes any other attached geometry; material, compliance,
    inflation pressure and FIFA certification are not established by this read.
    """
    if not isinstance(model, mujoco.MjModel):
        raise TypeError("native MuJoCo model required")
    if type(geom_id) is not int or not 0 <= geom_id < model.ngeom:
        raise ValueError("valid integer ball geometry ID required")
    if model.ngeom > 8192:
        raise ValueError("ball inspection budget exceeded")
    if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_SPHERE):
        raise ValueError("ball geometry must be spherical")
    body = int(model.geom_bodyid[geom_id])
    if not 0 < body < model.nbody or int(model.body_jntnum[body]) != 1:
        raise ValueError("ball must have one free joint on a non-world body")
    joint = int(model.body_jntadr[body])
    if not 0 <= joint < model.njnt or int(model.jnt_type[joint]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise ValueError("ball must have one free joint on a non-world body")
    radius = float(model.geom_size[geom_id, 0])
    mass = float(model.body_mass[body])
    inertia = np.array(model.body_inertia[body], dtype=np.float64, copy=True)
    if not math.isfinite(radius) or not 0 < radius <= 1:
        raise ValueError("finite bounded ball radius required")
    if not math.isfinite(mass) or not 0 < mass <= 100:
        raise ValueError("finite bounded ball body mass required")
    if not np.isfinite(inertia).all() or not np.all(inertia > 0):
        raise ValueError("positive finite principal inertia required")
    circumference = 2 * math.pi * radius
    # Only a round-off allowance, not a tunable task/physics tolerance.
    epsilon = 1e-12
    return NativeBallDimensions(
        geom_id=geom_id,
        body_id=body,
        radius_m=radius,
        circumference_m=circumference,
        body_mass_kg=mass,
        principal_inertia_kg_m2=tuple(map(float, inertia)),
        circumference_in_ifab_range=0.68 - epsilon <= circumference <= 0.70 + epsilon,
        mass_in_ifab_range=0.410 - epsilon <= mass <= 0.450 + epsilon,
    )
