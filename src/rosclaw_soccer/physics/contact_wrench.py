"""Read-only signed native contact forces for offline football diagnosis.

MuJoCo contact-frame x is the normal from geom 0 to geom 1. Wrenches
returned here act on the requested geometry, at the cached contact point.
Reading does not refresh kinematics or advance physics. After mj_step the
cached contact configuration can precede the integrated qpos; do not label
this as a fresh current-FK measurement or an execution receipt.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class GeomContactWrench:
    contact_id: int
    geom_id: int
    other_geom_id: int
    sample_time_sec: float
    signed_distance_m: float
    point_world_m: tuple[float, ...]
    normal_toward_geom_world: tuple[float, ...]
    force_world_n: tuple[float, ...]
    torque_at_contact_world_nm: tuple[float, ...]
    contact_normal_force_n: float


def read_geom_contact_wrenches(
    model: mujoco.MjModel, data: mujoco.MjData, *, geom_id: int
) -> tuple[GeomContactWrench, ...]:
    """Copy all cached contacts for one native rigid geom, including zero force.

    Forces are not filtered by body names or a success threshold. Flex contacts
    involving the requested geom are unsupported and refused. Contact torque
    is about the contact point, NOT the geom or body centre. Summing forces is
    meaningful; summing torques requires a common reference point first.
    """
    if not isinstance(model, mujoco.MjModel) or not isinstance(data, mujoco.MjData):
        raise TypeError("native MuJoCo model and data required")
    if data.model is not model:
        raise ValueError("data must belong to the exact supplied model")
    if type(geom_id) is not int or not 0 <= geom_id < model.ngeom:
        raise ValueError("valid integer geometry ID required")
    if model.ngeom > 8192 or not 0 <= data.ncon <= 8192:
        raise ValueError("contact probe budget exceeded")
    timestamp = float(data.time)
    if not math.isfinite(timestamp) or not 0 <= timestamp <= 3600:
        raise ValueError("finite bounded simulation time required")
    records = []
    for index in range(data.ncon):
        contact = data.contact[index]
        first, second = map(int, contact.geom)
        if geom_id not in (first, second):
            continue
        if not (0 <= first < model.ngeom and 0 <= second < model.ngeom) or first == second:
            raise ValueError("contact must bind two distinct rigid geometries")
        frame = np.array(contact.frame, dtype=np.float64, copy=True).reshape(3, 3)
        point = np.array(contact.pos, dtype=np.float64, copy=True)
        distance = float(contact.dist)
        if not np.isfinite(frame).all() or not np.isfinite(point).all():
            raise ValueError("nonfinite cached contact geometry")
        if not math.isfinite(distance):
            raise ValueError("nonfinite cached contact distance")
        if not np.allclose(frame @ frame.T, np.eye(3), rtol=0, atol=1e-8) or not math.isclose(
            float(np.linalg.det(frame)), 1.0, rel_tol=0, abs_tol=1e-8
        ):
            raise ValueError("invalid cached contact frame")
        local = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, index, local)
        if not np.isfinite(local).all():
            raise ValueError("nonfinite cached contact wrench")
        sign = 1.0 if geom_id == second else -1.0
        records.append(
            GeomContactWrench(
                contact_id=index,
                geom_id=geom_id,
                other_geom_id=first if geom_id == second else second,
                sample_time_sec=timestamp,
                signed_distance_m=distance,
                point_world_m=tuple(map(float, point)),
                normal_toward_geom_world=tuple(map(float, sign * frame[0])),
                force_world_n=tuple(map(float, sign * (frame.T @ local[:3]))),
                torque_at_contact_world_nm=tuple(map(float, sign * (frame.T @ local[3:]))),
                contact_normal_force_n=float(local[0]),
            )
        )
    return tuple(records)
