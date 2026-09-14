"""Read-only, robot-scoped support evidence for native MuJoCo simulations.

These are summed contact-force magnitudes, not a net wrench or a stability
certificate. Explicit ground geometry prevents walls and other actors from
being mistaken for this robot's ground support.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SupportForces:
    support_force_n: float
    other_robot_ground_force_n: float
    support_contacts: int
    other_robot_ground_contacts: int


def measure_support_forces(
    model: Any,
    data: Any,
    *,
    robot_root_body_id: int,
    support_body_ids: tuple[int, ...],
    ground_geom_ids: tuple[int, ...],
) -> SupportForces:
    """Inspect existing contacts without stepping or changing simulator state."""
    import mujoco

    def valid_id(value: int, count: int) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and 0 <= value < count

    if not valid_id(robot_root_body_id, model.nbody) or robot_root_body_id == 0:
        raise ValueError("robot root must be a non-world body")
    if not support_body_ids or not ground_geom_ids:
        raise ValueError("explicit support bodies and ground geoms are required")
    if any(not valid_id(body, model.nbody) for body in support_body_ids):
        raise ValueError("invalid support body")
    if any(not valid_id(geom, model.ngeom) for geom in ground_geom_ids):
        raise ValueError("invalid ground geom")

    def descendant(body: int, root: int) -> bool:
        while body:
            if body == root:
                return True
            body = int(model.body_parentid[body])
        return False

    if any(not descendant(body, robot_root_body_id) for body in support_body_ids):
        raise ValueError("support body is outside the robot subtree")
    if any(
        descendant(int(model.geom_bodyid[geom]), robot_root_body_id) for geom in ground_geom_ids
    ):
        raise ValueError("ground geom belongs to the robot")
    ground = set(ground_geom_ids)
    forces = [0.0, 0.0]
    counts = [0, 0]
    for index, contact in enumerate(data.contact):
        first, second = int(contact.geom1), int(contact.geom2)
        if (first in ground) == (second in ground):
            continue
        body = int(model.geom_bodyid[second if first in ground else first])
        if not descendant(body, robot_root_body_id):
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, force)
        if not np.isfinite(force).all():
            raise ValueError("nonfinite native contact force")
        bucket = 0 if any(descendant(body, root) for root in support_body_ids) else 1
        with np.errstate(over="ignore", invalid="ignore"):
            magnitude = float(np.linalg.norm(force[:3]))
        if not np.isfinite(magnitude) or not np.isfinite(forces[bucket] + magnitude):
            raise ValueError("nonfinite native contact force magnitude")
        forces[bucket] += magnitude
        counts[bucket] += 1
    return SupportForces(forces[0], forces[1], counts[0], counts[1])
