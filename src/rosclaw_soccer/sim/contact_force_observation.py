"""Owned body-contact force observations from explicitly solved rigid contacts.

Pure numeric reduction, independent of simulator and task. It neither solves
contacts nor attests that supplied forces came from physical execution.
"""

from typing import Any

import numpy as np


def maximum_contact_force_by_body(
    world_ids: Any,
    geom_pairs: Any,
    force_torque: Any,
    geom_body_ids: Any,
    *,
    target_geom_id: int,
    observed_body_ids: tuple[int, ...],
    world_count: int,
) -> np.ndarray[Any, Any]:
    """Maximum linear-force norm, not summed force or torque, per world/body.

    Inputs must contain only active rigid contacts. The six-vector convention
    is MuJoCo/MJWarp force-then-torque. No threshold or event is imposed here.
    """
    worlds, pairs, forces, bodies = [
        np.asarray(v) for v in (world_ids, geom_pairs, force_torque, geom_body_ids)
    ]
    if (
        type(world_count) is not int
        or not 1 <= world_count <= 4096
        or type(target_geom_id) is not int
        or type(observed_body_ids) is not tuple
        or not 1 <= len(observed_body_ids) <= 128
        or any(type(b) is not int or b < 0 for b in observed_body_ids)
        or len(set(observed_body_ids)) != len(observed_body_ids)
        or worlds.ndim != 1
        or len(worlds) > 1000000
        or pairs.shape != (len(worlds), 2)
        or forces.shape != (len(worlds), 6)
        or bodies.ndim != 1
        or not 1 <= len(bodies) <= 1000000
        or any(v.dtype.kind not in "iu" for v in (worlds, pairs, bodies))
        or forces.dtype.kind not in "fiu"
        or not np.isfinite(forces).all()
        or np.any(bodies < 0)
        or not 0 <= target_geom_id < len(bodies)
        or np.any((worlds < 0) | (worlds >= world_count))
        or np.any((pairs < 0) | (pairs >= len(bodies)))
        or np.any(pairs[:, 0] == pairs[:, 1])
    ):
        raise ValueError(
            "bounded active rigid contacts with explicit world/body identities required"
        )
    result: np.ndarray[Any, Any] = np.zeros((world_count, len(observed_body_ids)), dtype=np.float64)
    selected = np.any(pairs == target_geom_id, axis=1)
    other = np.where(pairs[selected, 0] == target_geom_id, pairs[selected, 1], pairs[selected, 0])
    other_bodies = bodies[other]
    with np.errstate(over="raise", invalid="raise"):
        try:
            norms = np.linalg.norm(forces[selected, :3].astype(np.float64), axis=1)
        except FloatingPointError as error:
            raise ValueError("finite linear contact force norms required") from error
    for column, body in enumerate(observed_body_ids):
        mask = other_bodies == body
        np.maximum.at(result[:, column], worlds[selected][mask], norms[mask])
    result.flags.writeable = False
    return result
