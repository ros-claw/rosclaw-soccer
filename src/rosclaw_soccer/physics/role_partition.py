"""Read-only native model ownership for independent player evidence.

A role's body/joint/actuator checks must not accidentally use another player's
state. This identifies topology, not permissions or real-world ownership.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco


@dataclass(frozen=True)
class NativeRolePartition:
    root_body_id: int
    body_ids: tuple[int, ...]
    geom_ids: tuple[int, ...]
    joint_ids: tuple[int, ...]
    actuator_ids: tuple[int, ...]


def partition_native_role(model: mujoco.MjModel, *, root_body_name: str) -> NativeRolePartition:
    """Return detached topology IDs below one exact named non-world body.

    Joint and joint-in-parent actuator transmissions are supported. A model with
    any other actuator transmission is refused: silently dropping tendon/site
    actuators could yield incomplete force accounting. Partitions may overlap
    when callers supply nested roots; team callers must check disjointness.
    """
    if not isinstance(model, mujoco.MjModel):
        raise TypeError("native MuJoCo model required")
    if not isinstance(root_body_name, str) or not root_body_name or len(root_body_name) > 256:
        raise ValueError("exact nonempty root body name required")
    if max(model.nbody, model.ngeom, model.njnt, model.nu) > 8192:
        raise ValueError("native role topology exceeds offline budget")
    root = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_body_name))
    if root <= 0:
        raise ValueError("role root must exist and cannot be the world")
    supported = {int(mujoco.mjtTrn.mjTRN_JOINT), int(mujoco.mjtTrn.mjTRN_JOINTINPARENT)}
    if any(int(value) not in supported for value in model.actuator_trntype):
        raise ValueError("unsupported actuator transmission prevents complete role accounting")
    if any(not 0 <= int(body) < model.nbody for body in model.geom_bodyid):
        raise ValueError("invalid geometry body binding")
    if any(not 1 <= int(body) < model.nbody for body in model.jnt_bodyid):
        raise ValueError("invalid joint body binding")
    bodies = {root}
    # Compiled MuJoCo bodies are parent-before-child; verify rather than assume
    # this invariant on a possibly edited model.
    for body in range(1, model.nbody):
        parent = int(model.body_parentid[body])
        if not 0 <= parent < body:
            raise ValueError("invalid compiled body parent ordering")
        if parent in bodies:
            bodies.add(body)
    joints = tuple(j for j in range(model.njnt) if int(model.jnt_bodyid[j]) in bodies)
    joint_set = set(joints)
    actuators = []
    for actuator in range(model.nu):
        joint = int(model.actuator_trnid[actuator, 0])
        if not 0 <= joint < model.njnt:
            raise ValueError("invalid actuator joint binding")
        if joint in joint_set:
            actuators.append(actuator)
    return NativeRolePartition(
        root_body_id=root,
        body_ids=tuple(sorted(bodies)),
        geom_ids=tuple(g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in bodies),
        joint_ids=joints,
        actuator_ids=tuple(actuators),
    )
