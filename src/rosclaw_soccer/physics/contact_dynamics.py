"""Read cached ground wrenches and solved hinge-joint dynamics, without stepping.

After mj_step, contact geometry and solved forces can precede the integrated
qpos/qvel. They are not refreshed here. Generalized constraint torque includes
all solver constraints, not only ground contact or joint limits.
"""

from collections import Counter
from typing import Any

import mujoco
import numpy as np

from rosclaw_soccer.physics.contact_wrench import read_geom_contact_wrenches


def read_ground_contact_dynamics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    ground_geom_id: int,
    joint_names: tuple[str, ...],
    support_body_names: tuple[str, ...],
) -> dict[str, Any]:
    """Copy observed values; no control, refresh, support threshold or admission."""
    if not isinstance(model, mujoco.MjModel) or not isinstance(data, mujoco.MjData):
        raise TypeError("native MuJoCo model and data required")
    if data.model is not model:
        raise ValueError("data must belong to the exact supplied model")
    if (
        type(ground_geom_id) is not int
        or not 0 <= ground_geom_id < model.ngeom
        or model.geom_bodyid[ground_geom_id] != 0
        or model.geom_type[ground_geom_id] != mujoco.mjtGeom.mjGEOM_PLANE
    ):
        raise ValueError("explicit static world-plane ground geometry required")
    for names, maximum in ((joint_names, 128), (support_body_names, 16)):
        if (
            type(names) is not tuple
            or not 1 <= len(names) <= maximum
            or any(type(name) is not str or not name or len(name) > 128 for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("bounded distinct explicit joint and support body names required")

    def identifier(kind: mujoco.mjtObj, name: str) -> int:
        index = mujoco.mj_name2id(model, kind, name)
        if index < 0:
            raise ValueError("requested observed model element missing")
        return int(index)

    joints = [identifier(mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]
    if any(model.jnt_type[index] != mujoco.mjtJoint.mjJNT_HINGE for index in joints):
        raise ValueError("hinge joints required for explicit radian and torque units")
    support = {identifier(mujoco.mjtObj.mjOBJ_BODY, name): name for name in support_body_names}
    if 0 in support:
        raise ValueError("world body cannot be a robot support body")
    dofs = np.asarray([model.jnt_dofadr[index] for index in joints])
    dynamics = {}
    for key, raw in (
        ("joint_generalized_constraint_torque_nm", data.qfrc_constraint),
        ("joint_generalized_actuator_torque_nm", data.qfrc_actuator),
        ("joint_generalized_passive_torque_nm", data.qfrc_passive),
        ("joint_generalized_bias_torque_nm", data.qfrc_bias),
        ("joint_solved_acceleration_rad_s2", data.qacc),
        ("recorded_joint_velocity_rad_s", data.qvel),
    ):
        value = np.array(raw[dofs], dtype=np.float64, copy=True)
        if not np.isfinite(value).all() or np.max(np.abs(value)) > 1e12:
            raise ValueError("finite bounded solved contact dynamics required")
        dynamics[key] = value.tolist()
    contacts = []
    forces = {name: np.zeros(3) for name in support_body_names}
    for wrench in read_geom_contact_wrenches(model, data, geom_id=ground_geom_id):
        body = int(model.geom_bodyid[wrench.other_geom_id])
        # Existing signed reader reports the force ON the requested ground geom.
        # The opposite force acts on the contacted robot/object, not the ground.
        force = -np.asarray(wrench.force_world_n)
        if body in support:
            forces[support[body]] += force
        contacts.append(
            dict(
                contact_id=wrench.contact_id,
                other_geom_id=wrench.other_geom_id,
                other_body_id=body,
                other_body_name=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body),
                force_on_other_body_world_n=force.tolist(),
                normal_force_n=wrench.contact_normal_force_n,
                signed_distance_m=wrench.signed_distance_m,
                contact_point_world_m=list(wrench.point_world_m),
            )
        )
    if not 0 <= data.nefc <= 65536:
        raise ValueError("bounded active constraint count required")
    types = Counter(int(value) for value in data.efc_type[: data.nefc])
    limit_names = []
    for index in range(data.nefc):
        if data.efc_type[index] != mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT:
            continue
        joint = int(data.efc_id[index])
        if not 0 <= joint < model.njnt:
            raise ValueError("active joint-limit constraint identity invalid")
        limit_names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint))
    return dict(
        schema="soccer.physics.cached_ground_contact_dynamics.v1",
        data_time_sec=float(data.time),
        measurement_epoch="cached solver geometry/forces; no integrated-state FK refresh",
        joint_names=list(joint_names),
        support_body_names=list(support_body_names),
        ground_contacts=contacts,
        support_body_ground_force_world_n={name: force.tolist() for name, force in forces.items()},
        active_constraint_type_counts={str(key): value for key, value in sorted(types.items())},
        active_joint_limit_constraint_names=limit_names,
        **dynamics,
        constraint_torque_is_not_ground_only=True,
        support_readiness_or_balance_qualified=False,
        new_physics_steps=0,
        kinematics_refreshed=False,
        controls_mutated=False,
    )
