"""Native static-reference rejection before G1 motion-controller experiments.

Inverse dynamics is a feasibility diagnostic, not an execution path or a
certificate of dynamic stability. The caller's model and state are never
modified. No renderer or hardware transport is involved.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS


@dataclass(frozen=True)
class G1PostureContact:
    first_geom: str
    second_geom: str
    signed_distance_m: float


@dataclass(frozen=True)
class G1StaticPostureAudit:
    qpos_sha256: str
    self_contacts: tuple[G1PostureContact, ...]
    inverse_joint_torque_nm: tuple[float, ...]
    unactuated_root_wrench: tuple[float, ...]
    maximum_joint_violation_rad: float
    rejection_reasons: tuple[str, ...]
    activation_ceiling: str = field(default="SIM_ONLY", init=False)
    promotion_eligible: bool = field(default=False, init=False)

    @property
    def reference_feasible(self) -> bool:
        """Static necessary conditions only; never a motion-success gate."""
        return not self.rejection_reasons


def audit_g1_static_posture(
    model: Any,
    qpos: ArrayLike,
    *,
    prefix: str = "",
    maximum_root_force_n: float = 0.05,
    maximum_root_moment_nm: float = 0.05,
) -> G1StaticPostureAudit:
    """Check one model-native pose, including robot-self collision geometry.

    Root wrench entries follow MuJoCo free-joint generalized-force order.
    Only direct, unit-gear torque motors are supported. Requested inverse
    torque is compared to hard limits *without clipping away infeasibility*.
    Ground contact is intentionally not classified as robot-self contact.
    This does not check environmental clearance, transitions, observation
    support, source licensing, balance robustness, or physical authorization.
    """
    import mujoco

    for value in (maximum_root_force_n, maximum_root_moment_nm):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("root residual ceilings must be finite and positive")
    position = np.asarray(qpos, dtype=np.float64)
    if position.shape != (model.nq,) or not np.all(np.isfinite(position)):
        raise ValueError("posture must be a finite native qpos vector")
    # Reject invalid quaternion input instead of letting normalization hide it.
    for joint in range(model.njnt):
        kind = model.jnt_type[joint]
        address = int(model.jnt_qposadr[joint])
        if kind == mujoco.mjtJoint.mjJNT_FREE:
            quaternion = position[address + 3 : address + 7]
        elif kind == mujoco.mjtJoint.mjJNT_BALL:
            quaternion = position[address : address + 4]
        else:
            continue
        if not np.isclose(np.linalg.norm(quaternion), 1.0, rtol=0.0, atol=1e-7):
            raise ValueError("posture quaternion must be normalized")
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "pelvis")
    joints = np.asarray(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + name)
            for name in G1_DDS_JOINT_NAMES
        ],
        dtype=np.int64,
    )
    if root < 1 or np.any(joints < 0):
        raise ValueError("G1 root or canonical joint contract is missing")
    root_joint = int(model.body_jntadr[root])
    if root_joint < 0 or model.jnt_type[root_joint] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("G1 pelvis must own a native free joint")
    if np.any(model.jnt_type[joints] != mujoco.mjtJoint.mjJNT_HINGE):
        raise ValueError("G1 canonical joints must be scalar hinges")
    body_ids = {root}
    for body in range(root + 1, model.nbody):
        if int(model.body_parentid[body]) in body_ids:
            body_ids.add(body)
    if any(int(model.jnt_bodyid[joint]) not in body_ids for joint in joints):
        raise ValueError("G1 joints must belong to the selected pelvis subtree")
    motor_indices = []
    for joint in joints:
        motors = np.flatnonzero(
            (model.actuator_trnid[:, 0] == joint)
            & (model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
        )
        if len(motors) != 1:
            raise ValueError("each G1 joint requires one direct torque motor")
        motor = int(motors[0])
        motor_indices.append(motor)
        if (
            model.actuator_gear[motor, 0] != 1.0
            or model.actuator_gaintype[motor] != mujoco.mjtGain.mjGAIN_FIXED
            or model.actuator_gainprm[motor, 0] != 1.0
            or model.actuator_biastype[motor] != mujoco.mjtBias.mjBIAS_NONE
            or model.actuator_dyntype[motor] != mujoco.mjtDyn.mjDYN_NONE
        ):
            raise ValueError("posture audit supports unit-gear direct torque motors only")
    data = mujoco.MjData(model)
    data.qpos[:] = position
    mujoco.mj_forward(model, data)
    data.qacc[:] = 0.0
    mujoco.mj_inverse(model, data)
    contacts = tuple(
        G1PostureContact(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1))
            or f"geom:{int(contact.geom1)}",
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2))
            or f"geom:{int(contact.geom2)}",
            float(contact.dist),
        )
        for contact in data.contact
        if int(model.geom_bodyid[contact.geom1]) in body_ids
        and int(model.geom_bodyid[contact.geom2]) in body_ids
        and contact.dist < 0.0
    )
    requested = np.asarray(data.qfrc_inverse[model.jnt_dofadr[joints]])
    root_dof = int(model.jnt_dofadr[root_joint])
    wrench = np.asarray(data.qfrc_inverse[root_dof : root_dof + 6])
    q = position[model.jnt_qposadr[joints]]
    ranges = model.jnt_range[joints]
    violation = float(
        np.max(
            np.where(
                model.jnt_limited[joints],
                np.maximum(0.0, np.maximum(ranges[:, 0] - q, q - ranges[:, 1])),
                0.0,
            )
        )
    )
    reasons = []
    if not np.all(np.isfinite(requested)) or not np.all(np.isfinite(wrench)):
        reasons.append("nonfinite_inverse_dynamics")
    if any(int(warning.number) > 0 for warning in data.warning):
        reasons.append("native_physics_warning")
    if not np.array_equal(data.qpos, position):
        reasons.append("native_state_changed_during_static_audit")
    if contacts:
        reasons.append("robot_self_penetration")
    if violation > 1e-5:
        reasons.append("joint_limit_violation")
    if np.any(np.abs(requested) > np.asarray(G1_HARD_TORQUE_LIMITS)):
        reasons.append("inverse_torque_exceeds_hard_limit")
    motors = np.asarray(motor_indices, dtype=np.int64)
    for limited, bounds, reason in (
        (
            model.actuator_ctrllimited[motors],
            model.actuator_ctrlrange[motors],
            "inverse_torque_exceeds_model_control_limit",
        ),
        (
            model.actuator_forcelimited[motors],
            model.actuator_forcerange[motors],
            "inverse_torque_exceeds_model_force_limit",
        ),
    ):
        if np.any(limited & ((requested < bounds[:, 0]) | (requested > bounds[:, 1]))):
            reasons.append(reason)
    if np.linalg.norm(wrench[:3]) > maximum_root_force_n:
        reasons.append("unactuated_root_force")
    if np.linalg.norm(wrench[3:]) > maximum_root_moment_nm:
        reasons.append("unactuated_root_moment")
    return G1StaticPostureAudit(
        qpos_sha256=hashlib.sha256(position.tobytes()).hexdigest(),
        self_contacts=contacts,
        inverse_joint_torque_nm=tuple(float(value) for value in requested),
        unactuated_root_wrench=tuple(float(value) for value in wrench),
        maximum_joint_violation_rad=violation,
        rejection_reasons=tuple(reasons),
    )
