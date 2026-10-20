"""Bounded root-local toe-reference adaptation, not executed foot IK.

Only six right-leg REFERENCE joints change. The actual body still requires
own-state SONIC inference and PD integration. FK/Jacobian residuals here are
not dynamic contact, balance, ball-direction or teacher qualification.
"""

from typing import Any

import mujoco
import numpy as np

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.providers.g1.sonic_pose_reference import SonicPoseReference
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def bounded_point_delta(
    jacobian: np.ndarray,
    displacement: np.ndarray,
    current_delta: np.ndarray,
    baseline: np.ndarray,
    limits: np.ndarray,
) -> np.ndarray:
    """Four-step caller uses damped least squares; box bounds never widened."""
    for a, shape in (
        (jacobian, (3, 6)),
        (displacement, (3,)),
        (current_delta, (6,)),
        (baseline, (6,)),
        (limits, (6, 2)),
    ):
        if type(a) is not np.ndarray or a.shape != shape or not np.isfinite(a).all():
            raise ValueError("finite six-joint reference kinematics required")
    if (
        np.any(limits[:, 0] >= limits[:, 1])
        or np.any(baseline < limits[:, 0])
        or np.any(baseline > limits[:, 1])
        or np.max(abs(current_delta)) > 0.1200000001
        or np.linalg.norm(displacement) > 0.2
    ):
        raise ValueError("original legal reference and bounded IK residual required")
    step = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 0.03**2 * np.eye(3), displacement)
    step *= min(1.0, 0.04 / max(float(np.max(abs(step))), 1e-12))
    lower = np.maximum(-0.12, limits[:, 0] - baseline)
    upper = np.minimum(0.12, limits[:, 1] - baseline)
    bounded: np.ndarray = np.clip(current_delta + step, lower, upper)
    return bounded


def offset_contact_reference(
    model: Any, reference: SonicPoseReference, *, offset_root_m: tuple[float, float, float]
) -> tuple[SonicPoseReference, dict[str, Any]]:
    """Original fixed phase, smooth local90..210 bump, <=4cm/component.

    Uses the original front support-sphere midpoint, NOT a mesh impact point.
    Root metadata is not SONIC Cartesian authority. No actual body, clock,
    target or torque is mutated. Zero is exact identity and allocates no data.
    """
    if (
        type(offset_root_m) is not tuple
        or len(offset_root_m) != 3
        or not np.isfinite(offset_root_m).all()
        or max(abs(x) for x in offset_root_m) > 0.04
        or np.linalg.norm(offset_root_m) > 0.06
    ):
        raise ValueError("explicit bounded immutable root-local reference offset required")
    report = dict(
        schema="soccer.rsi.contact_reference_offset.v1",
        source_reference_hash=reference.contract_hash,
        offset_root_m=list(offset_root_m),
        window_local_frames=[90, 210],
        temporal_weight="sin_squared_0_to_pi",
        maximum_joint_increment_rad=0.12,
        maximum_dls_step_rad=0.04,
        dls_iterations_per_pose=4,
        damping_m_per_rad=0.03,
        landmark="original_front_two_support_sphere_midpoint_not_mesh_contact",
        modified_reference_not_original_demonstration=any(offset_root_m),
        root_translation_is_tracker_authority=False,
        physics_qualified=False,
        physical_state_or_targets_modified=False,
        new_native_steps=0,
        new_neural_inferences=0,
        optimizer_updates=0,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    if not any(offset_root_m):
        return reference, dict(
            report,
            modified_reference_hash=reference.contract_hash,
            maximum_realized_offset_m=0.0,
            maximum_reference_delta_rad=0.0,
            maximum_fk_residual_m=0.0,
            zero_is_bit_exact_identity=True,
        )
    if (
        (model.nq, model.nv, model.nu) != (43, 41, 29)
        or model.opt.timestep != 0.002
        or not 211 <= len(reference.poses) <= 1001
    ):
        raise ValueError("original G1 native model and complete contact-phase reference required")
    foot = int(model.body("right_ankle_roll_link").id)
    spheres = np.flatnonzero(
        (model.geom_bodyid == foot)
        & (model.geom_type == mujoco.mjtGeom.mjGEOM_SPHERE)
        & (model.geom_contype != 0)
    )
    if len(spheres) != 4 or not np.all(model.geom_size[spheres, 0] == 0.005):
        raise ValueError("original four5mm foot support spheres required")
    toes = spheres[model.geom_pos[spheres, 0] == model.geom_pos[spheres, 0].max()]
    if len(toes) != 2:
        raise ValueError("original front two toe support landmarks required")
    joints = np.asarray([model.joint(n).id for n in G1_DDS_JOINT_NAMES])
    qi, vi = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    ids = np.arange(6, 12)
    limits = model.jnt_range[joints[ids]].copy()
    original = np.asarray(reference.poses, dtype=np.float64)
    poses = original.copy()
    scratch = mujoco.MjData(model)
    realized, residuals = [], []
    for frame in range(91, 210):
        row = original[frame]
        baseline = row[7 + ids]
        if np.any(baseline < limits[:, 0]) or np.any(baseline > limits[:, 1]):
            raise ValueError("source right-leg reference outside native limits; no silent repair")
        scratch.qpos[:7] = row[:7]
        scratch.qpos[qi] = row[7:36]
        mujoco.mj_kinematics(model, scratch)
        before = scratch.geom_xpos[toes].mean(axis=0).copy()
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, row[3:7])
        weight = float(np.sin(np.pi * (frame - 90) / 120) ** 2)
        request = rotation.reshape(3, 3) @ np.asarray(offset_root_m) * weight
        desired = before + request
        delta = np.zeros(6)
        for _ in range(4):
            mujoco.mj_kinematics(model, scratch)
            mujoco.mj_comPos(model, scratch)
            point = scratch.geom_xpos[toes].mean(axis=0).copy()
            jac = np.zeros((3, model.nv))
            mujoco.mj_jac(model, scratch, jac, None, point, foot)
            delta = bounded_point_delta(jac[:, vi[ids]], desired - point, delta, baseline, limits)
            scratch.qpos[qi[ids]] = baseline + delta
        mujoco.mj_kinematics(model, scratch)
        after = scratch.geom_xpos[toes].mean(axis=0)
        realized.append(float(np.linalg.norm(after - before)))
        residuals.append(float(np.linalg.norm(after - desired)))
        poses[frame, 7 + ids] = baseline + delta
    untouched = [i for i in range(36) if i not in 7 + ids]
    if (
        not np.array_equal(poses[:, untouched], original[:, untouched])
        or not np.array_equal(poses[:91], original[:91])
        or not np.array_equal(poses[210:], original[210:])
    ):
        raise ValueError("reference adaptation changed untargeted joints, phase or entry")
    reference_evidence = hash_json(
        dict(
            operation=report,
            modified_pose_bytes_hash=hash_bytes(poses.tobytes()),
        )
    )
    modified = SonicPoseReference(
        tuple(tuple(float(x) for x in row) for row in poses), reference_evidence
    )
    return modified, dict(
        report,
        modified_reference_hash=modified.contract_hash,
        maximum_realized_offset_m=max(realized),
        maximum_reference_delta_rad=float(np.max(abs(poses - original))),
        maximum_fk_residual_m=max(residuals),
        zero_is_bit_exact_identity=False,
    )
