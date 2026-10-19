"""Bounded foot-heading numerical proposal, not an execution permit.

Native use requires an explicit typed request and independent qualification.

Projects a small yaw correction into the instantaneous position-Jacobian null
space. First-order translation preservation is not dynamic balance, realized
foot motion, impact direction, or task improvement. World+x is the diagnostic
course direction, not a universal football heading or a learned motor policy.
"""

import math
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

MAX_HEADING_REQUEST_RAD = math.radians(10.0)
HEADING_DEADBAND_RAD = math.radians(10.0)
MAX_ADDITIONAL_JOINT_RAD = 0.1
MAX_TOTAL_INCREMENT_RAD = 0.35


def propose_foot_heading(
    linear_jacobian: np.ndarray,
    angular_jacobian: np.ndarray,
    foot_forward_axis_world: np.ndarray,
    navigation_target: np.ndarray,
    original_ik_increment: np.ndarray,
    joint_limits: np.ndarray,
    *,
    enabled: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a bounded numerical proposal and diagnostics; no simulator calls."""
    for a, shape in (
        (linear_jacobian, (3, 6)),
        (angular_jacobian, (3, 6)),
        (foot_forward_axis_world, (3,)),
        (navigation_target, (6,)),
        (original_ik_increment, (6,)),
        (joint_limits, (6, 2)),
    ):
        if (
            not isinstance(a, np.ndarray)
            or a.shape != shape
            or a.dtype != np.float64
            or not np.isfinite(a).all()
            or np.max(np.abs(a)) > 1e6
        ):
            raise ValueError("whole finite float64 six-joint kinematic proposal inputs required")
    if (
        type(enabled) is not bool
        or np.any(joint_limits[:, 0] >= joint_limits[:, 1])
        or np.max(np.abs(original_ik_increment)) > MAX_TOTAL_INCREMENT_RAD + 1e-12
        or not np.isclose(np.linalg.norm(foot_forward_axis_world), 1.0, atol=1e-8, rtol=0)
    ):
        raise ValueError(
            "explicit OFF/ON intent, unit forward axis and original increment limits required"
        )
    total = original_ik_increment
    heading = math.atan2(float(foot_forward_axis_world[1]), float(foot_forward_axis_world[0]))
    request = -math.copysign(
        min(max(abs(heading) - HEADING_DEADBAND_RAD, 0.0), MAX_HEADING_REQUEST_RAD), heading
    )
    requested = np.zeros(6)
    applied = np.zeros(6)
    yaw_jacobian = np.zeros(6)
    rank: int | None = None
    scale = 0.0
    nominal = navigation_target + original_ik_increment
    planar = float(np.dot(foot_forward_axis_world[:2], foot_forward_axis_world[:2]))
    status = "OFF_UNCHANGED"
    if enabled:
        if np.any(nominal < joint_limits[:, 0]) or np.any(nominal > joint_limits[:, 1]):
            status = "BASELINE_OUTSIDE_LIMITS_UNCHANGED"
        elif planar < 0.25 or abs(heading) > math.radians(60):
            status = "OUTSIDE_DECLARED_HEADING_DOMAIN_UNCHANGED"
        else:
            fx, fy, fz = foot_forward_axis_world
            yaw_jacobian = (
                angular_jacobian[2]
                - fz * (fx * angular_jacobian[0] + fy * angular_jacobian[1]) / planar
            )
            _, singular, vh = np.linalg.svd(linear_jacobian, full_matrices=True)
            rank = int(np.count_nonzero(singular > max(1e-10, float(singular[0]) * 1e-8)))
            null_basis = vh[rank:].T
            direction = null_basis @ (null_basis.T @ yaw_jacobian)
            authority = float(yaw_jacobian @ direction)
            status = "NO_POSITION_NULLSPACE_YAW_AUTHORITY_UNCHANGED"
            if abs(request) <= 1e-12:
                status = "ALREADY_ALIGNED_UNCHANGED"
            if authority > 1e-10 and abs(request) > 1e-12:
                requested = direction * (request / authority)
                scale = min(1.0, MAX_ADDITIONAL_JOINT_RAD / float(np.max(np.abs(requested))))
                for i, value in enumerate(requested):
                    if value > 0:
                        scale = min(
                            scale,
                            (MAX_TOTAL_INCREMENT_RAD - original_ik_increment[i]) / value,
                            (joint_limits[i, 1] - nominal[i]) / value,
                        )
                    elif value < 0:
                        scale = min(
                            scale,
                            (-MAX_TOTAL_INCREMENT_RAD - original_ik_increment[i]) / value,
                            (joint_limits[i, 0] - nominal[i]) / value,
                        )
                scale = max(0.0, scale)
                applied = requested * scale
                if np.max(np.abs(linear_jacobian @ applied)) > 1e-8:
                    applied = np.zeros(6)
                    scale = 0.0
                    status = "NUMERICAL_POSITION_LEAK_UNCHANGED"
                elif scale > 0:
                    total = original_ik_increment + applied
                    status = "BOUNDED_KINEMATIC_PROPOSAL_NOT_PHYSICS_QUALIFIED"
                else:
                    status = "NO_JOINT_LIMIT_ROOM_UNCHANGED"
    result = dict(
        schema="soccer.rsi.foot_heading_nullspace_proposal.v1",
        status=status,
        source_hash=hash_bytes(Path(__file__).read_bytes()),
        enabled=enabled,
        measured_foot_heading_rad=heading,
        requested_heading_change_rad=request if enabled else 0.0,
        additional_joint_request_rad=requested.tolist(),
        additional_joint_applied_rad=applied.tolist(),
        uniform_limit_scale=scale,
        position_jacobian_rank=rank,
        predicted_additional_heading_change_rad=float(yaw_jacobian @ applied),
        predicted_position_increment_m=(linear_jacobian @ applied).tolist(),
        final_ik_increment_rad=total.tolist(),
        maximum_heading_request_rad=MAX_HEADING_REQUEST_RAD,
        heading_deadband_rad=HEADING_DEADBAND_RAD,
        maximum_additional_joint_rad=MAX_ADDITIONAL_JOINT_RAD,
        maximum_total_ik_increment_rad=MAX_TOTAL_INCREMENT_RAD,
        linearization_only=True,
        no_integration_or_controller_calls=True,
        new_physical_steps=0,
        optimizer_updates=0,
        trained_policy=False,
        physics_qualified=False,
        runtime_execution_authorized=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    return total, dict(result, report_hash=hash_json(result))


def check_recorded_pose_proposal(
    model: Any, canonical_qpos: np.ndarray, foot_body_name: str, *, enabled: bool
) -> dict[str, Any]:
    """Forward-kinematic before/after only; no mj_step, PD, SONIC or control calls.

    Uses a recorded state selected retrospectively by the analysis caller. This
    is NOT a prospective actor observation or permission to change a trajectory.
    """
    import mujoco

    if (
        (model.nq, model.nv, model.nu) != (43, 41, 29)
        or canonical_qpos.shape != (43,)
        or canonical_qpos.dtype != np.float64
        or not np.isfinite(canonical_qpos).all()
        or foot_body_name not in ("left_ankle_roll_link", "right_ankle_roll_link")
        or any(
            not np.isclose(np.linalg.norm(v), 1.0, atol=1e-8, rtol=0)
            for v in (canonical_qpos[3:7], canonical_qpos[39:43])
        )
    ):
        raise ValueError("original canonical G1 pose and monitored foot required")
    joints = [int(model.joint(n).id) for n in G1_DDS_JOINT_NAMES]
    qi = np.asarray([int(model.jnt_qposadr[j]) for j in joints])
    vi = np.asarray([int(model.jnt_dofadr[j]) for j in joints])
    ball = int(model.body("ball").id)
    bq = int(model.jnt_qposadr[model.body_jntadr[ball]])
    if sorted([*range(7), *qi, *range(bq, bq + 7)]) != list(range(43)):
        raise ValueError("whole nonoverlapping canonical joint address mapping required")
    data = mujoco.MjData(model)
    data.qpos[:7], data.qpos[qi], data.qpos[bq : bq + 7] = (
        canonical_qpos[:7],
        canonical_qpos[7:36],
        canonical_qpos[36:43],
    )
    mujoco.mj_kinematics(model, data)
    # Official minimum Jacobian pipeline: kinematics -> comPos -> jacBody.
    # https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-jac
    mujoco.mj_comPos(model, data)
    foot = int(model.body(foot_body_name).id)
    linear, angular = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, linear, angular, foot)
    before_position = data.xpos[foot].copy()
    before_forward = data.xmat[foot].reshape(3, 3)[:, 0].copy()
    side = 0 if foot_body_name == "left_ankle_roll_link" else 1
    ids = np.arange(6 * side, 6 * side + 6)
    correction, proposal = propose_foot_heading(
        linear[:, vi[ids]],
        angular[:, vi[ids]],
        before_forward,
        canonical_qpos[7:36][ids].copy(),
        np.zeros(6),
        model.jnt_range[np.asarray(joints)[ids]].copy(),
        enabled=enabled,
    )
    data.qpos[qi[ids]] += correction
    mujoco.mj_kinematics(model, data)
    after_forward = data.xmat[foot].reshape(3, 3)[:, 0]
    before_heading = math.atan2(float(before_forward[1]), float(before_forward[0]))
    after_heading = math.atan2(float(after_forward[1]), float(after_forward[0]))
    result = dict(
        schema="soccer.rsi.recorded_pose_heading_proposal_check.v1",
        proposal=proposal,
        measured_original_heading_rad=before_heading,
        forward_kinematic_proposed_heading_rad=after_heading,
        body_origin_displacement_m=(data.xpos[foot] - before_position).tolist(),
        body_origin_displacement_norm_m=float(np.linalg.norm(data.xpos[foot] - before_position)),
        preserves_body_origin_not_contact_point=True,
        retrospective_pose_analysis_not_actor_observation=True,
        original_ik_and_parent_not_applied_here=True,
        physical_integration_performed=False,
        balance_or_ball_outcome_verified=False,
        new_physical_steps=0,
        new_neural_inferences=0,
        optimizer_updates=0,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    return dict(result, report_hash=hash_json(result))
