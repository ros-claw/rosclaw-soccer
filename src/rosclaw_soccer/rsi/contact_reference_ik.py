"""Actuated-only contact configuration for a future reference, NOT execution.

Foot surface placement, support pose and shin separation are solved together.
All modifications are on a scratch FK body. A feasible result is only geometry;
the original neural tracker and native physics must still validate any motion.
"""

from typing import Any

import numpy as np

from rosclaw_soccer.physics.bounded_joint_tasks import LinearJointTask, solve_bounded_joint_tasks
from rosclaw_soccer.rsi.ball_surface_tasks import measure_ball_surfaces


def solve_contact_reference_pose(
    model: Any, measured: Any, layout: Any, *, goal: np.ndarray
) -> dict[str, Any]:
    import mujoco

    if (
        layout.model is not model
        or type(goal) is not np.ndarray
        or goal.shape != (3,)
        or goal.dtype != np.float64
        or not np.isfinite(goal).all()
    ):
        raise ValueError("own typed FK model and finite world goal required")
    if not np.isfinite(measured.qpos).all() or not np.isfinite(measured.qvel).all():
        raise ValueError("finite actual seed required, no state repair")
    scratch = mujoco.MjData(model)
    mujoco.mj_copyData(scratch, model, measured)
    mujoco.mj_kinematics(model, scratch)
    mujoco.mj_comPos(model, scratch)
    ball_position = scratch.xpos[layout.ball].copy()
    direction = goal - ball_position
    direction[2] = 0
    norm = float(np.linalg.norm(direction))
    if norm < 0.1:
        raise ValueError("nondegenerate horizontal kick direction required")
    direction /= norm
    ball_geoms = np.flatnonzero(model.geom_bodyid == layout.ball)
    if len(ball_geoms) != 1:
        raise ValueError("one qualified sphere required")
    radius = float(model.geom_size[ball_geoms[0], 0])
    surfaces = measure_ball_surfaces(model, scratch, layout)["surfaces"]
    feet = [side for side in ("left", "right") if surfaces[side + "_foot"]["available"]]
    if len(feet) != 2:
        raise ValueError("two measured actual foot surfaces required")
    active = min(feet, key=lambda side: surfaces[side + "_foot"]["gap_m"])
    support = "right" if active == "left" else "left"
    active_body = int(model.body(layout.robot_prefix + active + "_ankle_roll_link").id)
    support_body = int(model.body(layout.robot_prefix + support + "_ankle_roll_link").id)
    local_point = surfaces[active + "_foot"]["local_point"].copy()
    contact_goal = ball_position - radius * direction
    support_position = scratch.xpos[support_body].copy()
    support_rotation = scratch.xmat[support_body].reshape(3, 3).copy()
    original = scratch.qpos[layout.qi].copy()
    original_qpos = scratch.qpos.copy()
    trace = []
    status = "ITERATION_BUDGET_EXHAUSTED"

    def measurements() -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, float]]:
        mujoco.mj_kinematics(model, scratch)
        mujoco.mj_comPos(model, scratch)
        foot_point = (
            scratch.xpos[active_body] + scratch.xmat[active_body].reshape(3, 3) @ local_point
        )
        rows = measure_ball_surfaces(model, scratch, layout)["surfaces"]
        rotation = scratch.xmat[support_body].reshape(3, 3)
        relative = support_rotation @ rotation.T
        rotation_error = 0.5 * np.array(
            [
                relative[2, 1] - relative[1, 2],
                relative[0, 2] - relative[2, 0],
                relative[1, 0] - relative[0, 1],
            ]
        )
        angle = float(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1)))
        shins = [rows[side + "_shin"] for side in ("left", "right")]
        metrics = dict(
            contact_point_error_m=float(np.linalg.norm(contact_goal - foot_point)),
            support_position_error_m=float(
                np.linalg.norm(support_position - scratch.xpos[support_body])
            ),
            support_angle_error_rad=angle,
            minimum_shin_gap_m=float(min(s["gap_m"] for s in shins)),
            foot_surface_gap_m=float(rows[active + "_foot"]["gap_m"]),
        )
        return foot_point, rotation_error, shins, metrics

    for iteration in range(16):
        foot_point, rotation_error, shins, metrics = measurements()
        trace.append(dict(iteration=iteration, **metrics))
        if (
            metrics["contact_point_error_m"] <= 0.005
            and metrics["support_position_error_m"] <= 0.005
            and metrics["support_angle_error_rad"] <= 0.02
            and metrics["minimum_shin_gap_m"] >= 0.015
            and -0.002 <= metrics["foot_surface_gap_m"] <= 0.005
        ):
            status = "KINEMATIC_CONTACT_CONFIGURATION_FEASIBLE"
            break
        if any(not s["smooth_separated"] for s in shins):
            status = "ABSTAIN_NONSMOOTH_SHIN_GEOMETRY"
            break
        if abs(contact_goal - foot_point).max() > 0.25:
            status = "ABSTAIN_LOCAL_CONTACT_OBJECTIVE_TOO_FAR"
            break
        active_jac = np.zeros((3, model.nv))
        support_jac = np.zeros((3, model.nv))
        rotation_jac = np.zeros((3, model.nv))
        mujoco.mj_jac(model, scratch, active_jac, None, foot_point, active_body)
        mujoco.mj_jac(
            model, scratch, support_jac, rotation_jac, scratch.xpos[support_body], support_body
        )
        tasks = [
            LinearJointTask(
                "foot_contact", active_jac[:, layout.vi].copy(), contact_goal - foot_point, 100.0
            ),
            LinearJointTask(
                "support_position",
                support_jac[:, layout.vi].copy(),
                support_position - scratch.xpos[support_body],
                100.0,
            ),
            LinearJointTask(
                "support_orientation",
                0.1 * rotation_jac[:, layout.vi].copy(),
                0.1 * rotation_error,
                100.0,
            ),
        ]
        for side, shin in zip(("left", "right"), shins, strict=True):
            jac = np.zeros((3, 29))
            jac[0] = shin["gap_jacobian_hg"]
            displacement = np.array([max(0.0, 0.02 - shin["gap_m"]), 0.0, 0.0])
            tasks.append(LinearJointTask(side + "_shin_clearance", jac, displacement, 100.0))
        current = scratch.qpos[layout.qi].copy()
        proposal = solve_bounded_joint_tasks(
            current,
            original,
            current,
            model.jnt_range[layout.joints].astype(np.float64),
            tuple(tasks),
            observation_step_cap_rad=0.2,
            command_step_cap_rad=0.1,
            nominal_trust_rad=0.35,
            posture_cost=0.1,
            active_joint_indices=tuple(range(12)),
        )
        if proposal.target_rad is None:
            status = proposal.status
            break
        candidate = np.array(proposal.target_rad)
        accepted = False
        for fraction in (1.0, 0.5, 0.25, 0.125):
            scratch.qpos[layout.qi] = current + fraction * (candidate - current)
            _, _, trial_shins, _ = measurements()
            if all(s["smooth_separated"] for s in trial_shins):
                accepted = True
                break
        if not accepted:
            scratch.qpos[layout.qi] = current
            status = "ABSTAIN_NO_SMOOTH_BOUNDED_STEP"
            break
    _, _, _, metrics = measurements()
    mask = np.ones(model.nq, dtype=bool)
    mask[layout.qi] = False
    if not np.array_equal(scratch.qpos[mask], original_qpos[mask]) or not np.array_equal(
        measured.qpos, original_qpos
    ):
        raise ValueError("reference IK modified actual state, root or ball")
    return dict(
        schema="soccer.contact_reference_geometry.v1",
        status=status,
        active_foot=active,
        contact_point_local=local_point.tolist(),
        contact_goal_world=contact_goal.tolist(),
        nominal_joints_hg=original.tolist(),
        reference_joints_hg=scratch.qpos[layout.qi].tolist(),
        maximum_reference_offset_rad=float(abs(scratch.qpos[layout.qi] - original).max()),
        metrics=metrics,
        iterations=trace,
        new_physical_steps=0,
        new_neural_calls=0,
        physically_qualified=False,
        hardware_authorized=False,
    )
