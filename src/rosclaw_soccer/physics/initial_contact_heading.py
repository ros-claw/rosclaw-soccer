"""Measured initial contact headings, not a task label or causal identification."""

import math
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES


def heading_deg(value: Any) -> float | None:
    v = np.asarray(value)
    if v.shape != (3,) or not np.isfinite(v).all():
        raise ValueError("finite measured world vector required")
    return (
        None
        if math.hypot(float(v[0]), float(v[1])) < 1e-12
        else math.degrees(math.atan2(float(v[1]), float(v[0])))
    )


def initial_contact_heading(row: dict[str, Any], arrays: dict[str, Any]) -> dict[str, Any]:
    validated = {}
    for key, width in (
        ("canonical_qpos_before_substep", 43),
        ("canonical_qvel_after_substep", 41),
        ("robot_contact_force_on_ball_world_n", 3),
    ):
        a = np.asarray(arrays.get(key))
        if a.shape != (3000, width) or a.dtype != np.float64 or not np.isfinite(a).all():
            raise ValueError("whole finite canonical G1 fixed-control replay arrays required")
        validated[key] = a
    arrays = validated
    event = row["first_foot_contact_event"]
    feet = ("left_ankle_roll_link", "right_ankle_roll_link")
    foot_events = [e for e in row["robot_contact_events_over_1n"] if e["other_body_name"] in feet]
    if (event is None) != (not foot_events):
        raise ValueError("first-foot presence must agree with all original contact events")
    result: dict[str, Any] = dict(
        case_id=row["case_id"],
        source_receipt_hash=row["source_receipt_hash"],
        original_outcome=row["original_outcome"],
        contact=None,
    )
    if event is not None:
        first = event["substep_index"]
        if (
            type(first) is not int
            or not 0 <= first < 3000
            or event not in row["robot_contact_events_over_1n"]
            or first != min(e["substep_index"] for e in foot_events)
            or event["other_body_name"] not in feet
            or type(event["force_norm_n"]) not in (int, float)
            or not math.isfinite(event["force_norm_n"])
            or event["force_norm_n"] <= 1
        ):
            raise ValueError("actual source-owned monitored foot event over1N required")
        end = min(first + 10, 3000)
        impulse = arrays["robot_contact_force_on_ball_world_n"][first:end].sum(axis=0) * 0.002
        velocity = arrays["canonical_qvel_after_substep"][end - 1, 35:38]
        center = arrays["canonical_qpos_before_substep"][first, 36:39]
        origin = np.asarray(event["other_body_origin_world_m"], dtype=np.float64)
        normal = np.asarray(event["normal_other_to_ball_world"], dtype=np.float64)
        heading_deg(origin)
        if (
            normal.shape != (3,)
            or not np.isfinite(normal).all()
            or not np.isclose(np.linalg.norm(normal), 1.0, atol=1e-6, rtol=0)
        ):
            raise ValueError("unit physical other-to-ball contact normal required")
        result["contact"] = dict(
            first_foot_substep=first,
            body=event["other_body_name"],
            first_normal_heading_deg=heading_deg(normal),
            first_force_heading_deg=heading_deg(event["force_on_ball_world_n"]),
            robot_impulse_initial_window_world_n_s=impulse.tolist(),
            robot_impulse_initial_window_heading_deg=heading_deg(impulse),
            ball_velocity_after_initial_window_world_m_s=velocity.tolist(),
            ball_velocity_after_initial_window_heading_deg=heading_deg(velocity),
            ball_center_minus_foot_body_origin_world_m=(center - origin).tolist(),
            initial_window_start_substep=first,
            initial_window_end_exclusive=end,
            initial_window_duration_s=(end - first) * 0.002,
            initial20ms_fully_observed=(end - first) == 10,
        )
    return dict(
        result,
        foot_body_origin_not_surface_distance=True,
        first_collision_not_all_later_contacts=True,
        headings_not_new_task_gates=True,
        causality_identified=False,
        new_physical_steps=0,
        optimizer_updates=0,
        hardware_authorized=False,
        promotion_authorized=False,
    )


def reconstruct_contact_orientation(
    model: Any, row: dict[str, Any], arrays: dict[str, Any]
) -> dict[str, Any] | None:
    """Actual collision-stage FK from saved before-substep qpos, not integration."""
    import mujoco

    # Validate the canonical arrays and event before allocating even a kinematic state.
    initial_contact_heading(row, arrays)
    event = row["first_foot_contact_event"]
    if event is None:
        return None
    if (model.nq, model.nv, model.nu) != (43, 41, 29):
        raise ValueError("original canonical43/41/29 G1 world required")
    q = np.asarray(arrays["canonical_qpos_before_substep"])[event["substep_index"]]
    qi = np.asarray([int(model.jnt_qposadr[model.joint(n).id]) for n in G1_DDS_JOINT_NAMES])
    ball = int(model.body("ball").id)
    bq = int(model.jnt_qposadr[model.body_jntadr[ball]])
    if sorted([*range(7), *qi, *range(bq, bq + 7)]) != list(range(43)) or any(
        not np.isclose(np.linalg.norm(v), 1.0, atol=1e-8, rtol=0) for v in (q[3:7], q[39:43])
    ):
        raise ValueError("canonical complete state and normalized root/ball quaternions required")
    data = mujoco.MjData(model)
    data.qpos[:7], data.qpos[qi], data.qpos[bq : bq + 7] = q[:7], q[7:36], q[36:43]
    mujoco.mj_kinematics(model, data)
    foot = int(model.body(event["other_body_name"]).id)
    error = float(np.max(np.abs(data.xpos[foot] - event["other_body_origin_world_m"])))
    if error > 1e-10:
        raise ValueError("reconstructed original collision-stage foot origin differs")
    rotation = data.xmat[foot].reshape(3, 3)
    pelvis_rotation = data.xmat[int(model.body("pelvis").id)].reshape(3, 3)
    normal = np.asarray(event["normal_other_to_ball_world"], dtype=np.float64)
    return dict(
        foot_forward_heading_world_deg=heading_deg(rotation[:, 0]),
        pelvis_forward_heading_world_deg=heading_deg(pelvis_rotation[:, 0]),
        normal_in_foot_body_frame=(rotation.T @ normal).tolist(),
        foot_origin_maximum_reconstruction_error_m=error,
        kinematics_from_original_collision_clock=True,
        joint_angles_not_intervened=True,
        dynamic_counterfactual_performed=False,
        new_physical_steps=0,
    )
