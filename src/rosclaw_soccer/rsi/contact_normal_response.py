"""Goal-side foot-position experiments from actual solver contact geometry."""

from typing import Any

import numpy as np


def measured_side_offset(
    events: list[dict[str, Any]],
    goal: np.ndarray,
    qpos: np.ndarray,
    *,
    initial_time: float,
    role: int,
) -> np.ndarray:
    if (
        qpos.ndim != 2
        or qpos.shape[1] != 43
        or goal.shape != (3,)
        or role not in (0, 1)
        or not np.isfinite(qpos).all()
        or not np.isfinite(goal).all()
        or not np.isfinite(initial_time)
    ):
        raise ValueError("actual canonical native geometry and intended foot required")
    total = np.zeros(2)
    weight = 0.0
    for event in events:
        if event["role"] != role:
            continue
        step = (event["solver_time_s"] - initial_time) / 0.002
        index = round(step)
        if abs(step - index) > 1e-6 or not 1 <= index < len(qpos):
            raise ValueError("contact must bind exact prior native body pose, never interpolation")
        w, x, y, z = qpos[index - 1, 3:7]
        if abs(w * w + x * x + y * y + z * z - 1) > 1e-8:
            raise ValueError("actual unit pelvis orientation required")
        rotation = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )
        center = np.array(event["ball_solver_center_world"], np.float64)
        point = np.array(event["contact_position_world"], np.float64)
        axis = goal[:2] - center[:2]
        if np.linalg.norm(axis) < 1e-8:
            raise ValueError("nondegenerate actual goal direction required")
        axis /= np.linalg.norm(axis)
        side = np.array([-axis[1], axis[0], 0.0])
        lateral = float((center - point) @ side)
        force = float(event["force_norm_N"])
        if not np.isfinite(force) or force <= 10 or not np.isfinite(lateral):
            raise ValueError("actual intended-foot thresholded contact required")
        total += force * (rotation.T @ (side * lateral))[:2]
        weight += force
    if not weight:
        raise ValueError("observed intended-foot contact required")
    result = total / weight
    if not np.isfinite(result).all() or np.linalg.norm(result) > 0.08:
        raise ValueError("local contact centering outside declared exploratory authority")
    return result


def contact_normal_proposals(
    base: np.ndarray, *, center: int, contact_control: int, local_xy_m: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    if (
        base.shape != (2, 6)
        or not np.isfinite(base).all()
        or (np.abs(base) > 1).any()
        or local_xy_m.shape != (2,)
        or not np.isfinite(local_xy_m).all()
        or np.linalg.norm(local_xy_m) > 0.08
        or type(center) is not int
        or not 30 <= center <= 200
        or type(contact_control) is not int
        or not 0 <= contact_control < 300
    ):
        raise ValueError("actual bounded foot position and source contact phase required")
    t = (contact_control - np.array([center - 12, center + 8])) / 30.0
    weights = np.where(np.abs(t) < 1, (1 - t * t) ** 3, 0.0)
    if weights.sum() < 0.25:
        raise ValueError("contact outside actual foot-intent phase authority")
    delta = local_xy_m / (np.array([0.04, 0.035]) * weights.sum())
    multipliers = np.array([0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, -0.25])
    proposals = np.repeat(base[None], 11, axis=0)
    proposals[:, :, :2] = np.clip(
        base[None, :, :2] + multipliers[:, None, None] * delta[None, None, :], -1, 1
    )
    return proposals, dict(
        measured_goal_side_centering_body_xy_m=local_xy_m.tolist(),
        center=center,
        actual_contact_control=contact_control,
        phase_weights=weights.tolist(),
        multipliers=multipliers.tolist(),
        xy_delta_per_basis=delta.tolist(),
        height_and_rotation_latents_unchanged=True,
        physical_success_predicted=False,
    )
