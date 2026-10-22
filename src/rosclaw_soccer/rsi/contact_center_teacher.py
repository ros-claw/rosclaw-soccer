"""Measured TRAIN contact centering, not learned policy or measured force direction."""

from typing import Any

import numpy as np


def measured_centering(events: list[dict[str, Any]], goal: np.ndarray) -> tuple[np.ndarray, int]:
    if goal.shape != (3,) or not np.isfinite(goal).all():
        raise ValueError("finite world goal required")
    rows = [e for e in events if e["phase"] == "kick" and e["role"] == 0]
    if not rows:
        raise ValueError("actual left-foot solver events required")
    corrections, weights = [], []
    for e in rows:
        center = np.asarray(e["ball_solver_center_world"], dtype=np.float64)
        point = np.asarray(e["contact_position_world"], dtype=np.float64)
        weight = e["force_norm_N"]
        if (
            center.shape != (3,)
            or point.shape != (3,)
            or not np.isfinite(np.r_[center, point, weight]).all()
            or weight <= 10
            or e["robot_body"] != "left_ankle_roll_link"
        ):
            raise ValueError("finite thresholded actual left ankle events required")
        heading = np.r_[goal[:2] - center[:2], 0.0]
        if np.linalg.norm(heading) < 1e-8:
            raise ValueError("nondegenerate goal heading required")
        # New protocol explicitly holds the existing .11m sphere fixed.
        desired_radial = 0.11 * heading / np.linalg.norm(heading)
        corrections.append(center - point - desired_radial)
        weights.append(weight)
    offset = np.average(corrections, axis=0, weights=weights)
    peak = max(rows, key=lambda e: e["force_norm_N"])
    frame = peak["reference_frame"]
    if np.linalg.norm(offset) > 0.08 or type(frame) is not int or not 80 <= frame <= 160:
        raise ValueError("measured bounded TRAIN centering/phase required")
    return offset, frame


def current_body_offset(
    world_offset: np.ndarray, pelvis_rotation: np.ndarray, frame: int, center: int
) -> np.ndarray:
    if (
        world_offset.shape != (3,)
        or pelvis_rotation.shape != (3, 3)
        or not np.isfinite(world_offset).all()
        or not np.isfinite(pelvis_rotation).all()
        or np.linalg.norm(world_offset) > 0.08
        or not np.allclose(pelvis_rotation.T @ pelvis_rotation, np.eye(3), atol=1e-8)
        or not np.isclose(np.linalg.det(pelvis_rotation), 1, atol=1e-8)
        or type(frame) is not int
        or type(center) is not int
    ):
        raise ValueError("bounded measured offset and current proper body rotation required")
    t = (frame - center) / 24.0
    weight = (1 - t * t) ** 3 if abs(t) < 1 else 0.0
    return np.asarray(np.r_[pelvis_rotation.T @ world_offset * weight, np.zeros(3)])
