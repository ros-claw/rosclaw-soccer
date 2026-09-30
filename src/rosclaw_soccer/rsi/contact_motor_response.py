"""Measured pre-contact control authority; never a policy or promotion gate."""

from __future__ import annotations

from typing import Any

import numpy as np


def precontact_response(
    parent: Any, candidate: Any, parent_first: int | None, candidate_first: int | None
) -> dict[str, Any]:
    shapes = {
        "root_pose_xyzw_m": (300, 1, 7),
        "joint_target_rad": (300, 1, 29),
        "foot_geometry_position_before_step_m": (300, 1, 4, 3),
        "ball_position_before_step_m": (300, 1, 3),
    }
    if any(
        key not in trace or trace[key].shape != shape or not np.isfinite(trace[key]).all()
        for trace in (parent, candidate)
        for key, shape in shapes.items()
    ) or any(
        first is not None and (type(first) is not int or not 0 <= first < 300)
        for first in (parent_first, candidate_first)
    ):
        raise ValueError("aligned finite independent body traces required")
    # Exclude the first post-step collision response from BOTH branches. The
    # observations here are pre-step, so the contact frame itself is allowed.
    stop = min(299 if first is None else first for first in (parent_first, candidate_first))
    active = slice(0, stop + 1)
    root = candidate["root_pose_xyzw_m"][active, 0, :2] - parent["root_pose_xyzw_m"][active, 0, :2]
    feet = (
        candidate["foot_geometry_position_before_step_m"][active, 0, :2]
        - parent["foot_geometry_position_before_step_m"][active, 0, :2]
    )
    target = candidate["joint_target_rad"][active, 0] - parent["joint_target_rad"][active, 0]
    differences = np.flatnonzero(np.max(np.abs(target), axis=1) > 1e-6)
    return {
        "last_compared_precontact_frame": stop,
        "first_joint_target_difference_frame": int(differences[0]) if len(differences) else None,
        "max_root_xy_response_before_contact_m": float(np.max(np.linalg.norm(root, axis=-1))),
        "max_foot_link_response_before_contact_m": float(np.max(np.linalg.norm(feet, axis=-1))),
        "max_joint_target_response_before_contact_rad": float(np.max(np.abs(target))),
        "not_claimed": (
            "link origins are not exact collider clearance; response does not prove task success"
        ),
    }
