"""Anatomical, sustained physical receiving evidence from a shared-world trace.

This read-only evaluator grants no motor or promotion authority. It never
infers control from a single contact or from a video frame.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def evaluate_receiving_possession(
    *,
    ball_pose: NDArray[np.float64],
    ball_velocity: NDArray[np.float64],
    pelvis_pose: NDArray[np.float64],
    contact_agent_code: NDArray[np.integer[Any]],
    contact_foot_code: NDArray[np.integer[Any]],
    nonfoot_agent_code: NDArray[np.integer[Any]],
    agent_code: int,
    body_safe: bool,
) -> dict[str, Any]:
    """Require own clean foot contact, then 0.2 s near/still relative possession.

    The first 1 s after first own foot contact is inspected at 50 Hz. Any own
    non-foot ball contact is terminal; another player's touch breaks a hold.
    """
    if type(agent_code) is not int or agent_code <= 0 or type(body_safe) is not bool:
        raise ValueError("positive player code and independent body safety required")
    arrays = (
        (ball_pose, 7),
        (ball_velocity, 6),
        (pelvis_pose, 7),
    )
    n = len(ball_pose) if isinstance(ball_pose, np.ndarray) and ball_pose.ndim == 2 else -1
    if not 51 <= n <= 10000:
        raise ValueError("complete bounded 50 Hz physical trace required")
    for pose_array, width in arrays:
        if (
            not isinstance(pose_array, np.ndarray)
            or pose_array.shape != (n, width)
            or pose_array.dtype.kind != "f"
            or not np.isfinite(pose_array).all()
        ):
            raise ValueError("finite aligned physical poses and velocities required")
    for contact_array in (contact_agent_code, contact_foot_code, nonfoot_agent_code):
        if (
            not isinstance(contact_array, np.ndarray)
            or contact_array.shape != (n,)
            or contact_array.dtype.kind not in "iu"
            or np.any(contact_array < 0)
        ):
            raise ValueError("aligned nonnegative anatomical contact codes required")
    own_foot = np.flatnonzero((contact_agent_code == agent_code) & (contact_foot_code > 0))
    own_nonfoot = np.flatnonzero(nonfoot_agent_code == agent_code)
    first_foot = int(own_foot[0]) if len(own_foot) else None
    first_nonfoot = int(own_nonfoot[0]) if len(own_nonfoot) else None
    result: dict[str, Any] = {
        "schema": "soccer.receiving_sustained_possession.v1",
        "agent_code": agent_code,
        "body_safe": body_safe,
        "first_foot_frame": first_foot,
        "first_nonfoot_frame": first_nonfoot,
        "retained_start_frame": None,
        "best_contiguous_frames": 0,
        "qualified": False,
        "reason": "NO_OWN_FOOT_CONTACT",
    }
    if first_foot is None:
        return result
    if first_nonfoot is not None:
        result["reason"] = "OWN_NONFOOT_CONTACT"
        return result
    if not body_safe:
        result["reason"] = "BODY_UNSAFE"
        return result
    ball_xy = ball_pose[:, :2]
    pelvis_xy = pelvis_pose[:, :2]
    pelvis_vxy = np.gradient(pelvis_xy, 0.02, axis=0)
    distance = np.linalg.norm(ball_xy - pelvis_xy, axis=1)
    ball_speed = np.linalg.norm(ball_velocity[:, :2], axis=1)
    relative_speed = np.linalg.norm(ball_velocity[:, :2] - pelvis_vxy, axis=1)
    foreign_contact = ((contact_agent_code != 0) & (contact_agent_code != agent_code)) | (
        (nonfoot_agent_code != 0) & (nonfoot_agent_code != agent_code)
    )
    eligible = (
        (distance <= 0.45) & (ball_speed <= 0.35) & (relative_speed <= 0.25) & ~foreign_contact
    )
    end = min(n, first_foot + 51)
    current = 0
    best = 0
    for frame in range(first_foot, end):
        current = current + 1 if bool(eligible[frame]) else 0
        if current > best:
            best = current
        if current >= 10:
            result["retained_start_frame"] = frame - 9
            result["qualified"] = True
            result["reason"] = "SUSTAINED_POSSESSION"
            break
    result["best_contiguous_frames"] = best
    result["first_second_min_distance_m"] = float(np.min(distance[first_foot:end]))
    result["first_second_min_ball_speed_mps"] = float(np.min(ball_speed[first_foot:end]))
    result["first_second_min_relative_speed_mps"] = float(np.min(relative_speed[first_foot:end]))
    if not result["qualified"]:
        result["reason"] = "NO_SUSTAINED_NEAR_SLOW_CONTROL"
    return result
