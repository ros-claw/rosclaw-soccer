"""Goal-conditioned receiving credit; never replaces physical success checks."""

from __future__ import annotations

from typing import Any

import numpy as np

from rosclaw_soccer.training.receiving_capture_credit import capture_retention_window


def receiving_transition_window(
    trace: dict[str, Any],
    *,
    agent_ids: tuple[str, ...],
    agent_id: str,
    start: int,
    frames: int,
    next_target_xy: tuple[float, float],
    required_frames: int = 125,
) -> tuple[np.ndarray, dict[str, Any]]:
    target = np.asarray(next_target_xy, dtype=np.float64)
    if target.shape != (2,) or not np.isfinite(target).all():
        raise ValueError("finite explicit next-skill target required")
    reward, outcome = capture_retention_window(
        trace,
        agent_ids=agent_ids,
        agent_id=agent_id,
        start=start,
        frames=frames,
        required_frames=required_frames,
    )
    window = slice(start, start + frames)
    ball = np.asarray(trace["ball_pose"])[window, :2]
    body = np.asarray(trace[agent_id.replace(".", "_") + "_pelvis_pose"])[window]
    direction = target - ball
    length = np.linalg.norm(direction, axis=1)
    if np.any(length < 1e-6):
        raise ValueError("next-skill target must be distinct from the ball")
    direction = direction / length[:, None]
    relative = ball - body[:, :2]
    depth = np.sum(relative * direction, axis=1)
    lateral = np.abs(relative[:, 0] * direction[:, 1] - relative[:, 1] * direction[:, 0])
    qw, qx, qy, qz = body[:, 3:7].T
    heading = np.stack((1 - 2 * (qy * qy + qz * qz), 2 * (qw * qz + qx * qy)), axis=1)
    heading_length = np.linalg.norm(heading, axis=1)
    if np.any(heading_length < 1e-6):
        raise ValueError("horizontal receiving heading required")
    alignment = np.sum(heading / heading_length[:, None] * direction, axis=1)
    # A broad entry pocket, not a new success gate or a desired joint pose.
    # Costs remain active without contact, so avoiding the ball cannot hide them.
    geometry_cost = 0.04 * (
        np.minimum((np.maximum(0.30 - depth, 0) + np.maximum(depth - 0.65, 0)) / 0.35, 4)
        + np.minimum(np.maximum(lateral - 0.20, 0) / 0.35, 4)
        + (1 - np.clip(alignment, -1, 1)) / 2
    )
    reward = np.asarray(reward - geometry_cost, dtype=np.float32)
    return reward, {
        **outcome,
        "schema": "soccer.receiving_transition_credit.v1",
        "next_target_xy": target.tolist(),
        "transition_geometry_cost": float(geometry_cost.sum()),
        "terminal_depth_m": float(depth[-1]),
        "terminal_lateral_m": float(lateral[-1]),
        "terminal_heading_alignment": float(alignment[-1]),
        "shaped_return": float(reward.sum()),
        "transition_success_claim": False,
    }
