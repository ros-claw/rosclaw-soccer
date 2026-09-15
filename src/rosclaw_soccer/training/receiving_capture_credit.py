"""Explicit capture-oriented learning credit; the physical success gate is unchanged."""

from __future__ import annotations

from typing import Any

import numpy as np

from rosclaw_soccer.training.receiving_rollout import receiving_window


def capture_retention_window(
    trace: dict[str, Any],
    *,
    agent_ids: tuple[str, ...],
    agent_id: str,
    start: int,
    frames: int,
    required_frames: int = 100,
) -> tuple[np.ndarray, dict[str, Any]]:
    # Validate the full evidence through the existing physical scoring contract.
    reward, outcome = receiving_window(
        trace,
        agent_ids=agent_ids,
        agent_id=agent_id,
        start=start,
        frames=frames,
        required_frames=required_frames,
    )
    window = slice(start, start + frames)
    ball = np.asarray(trace["ball_pose"])[window, :3]
    key = agent_id.replace(".", "_")
    distance = np.minimum(
        *[
            np.linalg.norm(np.asarray(trace[key + suffix])[window] - ball, axis=1)
            for suffix in ("_left_foot_position", "_right_foot_position")
        ]
    )
    speed = np.linalg.norm(np.asarray(trace["ball_velocity"])[window, :3], axis=1)
    # Apply on every admitted frame, not only after a rewarded touch. This
    # avoids making the cost itself disappear by refusing all foot contacts.
    cost = 0.04 * (np.minimum(distance / 0.35, 4) + np.minimum(speed / 0.35, 4))
    reward = np.asarray(reward - cost, dtype=np.float32)
    return reward, {
        **outcome,
        "schema": "soccer.receiving_capture_retention.v2",
        "success_contract": outcome["schema"],
        "dense_distance_speed_cost": float(cost.sum()),
        "shaped_return": float(reward.sum()),
    }
