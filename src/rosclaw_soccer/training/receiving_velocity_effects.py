"""Kinematic diagnostics for SIM_ONLY receiving trajectories.

These 50 Hz quantities describe observed effects, not contact impulses or
permission to promote a policy. The authoritative receiving gate is unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def _array(trace: dict[str, Any], key: str, frames: int, width: int) -> NDArray[np.float64]:
    value = np.asarray(trace[key], dtype=np.float64)
    if value.ndim != 2 or value.shape[0] != frames or value.shape[1] < width:
        raise ValueError(f"invalid {key} trace shape")
    if not np.isfinite(value).all():
        raise ValueError(f"nonfinite {key} trace")
    return value


def receiving_velocity_effects(
    trace: dict[str, Any], *, agent_id: str, agent_code: int, tail: slice = slice(110, 120)
) -> dict[str, Any]:
    """Measure first own-foot contact and its subsequent ball/body response.

    The ball velocity change is a frame-level observation; other contacts,
    gravity and ground effects may contribute. It is deliberately not called
    a force or impulse estimate.
    """
    contact = np.asarray(trace["ball_contact_agent_code"])
    foot_code = np.asarray(trace["ball_contact_foot_code"])
    nonfoot = np.asarray(trace["ball_nonfoot_contact_agent_code"])
    frames = len(contact)
    if frames < 3 or contact.shape != foot_code.shape or contact.shape != nonfoot.shape:
        raise ValueError("inconsistent contact traces")
    if not isinstance(agent_code, int) or agent_code <= 0:
        raise ValueError("positive agent code required")
    if not isinstance(agent_id, str) or not agent_id or "." not in agent_id:
        raise ValueError("qualified agent id required")
    ball_pose = _array(trace, "ball_pose", frames, 3)
    ball_velocity = _array(trace, "ball_velocity", frames, 3)
    pelvis = _array(trace, f"{agent_id.replace('.', '_')}_pelvis_pose", frames, 3)
    left = _array(trace, f"{agent_id.replace('.', '_')}_left_foot_position", frames, 3)
    right = _array(trace, f"{agent_id.replace('.', '_')}_right_foot_position", frames, 3)
    own_foot = np.flatnonzero((contact == agent_code) & (foot_code > 0))
    if not len(own_foot) or own_foot[0] < 1 or own_foot[-1] >= frames - 1:
        raise ValueError("complete own-foot contact window required")
    first = int(own_foot[0])
    end = first
    while end + 1 < frames and contact[end + 1] == agent_code and foot_code[end + 1] > 0:
        end += 1
    incoming = ball_velocity[first - 1, :2]
    outgoing = ball_velocity[end, :2]
    incoming_speed = float(np.linalg.norm(incoming))
    if incoming_speed < 1e-6:
        raise ValueError("nonzero incoming ball velocity required")
    direction = incoming / incoming_speed
    outgoing_parallel = float(np.dot(outgoing, direction))
    outgoing_lateral = float(direction[0] * outgoing[1] - direction[1] * outgoing[0])
    distance = np.minimum(
        np.linalg.norm(left - ball_pose[:, :3], axis=1),
        np.linalg.norm(right - ball_pose[:, :3], axis=1),
    )
    tail_indices = np.arange(frames)[tail]
    if not len(tail_indices) or np.any(tail_indices <= end):
        raise ValueError("tail must be nonempty and after first contact")
    own_nonfoot_after = int(np.count_nonzero(nonfoot[first:] == agent_code))
    return {
        "schema": "rosclaw_soccer.receiving_contact_diagnostics.v1",
        "agent_id": agent_id,
        "first_foot_frame": first,
        "first_contact_end_frame": end,
        "own_nonfoot_frames_after_contact": own_nonfoot_after,
        "incoming_ball_speed_mps": incoming_speed,
        "outgoing_ball_speed_mps": float(np.linalg.norm(outgoing)),
        "outgoing_parallel_mps": outgoing_parallel,
        "outgoing_lateral_mps": outgoing_lateral,
        "observed_parallel_speed_removed_mps": incoming_speed - outgoing_parallel,
        "pelvis_x_displacement_first_to_tail_m": float(
            pelvis[tail_indices[-1], 0] - pelvis[first, 0]
        ),
        "tail_maximum_ball_speed_mps": float(
            np.max(np.linalg.norm(ball_velocity[tail_indices, :3], axis=1))
        ),
        "tail_maximum_foot_distance_m": float(np.max(distance[tail_indices])),
        "task_contact_diagnostic_passed": bool(
            own_nonfoot_after == 0
            and float(np.max(np.linalg.norm(ball_velocity[tail_indices, :3], axis=1))) <= 0.35
            and float(np.max(distance[tail_indices])) <= 0.35
        ),
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
    }
