"""Strict numeric team trace extraction for SIM evaluation and visualization.

Frames are observation snapshots, not substep safety certificates. Physical
receipts, source/model identity and collision evidence remain separate gates.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np

ROSTER = tuple(
    f"{team}_{role}"
    for team in ("red", "blue")
    for role in ("defender", "finisher", "goalkeeper", "playmaker")
)


def extract_team_trace(trace: Mapping[str, Any]) -> dict[str, np.ndarray]:
    if not isinstance(trace, Mapping) or "ball_pose" not in trace:
        raise ValueError("recorded shared-world trace required")
    ball = np.asarray(trace["ball_pose"])
    if ball.ndim != 2 or ball.shape[1] != 7 or not 1 <= len(ball) <= 100000:
        raise ValueError("bounded ball pose timeline required")
    frames = len(ball)
    expected = {
        "time": (frames,),
        "ball_pose": (frames, 7),
        "ball_velocity": (frames, 6),
        **{
            key: (frames,)
            for key in (
                "ball_contact_agent_code",
                "ball_contact_effector_code",
                "ball_contact_force_n",
                "ball_nonfoot_contact_agent_code",
                "ball_nonfoot_contact_force_n",
                "robot_robot_contact_count",
            )
        },
    }
    for agent in ROSTER:
        expected.update(
            {
                f"{agent}_pelvis_pose": (frames, 7),
                f"{agent}_joint_position": (frames, 29),
                f"{agent}_joint_velocity": (frames, 29),
                f"{agent}_joint_torque": (frames, 29),
                f"{agent}_joint_safety_margin_rad": (frames, 29),
                f"{agent}_left_foot_position": (frames, 3),
                f"{agent}_right_foot_position": (frames, 3),
            }
        )
    result = {}
    for key, shape in expected.items():
        if key not in trace:
            raise ValueError(f"missing shared-world trace: {key}")
        value = np.asarray(trace[key])
        if value.shape != shape or value.dtype.kind not in "fiu" or not np.isfinite(value).all():
            raise ValueError(f"invalid shared-world trace: {key}")
        result[key] = value.copy()
    if np.any(np.diff(result["time"]) <= 0):
        raise ValueError("strictly increasing simulator time required")
    return result
