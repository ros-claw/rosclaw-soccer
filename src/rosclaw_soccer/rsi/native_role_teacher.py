"""Read-only counterfactual coached labels for visited native role states.

These are teacher queries, NOT proof of a physical action's outcome. Caller
must validate all original coached command labels before using off-policy data.
"""

import math
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.native_team_intent import (
    ROLES,
    clearance_command,
    command_toward,
    finisher_intercept_target,
    foot_tracking_command,
    role_target,
)


def coached_native_role_label(
    data: Any,
    kinematics: Any,
    layouts: dict[str, Any],
    *,
    team: str,
    role: str,
    previous_world_command: np.ndarray,
) -> np.ndarray:
    if team not in ("red", "blue") or role not in ROLES:
        raise ValueError("explicit native team and role required for teacher query")
    prefix = team + "_" + role + "/"
    lo = layouts[prefix]
    q, _ = lo.canonical(data)
    quat = q[3:7]
    yaw = math.atan2(
        2 * (quat[0] * quat[3] + quat[1] * quat[2]),
        1 - 2 * (quat[2] ** 2 + quat[3] ** 2),
    )
    ball = data.qpos[lo.bq : lo.bq + 2].copy()
    ball_velocity = data.qvel[lo.bv : lo.bv + 2].copy()
    goal = role_target(team, role, ball)
    if role == "finisher":
        goal, _, _ = finisher_intercept_target(team, q[:2], ball, ball_velocity)
    command = command_toward(
        q[:2], yaw, goal, ball, maximum_forward_speed=1.0 if role == "finisher" else 0.8
    )
    if role == "finisher" and np.linalg.norm(ball - q[:2]) < 1.0:
        feet = [lo.model.body(prefix + side + "_ankle_roll_link").id for side in ("left", "right")]
        command, _ = foot_tracking_command(
            q[:2],
            yaw,
            kinematics.xpos[feet, :2].copy(),
            ball,
            ball_velocity,
            previous_world_command,
        )
    neighbors = np.array(
        [
            data.qpos[other.rq : other.rq + 2] - q[:2]
            for other_prefix, other in layouts.items()
            if other_prefix != prefix
        ]
    )
    command, _, _ = clearance_command(
        command, yaw, neighbors, maximum_speed_mps=1.0 if role == "finisher" else 0.8
    )
    return command
