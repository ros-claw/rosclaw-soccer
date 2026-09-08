"""Reuse the same compliant net law under a half-turn, without pose writes."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from rosclaw_soccer.world.field import (
    G1CompliantGoalNetState,
    G1TrainingGoalSpec,
    apply_g1_compliant_goal_net_force,
)


def apply_opposite_goal_net_force(
    data: Any,
    *,
    ball_body_id: int,
    ball_qpos: int,
    ball_qvel: int,
    spec: G1TrainingGoalSpec,
    left_goal_plane_x_m: float,
    state: G1CompliantGoalNetState,
) -> None:
    # Only a three-coordinate observation is transformed. Never change qpos/qvel.
    position = np.array(data.qpos[ball_qpos : ball_qpos + 3], copy=True)
    velocity = np.array(data.qvel[ball_qvel : ball_qvel + 3], copy=True)
    position[0] = spec.plane_x_m + left_goal_plane_x_m - position[0]
    position[1] *= -1.0
    velocity[:2] *= -1.0
    proxy = SimpleNamespace(qpos=position, qvel=velocity, xfrc_applied=np.zeros((1, 6)))
    apply_g1_compliant_goal_net_force(
        proxy,
        ball_body_id=0,
        ball_qpos=0,
        ball_qvel=0,
        spec=spec,
        capture_depth_m=max(0.20, 0.80 * spec.depth_m),
        stiffness_n_m=180.0,
        damping_n_s_m=10.0,
        state=state,
    )
    force = proxy.xfrc_applied[0, :3]
    force[:2] *= -1.0
    data.xfrc_applied[ball_body_id, :3] += force
