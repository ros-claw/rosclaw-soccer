"""Episode-bound football rewards; no teleport, restart, or success authority."""

from __future__ import annotations

from typing import Any

import numpy as np

from rosclaw_soccer.world.match_boundary import ball_exit_reason


def in_play_rewards(
    rewards: np.ndarray, trace: dict[str, Any], agent_ids: tuple[str, ...]
) -> np.ndarray:
    """Cut all post-exit credit and score the first measured boundary event.

    A goal gives each scoring teammate +0.5 and each opponent -0.5. An out
    charges each teammate of the last unambiguous observed touching actor
    -0.25. Simultaneous different actors clear that attribution. These are
    declared training rewards, not referee/scorer or promotion certificates.
    """
    count = len(trace["time"])
    geometry = np.asarray(trace.get("pitch_boundary_geometry", ()))
    ball = np.asarray(trace["ball_pose"])
    if (
        rewards.shape != (count, 8)
        or not np.isfinite(rewards).all()
        or count < 1
        or len(agent_ids) != 8
        or len(set(agent_ids)) != 8
        or any(a.split(".")[0] not in {"red", "blue"} for a in agent_ids)
        or geometry.shape != (count, 6)
        or not np.isfinite(geometry).all()
        or not np.array_equal(geometry, np.broadcast_to(geometry[0], geometry.shape))
        or ball.ndim != 2
        or ball.shape[0] != count
        or ball.shape[1] < 3
        or not np.isfinite(ball).all()
    ):
        raise ValueError("bound finite in-play reward geometry required")
    left, right, half_width, radius, width, height = geometry[0]
    if not (left < right and half_width == 3.0 and 0 < radius < min(width, height) / 2):
        raise ValueError("unsupported declared training pitch geometry")
    contacts = []
    for key in ("ball_contact", "ball_nonfoot_contact"):
        codes, force = np.asarray(trace[key + "_agent_code"]), np.asarray(trace[key + "_force_n"])
        if (
            codes.shape != (count,)
            or force.shape != (count,)
            or not np.isin(codes, np.arange(9)).all()
            or not np.isfinite(force).all()
            or np.any(force < 0)
        ):
            raise ValueError("finite aligned physical contact evidence required")
        contacts.append((codes, force))
    result = rewards.copy()
    previous_actor = None
    for frame in range(count):
        observed = {int(c[frame]) for c, f in contacts if c[frame] > 0 and f[frame] > 1e-6}
        if observed:
            previous_actor = next(iter(observed)) if len(observed) == 1 else None
        reason = ball_exit_reason(
            tuple(float(v) for v in ball[frame, :3]),
            left_x=left,
            right_x=right,
            radius=radius,
            goal_width=width,
            goal_height=height,
        )
        if reason is None:
            continue
        if frame == 0:
            raise ValueError("in-play episode must start inside the declared pitch")
        # This terminal objective does not claim potential-shaping invariance.
        terminal_cost = np.minimum(result[frame], 0.0)
        result[frame:] = 0.0
        result[frame] = terminal_cost
        if reason in {"BLUE_GOAL_CROSSING", "RED_GOAL_CROSSING"}:
            scoring_team = "blue" if reason == "BLUE_GOAL_CROSSING" else "red"
            result[frame] += [0.5 if a.split(".")[0] == scoring_team else -0.5 for a in agent_ids]
        elif previous_actor is not None:
            team = agent_ids[previous_actor - 1].split(".")[0]
            result[frame] += [-0.25 if a.split(".")[0] == team else 0.0 for a in agent_ids]
        break
    return result
