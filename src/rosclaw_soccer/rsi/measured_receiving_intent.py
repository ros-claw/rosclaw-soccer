"""Explicit geometry teacher for a SIM receiver, not a learned team policy.

Only measured ball/body states and declared role anchors enter this intent.
The learned velocity backend owns all29 joint targets; this function cannot
write physics or command a joint. No pre-scripted ball flight or contact time.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReceivingIntent:
    role_state: str
    target_xy: tuple[float, float]
    command_body: tuple[float, float, float]
    launch_seen: bool


def receiving_intent(
    *,
    root_xy: np.ndarray,
    yaw: float,
    ball_xy: np.ndarray,
    ball_velocity_xy: np.ndarray,
    home_xy: np.ndarray,
    pass_direction: np.ndarray,
    launch_seen: bool,
    ready_anchor_xy: np.ndarray | None = None,
) -> ReceivingIntent:
    for value in (root_xy, ball_xy, ball_velocity_xy, home_xy, pass_direction):
        if value.shape != (2,) or not np.isfinite(value).all():
            raise ValueError("finite measured planar receiving state required")
    if (
        type(launch_seen) is not bool
        or not np.isfinite(yaw)
        or abs(np.linalg.norm(pass_direction) - 1) > 1e-10
    ):
        raise ValueError("explicit role history, finite yaw and unit pass direction required")
    if ready_anchor_xy is not None and (
        ready_anchor_xy.shape != (2,) or not np.isfinite(ready_anchor_xy).all()
    ):
        raise ValueError("finite communicated receiving anchor required")
    launch = launch_seen or float(ball_velocity_xy @ pass_direction) > 0.4
    if launch:
        speed2 = float(ball_velocity_xy @ ball_velocity_xy)
        eta = float(np.clip((root_xy - ball_xy) @ ball_velocity_xy / max(speed2, 0.01), 0, 0.5))
        lateral = np.array([-pass_direction[1], pass_direction[0]])
        target = ball_xy + eta * ball_velocity_xy + 0.35 * pass_direction + 0.12 * lateral
        phase = "INTERCEPT" if float((root_xy - ball_xy) @ pass_direction) > 0 else "RECOVER_LOOSE"
    else:
        target = home_xy.copy() if ready_anchor_xy is None else ready_anchor_xy.copy()
        phase = (
            "MOVE_SUPPORT"
            if ready_anchor_xy is not None and np.linalg.norm(target - root_xy) > 0.12
            else "WAIT_PASS"
        )
    velocity = 1.4 * (target - root_xy)
    norm = float(np.linalg.norm(velocity))
    if norm > 1:
        velocity /= norm
    c, s = np.cos(yaw), np.sin(yaw)
    delta = ball_xy - root_xy
    wanted = float(np.arctan2(delta[1], delta[0])) if np.linalg.norm(delta) > 0.15 else yaw
    angle = float(np.arctan2(np.sin(wanted - yaw), np.cos(wanted - yaw)))
    command = (
        float(np.clip(c * velocity[0] + s * velocity[1], -0.5, 1)),
        float(np.clip(-s * velocity[0] + c * velocity[1], -0.5, 0.5)),
        float(np.clip(1.5 * angle, -1, 1)),
    )
    return ReceivingIntent(phase, (float(target[0]), float(target[1])), command, bool(launch))
