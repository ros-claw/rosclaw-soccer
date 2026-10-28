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
    mode: str = "pursuit-v1",
    capture_target_xy: np.ndarray | None = None,
    lateral_offset_m: float = 0.12,
    longitudinal_offset_m: float = 0.35,
) -> ReceivingIntent:
    if (
        type(mode) is not str
        or mode
        not in ("pursuit-v1", "anchor-plane-v2", "projected-capture-v3", "velocity-match-v4")
        or (mode != "pursuit-v1" and ready_anchor_xy is None)
        or (mode == "projected-capture-v3" and capture_target_xy is None)
        or type(lateral_offset_m) is not float
        or not np.isfinite(lateral_offset_m)
        or abs(lateral_offset_m) > 0.4
        or type(longitudinal_offset_m) is not float
        or not np.isfinite(longitudinal_offset_m)
        or not 0.0 <= longitudinal_offset_m <= 0.8
    ):
        raise ValueError("explicit receiving mode and communicated anchor for plane mode required")
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
    if capture_target_xy is not None and (
        capture_target_xy.shape != (2,) or not np.isfinite(capture_target_xy).all()
    ):
        raise ValueError("finite communicated capture target required")
    launch = launch_seen or float(ball_velocity_xy @ pass_direction) > 0.4
    if launch:
        speed2 = float(ball_velocity_xy @ ball_velocity_xy)
        eta = float(np.clip((root_xy - ball_xy) @ ball_velocity_xy / max(speed2, 0.01), 0, 0.5))
        lateral = np.array([-pass_direction[1], pass_direction[0]])
        target = (
            ball_xy
            + eta * ball_velocity_xy
            + longitudinal_offset_m * pass_direction
            + lateral_offset_m * lateral
        )
        phase = "INTERCEPT" if float((root_xy - ball_xy) @ pass_direction) > 0 else "RECOVER_LOOSE"
        plane_supported = mode == "anchor-plane-v2"
        if mode == "projected-capture-v3":
            assert capture_target_xy is not None
            toward = float(ball_velocity_xy @ pass_direction)
            if toward > 0.4:
                # A causal constant-velocity prediction, NOT the future scored
                # arrival. Precision-plane tracking is appropriate only while
                # the currently observed trajectory intersects that plane
                # within the task's existing .1m lateral precision domain.
                # Otherwise pursue/recover the real loose ball, including after
                # it crosses the target plane. No trial ID or clock threshold.
                arrival = float((capture_target_xy - ball_xy) @ pass_direction) / toward
                miss = float((ball_xy + arrival * ball_velocity_xy - capture_target_xy) @ lateral)
                plane_supported = arrival >= 0 and abs(miss) <= 0.1
        if plane_supported:
            # Keep the longitudinal ROLE destination instead of chasing a slow
            # pass upstream. Lateral prediction, actual ball-facing, gains and
            # all velocity limits remain unchanged. This is a geometric
            # teacher ablation, NOT a learned receiver or proof of a catch.
            assert ready_anchor_xy is not None
            target = target + float((ready_anchor_xy - target) @ pass_direction) * pass_direction
            phase = "TRACK_RECEIVING_PLANE"
    else:
        target = home_xy.copy() if ready_anchor_xy is None else ready_anchor_xy.copy()
        phase = (
            "MOVE_SUPPORT"
            if ready_anchor_xy is not None and np.linalg.norm(target - root_xy) > 0.12
            else "WAIT_PASS"
        )
    velocity = 1.4 * (target - root_xy)
    if mode == "velocity-match-v4" and launch:
        along = float((root_xy - ball_xy) @ pass_direction)
        incoming = float(ball_velocity_xy @ pass_direction)
        if 0.0 < along <= 1.8 and incoming > 0.4:
            # A bounded causal interception teacher, not a contact model:
            # use only current measured position/velocity, preserve the chosen
            # foot-side offset and ask the locomotion actor to match the ball.
            eta = float(np.clip(along / incoming, 0.0, 0.5))
            lateral = np.array([-pass_direction[1], pass_direction[0]])
            target = (
                ball_xy
                + eta * ball_velocity_xy
                + longitudinal_offset_m * pass_direction
                + lateral_offset_m * lateral
            )
            velocity = ball_velocity_xy + 1.4 * (target - root_xy)
            phase = "VELOCITY_MATCH_INTERCEPT"
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
