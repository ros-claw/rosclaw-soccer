"""Coached role baseline for native team locomotion, not a learned match policy."""

import math

import numpy as np

ROLES = ("goalkeeper", "defender", "playmaker", "finisher")


def role_target(team: str, role: str, ball_xy: np.ndarray) -> np.ndarray:
    ball = np.asarray(ball_xy, dtype=np.float64)
    if (
        team not in ("red", "blue")
        or role not in ROLES
        or ball.shape != (2,)
        or not np.isfinite(ball).all()
    ):
        raise ValueError("known symmetric team/role and measured finite ball required")
    direction = 1.0 if team == "red" else -1.0
    if role == "goalkeeper":
        return np.array([-direction * 8.0, np.clip(ball[1], -1.5, 1.5)])
    if role == "defender":
        return np.array(
            [np.clip(ball[0] - direction * 2.5, -6.0, 6.0), np.clip(ball[1] * 0.8, -3.0, 3.0)]
        )
    if role == "playmaker":
        return np.array(
            [
                np.clip(ball[0] - direction * 1.5, -6.0, 6.0),
                np.clip(ball[1] + direction * 1.8, -4.0, 4.0),
            ]
        )
    return np.asarray(ball - np.array([direction * 0.25, 0.0]), dtype=np.float64)


def command_toward(
    root_xy: np.ndarray, yaw: float, target_xy: np.ndarray, ball_xy: np.ndarray
) -> np.ndarray:
    root, target, ball = (np.asarray(x, dtype=np.float64) for x in (root_xy, target_xy, ball_xy))
    if any(
        x.shape != (2,) or not np.isfinite(x).all() for x in (root, target, ball)
    ) or not math.isfinite(yaw):
        raise ValueError("actual finite world positions and body yaw required")
    delta = target - root
    distance = float(np.linalg.norm(delta))
    facing = ball - root if distance < 0.15 else delta
    desired = (
        math.atan2(float(facing[1]), float(facing[0])) if np.linalg.norm(facing) > 1e-9 else yaw
    )
    error = math.atan2(math.sin(desired - yaw), math.cos(desired - yaw))
    speed = 0.0 if distance < 0.15 else min(0.8, distance)
    # Exact pretrained v0 limits: x[-.5,1],y[-.5,.5],yaw[-1,1]. No sprint claim.
    return np.array(
        [
            np.clip(speed * math.cos(error), -0.5, 1.0),
            np.clip(speed * math.sin(error), -0.5, 0.5),
            np.clip(2 * error, -1.0, 1.0),
        ],
        np.float32,
    )
