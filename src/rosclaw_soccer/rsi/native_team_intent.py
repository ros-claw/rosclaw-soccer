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
    root_xy: np.ndarray,
    yaw: float,
    target_xy: np.ndarray,
    ball_xy: np.ndarray,
    *,
    maximum_forward_speed: float = 0.8,
) -> np.ndarray:
    root, target, ball = (np.asarray(x, dtype=np.float64) for x in (root_xy, target_xy, ball_xy))
    if (
        any(x.shape != (2,) or not np.isfinite(x).all() for x in (root, target, ball))
        or not math.isfinite(yaw)
        or not math.isfinite(maximum_forward_speed)
        or not 0.8 <= maximum_forward_speed <= 1.0
    ):
        raise ValueError("actual finite world positions and body yaw required")
    delta = target - root
    distance = float(np.linalg.norm(delta))
    facing = ball - root if distance < 0.15 else delta
    desired = (
        math.atan2(float(facing[1]), float(facing[0])) if np.linalg.norm(facing) > 1e-9 else yaw
    )
    error = math.atan2(math.sin(desired - yaw), math.cos(desired - yaw))
    speed = 0.0 if distance < 0.15 else min(maximum_forward_speed, distance)
    # Exact pretrained v0 limits: x[-.5,1],y[-.5,.5],yaw[-1,1]. No sprint claim.
    return np.array(
        [
            np.clip(speed * math.cos(error), -0.5, 1.0),
            np.clip(speed * math.sin(error), -0.5, 0.5),
            np.clip(2 * error, -1.0, 1.0),
        ],
        np.float32,
    )


def finisher_intercept_target(
    team: str, root_xy: np.ndarray, ball_xy: np.ndarray, ball_velocity_xy: np.ndarray
) -> tuple[np.ndarray, float, bool]:
    from rosclaw_soccer.world.ground_intercept import propose_ground_intercept

    if team not in ("red", "blue"):
        raise ValueError("known team required")
    root, ball, velocity = (
        np.asarray(v, dtype=np.float64) for v in (root_xy, ball_xy, ball_velocity_xy)
    )
    if any(v.shape != (2,) or not np.isfinite(v).all() for v in (root, ball, velocity)):
        raise ValueError("actual finite planar motion required")
    if np.linalg.norm(velocity) < 0.95:
        proposal = propose_ground_intercept(
            root_xy=(float(root[0]), float(root[1])),
            ball_xy=(float(ball[0]), float(ball[1])),
            ball_velocity_xy=(float(velocity[0]), float(velocity[1])),
            root_speed_mps=0.95,
            contact_standoff_m=0.25,
            maximum_horizon_sec=2.0,
        )
        predicted = np.asarray(proposal.predicted_ball_xy)
        horizon, reachable = proposal.horizon_sec, proposal.kinematically_reachable
    else:
        # Measured fast balls are not clipped into the slow-ball model's domain.
        predicted = ball + 0.2 * velocity
        horizon, reachable = 0.2, False
    offset = 0.25 if team == "red" else -0.25
    return np.asarray(predicted - np.array([offset, 0.0]), dtype=np.float64), horizon, reachable
