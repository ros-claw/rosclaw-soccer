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


def foot_tracking_command(
    root_xy: np.ndarray,
    yaw: float,
    foot_positions_xy: np.ndarray,
    ball_xy: np.ndarray,
    ball_velocity_xy: np.ndarray,
    previous_world_command: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Near-ball coached velocity proposal using actual feet, not root standoff.

    Reuses the bounded planar interception primitive. No simulator writes,
    learned skill or possession assertion. Clock owned by caller:20ms.
    """
    import torch

    from rosclaw_soccer.skills.planar_interception import (
        PlanarInterceptionConfig,
        propose_planar_interception,
    )

    root, feet, ball, velocity, previous = (
        np.asarray(x, dtype=np.float64)
        for x in (root_xy, foot_positions_xy, ball_xy, ball_velocity_xy, previous_world_command)
    )
    if (
        feet.shape != (2, 2)
        or any(x.shape != (2,) for x in (root, ball, velocity, previous))
        or any(not np.isfinite(x).all() for x in (root, feet, ball, velocity, previous))
        or not math.isfinite(yaw)
        or np.linalg.norm(previous) > 1.0 + 1e-6
    ):
        raise ValueError("finite actual feet, ball and bounded previous world command required")
    foot = feet[int(np.argmin(np.linalg.norm(feet - ball, axis=1)))]
    proposed = propose_planar_interception(
        target_position_xy=torch.tensor(ball, dtype=torch.float32),
        target_velocity_xy=torch.tensor(velocity, dtype=torch.float32),
        effector_position_xy=torch.tensor(foot, dtype=torch.float32),
        previous_velocity_xy=torch.tensor(previous, dtype=torch.float32),
        config=PlanarInterceptionConfig((0.0, 0.0), 1.5, 1.0, 0.08),
    ).numpy()
    c, s = math.cos(yaw), math.sin(yaw)
    vx = float(np.clip(c * proposed[0] + s * proposed[1], -0.5, 1.0))
    vy = float(np.clip(-s * proposed[0] + c * proposed[1], -0.5, 0.5))
    desired = math.atan2(float(proposed[1]), float(proposed[0]))
    error = math.atan2(math.sin(desired - yaw), math.cos(desired - yaw))
    command = np.array([vx, vy, np.clip(2 * error, -1.0, 1.0)], dtype=np.float32)
    # Store the actual domain-clipped command, not an unapplied proposal.
    actual_world = np.array([c * command[0] - s * command[1], s * command[0] + c * command[1]])
    return command, actual_world
