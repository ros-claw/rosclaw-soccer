"""Explicit half-turn frames for read-only navigation observations and deltas.

No robot model, team colour, field centre, policy or simulator is inferred.
Callers must bind the declared transform into their policy/evidence contract.
Projection does not authorize motion or establish cross-frame task competence.
"""

from __future__ import annotations

from dataclasses import replace

from rosclaw_soccer.skills.team.navigation_option import NavigationDelta, NavigationObservation
from rosclaw_soccer.training.receiving_frame import receiving_frame_translation


def canonical_navigation_observation(
    observation: NavigationObservation,
    *,
    half_turn: bool,
    translation_xy_m: tuple[float, float],
) -> NavigationObservation:
    """Return a validated copy, using XY'=translation-XY when enabled.

    World linear velocities change sign in XY; yaw rates and vertical values
    do not. Quaternion left multiplication rotates world heading without
    changing local body orientation. Agent names, neighbors' identities,
    roles, intent and physical timestamps remain unchanged.
    """
    if not isinstance(observation, NavigationObservation) or type(half_turn) is not bool:
        raise ValueError("typed navigation observation and explicit half-turn required")
    observation.__post_init__()
    tx, ty = receiving_frame_translation(translation_xy_m)
    if not half_turn:
        return replace(observation)

    def position3(value: tuple[float, float, float]) -> tuple[float, float, float]:
        return tx - value[0], ty - value[1], value[2]

    def velocity(value: tuple[float, float, float]) -> tuple[float, float, float]:
        return -value[0], -value[1], value[2]

    pose = observation.body_pose
    w, x, y, z = pose[3:]
    return replace(
        observation,
        body_pose=(tx - pose[0], ty - pose[1], pose[2], -z, -y, x, w),
        body_velocity=velocity(observation.body_velocity),
        ball_position=position3(observation.ball_position),
        ball_velocity=velocity(observation.ball_velocity),
        task_target=position3(observation.task_target),
        steering_target=(tx - observation.steering_target[0], ty - observation.steering_target[1]),
        baseline_command=velocity(observation.baseline_command),
        previous_command=velocity(observation.previous_command),
        neighbors=tuple((agent, tx - px, ty - py) for agent, px, py in observation.neighbors),
        effector_positions=tuple(
            (name, tx - px, ty - py, pz) for name, px, py, pz in observation.effector_positions
        ),
    )


def canonical_navigation_delta(delta: NavigationDelta, *, half_turn: bool) -> NavigationDelta:
    """Rotate a bounded velocity correction back; a half turn is its inverse."""
    if not isinstance(delta, NavigationDelta) or type(half_turn) is not bool:
        raise ValueError("typed bounded navigation delta and explicit half-turn required")
    delta.__post_init__()
    x, y, yaw = delta.velocity_delta
    return replace(delta, velocity_delta=(-x, -y, yaw) if half_turn else delta.velocity_delta)
