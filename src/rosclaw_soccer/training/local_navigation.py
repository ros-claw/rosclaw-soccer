"""Local 39/3 navigation learning above frozen locomotion, never joint authority.

Features are world-frame, not rotation-invariant. Optional feature 40 is a
training-only influence bit excluded from actor and critic inputs. The runtime
still owns clearance, speed, acceleration, recovery and post-reception guards.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from rosclaw_soccer.skills.team.navigation_option import NavigationObservation

LOCAL_NAVIGATION_CONTRACT = "soccer.local_navigation_world_heading_39.v1"


def local_navigation_features(observation: NavigationObservation) -> np.ndarray:
    """Explicit heading, task, ball, previous command and three closest players.

    Neighbors are selected by measured planar distance, ties by agent ID, with
    missing neighbors padded by (0, 0). Relative positions use the world frame.
    Each field is clipped to [-10, 10]; no current proposed action is observed.
    """
    if not isinstance(observation, NavigationObservation):
        raise ValueError("typed read-only navigation observation required")
    observation.__post_init__()
    p = np.asarray(observation.body_pose[:3])
    w, x, y, z = observation.body_pose[3:]
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    neighbors = sorted(
        observation.neighbors, key=lambda n: ((n[1] - p[0]) ** 2 + (n[2] - p[1]) ** 2, n[0])
    )[:3]
    relative = [(n[1] - p[0], n[2] - p[1]) for n in neighbors]
    relative.extend([(0.0, 0.0)] * (3 - len(relative)))
    intent = (
        0
        if observation.intent in ("pass", "distribute")
        else 1
        if observation.intent == "shoot"
        else 2
        if observation.intent in ("receive", "intercept")
        else 3
    )
    roles = ("playmaker", "finisher", "defender", "goalkeeper")
    if observation.role not in roles:
        raise ValueError("explicit football role required for local navigation features")
    features = np.concatenate(
        (
            (math.sin(yaw), math.cos(yaw), p[2]),
            observation.body_velocity,
            (2 * (x * z - w * y), 2 * (y * z + w * x)),
            (np.asarray(observation.ball_position) - p) / 3,
            np.asarray(observation.ball_velocity) / 3,
            (np.asarray(observation.task_target) - p) / 5,
            np.asarray(observation.steering_target) - p[:2],
            observation.baseline_command,
            observation.previous_command,
            np.asarray(relative).reshape(6) / 2,
            np.eye(4)[intent],
            np.eye(4)[roles.index(observation.role)],
        )
    )
    if features.shape != (39,) or not np.isfinite(features).all():
        raise ValueError("finite 39-feature local navigation contract required")
    return np.clip(features, -10, 10).astype(np.float32)


def local_navigation_delta(raw: np.ndarray) -> tuple[float, float, float]:
    """Bound Gaussian latents; likelihoods must be computed before this mapping."""
    value = np.asarray(raw)
    if value.shape != (3,) or value.dtype.kind != "f" or not np.isfinite(value).all():
        raise ValueError("finite three-axis navigation latent required")
    delta = np.tanh(value.astype(np.float64)) * np.asarray((0.25, 0.25, 0.4))
    norm = float(np.linalg.norm(delta[:2]))
    if norm > 0.25:
        delta[:2] *= 0.25 / norm
    return float(delta[0]), float(delta[1]), float(delta[2])


def build_local_navigation_actor_critic(*, training_influence: bool = False) -> Any:
    """Optional Torch, zero initial mean. The influence bit is never a feature."""
    import torch

    if type(training_influence) is not bool:
        raise ValueError("explicit training influence contract required")

    class ActorCritic(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()

            def network(output: int) -> Any:
                return torch.nn.Sequential(
                    torch.nn.Linear(39, 64),
                    torch.nn.Tanh(),
                    torch.nn.Linear(64, 64),
                    torch.nn.Tanh(),
                    torch.nn.Linear(64, output),
                )

            self.actor, self.critic = network(3), network(1)
            torch.nn.init.zeros_(self.actor[-1].weight)
            torch.nn.init.zeros_(self.actor[-1].bias)
            self.logstd = torch.nn.Parameter(torch.full((3,), -1.0))

        def forward(self, observation: Any) -> tuple[Any, Any]:
            if (
                not isinstance(observation, torch.Tensor)
                or observation.ndim != 2
                or observation.shape[1] != 39 + int(training_influence)
                or not 1 <= len(observation) <= 65536
                or observation.dtype != torch.float32
                or observation.layout != torch.strided
                or not bool(torch.isfinite(observation).all())
                or bool((observation.abs() > 10).any())
                or training_influence
                and not bool(((observation[:, -1] == 0) | (observation[:, -1] == 1)).all())
            ):
                raise ValueError("finite explicit local navigation features required")
            mean = self.actor(observation[:, :39])
            value = self.critic(observation[:, :39]).squeeze(-1)
            if not bool(torch.isfinite(mean).all() and torch.isfinite(value).all()):
                raise FloatingPointError("nonfinite local navigation actor or critic output")
            return mean, value

    return ActorCritic()
