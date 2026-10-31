"""Local 39/3 navigation learning above frozen locomotion, never joint authority.

Features are world-frame, not rotation-invariant. Optional feature 40 is a
training-only influence bit excluded from actor and critic inputs. The runtime
still owns clearance, speed, acceleration, recovery and post-reception guards.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from rosclaw_soccer.skills.team.navigation_option import NavigationObservation

LOCAL_NAVIGATION_CONTRACT = "soccer.local_navigation_world_heading_39.v1"
TEAM_NAVIGATION_CONTRACT = "soccer.local_navigation_world_heading_team_45.v1"


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


def team_navigation_features(observation: NavigationObservation) -> np.ndarray:
    """Append measured neighbor presence/ally bits, without changing legacy fields.

    Each of the same three nearest neighbors contributes (present, ally).
    Absent entries are (0, 0), opponents (1, 0), teammates (1, 1). This is
    simulation research context, not permission to act or a success label.
    """
    base = local_navigation_features(observation)

    def team(agent_id: str) -> str:
        prefix, separator, role = agent_id.partition(".")
        if not separator or not role or prefix not in ("red", "blue"):
            raise ValueError("explicit red/blue team-qualified player required")
        return prefix

    own_team = team(observation.agent_id)
    # Validate all IDs, even if a farther player is outside the selected three.
    for neighbor in observation.neighbors:
        team(neighbor[0])
    x, y = observation.body_pose[:2]
    nearest = sorted(
        observation.neighbors, key=lambda n: ((n[1] - x) ** 2 + (n[2] - y) ** 2, n[0])
    )[:3]
    flags = [(1.0, float(team(n[0]) == own_team)) for n in nearest]
    flags.extend([(0.0, 0.0)] * (3 - len(flags)))
    return np.concatenate((base, np.asarray(flags, dtype=np.float32).reshape(6)))


def lift_local_navigation_state(state: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Zero-extend a legacy 39-input actor/critic; caller authenticates weights."""
    shapes = {
        "logstd": (3,),
        "actor.0.weight": (64, 39),
        "actor.0.bias": (64,),
        "actor.2.weight": (64, 64),
        "actor.2.bias": (64,),
        "actor.4.weight": (3, 64),
        "actor.4.bias": (3,),
        "critic.0.weight": (64, 39),
        "critic.0.bias": (64,),
        "critic.2.weight": (64, 64),
        "critic.2.bias": (64,),
        "critic.4.weight": (1, 64),
        "critic.4.bias": (1,),
    }
    if not isinstance(state, Mapping) or set(state) != set(shapes):
        raise ValueError("complete legacy navigation actor/critic state required")
    result = {}
    for key, shape in shapes.items():
        value = np.asarray(state[key])
        if value.dtype != np.float32 or value.shape != shape or not np.isfinite(value).all():
            raise ValueError("finite float32 legacy parameters with exact shapes required")
        result[key] = np.pad(value, ((0, 0), (0, 6))) if key.endswith(".0.weight") else value.copy()
    return result


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


def build_local_navigation_actor_critic(
    *, training_influence: bool = False, team_context: bool = False
) -> Any:
    """Optional Torch, zero initial mean. The influence bit is never a feature."""
    import torch

    if type(training_influence) is not bool or type(team_context) is not bool:
        raise ValueError("explicit training influence and team context contracts required")
    width = 45 if team_context else 39

    class ActorCritic(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()

            def network(output: int) -> Any:
                return torch.nn.Sequential(
                    torch.nn.Linear(width, 64),
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
                or observation.shape[1] != width + int(training_influence)
                or not 1 <= len(observation) <= 65536
                or observation.dtype != torch.float32
                or observation.layout != torch.strided
                or not bool(torch.isfinite(observation).all())
                or bool((observation.abs() > 10).any())
                or training_influence
                and not bool(((observation[:, -1] == 0) | (observation[:, -1] == 1)).all())
            ):
                raise ValueError("finite explicit local navigation features required")
            if team_context:
                flags = observation[:, 39:45].reshape(-1, 3, 2)
                present, ally = flags[:, :, 0], flags[:, :, 1]
                if (
                    not bool(((flags == 0) | (flags == 1)).all())
                    or bool((ally > present).any())
                    or bool((present[:, 1:] > present[:, :-1]).any())
                ):
                    raise ValueError("valid ordered teammate/opponent presence bits required")
            mean = self.actor(observation[:, :width])
            value = self.critic(observation[:, :width]).squeeze(-1)
            if not bool(torch.isfinite(mean).all() and torch.isfinite(value).all()):
                raise FloatingPointError("nonfinite local navigation actor or critic output")
            return mean, value

    return ActorCritic()
