"""Simulation-only, tick-wise contact feedback over a frozen coupled policy.

Explicit 169/32 contract: 139 legacy features, then current and previous
15-feature foot/ball measurements. No hidden state, simulator, action mapping,
training lease or policy promotion is owned here. The caller supplies causal
measurements and retains the existing bounded residual controller.
"""

from collections.abc import Mapping
from typing import Any

CONTACT_FEATURE_SIZE = 15
CONTACT_OBSERVATION_SIZE = 169


def contact_features(
    *,
    foot_position: Any,
    foot_velocity: Any,
    ball_position: Any,
    ball_velocity: Any,
    launch_direction: Any,
    stance_force: Any,
    interval_valid: Any,
) -> Any:
    """World-frame measurements to launch-frame relative features, left then right.

    Relative positions are metres, velocities divided by 5 m/s, stance normal
    forces divided by 300 N. Features are clipped, not safety certificates.
    Finite-difference velocities and prior-interval forces are unavailable at
    reset: mark interval_valid=0, which zeros these channels explicitly.
    """
    import torch

    if not isinstance(ball_position, torch.Tensor) or ball_position.ndim != 2:
        raise ValueError("batched measured ball position required")
    n = len(ball_position)
    values = (
        (foot_position, (n, 2, 3)),
        (foot_velocity, (n, 2, 3)),
        (ball_position, (n, 3)),
        (ball_velocity, (n, 3)),
        (launch_direction, (n, 2)),
        (stance_force, (n, 2)),
        (interval_valid, (n, 1)),
    )
    if not 1 <= n <= 65536 or any(
        not isinstance(value, torch.Tensor)
        or value.shape != shape
        or value.dtype != torch.float32
        or value.device != ball_position.device
        or not bool(torch.isfinite(value).all())
        or bool((value.abs() > 1e6).any())
        for value, shape in values
    ):
        raise ValueError("finite aligned float32 contact measurements required")
    if (
        bool((stance_force < 0).any())
        or not bool(((interval_valid == 0) | (interval_valid == 1)).all())
        or bool((torch.linalg.vector_norm(launch_direction, dim=1) - 1).abs().gt(1e-4).any())
    ):
        raise ValueError("nonnegative forces, binary validity and unit launch ray required")

    def rotate(vector: Any) -> Any:
        x, y = launch_direction[:, 0:1], launch_direction[:, 1:2]
        return torch.stack(
            (
                vector[..., 0] * x + vector[..., 1] * y,
                -vector[..., 0] * y + vector[..., 1] * x,
                vector[..., 2],
            ),
            dim=-1,
        )

    position = rotate(ball_position[:, None] - foot_position).clamp(-5, 5).reshape(n, 6)
    velocity = rotate((ball_velocity[:, None] - foot_velocity) / 5).clamp(-5, 5)
    velocity = velocity.reshape(n, 6) * interval_valid
    support = (stance_force / 300).clamp(0, 1) * interval_valid
    result = torch.cat((position, velocity, support, interval_valid), dim=1)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("contact feature arithmetic overflow")
    return result


def build_contact_feedback_actor_critic(parent_state: Mapping[str, Any]) -> Any:
    """Copy/freeze the 139/32 parent; train bounded actor correction and critic.

    Zero-initialized final layers preserve parent means/values exactly initially.
    A fresh trainable logstd starts at -3.5. Rollout and PPO must explicitly use
    the same exploration floor; legacy checkpoint contracts are not inferred.
    """
    import torch

    from rosclaw_soccer.training.coupled_ball_residual import (
        build_coupled_ball_residual_actor_critic,
    )

    if (
        not isinstance(parent_state, Mapping)
        or not parent_state
        or any(
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.float32
            or not bool(torch.isfinite(value).all())
            for value in parent_state.values()
        )
    ):
        raise ValueError("explicit finite float32 parent tensors required")

    class ContactActorCritic(torch.nn.Module):  # type: ignore[misc]  # Optional lazy Torch import.
        def __init__(self) -> None:
            super().__init__()
            self.parent = build_coupled_ball_residual_actor_critic()
            self.parent.load_state_dict(parent_state, strict=True)
            if any(not bool(torch.isfinite(v).all()) for v in self.parent.state_dict().values()):
                raise ValueError("finite parent policy required")
            self.parent.requires_grad_(False)
            self.actor = torch.nn.Sequential(
                torch.nn.Linear(169, 128),
                torch.nn.Tanh(),
                torch.nn.Linear(128, 128),
                torch.nn.Tanh(),
                torch.nn.Linear(128, 32),
            )
            self.critic = torch.nn.Sequential(
                torch.nn.Linear(169, 128),
                torch.nn.Tanh(),
                torch.nn.Linear(128, 128),
                torch.nn.Tanh(),
                torch.nn.Linear(128, 1),
            )
            for layer in (self.actor[-1], self.critic[-1]):
                torch.nn.init.zeros_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
            self.logstd = torch.nn.Parameter(torch.full((32,), -3.5))

        def forward(self, observation: Any) -> tuple[Any, Any]:
            if (
                not isinstance(observation, torch.Tensor)
                or observation.ndim != 2
                or observation.shape[1] != 169
                or not 1 <= len(observation) <= 65536
                or observation.dtype != torch.float32
                or not bool(torch.isfinite(observation).all())
                or bool((observation[:, 139:].abs() > 5).any())
            ):
                raise ValueError("explicit finite 169-feature contact feedback contract required")
            for start in (139, 154):
                support = observation[:, start + 12 : start + 14]
                valid = observation[:, start + 14]
                if bool(((support < 0) | (support > 1)).any()) or not bool(
                    ((valid == 0) | (valid == 1)).all()
                ):
                    raise ValueError("normalized stance and binary history validity required")
                if bool((observation[valid == 0, start + 6 : start + 14] != 0).any()):
                    raise ValueError("unavailable interval velocities and forces must be zero")
            with torch.no_grad():
                mean, value = self.parent(observation[:, :139].contiguous())
            mean = mean + 0.3 * torch.tanh(self.actor(observation))
            value = value + self.critic(observation).squeeze(-1)
            if not bool(torch.isfinite(mean).all() and torch.isfinite(value).all()):
                raise FloatingPointError("nonfinite contact actor or critic output")
            return mean, value

    return ContactActorCritic()
