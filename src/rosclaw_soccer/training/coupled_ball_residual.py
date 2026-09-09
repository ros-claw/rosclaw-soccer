"""Jointly learned navigation and joint residuals for isolated soccer training.

This is a NEW 139-feature/32-action contract, not a compatible replacement for
the 133/136-feature motor actors. The frozen recurrent foundation advances only
after navigation selection, once per control tick. No simulator or activation
authority is owned here; shared-world integration needs a separate adapter.
"""

from typing import Any

from rosclaw_soccer.training.ball_residual import (
    advance_ball_residual,
    build_ball_residual_actor_critic,
)

COUPLED_OBSERVATION_SIZE = 139
COUPLED_ACTION_SIZE = 32
NAVIGATION_RESIDUAL_LIMITS = (0.7, 0.7, 0.8)


def build_coupled_ball_residual_actor_critic() -> Any:
    """133 motor features (PREVIOUS foundation), 3 prior nav residuals, 3 goal.

    Prior navigation residuals are divided by NAVIGATION_RESIDUAL_LIMITS. Goal is
    a declared fixed world-frame unit launch direction XY and desired speed/2.5.
    Outputs are 29 joint residual latents and 3 navigation residual latents. Neither
    policy outputs nor training rewards authorize real motion or policy promotion.
    """
    import torch

    class CoupledActorCritic(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            parent = build_ball_residual_actor_critic()
            self.actor, self.critic = parent.actor, parent.critic
            self.actor[0] = torch.nn.Linear(139, 128)
            self.critic[0] = torch.nn.Linear(139, 128)
            self.actor[-1] = torch.nn.Linear(128, 32)
            torch.nn.init.zeros_(self.actor[-1].weight)
            torch.nn.init.zeros_(self.actor[-1].bias)
            self.logstd = torch.nn.Parameter(torch.full((32,), -1.0))

        def forward(self, observation: Any) -> tuple[Any, Any]:
            if (
                observation.ndim != 2
                or observation.shape[1] != 139
                or not 1 <= observation.shape[0] <= 65536
                or not bool(torch.isfinite(observation).all())
                or bool((observation.abs() > 10).any())
                or bool((observation[:, 133:136].abs() > 1.000001).any())
                or bool(
                    (torch.linalg.vector_norm(observation[:, -3:-1], dim=1) - 1)
                    .abs()
                    .gt(1e-4)
                    .any()
                )
                or bool(((observation[:, -1] < 0.2) | (observation[:, -1] > 1)).any())
            ):
                raise ValueError("finite coupled motor state and explicit launch-ray goal required")
            mean, value = self.actor(observation), self.critic(observation).squeeze(-1)
            if not bool(torch.isfinite(mean).all() and torch.isfinite(value).all()):
                raise FloatingPointError("nonfinite coupled actor or critic output")
            return mean, value

    return CoupledActorCritic()


def advance_coupled_residual(
    raw_action: Any, previous_joint: Any, previous_navigation: Any
) -> tuple[Any, Any]:
    """Preserve old joint envelope; smooth bounded world-navigation residuals."""
    import torch

    if (
        not isinstance(raw_action, torch.Tensor)
        or raw_action.ndim != 2
        or raw_action.shape[1] != 32
        or not 1 <= len(raw_action) <= 4096
        or raw_action.dtype != torch.float32
        or not bool(torch.isfinite(raw_action).all())
        or not isinstance(previous_joint, torch.Tensor)
        or previous_joint.shape != (len(raw_action), 29)
        or previous_joint.device != raw_action.device
        or previous_joint.dtype != raw_action.dtype
        or not bool(torch.isfinite(previous_joint).all())
        or not isinstance(previous_navigation, torch.Tensor)
        or previous_navigation.shape != (len(raw_action), 3)
        or previous_navigation.device != raw_action.device
        or previous_navigation.dtype != raw_action.dtype
        or not bool(torch.isfinite(previous_navigation).all())
    ):
        raise ValueError("finite aligned coupled action and navigation history required")
    limits = raw_action.new_tensor(NAVIGATION_RESIDUAL_LIMITS)
    if bool((previous_navigation.abs() > limits + 1e-7).any()):
        raise ValueError("previous navigation residual exceeds its envelope")
    joint = advance_ball_residual(raw_action[:, :29], previous_joint)
    desired = limits * torch.tanh(raw_action[:, 29:])
    change = 0.2 * (desired - previous_navigation)
    xy = change[:, :2]
    scale = (0.012 / torch.linalg.vector_norm(xy, dim=1).clamp_min(1e-9)).clamp_max(1)
    navigation = previous_navigation + torch.cat(
        (xy * scale[:, None], change[:, 2:3].clamp(-0.04, 0.04)), dim=1
    )
    return joint, navigation


def compose_coupled_navigation(nominal: Any, residual: Any) -> Any:
    """Cap combined world XY speed at 0.7 m/s and yaw rate at 0.8 rad/s.

    This does not replace shared-world player/pitch clearance. Residual smoothing
    also does not promise an acceleration bound for a changing nominal command.
    """
    import torch

    if (
        any(
            not isinstance(v, torch.Tensor)
            or v.ndim != 2
            or v.shape[1] != 3
            or not 1 <= len(v) <= 4096
            or v.dtype != torch.float32
            or not bool(torch.isfinite(v).all())
            for v in (nominal, residual)
        )
        or nominal.shape != residual.shape
        or nominal.device != residual.device
        or bool((torch.linalg.vector_norm(nominal[:, :2], dim=1) > 0.700001).any())
        or bool((nominal[:, 2].abs() > 0.800001).any())
        or bool((residual.abs() > residual.new_tensor(NAVIGATION_RESIDUAL_LIMITS) + 1e-7).any())
    ):
        raise ValueError("bounded aligned nominal and learned navigation required")
    command = nominal + residual
    scale = (0.7 / torch.linalg.vector_norm(command[:, :2], dim=1).clamp_min(1e-9)).clamp_max(1)
    return torch.cat((command[:, :2] * scale[:, None], command[:, 2:3].clamp(-0.8, 0.8)), dim=1)
