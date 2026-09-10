"""Learn anticipatory reception navigation while retaining a frozen motor actor.

The 136/3 policy sees 133 motor features using the PREVIOUS foundation target,
then three normalized previous navigation residuals. After selecting navigation,
the caller advances its frozen foundation once and evaluates the existing motor
actor with the CURRENT foundation. This avoids action-conditioned observations
leaking into the navigation likelihood and does not reset either motor history.
No simulator, controller, lease, activation, or hardware authority lives here.
"""

from typing import Any

from rosclaw_soccer.training.ball_residual import build_ball_residual_actor_critic
from rosclaw_soccer.training.coupled_ball_residual import (
    NAVIGATION_RESIDUAL_LIMITS,
    advance_coupled_residual,
)


def build_reception_navigation_actor_critic() -> Any:
    """New explicit contract, not a compatible replacement for a motor actor."""
    import torch

    class NavigationActorCritic(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            parent = build_ball_residual_actor_critic()
            self.actor, self.critic = parent.actor, parent.critic
            self.actor[0] = torch.nn.Linear(136, 128)
            self.critic[0] = torch.nn.Linear(136, 128)
            self.actor[-1] = torch.nn.Linear(128, 3)
            torch.nn.init.zeros_(self.actor[-1].weight)
            torch.nn.init.zeros_(self.actor[-1].bias)
            self.logstd = torch.nn.Parameter(torch.full((3,), -2.0))

        def forward(self, observation: Any) -> tuple[Any, Any]:
            if (
                not isinstance(observation, torch.Tensor)
                or observation.ndim != 2
                or observation.shape[1] != 136
                or not 1 <= len(observation) <= 65536
                or observation.dtype != torch.float32
                or observation.layout != torch.strided
                or not bool(torch.isfinite(observation).all())
                or bool((observation.abs() > 10).any())
                or bool((observation[:, -3:].abs() > 1.000001).any())
            ):
                raise ValueError("finite 136-feature navigation state and history required")
            mean, value = self.actor(observation), self.critic(observation).squeeze(-1)
            if not bool(torch.isfinite(mean).all() and torch.isfinite(value).all()):
                raise FloatingPointError("nonfinite navigation actor or critic output")
            return mean, value

    return NavigationActorCritic()


def advance_reception_navigation(raw_action: Any, previous_navigation: Any) -> Any:
    """Reuse the exact established navigation envelope; never emit joint actions.

    The caller must still project the composed command through current teammate,
    opponent and pitch clearance. Smoothing is not an acceleration guarantee for
    a separately changing nominal command.
    """
    import torch

    if (
        not isinstance(raw_action, torch.Tensor)
        or raw_action.ndim != 2
        or raw_action.shape[1] != 3
        or not 1 <= len(raw_action) <= 4096
        or raw_action.dtype != torch.float32
        or raw_action.requires_grad
        or raw_action.layout != torch.strided
        or not isinstance(previous_navigation, torch.Tensor)
        or previous_navigation.requires_grad
        or previous_navigation.layout != torch.strided
    ):
        raise ValueError("detached three-action navigation and history required")
    zeros = raw_action.new_zeros((len(raw_action), 29))
    _, navigation = advance_coupled_residual(
        torch.cat((zeros, raw_action), dim=1), zeros, previous_navigation
    )
    return navigation


def normalized_reception_navigation(previous_navigation: Any) -> Any:
    """Validate a history without modifying it, then normalize its three axes."""
    import torch

    if not isinstance(previous_navigation, torch.Tensor):
        raise ValueError("navigation tensor history required")
    advance_reception_navigation(torch.zeros_like(previous_navigation), previous_navigation)
    return previous_navigation / previous_navigation.new_tensor(NAVIGATION_RESIDUAL_LIMITS)
