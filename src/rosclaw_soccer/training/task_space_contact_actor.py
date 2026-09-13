"""Three-axis contact exploration over a frozen, explicit G1 motion parent.

The learner proposes latent foot forces, not motor torques. The caller owns
current geometry, selected-foot provenance, guarded mapping, physics, learning
leases and independent qualification. This module owns neural math only.
"""

from collections.abc import Mapping
from typing import Any


def build_task_space_contact_actor_critic(
    parent_state: Mapping[str, Any],
    mean: Any,
    scale: Any,
    *,
    critic_mean: Any,
    critic_scale: Any,
) -> Any:
    """Explicit 170/3: old 169 contact features plus a latched binary foot ID.

    A separate training-only learning bit, if used, makes the wrapper 171/3.
    Initial force means are zero and values equal the frozen contact parent.
    Both learning heads and logstd train; all parent parameters and scaling
    buffers remain frozen. No exam data or fitting occurs in this constructor.
    """
    import torch

    from rosclaw_soccer.training.contact_feedback_actor import (
        build_normalized_contact_actor_critic,
    )
    from rosclaw_soccer.training.frozen_feature_normalization import normalize_network_inputs

    parent = build_normalized_contact_actor_critic(
        parent_state, mean, scale, critic_mean=critic_mean, critic_scale=critic_scale
    )
    parent.requires_grad_(False)

    class TaskSpaceContactActorCritic(torch.nn.Module):
        actor: Any
        critic: Any

        def __init__(self) -> None:
            super().__init__()
            self.parent = parent
            for name, width, centre, spread in (
                ("actor", 3, mean, scale),
                ("critic", 1, critic_mean, critic_scale),
            ):
                network = torch.nn.Sequential(
                    torch.nn.Linear(170, 128),
                    torch.nn.Tanh(),
                    torch.nn.Linear(128, 128),
                    torch.nn.Tanh(),
                    torch.nn.Linear(128, width),
                )
                output: Any = network[-1]
                torch.nn.init.zeros_(output.weight)
                torch.nn.init.zeros_(output.bias)
                setattr(
                    self,
                    name,
                    normalize_network_inputs(
                        network,
                        torch.cat((centre, centre.new_tensor([0.5]))),
                        torch.cat((spread, spread.new_tensor([0.5]))),
                    ),
                )
            self.logstd = torch.nn.Parameter(torch.full((3,), -2.5))

        def forward(self, observation: Any) -> tuple[Any, Any]:
            if (
                not isinstance(observation, torch.Tensor)
                or observation.ndim != 2
                or observation.shape[1] != 170
                or observation.dtype != torch.float32
                or not 1 <= len(observation) <= 65536
                or not bool(((observation[:, 169] == 0) | (observation[:, 169] == 1)).all())
            ):
                raise ValueError("explicit contact features and binary selected foot required")
            with torch.no_grad():
                _, value = self.parent(observation[:, :169].contiguous())
            mean = 2 * torch.tanh(self.actor(observation))
            value = value + self.critic(observation).squeeze(-1)
            if not bool(torch.isfinite(mean).all() and torch.isfinite(value).all()):
                raise FloatingPointError("nonfinite task-space contact learner")
            return mean, value

    return TaskSpaceContactActorCritic()
