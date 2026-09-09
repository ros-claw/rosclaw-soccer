"""Goal-conditioned soccer residual; no action or promotion authority."""

from typing import Any

from rosclaw_soccer.training.ball_residual import build_ball_residual_actor_critic


def build_directed_ball_residual_actor_critic() -> Any:
    """133 motor features plus world-frame unit direction XY and speed / 2.5.

    Outputs remain 29 joint-target residual proposals, never raw torque.
    The optional Torch backend is imported only at explicit construction.
    """
    import torch

    class DirectedActorCritic(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            parent = build_ball_residual_actor_critic()
            self.actor, self.critic, self.logstd = parent.actor, parent.critic, parent.logstd
            for network in (self.actor, self.critic):
                network[0] = torch.nn.Linear(136, 128)

        def forward(self, observation: Any) -> tuple[Any, Any]:
            if (
                observation.ndim != 2
                or observation.shape[1] != 136
                or not 1 <= observation.shape[0] <= 65536
                or not bool(torch.isfinite(observation).all())
                or bool((observation.abs() > 10).any())
                or bool(
                    (torch.linalg.vector_norm(observation[:, -3:-1], dim=1) - 1)
                    .abs()
                    .gt(1e-4)
                    .any()
                )
                or bool(((observation[:, -1] < 0.2) | (observation[:, -1] > 1)).any())
            ):
                raise ValueError(
                    "finite motor state, unit direction and bounded pass speed required"
                )
            mean, value = self.actor(observation), self.critic(observation).squeeze(-1)
            if not bool(torch.isfinite(mean).all() and torch.isfinite(value).all()):
                raise FloatingPointError("nonfinite directed actor or critic output")
            return mean, value

    return DirectedActorCritic()


def initialize_directed_from_capture(directed: Any, capture: Any) -> None:
    """Copy frozen source numerics, zero new inputs; caller starts a NEW optimizer.

    This migration does not claim skill retention or transfer an old optimizer
    whose parameter shapes no longer match. The source model is never mutated.
    """
    import torch

    state = directed.state_dict()
    source = capture.state_dict()
    if set(state) != set(source):
        raise ValueError("directed/capture architectures do not match")
    migrated = {}
    for name, value in source.items():
        if not bool(torch.isfinite(value).all()):
            raise ValueError("nonfinite source weights")
        if name in ("actor.0.weight", "critic.0.weight"):
            if tuple(value.shape) != (128, 133) or tuple(state[name].shape) != (128, 136):
                raise ValueError("invalid directed input migration shape")
            expanded = torch.zeros_like(state[name])
            expanded[:, :133] = value.to(expanded)
            migrated[name] = expanded
        else:
            if value.shape != state[name].shape:
                raise ValueError("invalid directed parameter shape")
            migrated[name] = value.detach().to(state[name]).clone()
    directed.load_state_dict(migrated, strict=True)
