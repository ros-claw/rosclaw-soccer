"""Unclipped return fitting with actor weights AND optimizer slots frozen."""

import copy
from collections.abc import Callable
from typing import Any

import torch

from rosclaw_soccer.rsi.dual_onpolicy_checkpoint import same_training_state
from rosclaw_soccer.rsi.dual_recurrent_actor_critic import MotorCritic


def regress_motor_critic(
    critic: MotorCritic,
    optimizer: torch.optim.Optimizer,
    actor_parameters: list[torch.Tensor],
    features: torch.Tensor,
    returns: torch.Tensor,
    *,
    steps: int,
    on_update: Callable[[int], None] | None = None,
) -> list[dict[str, Any]]:
    if (
        type(steps) is not int
        or not 1 <= steps <= 64
        or features.ndim != 2
        or features.shape[1] != 288
        or returns.shape != (len(features),)
        or not len(features)
        or features.requires_grad
        or returns.requires_grad
        or features.dtype != torch.float32
        or returns.dtype != features.dtype
        or returns.device != features.device
        or not torch.isfinite(features).all()
        or not torch.isfinite(returns).all()
    ):
        raise ValueError("finite detached actual own-history features/returns required")
    owned = [id(p) for group in optimizer.param_groups for p in group["params"]]
    expected = [id(p) for p in actor_parameters] + [id(p) for p in critic.parameters()]
    if len(actor_parameters) != 16 or owned != expected or len(set(owned)) != 20:
        raise ValueError("all16 actor and4 critic optimizer parameters must have exact ownership")
    before = [p.detach().clone() for p in actor_parameters]
    slots = [copy.deepcopy(optimizer.state[p]) for p in actor_parameters]
    history = []
    for i in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = 0.5 * (critic(features) - returns).square().mean()
        if not torch.isfinite(loss):
            raise ValueError("nonfinite critic regression loss")
        loss.backward()
        if any(p.grad is not None for p in actor_parameters):
            raise ValueError("critic-only regression cannot create motor gradients")
        gradient = torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        if not torch.isfinite(gradient):
            raise ValueError("nonfinite critic regression gradient")
        optimizer.step()
        if on_update is not None:
            on_update(i + 1)
        with torch.no_grad():
            mse = (critic(features) - returns).square().mean()
        if not torch.isfinite(mse):
            raise ValueError("critic regression became nonfinite")
        history.append(
            dict(
                update=i + 1,
                pre_step_MSE=2 * float(loss.detach()),
                post_step_MSE=float(mse),
                gradient_norm=float(gradient),
            )
        )
    if any(
        not torch.equal(p.detach(), v) for p, v in zip(actor_parameters, before, strict=True)
    ) or any(
        not same_training_state(optimizer.state[p], s)
        for p, s in zip(actor_parameters, slots, strict=True)
    ):
        raise ValueError("critic fitting changed an actor weight or actor optimizer state")
    return history
