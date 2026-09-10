"""Supervised retention of successful Soccer motor actions, separate from PPO.

This updater neither certifies the demonstrations nor promotes its output.
The caller owns evidence verification, Core plasticity leases and physical exams.
Torch is optional until the updater is called.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SuccessDistillationConfig:
    steps: int = 1000
    learning_rate: float = 0.001

    def __post_init__(self) -> None:
        if (
            type(self.steps) is not int
            or not 1 <= self.steps <= 10000
            or isinstance(self.learning_rate, bool)
            or not math.isfinite(self.learning_rate)
            or not 0 < self.learning_rate <= 0.001
        ):
            raise ValueError("bounded supervised update configuration required")


def distill_successful_motor_actions(
    agent: Any,
    *,
    observations: Any,
    raw_actions: Any,
    sample_weights: Any,
    config: SuccessDistillationConfig | None = None,
) -> dict[str, Any]:
    """Fit the existing 133/29 actor only, with a fresh, locally owned Adam.

    Inputs are detached float32 training tensors, NOT fresh PPO likelihoods.
    All inputs and actor ownership are checked before updates. On optimizer
    failure the caller must discard/restore the candidate, just as with PPO.
    Only actual successful optimizer.step calls enter the returned step count.
    Critic/noise weights and pre-existing requires_grad flags are untouched.
    """
    import torch

    active = config or SuccessDistillationConfig()
    values = (observations, raw_actions, sample_weights)
    if any(not isinstance(value, torch.Tensor) for value in values):
        raise ValueError("detached training tensors required")
    if observations.ndim != 2 or not 2 <= len(observations) <= 65536:
        raise ValueError("bounded sample/feature axes required")
    size = len(observations)
    shapes = ((size, 133), (size, 29), (size,))
    for value, shape in zip(values, shapes, strict=True):
        if (
            value.shape != shape
            or value.dtype != torch.float32
            or value.layout != torch.strided
            or value.requires_grad
            or value.device != observations.device
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError("finite matching detached float32 training tensors required")
    if (
        bool((observations.abs() > 10).any())
        or bool((raw_actions.abs() > 20).any())
        or bool((sample_weights < 0).any())
        or abs(float(sample_weights.sum()) - 1.0) > 1e-5
        or int((sample_weights > 0).sum()) < 2
    ):
        raise ValueError("bounded actions/observations and normalized sample weights required")
    actor = agent.actor
    actor_parameters = tuple(actor.parameters())
    owned = {id(parameter) for parameter in actor_parameters}
    if not owned or any(
        not parameter.requires_grad
        or parameter.dtype != torch.float32
        or parameter.device != observations.device
        or not bool(torch.isfinite(parameter).all())
        for parameter in actor_parameters
    ):
        raise ValueError("finite trainable float32 actor required")
    if any(
        id(parameter) in owned
        for name, parameter in agent.named_parameters(remove_duplicate=False)
        if not name.startswith("actor.")
    ):
        raise ValueError("actor must not share parameters with protected components")
    protected = {
        name: tensor.detach().clone()
        for name, tensor in agent.state_dict().items()
        if not name.startswith("actor.")
    }
    with torch.no_grad():
        prediction = actor(observations)
        if prediction.shape != raw_actions.shape or not bool(torch.isfinite(prediction).all()):
            raise ValueError("finite 29-action actor output required")
        initial_loss = float(((prediction - raw_actions).square().mean(1) * sample_weights).sum())
    optimizer = torch.optim.Adam(actor_parameters, lr=active.learning_rate)
    executed_steps = 0
    for _ in range(active.steps):
        prediction = actor(observations)
        loss = ((prediction - raw_actions).square().mean(1) * sample_weights).sum()
        if not bool(torch.isfinite(loss)):
            raise ValueError("nonfinite supervised loss; discard candidate")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor_parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        executed_steps += 1
    with torch.no_grad():
        final_loss = float(
            ((actor(observations) - raw_actions).square().mean(1) * sample_weights).sum()
        )
    if not math.isfinite(final_loss) or any(
        not torch.equal(agent.state_dict()[name], tensor) for name, tensor in protected.items()
    ):
        raise ValueError("nonfinite candidate or protected state changed; discard candidate")
    return {
        "optimizer_steps": executed_steps,
        "samples": size,
        "initial_weighted_mse": initial_loss,
        "final_weighted_mse": final_loss,
        "protected_state_unchanged": True,
        "method": "SUPERVISED_SUCCESS_DISTILLATION_NOT_PPO",
        "activation_ceiling": "SIM_ONLY",
        "promotion_eligible": False,
    }
