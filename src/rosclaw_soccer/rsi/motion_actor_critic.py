"""SIM-only recurrent Gaussian pre-FSQ actor-critic, frozen perception/muscle prior."""

import math
from typing import Any


def create_motion_actor_critic() -> Any:
    import torch
    from torch import nn

    from rosclaw_soccer.rsi.neural_motion_intention import create_motion_intention_student

    class MotionActorCritic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base = create_motion_intention_student()
            # First online pilot: preserve the learned perceptual/memory trunk.
            for module in (self.base.encoder, self.base.memory):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
            self.log_std = nn.Parameter(torch.full((64,), math.log(0.02)))

        def parameters_for_state(
            self, observation: Any, hidden: Any = None
        ) -> tuple[Any, Any, Any, Any]:
            if (
                observation.ndim != 3
                or observation.shape[-1] != 108
                or observation.shape[0] < 1
                or not 1 <= observation.shape[1] <= 300
                or observation.dtype != torch.float32
                or not torch.isfinite(observation).all()
            ):
                raise ValueError("finite current float32 recurrent observations required")
            if (
                not torch.isfinite(self.base.observation_mean).all()
                or not torch.isfinite(self.base.observation_std).all()
                or (self.base.observation_std < 0.01).any()
                or not torch.isfinite(self.log_std).all()
            ):
                raise ValueError("finite normalizer and exploration parameters required")
            if hidden is not None and (
                hidden.shape != (1, observation.shape[0], 128)
                or hidden.dtype != torch.float32
                or not torch.isfinite(hidden).all()
            ):
                raise ValueError("finite owned recurrent memory required")
            x = ((observation - self.base.observation_mean) / self.base.observation_std).clamp(
                -10, 10
            )
            memory, next_hidden = self.base.memory(self.base.encoder(x), hidden)
            mean = self.base.actor(memory)
            std = self.log_std.clamp(math.log(0.005), math.log(0.05)).exp().expand_as(mean)
            value = self.base.critic(memory).squeeze(-1)
            if not (
                torch.isfinite(mean).all()
                and torch.isfinite(value).all()
                and torch.isfinite(next_hidden).all()
            ):
                raise ValueError("nonfinite actor-critic output, no fallback")
            return mean, std, value, next_hidden

        def sample(
            self, observation: Any, hidden: Any, *, generator: Any, stochastic: bool
        ) -> dict[str, Any]:
            if type(stochastic) is not bool:
                raise ValueError("explicit sampling mode required")
            mean, std, value, next_hidden = self.parameters_for_state(observation, hidden)
            noise = (
                torch.randn(mean.shape, dtype=mean.dtype, device=mean.device, generator=generator)
                if stochastic
                else torch.zeros_like(mean)
            )
            raw = mean + std * noise
            token = (
                torch.round(15.515501022338867 * torch.tanh(raw + 0.03223699703812599) - 0.5) / 16.0
            )
            distribution: Any = torch.distributions.Normal(mean, std)
            return dict(
                mean=mean,
                std=std,
                noise=noise,
                raw=raw,
                token=token,
                value=value,
                hidden=next_hidden,
                log_prob=distribution.log_prob(raw).sum(-1),
                entropy=distribution.entropy().sum(-1),
            )

        def evaluate_raw(self, observation: Any, raw: Any) -> tuple[Any, Any, Any, Any, Any]:
            mean, std, value, _ = self.parameters_for_state(observation)
            if (
                raw.shape != mean.shape
                or raw.dtype != torch.float32
                or not torch.isfinite(raw).all()
            ):
                raise ValueError("finite recorded PRE-quantization action required")
            distribution: Any = torch.distributions.Normal(mean, std)
            return (
                distribution.log_prob(raw).sum(-1),
                distribution.entropy().sum(-1),
                value,
                mean,
                std,
            )

    return MotionActorCritic()
