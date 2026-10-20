"""Causal SIM-only64 motion-intention model; frozen muscle decoder is separate.

Uses the exact32-level FSQ constants inspected in the pinned local encoder ONNX,
not a guessed unit-sphere normalization. No reference, teacher ID or future state
enters this network. This factory alone is not a trained or qualified policy.
"""

from typing import Any


def create_motion_intention_student(*, hidden_size: int = 128) -> Any:
    if type(hidden_size) is not int or not 32 <= hidden_size <= 512:
        raise ValueError("bounded recurrent motion-intention architecture required")
    import torch
    from torch import nn

    class MotionIntention(nn.Module):
        observation_mean: torch.Tensor
        observation_std: torch.Tensor

        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("observation_mean", torch.zeros(108))
            self.register_buffer("observation_std", torch.ones(108))
            self.encoder = nn.Sequential(
                nn.Linear(108, hidden_size), nn.ELU(), nn.LayerNorm(hidden_size)
            )
            self.memory = nn.GRU(hidden_size, hidden_size, batch_first=True)
            self.actor = nn.Linear(hidden_size, 64)
            self.critic = nn.Linear(hidden_size, 1)
            nn.init.normal_(self.actor.weight, std=0.001)
            nn.init.zeros_(self.actor.bias)

        def initialize_training_mean(self, mean_token: Any) -> None:
            """Explicit train-only baseline initialization, never from validation."""
            if (
                mean_token.shape != (64,)
                or mean_token.dtype != torch.float32
                or not torch.isfinite(mean_token).all()
                or (mean_token < -1).any()
                or (mean_token > 0.9375).any()
            ):
                raise ValueError("finite64 training-only mean in original FSQ domain required")
            inverse = (
                torch.atanh((16.0 * mean_token + 0.5) / 15.515501022338867) - 0.03223699703812599
            )
            with torch.no_grad():
                self.actor.bias.copy_(inverse.to(self.actor.bias.device))

        def forward(self, observation: Any, hidden: Any = None) -> tuple[Any, Any, Any]:
            if (
                observation.ndim != 3
                or observation.shape[-1] != 108
                or observation.shape[0] < 1
                or not 1 <= observation.shape[1] <= 300
                or observation.dtype != torch.float32
                or not torch.isfinite(observation).all()
            ):
                raise ValueError("finite float32 batch,time,108 current observations required")
            if (
                not torch.isfinite(self.observation_mean).all()
                or not torch.isfinite(self.observation_std).all()
                or (self.observation_std < 0.01).any()
            ):
                raise ValueError("finite training-only observation normalizer required")
            if hidden is not None and (
                hidden.shape != (1, observation.shape[0], hidden_size)
                or hidden.dtype != torch.float32
                or not torch.isfinite(hidden).all()
            ):
                raise ValueError("finite owned recurrent motion memory required")
            x = ((observation - self.observation_mean) / self.observation_std).clamp(-10, 10)
            memory, next_hidden = self.memory(self.encoder(x), hidden)
            raw = self.actor(memory)
            bounded = 15.515501022338867 * torch.tanh(raw + 0.03223699703812599) - 0.5
            continuous = bounded / 16.0
            discrete = torch.round(bounded) / 16.0
            token = continuous + (discrete - continuous).detach()
            # Explicit value substitution preserves exact quantized inference
            # bits while retaining the straight-through continuous gradient.
            if not self.training:
                token = discrete
            value = self.critic(memory).squeeze(-1)
            if not (
                torch.isfinite(token).all()
                and torch.isfinite(value).all()
                and torch.isfinite(next_hidden).all()
            ):
                raise ValueError("nonfinite motion intent; no fallback token synthesized")
            return token, value, next_hidden

    return MotionIntention()
