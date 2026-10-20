"""Isolated measured-geometry branch over a frozen motion parent, SIM research.

Old perception, recurrent memory, actor, critic and normalizers remain frozen.
New zero-initialized bounded raw64 head does not blend incompatible FSQ tokens.
No world step, motor, promotion, hardware or future observation is available.
"""

import math


def create_kinematic_motion_residual(parent, extractor):
    import numpy as np
    import torch
    from torch import nn

    if parent.training or any(p.requires_grad for p in parent.parameters()):
        raise ValueError("explicitly frozen eval-only actor-critic parent required")

    class KinematicMotionResidual(nn.Module):
        def __init__(self):
            super().__init__()
            self.parent = parent
            self.extractor = extractor
            self.register_buffer("geometry_mean", torch.zeros(26))
            self.register_buffer("geometry_std", torch.ones(26))
            self.adapter = nn.Sequential(nn.Linear(262, 128), nn.ELU(), nn.Linear(128, 64))
            self.value_adapter = nn.Sequential(nn.Linear(262, 64), nn.ELU(), nn.Linear(64, 1))
            for head in (self.adapter[-1], self.value_adapter[-1]):
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
            self.log_std = nn.Parameter(parent.log_std.detach().clone())
            self.sample_calls = 0

        def network_call_counts(self):
            if self.sample_calls != 300:
                raise ValueError("complete current H300 inference stream required")
            return dict(policy_invocations=300, intention_network_calls=600)

        def train(self, mode=True):
            super().train(mode)
            self.parent.eval()
            return self

        def geometry(self, observation):
            if observation.ndim != 3 or observation.shape[-1] != 108:
                raise ValueError("batch,time,current108 required")
            raw = observation.detach().cpu().numpy()
            result = np.stack(
                [self.extractor.from_student_features(x) for x in raw.reshape(-1, 108)]
            )
            return torch.from_numpy(result.reshape(*raw.shape[:2], 26)).to(observation.device)

        def parameters_for_state(self, observation, hidden=None, *, geometry=None):
            if (
                observation.ndim != 3
                or observation.shape[-1] != 108
                or observation.dtype != torch.float32
                or not torch.isfinite(observation).all()
                or not 1 <= observation.shape[1] <= 300
                or observation.shape[0] < 1
            ):
                raise ValueError("finite current recurrent float32 observations required")
            base = self.parent.base
            if (
                not torch.isfinite(base.observation_mean).all()
                or not torch.isfinite(base.observation_std).all()
                or (base.observation_std < 0.01).any()
                or not torch.isfinite(self.geometry_mean).all()
                or not torch.isfinite(self.geometry_std).all()
                or (self.geometry_std < 0.01).any()
                or not torch.isfinite(self.log_std).all()
            ):
                raise ValueError("finite well-conditioned frozen/current normalizers required")
            if hidden is not None and (
                hidden.shape != (1, observation.shape[0], 128)
                or hidden.dtype != torch.float32
                or not torch.isfinite(hidden).all()
            ):
                raise ValueError("owned current128 recurrent hidden required")
            x = ((observation - base.observation_mean) / base.observation_std).clamp(-10, 10)
            with torch.no_grad():
                memory, next_hidden = base.memory(base.encoder(x), hidden)
                original = base.actor(memory)
                original_value = base.critic(memory).squeeze(-1)
            geo = self.geometry(observation) if geometry is None else geometry
            if (
                geo.shape != (*observation.shape[:2], 26)
                or geo.dtype != torch.float32
                or not torch.isfinite(geo).all()
            ):
                raise ValueError("same-current-state finite geometry cache required")
            g = ((geo - self.geometry_mean) / self.geometry_std).clamp(-10, 10)
            context = torch.cat((x, memory.detach(), g), dim=-1)
            mean = original + 0.125 * torch.tanh(self.adapter(context))
            value = original_value + self.value_adapter(context).squeeze(-1)
            std = self.log_std.clamp(math.log(0.005), math.log(0.05)).exp().expand_as(mean)
            if not all(torch.isfinite(t).all() for t in (mean, std, value, next_hidden)):
                raise ValueError("nonfinite isolated geometry branch; no fallback")
            return mean, std, value, next_hidden

        def sample(self, observation, hidden, *, generator, stochastic):
            if type(stochastic) is not bool:
                raise ValueError("explicit deterministic/stochastic mode required")
            if hidden is None:
                self.sample_calls = 0
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
            distribution = torch.distributions.Normal(mean, std)
            self.sample_calls += 1
            return dict(
                mean=mean,
                std=std,
                value=value,
                hidden=next_hidden,
                noise=noise,
                raw=raw,
                token=token,
                log_prob=distribution.log_prob(raw).sum(-1),
                entropy=distribution.entropy().sum(-1),
            )

        def evaluate_raw(self, observation, raw):
            mean, std, value, _ = self.parameters_for_state(observation)
            if (
                raw.shape != mean.shape
                or raw.dtype != torch.float32
                or not torch.isfinite(raw).all()
            ):
                raise ValueError("complete finite prequantization actions required")
            distribution = torch.distributions.Normal(mean, std)
            return (
                distribution.log_prob(raw).sum(-1),
                distribution.entropy().sum(-1),
                value,
                mean,
                std,
            )

    return KinematicMotionResidual()
