"""Head-only kick PPO on own raw actions; no collector, activation or hardware IO."""

from typing import Any, cast

import torch
from torch import nn

from rosclaw_soccer.rsi.correlated_motor_exploration import conditional_logp
from rosclaw_soccer.rsi.kick_prior_actor import KickPriorActor
from rosclaw_soccer.rsi.motor_sampling_law import history_motor_kl


def freeze_kick_features(actor: KickPriorActor) -> list[nn.Parameter]:
    for name, parameter in actor.named_parameters():
        parameter.requires_grad_(name.startswith("actor.6."))
    return [p for p in actor.parameters() if p.requires_grad]


class KickValueHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        final = nn.Linear(64, 1)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.net = nn.Sequential(nn.Linear(130, 64), nn.Tanh(), final)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != 130 or not torch.isfinite(features).all():
            raise ValueError("detached128 motor features plus two actual clocks required")
        if features.requires_grad:
            raise ValueError("value gradients must not modify the motor actor")
        return cast(torch.Tensor, self.net(features).squeeze(-1))


@torch.no_grad()
def value_features(
    actor: KickPriorActor, observation: torch.Tensor, frames: torch.Tensor
) -> torch.Tensor:
    if (
        observation.ndim != 2
        or observation.shape[1] != 547
        or (
            frames.shape != (len(observation),)
            or frames.dtype != torch.int64
            or not bool(((frames >= 0) & (frames < 360)).all())
            or not torch.isfinite(observation).all()
        )
    ):
        raise ValueError("actual own observations and bounded physical control frames required")
    latent = actor.actor[:6](observation)
    elapsed = frames.to(observation.dtype) / 359
    remaining_task = (299 - frames).clamp_min(0).to(observation.dtype) / 299
    return torch.cat((latent, elapsed[:, None], remaining_task[:, None]), dim=1).detach()


def own_ppo_objective(
    actor: KickPriorActor,
    episodes: list[dict[str, Any]],
    anchors: torch.Tensor,
    anchor_means: torch.Tensor,
    log_std: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if not episodes or any(
        p.requires_grad for n, p in actor.named_parameters() if not n.startswith("actor.6.")
    ):
        raise ValueError("own episodes and frozen original motor features required")
    losses, kls, ratios = [], [], []
    for episode in episodes:
        if (
            episode.get("actual_pd_commitment_verified") is not True
            or episode.get("external_teacher") is not False
        ):
            raise ValueError(
                "verified own raw-action trajectories, not assisted teacher data required"
            )
        obs = episode["observations"]
        n = len(obs)
        for key, shape in (
            ("old_means", (n, 29)),
            ("raw_actions", (n, 29)),
            ("old_logp", (n,)),
            ("advantages", (n,)),
        ):
            value = episode[key]
            if (
                not 1 <= n <= 360
                or value.shape != shape
                or value.dtype != torch.float32
                or (value.requires_grad or not torch.isfinite(value).all())
            ):
                raise ValueError("finite detached own raw-action PPO batch required: " + key)
        old = episode["old_means"]
        action = episode["raw_actions"]
        replay = conditional_logp(old, action, log_std, rho=0.9)
        if float((replay - episode["old_logp"]).abs().max()) > 0.002:
            raise ValueError("old likelihood does not match raw AR law")
        means = actor(obs)
        logp = conditional_logp(means, action, log_std, rho=0.9)
        ratio = (logp - episode["old_logp"]).exp()
        if not torch.isfinite(ratio).all():
            raise ValueError("nonfinite own-action likelihood ratio")
        advantage = episode["advantages"]
        losses.append(-torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage).mean())
        kls.append(
            history_motor_kl("gaussian-ar1", old, means, action, None, None, log_std, rho=0.9)
        )
        ratios.append(ratio.detach().max())
    current_anchor = actor(anchors)
    if (
        anchor_means.shape != current_anchor.shape
        or anchor_means.requires_grad
        or (not torch.isfinite(anchor_means).all())
    ):
        raise ValueError("original frozen prior means at actual causal anchors required")
    anchor_kl = 0.5 * ((current_anchor - anchor_means) / log_std.exp()).square().sum(-1).mean()
    own_kl = torch.stack(kls).mean()
    loss = torch.stack(losses).mean() + 0.1 * anchor_kl
    return loss, dict(
        mean_kl=torch.maximum(own_kl, anchor_kl),
        own_conditional_kl=own_kl,
        original_anchor_kl=anchor_kl,
        maximum_raw_likelihood_ratio=torch.stack(ratios).max(),
        raw_anchor_max_change=(current_anchor - anchor_means).abs().max(),
    )
