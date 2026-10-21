"""Optional SIM PPO with actual deterministic far-domain actions.

Far/missing-ball rows train the critic, never a Gaussian motion proposal or an
actor likelihood. Near-ball exploration is scaled by the measured smooth gate.
This is a new learning intent, not the old contact24 loss or deployment.
"""

import math
from typing import Any

import torch


def approach_sample(policy: Any, obs: torch.Tensor) -> dict[str, torch.Tensor]:
    mean = policy.actor(obs)
    gate = policy.actor.measured_gate(obs)
    plastic = gate > 0
    action = mean.clone()
    logp = torch.zeros(len(obs), dtype=obs.dtype, device=obs.device)
    if plastic.any():
        std = policy.log_std.clamp(math.log(0.01), math.log(0.1)).exp()
        dist: Any = torch.distributions.Normal(
            mean[plastic], std * gate[plastic, None].clamp_min(1e-4)
        )
        action[plastic] = dist.sample()
        logp[plastic] = dist.log_prob(action[plastic]).sum(-1)
    return dict(action=action, old_logp=logp, plastic=plastic, old_value=policy.value(obs))


def approach_ppo_loss(
    policy: Any, data: dict[str, torch.Tensor], *, value_only: bool
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    obs, old_value, returns = data["obs"], data["old_value"], data["returns"]
    value = policy.value(obs)
    zero = torch.zeros((), dtype=obs.dtype, device=obs.device)
    if value_only:
        loss = 0.5 * (value - returns).square().mean()
        if not torch.isfinite(loss):
            raise ValueError("nonfinite long-course critic loss")
        return loss, dict(value=loss.detach(), approximate_kl=zero, plastic_rows=zero)
    plastic = policy.actor.measured_gate(obs) > 0
    if not torch.equal(plastic, data["plastic"]):
        raise ValueError("stored exploration domain differs from actual measured observations")
    clipped = old_value + (value - old_value).clamp(-0.2, 0.2)
    critic = 0.5 * torch.maximum((value - returns).square(), (clipped - returns).square()).mean()
    actor_loss = anchor_kl = approximate_kl = entropy = zero
    if plastic.any():
        near = obs[plastic]
        mean = policy.actor(near)
        gate = policy.actor.measured_gate(near)
        std = policy.log_std.clamp(math.log(0.01), math.log(0.1)).exp()
        dist: Any = torch.distributions.Normal(mean, std * gate[:, None].clamp_min(1e-4))
        logratio = dist.log_prob(data["action"][plastic]).sum(-1) - data["old_logp"][plastic]
        if not torch.isfinite(logratio).all() or logratio.abs().max() > 40:
            raise ValueError("invalid actual near-domain on-policy likelihood")
        ratio, advantage = logratio.exp(), data["advantage"][plastic]
        actor_loss = -torch.minimum(ratio * advantage, ratio.clamp(0.9, 1.1) * advantage).mean()
        with torch.no_grad():
            anchor = policy.baseline(near)
        anchor_kl = ((mean - anchor).square() / (2 * 0.1**2)).sum(-1).mean()
        approximate_kl = ((ratio - 1) - logratio).mean()
        entropy = dist.entropy().sum(-1).mean()
    loss = actor_loss + critic + 0.01 * anchor_kl - 0.0002 * entropy
    if not torch.isfinite(loss):
        raise ValueError("nonfinite long-approach PPO objective")
    return loss, dict(
        actor=actor_loss.detach(),
        value=critic.detach(),
        baseline_anchor_kl=anchor_kl.detach(),
        approximate_kl=approximate_kl.detach(),
        plastic_rows=plastic.sum().detach(),
    )
