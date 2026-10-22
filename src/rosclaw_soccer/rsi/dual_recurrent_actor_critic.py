"""Own-dual-history AR(1) PPO objective, not a trained or admitted policy.

The critic cannot alter the actor through value gradients. Pure learner
histories only: a teacher-rescued episode is never on-policy motor evidence.
Recompute every mean from zero; no teacher/old hidden-state inputs.
"""

from typing import Any, cast

import torch
from torch import nn

from rosclaw_soccer.rsi.correlated_motor_exploration import (
    conditional_logp,
    conditional_parameters,
)
from rosclaw_soccer.rsi.dual_recurrent_motor import DualRecurrentMotor


class MotorCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        head = nn.Linear(128, 1)
        self.net = nn.Sequential(nn.Linear(288, 128), nn.Tanh(), head)
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.net(features).squeeze(-1))


@torch.no_grad()
def own_history_conditional_kl(
    actor: DualRecurrentMotor, episode: dict[str, Any], log_std: torch.Tensor, *, rho: float
) -> float:
    """Trial-step KL without exponentiating a large likelihood ratio.

    Large finite trials must backtrack, not crash the stricter PPO loss guard.
    Neither the likelihood guard nor the trust-region threshold is relaxed.
    """
    means, _ = actor(episode["observations"][:, None], actor.initial_state())
    location, scale = conditional_parameters(means[:, 0], episode["actions"], log_std, rho=rho)
    old_location, _ = conditional_parameters(
        episode["old_means"], episode["actions"], log_std, rho=rho
    )
    return float(0.5 * ((location - old_location) / scale).square().sum(-1).mean())


def own_history_loss(
    actor: DualRecurrentMotor,
    critic: MotorCritic,
    episode: dict[str, Any],
    log_std: torch.Tensor,
    *,
    rho: float,
    value_only: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if (
        type(actor) is not DualRecurrentMotor
        or episode.get("assisted") is not False
        or episode.get("all_actor_actions_actually_executed") is not True
        or episode.get("authority_rejected") is not False
        or episode.get("natural_terminal_observed") is not True
    ):
        raise ValueError("complete pure-learner actually executed option required")
    obs, actions = episode["observations"], episode["actions"]
    if obs.ndim != 2 or obs.shape[1] != 160 or not 1 <= len(obs) <= 400:
        raise ValueError("whole unpadded own observation history required")
    n = len(obs)
    tensors = {
        "actions": (n, 29),
        "old_means": (n, 29),
        "old_logp": (n,),
        "old_values": (n,),
        "returns": (n,),
        "advantages": (n,),
    }
    for name, shape in tensors.items():
        x = episode[name]
        if (
            not isinstance(x, torch.Tensor)
            or x.shape != shape
            or x.dtype != obs.dtype
            or x.device != obs.device
            or not torch.isfinite(x).all()
        ):
            raise ValueError("finite same-device actual episode field required: " + name)
    with torch.set_grad_enabled(not value_only):
        means, _, evidence = actor.forward_with_evidence(obs[:, None], actor.initial_state())
    means = means[:, 0]
    values = critic(evidence["critic_features"][:, 0].detach())
    returns, old_values = episode["returns"], episode["old_values"]
    clipped = old_values + (values - old_values).clamp(-0.2, 0.2)
    value_loss = (
        0.5 * torch.maximum((values - returns).square(), (clipped - returns).square()).mean()
    )
    if value_only:
        return value_loss, dict(actor_rows=0, critic_rows=n)
    # Check source likelihood against its own declared actual-action sequence.
    with torch.no_grad():
        source_logp = conditional_logp(episode["old_means"], actions, log_std, rho=rho)
        if not torch.allclose(source_logp, episode["old_logp"], atol=1e-4, rtol=1e-6):
            raise ValueError("old AR likelihood must bind actual means/actions, not IID fiction")
    logp = conditional_logp(means, actions, log_std, rho=rho)
    ratio_log = logp - episode["old_logp"]
    if not torch.isfinite(ratio_log).all() or ratio_log.abs().max() > 40:
        raise ValueError("invalid whole-history AR likelihood ratio")
    ratio = ratio_log.exp()
    advantage = episode["advantages"]
    policy = -torch.minimum(ratio * advantage, ratio.clamp(0.9, 1.1) * advantage).mean()
    new_loc, scale = conditional_parameters(means, actions, log_std, rho=rho)
    old_loc, _ = conditional_parameters(episode["old_means"], actions, log_std, rho=rho)
    conditional_kl = 0.5 * ((new_loc - old_loc) / scale).square().sum(-1).mean()
    anchor = evidence["reference_mean"][:, 0]
    anchor_kl = 0.5 * ((means - anchor) / 0.02).square().sum(-1).mean()
    loss = policy + value_loss + 0.02 * anchor_kl
    if not torch.isfinite(loss):
        raise ValueError("finite actor/critic full-history objective required")
    return loss, dict(
        actor_rows=n,
        critic_rows=n,
        conditional_kl=conditional_kl.detach(),
        value_loss=value_loss.detach(),
        anchor_kl=anchor_kl.detach(),
        policy_loss=policy.detach(),
    )
