"""Bounded AR on-policy receiver actor/critic objective; no optimizer or physics."""

from typing import Any

import torch
from torch import nn

from rosclaw_soccer.rsi.anchored_receiving_motor import AnchoredReceivingMotor
from rosclaw_soccer.rsi.motor_sampling_law import history_motor_kl, history_motor_logp
from rosclaw_soccer.rsi.precontact_receiving_motor import PrecontactReceivingMotor


class ReceiverCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(112, 128), nn.Tanh(), nn.Linear(128, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def episode_balanced_value_loss(
    critic: ReceiverCritic, episodes: list[dict[str, Any]]
) -> torch.Tensor:
    """Give each complete sampled episode equal critic weight despite different lengths."""
    if type(critic) is not ReceiverCritic or not episodes:
        raise ValueError("explicit critic and nonempty complete episode set required")
    losses = []
    for episode in episodes:
        obs, returns = episode["observations"], episode["returns"]
        if (
            not isinstance(obs, torch.Tensor)
            or obs.ndim != 2
            or obs.shape[1] != 112
            or not 1 <= len(obs) <= 300
            or not isinstance(returns, torch.Tensor)
            or returns.shape != (len(obs),)
            or obs.dtype != torch.float32
            or returns.dtype != torch.float32
            or returns.device != obs.device
            or not torch.isfinite(obs).all()
            or not torch.isfinite(returns).all()
        ):
            raise ValueError("finite complete own observations and returns required")
        losses.append(0.5 * (critic(obs) - returns).square().mean())
    result = torch.stack(losses).mean()
    if not torch.isfinite(result):
        raise ValueError("finite episode-balanced critic objective required")
    return result


def receiver_own_history_loss(
    actor: AnchoredReceivingMotor,
    critic: ReceiverCritic,
    episode: dict[str, Any],
    log_std: torch.Tensor,
    *,
    rho: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Complete actual sampled sequence with detached critic and frozen anchor.

    Does not admit a dataset or perform an optimizer step. External caller must
    separately prove physical execution, own replay and measured outcomes.
    """
    if (
        type(actor) not in (AnchoredReceivingMotor, PrecontactReceivingMotor)
        or type(critic) is not ReceiverCritic
        or episode.get("sampling_law") != "box-truncated-ar1"
        or any(
            episode.get(k) is not True
            for k in (
                "all_actor_actions_actually_executed",
                "density_audited",
                "full_task_outcome_observed",
            )
        )
        or any(
            episode.get(k) is not False
            for k in ("assisted", "external_teacher", "authority_rejected")
        )
        or actor.baseline.training
        or any(p.requires_grad for p in actor.baseline.parameters())
    ):
        raise ValueError("audited pure own-history receiver data and frozen anchor required")
    obs = episode["observations"]
    if (
        not isinstance(obs, torch.Tensor)
        or obs.ndim != 2
        or obs.shape[1] != 112
        or not 1 <= len(obs) <= 300
    ):
        raise ValueError("bounded complete own 112D observation history required")
    n = len(obs)
    shapes = {
        "observations": (n, 112),
        "actions": (n, 29),
        "old_means": (n, 29),
        "reference": (n, 29),
        "old_logp": (n,),
        "returns": (n,),
        "advantages": (n,),
        "action_scale": (29,),
    }
    for name, shape in shapes.items():
        value = episode[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != shape
            or value.dtype != torch.float32
            or value.device != obs.device
            or not torch.isfinite(value).all()
        ):
            raise ValueError("finite complete own sequence required: " + name)
    if (
        not torch.equal(episode["action_scale"], actor.action_scale)
        or log_std.shape != (29,)
        or log_std.dtype != torch.float32
        or log_std.device != obs.device
        or log_std.requires_grad
    ):
        raise ValueError("exact frozen reference scale and declared noise required")
    with torch.no_grad():
        old = history_motor_logp(
            "box-truncated-ar1",
            episode["old_means"],
            episode["actions"],
            episode["reference"],
            episode["action_scale"],
            log_std,
            rho=rho,
        )
        if not torch.allclose(old, episode["old_logp"], rtol=1e-6, atol=1e-4):
            raise ValueError("full actual normalized sampling density required")
    means, evidence = actor.forward_with_evidence(obs, episode["reference"])
    values = critic(evidence["critic_features"].detach())
    value_loss = 0.5 * (values - episode["returns"]).square().mean()
    logp = history_motor_logp(
        "box-truncated-ar1",
        means,
        episode["actions"],
        episode["reference"],
        episode["action_scale"],
        log_std,
        rho=rho,
    )
    log_ratio = logp - episode["old_logp"]
    if not torch.isfinite(log_ratio).all() or log_ratio.abs().max() > 40:
        raise ValueError("finite trusted complete-history likelihood ratio required")
    ratio = log_ratio.exp()
    policy_loss = -torch.minimum(
        ratio * episode["advantages"], ratio.clamp(0.9, 1.1) * episode["advantages"]
    ).mean()
    anchor_loss = ((means - episode["reference"]) * actor.action_scale).square().mean()
    kl = history_motor_kl(
        "box-truncated-ar1",
        episode["old_means"],
        means,
        episode["actions"],
        episode["reference"],
        episode["action_scale"],
        log_std,
        rho=rho,
    )
    loss = policy_loss + value_loss + 0.02 * anchor_loss
    if not torch.isfinite(loss):
        raise ValueError("finite receiver actor/critic objective required")
    return loss, dict(
        actor_rows=n,
        critic_rows=n,
        conditional_kl=kl.detach(),
        policy_loss=policy_loss.detach(),
        value_loss=value_loss.detach(),
        anchor_loss=anchor_loss.detach(),
        optimizer_steps_executed=0,
        physical_retention_proven=False,
        hardware_authorized=False,
    )
