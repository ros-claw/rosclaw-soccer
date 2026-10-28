"""Optional SIM contact24 PPO math with explicit timeout bootstrap and masks.

The established177 actor is transferred, the177 critic starts fresh. No legacy
304 critic/Adam transfer. This file grants no deployment or promotion authority.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray


def enable_ball_only_actor_gradient(actor: Any) -> Any:
    """Expose only current ball position/velocity columns to the actor optimizer.

    This is a SIM-only experimental parameterization, not a trained policy or
    proof the policy actually uses perception. The caller must separately
    assign zero learning rate to all other actor parameters.
    """
    import torch
    from torch import nn

    layer = actor.mlp[0]
    if not isinstance(layer, nn.Linear) or layer.weight.shape != (512, 177):
        raise ValueError("explicit velocity98/reference70/ball9 actor ABI required")
    mask = torch.zeros_like(layer.weight)
    mask[:, 168:174] = 1
    return layer.weight.register_hook(lambda gradient: gradient * mask)


def guarded_ppo_optimizer_step(
    policy: Any, optimizer: Any, obs: Any, action: Any, old_logp: Any, *, kl_limit: float = 0.02
) -> tuple[bool, float]:
    """Roll back both network and optimizer if post-update rollout KL is high.

    The caller has already computed/clipped gradients. KL is measured on all
    valid samples from the actual rollout, not just the update minibatch.
    This is an empirical on-policy guard, not a safety or generalization proof.
    """
    import copy
    import math

    import torch

    if not math.isfinite(kl_limit) or kl_limit <= 0:
        raise ValueError("positive finite KL limit required")
    before_policy = copy.deepcopy(policy.state_dict())
    before_optimizer = copy.deepcopy(optimizer.state_dict())
    try:
        optimizer.step()
        with torch.no_grad():
            policy.log_std.clamp_(math.log(0.01), math.log(0.1))
            log_ratio = policy.distribution(obs).log_prob(action).sum(-1) - old_logp
            if not torch.isfinite(log_ratio).all() or log_ratio.abs().max() > 40:
                raise ValueError("nonfinite or extreme post-update likelihood ratio")
            kl = float(((log_ratio.exp() - 1) - log_ratio).mean())
            if not math.isfinite(kl):
                raise ValueError("finite post-update KL required")
            if any(not torch.isfinite(v).all() for v in policy.state_dict().values()):
                raise ValueError("finite post-update model required")
        if kl > kl_limit:
            policy.load_state_dict(before_policy, strict=True)
            optimizer.load_state_dict(before_optimizer)
            return False, kl
        return True, kl
    except Exception:
        policy.load_state_dict(before_policy, strict=True)
        optimizer.load_state_dict(before_optimizer)
        raise


def course_gae(
    reward: NDArray[np.float32],
    value: NDArray[np.float32],
    next_value: NDArray[np.float32],
    terminated: NDArray[np.bool_],
    truncated: NDArray[np.bool_],
    valid: NDArray[np.bool_],
    *,
    maximum_controls: int = 24,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    shape = reward.shape
    if (
        type(maximum_controls) is not int
        or maximum_controls not in (24, 40, 60)
        or len(shape) != 2
        or not 1 <= shape[0] <= maximum_controls
        or shape[1] < 1
    ):
        raise ValueError("explicit bounded time-major course required")
    for x in (reward, value, next_value):
        if x.shape != shape or x.dtype != np.float32 or not np.isfinite(x).all():
            raise ValueError("finite aligned float32 course values required")
    for mask in (terminated, truncated, valid):
        if mask.shape != shape or mask.dtype != np.bool_:
            raise ValueError("aligned boolean episode masks required")
    if (terminated & truncated).any() or ((terminated | truncated) & ~valid).any():
        raise ValueError("terminal event must belong to exactly one valid transition")
    advantage = np.zeros_like(reward)
    carry = np.zeros(shape[1], dtype=np.float32)
    for t in range(shape[0] - 1, -1, -1):
        delta = reward[t] + np.float32(0.99) * next_value[t] * (~terminated[t]) - value[t]
        carry = (
            delta + np.float32(0.99 * 0.95) * carry * (~(terminated[t] | truncated[t]))
        ) * valid[t]
        advantage[t] = carry
    return advantage, np.where(valid, advantage + value, 0).astype(np.float32)


def make_contact_actor_critic(actor: Any) -> Any:
    import copy
    import math

    import torch
    from torch import nn

    class NativeContactActorCritic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.actor = actor
            self.baseline = copy.deepcopy(actor).eval()
            for p in self.baseline.parameters():
                p.requires_grad_(False)
            last = nn.Linear(128, 1)
            self.critic = nn.Sequential(
                nn.Linear(177, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), last
            )
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
            self.log_std = nn.Parameter(torch.full((29,), math.log(0.05)))

        def value(self, obs: Any) -> Any:
            if (
                obs.ndim != 2
                or obs.shape[1] != 177
                or obs.dtype != torch.float32
                or not torch.isfinite(obs).all()
            ):
                raise ValueError("finite actual177 critic observations required")
            x = torch.cat(
                ((obs[:, :98] - self.actor.prior_mean) / self.actor.prior_denominator, obs[:, 98:]),
                dim=1,
            )
            return self.critic(x).squeeze(-1)

        def distribution(self, obs: Any) -> Any:
            mean = self.actor(obs)
            std = self.log_std.clamp(math.log(0.01), math.log(0.1)).exp().expand_as(mean)
            return torch.distributions.Normal(mean, std)

    return NativeContactActorCritic()


def native_ppo_loss(
    policy: Any,
    obs: Any,
    action: Any,
    old_logp: Any,
    old_value: Any,
    advantage: Any,
    returns: Any,
    rehearsal_obs: Any,
    *,
    value_only: bool = False,
) -> tuple[Any, dict[str, Any]]:
    import torch

    value = policy.value(obs)
    if value_only:
        loss = 0.5 * (value - returns).square().mean()
        return loss, dict(value=loss.detach(), approximate_kl=torch.zeros((), device=obs.device))
    distribution = policy.distribution(obs)
    logp = distribution.log_prob(action).sum(-1)
    log_ratio = logp - old_logp
    if not torch.isfinite(log_ratio).all() or log_ratio.abs().max() > 40:
        raise ValueError("invalid on-policy likelihood ratio")
    ratio = log_ratio.exp()
    actor_loss = -torch.minimum(ratio * advantage, ratio.clamp(0.9, 1.1) * advantage).mean()
    clipped_value = old_value + (value - old_value).clamp(-0.2, 0.2)
    critic_loss = (
        0.5 * torch.maximum((value - returns).square(), (clipped_value - returns).square()).mean()
    )
    with torch.no_grad():
        anchor = policy.baseline(obs)
        rehearsal_target = policy.baseline(rehearsal_obs)
    anchor_kl = ((distribution.loc - anchor).square() / (2 * 0.1**2)).sum(-1).mean()
    rehearsal = (policy.actor(rehearsal_obs) - rehearsal_target).square().mean()
    loss = (
        actor_loss
        + critic_loss
        + 0.01 * anchor_kl
        + rehearsal
        - 0.0002 * distribution.entropy().sum(-1).mean()
    )
    if not torch.isfinite(loss):
        raise ValueError("nonfinite native PPO objective")
    return loss, dict(
        actor=actor_loss.detach(),
        value=critic_loss.detach(),
        baseline_anchor_kl=anchor_kl.detach(),
        rehearsal_mse=rehearsal.detach(),
        approximate_kl=((ratio - 1) - log_ratio).mean().detach(),
    )
