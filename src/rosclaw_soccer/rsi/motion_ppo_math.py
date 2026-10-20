"""Finite recurrent PPO math; no rollout, world or policy activation authority."""

from typing import Any

import numpy as np


def full_horizon_gae(rewards: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if (
        rewards.shape != values.shape
        or rewards.ndim != 2
        or rewards.shape[0] < 1
        or rewards.shape[1] != 300
        or rewards.dtype != np.float32
        or values.dtype != np.float32
        or not np.isfinite(rewards).all()
        or not np.isfinite(values).all()
    ):
        raise ValueError("finite full H300 rewards and own critic values required")
    advantages = np.zeros_like(rewards)
    carry = np.zeros(rewards.shape[0], dtype=np.float32)
    following = np.zeros_like(carry)  # True fixed terminal, never time-limit bootstrap.
    for t in range(299, -1, -1):
        delta = rewards[:, t] + np.float32(0.99) * following - values[:, t]
        carry = delta + np.float32(0.99 * 0.95) * carry
        advantages[:, t] = carry
        following = values[:, t]
    returns = advantages + values
    advantages[:, :20] = 0
    return advantages, returns


def clipped_ppo_loss(
    policy: Any,
    reference: Any,
    x: Any,
    raw: Any,
    old_log_prob: Any,
    old_values: Any,
    advantages: Any,
    returns: Any,
    *,
    quantized_likelihood: bool = False,
) -> tuple[Any, dict[str, Any]]:
    import torch

    lp, entropy, value, mean, std = policy.evaluate_raw(x, raw)
    with torch.no_grad():
        _, _, _, ref_mean, ref_std = reference.evaluate_raw(x, raw)
    if type(quantized_likelihood) is not bool:
        raise ValueError("explicit likelihood mode required")
    if old_log_prob.shape != lp.shape or not torch.isfinite(old_log_prob).all():
        raise ValueError("finite original on-policy scalar sequence required")
    if quantized_likelihood:
        from rosclaw_soccer.rsi.fsq_gaussian_mass import fsq_log_mass

        token = torch.round(15.515501022338867 * torch.tanh(raw + 0.03223699703812599) - 0.5) / 16
        lp = fsq_log_mass(mean, std, token).sum(-1)
        # The reference is the actual unchanged collection policy; integrate
        # its original Gaussian over the SAME observed symbols, not raw density.
        old_log_prob = fsq_log_mass(ref_mean, ref_std, token).sum(-1)
    mask = torch.zeros_like(lp, dtype=torch.bool)
    mask[:, 20:] = True
    for a in (old_log_prob, old_values, advantages, returns):
        if a.shape != lp.shape or not torch.isfinite(a).all():
            raise ValueError("complete finite on-policy scalar sequences required")
    adv = advantages[mask]
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    log_ratio = lp[mask] - old_log_prob[mask]
    if not torch.isfinite(log_ratio).all() or log_ratio.abs().max() > 40:
        raise ValueError("invalid or stale on-policy likelihood ratio")
    ratio = log_ratio.exp()
    actor = -torch.minimum(ratio * adv, ratio.clamp(0.8, 1.2) * adv).mean()
    value_clipped = old_values[mask] + (value[mask] - old_values[mask]).clamp(-0.2, 0.2)
    critic = torch.maximum(
        (value[mask] - returns[mask]).square(), (value_clipped - returns[mask]).square()
    ).mean()
    kl = (
        (
            torch.log(ref_std / std)
            + (std.square() + (mean - ref_mean).square()) / (2 * ref_std.square())
            - 0.5
        )
        .sum(-1)[mask]
        .mean()
    )
    loss = actor + 0.5 * critic + 0.01 * kl - 0.0002 * entropy[mask].mean()
    if not torch.isfinite(loss):
        raise ValueError("nonfinite PPO loss")
    return loss, dict(
        actor=actor,
        critic=critic,
        prior_kl=kl,
        entropy=entropy[mask].mean(),
        approx_old_kl=((ratio - 1) - log_ratio).mean(),
    )
