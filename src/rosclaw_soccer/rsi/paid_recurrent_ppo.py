"""Finite-option recurrent PPO math; no stale-hidden-state approximation.

Each update recomputes one complete, unpadded episode from zero h/c. The
declared finite source motion followed by the held-target tail is a natural
task terminal, NOT an unobserved time-limit truncation. No bootstrapped value
is invented beyond the source's final frame.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import torch

GAMMA = 0.995


def backtracked_adam_update(
    optimizer: Any,
    parameters: list[torch.Tensor],
    evaluate_kl: Callable[[], float],
    *,
    maximum_kl: float = 0.015,
    maximum_attempts: int = 8,
) -> dict[str, Any]:
    """Commit one actual Adam step only after a full TRAIN-history KL check.

    Rejected trials restore BOTH weights and Adam moments/step counters. The
    already-computed gradients are reused at successively smaller learning
    rates. This does not assert physical safety: new physics exams still gate.
    """
    import copy

    if not 0 < maximum_kl <= 0.015 or not 1 <= maximum_attempts <= 8:
        raise ValueError("bounded fixed trust-region protocol required")
    expected = [id(p) for group in optimizer.param_groups for p in group["params"]]
    if (
        not expected
        or len(set(expected)) != len(expected)
        or [id(p) for p in parameters] != expected
    ):
        raise ValueError("all optimizer parameters in exact unique order required for rollback")
    original = [p.detach().clone() for p in parameters]
    saved = copy.deepcopy(optimizer.state_dict())
    rates = [g["lr"] for g in optimizer.param_groups]
    attempts = []

    def restore() -> None:
        with torch.no_grad():
            for p, value in zip(parameters, original, strict=True):
                p.copy_(value)
        optimizer.load_state_dict(copy.deepcopy(saved))

    try:
        for attempt in range(maximum_attempts):
            if attempt:
                restore()
            for group, rate in zip(optimizer.param_groups, rates, strict=True):
                group["lr"] = rate * 0.5**attempt
            optimizer.step()
            kl = evaluate_kl()
            accepted = np.isfinite(kl) and 0 <= kl <= maximum_kl
            attempts.append(
                dict(
                    scale=0.5**attempt, kl=kl if np.isfinite(kl) else None, accepted=bool(accepted)
                )
            )
            if accepted:
                return dict(applied=True, attempts=attempts)
        restore()
        return dict(applied=False, attempts=attempts)
    except Exception:
        restore()
        raise


def finite_option_gae(
    value: np.ndarray, terminal_reward: float, tail_controls: int
) -> tuple[np.ndarray, np.ndarray]:
    if (
        value.ndim != 1
        or len(value) < 1
        or value.dtype != np.float32
        or not np.isfinite(value).all()
        or not np.isfinite(terminal_reward)
        or not 0 <= tail_controls <= 60
    ):
        raise ValueError("finite source episode with bounded actual held tail required")
    reward = np.zeros_like(value)
    reward[-1] = terminal_reward * GAMMA**tail_controls
    advantage = np.zeros_like(value)
    carry, next_value = 0.0, 0.0
    for t in range(len(value) - 1, -1, -1):
        delta = float(reward[t]) + GAMMA * next_value - float(value[t])
        carry = delta + GAMMA * 0.95 * carry
        advantage[t] = carry
        next_value = float(value[t])
    return advantage, (advantage + value).astype(np.float32)


def recurrent_episode_loss(
    actor: Any,
    anchor: Any,
    critic: Any,
    log_std: torch.Tensor,
    episode: dict[str, torch.Tensor],
    *,
    value_only: bool,
) -> tuple[torch.Tensor, dict[str, Any]]:
    obs = episode["obs"]
    if obs.ndim != 2 or obs.shape[1] != 160 or not 1 <= len(obs) <= 300:
        raise ValueError("one unpadded complete actual episode required")
    x = (obs - actor.mean) / actor.denominator
    value = critic(x).squeeze(-1)
    old = episode["value"]
    returns = episode["returns"]
    zero = torch.zeros((), device=obs.device)
    if value_only:
        loss = 0.5 * (value - returns).square().mean()
        return loss, dict(approximate_kl=zero, actor_rows=0, anchor_rows=0, critic_rows=len(obs))
    state = (torch.zeros(2, 1, 128, device=obs.device), torch.zeros(2, 1, 128, device=obs.device))
    mean, _ = actor(obs[:, None], state)
    mean = mean[:, 0]
    std = log_std.clamp(np.log(0.01), np.log(0.05)).exp()
    dist: Any = torch.distributions.Normal(mean, std)
    logratio = dist.log_prob(episode["action"]).sum(-1) - episode["logp"]
    if not torch.isfinite(logratio).all() or logratio.abs().max() > 40:
        raise ValueError("invalid full-history on-policy likelihood ratio")
    ratio = logratio.exp()
    policy = -torch.minimum(
        ratio * episode["advantage"], ratio.clamp(0.9, 1.1) * episode["advantage"]
    ).mean()
    clipped_value = old + (value - old).clamp(-0.2, 0.2)
    value_loss = (
        0.5 * torch.maximum((value - returns).square(), (clipped_value - returns).square()).mean()
    )
    with torch.no_grad():
        reference, _ = anchor(obs[:, None], state)
    anchor_kl = ((mean - reference[:, 0]).square() / (2 * 0.02**2)).sum(-1).mean()
    loss = policy + value_loss + 0.02 * anchor_kl - 0.0001 * dist.entropy().sum(-1).mean()
    if not torch.isfinite(loss):
        raise ValueError("nonfinite full-history recurrent PPO objective")
    return loss, dict(
        approximate_kl=((ratio - 1) - logratio).mean().detach(),
        anchor_kl=anchor_kl.detach(),
        value_loss=value_loss.detach(),
        actor_rows=len(obs),
        anchor_rows=len(obs),
        critic_rows=len(obs),
    )
