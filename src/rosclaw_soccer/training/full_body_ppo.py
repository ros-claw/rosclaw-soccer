"""Bounded, replayable PPO updates for the Soccer full-body residual backend.

Only optimizer math lives here. Rollout collection, Core learning leases,
frozen-teacher audits, candidate retention and deployment are separate owners.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FullBodyPPOUpdateConfig:
    epochs: int = 8
    minibatch_size: int = 1024
    gamma: float = 0.995
    trace_decay: float = 0.98
    target_kl: float = 0.02
    observation_size: int = 133

    def __post_init__(self) -> None:
        if (
            type(self.epochs) is not int
            or type(self.observation_size) is not int
            or self.observation_size not in (133, 136)
            or not 1 <= self.epochs <= 16
            or type(self.minibatch_size) is not int
            or not 1 <= self.minibatch_size <= 4096
            or not all(math.isfinite(x) for x in (self.gamma, self.trace_decay, self.target_kl))
            or not 0.9 <= self.gamma < 1
            or not 0.9 <= self.trace_decay < 1
            or not 0 < self.target_kl <= 0.05
        ):
            raise ValueError("bounded PPO update configuration required")


def update_full_body_ppo(
    agent: Any,
    optimizer: Any,
    rollout: Mapping[str, Any],
    config: FullBodyPPOUpdateConfig | None = None,
) -> dict[str, Any]:
    """Validate on-policy evidence before updating; no candidate activation.

    Caller owns rollback if an optimizer raises. Episodic horizons are terminal
    (zero bootstrap), and dead worlds cannot silently revive inside a rollout.
    Restore Torch RNG and optimizer state to replay the same minibatch order.
    """
    import torch

    active = config or FullBodyPPOUpdateConfig()
    required = {"obs", "raw", "logp", "value", "reward", "alive", "next_alive"}
    if set(rollout) != required:
        raise ValueError("complete on-policy rollout fields required")
    data = dict(rollout)
    if data["reward"].ndim != 2:
        raise ValueError("episodic time/world reward axes required")
    horizon, worlds = data["reward"].shape
    if not 1 <= horizon <= 4096 or not 1 <= worlds <= 4096 or horizon * worlds > 65536:
        raise ValueError("bounded rollout dimensions required")
    for name, value in data.items():
        shape: tuple[int, ...] = (horizon, worlds, active.observation_size if name == "obs" else 29)
        if name not in {"obs", "raw"}:
            shape = (horizon, worlds)
        if value.shape != shape or value.requires_grad or not bool(torch.isfinite(value).all()):
            raise ValueError("finite detached rollout tensors with matching shapes required")
    for name in ("alive", "next_alive"):
        if not bool(((data[name] == 0) | (data[name] == 1)).all()):
            raise ValueError("binary episode activity required")
    if bool((data["next_alive"] > data["alive"]).any()) or not bool(
        (data["alive"][1:] == data["next_alive"][:-1]).all()
    ):
        raise ValueError("rollout contains an undeclared episode reset")
    expected = {id(p) for p in agent.parameters() if p.requires_grad}
    actual = [id(p) for group in optimizer.param_groups for p in group["params"]]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("optimizer does not exclusively own the declared learner")
    keep = data["alive"].reshape(-1) > 0
    if int(keep.sum()) < 2:
        raise ValueError("at least two active learning samples required")
    obs = data["obs"].reshape(-1, active.observation_size)[keep]
    raw = data["raw"].reshape(-1, 29)[keep]
    old = data["logp"].reshape(-1)[keep]
    with torch.no_grad():
        mean, value = agent(obs)
        distribution: Any = torch.distributions.Normal(mean, agent.logstd.clamp(-2.5, -0.3).exp())
        if not bool((distribution.log_prob(raw).sum(1) - old).abs().max() <= 1e-3) or not bool(
            (value - data["value"].reshape(-1)[keep]).abs().max() <= 1e-3
        ):
            raise ValueError("rollout likelihood or critic differs from the starting learner")
    advantage = torch.zeros_like(data["reward"])
    carry = torch.zeros(worlds, device=obs.device)
    for frame in reversed(range(horizon)):
        following = (
            data["value"][frame + 1]
            if frame + 1 < horizon
            else torch.zeros(worlds, device=obs.device)
        )
        delta = (
            data["reward"][frame]
            + active.gamma * data["next_alive"][frame] * following
            - data["value"][frame]
        )
        carry = (
            delta + active.gamma * active.trace_decay * data["next_alive"][frame] * carry
        ) * data["alive"][frame]
        advantage[frame] = carry
    returns = (advantage + data["value"]).reshape(-1)[keep]
    adv = advantage.reshape(-1)[keep]
    adv = (adv - adv.mean()) / (adv.std() + 1e-6)
    steps, kl, completed = 0, 0.0, 0
    for epoch in range(active.epochs):
        permutation: Any = torch.randperm(len(obs), device=obs.device)
        for indices in permutation.split(active.minibatch_size):
            mean, value = agent(obs[indices])
            distribution = torch.distributions.Normal(mean, agent.logstd.clamp(-2.5, -0.3).exp())
            logp = distribution.log_prob(raw[indices]).sum(1)
            ratio = (logp - old[indices]).exp()
            loss = (
                -torch.minimum(ratio * adv[indices], ratio.clamp(0.8, 1.2) * adv[indices]).mean()
                + 0.5 * (value - returns[indices]).square().mean()
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite full-body PPO loss")
            optimizer.zero_grad()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError("nonfinite full-body PPO gradient")
            optimizer.step()
            steps += 1
        with torch.no_grad():
            mean, _ = agent(obs)
            distribution = torch.distributions.Normal(mean, agent.logstd.clamp(-2.5, -0.3).exp())
            logp = distribution.log_prob(raw).sum(1)
            delta = logp - old
            kl = float((delta.exp() - 1 - delta).mean())
        if not math.isfinite(kl):
            raise FloatingPointError("nonfinite full-body PPO divergence")
        completed = epoch + 1
        if kl > active.target_kl:
            break
    return {
        "optimizer_steps": steps,
        "approx_kl": kl,
        "epochs_completed": completed,
        "active_samples": len(obs),
        "on_policy_inputs_verified": True,
        "activation_ceiling": "SIM_ONLY",
        "promotion_eligible": False,
    }
