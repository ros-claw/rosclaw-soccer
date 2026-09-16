"""Bounded, replayable PPO updates for the Soccer full-body residual backend.

Only optimizer math lives here. Rollout collection, Core learning leases,
frozen-teacher audits, candidate retention and deployment are separate owners.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from rosclaw_soccer.training.stateless_forward import stateless_actor_critic_rows


@dataclass(frozen=True)
class FullBodyPPOUpdateConfig:
    epochs: int = 8
    minibatch_size: int = 1024
    gamma: float = 0.995
    trace_decay: float = 0.98
    target_kl: float = 0.02
    observation_size: int = 133
    action_size: int = 29
    minimum_log_std: float = -2.5
    learning_observation_index: int | None = None
    critic_all_active: bool = False
    episode_balanced_actor: bool = False
    group_relative_actor: bool = False
    binary_terminal_outcome: bool = False
    stateless_rowwise_forward: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.critic_all_active) is not bool
            or type(self.stateless_rowwise_forward) is not bool
            or type(self.episode_balanced_actor) is not bool
            or type(self.group_relative_actor) is not bool
            or type(self.binary_terminal_outcome) is not bool
            or (self.group_relative_actor and not self.episode_balanced_actor)
            or type(self.epochs) is not int
            or type(self.observation_size) is not int
            or type(self.action_size) is not int
            or (self.observation_size, self.action_size)
            not in (
                (39, 3),  # Read-only world-heading local navigation, frozen joint foundation.
                (40, 3),  # Same navigation features plus training-only influence bit.
                (133, 29),
                (135, 29),  # Existing receiving fields plus explicit world heading sin/cos.
                (138, 29),  # Heading fields plus an explicit next-skill target offset.
                (138, 3),  # Explicit local motor-synergy actions, not joint targets.
                (136, 29),
                (554, 29),  # Full kick547 + goal3 + rootvelocity3 + learning bit.
                (583, 29),  # Full kick554 plus explicit preceding sampled raw action29.
                (139, 32),
                (140, 32),  # 139 policy features plus training-only binary learning bit.
                (169, 32),  # Explicit foot/ball feedback plus one measured interval history.
                (170, 32),  # 169 policy features plus training-only binary learning bit.
                (170, 3),  # Contact features plus explicit selected foot; force latents.
                (171, 3),  # Same task-space policy plus training-only learning bit.
                (170, 6),  # Explicit carry force and bounded navigation corrections.
                (171, 6),  # Carry policy plus training-only influence bit.
                (182, 6),  # Local carry features plus explicit12 task-context features.
                (183, 6),  # Contextual carry plus training-only influence bit.
                (182, 32),  # Contextual full-body carry:29 joints and3 navigation latents.
                (183, 32),  # Full-body carry plus training-only influence bit.
                (136, 3),
                (143, 3),
                (143, 29),
                (149, 29),  # Receiving143 plus explicit bilateral foot-relative velocity6.
                (150, 29),  # Same feedback plus a separate binary plasticity mask.
                (136, 30),
                (137, 30),  # 136-state actor plus causal binary successor feature.
                (142, 30),
            )
            or not 1 <= self.epochs <= 16
            or type(self.minibatch_size) is not int
            or not 1 <= self.minibatch_size <= 4096
            or not all(math.isfinite(x) for x in (self.gamma, self.trace_decay, self.target_kl))
            or (
                (
                    type(self.gamma) not in (int, float)
                    or type(self.trace_decay) not in (int, float)
                    or self.gamma != 1.0
                    or self.trace_decay != 1.0
                )
                if self.binary_terminal_outcome
                else (not 0.9 <= self.gamma < 1 or not 0.9 <= self.trace_decay < 1)
            )
            or not 0 < self.target_kl <= 0.05
            or type(self.minimum_log_std) not in (float, int)
            or not math.isfinite(self.minimum_log_std)
            or not -6.0 <= self.minimum_log_std <= -2.5
            or (
                self.learning_observation_index is not None
                and (
                    type(self.learning_observation_index) is not int
                    or not 0 <= self.learning_observation_index < self.observation_size
                )
            )
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
    A smaller exploration floor must also be used by the rollout collector;
    starting-likelihood validation rejects mismatched distributions. The default
    retains the historical -2.5 floor exactly. This does not change action limits.

    Stateless row-wise forwarding is separately opt-in. It uses singleton
    arithmetic for rollout-likelihood checks, actor/critic updates and KL checks;
    it never lowers the likelihood tolerance or adapts recurrent controllers.
    An optional binary observation feature selects optimization samples only:
    all live likelihoods/values are still verified, and GAE uses the continuous
    episode. Feature provenance and candidate admission remain caller-owned.
    With critic_all_active, value targets cover all live samples once per
    completed epoch; only actor likelihood loss uses the learning window.
    Optional episode balancing gives each world with selected actor samples
    equal total policy-loss weight, instead of favouring longer action windows.
    It changes neither GAE, advantage normalization, critic weights nor sample
    selection. Divergence checks retain both time-step and balanced views.
    A separate group-relative option replaces only the actor advantage with
    normalized discounted episode outcomes within explicit episode_group IDs.
    It requires episode balancing; critic GAE and validation stay unchanged.
    This is a continuous-action research variant, not full paper GRPO.
    Binary terminal outcome mode explicitly uses gamma=lambda=1, requires
    complete zero-padded episodes and only terminal binary rewards. It cannot
    silently reinterpret dense shaped rewards as success-probability learning.
    The default path preserves historical optimizer ordering and RNG usage.
    """
    import torch

    active = config or FullBodyPPOUpdateConfig()
    required = {"obs", "raw", "logp", "value", "reward", "alive", "next_alive"}
    if active.group_relative_actor:
        required.add("episode_group")
    if set(rollout) != required:
        raise ValueError("complete on-policy rollout fields required")
    data = dict(rollout)
    group_ids = data.pop("episode_group", None)
    if any(not isinstance(value, torch.Tensor) for value in data.values()):
        raise ValueError("complete detached rollout tensors required")
    if data["reward"].ndim != 2:
        raise ValueError("episodic time/world reward axes required")
    horizon, worlds = data["reward"].shape
    if not 1 <= horizon <= 4096 or not 1 <= worlds <= 4096 or horizon * worlds > 65536:
        raise ValueError("bounded rollout dimensions required")
    for name, value in data.items():
        shape: tuple[int, ...] = (
            horizon,
            worlds,
            active.observation_size if name == "obs" else active.action_size,
        )
        if name not in {"obs", "raw"}:
            shape = (horizon, worlds)
        if (
            value.shape != shape
            or value.requires_grad
            or value.layout != torch.strided
            or torch.is_complex(value)
            or value.device != data["obs"].device
            or (name not in {"alive", "next_alive"} and not torch.is_floating_point(value))
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError("finite detached rollout tensors with matching shapes required")
    for name in ("alive", "next_alive"):
        if not bool(((data[name] == 0) | (data[name] == 1)).all()):
            raise ValueError("binary episode activity required")
    if bool((data["next_alive"] > data["alive"]).any()) or not bool(
        (data["alive"][1:] == data["next_alive"][:-1]).all()
    ):
        raise ValueError("rollout contains an undeclared episode reset")
    if active.binary_terminal_outcome:
        terminal = (data["alive"] == 1) & (data["next_alive"] == 0)
        if (
            bool((data["next_alive"][-1] != 0).any())
            or any(bool((value[data["alive"] == 0] != 0).any()) for value in data.values())
            or not bool(((data["reward"] == 0) | (data["reward"] == 1)).all())
            or bool(((data["reward"] != 0) & ~terminal).any())
        ):
            raise ValueError("complete zero-padded binary terminal outcome episodes required")
    expected = {id(p) for p in agent.parameters() if p.requires_grad}
    actual = [id(p) for group in optimizer.param_groups for p in group["params"]]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("optimizer does not exclusively own the declared learner")
    keep = data["alive"].reshape(-1) > 0
    if int(keep.sum()) < 2:
        raise ValueError("at least two active learning samples required")
    learning_keep = keep
    if active.learning_observation_index is not None:
        feature = data["obs"][..., active.learning_observation_index]
        if not bool(((feature == 0) | (feature == 1)).all()):
            raise ValueError("binary learning-window observation required")
        learning_keep = keep & (feature.reshape(-1) == 1)
        if int(learning_keep.sum()) < 2:
            raise ValueError("at least two selected learning samples required")
    obs = data["obs"].reshape(-1, active.observation_size)[keep]
    raw = data["raw"].reshape(-1, active.action_size)[keep]
    old = data["logp"].reshape(-1)[keep]
    with torch.no_grad():
        mean, value = (
            stateless_actor_critic_rows(agent, obs)
            if active.stateless_rowwise_forward
            else agent(obs)
        )
        distribution: Any = torch.distributions.Normal(
            mean, agent.logstd.clamp(active.minimum_log_std, -0.3).exp()
        )
        if not bool((distribution.log_prob(raw).sum(1) - old).abs().max() <= 1e-3) or not bool(
            (value - data["value"].reshape(-1)[keep]).abs().max() <= 1e-3
        ):
            raise ValueError("rollout likelihood or critic differs from the starting learner")
    all_obs = obs
    all_keep = keep
    if active.learning_observation_index is not None:
        keep = learning_keep
        obs = data["obs"].reshape(-1, active.observation_size)[keep]
        raw = data["raw"].reshape(-1, active.action_size)[keep]
        old = data["logp"].reshape(-1)[keep]
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
    all_returns = (advantage + data["value"]).reshape(-1)[all_keep]
    adv = advantage.reshape(-1)[keep]
    if active.group_relative_actor:
        from rosclaw_soccer.training.episode_group_advantage import group_relative_advantages

        discount = active.gamma ** torch.arange(
            horizon, device=obs.device, dtype=data["reward"].dtype
        )
        outcomes = (data["reward"] * data["alive"] * discount[:, None]).sum(0)
        outcome_advantage = group_relative_advantages(outcomes, group_ids)
        adv = outcome_advantage.expand(horizon, worlds).reshape(-1)[keep]
    else:
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
    actor_weights = None
    if active.episode_balanced_actor:
        counts = keep.reshape(horizon, worlds).sum(0)
        represented = (counts > 0).sum()
        per_world = len(obs) / (represented * counts.clamp_min(1))
        actor_weights = per_world.expand(horizon, worlds).reshape(-1)[keep]
    steps, kl, completed = 0, 0.0, 0
    for epoch in range(active.epochs):
        permutation: Any = torch.randperm(len(obs), device=obs.device)
        batches = permutation.split(active.minibatch_size)
        critic_batches = (
            torch.tensor_split(torch.randperm(len(all_obs), device=obs.device), len(batches))
            if active.critic_all_active
            else ()
        )
        for batch_index, indices in enumerate(batches):
            mean, value = (
                stateless_actor_critic_rows(agent, obs[indices])
                if active.stateless_rowwise_forward
                else agent(obs[indices])
            )
            distribution = torch.distributions.Normal(
                mean, agent.logstd.clamp(active.minimum_log_std, -0.3).exp()
            )
            logp = distribution.log_prob(raw[indices]).sum(1)
            ratio = (logp - old[indices]).exp()
            surrogate = torch.minimum(ratio * adv[indices], ratio.clamp(0.8, 1.2) * adv[indices])
            policy_loss = -(
                surrogate if actor_weights is None else surrogate * actor_weights[indices]
            ).mean()
            loss = policy_loss
            if not active.critic_all_active:
                loss = loss + 0.5 * (value - returns[indices]).square().mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite full-body PPO loss")
            optimizer.zero_grad()
            loss.backward()
            if active.critic_all_active:
                critic_indices: Any = critic_batches[batch_index]
                # Bound forward/backward memory without changing the mean loss.
                for chunk in critic_indices.split(active.minibatch_size):
                    _, prediction = (
                        stateless_actor_critic_rows(agent, all_obs[chunk])
                        if active.stateless_rowwise_forward
                        else agent(all_obs[chunk])
                    )
                    critic_loss = (
                        0.5 * (prediction - all_returns[chunk]).square().sum() / len(critic_indices)
                    )
                    if not bool(torch.isfinite(critic_loss)):
                        raise FloatingPointError("nonfinite full-episode critic loss")
                    critic_loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError("nonfinite full-body PPO gradient")
            optimizer.step()
            steps += 1
        with torch.no_grad():
            mean, _ = (
                stateless_actor_critic_rows(agent, obs)
                if active.stateless_rowwise_forward
                else agent(obs)
            )
            distribution = torch.distributions.Normal(
                mean, agent.logstd.clamp(active.minimum_log_std, -0.3).exp()
            )
            logp = distribution.log_prob(raw).sum(1)
            delta = logp - old
            kl = float((delta.exp() - 1 - delta).mean())
            if actor_weights is not None:
                balanced_kl = float(((delta.exp() - 1 - delta) * actor_weights).mean())
                if not math.isfinite(balanced_kl):
                    raise FloatingPointError("nonfinite balanced full-body PPO divergence")
                kl = max(kl, balanced_kl)
        if not math.isfinite(kl):
            raise FloatingPointError("nonfinite full-body PPO divergence")
        completed = epoch + 1
        if kl > active.target_kl:
            break
    result = {
        "optimizer_steps": steps,
        "approx_kl": kl,
        "epochs_completed": completed,
        "active_samples": len(obs),
        "on_policy_inputs_verified": True,
        "minimum_log_std": active.minimum_log_std,
        "activation_ceiling": "SIM_ONLY",
        "promotion_eligible": False,
    }
    if active.learning_observation_index is not None:
        result.update(
            learning_observation_index=active.learning_observation_index,
            verified_active_samples=int(data["alive"].sum()),
        )
    if active.critic_all_active:
        result.update(critic_all_active=True, critic_samples_per_epoch=len(all_obs))
    if actor_weights is not None:
        result.update(
            episode_balanced_actor=True,
            actor_weight_min=float(actor_weights.min()),
            actor_weight_max=float(actor_weights.max()),
            actor_represented_worlds=int(represented),
        )
    if active.group_relative_actor:
        result.update(
            group_relative_actor=True, actor_task_groups=int(torch.unique(group_ids).numel())
        )
    if active.binary_terminal_outcome:
        result.update(binary_terminal_outcome=True)
    if active.stateless_rowwise_forward:
        result.update(stateless_rowwise_forward=True)
    return result
