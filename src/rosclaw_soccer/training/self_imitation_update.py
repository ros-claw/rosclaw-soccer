"""Bounded off-policy self-imitation math, separate from on-policy PPO.

Inspired by Oh et al., Self-Imitation Learning (ICML 2018),
https://arxiv.org/abs/1806.05635. This is not their prioritized-replay
implementation: callers own qualified demonstration selection, discounted
returns, sampling, RNG/checkpoints, retention exams and promotion gates.
"""

from collections.abc import Mapping
from typing import Any


def update_self_imitation(agent: Any, optimizer: Any, batch: Mapping[str, Any]) -> dict[str, Any]:
    """One actor/critic step using positive return gaps and a binary actor mask.

    Inputs are detached 169-feature observations, 32 raw latent actions,
    Monte Carlo returns and actor masks. No behavior likelihood is accepted,
    and this function never represents replay data as on-policy experience.
    Positive advantages are capped at one; one-sided Huber value weight is
    .01, Gaussian logstd floor -4, global gradient bound .5. No entropy bonus.
    Caller owns rollback if optimizer.step fails.
    """
    import torch

    if not isinstance(batch, Mapping) or set(batch) != {"obs", "raw", "returns", "actor_mask"}:
        raise ValueError("explicit off-policy self-imitation fields required")
    observation = batch["obs"]
    if not isinstance(observation, torch.Tensor) or observation.ndim != 2:
        raise ValueError("batched self-imitation observations required")
    n = len(observation)
    if not 1 <= n <= 4096:
        raise ValueError("bounded self-imitation minibatch required")
    shapes = {"obs": (n, 169), "raw": (n, 32), "returns": (n,), "actor_mask": (n,)}
    for key, value in batch.items():
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != shapes[key]
            or value.dtype != torch.float32
            or value.device != observation.device
            or value.requires_grad
            or not bool(torch.isfinite(value).all())
            or bool((value.abs() > 1e6).any())
        ):
            raise ValueError("finite aligned detached float32 self-imitation tensors required")
    if not bool(((batch["actor_mask"] == 0) | (batch["actor_mask"] == 1)).all()):
        raise ValueError("binary actor learning mask required")
    expected = {id(p) for p in agent.parameters() if p.requires_grad}
    actual = [id(p) for group in optimizer.param_groups for p in group["params"]]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("optimizer must exclusively own trainable learner parameters")
    mean, value = agent(observation)
    if (
        mean.shape != (n, 32)
        or value.shape != (n,)
        or agent.logstd.shape != (32,)
        or not bool(torch.isfinite(mean).all() and torch.isfinite(value).all())
        or not bool(torch.isfinite(agent.logstd).all())
    ):
        raise ValueError("finite explicit actor/critic output contract required")
    gap = (batch["returns"] - value).clamp_min(0)
    advantage = gap.detach().clamp_max(1)
    actor_keep = (batch["actor_mask"] == 1) & (advantage > 0)
    positive = int((advantage > 0).sum())
    result = dict(
        optimizer_steps=0,
        positive_value_samples=positive,
        positive_actor_samples=int(actor_keep.sum()),
        sample_count=n,
        on_policy=False,
        activation_ceiling="SIM_ONLY",
        promotion_eligible=False,
    )
    if positive == 0:
        return result  # Do not advance Adam momentum on an empty positive batch.
    critic_loss = 0.01 * torch.where(gap < 1, 0.5 * gap.square(), gap - 0.5).mean()
    loss: Any = critic_loss
    if bool(actor_keep.any()):
        distribution: Any = torch.distributions.Normal(
            mean[actor_keep], agent.logstd.clamp(-4, -0.3).exp()
        )
        logp = distribution.log_prob(batch["raw"][actor_keep]).sum(1)
        loss = loss - (logp * advantage[actor_keep]).mean()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("nonfinite self-imitation loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
    if not bool(torch.isfinite(norm)):
        raise FloatingPointError("nonfinite self-imitation gradient")
    optimizer.step()
    result.update(
        optimizer_steps=1, loss=float(loss.detach()), gradient_norm_before_clip=float(norm)
    )
    return result
