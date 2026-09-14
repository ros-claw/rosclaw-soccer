"""Transactional backtracking for an exclusively owned, stateless PPO learner.

Never run this around a live serving policy. It temporarily tries candidate
parameters, measures raw Gaussian divergence on the collected learning states,
and restores rejected trials. This is not physical safety or skill retention.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .full_body_ppo import FullBodyPPOUpdateConfig, update_full_body_ppo
from .torch_rng_snapshot import capture_torch_rng, restore_torch_rng


@dataclass(frozen=True)
class PPOTrustRegion:
    maximum_mean_kl: float = 0.01
    maximum_state_kl: float = 0.05
    backtrack_factor: float = 0.1
    maximum_attempts: int = 4

    def __post_init__(self) -> None:
        values = (self.maximum_mean_kl, self.maximum_state_kl, self.backtrack_factor)
        if (
            any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
            or not 0 < self.maximum_mean_kl <= self.maximum_state_kl <= 1
            or not 0 < self.backtrack_factor < 1
            or type(self.maximum_attempts) is not int
            or not 1 <= self.maximum_attempts <= 8
        ):
            raise ValueError("bounded explicit PPO trust region required")


def guarded_full_body_ppo_update(
    agent: Any,
    optimizer: Any,
    data: Mapping[str, Any],
    config: FullBodyPPOUpdateConfig,
    *,
    trust_region: PPOTrustRegion | None = None,
    cuda_devices: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Retry the same on-policy update from its original state at smaller LRs.

    Accepted candidates satisfy both analytic old→new Gaussian KL limits on
    every selected collected state. This is not a global policy/state bound.
    All attempts count toward ``optimizer_steps``; only the last accepted
    attempt contributes ``applied_optimizer_steps``. Rejected trials restore
    model/buffers, parameter gradients, optimizer, modes and explicit Torch
    RNGs. CPU/CUDA backend failures can prevent restoration: quarantine the
    learner and retain its durable prior checkpoint if restoration raises.
    Arbitrary external side effects or custom non-state-dict caches are not
    covered; use a stateless numerical model with deterministic forward passes
    in an exclusively owned process (no dropout or mutable forward caches).
    """
    import torch

    region = PPOTrustRegion() if trust_region is None else trust_region
    if not isinstance(config, FullBodyPPOUpdateConfig) or not isinstance(region, PPOTrustRegion):
        raise ValueError("validated PPO and trust-region configs required")
    parameters = list(agent.parameters())
    owned = {id(p) for p in parameters}
    optimized = [p for group in optimizer.param_groups for p in group["params"]]
    if (
        not optimized
        or len({id(p) for p in optimized}) != len(optimized)
        or any(id(p) not in owned for p in optimized)
    ):
        raise ValueError("optimizer must own only distinct learner parameters")
    tensors = (
        parameters
        + list(agent.buffers())
        + [value for value in data.values() if isinstance(value, torch.Tensor)]
    )
    if any(t.device.type not in ("cpu", "cuda") for t in tensors):
        raise ValueError("only explicitly snapshotted CPU/CUDA learners supported")
    used_cuda = {str(t.device) for t in tensors if t.device.type == "cuda"}
    if type(cuda_devices) is not tuple or set(cuda_devices) != used_cuda:
        raise ValueError("exact explicit CUDA RNG devices required")
    original_lrs = [group["lr"] for group in optimizer.param_groups]
    if any(
        type(lr) not in (int, float) or not math.isfinite(lr) or not 0 <= lr <= 1
        for lr in original_lrs
    ):
        raise ValueError("finite scalar learning rates in [0, 1] required")
    state = copy.deepcopy(agent.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    gradients = [None if p.grad is None else p.grad.detach().clone() for p in parameters]
    modes = [(module, module.training) for module in agent.modules()]
    rng = capture_torch_rng(cuda_devices=cuda_devices)

    def restore() -> None:
        agent.load_state_dict(state, strict=True)
        optimizer.load_state_dict(copy.deepcopy(optimizer_state))
        for parameter, gradient in zip(parameters, gradients, strict=True):
            parameter.grad = None if gradient is None else gradient.clone()
        for module, mode in modes:
            module.training = mode
        restore_torch_rng(rng, cuda_devices=cuda_devices)

    attempts: list[dict[str, Any]] = []
    steps = 0
    try:
        obs = data["obs"].reshape(-1, config.observation_size)
        selected = data["alive"].reshape(-1) == 1
        if config.learning_observation_index is not None:
            selected = selected & (obs[:, config.learning_observation_index] == 1)
        obs = obs[selected]
        if len(obs) < 2 or not bool(torch.isfinite(obs).all()):
            raise ValueError("at least two finite selected learning states required")
        with torch.no_grad():
            old_mean, _ = agent(obs)
            old_mean = old_mean.detach().to(torch.float64).clone()
            old_logstd = (
                agent.logstd.detach().clamp(config.minimum_log_std, -0.3).to(torch.float64).clone()
            )
            if old_mean.shape != (len(obs), config.action_size) or old_logstd.shape != (
                config.action_size,
            ):
                raise ValueError("one diagonal Gaussian action vector per state required")
            if not bool(torch.isfinite(old_mean).all()) or not bool(
                torch.isfinite(old_logstd).all()
            ):
                raise ValueError("finite starting Gaussian policy required")
        for attempt in range(region.maximum_attempts):
            restore()
            rates = [float(lr * region.backtrack_factor**attempt) for lr in original_lrs]
            for group, lr in zip(optimizer.param_groups, rates, strict=True):
                group["lr"] = lr
            result = update_full_body_ppo(agent, optimizer, data, config)
            steps += result["optimizer_steps"]
            with torch.no_grad():
                new_mean, _ = agent(obs)
                new_logstd = (
                    agent.logstd.detach().clamp(config.minimum_log_std, -0.3).to(torch.float64)
                )
                variance_ratio = (
                    (2 * old_logstd).exp() + (old_mean - new_mean.to(torch.float64)).square()
                ) / (2 * new_logstd).exp()
                per_state = (
                    (new_logstd - old_logstd + 0.5 * (variance_ratio - 1)).sum(-1).clamp_min(0)
                )
                mean_kl = float(per_state.mean())
                maximum_kl = float(per_state.max())
            finite = math.isfinite(mean_kl) and math.isfinite(maximum_kl)
            accepted = (
                finite
                and mean_kl <= region.maximum_mean_kl
                and maximum_kl <= region.maximum_state_kl
            )
            attempts.append(
                dict(
                    learning_rates=rates,
                    update=result,
                    accepted=accepted,
                    mean_kl=mean_kl if finite else None,
                    maximum_state_kl=maximum_kl if finite else None,
                )
            )
            if accepted:
                return dict(
                    **{k: v for k, v in result.items() if k != "optimizer_steps"},
                    accepted=True,
                    status="ACCEPTED_WITHIN_COLLECTED_STATE_TRUST_REGION",
                    optimizer_steps=steps,
                    applied_optimizer_steps=result["optimizer_steps"],
                    attempts=attempts,
                    trust_region_mean_kl=mean_kl,
                    trust_region_maximum_state_kl=maximum_kl,
                )
        restore()
        return dict(
            accepted=False,
            status="REJECTED_AND_RESTORED",
            optimizer_steps=steps,
            applied_optimizer_steps=0,
            attempts=attempts,
            trust_region_mean_kl=0.0,
            trust_region_maximum_state_kl=0.0,
            activation_ceiling="SIM_ONLY",
            promotion_eligible=False,
        )
    except BaseException:
        restore()
        raise
