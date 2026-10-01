"""Full-batch PPO with the actual history-conditioned AR(1) likelihood."""

import copy
import re
from pathlib import Path
from typing import Any

import numpy as np
from rosclaw.growth.correlated_exploration import conditional_mean

from rosclaw_soccer.rsi.smooth_memory_motor import (
    CompiledSmoothMemoryMotor,
    make_preview,
    validate_model,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def fit_update(model: dict[str, Any], arrays: Any, *, batch_hash: str, rho: float = 0.9) -> Any:
    import torch

    validate_model(model)
    x, phase, action, old_logp, returns, marginal_std, groups = [
        np.asarray(arrays[k])
        for k in (
            "observation",
            "phase_index",
            "latent_action",
            "old_log_probability",
            "terminal_return",
            "std_raw",
            "trajectory_index",
        )
    ]
    n = len(x)
    if (
        model["generation"] >= 32
        or not 1080 <= n <= 200000
        or n % 270
        or x.shape != (n, 134)
        or action.shape != (n, 12)
        or any(a.shape != (n,) for a in (phase, old_logp, returns, marginal_std, groups))
        or not all(
            np.isfinite(a).all() for a in (x, phase, action, old_logp, returns, marginal_std)
        )
        or phase.dtype.kind not in "iu"
        or groups.dtype.kind not in "iu"
        or set(phase.tolist()) != {0, 1, 2}
        or not np.array_equal(groups, np.repeat(np.arange(n // 270), 270))
        or np.any((marginal_std < 0.01) | (marginal_std > 0.15))
        or type(rho) not in (float, int)
        or not np.isfinite(rho)
        or not 0 <= rho <= 0.95
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", batch_hash)
    ):
        raise ValueError("complete ordered finite current-parent AR physical batch required")
    if any(
        not np.all(returns[groups == g] == returns[groups == g][0])
        or not np.all(marginal_std[groups == g] == marginal_std[groups == g][0])
        for g in range(n // 270)
    ):
        raise ValueError("one actual return and declared marginal scale per trajectory required")
    first = np.arange(n) % 270 == 0
    std = marginal_std * np.where(first, 1.0, np.sqrt(1 - rho**2))
    if np.any(std < 0.01):
        raise ValueError("bounded conditional innovation scale required")
    previous_action = np.roll(action, 1, axis=0)

    def conditioned(mean: Any) -> Any:
        previous = np.roll(mean, 1, axis=0)
        return np.where(
            first[:, None], mean, conditional_mean(mean, previous, previous_action, rho)
        )

    decoder = CompiledSmoothMemoryMotor(make_preview(model))
    current = np.stack([decoder.raw_mean(v, int(p)) for v, p in zip(x, phase, strict=True)])
    conditional = conditioned(current)
    density = np.sum(
        -0.5 * ((action - conditional) / std[:, None]) ** 2
        - np.log(std[:, None])
        - 0.5 * np.log(2 * np.pi),
        axis=1,
    )
    if not np.allclose(density, old_logp, atol=1e-8, rtol=0):
        raise ValueError("batch is not the actual current history-conditioned actor")
    phi = np.stack([decoder._parent.features(v) for v in x])
    context = np.column_stack((phi[:, :134], phase))
    gates = decoder._guard.gates(context)
    original_layers = [
        (np.asarray(v["weight"]), np.asarray(v["bias"])) for v in model["residual_layers"]
    ]
    hidden = context
    for weight, bias in original_layers:
        hidden = np.tanh(hidden @ weight.T + bias)
    baseline = current - 0.05 * gates[:, None] * hidden
    target_mean, target_scale = float(returns.mean()), max(float(returns.std()), 1.0)
    targets = (returns - target_mean) / target_scale
    values = np.zeros(n)
    critic = np.zeros((3, 512))

    def regression(ids: Any) -> Any:
        design = phi[ids]
        if len(design) < 50:
            raise ValueError("whole-trajectory critic cross-fit support too small")
        return np.linalg.solve(design.T @ design + 0.01 * np.eye(512), design.T @ targets[ids])

    for p in range(3):
        for fold in range(4):
            test = (phase == p) & (groups % 4 == fold)
            values[test] = phi[test] @ regression((phase == p) & (groups % 4 != fold))
        critic[p] = regression(phase == p)
    advantage = targets - values
    advantage = (advantage - advantage.mean()) / max(float(advantage.std()), 1e-6)
    torch.set_num_threads(4)
    torch.manual_seed(202610348)
    torch.use_deterministic_algorithms(True)
    weights = [torch.nn.Parameter(torch.tensor(w, dtype=torch.float64)) for w, _ in original_layers]
    biases = [torch.nn.Parameter(torch.tensor(b, dtype=torch.float64)) for _, b in original_layers]
    parameters = weights + biases
    optimizer = torch.optim.Adam(parameters, lr=1e-4)
    inp, base, gate, desired, noise, marginal_noise, importance = [
        torch.tensor(v, dtype=torch.float64)
        for v in (
            context,
            baseline,
            gates[:, None],
            action,
            std[:, None],
            marginal_std[:, None],
            advantage,
        )
    ]
    first_mask = torch.tensor(first[:, None])
    prev_action = torch.tensor(previous_action, dtype=torch.float64)

    def mean() -> Any:
        hidden = inp
        for w, b in zip(weights, biases, strict=True):
            hidden = torch.tanh(hidden @ w.T + b)
        return base + 0.05 * gate * hidden

    def condition(mu: Any) -> Any:
        # Previous candidate means remain in the gradient graph. Substituting
        # detached old means here would silently train the wrong policy.
        return torch.where(first_mask, mu, mu + rho * (prev_action - mu.roll(1, dims=0)))

    def log_probability(mu: Any) -> Any:
        return (
            -0.5 * ((desired - mu) / noise).square() - noise.log() - 0.5 * np.log(2 * np.pi)
        ).sum(dim=1)

    with torch.no_grad():
        original = mean().clone()
        original_conditional = condition(original).clone()
        logp0 = log_probability(original_conditional).clone()

    def objective() -> tuple[Any, Any, Any]:
        mu = mean()
        conditional = condition(mu)
        ratio = torch.exp(log_probability(conditional) - logp0)
        kl = ((conditional - original_conditional).square() / (2 * noise.square())).sum(1).mean()
        marginal_kl = ((mu - original).square() / (2 * marginal_noise.square())).sum(1).mean()
        loss = -torch.minimum(ratio * importance, ratio.clamp(0.8, 1.2) * importance).mean()
        return loss + 10 * kl, kl, marginal_kl

    history = [float(objective()[0].detach())]
    for _ in range(160):
        optimizer.zero_grad()
        loss, _, _ = objective()
        if not torch.isfinite(loss):
            raise ValueError("nonfinite AR policy gradient")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1)
        previous = [p.detach().clone() for p in parameters]
        optimizer.step()
        directions = [p.detach().clone() - old for p, old in zip(parameters, previous, strict=True)]
        accepted = False
        for reduction in range(13):
            with torch.no_grad():
                for parameter, old, direction in zip(parameters, previous, directions, strict=True):
                    parameter.copy_(old + (0.5**reduction) * direction)
                trial, kl, marginal_kl = objective()
                if (
                    torch.isfinite(trial)
                    and max(float(kl), float(marginal_kl)) <= 0.0049
                    and float(trial) < history[-1] - 1e-9
                ):
                    history.append(float(trial))
                    accepted = True
                    break
        if not accepted:
            with torch.no_grad():
                for parameter, old in zip(parameters, previous, strict=True):
                    parameter.copy_(old)
            break
    if len(history) == 1:
        raise ValueError("no actual smooth-memory learning")
    result = {k: copy.deepcopy(v) for k, v in model.items() if k != "model_hash"}
    result["residual_layers"] = [
        dict(weight=w.detach().numpy().tolist(), bias=b.detach().numpy().tolist())
        for w, b in zip(weights, biases, strict=True)
    ]
    result["generation"] += 1
    receipt = dict(
        algorithm="SMOOTH_MEMORY_AR1_PPO_MC_TERMINAL",
        rho=float(rho),
        all_residual_layers_trainable=True,
        frozen_parent=True,
        distributional_retention_guaranteed=False,
        physical_batch_hash=batch_hash,
        learner_parent_hash=model["model_hash"],
        optimizer_source_hash=hash_bytes(Path(__file__).read_bytes()),
        completed_optimizer_steps=len(history) - 1,
        full_batch_loss_history=history,
        exact_mean_conditional_kl=0.0,
        exact_mean_marginal_kl=0.0,
        critic_crossfit_unit="whole_rollout",
        critic_crossfit_folds=4,
        critic_ridge=0.01,
        critic_target_mean=target_mean,
        critic_target_scale=target_scale,
        physical_rollout_count=n // 270,
        frame_sample_count=n,
        gamma=1.0,
        gae_lambda=1.0,
        learning_rate=1e-4,
        kl_penalty=10.0,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    result["learning_receipt"] = receipt
    result["critic_readout"] = critic.tolist()
    result["model_hash"] = hash_json(result)
    candidate = CompiledSmoothMemoryMotor(make_preview(result))
    final = np.stack([candidate.raw_mean(v, int(p)) for v, p in zip(x, phase, strict=True)])
    receipt["exact_mean_conditional_kl"] = float(
        np.mean(np.sum((conditioned(final) - conditional) ** 2 / (2 * std[:, None] ** 2), axis=1))
    )
    receipt["exact_mean_marginal_kl"] = float(
        np.mean(np.sum((final - current) ** 2 / (2 * marginal_std[:, None] ** 2), axis=1))
    )
    if any(
        not np.isfinite(receipt[k]) or receipt[k] > 0.005
        for k in ("exact_mean_conditional_kl", "exact_mean_marginal_kl")
    ):
        raise ValueError("actual AR conditional or marginal KL exceeded declared budget")
    result.pop("model_hash")
    result["model_hash"] = hash_json(result)
    validate_model(result)
    return result
