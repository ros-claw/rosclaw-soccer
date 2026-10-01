"""Actual on-policy PPO for the plastic residual MLP above a frozen later parent.

The caller must supply independently reconstructed physical rollouts. This
optimizer checks exact current behavior density and freezes all parent/memory
data. A learned candidate still requires independent physical qualification.
"""

import copy
import re
from pathlib import Path
from typing import Any

import numpy as np
from rosclaw.growth.anchor_kernel import AnchorKernelGuard

from rosclaw_soccer.rsi.kernel_guarded_step_network import latents
from rosclaw_soccer.rsi.output_memory_step_motor import (
    CompiledOutputMemoryMotor,
    make_preview,
    validate_model,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def fit_update(model: dict[str, Any], arrays: Any, *, batch_hash: str) -> dict[str, Any]:
    import torch

    validate_model(model)
    x, phase, action, old_logp, returns, std, groups = [
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
        or not 100 <= n <= 200000
        or x.shape != (n, 134)
        or action.shape != (n, 12)
        or any(a.shape != (n,) for a in (phase, old_logp, returns, std, groups))
        or not all(np.isfinite(a).all() for a in (x, phase, action, old_logp, returns, std, groups))
        or phase.dtype.kind not in "iu"
        or groups.dtype.kind not in "iu"
        or set(phase.tolist()) != {0, 1, 2}
        or np.any(groups < 0)
        or np.any((std < 0.01) | (std > 0.15))
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", batch_hash)
    ):
        raise ValueError("complete finite current-parent physical batch required")
    if any(not np.all(returns[groups == g] == returns[groups == g][0]) for g in set(groups)):
        raise ValueError("one actual terminal return per whole trajectory required")
    decoder = CompiledOutputMemoryMotor(make_preview(model))
    current = np.stack([decoder.raw_mean(v, int(p)) for v, p in zip(x, phase, strict=True)])
    density = np.sum(
        -0.5 * ((action - current) / std[:, None]) ** 2
        - np.log(std[:, None])
        - 0.5 * np.log(2 * np.pi),
        axis=1,
    )
    if not np.allclose(density, old_logp, atol=1e-8, rtol=0):
        raise ValueError("batch density is not the actual current output-memory actor")
    phi = latents(model["frozen_parent"], x)
    context = np.column_stack((phi[:, :134], phase))
    frozen_layers = [
        (np.asarray(v["weight"]), np.asarray(v["bias"])) for v in model["residual_layers"]
    ]
    hidden = context
    for weight, bias in frozen_layers:
        hidden = np.tanh(hidden @ weight.T + bias)
    guard = AnchorKernelGuard(
        model["output_memory"]["observations"], bandwidth=model["output_memory"]["bandwidth"]
    )
    gates = guard.gates(context)
    # Remove the current learned residual to obtain the unchanged memory-blended
    # parent baseline; adding it back reproduces current inference within roundoff.
    baseline = current - 0.05 * gates[:, None] * hidden
    target_mean, target_scale = float(returns.mean()), max(float(returns.std()), 1.0)
    targets = (returns - target_mean) / target_scale
    value = np.zeros(n)
    critic = np.zeros((3, 512))

    def regression(ids: Any) -> Any:
        design = phi[ids]
        if len(design) < 50:
            raise ValueError("whole-trajectory cross-fit support too small")
        return np.linalg.solve(design.T @ design + 0.01 * np.eye(512), design.T @ targets[ids])

    for p in range(3):
        for fold in range(4):
            test = (phase == p) & (groups % 4 == fold)
            value[test] = phi[test] @ regression((phase == p) & (groups % 4 != fold))
        critic[p] = regression(phase == p)
    advantage = targets - value
    advantage = (advantage - advantage.mean()) / max(float(advantage.std()), 1e-6)
    torch.set_num_threads(4)
    torch.manual_seed(202610343)
    torch.use_deterministic_algorithms(True)
    weights = [torch.nn.Parameter(torch.tensor(w, dtype=torch.float32)) for w, _ in frozen_layers]
    biases = [torch.nn.Parameter(torch.tensor(b, dtype=torch.float32)) for _, b in frozen_layers]
    parameters = weights + biases
    optimizer = torch.optim.Adam(parameters, lr=1e-4)
    inp = torch.tensor(context, dtype=torch.float32)
    base = torch.tensor(baseline, dtype=torch.float32)
    gate = torch.tensor(gates[:, None], dtype=torch.float32)
    noise = torch.tensor(std[:, None], dtype=torch.float32)
    desired = torch.tensor(action, dtype=torch.float32)
    importance = torch.tensor(advantage, dtype=torch.float32)

    def mean() -> Any:
        hidden = inp
        for w, b in zip(weights, biases, strict=True):
            hidden = torch.tanh(hidden @ w.T + b)
        return base + 0.05 * gate * hidden

    with torch.no_grad():
        original = mean().clone()
        logp0 = (
            -0.5 * ((desired - original) / noise).square() - noise.log() - 0.5 * np.log(2 * np.pi)
        ).sum(dim=1)

    def objective() -> tuple[Any, Any]:
        mu = mean()
        logp = (
            -0.5 * ((desired - mu) / noise).square() - noise.log() - 0.5 * np.log(2 * np.pi)
        ).sum(dim=1)
        ratio = torch.exp(logp - logp0)
        kl = ((mu - original).square() / (2 * noise.square())).sum(dim=1).mean()
        loss = -torch.minimum(ratio * importance, ratio.clamp(0.8, 1.2) * importance).mean()
        return loss + 10 * kl, kl

    history = [float(objective()[0].detach())]
    reductions = 0
    for _ in range(160):
        optimizer.zero_grad()
        loss, _ = objective()
        if not torch.isfinite(loss):
            raise ValueError("nonfinite actual motor policy gradient")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        previous = [p.detach().clone() for p in parameters]
        optimizer.step()
        directions = [p.detach().clone() - old for p, old in zip(parameters, previous, strict=True)]
        accepted = False
        for reduction in range(13):
            with torch.no_grad():
                for parameter, old, d in zip(parameters, previous, directions, strict=True):
                    parameter.copy_(old + (0.5**reduction) * d)
                trial, kl = objective()
                if (
                    torch.isfinite(trial)
                    and float(kl) <= 0.0049
                    and float(trial) < history[-1] - 1e-9
                ):
                    history.append(float(trial))
                    reductions += reduction
                    accepted = True
                    break
        if not accepted:
            with torch.no_grad():
                for parameter, old in zip(parameters, previous, strict=True):
                    parameter.copy_(old)
            break
    if len(history) == 1:
        raise ValueError("no actual residual network learning")
    result = {k: copy.deepcopy(v) for k, v in model.items() if k != "model_hash"}
    result["residual_layers"] = [
        dict(
            weight=w.detach().numpy().astype(float).tolist(),
            bias=b.detach().numpy().astype(float).tolist(),
        )
        for w, b in zip(weights, biases, strict=True)
    ]
    result["generation"] += 1
    # Provisional receipt permits the exact frozen NumPy inference audit below.
    receipt = dict(
        algorithm="OUTPUT_MEMORY_MLP_PPO_MC_TERMINAL",
        all_residual_layers_trainable=True,
        frozen_parent=True,
        distributional_retention_guaranteed=False,
        physical_batch_hash=batch_hash,
        learner_parent_hash=model["model_hash"],
        optimizer_source_hash=hash_bytes(Path(__file__).read_bytes()),
        completed_optimizer_steps=len(history) - 1,
        full_batch_loss_history=history,
        backtracking_reductions=reductions,
        exact_mean_latent_kl=0.0,
        critic_crossfit_unit="whole_rollout",
        critic_crossfit_folds=4,
        critic_ridge=0.01,
        critic_target_mean=target_mean,
        critic_target_scale=target_scale,
        physical_rollout_count=len(set(groups.tolist())),
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
    candidate = CompiledOutputMemoryMotor(make_preview(result))
    final = np.stack([candidate.raw_mean(v, int(p)) for v, p in zip(x, phase, strict=True)])
    kl = float(np.mean(np.sum((final - current) ** 2 / (2 * std[:, None] ** 2), axis=1)))
    if not np.isfinite(kl) or kl > 0.005:
        raise ValueError("actual frozen inference exceeded cumulative KL budget")
    receipt["exact_mean_latent_kl"] = kl
    result.pop("model_hash")
    result["model_hash"] = hash_json(result)
    validate_model(result)
    return result
