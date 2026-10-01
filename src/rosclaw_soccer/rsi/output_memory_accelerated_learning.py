"""Paired learning-rate experiment; historical PPO artifacts stay unchanged.

Same causal actor, critic, objective, 160-step ceiling, residual cap and KL
budget as the historical learner. Only the optimizer learning rate is varied.
No physical gain or activation follows from decreasing the training objective.
"""

import copy
import re
from pathlib import Path
from typing import Any

import numpy as np
from rosclaw.growth.anchor_kernel import AnchorKernelGuard

from rosclaw_soccer.rsi import bounded_residual_policy_gradient as optimizer_module
from rosclaw_soccer.rsi.kernel_guarded_step_network import latents
from rosclaw_soccer.rsi.output_memory_step_motor import (
    CompiledOutputMemoryMotor,
    make_preview,
    validate_model,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def fit_update(
    model: dict[str, Any], arrays: Any, *, batch_hash: str, learning_rate: float = 4e-4
) -> dict[str, Any]:
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
        or type(learning_rate) not in (float, int)
        or not np.isfinite(learning_rate)
        or not 1e-5 <= learning_rate <= 1e-3
    ):
        raise ValueError("complete finite bounded current-parent physical batch required")
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
    hidden = context
    for layer in model["residual_layers"]:
        hidden = np.tanh(hidden @ np.asarray(layer["weight"]).T + np.asarray(layer["bias"]))
    guard = AnchorKernelGuard(
        model["output_memory"]["observations"], bandwidth=model["output_memory"]["bandwidth"]
    )
    gates = guard.gates(context)
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
    optimized = optimizer_module.optimize_residual(
        model["residual_layers"],
        context,
        baseline,
        gates,
        action,
        std,
        advantage,
        learning_rate=learning_rate,
    )
    result = {k: copy.deepcopy(v) for k, v in model.items() if k != "model_hash"}
    result["residual_layers"] = optimized["layers"]
    result["generation"] += 1
    receipt = dict(
        algorithm="OUTPUT_MEMORY_MLP_PPO_MC_TERMINAL",
        all_residual_layers_trainable=True,
        frozen_parent=True,
        distributional_retention_guaranteed=False,
        physical_batch_hash=batch_hash,
        learner_parent_hash=model["model_hash"],
        optimizer_source_hash=hash_bytes(Path(__file__).read_bytes()),
        numeric_optimizer_source_hash=hash_bytes(Path(optimizer_module.__file__).read_bytes()),
        learning_rate_experiment=True,
        completed_optimizer_steps=optimized["completed_optimizer_steps"],
        full_batch_loss_history=optimized["full_batch_loss_history"],
        backtracking_reductions=optimized["backtracking_reductions"],
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
        learning_rate=float(learning_rate),
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
