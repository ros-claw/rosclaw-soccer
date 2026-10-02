"""Adapter from sealed physical motor batches to Core's generic AR optimizer."""

import copy
import re
from pathlib import Path
from typing import Any

import numpy as np
import rosclaw.growth.correlated_residual_gradient as gradient_module
from rosclaw.growth.correlated_residual_gradient import (
    ResidualGradientConfig,
    conditional_means,
    fit_correlated_residual,
    terminal_crossfit_advantages,
)

from rosclaw_soccer.rsi.current_memory_motor import (
    CompiledCurrentMemoryMotor,
    make_preview,
    validate_model,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def fit_update(model: dict[str, Any], arrays: Any, *, batch_hash: str) -> dict[str, Any]:
    validate_model(model)
    if model["generation"] != 0 or not re.fullmatch(r"sha256:[0-9a-f]{64}", batch_hash):
        raise ValueError("declared zero-addition initial current-parent learning batch required")
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
    if x.ndim != 2:
        raise ValueError("complete ordered current-parent physical batch required")
    n = len(x)
    if (
        not 1080 <= n <= 200000
        or n % 270
        or x.shape != (n, 134)
        or action.shape != (n, 12)
        or any(a.shape != (n,) for a in (phase, old_logp, returns, std, groups))
        or phase.dtype.kind not in "iu"
        or groups.dtype.kind not in "iu"
        or set(phase.tolist()) != {0, 1, 2}
        or not np.array_equal(groups, np.repeat(np.arange(n // 270), 270))
        or not all(np.isfinite(a).all() for a in (x, action, old_logp, returns, std))
        or any(not np.all(std[groups == g] == std[groups == g][0]) for g in range(n // 270))
    ):
        raise ValueError("complete ordered current-parent physical batch required")
    decoder = CompiledCurrentMemoryMotor(make_preview(model))
    current = np.stack([decoder.raw_mean(v, int(p)) for v, p in zip(x, phase, strict=True)])
    phi = np.stack([decoder.features(v) for v in x])
    context = np.column_stack((phi[:, :134], phase))
    gates = decoder._guard.gates(context)
    prepared = terminal_crossfit_advantages(phi, phase, groups, returns)
    layers = [(np.asarray(v["weight"]), np.asarray(v["bias"])) for v in model["residual_layers"]]
    numeric = fit_correlated_residual(
        layers=layers,
        context=context,
        baseline=current,
        gates=gates,
        actions=action,
        marginal_std=std,
        first=np.arange(n) % 270 == 0,
        advantages=prepared["advantages"],
        old_log_probability=old_logp,
        config=ResidualGradientConfig(
            residual_cap=model["raw_residual_cap"], learning_rate=model["learning_rate"]
        ),
    )
    result = {k: copy.deepcopy(v) for k, v in model.items() if k != "model_hash"}
    result["residual_layers"] = numeric.pop("layers")
    result["generation"] = 1
    receipt = dict(
        **numeric,
        physical_batch_hash=batch_hash,
        learner_parent_hash=model["model_hash"],
        behavior_model_hash=model["baseline"]["base_model"]["model_hash"],
        behavior_mean_equivalence="GLOBALLY_EXACT_ZERO_ADDITION_TO_SAME_FROZEN_NN",
        optimizer_source_hash=hash_bytes(Path(gradient_module.__file__).read_bytes()),
        adapter_source_hash=hash_bytes(Path(__file__).read_bytes()),
        physical_rollout_count=n // 270,
        frame_sample_count=n,
        protected_memory_hash=model["baseline"]["memory"]["memory_hash"],
        protected_memory_rows=len(model["baseline"]["memory"]["observations"]),
        protected_anchor_contexts=len(model["baseline"]["consolidation_manifest"]["records"]),
        gamma=1.0,
        gae_lambda=1.0,
        critic_crossfit_unit="whole_rollout",
        critic_crossfit_folds=4,
        critic_target_mean=prepared["target_mean"],
        critic_target_scale=prepared["target_scale"],
    )
    result["learning_receipt"] = receipt
    result["critic_readout"] = prepared["critic_readout"].tolist()
    result["model_hash"] = hash_json(result)
    candidate = CompiledCurrentMemoryMotor(make_preview(result))
    final = np.stack([candidate.raw_mean(v, int(p)) for v, p in zip(x, phase, strict=True)])
    first = np.arange(n) % 270 == 0
    innovation = std * np.where(first, 1.0, np.sqrt(1 - 0.9**2))
    receipt["exact_mean_conditional_kl"] = float(
        np.mean(
            np.sum(
                (
                    conditional_means(final, action, first, 0.9)
                    - conditional_means(current, action, first, 0.9)
                )
                ** 2
                / (2 * innovation[:, None] ** 2),
                axis=1,
            )
        )
    )
    receipt["exact_mean_marginal_kl"] = float(
        np.mean(
            np.sum(
                (final - current) ** 2 / (2 * std[:, None] ** 2),
                axis=1,
            )
        )
    )
    result.pop("model_hash")
    result["model_hash"] = hash_json(result)
    validate_model(result)
    return result
