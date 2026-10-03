"""Actual-state regression adapter; still no rollout, activation or motion IO."""

import copy
import re
from pathlib import Path
from typing import Any

import numpy as np
import rosclaw.growth.proposal_advantage_regression as regression_module
from rosclaw.growth.correlated_residual_gradient import (
    conditional_means,
    terminal_crossfit_advantages,
)
from rosclaw.growth.proposal_advantage_regression import fit_proposal_advantage_residual
from rosclaw.growth.sample_weighting import balanced_partition_weights

from rosclaw_soccer.rsi.proposal_memory_motor import (
    CompiledProposalMemoryMotor,
    make_preview,
    validate_model,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def fit_update(model: dict[str, Any], arrays: Any, *, batch_hash: str) -> dict[str, Any]:
    validate_model(model)
    if model["generation"] != 0 or not re.fullmatch(r"sha256:[0-9a-f]{64}", batch_hash):
        raise ValueError("zero-addition regression initial model and sealed batch required")
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
        raise ValueError("complete ordered physical regression trajectories required")
    n = len(x)
    if (
        not 1080 <= n <= 200000
        or n % 270
        or x.shape != (n, 134)
        or action.shape != (n, 12)
        or any(v.shape != (n,) for v in (phase, old_logp, returns, std, groups))
        or phase.dtype.kind not in "iu"
        or groups.dtype.kind not in "iu"
        or set(phase.tolist()) != {0, 1, 2}
        or not np.array_equal(groups, np.repeat(np.arange(n // 270), 270))
        or not all(np.isfinite(v).all() for v in (x, action, old_logp, returns, std))
        or any(not np.all(std[groups == g] == std[groups == g][0]) for g in range(n // 270))
    ):
        raise ValueError("complete ordered physical regression trajectories required")
    decoder = CompiledProposalMemoryMotor(make_preview(model))
    current = np.stack([decoder.raw_mean(v, int(p)) for v, p in zip(x, phase, strict=True)])
    phi = np.stack([decoder.features(v) for v in x])
    context = np.column_stack((phi[:, :134], phase))
    prepared = terminal_crossfit_advantages(phi, phase, groups, returns)
    numeric = fit_proposal_advantage_residual(
        layers=[(np.asarray(v["weight"]), np.asarray(v["bias"])) for v in model["residual_layers"]],
        context=context,
        baseline=current,
        gates=decoder._guard.gates(context),
        actions=action,
        marginal_std=std,
        first=np.arange(n) % 270 == 0,
        advantages=prepared["advantages"],
        old_log_probability=old_logp,
        config=regression_module.ProposalAdvantageRegressionConfig(
            maximum_mean_kl=model["maximum_mean_kl"]
        ),
        sample_weights=(
            balanced_partition_weights(phase)
            if model["loss_weighting_profile"] == "equal-contact-phase-mass"
            else None
        ),
    )
    result = {k: copy.deepcopy(v) for k, v in model.items() if k != "model_hash"}
    result["residual_layers"] = numeric.pop("layers")
    result["generation"] = 1
    initial = model["initial_actor"]
    receipt = dict(
        **numeric,
        physical_batch_hash=batch_hash,
        learner_parent_hash=model["model_hash"],
        behavior_model_hash=initial["baseline"]["base_model"]["model_hash"],
        optimizer_source_hash=hash_bytes(Path(regression_module.__file__).read_bytes()),
        adapter_source_hash=hash_bytes(Path(__file__).read_bytes()),
        physical_rollout_count=n // 270,
        frame_sample_count=n,
        protected_memory_hash=initial["baseline"]["memory"]["memory_hash"],
        protected_memory_rows=len(initial["baseline"]["memory"]["observations"]),
        protected_anchor_contexts=len(initial["baseline"]["consolidation_manifest"]["records"]),
        critic_kind="WHOLE_TRAJECTORY_CROSSFIT_MC_NOT_TD_LAMBDA",
        critic_crossfit_folds=4,
        critic_target_mean=prepared["target_mean"],
        critic_target_scale=prepared["target_scale"],
        loss_weighting_profile=model["loss_weighting_profile"],
    )
    if model["loss_weighting_profile"] == "equal-contact-phase-mass":
        receipt["phase_frame_counts"] = [int(np.sum(phase == p)) for p in range(3)]
    result["learning_receipt"] = receipt
    result["critic_readout"] = prepared["critic_readout"].tolist()
    result["model_hash"] = hash_json(result)
    candidate = CompiledProposalMemoryMotor(make_preview(result))
    final = np.stack([candidate.raw_mean(v, int(p)) for v, p in zip(x, phase, strict=True)])
    first = np.arange(n) % 270 == 0
    innovation = std * np.where(first, 1, np.sqrt(1 - 0.9**2))
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
        np.mean(np.sum((final - current) ** 2 / (2 * std[:, None] ** 2), axis=1))
    )
    result.pop("model_hash")
    result["model_hash"] = hash_json(result)
    validate_model(result)
    return result
