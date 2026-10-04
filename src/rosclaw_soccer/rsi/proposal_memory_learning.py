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
from rosclaw.growth.event_credit_partitions import event_credit_partitions
from rosclaw.growth.proposal_advantage_regression import fit_proposal_advantage_residual
from rosclaw.growth.sample_weighting import balanced_partition_weights

from rosclaw_soccer.rsi.domain_memory_protection import protection_identity
from rosclaw_soccer.rsi.proposal_decoder_selection import (
    compilation_contract,
    select_proposal_decoder,
)
from rosclaw_soccer.rsi.proposal_memory_motor import (
    critic_kind,
    make_preview,
    validate_model,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def fit_update(
    model: dict[str, Any],
    arrays: Any,
    *,
    batch_hash: str,
    measured_event_frames: Any = None,
    event_evidence_hash: str | None = None,
    trajectory_context_ids: Any = None,
    context_evidence_hash: str | None = None,
    neural_critic_fit_results: Any = None,
    numeric_implementation: str = "reference",
) -> dict[str, Any]:
    numeric_contract = compilation_contract(numeric_implementation)
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
    profile = model["loss_weighting_profile"]
    partitions = None
    events = None
    if profile == "equal-first-contact-lead-mass":
        events = np.asarray(measured_event_frames)
        if (
            events.shape != (n // 270,)
            or events.dtype.kind not in "iu"
            or not isinstance(event_evidence_hash, str)
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", event_evidence_hash)
        ):
            raise ValueError("complete independently bound offline event labels required")
        partitions = event_credit_partitions(
            groups,
            np.tile(np.arange(30, 300), n // 270),
            events,
            lead_frames=8,
            after_frames=20,
        )
        weights = balanced_partition_weights(partitions)
    else:
        if measured_event_frames is not None or event_evidence_hash is not None:
            raise ValueError(
                "event labels cannot silently alter the declared phase/uniform objective"
            )
        weights = (
            balanced_partition_weights(phase) if profile == "equal-contact-phase-mass" else None
        )
    decoder = select_proposal_decoder(make_preview(model), implementation=numeric_implementation)
    current = np.stack([decoder.raw_mean(v, int(p)) for v, p in zip(x, phase, strict=True)])
    phi = np.stack([decoder.features(v) for v in x])
    context = np.column_stack((phi[:, :134], phase))
    context_ids = None
    neural_profile = model.get("critic_profile") == "whole-context-neural"
    if model.get("critic_profile", "whole-rollout") in ("whole-context", "whole-context-neural"):
        from rosclaw.growth.context_crossfit import context_crossfit_advantages

        if not isinstance(context_evidence_hash, str) or not re.fullmatch(
            r"sha256:[0-9a-f]{64}", context_evidence_hash
        ):
            raise ValueError("independently bound complete context labels required")
        context_ids = np.asarray(trajectory_context_ids)
        if neural_profile:
            from rosclaw.growth.neural_context_advantages import neural_context_advantages

            prepared = neural_context_advantages(
                phi,
                phase,
                groups,
                returns,
                trajectory_context_ids=context_ids,
                fold_fit_results=neural_critic_fit_results,
            )
        else:
            if neural_critic_fit_results is not None:
                raise ValueError("neural fit cannot silently alter the linear critic profile")
            prepared = context_crossfit_advantages(
                phi, phase, groups, returns, trajectory_context_ids=context_ids
            )
    else:
        if (
            trajectory_context_ids is not None
            or context_evidence_hash is not None
            or neural_critic_fit_results is not None
        ):
            raise ValueError("context labels cannot silently alter the declared critic objective")
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
            maximum_mean_kl=model["maximum_mean_kl"],
            compute_device=model.get("optimizer_compute_device", "cpu"),
            likelihood_profile=model.get("optimizer_likelihood_profile", "marginal"),
        ),
        sample_weights=weights,
    )
    result = {k: copy.deepcopy(v) for k, v in model.items() if k != "model_hash"}
    result["residual_layers"] = numeric.pop("layers")
    result["generation"] = 1
    initial = model["initial_actor"]
    protected_hash, protected_rows, protected_contexts = protection_identity(model)
    receipt = dict(
        **numeric,
        physical_batch_hash=batch_hash,
        learner_parent_hash=model["model_hash"],
        behavior_model_hash=initial["baseline"]["base_model"]["model_hash"],
        optimizer_source_hash=hash_bytes(Path(regression_module.__file__).read_bytes()),
        adapter_source_hash=hash_bytes(Path(__file__).read_bytes()),
        physical_rollout_count=n // 270,
        frame_sample_count=n,
        protected_memory_hash=protected_hash,
        protected_memory_rows=protected_rows,
        protected_anchor_contexts=protected_contexts,
        critic_kind=critic_kind(model.get("critic_profile", "whole-rollout")),
        critic_crossfit_folds=4,
        critic_target_mean=prepared["target_mean"],
        critic_target_scale=prepared["target_scale"],
        loss_weighting_profile=model["loss_weighting_profile"],
    )
    if context_ids is not None:
        receipt.update(
            core_context_source_hash=model["core_context_source_hash"],
            context_labels_hash=hash_json(
                {
                    "trajectory_context_ids": context_ids.tolist(),
                    "context_evidence_hash": context_evidence_hash,
                }
            ),
            context_evidence_hash=context_evidence_hash,
            independent_critic_contexts=len(np.unique(context_ids)),
            overlapping_context_count=0,
            context_is_actor_observation=False,
            critic_readout_unit="raw_terminal_return",
        )
    if numeric_contract is not None:
        receipt["numeric_preparation"] = numeric_contract
    if model["loss_weighting_profile"] == "equal-contact-phase-mass":
        receipt["phase_frame_counts"] = [int(np.sum(phase == p)) for p in range(3)]
    if partitions is not None and events is not None:
        receipt.update(
            event_partition_frame_counts=[int(np.sum(partitions == p)) for p in range(5)],
            credit_lead_frames=8,
            credit_after_frames=20,
            future_event_is_actor_observation=False,
            future_event_used_only_as_offline_label=True,
            core_event_source_hash=model["core_event_source_hash"],
            event_label_hash=hash_json(
                {
                    "measured_event_frames": events.tolist(),
                    "event_evidence_hash": event_evidence_hash,
                }
            ),
            event_evidence_hash=event_evidence_hash,
        )
    result["learning_receipt"] = receipt
    if neural_profile:
        result["neural_critic_fit_results"] = copy.deepcopy(neural_critic_fit_results)
        receipt.update(
            critic_model_hashes=prepared["critic_model_hashes"],
            core_prediction_source_hash=model["core_prediction_source_hash"],
            critic_readout_role="LINEAR_DIAGNOSTIC_ONLY_NEURAL_USED_FOR_ADVANTAGES",
        )
        result["critic_readout"] = prepared["linear_diagnostic_critic_readout"].tolist()
    else:
        result["critic_readout"] = prepared["critic_readout"].tolist()
    result["model_hash"] = hash_json(result)
    candidate = select_proposal_decoder(make_preview(result), implementation=numeric_implementation)
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
