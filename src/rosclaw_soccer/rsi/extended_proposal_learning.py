"""One source-bound frozen-batch budget comparison; not a new rollout cycle."""

import re
from typing import Any

import numpy as np
from rosclaw.growth.correlated_residual_gradient import (
    conditional_means,
    terminal_crossfit_advantages,
)
from rosclaw.growth.event_credit_partitions import event_credit_partitions
from rosclaw.growth.extended_proposal_regression import (
    ExtendedProposalRegressionConfig,
    fit_extended_proposal_residual,
)
from rosclaw.growth.neural_context_advantages import neural_context_advantages
from rosclaw.growth.sample_weighting import balanced_partition_weights

from rosclaw_soccer.rsi.extended_proposal_motor import (
    CompiledExtendedProposalMotor,
    make_model,
    make_preview,
)
from rosclaw_soccer.rsi.proposal_decoder_selection import select_proposal_decoder
from rosclaw_soccer.rsi.proposal_memory_motor import make_preview as parent_preview
from rosclaw_soccer.rsi.proposal_memory_motor import validate_model as validate_parent
from rosclaw_soccer.sim.contracts import hash_json


def fit_extended_update(
    parent: dict[str, Any],
    arrays: Any,
    *,
    batch_hash: str,
    behavior_model_hash: str,
    critic_evidence_hash: str,
    maximum_inner_steps: int,
    trajectory_context_ids: Any = None,
    neural_critic_fit_results: Any = None,
    measured_event_frames: Any = None,
) -> dict[str, Any]:
    """Original physical batch/critic must be authenticated by caller separately.

    Immediate behavior identity, original likelihood, caps, gate and unchanged
    offline weighting are mandatory. Source contracts are checked by parent
    validation. No generation receipt or executed episode is relabelled.
    """
    validate_parent(parent)
    if behavior_model_hash != parent["model_hash"]:
        raise ValueError("exact immediate behavior model required")
    if any(
        not isinstance(v, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", v)
        for v in (batch_hash, critic_evidence_hash)
    ):
        raise ValueError("complete externally authenticated batch and critic identities required")
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
        not 1080 <= n <= 200000
        or n % 270
        or x.shape != (n, 134)
        or action.shape != (n, 12)
        or any(v.shape != (n,) for v in (phase, old_logp, returns, std, groups))
        or phase.dtype.kind not in "iu"
        or groups.dtype.kind not in "iu"
        or set(phase.tolist()) != {0, 1, 2}
        or any(v.dtype.kind not in "fiu" for v in (x, action, old_logp, returns, std))
        or not np.array_equal(groups, np.repeat(np.arange(n // 270), 270))
        or not all(np.isfinite(v).all() for v in (x, action, old_logp, returns, std))
        or any(not np.all(std[groups == g] == std[groups == g][0]) for g in range(n // 270))
    ):
        raise ValueError("complete ordered original behavior trajectories required")
    decoder = select_proposal_decoder(parent_preview(parent), implementation="bounded_snapshot")
    phi = np.stack([decoder.features(v) for v in x])
    context = np.column_stack((phi[:, :134], phase))
    current = np.stack([decoder.raw_mean(v, int(p)) for v, p in zip(x, phase, strict=True)])
    baseline = np.stack(
        [decoder._parent.raw_mean(v, int(p)) for v, p in zip(x, phase, strict=True)]
    )
    profile = parent["loss_weighting_profile"]
    weights = None
    if profile == "equal-first-contact-lead-mass":
        events = np.asarray(measured_event_frames)
        if events.shape != (n // 270,) or events.dtype.kind not in "iu":
            raise ValueError("complete independently bound original event labels required")
        partitions = event_credit_partitions(
            groups, np.tile(np.arange(30, 300), n // 270), events, lead_frames=8, after_frames=20
        )
        weights = balanced_partition_weights(partitions)
    else:
        if measured_event_frames is not None:
            raise ValueError("event weighting cannot alter the original objective")
        if profile == "equal-contact-phase-mass":
            weights = balanced_partition_weights(phase)
    if parent.get("critic_profile", "whole-rollout") == "whole-context-neural":
        prepared = neural_context_advantages(
            phi,
            phase,
            groups,
            returns,
            trajectory_context_ids=trajectory_context_ids,
            fold_fit_results=neural_critic_fit_results,
        )
    else:
        if (
            parent.get("critic_profile", "whole-rollout") != "whole-rollout"
            or trajectory_context_ids is not None
            or neural_critic_fit_results is not None
        ):
            raise ValueError("explicit supported original critic objective required")
        prepared = terminal_crossfit_advantages(phi, phase, groups, returns)
    numeric = fit_extended_proposal_residual(
        layers=[
            (np.asarray(v["weight"]), np.asarray(v["bias"])) for v in parent["residual_layers"]
        ],
        context=context,
        baseline=baseline,
        gates=decoder._guard.gates(context),
        actions=action,
        marginal_std=std,
        first=np.arange(n) % 270 == 0,
        advantages=prepared["advantages"],
        old_log_probability=old_logp,
        config=ExtendedProposalRegressionConfig(
            steps=maximum_inner_steps,
            maximum_mean_kl=parent["maximum_mean_kl"],
            compute_device=parent.get("optimizer_compute_device", "cpu"),
            likelihood_profile=parent.get("optimizer_likelihood_profile", "marginal"),
        ),
        sample_weights=weights,
    )
    layers = numeric.pop("layers")
    numeric.update(
        physical_batch_hash=batch_hash,
        critic_evidence_hash=critic_evidence_hash,
        learner_parent_hash=parent["model_hash"],
        behavior_model_hash=behavior_model_hash,
        physical_rollout_count=n // 270,
        frame_sample_count=n,
        context_is_actor_observation=False,
        future_event_is_actor_observation=False,
        offline_training_labels_hash=hash_json(
            dict(
                context_ids=None
                if trajectory_context_ids is None
                else np.asarray(trajectory_context_ids).tolist(),
                event_frames=None
                if measured_event_frames is None
                else np.asarray(measured_event_frames).tolist(),
                critic_model_hashes=prepared.get("critic_model_hashes"),
            )
        ),
        loss_weighting_profile=profile,
        critic_profile=parent.get("critic_profile", "whole-rollout"),
    )
    model = make_model(
        parent,
        maximum_inner_steps=maximum_inner_steps,
        residual_layers=layers,
        learning_receipt=numeric,
    )
    actual = CompiledExtendedProposalMotor(make_preview(model))
    final = np.stack([actual.raw_mean(v, int(p)) for v, p in zip(x, phase, strict=True)])
    first = np.arange(n) % 270 == 0
    innovation = std * np.where(first, 1.0, np.sqrt(1 - 0.9**2))
    numeric["exact_mean_conditional_kl"] = float(
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
    numeric["exact_mean_marginal_kl"] = float(
        np.mean(np.sum((final - current) ** 2 / (2 * std[:, None] ** 2), axis=1))
    )
    return make_model(
        parent,
        maximum_inner_steps=maximum_inner_steps,
        residual_layers=layers,
        learning_receipt=numeric,
    )
