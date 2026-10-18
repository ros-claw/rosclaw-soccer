"""Offline reference imitation over supplied numeric rows; no physical admission.

Reuses Core's causal sequence learner without changing its .2 output cap: the
target is expressed in .2/.35 units, then exported to the existing .35 decoder.
The caller must authenticate physical source files and teacher eligibility.
"""

import numpy as np
from rosclaw.growth.causal_residual_memory import CausalResidualMemory, initial_parameters
from rosclaw.growth.recurrent_residual_imitation import (
    RecurrentResidualImitationConfig,
    fit_recurrent_residual_imitation,
)

from rosclaw_soccer.rsi.reference_student_execution import make_model
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def prepare_numeric_batch(observations, targets, *, partitions, source_seeds, eligible):
    """Validate every supplied episode; fit statistics on train teachers only.

    Rows include failures, but zero-weight failures are not negative learning.
    Source IDs/partitions/eligibility never enter the actor's input vector.
    """
    x, y = np.asarray(observations), np.asarray(targets)
    n = len(partitions)
    if (
        not 4 <= n <= 32
        or x.shape != (n, 300, 135)
        or y.shape != (n, 300, 12)
        or x.dtype != np.float64
        or y.dtype != np.float64
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or np.max(np.abs(x)) > 1e6
        or np.max(np.abs(y)) > 0.35 + 1e-12
        or np.any(y[:, :30] != 0)
        or not np.array_equal(x[:, :, -1], np.tile((300 - np.arange(300)) / 300, (n, 1)))
        or len(source_seeds) != n
        or len(eligible) != n
        or any(type(v) is not int or not 0 <= v < 2**32 for v in source_seeds)
        or any(type(v) is not bool for v in eligible)
        or any(v not in ("train", "validation") for v in partitions)
    ):
        raise ValueError("complete finite aligned 300-frame reference episodes required")
    train = np.array([p == "train" and e for p, e in zip(partitions, eligible, strict=True)])
    validation = np.array(
        [p == "validation" and e for p, e in zip(partitions, eligible, strict=True)]
    )
    seed_partitions = {}
    for seed, partition in zip(source_seeds, partitions, strict=True):
        if seed_partitions.setdefault(seed, partition) != partition:
            raise ValueError("all perturbations and controllers of a source must share one split")
    if len(
        {s for s, selected in zip(source_seeds, train, strict=True) if selected}
    ) < 2 or not np.any(validation):
        raise ValueError("at least two eligible train sources and one validation source required")
    # Remaining-time units stay literal, as required by the execution decoder.
    mean = x[train].mean(axis=(0, 1))
    scale = np.maximum(x[train].std(axis=(0, 1)), 0.05)
    mean[-1], scale[-1] = 0, 1
    normalized = np.clip((x - mean) / scale, -8, 8)
    gates = np.ones((n, 300), dtype=np.float64)
    gates[:, :30] = 0
    # Label-dependent weighting is OFFLINE loss only, never an actor feature.
    weights = train[:, None] * (1.0 + 4.0 * np.any(np.abs(y) > 0.001, axis=2))
    weights[:, :30] = 0
    return dict(
        original_observations=x.copy(),
        normalized=normalized,
        mean=mean,
        scale=scale,
        gates=gates,
        weights=weights,
        train=train,
        validation=validation,
        targets=y.copy(),
        partitions=list(partitions),
        source_seeds=list(source_seeds),
        eligible=list(eligible),
    )


def numeric_errors(parameters, batch):
    """Teacher-state diagnostic, NOT student's own-action/body evaluation."""
    x, y = batch["normalized"], batch["targets"]
    rows = []
    for i in range(len(x)):
        memory = CausalResidualMemory(parameters)
        prediction = np.stack([memory.step(v, index=t) for t, v in enumerate(x[i])]) * 0.35
        prediction[:30] = 0
        rows.append(
            dict(
                partition=batch["partitions"][i],
                source_seed=batch["source_seeds"][i],
                eligible=batch["eligible"][i],
                mean_squared_reference_error_rad2=float(
                    np.mean((prediction[30:] - y[i, 30:]) ** 2)
                ),
                maximum_reference_error_rad=float(np.max(np.abs(prediction[30:] - y[i, 30:]))),
            )
        )
    return rows


def fit_numeric_reference_student(batch, *, world_hash, parent_model_hash, seed, steps=128):
    """Finite CPU numeric fit; eligibility remains a caller-supplied assertion.

    Validation states never enter the optimizer or normalization. No best-of
    validation checkpoint choice, online interaction, RL or policy activation.
    """
    # Re-validate and derive statistics instead of trusting a mutable batch mask.
    raw = batch.get("original_observations")
    if raw is None:
        raise ValueError("original whole observations required before fitting")
    verified = prepare_numeric_batch(
        raw,
        batch["targets"],
        partitions=batch["partitions"],
        source_seeds=batch["source_seeds"],
        eligible=batch["eligible"],
    )
    for key in ("mean", "scale", "normalized", "weights", "gates", "train", "validation"):
        if not np.array_equal(verified[key], batch[key]):
            raise ValueError("unchanged complete numeric preparation and split required")
    config = RecurrentResidualImitationConfig(
        steps=steps, batch_episodes=4, learning_rate=0.0004, residual_cap=0.2, seed=seed
    )
    config.validate()
    parameters = initial_parameters(135, 12, hidden_dimension=64, seed=seed)
    # Hard-subset the train sources, not merely zero validation weights.
    selected = batch["train"]
    x = batch["normalized"][selected]
    gates, weight = batch["gates"][selected], batch["weights"][selected]
    target = batch["targets"][selected] * (0.2 / 0.35)
    fit = fit_recurrent_residual_imitation(
        parameters=parameters,
        context=x,
        baseline=np.zeros_like(target),
        gates=gates,
        targets=target,
        training_weights=weight,
        config=config,
    )
    model = make_model(
        fit["parameters"],
        mean=batch["mean"],
        scale=batch["scale"],
        world_hash=world_hash,
        parent_model_hash=parent_model_hash,
    )
    report = dict(
        schema="soccer.rsi.reference_numeric_imitation_fit.v1",
        numeric_model=model,
        initial_parameters_hash=hash_json(parameters),
        learning_receipt=fit,
        normalization_hash=hash_json(
            dict(mean=batch["mean"].tolist(), scale=batch["scale"].tolist())
        ),
        whole_observations_hash=hash_bytes(raw.tobytes()),
        whole_reference_labels_hash=hash_bytes(batch["targets"].tobytes()),
        train_source_seeds=sorted(
            {s for s, t in zip(batch["source_seeds"], selected, strict=True) if t}
        ),
        validation_source_seeds=sorted(
            {
                s
                for s, p in zip(batch["source_seeds"], batch["partitions"], strict=True)
                if p == "validation"
            }
        ),
        input_episode_count=len(raw),
        optimized_episode_count=int(selected.sum()),
        original_ineligible_episode_count=int(np.count_nonzero(np.logical_not(batch["eligible"]))),
        initial_same_state_diagnostics=numeric_errors(parameters, batch),
        final_same_state_diagnostics=numeric_errors(fit["parameters"], batch),
        validation_excluded_from_gradient_and_normalization=True,
        normalization_fit_only_on_eligible_train_episodes=True,
        previous_reference_inputs_are_teacher_forced=True,
        offline_reference_loss_not_projected_execution_loss=True,
        failed_episodes_are_retained_not_negative_gradient_examples=True,
        target_units_ratio=0.2 / 0.35,
        original_core_learner_cap=0.2,
        original_execution_decoder_cap_rad=0.35,
        physical_source_authentication_performed_here=False,
        teacher_eligibility_authenticated_here=False,
        training_provenance_authenticated=False,
        online_rl_claimed=False,
        own_body_student_evaluated=False,
        teacher_qualified=False,
        dataset_admission_authorized=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    report["report_hash"] = hash_json(report)
    return report
