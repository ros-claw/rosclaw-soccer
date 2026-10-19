"""Small offline SIM action-outcome model, not an actor or qualified teacher.

Learns measured success/safety of thirteen interventions from121 current
features. Future outcomes are labels ONLY. No policy execution, action
selection, online RL or hardware authority is provided by this module.
"""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.postcontact_learning_observation import WIDTH
from rosclaw_soccer.sim.contracts import hash_json

SCHEMA = "soccer.rsi.postcontact_outcome_model.v1"
HIDDEN = 32
ACTIONS = 13
EPOCHS = 500
SEEDS = (1019, 2027, 4093)


def require_observations(features: np.ndarray) -> None:
    if (
        type(features) is not np.ndarray
        or features.dtype != np.float32
        or features.ndim != 2
        or features.shape[1] != WIDTH
        or not 1 <= len(features) <= 4096
        or not np.isfinite(features).all()
        or np.max(abs(features)) > 1e6
    ):
        raise ValueError("bounded complete float32 current-state feature matrix required")


def weights(report: dict[str, Any]) -> dict[str, np.ndarray]:
    if (
        report.get("schema") != SCHEMA
        or report.get("report_hash")
        != hash_json({k: v for k, v in report.items() if k != "report_hash"})
        or report.get("mode") != "SIM_ONLY"
        or report.get("model_role") != "diagnostic_action_outcome_predictor_not_actor"
        or any(
            report.get(k) is not False
            for k in (
                "teacher_qualified",
                "motor_policy_activation_authorized",
                "online_rl_claimed",
                "hardware_authorized",
                "promotion_authorized",
            )
        )
    ):
        raise ValueError("sealed SIM-only diagnostic model without motor authority required")
    values = {}
    for name, shape in dict(
        center=(WIDTH,),
        scale=(WIDTH,),
        w1=(WIDTH, HIDDEN),
        b1=(HIDDEN,),
        w2=(HIDDEN, 2 * ACTIONS),
        b2=(2 * ACTIONS,),
    ).items():
        a = np.asarray(report["parameters"][name], dtype=np.float32)
        if a.shape != shape or not np.isfinite(a).all() or np.max(abs(a)) > 1e6:
            raise ValueError("complete finite bounded diagnostic model parameters required")
        values[name] = a
    if np.any(values["scale"] < 0.01):
        raise ValueError("training-only normalization scale floor required")
    return values


def predict(report: dict[str, Any], features: np.ndarray) -> np.ndarray:
    """Return [N,13,2] probabilities: measured task pass, measured safe.

    There is intentionally no torque, action lease or Foundation interface.
    """
    require_observations(features)
    p = weights(report)
    x = np.clip((features - p["center"]) / p["scale"], -10.0, 10.0)
    hidden = np.tanh(x @ p["w1"] + p["b1"])
    logits = np.clip(hidden @ p["w2"] + p["b2"], -30.0, 30.0)
    result: np.ndarray = (1.0 / (1.0 + np.exp(-logits))).reshape(len(features), ACTIONS, 2)
    if not np.isfinite(result).all():
        raise ValueError("finite diagnostic action probabilities required")
    return result


def fit(
    features: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    *,
    seed: int,
    dataset_hash: str,
    training_context_indices: list[int],
) -> dict[str, Any]:
    """Bounded CPU full-batch BCE; only explicitly supplied training rows.

    Censored quality must be masked from the pass head. Safety remains a
    measured label. No imitation of a preferred action or dense-reward actor.
    """
    require_observations(features)
    shape = (len(features), ACTIONS, 2)
    if (
        type(seed) is not int
        or seed not in SEEDS
        or type(dataset_hash) is not str
        or not dataset_hash.startswith("sha256:")
        or type(training_context_indices) is not list
        or len(training_context_indices) != len(features)
        or len(set(training_context_indices)) != len(features)
        or any(type(i) is not int or not 0 <= i < 12 for i in training_context_indices)
        or type(labels) is not np.ndarray
        or labels.dtype != np.bool_
        or labels.shape != shape
        or type(mask) is not np.ndarray
        or mask.dtype != np.bool_
        or mask.shape != shape
        or not np.any(mask[:, :, 0])
        or not np.any(mask[:, :, 1])
        or not np.all(mask.any(axis=(1, 2)))
        or np.any(labels[:, :, 0] & ~labels[:, :, 1])
    ):
        raise ValueError("explicit local-only measured labels and quality/safety masks required")
    # Lazy optional dependency; never change global torch RNG/default device.
    import torch

    center = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), np.float32(0.01))
    x = torch.from_numpy(np.clip((features - center) / scale, -10.0, 10.0).copy())
    y = torch.from_numpy(labels.reshape(len(features), -1).astype(np.float32))
    m = torch.from_numpy(mask.reshape(len(features), -1).astype(np.float32))
    rng = np.random.default_rng(seed)
    w1 = torch.tensor(
        rng.normal(0, 1 / np.sqrt(WIDTH), (WIDTH, HIDDEN)),
        dtype=torch.float32,
        device="cpu",
        requires_grad=True,
    )
    b1 = torch.zeros(HIDDEN, dtype=torch.float32, device="cpu", requires_grad=True)
    w2 = torch.tensor(
        rng.normal(0, 1 / np.sqrt(HIDDEN), (HIDDEN, 2 * ACTIONS)),
        dtype=torch.float32,
        device="cpu",
        requires_grad=True,
    )
    b2 = torch.zeros(2 * ACTIONS, dtype=torch.float32, device="cpu", requires_grad=True)
    parameters = [w1, b1, w2, b2]
    optimizer = torch.optim.Adam(parameters, lr=0.003)
    losses = []
    for _ in range(EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        logits = torch.tanh(x @ w1 + b1) @ w2 + b2
        loss = (
            torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none") * m
        ).sum() / m.sum()
        loss = loss + 1e-4 * (w1.square().mean() + w2.square().mean())
        if not torch.isfinite(loss):
            raise ValueError("nonfinite outcome-model loss, no retry or activation")
        loss.backward()  # type: ignore[no-untyped-call]
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
            raise ValueError("nonfinite outcome-model gradient, no retry or activation")
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    report = dict(
        schema=SCHEMA,
        mode="SIM_ONLY",
        model_role="diagnostic_action_outcome_predictor_not_actor",
        seed=seed,
        dataset_hash=dataset_hash,
        training_context_indices=training_context_indices.copy(),
        independent_training_states=len(features),
        action_outcome_pairs=int(mask.any(axis=2).sum()),
        optimizer_updates=EPOCHS,
        training_loss_initial=losses[0],
        training_loss_final=losses[-1],
        parameters=dict(
            center=center.tolist(),
            scale=scale.tolist(),
            **{
                name: p.detach().numpy().tolist()
                for name, p in zip(("w1", "b1", "w2", "b2"), parameters, strict=True)
            },
        ),
        normalization_fitted_on_training_rows_only=True,
        teacher_qualified=False,
        motor_policy_activation_authorized=False,
        online_rl_claimed=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    report["report_hash"] = hash_json(report)
    weights(report)
    return report
