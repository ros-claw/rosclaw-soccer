"""Capacity-expanding contextual neural residual, with a frozen predecessor.

Frozen 13-input nonlinear features and a learned protected 37-output head.
Not task-ID routing, torque PPO, full Progressive Neural Networks, or authority.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import rosclaw.growth.advantage_weights as weight_module
import rosclaw.growth.anchor_plane as plane_module
from rosclaw.growth.advantage_weights import statewise_advantage_weights
from rosclaw.growth.anchor_plane import AnchorProtectionPlane, fit_protected_readout

from rosclaw_soccer.rsi.bootstrap_motor_execution import context_at30
from rosclaw_soccer.rsi.contact_motor_phase import make_policy
from rosclaw_soccer.rsi.online_motor_actor_critic import (
    normalized_actor as parent_actor,
)
from rosclaw_soccer.rsi.online_motor_actor_critic import (
    terminal_return,
)
from rosclaw_soccer.rsi.online_motor_actor_critic import (
    validate_model as validate_parent,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

SCHEMA = "soccer.rsi.progressive_contextual_actor_critic.v312"
WIDTH = 64
DIMENSION = 13 + WIDTH + 1


def latent(model: dict[str, Any], observation: Any) -> np.ndarray[Any, Any]:
    values = np.asarray(observation, dtype=np.float64)
    if values.shape != (13,) or not np.isfinite(values).all():
        raise ValueError("thirteen causal finite features required, not course identities")
    encoder = model["encoder"]
    mean, scale = np.asarray(encoder["mean"]), np.asarray(encoder["scale"])
    weight, bias = np.asarray(encoder["weight"]), np.asarray(encoder["bias"])
    if (
        mean.shape != (13,)
        or scale.shape != (13,)
        or np.any(scale < 0.05)
        or weight.shape != (WIDTH, 13)
        or bias.shape != (WIDTH,)
        or not all(np.isfinite(v).all() for v in (mean, scale, weight, bias))
    ):
        raise ValueError("bounded frozen encoder contract required")
    # Broader consumed-state scale, independent of narrow predecessor scale.
    normalized = np.clip((values - mean) / scale, -8.0, 8.0)
    return np.concatenate((normalized, np.tanh(weight @ normalized + bias), [1.0]))


def validate_model(model: dict[str, Any]) -> AnchorProtectionPlane:
    if (
        model.get("schema") != SCHEMA
        or model.get("activation_ceiling") != "SIM_ONLY"
        or model.get("learning_kind") != "capacity_expanded_protected_contextual_awr_inspired"
        or any(
            model.get(k) is not False
            for k in (
                "runtime_execution_authorized",
                "promotion_authorized",
                "hardware_authorized",
                "fresh_holdout_open_authorized",
            )
        )
        or model.get("source_hash") != hash_bytes(Path(__file__).read_bytes())
        or model.get("core_plane_source_hash")
        != hash_bytes(Path(plane_module.__file__).read_bytes())
        or model.get("weighting_source_hash")
        != hash_bytes(Path(weight_module.__file__).read_bytes())
        or model.get("model_hash")
        != hash_json({k: v for k, v in model.items() if k != "model_hash"})
    ):
        raise ValueError("unsealed progressive contextual model")
    validate_parent(model["frozen_parent"])
    if model.get("encoder", {}).get("frozen") is not True:
        raise ValueError("retention requires a frozen encoder")
    protected = model["protected_features"]
    if not 9 <= len(protected) <= 64:
        raise ValueError("explicit bounded retention observations required")
    plane = AnchorProtectionPlane.from_dict(model["anchor_plane"])
    rebuilt = AnchorProtectionPlane(np.stack([latent(model, f) for f in protected]))
    if plane.to_dict() != rebuilt.to_dict():
        raise ValueError("encoder changed its protection plane")
    if plane.dimension != DIMENSION or plane.rank >= DIMENSION:
        raise ValueError("nonempty plastic capacity required")
    for key, shape in (("plastic_readout", (37, DIMENSION)), ("value_readout", (DIMENSION,))):
        value = np.asarray(model[key])
        if value.shape != shape or not np.isfinite(value).all() or np.max(np.abs(value)) > 1e4:
            raise ValueError("finite bounded aligned learned head required")
    if type(model.get("generation")) is not int or model["generation"] < 0:
        raise ValueError("nonnegative generation required")
    return plane


def _seal(content: dict[str, Any]) -> dict[str, Any]:
    result = {k: v for k, v in content.items() if k != "model_hash"}
    result["model_hash"] = hash_json(result)
    validate_model(result)
    return result


def make_model(
    parent: dict[str, Any],
    contexts: list[list[float]],
    protected: list[list[float]],
    *,
    learning_report_hash: str,
    fresh_protocol_hash: str,
) -> dict[str, Any]:
    validate_parent(parent)
    values = np.asarray(contexts, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1] != 13
        or not 12 <= len(values) <= 4096
        or not np.isfinite(values).all()
    ):
        raise ValueError("complete consumed physical contexts required")
    rng = np.random.default_rng(20261001312)
    model = dict(
        schema=SCHEMA,
        activation_ceiling="SIM_ONLY",
        learning_kind="capacity_expanded_protected_contextual_awr_inspired",
        frozen_parent=parent,
        encoder=dict(
            mean=values.mean(axis=0).tolist(),
            scale=np.maximum(values.std(axis=0), 0.05).tolist(),
            weight=(rng.normal(size=(WIDTH, 13)) / np.sqrt(13)).tolist(),
            bias=rng.uniform(-1, 1, size=WIDTH).tolist(),
            frozen=True,
        ),
        protected_features=protected,
        plastic_readout=np.zeros((37, DIMENSION)).tolist(),
        value_readout=np.zeros(DIMENSION).tolist(),
        generation=0,
        parent_model_hash=parent["model_hash"],
        learning_report_hash=learning_report_hash,
        fresh_quarantine_protocol_hash=fresh_protocol_hash,
        source_hash=hash_bytes(Path(__file__).read_bytes()),
        core_plane_source_hash=hash_bytes(Path(plane_module.__file__).read_bytes()),
        weighting_source_hash=hash_bytes(Path(weight_module.__file__).read_bytes()),
        runtime_execution_authorized=False,
        promotion_authorized=False,
        hardware_authorized=False,
        fresh_holdout_open_authorized=False,
    )
    model["anchor_plane"] = AnchorProtectionPlane(
        np.stack([latent(model, f) for f in protected])
    ).to_dict()
    return _seal(model)


def normalized_actor(model: dict[str, Any], observation: Any) -> np.ndarray[Any, Any]:
    plane = validate_model(model)
    frozen = parent_actor(model["frozen_parent"], observation)
    delta = np.asarray(model["plastic_readout"]) @ plane.project(latent(model, observation))
    return cast(np.ndarray[Any, Any], np.clip(frozen + delta, -1.0, 1.0))


def actor_parameters(model: dict[str, Any], observation: Any) -> np.ndarray[Any, Any]:
    normalized = normalized_actor(model, observation)
    return np.concatenate((normalized[:36] * 0.16, [normalized[36] * 0.3 - 0.05]))


def update_from_physics(
    model: dict[str, Any], samples: list[dict[str, Any]], report_hash: str
) -> dict[str, Any]:
    plane = validate_model(model)
    unique: dict[str, dict[str, Any]] = {}
    for sample in samples:
        key = hash_json(dict(observation=sample["observation"], action=sample["normalized_action"]))
        if key in unique and terminal_return(unique[key]["outcome"]) != terminal_return(
            sample["outcome"]
        ):
            raise ValueError("identical physical input/action has inconsistent return")
        unique.setdefault(key, sample)
    records = list(unique.values())
    if not 1 <= len(records) <= 4096:
        raise ValueError("bounded nonempty physical replay required")
    x = np.stack([latent(model, s["observation"]) for s in records])
    actions = np.asarray([s["normalized_action"] for s in records])
    if (
        actions.shape != (len(records), 37)
        or not np.isfinite(actions).all()
        or np.max(np.abs(actions)) > 1
    ):
        raise ValueError("bounded actually executed motor actions required")
    returns = np.asarray([terminal_return(s["outcome"]) for s in records])
    critic = np.linalg.solve(x.T @ x + 1e-4 * np.eye(DIMENSION), x.T @ returns)
    observations = np.asarray([s["observation"] for s in records])
    # Evaluate V once per identical state. A batched BLAS reduction can differ
    # at the last bit between repeated rows; those are not different states.
    # Keep the Core consistency check strict rather than widening its tolerance.
    state_values: dict[str, float] = {}
    values = []
    for observation, features in zip(observations, x, strict=True):
        key = hash_json(observation.tolist())
        if key not in state_values:
            state_values[key] = float(features @ critic)
        values.append(state_values[key])
    weights = statewise_advantage_weights(
        observations, returns, np.asarray(values), temperature=0.5
    )
    frozen = np.stack([parent_actor(model["frozen_parent"], s["observation"]) for s in records])
    head = fit_protected_readout(
        plane,
        current_readout=model["plastic_readout"],
        latents=x,
        residual_targets=actions - frozen,
        sample_weights=weights,
        ridge=1e-4,
    )
    result = _seal(
        dict(
            model,
            plastic_readout=head.tolist(),
            value_readout=critic.tolist(),
            generation=model["generation"] + 1,
            parent_model_hash=model["model_hash"],
            learning_report_hash=report_hash,
        )
    )
    for features in model["protected_features"]:
        if not np.array_equal(
            actor_parameters(result, features), actor_parameters(model, features)
        ):
            raise ValueError("update changed a protected physical observation")
    return result


def configure_preview(
    model: dict[str, Any], body: Any, first_contact: int | None
) -> dict[str, Any]:
    if first_contact is not None and first_contact < 30:
        raise ValueError("context must precede contact")
    features = context_at30(body)
    values = actor_parameters(model, features)
    policy = make_policy(values[:36].reshape(3, 12), float(values[36]), model["model_hash"])
    policy["progressive_motor_proof"] = dict(
        schema="soccer.rsi.progressive_motor_sim_preview_proof.v1",
        model=model,
        context=list(features),
        decision_frame=30,
        residual_zero_before_decision=True,
        qualification="UNQUALIFIED_SIM_COUNTERFACTUAL",
        promotion_authorized=False,
    )
    policy.pop("policy_hash")
    policy["policy_hash"] = hash_json(policy)
    return policy


def audit_preview(policy: dict[str, Any], body: Any, force: np.ndarray[Any, Any]) -> None:
    proof = policy["progressive_motor_proof"]
    if (
        force.shape != (300, 1, 6)
        or np.any(force[:30] > 1)
        or configure_preview(proof["model"], body, None) != policy
    ):
        raise ValueError("progressive policy differs from causal physical context")
