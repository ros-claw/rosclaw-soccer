"""SIM-only episodic contextual actor/critic with protected neural features.

One stochastic 37-parameter option decision precedes physical execution.
Returns come from physics, not critic predictions. AWR-inspired updates use a
learned state-value baseline and a generic Core null-space readout constraint.
Not full-trajectory PPO, not a torque actor, and never hardware authorization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import rosclaw.growth.anchor_plane as plane_module
from rosclaw.growth.anchor_plane import AnchorProtectionPlane, fit_protected_readout

from rosclaw_soccer.rsi import motor_bootstrap_network as base_module
from rosclaw_soccer.rsi.bootstrap_motor_execution import context_at30
from rosclaw_soccer.rsi.contact_motor_phase import make_policy
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

SCHEMA = "soccer.rsi.protected_contextual_actor_critic.v308"


def latent(base: dict[str, Any], features: Any) -> np.ndarray[Any, Any]:
    values = np.asarray(features, dtype=np.float64)
    if values.shape != (13,) or not np.isfinite(values).all():
        raise ValueError("thirteen finite causal proprioceptive features required")
    actor = base["actor"]
    first = actor["layers"][0]
    hidden = np.tanh(
        np.asarray(first["weight"])
        @ ((values - np.asarray(actor["mean"])) / np.asarray(actor["scale"]))
        + np.asarray(first["bias"])
    )
    return np.concatenate((hidden, [1.0]))


def _seal(model: dict[str, Any]) -> dict[str, Any]:
    model = {k: v for k, v in model.items() if k != "model_hash"}
    model["model_hash"] = hash_json(model)
    validate_model(model)
    return model


def make_model(
    base: dict[str, Any], protected_features: list[list[float]], parent_report_hash: str
) -> dict[str, Any]:
    base_module.validate_model(base)
    if len(protected_features) != 9:
        raise ValueError("all nine physically successful predecessor contexts must be protected")
    plane = AnchorProtectionPlane(np.stack([latent(base, x) for x in protected_features]))
    return _seal(
        {
            "schema": SCHEMA,
            "activation_ceiling": "SIM_ONLY",
            "learning_kind": "physical_contextual_actor_critic_awr_inspired",
            "base_model": base,
            "protected_features": protected_features,
            "anchor_plane": plane.to_dict(),
            "plastic_readout": np.zeros((37, 33)).tolist(),
            "value_readout": np.zeros(33).tolist(),
            "generation": 0,
            "parent_model_hash": base["model_hash"],
            "learning_report_hash": parent_report_hash,
            "source_hash": hash_bytes(Path(__file__).read_bytes()),
            "core_plane_source_hash": hash_bytes(Path(plane_module.__file__).read_bytes()),
            "fresh_quarantine_protocol_hash": base["fresh_quarantine_protocol_hash"],
            "runtime_execution_authorized": False,
            "promotion_authorized": False,
            "hardware_authorized": False,
            "fresh_holdout_open_authorized": False,
        }
    )


def validate_model(model: dict[str, Any]) -> AnchorProtectionPlane:
    if (
        model.get("schema") != SCHEMA
        or model.get("activation_ceiling") != "SIM_ONLY"
        or model.get("learning_kind") != "physical_contextual_actor_critic_awr_inspired"
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
        or model.get("model_hash")
        != hash_json({k: v for k, v in model.items() if k != "model_hash"})
    ):
        raise ValueError("unsealed protected online contextual model")
    base_module.validate_model(model["base_model"])
    features = model["protected_features"]
    if len(features) != 9:
        raise ValueError("nine protected observation anchors required")
    plane = AnchorProtectionPlane.from_dict(model["anchor_plane"])
    rebuilt = AnchorProtectionPlane(np.stack([latent(model["base_model"], x) for x in features]))
    if rebuilt.to_dict() != plane.to_dict():
        raise ValueError("encoder or protected latent contract changed")
    for key, shape in (("plastic_readout", (37, 33)), ("value_readout", (33,))):
        value = np.asarray(model.get(key), dtype=np.float64)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError("finite aligned online neural readout required")
    if type(model.get("generation")) is not int or model["generation"] < 0:
        raise ValueError("nonnegative online generation required")
    return plane


def normalized_actor(model: dict[str, Any], features: Any) -> np.ndarray[Any, Any]:
    plane = validate_model(model)
    values = np.asarray(features, dtype=np.float64)
    base = base_module._evaluate(model["base_model"]["actor"], values)
    projected = plane.project(latent(model["base_model"], values))
    output = base + np.asarray(model["plastic_readout"]) @ projected
    if not np.isfinite(output).all():
        raise ValueError("nonfinite contextual actor proposal")
    return cast(np.ndarray[Any, Any], np.clip(output, -1.0, 1.0))


def actor_parameters(model: dict[str, Any], features: Any) -> np.ndarray[Any, Any]:
    values = normalized_actor(model, features)
    return np.concatenate((values[:36] * 0.16, [values[36] * 0.3 - 0.05]))


def terminal_return(outcome: dict[str, Any]) -> float:
    keys = ("reward", "minimum_pelvis_z_m", "maximum_lateral_excursion_m")
    if not all(type(outcome[k]) in (float, int) and np.isfinite(outcome[k]) for k in keys) or any(
        type(outcome[k]) is not bool for k in ("high_quality", "clean_foot_only")
    ):
        raise ValueError("finite independently measured terminal outcome required")
    return float(
        outcome["reward"]
        + 10 * outcome["high_quality"]
        - 8 * (not outcome["clean_foot_only"])
        - 20 * (outcome["maximum_lateral_excursion_m"] > 4)
        - 100 * (outcome["minimum_pelvis_z_m"] < 0.65)
    )


def update_from_physics(
    model: dict[str, Any], samples: list[dict[str, Any]], report_hash: str
) -> dict[str, Any]:
    """Fit neural value baseline then advantage-weighted plastic actor readout."""
    plane = validate_model(model)
    if not samples:
        raise ValueError("physical replay cannot be empty")
    base = model["base_model"]
    x = np.stack([latent(base, s["observation"]) for s in samples])
    returns = np.asarray([terminal_return(s["outcome"]) for s in samples])
    actions = np.asarray([s["normalized_action"] for s in samples], dtype=np.float64)
    if (
        actions.shape != (len(samples), 37)
        or not np.isfinite(actions).all()
        or np.max(np.abs(actions)) > 1
    ):
        raise ValueError("bounded physical action evidence required")
    # Repeated deterministic anchors do not increase the effective replay size.
    unique: dict[str, int] = {}
    for index, sample in enumerate(samples):
        key = hash_json(
            {"observation": sample["observation"], "action": sample["normalized_action"]}
        )
        if key in unique and returns[index] != returns[unique[key]]:
            raise ValueError("identical replay input/action has inconsistent return")
        unique.setdefault(key, index)
    ids = list(unique.values())
    x, returns, actions = x[ids], returns[ids], actions[ids]
    value_readout = np.linalg.solve(x.T @ x + 1e-8 * np.eye(33), x.T @ returns)
    advantages = returns - x @ value_readout
    weights = np.exp(np.clip(advantages / 0.5, -8.0, 8.0))
    base_logits = np.stack(
        [
            base_module._evaluate(base["actor"], np.asarray(samples[index]["observation"]))
            for index in ids
        ]
    )
    plastic = fit_protected_readout(
        plane,
        current_readout=model["plastic_readout"],
        latents=x,
        residual_targets=actions - base_logits,
        sample_weights=weights,
    )
    changed = {k: v for k, v in model.items() if k != "model_hash"}
    changed.update(
        plastic_readout=plastic.tolist(),
        value_readout=value_readout.tolist(),
        generation=model["generation"] + 1,
        parent_model_hash=model["model_hash"],
        learning_report_hash=report_hash,
    )
    result = _seal(changed)
    for features in model["protected_features"]:
        if not np.array_equal(
            actor_parameters(result, features), base_module.actor_parameters(base, tuple(features))
        ):
            raise ValueError("plastic update changed a protected neural output")
    return result


def configure_preview(
    model: dict[str, Any], body: Any, first_contact: int | None
) -> dict[str, Any]:
    if first_contact is not None and first_contact < 30:
        raise ValueError("online context must precede contact")
    features = context_at30(body)
    parameters = actor_parameters(model, features)
    policy = make_policy(parameters[:36].reshape(3, 12), float(parameters[36]), model["model_hash"])
    policy["online_motor_proof"] = {
        "model": model,
        "context": list(features),
        "decision_frame": 30,
        "qualification": "UNQUALIFIED_SIM_COUNTERFACTUAL",
        "promotion_authorized": False,
    }
    policy.pop("policy_hash")
    policy["policy_hash"] = hash_json(policy)
    return policy


def audit_preview(policy: dict[str, Any], body: Any, force: np.ndarray[Any, Any]) -> None:
    proof = policy["online_motor_proof"]
    if (
        not isinstance(proof, dict)
        or proof.get("decision_frame") != 30
        or proof.get("qualification") != "UNQUALIFIED_SIM_COUNTERFACTUAL"
        or proof.get("promotion_authorized") is not False
        or force.shape != (300, 1, 6)
        or np.any(force[:30] > 1)
    ):
        raise ValueError("invalid online simulator preview proof")
    if configure_preview(proof["model"], body, None) != policy:
        raise ValueError("online neural parameters disagree with measured causal state")
