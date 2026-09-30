"""Statewise stable replay recalibration of the protected contextual motor actor.

Same frozen inference architecture as v308. Only the declared training rule
changes; source-sealed weights still need independent physical qualification.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rosclaw.growth.advantage_weights as weight_module
from rosclaw.growth.advantage_weights import statewise_advantage_weights
from rosclaw.growth.anchor_plane import fit_protected_readout

from rosclaw_soccer.rsi import motor_bootstrap_network as base_module
from rosclaw_soccer.rsi.online_motor_actor_critic import (
    actor_parameters,
    latent,
    terminal_return,
    validate_model,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def stable_update(
    model: dict[str, Any], samples: list[dict[str, Any]], report_hash: str
) -> dict[str, Any]:
    plane = validate_model(model)
    if not samples:
        raise ValueError("physical replay required")
    unique: dict[str, dict[str, Any]] = {}
    for sample in samples:
        key = hash_json(
            {"observation": sample["observation"], "action": sample["normalized_action"]}
        )
        if key in unique and terminal_return(sample["outcome"]) != terminal_return(
            unique[key]["outcome"]
        ):
            raise ValueError("identical state/action has conflicting physical returns")
        unique.setdefault(key, sample)
    replay = list(unique.values())
    observations = [s["observation"] for s in replay]
    x = np.stack([latent(model["base_model"], s) for s in observations])
    returns = np.asarray([terminal_return(s["outcome"]) for s in replay])
    actions = np.asarray([s["normalized_action"] for s in replay], dtype=np.float64)
    if (
        actions.shape != (len(replay), 37)
        or not np.isfinite(actions).all()
        or np.max(np.abs(actions)) > 1
    ):
        raise ValueError("bounded physical motor actions required")
    value = np.linalg.solve(x.T @ x + 1e-8 * np.eye(33), x.T @ returns)
    predictions: dict[tuple[float, ...], float] = {}
    for row, features in zip(x, observations, strict=True):
        predictions.setdefault(tuple(features), float(row @ value))
    weights = statewise_advantage_weights(
        observations, returns, [predictions[tuple(s)] for s in observations]
    )
    baseline = np.stack(
        [base_module._evaluate(model["base_model"]["actor"], np.asarray(s)) for s in observations]
    )
    plastic = fit_protected_readout(
        plane,
        current_readout=model["plastic_readout"],
        latents=x,
        residual_targets=actions - baseline,
        sample_weights=weights,
    )
    updated = {k: v for k, v in model.items() if k != "model_hash"}
    updated.update(
        plastic_readout=plastic.tolist(),
        value_readout=value.tolist(),
        generation=model["generation"] + 1,
        parent_model_hash=model["model_hash"],
        learning_report_hash=report_hash,
        update_rule="statewise_stable_advantage_v309",
        update_source_hash=hash_bytes(Path(__file__).read_bytes()),
        weighting_source_hash=hash_bytes(Path(weight_module.__file__).read_bytes()),
    )
    updated["model_hash"] = hash_json(updated)
    validate_stable_model(updated)
    for features in model["protected_features"]:
        if not np.array_equal(
            actor_parameters(updated, features),
            base_module.actor_parameters(model["base_model"], tuple(features)),
        ):
            raise ValueError("stable actor update changed a protected output")
    return updated


def validate_stable_model(model: dict[str, Any]) -> None:
    validate_model(model)
    if (
        model.get("update_rule") != "statewise_stable_advantage_v309"
        or model.get("update_source_hash") != hash_bytes(Path(__file__).read_bytes())
        or model.get("weighting_source_hash")
        != hash_bytes(Path(weight_module.__file__).read_bytes())
    ):
        raise ValueError("unsealed stable actor update rule")
