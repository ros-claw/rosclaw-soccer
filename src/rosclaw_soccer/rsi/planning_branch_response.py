"""Separate response domain for unconditional native planning branches.

Reuses the existing Core-backed matched regressor, without relabelling branch
forecasts as actual safety-guarded H300 executions or fitting an actor.
"""

import re
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.observed_ball_response import fit_matched_effect, predict_matched_effect
from rosclaw_soccer.sim.contracts import hash_json

SCHEMA = "soccer.research.unconditional_planning_branch_response.v1"
LAW = "UNCONDITIONAL_60MS_PLANNING_BRANCH_NOT_GUARDED_EXECUTION"


def fit_planning_response(
    baseline_features: np.ndarray,
    action_features: np.ndarray,
    baseline_delta_velocity: np.ndarray,
    action_delta_velocity: np.ndarray,
    *,
    source_dataset_hash: str,
    seed: int,
    steps: int = 256,
) -> dict[str, Any]:
    """Caller authenticates physics, matching and train-only ancestry first."""
    if (
        type(source_dataset_hash) is not str
        or re.fullmatch(r"sha256:[0-9a-f]{64}", source_dataset_hash) is None
    ):
        raise ValueError("explicit source dataset commitment required")
    fitted = fit_matched_effect(
        baseline_features,
        action_features,
        baseline_delta_velocity,
        action_delta_velocity,
        seed=seed,
        steps=steps,
    )
    model = dict(
        schema=SCHEMA,
        conditioning_law=LAW,
        source_dataset_hash=source_dataset_hash,
        mode="SIM_ONLY",
        response_model=fitted,
        source_or_physics_authenticated_here=False,
        guarded_execution_response_qualified=False,
        actor_optimizer_updates=0,
        teacher_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    model["model_hash"] = hash_json(model)
    return model


def predict_planning_response(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    if (
        type(model) is not dict
        or model.get("schema") != SCHEMA
        or model.get("conditioning_law") != LAW
        or model.get("mode") != "SIM_ONLY"
        or type(model.get("source_dataset_hash")) is not str
        or re.fullmatch(r"sha256:[0-9a-f]{64}", model["source_dataset_hash"]) is None
        or model.get("model_hash")
        != hash_json({k: v for k, v in model.items() if k != "model_hash"})
        or type(model.get("actor_optimizer_updates")) is not int
        or model["actor_optimizer_updates"] != 0
        or any(
            model.get(flag) is not False
            for flag in (
                "source_or_physics_authenticated_here",
                "guarded_execution_response_qualified",
                "teacher_qualified",
                "hardware_authorized",
                "promotion_authorized",
            )
        )
    ):
        raise ValueError("sealed planning-only response with exact conditioning law required")
    return predict_matched_effect(model["response_model"], features)
