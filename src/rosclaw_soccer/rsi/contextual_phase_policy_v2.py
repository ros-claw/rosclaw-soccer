"""SIM_ONLY phase actor with measured-context support safeguard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.contextual_phase_policy import (
    FEATURE_NAMES,
    PHASE_ACTIONS,
    context_features,
    fit_action_values,
    select_actions,
)
from rosclaw_soccer.sim.contracts import hash_json

SUPPORT_THRESHOLDS = (0.5, 1.0, 2.0)


def support_aware_actions(
    features: np.ndarray[Any, Any],
    weights: np.ndarray[Any, Any],
    support_contexts: np.ndarray[Any, Any],
    threshold: float,
) -> np.ndarray[Any, Any]:
    if (
        features.ndim != 2
        or features.shape[1] != len(FEATURE_NAMES)
        or support_contexts.ndim != 2
        or support_contexts.shape[1] != len(FEATURE_NAMES)
        or len(support_contexts) < 1
        or not np.isfinite(features).all()
        or not np.isfinite(support_contexts).all()
        or threshold not in SUPPORT_THRESHOLDS
    ):
        raise ValueError("finite supported phase context required")
    selected = select_actions(features, weights)
    # Bias and near-constant snapshot foot coordinate are not distances.
    distances = np.linalg.norm(features[:, None, 1:4] - support_contexts[None, :, 1:4], axis=2)
    selected[np.min(distances, axis=1) > threshold] = 0.0
    return selected


def load_phase_actor_v2(
    path: Path,
) -> tuple[str, np.ndarray[Any, Any], np.ndarray[Any, Any], float]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != "rsi_contextual_sonic_phase_actor_v2"
        or manifest.get("activation_ceiling") != "SIM_ONLY"
        or manifest.get("phase_actions_frames") != list(PHASE_ACTIONS)
        or manifest.get("feature_names") != list(FEATURE_NAMES)
        or manifest.get("support_threshold") not in SUPPORT_THRESHOLDS
        or manifest.get("promotion_authorized") is not False
        or manifest.get("actor_hash")
        != hash_json({key: value for key, value in manifest.items() if key != "actor_hash"})
    ):
        raise ValueError("unauthenticated SIM_ONLY supported phase actor")
    weights = np.asarray(manifest.get("weights"), dtype=np.float64)
    contexts = np.asarray(manifest.get("support_contexts"), dtype=np.float64)
    if (
        weights.shape != (len(FEATURE_NAMES), len(PHASE_ACTIONS))
        or contexts.ndim != 2
        or contexts.shape[1] != len(FEATURE_NAMES)
        or not 35 <= len(contexts) <= 512
        or not np.isfinite(weights).all()
        or not np.isfinite(contexts).all()
    ):
        raise ValueError("invalid supported phase actor weights or contexts")
    return manifest["actor_hash"], weights, contexts, float(manifest["support_threshold"])


__all__ = (
    "FEATURE_NAMES",
    "PHASE_ACTIONS",
    "SUPPORT_THRESHOLDS",
    "context_features",
    "fit_action_values",
    "support_aware_actions",
    "load_phase_actor_v2",
)
