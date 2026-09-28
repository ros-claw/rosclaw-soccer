"""Small, auditable SIM_ONLY phase actor fitted to physical contact outcomes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import hash_json

PHASE_ACTIONS = (0.0, 3.0, 6.0)
FEATURE_NAMES = (
    "bias",
    "ball_minus_root_x_centered_scaled",
    "ball_minus_root_y_scaled",
    "ball_vx_scaled",
    "right_foot_minus_right_knee_x_centered_scaled",
)


def context_features(
    root_pose: np.ndarray[Any, Any],
    ball_position: np.ndarray[Any, Any],
    ball_velocity: np.ndarray[Any, Any],
    foot_positions: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    """Only use measured snapshot-time state; no future contact data."""
    count = len(root_pose)
    if (
        root_pose.shape != (count, 7)
        or ball_position.shape != (count, 3)
        or ball_velocity.shape != (count, 3)
        or foot_positions.shape != (count, 4, 3)
        or count < 1
        or not all(
            np.isfinite(v).all() for v in (root_pose, ball_position, ball_velocity, foot_positions)
        )
    ):
        raise ValueError("finite measured snapshot context required")
    gap = ball_position - root_pose[:, :3]
    foot_ahead = foot_positions[:, 1, 0] - foot_positions[:, 3, 0]
    return np.column_stack(
        (
            np.ones(count),
            (gap[:, 0] - 2.0) / 0.5,
            gap[:, 1] / 0.2,
            ball_velocity[:, 0] / 0.5,
            (foot_ahead + 0.15) / 0.15,
        )
    )


def fit_action_values(
    features: np.ndarray[Any, Any], rewards: np.ndarray[Any, Any], ridge: float
) -> np.ndarray[Any, Any]:
    if (
        features.ndim != 2
        or features.shape[1] != len(FEATURE_NAMES)
        or rewards.shape != (len(features), len(PHASE_ACTIONS))
        or not np.isfinite(features).all()
        or not np.isfinite(rewards).all()
        or ridge not in (0.1, 1.0, 10.0)
    ):
        raise ValueError("bounded physical reward regression required")
    penalty = np.eye(features.shape[1]) * ridge
    penalty[0, 0] = 0.0
    return np.linalg.solve(features.T @ features + penalty, features.T @ rewards)


def select_actions(
    features: np.ndarray[Any, Any], weights: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    if (
        features.ndim != 2
        or features.shape[1] != len(FEATURE_NAMES)
        or weights.shape != (len(FEATURE_NAMES), len(PHASE_ACTIONS))
        or not np.isfinite(features).all()
        or not np.isfinite(weights).all()
    ):
        raise ValueError("invalid contextual phase actor")
    values = features @ weights
    chosen: np.ndarray[Any, Any] = np.asarray(PHASE_ACTIONS)[np.argmax(values, axis=1)]
    return chosen


def load_phase_actor(path: Path) -> tuple[str, np.ndarray[Any, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != "rsi_contextual_sonic_phase_actor_v1"
        or manifest.get("activation_ceiling") != "SIM_ONLY"
        or manifest.get("phase_actions_frames") != list(PHASE_ACTIONS)
        or manifest.get("feature_names") != list(FEATURE_NAMES)
        or manifest.get("promotion_authorized") is not False
        or manifest.get("actor_hash")
        != hash_json({key: value for key, value in manifest.items() if key != "actor_hash"})
    ):
        raise ValueError("unauthenticated SIM_ONLY contextual phase actor")
    weights = np.asarray(manifest.get("weights"), dtype=np.float64)
    if weights.shape != (len(FEATURE_NAMES), len(PHASE_ACTIONS)) or not np.isfinite(weights).all():
        raise ValueError("invalid contextual phase actor weights")
    return manifest["actor_hash"], weights
