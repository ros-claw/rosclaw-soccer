"""Causal predicted-contact gait-phase features for SIM_ONLY first touch."""

from __future__ import annotations

from typing import Any

import numpy as np

GAIT_PERIOD_FRAMES = 40.0
RAW_FEATURE_NAMES = ("bias", "gap_x_scaled", "gap_y_scaled", "ball_vx_scaled", "root_vx_scaled")
ACTION_FEATURE_NAMES = RAW_FEATURE_NAMES + (
    "sin_predicted_gait_phase",
    "cos_predicted_gait_phase",
    "lateral_times_sin_phase",
    "lateral_times_cos_phase",
)


def current_context(
    root_pose: np.ndarray[Any, Any],
    root_velocity: np.ndarray[Any, Any],
    ball_position: np.ndarray[Any, Any],
    ball_velocity: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    count = len(root_pose)
    if (
        count < 1
        or root_pose.shape != (count, 7)
        or root_velocity.shape != (count, 6)
        or ball_position.shape != (count, 3)
        or ball_velocity.shape != (count, 3)
        or not all(
            np.isfinite(value).all()
            for value in (root_pose, root_velocity, ball_position, ball_velocity)
        )
    ):
        raise ValueError("finite measured ball and body context required")
    gap = ball_position - root_pose[:, :3]
    return np.column_stack(
        (
            np.ones(count),
            (gap[:, 0] - 2.0) / 0.5,
            gap[:, 1] / 0.2,
            ball_velocity[:, 0] / 0.5,
            root_velocity[:, 0],
        )
    )


def fit_contact_time(
    context: np.ndarray[Any, Any], first_contact_offsets: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    if (
        context.ndim != 2
        or context.shape[1] != len(RAW_FEATURE_NAMES)
        or first_contact_offsets.shape != (len(context),)
        or len(context) < 8
        or not np.isfinite(context).all()
        or not np.isfinite(first_contact_offsets).all()
        or np.any(first_contact_offsets < 15)
        or np.any(first_contact_offsets > 120)
    ):
        raise ValueError("bounded development contact-time labels required")
    design = np.column_stack(
        (
            context[:, 0],
            context[:, 1],
            context[:, 2],
            context[:, 3],
            context[:, 4],
            context[:, 1] * context[:, 3],
        )
    )
    penalty = np.eye(design.shape[1])
    penalty[0, 0] = 0
    fitted: np.ndarray[Any, Any] = np.linalg.solve(
        design.T @ design + penalty, design.T @ first_contact_offsets
    )
    return fitted


def predict_contact_time(
    context: np.ndarray[Any, Any], weights: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    if (
        context.ndim != 2
        or context.shape[1] != len(RAW_FEATURE_NAMES)
        or weights.shape != (6,)
        or not np.isfinite(context).all()
        or not np.isfinite(weights).all()
    ):
        raise ValueError("invalid causal contact-time predictor")
    design = np.column_stack(
        (
            context[:, 0],
            context[:, 1],
            context[:, 2],
            context[:, 3],
            context[:, 4],
            context[:, 1] * context[:, 3],
        )
    )
    prediction: np.ndarray[Any, Any] = np.clip(design @ weights, 15.0, 120.0)
    return prediction


def gait_phase_features(
    context: np.ndarray[Any, Any], predicted_contact_offsets: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    if (
        context.ndim != 2
        or context.shape[1] != len(RAW_FEATURE_NAMES)
        or predicted_contact_offsets.shape != (len(context),)
        or not np.isfinite(context).all()
        or not np.isfinite(predicted_contact_offsets).all()
    ):
        raise ValueError("invalid predicted gait phase context")
    phase = 2 * np.pi * (30.0 + predicted_contact_offsets) / GAIT_PERIOD_FRAMES
    sine = np.sin(phase)
    cosine = np.cos(phase)
    return np.column_stack((context, sine, cosine, context[:, 2] * sine, context[:, 2] * cosine))


def fit_action_values(
    features: np.ndarray[Any, Any], rewards: np.ndarray[Any, Any], ridge: float
) -> np.ndarray[Any, Any]:
    if (
        features.ndim != 2
        or features.shape[1] != len(ACTION_FEATURE_NAMES)
        or rewards.shape != (len(features), 3)
        or not np.isfinite(features).all()
        or not np.isfinite(rewards).all()
        or ridge not in (0.1, 1.0, 10.0)
    ):
        raise ValueError("bounded physical action value regression required")
    penalty = np.eye(features.shape[1]) * ridge
    penalty[0, 0] = 0
    solved: np.ndarray[Any, Any] = np.linalg.solve(
        features.T @ features + penalty, features.T @ rewards
    )
    return solved


def select_phase_actions(
    features: np.ndarray[Any, Any], weights: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    if (
        features.ndim != 2
        or features.shape[1] != len(ACTION_FEATURE_NAMES)
        or weights.shape != (len(ACTION_FEATURE_NAMES), 3)
        or not np.isfinite(features).all()
        or not np.isfinite(weights).all()
    ):
        raise ValueError("invalid learned gait-phase action value")
    selected: np.ndarray[Any, Any] = np.asarray((0.0, 3.0, 6.0))[
        np.argmax(features @ weights, axis=1)
    ]
    return selected
