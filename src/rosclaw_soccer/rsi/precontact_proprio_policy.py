"""Bounded SIM_ONLY frame-30 proprioceptive approach-risk decision."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import hash_json

FEATURE_NAMES = (
    "ball_minus_root_x_m",
    "ball_minus_root_y_m",
    "ball_vx_m_s",
    "root_vx_m_s",
    "root_vy_m_s",
    "left_foot_ball_distance_m",
    "right_foot_ball_distance_m",
    "left_knee_ball_distance_m",
    "right_knee_ball_distance_m",
    "left_foot_height_m",
    "right_foot_height_m",
    "left_knee_ball_closing_m_s",
    "right_knee_ball_closing_m_s",
)
V300_HASH = "sha256:0bace60dad016fe04ac7a1e5c0fc5c4d90b39dbbb85c08f465a8a6a402f0e2ba"


def proprio_vector(
    root_pose: np.ndarray,
    root_velocity: np.ndarray,
    ball20: np.ndarray,
    ball30: np.ndarray,
    ball_velocity30: np.ndarray,
    geometry20: np.ndarray,
    geometry30: np.ndarray,
) -> tuple[float, ...]:
    """Use only measurements available at or before control frame 30."""
    arrays = (root_pose, root_velocity, ball20, ball30, ball_velocity30, geometry20, geometry30)
    if tuple(value.shape for value in arrays) != (
        (7,),
        (6,),
        (3,),
        (3,),
        (3,),
        (4, 3),
        (4, 3),
    ) or not all(np.isfinite(value).all() for value in arrays):
        raise ValueError("finite aligned precontact body/ball arrays required")
    distance20 = np.linalg.norm(geometry20 - ball20, axis=1)
    distance30 = np.linalg.norm(geometry30 - ball30, axis=1)
    values = (
        float(ball30[0] - root_pose[0]),
        float(ball30[1] - root_pose[1]),
        float(ball_velocity30[0]),
        float(root_velocity[0]),
        float(root_velocity[1]),
        *(float(value) for value in distance30),
        float(geometry30[0, 2]),
        float(geometry30[1, 2]),
        float((distance20[2] - distance30[2]) / 0.2),
        float((distance20[3] - distance30[3]) / 0.2),
    )
    if len(values) != len(FEATURE_NAMES):
        raise ValueError("thirteen causal proprioceptive features required")
    return values


def validate_policy(policy: dict[str, Any]) -> tuple[dict[str, Any], str]:
    expected_hash = hash_json({key: value for key, value in policy.items() if key != "policy_hash"})
    if (
        policy.get("schema") != "rsi_precontact_proprio_approach_policy_v1"
        or policy.get("activation_ceiling") != "SIM_ONLY"
        or policy.get("training_report_hash") != V300_HASH
        or policy.get("feature_names") != list(FEATURE_NAMES)
        or policy.get("decision_frame") != 30
        or policy.get("threshold") != 0.2
        or policy.get("aggressive_gain") != 1.2
        or policy.get("fallback_gain") != 0.8
        or policy.get("promotion_authorized") is not False
        or policy.get("policy_hash") != expected_hash
    ):
        raise ValueError("unsealed SIM_ONLY proprioceptive policy")
    for key in ("mean", "scale", "coefficients"):
        value = policy.get(key)
        if (
            not isinstance(value, list)
            or len(value) != len(FEATURE_NAMES)
            or not all(type(item) in (float, int) and math.isfinite(item) for item in value)
        ):
            raise ValueError("finite thirteen-feature policy weights required")
    if (
        not all(value > 0 for value in policy["scale"])
        or type(policy.get("intercept"))
        not in (
            float,
            int,
        )
        or not math.isfinite(policy["intercept"])
    ):
        raise ValueError("finite scaled policy intercept required")
    return policy, expected_hash


def load_policy(path: Path) -> tuple[dict[str, Any], str]:
    return validate_policy(json.loads(path.read_text(encoding="utf-8")))


def risk_probability(policy: dict[str, Any], features: tuple[float, ...]) -> float:
    validate_policy(policy)
    if len(features) != len(FEATURE_NAMES) or not all(math.isfinite(value) for value in features):
        raise ValueError("finite causal proprioceptive features required")
    logit = policy["intercept"] + sum(
        weight * (value - mean) / scale
        for value, mean, scale, weight in zip(
            features, policy["mean"], policy["scale"], policy["coefficients"], strict=True
        )
    )
    if not math.isfinite(logit):
        raise ValueError("nonfinite risk logit")
    return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, logit))))
