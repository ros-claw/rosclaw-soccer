"""Explicit SIM perception ABI: velocity98 + reference70 + current ball/goal9.

The last nine values are metres, m/s, metres expressed in the current pelvis
basis: ball minus pelvis position, ball minus pelvis world linear velocity,
and target minus pelvis position. Relative velocity is not a rotating-frame
coordinate derivative. No clipping, cached launch position, or future state.
"""

import math
from typing import Any

import numpy as np

BALL_AWARE_ABI = "soccer.velocity98.reference70.current_ball_goal9.v1"
BALL_AWARE_DIM = 177


def require_ball_aware_qualification(report: dict[str, Any]) -> None:
    """Fail-closed structural gate; caller must authenticate all files and unit exit."""
    if (
        report.get("status") != "BALL_AWARE_ABI_SENSOR_QUALIFICATION_NOT_LEARNED_SKILL"
        or report.get("ball_aware_abi") != BALL_AWARE_ABI
        or report.get("actor_observation_dim") != 177
        or report.get("critic_observation_dim") != 304
        or report.get("checkpoint_file") != "ball-aware-initialization.pt"
        or type(report.get("new_optimizer_steps")) is not int
        or report["new_optimizer_steps"] != 0
        or report.get("optimizer_state_migrated") is not False
        or report.get("weights_only_checkpoint_reload") is not True
        or any(
            report.get(k) is not False
            for k in (
                "goal_complete",
                "football_skill_proven",
                "hardware_authorized",
                "promotion_authorized",
            )
        )
    ):
        raise ValueError("independent non-promoted new perception/sensor qualification required")
    for key in ("perception_max_error", "actor_migration_max_error"):
        value = report.get(key)
        if not isinstance(value, (float, int)) or isinstance(value, bool):
            raise ValueError("numeric measured perception/migration error required")
        if not math.isfinite(value) or not 0 <= value <= 1e-5:
            raise ValueError(
                "measured finite perception/migration error within original bounds required"
            )
    steps = report.get("episode_gpu_world_steps")
    if (
        type(steps) is not int
        or not 80 <= steps <= 24000
        or steps % 80
        or (
            report.get("actual_neural_rows") != steps // 5
            or report.get("perception_native_fk_calls") != steps // 10
            or report.get("fixture_gpu_world_steps") != 80
            or report.get("fixture_native_steps") != 80
            or report.get("full_reference_episodes_completed") != [True] * 8
            or report.get("failed") != [False] * 8
            or report.get("right_censored") != [False] * 8
        )
    ):
        raise ValueError(
            "complete bounded eight-source measured trajectories and fixtures required"
        )
    cal = report.get("sensor_calibration")
    expected = [[i % 2 == 0, i % 2 == 1] if i < 6 else [False, False] for i in range(8)]
    if not isinstance(cal, dict) or any(
        cal.get(k) != expected for k in ("expected_labels", "gpu_labels", "native_labels")
    ):
        raise ValueError("all left/right/negative physical contact calibration labels required")
    for key, limit in (
        ("native_sensor_vs_collision_max_error_n", 1e-5),
        ("max_qpos_error_mixed_units", 0.001),
        ("max_qvel_error_mixed_units", 0.02),
    ):
        value = cal.get(key)
        if not isinstance(value, (float, int)) or isinstance(value, bool):
            raise ValueError("numeric actual contact and physics error required")
        if not math.isfinite(value) or not 0 <= value <= limit:
            raise ValueError("independent native actual contact and physics agreement required")


def ball_goal9(
    *,
    pelvis_rotation: np.ndarray,
    pelvis_position: np.ndarray,
    pelvis_linear_velocity: np.ndarray,
    ball_position: np.ndarray,
    ball_linear_velocity: np.ndarray,
    target_position: np.ndarray,
) -> np.ndarray:
    for value, shape in (
        (pelvis_rotation, (3, 3)),
        (pelvis_position, (3,)),
        (pelvis_linear_velocity, (3,)),
        (ball_position, (3,)),
        (ball_linear_velocity, (3,)),
        (target_position, (3,)),
    ):
        if (
            not isinstance(value, np.ndarray)
            or value.shape != shape
            or value.dtype not in (np.float32, np.float64)
            or not np.isfinite(value).all()
        ):
            raise ValueError("finite typed measured pelvis/ball/target geometry required")
    if (
        not np.allclose(pelvis_rotation.T @ pelvis_rotation, np.eye(3), rtol=0, atol=1e-5)
        or abs(np.linalg.det(pelvis_rotation) - 1) > 1e-5
    ):
        raise ValueError("proper current pelvis rotation required")
    value = np.concatenate(
        [
            pelvis_rotation.T @ delta
            for delta in (
                ball_position - pelvis_position,
                ball_linear_velocity - pelvis_linear_velocity,
                target_position - pelvis_position,
            )
        ]
    ).astype(np.float32)
    if not np.isfinite(value).all():
        raise ValueError("representable finite perception9 required")
    return value
