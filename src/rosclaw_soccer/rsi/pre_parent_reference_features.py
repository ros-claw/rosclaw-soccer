"""Causal 50 Hz inputs for a future pre-parent leg-reference student.

Reuse the measured-state encoder, but never feed the current teacher increment
back as its own input. Navigation is the committed *pre-contact* target. The
parent motor is downstream and must be re-evaluated after a student proposal.
No dataset admission, fitting, simulator execution or hardware authority here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.rsi.contact_reference_labels import reference_labels
from rosclaw_soccer.rsi.step_motor_features import FEATURE_NAMES as MOTOR_FEATURE_NAMES
from rosclaw_soccer.rsi.stochastic_step_execution import features_at_frame

FEATURE_NAMES = tuple(
    name.replace("nominal_target:", "navigation_target_before_contact:").replace(
        "previous_motor_delta:", "previous_applied_reference_increment:"
    )
    for name in MOTOR_FEATURE_NAMES
) + ("remaining_task_time_fraction",)


def actor_features(
    measured_history: dict[str, Any],
    *,
    frame: int,
    navigation_target: NDArray[np.float64],
    previous_applied_reference: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Current measured SIM inputs; runtime caller supplies its own past action.

    Ball/body geometry is the same current kinematic SIM observation used by the
    existing actor, not a real sensor qualification. No teacher phase, outcome,
    catalog ID, oracle route or post-integration force is a feature.
    """
    if type(frame) is not int or not 0 <= frame < 300:
        raise ValueError("original finite-task frame in [0,300) required")
    for array, shape in ((navigation_target, (29,)), (previous_applied_reference, (12,))):
        if (
            type(array) is not np.ndarray
            or array.shape != shape
            or array.dtype != np.float64
            or not np.isfinite(array).all()
        ):
            raise ValueError("actual float64 pre-contact target and past reference required")
    if np.max(np.abs(previous_applied_reference)) > 0.35 + 1e-12 or (
        frame == 0 and np.any(previous_applied_reference != 0)
    ):
        raise ValueError("bounded actually applied previous reference, initially zero, required")
    previous_force = (
        np.asarray(measured_history["force_n"])[frame - 1, 0]
        if frame
        else np.zeros(6, dtype=np.float64)
    )
    encoded = features_at_frame(
        measured_history,
        frame=frame,
        nominal_target=navigation_target,
        previous=previous_applied_reference,
        previous_contact_forces=previous_force,
    )
    return np.concatenate((encoded, [(300 - frame) / 300]))


def trace_observations(trace: dict[str, Any]) -> dict[str, NDArray[np.float64]]:
    """Same-state diagnostic reconstruction, not a training bank export."""
    required = {
        "joint_position_rad": (300, 1, 29),
        "joint_velocity_rad_s": (300, 1, 29),
        "root_pose_xyzw_m": (300, 1, 7),
        "root_velocity_world": (300, 1, 6),
        "ball_position_before_step_m": (300, 1, 3),
        "ball_linear_velocity_before_step_m_s": (300, 1, 3),
        "foot_geometry_position_before_step_m": (300, 1, 4, 3),
        "force_n": (300, 1, 6),
    }
    for key, shape in required.items():
        a = trace.get(key)
        if (
            type(a) is not np.ndarray
            or a.shape != shape
            or a.dtype != np.float64
            or not np.isfinite(a).all()
        ):
            raise ValueError("whole finite unchanged measured-state clock required")
    labels = reference_labels(trace)
    navigation = labels["navigation_joint_target_rad"]
    delta = labels["pre_parent_reference_increment_rad"]
    observations = np.stack(
        [
            actor_features(
                trace,
                frame=frame,
                navigation_target=navigation[frame, 0],
                previous_applied_reference=delta[frame - 1, 0] if frame else np.zeros(12),
            )
            for frame in range(300)
        ]
    )
    if observations.shape != (300, len(FEATURE_NAMES)):
        raise ValueError("complete original 135-dimensional causal frame sequence required")
    return dict(observation=observations, same_state_reference_target_rad=delta[:, 0].copy())
