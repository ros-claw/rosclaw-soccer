"""Bounded leg-action proposal for private SIM counterfactual rollouts.

This is an experiment codec, not a trained player or a hardware safety gate.
The caller must use the returned action as the next observation's action
history, and judge the proposed target by actual shared-world physics.
"""

import numpy as np

from rosclaw_soccer.providers.g1.velocity_policy import VelocityPolicyParameters, target29


def receiver_counterfactual_residual(
    nominal_action: np.ndarray,
    leg_delta_rad: np.ndarray,
    params: VelocityPolicyParameters,
    *,
    amplitude: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode a small leg-only position-target correction in policy units."""
    if (
        type(nominal_action) is not np.ndarray
        or nominal_action.shape != (29,)
        or nominal_action.dtype != np.float32
        or not np.isfinite(nominal_action).all()
        or type(leg_delta_rad) is not np.ndarray
        or leg_delta_rad.shape != (12,)
        or leg_delta_rad.dtype != np.float64
        or not np.isfinite(leg_delta_rad).all()
        or np.max(np.abs(leg_delta_rad)) > 0.15
        or type(params) is not VelocityPolicyParameters
        or type(amplitude) is not float
        or not np.isfinite(amplitude)
        or not 0.0 <= amplitude <= 1.0
    ):
        raise ValueError("finite own action and bounded explicit SIM leg correction required")
    nominal_target = target29(nominal_action, params)
    action = nominal_action.copy()
    scale = np.asarray(params.action_scale[:12], dtype=np.float32)
    action[:12] += (amplitude * leg_delta_rad / scale).astype(np.float32)
    target = target29(action, params)
    if (
        not np.array_equal(action[12:], nominal_action[12:])
        or np.max(np.abs(target[:12] - nominal_target[:12])) > 0.150001
        or not np.array_equal(target[12:], nominal_target[12:])
    ):
        raise ValueError("leg-only bounded correction and exact encoded target required")
    return action, target
