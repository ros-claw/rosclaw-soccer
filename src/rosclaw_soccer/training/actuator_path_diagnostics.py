"""Numerical diagnostics for recorded simulation motor authority.

This helper has no simulator or motor handle. The caller is responsible for
authenticating and time-aligning measured input arrays before calling it.
Differences between predicted and applied torques are evidence for further
investigation, never proof of their cause or of a safe policy.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def diagnose_actuator_path(
    *,
    position_rad: NDArray[np.floating[Any]],
    velocity_radps: NDArray[np.floating[Any]],
    target_rad: NDArray[np.floating[Any]],
    kp: NDArray[np.floating[Any]],
    kd: NDArray[np.floating[Any]],
    extra_torque_nm: NDArray[np.floating[Any]],
    applied_torque_nm: NDArray[np.floating[Any]],
    torque_cap_nm: NDArray[np.floating[Any]],
    joint_names: tuple[str, ...],
    tolerance_nm: float = 1e-8,
) -> dict[str, Any]:
    """Compare aligned requested, independently capped and applied torque.

    Every history is shaped ``[substep, joint]`` at the same pre-integration
    boundary. ``extra_torque_nm`` is the logged additive contribution before
    safety projection, not an inferred residual. The applied command must
    already respect the supplied positive per-joint absolute torque caps.
    Counts are joint-substeps, not independent events; the outcome does not
    identify which downstream mechanism produced a difference.
    """
    histories = (
        position_rad,
        velocity_radps,
        target_rad,
        kp,
        kd,
        extra_torque_nm,
        applied_torque_nm,
    )
    if (
        any(not isinstance(value, np.ndarray) for value in (*histories, torque_cap_nm))
        or position_rad.ndim != 2
        or not 1 <= position_rad.shape[0] <= 1_000_000
        or not 1 <= position_rad.shape[1] <= 256
        or any(value.shape != position_rad.shape for value in histories)
        or torque_cap_nm.shape != (position_rad.shape[1],)
        or any(value.dtype.kind not in "fiu" for value in (*histories, torque_cap_nm))
        or any(not np.isfinite(value).all() for value in (*histories, torque_cap_nm))
        or type(joint_names) is not tuple
        or len(joint_names) != position_rad.shape[1]
        or any(type(name) is not str or not name or len(name) > 128 for name in joint_names)
        or len(set(joint_names)) != len(joint_names)
        or type(tolerance_nm) not in (int, float)
        or not np.isfinite(tolerance_nm)
        or not 0 < tolerance_nm <= 1e-3
        or np.any(kp < 0)
        or np.any(kd < 0)
        or np.any(torque_cap_nm <= tolerance_nm)
    ):
        raise ValueError("finite, aligned, named actuator histories and positive caps required")

    # Compute in float64, not the possibly lower-precision evidence dtype.
    q, velocity, target, stiffness, damping, extra, applied = (
        value.astype(np.float64) for value in histories
    )
    caps = torque_cap_nm.astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        requested = stiffness * (target - q) - damping * velocity + extra
    if not np.isfinite(requested).all():
        raise ValueError("nonfinite reconstructed request")
    if np.any(np.abs(applied) > caps + tolerance_nm):
        raise ValueError("applied torque exceeds supplied cap")
    clipped_request = np.clip(requested, -caps, caps)
    changed = np.abs(applied - clipped_request) > tolerance_nm
    capped = np.abs(applied) >= caps - tolerance_nm
    reversed_sign = (
        (np.abs(requested) > tolerance_nm)
        & (np.abs(applied) > tolerance_nm)
        & (np.signbit(requested) != np.signbit(applied))
    )
    by_joint = []
    for index, name in enumerate(joint_names):
        by_joint.append(
            {
                "joint": name,
                "post_cap_difference_substeps": int(changed[:, index].sum()),
                "at_cap_substeps": int(capped[:, index].sum()),
                "sign_reversal_substeps": int(reversed_sign[:, index].sum()),
                "max_abs_requested_nm": float(np.max(np.abs(requested[:, index]))),
                "max_abs_applied_nm": float(np.max(np.abs(applied[:, index]))),
            }
        )
    return {
        "schema": "soccer.actuator_path_diagnostics.v1",
        "substeps": int(position_rad.shape[0]),
        "joints": by_joint,
        "input_evidence_authenticated": False,
        "cause_inferred": False,
        "training_authorized": False,
        "promotion_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
