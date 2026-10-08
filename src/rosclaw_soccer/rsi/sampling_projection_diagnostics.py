"""Offline inspection of the contact actor's actual leg-target projection.

Caller authenticates the full physical witnesses and compiled joint bounds.
This checks numeric transport, not new dynamics, action causality or promotion.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def projection_diagnostics(
    latent_actions: Any,
    nominal_targets: Any,
    measured_deltas: Any,
    joint_limits: Any,
    *,
    cap_rad: float,
    slew_rad: float,
    first_contact_frame: int | None,
) -> dict[str, Any]:
    """Replay all 300 frames of the unchanged 12-leg-joint target envelope.

    Counts are coordinate/frame counts, not contact duration or impulses.
    Same-observation projection replay never proves counterfactual dynamics.
    No targets are emitted for runtime consumption.
    """
    if (
        type(cap_rad) not in (int, float)
        or type(slew_rad) not in (int, float)
        or not np.isfinite([cap_rad, slew_rad]).all()
        or not 0 < slew_rad <= cap_rad < 1
        or (
            first_contact_frame is not None
            and (type(first_contact_frame) is not int or not 0 <= first_contact_frame < 300)
        )
    ):
        raise ValueError("finite actual contact envelope and measured event required")
    inputs = [
        np.asarray(v) for v in (latent_actions, nominal_targets, measured_deltas, joint_limits)
    ]
    if any(
        v.shape != shape or v.dtype.kind not in "fiu" or not np.isfinite(v).all()
        for v, shape in zip(inputs, ((300, 12), (300, 12), (300, 12), (12, 2)), strict=True)
    ):
        raise ValueError("complete finite 300-frame 12-joint arrays required")
    latent, nominal, measured, limits = [v.astype(np.float64, copy=False) for v in inputs]
    if np.any(limits[:, 0] >= limits[:, 1]) or np.any(measured[:30] != 0):
        raise ValueError("ordered compiled bounds and zero pre-control deltas required")
    if np.max(np.abs(measured)) > cap_rad + 1e-5:
        raise ValueError("actual deltas violate the original motor envelope")
    desired = cap_rad * np.tanh(latent)
    slew_changed = np.zeros((300, 12), dtype=np.bool_)
    limits_changed = np.zeros((300, 12), dtype=np.bool_)
    transport_error = np.zeros((300, 12), dtype=np.float64)
    replay_error = 0.0
    previous = np.zeros(12, dtype=np.float64)
    for frame in range(30, 300):
        proposed = previous + np.clip(desired[frame] - previous, -slew_rad, slew_rad)
        lower = np.maximum(np.minimum(0.0, limits[:, 0] - nominal[frame]), -cap_rad)
        upper = np.minimum(np.maximum(0.0, limits[:, 1] - nominal[frame]), cap_rad)
        actual = np.clip(proposed, lower, upper)
        error = float(np.max(np.abs(actual - measured[frame])))
        replay_error = max(replay_error, error)
        if not np.array_equal(actual, measured[frame]):
            raise ValueError("actual motor deltas do not EXACTLY replay the original projection")
        slew_changed[frame] = proposed != desired[frame]
        limits_changed[frame] = actual != proposed
        transport_error[frame] = actual - desired[frame]
        previous = actual

    def summarize(start: int, stop: int) -> dict[str, Any]:
        active = stop > start
        error = transport_error[start:stop]
        return {
            "frame_interval": [start, stop],
            "interval_stop_exclusive": True,
            "coordinate_rows": (stop - start) * 12,
            "slew_changed_coordinate_rows": int(slew_changed[start:stop].sum()),
            "joint_limit_changed_coordinate_rows": int(limits_changed[start:stop].sum()),
            "desired_to_actual_rms_rad": float(np.sqrt(np.mean(error**2))) if active else None,
            "desired_to_actual_max_abs_rad": float(np.max(np.abs(error))) if active else None,
        }

    contact = 300 if first_contact_frame is None else max(30, first_contact_frame)
    horizon = 300 if first_contact_frame is None else min(300, max(30, first_contact_frame + 61))
    return {
        "schema": "soccer.rsi.offline_contact_projection_diagnostics.v1",
        "controlled_frames": 270,
        "coordinate_rows": 3240,
        "actual_cap_rad": cap_rad,
        "actual_slew_rad": slew_rad,
        "complete_original_numeric_projection_exact": True,
        "projection_replay_max_abs_error_rad": replay_error,
        "full_controlled": summarize(30, 300),
        "before_first_contact": summarize(30, contact),
        "first_contact_through_plus60": summarize(contact, horizon),
        "after_contact_horizon": summarize(horizon, 300),
        "first_contact_frame": first_contact_frame,
        "source_witnesses_authenticated_here": False,
        "new_physical_executions": 0,
        "optimizer_steps": 0,
        "counterfactual_dynamics_claimed": False,
        "runtime_execution_authorized": False,
        "promotion_authorized": False,
        "hardware_authorized": False,
    }
