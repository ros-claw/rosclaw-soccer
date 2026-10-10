"""Measured legacy approach diagnostics, not a controller or causal proof.

This applies only to the frozen native first-touch command law. Foot/body
positions are geometry centres, not surface distances or collision witnesses.
Input provenance and independent dynamics replay remain the caller's job.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def diagnose_native_approach(trace: dict[str, Any]) -> dict[str, Any]:
    """Inspect all300 measured frames and check the legacy navigation law.

    Do not use a smaller prefix or silently repair missing/nonfinite arrays.
    Contact is determined only by the recorded force threshold, not distance.
    """
    shapes = {
        "root_pose_xyzw_m": (300, 1, 7),
        "ball_position_before_step_m": (300, 1, 3),
        "foot_geometry_position_before_step_m": (300, 1, 4, 3),
        "navigation_command": (300, 1, 3),
        "force_n": (300, 1, 6),
        "motor_delta_rad": (300, 1, 12),
    }
    if type(trace) is not dict:
        raise ValueError("complete ordinary native trajectory mapping required")
    arrays: dict[str, Any] = {}
    for name, shape in shapes.items():
        raw = np.asarray(trace.get(name))
        if (
            raw.shape != shape
            or raw.dtype.kind not in "fiu"
            or not np.isfinite(raw).all()
            or np.max(np.abs(raw)) > 1e6
        ):
            raise ValueError("complete bounded finite300-frame native arrays required")
        arrays[name] = np.array(raw, dtype=np.float64, copy=True)
    root = arrays["root_pose_xyzw_m"][:, 0, :3]
    ball = arrays["ball_position_before_step_m"][:, 0]
    feet = arrays["foot_geometry_position_before_step_m"][:, 0]
    force = arrays["force_n"][:, 0]
    command = arrays["navigation_command"][:, 0]
    if np.any(force < 0):
        raise ValueError("recorded contact force norms must be nonnegative")
    contacts = np.flatnonzero(np.any(force > 1, axis=1))
    first = int(contacts[0]) if len(contacts) else None
    before = np.arange(300) <= (299 if first is None else first)
    gap = ball - root
    enabled = (ball[0, 1] < 0) & (gap[:, 0] > 0.95) & before
    expected = np.column_stack(
        (
            np.full(300, 1.4),
            np.where(enabled, np.clip(1.2 * gap[:, 1], -0.2, 0.2), 0.0),
            np.zeros(300),
        )
    )
    if not np.array_equal(command, expected):
        raise ValueError("trace differs from exact frozen native navigation command law")
    distance = np.linalg.norm(feet - ball[:, None], axis=-1)
    stop = 300 if first is None else first + 1
    nearest = np.unravel_index(np.argmin(distance[:stop]), (stop, 4))
    frame, foot = int(nearest[0]), int(nearest[1])
    local = feet[frame, foot] - ball[frame]
    close = before & (np.abs(gap[:, 0]) <= 0.95)
    return dict(
        schema="soccer.rsi.native_approach_diagnostics.v1",
        frames=300,
        first_contact_frame=first,
        recorded_contact_detected=first is not None,
        exact_legacy_navigation_commands_reconstructed=True,
        initial_y_disables_legacy_lateral_tracking=bool(ball[0, 1] >= 0),
        precontact_lateral_command_active_frames=int(np.count_nonzero(command[before, 1])),
        precontact_close_longitudinal_frames=int(np.count_nonzero(close)),
        precontact_close_with_zero_lateral_command_frames=int(
            np.count_nonzero(close & (command[:, 1] == 0))
        ),
        root_overtook_ball_before_contact=bool(np.any(before & (gap[:, 0] <= 0))),
        nearest_precontact_foot_centre_frame=frame,
        nearest_precontact_foot_centre_index=foot,
        nearest_precontact_foot_centre_distance_m=float(distance[frame, foot]),
        nearest_precontact_foot_centre_minus_ball_xyz_m=local.tolist(),
        navigation_command_at_nearest_centre=command[frame].tolist(),
        peak_abs_recorded_motor_delta_rad=float(np.max(np.abs(arrays["motor_delta_rad"]))),
        geometry_surface_distance_checked=False,
        input_receipts_authenticated_here=False,
        physics_replayed_here=False,
        neural_policy_calls_recomputed_here=False,
        causal_failure_reason_proven=False,
        training_data_admitted=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
