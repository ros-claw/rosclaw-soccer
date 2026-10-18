"""Measured post-contact ball path diagnostics, not an impulse decomposition.

Frame aggregates cannot localize substep impacts or separate foot and ground
impulses. Labels remain the frozen finite-horizon first-touch contract.
"""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch


def diagnose(trace: dict[str, Any]) -> dict[str, Any]:
    outcome = label_first_touch(trace)
    for key in ("ball_linear_velocity_before_step_m_s", "ball_position_before_step_m"):
        array = np.asarray(trace.get(key))
        if array.shape != (300, 1, 3) or array.dtype != np.float64 or not np.isfinite(array).all():
            raise ValueError("complete measured finite ball velocity/position required")
    first = outcome.first_foot_contact_frame
    if first is None:
        return dict(outcome=outcome.to_dict(), windows=[], no_foot_contact=True)
    force = trace["force_n"][:, 0]
    position = trace["ball_position_after_step_m"][:, 0]
    velocity = trace["ball_linear_velocity_before_step_m_s"][:, 0]
    windows = []
    for length in (1, 5, 15, 30, 60):
        if first + length >= 300:
            continue
        displacement = position[first + length] - position[first]
        active = np.any(force[first + 1 : first + length + 1, :2] > 1, axis=1)
        windows.append(
            dict(
                frames_after_first_contact=length,
                displacement_m=displacement.tolist(),
                forward_displacement_m=float(displacement[0]),
                lateral_displacement_m=float(displacement[1]),
                absolute_lateral_over_forward=float(
                    abs(displacement[1]) / max(float(displacement[0]), 0.01)
                ),
                later_foot_contact_frames=int(np.count_nonzero(active)),
                later_left_foot_frames=int(
                    np.count_nonzero(force[first + 1 : first + length + 1, 0] > 1)
                ),
                later_right_foot_frames=int(
                    np.count_nonzero(force[first + 1 : first + length + 1, 1] > 1)
                ),
            )
        )
    later = np.flatnonzero(np.any(force[first + 1 :, :2] > 1, axis=1)) + first + 1
    return dict(
        outcome=outcome.to_dict(),
        first_contact_left_foot=bool(force[first, 0] > 1),
        first_contact_right_foot=bool(force[first, 1] > 1),
        velocity_before_first_frame_m_s=velocity[first].tolist(),
        velocity_before_next_frame_m_s=velocity[first + 1].tolist() if first < 299 else None,
        measured_first_frame_net_velocity_change_m_s=(
            velocity[first + 1] - velocity[first]
        ).tolist()
        if first < 299
        else None,
        net_velocity_change_is_not_isolated_foot_impulse=True,
        first_contact_frame_has_no_following_velocity_sample=first == 299,
        later_foot_contact_frames=later.tolist(),
        windows=windows,
        frame_duration_s=0.02,
        new_physics_steps=0,
        optimizer_updates=0,
        causal_intervention_performed=False,
        hardware_authorized=False,
    )
