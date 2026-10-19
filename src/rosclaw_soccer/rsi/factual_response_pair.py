"""Extract a measured response pair after caller-owned physical authentication."""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.foot_heading_parity import original_arrays
from rosclaw_soccer.rsi.observed_ball_response import observation
from rosclaw_soccer.rsi.postcontact_release_control import ACTION_NAMES


def wrench_wish(name: str) -> np.ndarray:
    if type(name) is not str or name not in ACTION_NAMES:
        raise ValueError("one original named physical action required")
    result = np.zeros(4, dtype=np.float64)
    if name == "baseline":
        return result
    if name.startswith("lateral-") or name.startswith("lateral+"):
        lateral, yaw = name.split("_")
        result[1], result[3] = float(lateral[7:]), float(yaw[3:])
    else:
        axis, sign = name.split("_")
        result[("forward", "lateral", "lift", "yaw").index(axis)] = 1.0 if sign == "plus" else -1.0
    return result


def extract_pair(
    baseline: dict[str, np.ndarray],
    action: dict[str, np.ndarray],
    *,
    name: str,
    trigger_frame: int | None,
) -> dict[str, Any]:
    """Actual80ms labels only; never fill no-contact/zero aliases with zero effects.

    This checks tensor ABI and exact matched prefix, not source integrity or
    native physics. Caller authenticates the ordered cases, policy/noise,
    source hashes, literal action attribution and physical reexecution first.
    All original task outcomes (including skipped labels) must remain upstream.
    """
    wish = wrench_wish(name)
    original_arrays(baseline)
    original_arrays({k: v for k, v in action.items() if not k.startswith("postcontact_release_")})
    if set(action) != set(baseline) | {
        "postcontact_release_requested_offset_nm",
        "postcontact_release_applied_offset_nm",
        "postcontact_release_forecast_steps",
    }:
        raise ValueError("exact named-release trace and unchanged original baseline ABI required")
    request = action["postcontact_release_requested_offset_nm"]
    if (
        request.shape != (300, 1, 29)
        or request.dtype != np.float64
        or not np.isfinite(request).all()
    ):
        raise ValueError("complete actual requested action tensor required")
    if trigger_frame is None or type(trigger_frame) is int and not 31 <= trigger_frame <= 295:
        return dict(status="SKIPPED", reason="NO_OBSERVED_80MS_DECISION_TAIL")
    if type(trigger_frame) is not int:
        raise ValueError("integer actual trigger or explicit absence required")
    f = trigger_frame
    for key in baseline:
        if action[key][:f].tobytes() != baseline[key][:f].tobytes():
            raise ValueError("whole original before-action prefix must match bitwise")
    post_action_fields = {
        "pd_requested_torque_nm",
        "torque_nm",
        "actual_actuator_force_nm",
        "pelvis_z_per_substep_m",
        "force_n",
        "ball_position_after_step_m",
    }
    for key in set(baseline) - post_action_fields:
        if action[key][f].tobytes() != baseline[key][f].tobytes():
            raise ValueError("whole current before-action state and controller must match bitwise")
    if not np.any(wish) or not np.any(request[f, 0]):
        return dict(status="SKIPPED", reason="ACTUAL_ZERO_BASELINE_ALIAS_OR_CANCELLATION")
    previous = action["force_n"][f - 1, 0]
    side = int(np.argmax(previous[:2]))
    states = []
    labels = []
    for trace, coefficients in ((baseline, np.zeros(4)), (action, wish)):
        states.append(
            observation(
                frame=f,
                side=side,
                foot_xyz=trace["foot_geometry_position_before_step_m"][f, 0, side].copy(),
                previous_foot_xyz=trace["foot_geometry_position_before_step_m"][
                    f - 1, 0, side
                ].copy(),
                ball_xyz=trace["ball_position_before_step_m"][f, 0].copy(),
                ball_velocity=trace["ball_linear_velocity_before_step_m_s"][f, 0].copy(),
                leg_joint_angles=trace["joint_position_rad"][f, 0, 6 * side : 6 * side + 6].copy(),
                root_angular_velocity=trace["root_velocity_world"][f, 0, 3:].copy(),
                previous_forces=trace["force_n"][f - 1, 0].copy(),
                requested_wrench_coefficients=coefficients,
            )
        )
        labels.append(
            (
                trace["ball_linear_velocity_before_step_m_s"][f + 4, 0]
                - trace["ball_linear_velocity_before_step_m_s"][f, 0]
            )
            * np.array([1.0, 1.0 if side == 0 else -1.0, 1.0])
        )
    if not np.array_equal(states[0][:12], states[1][:12]) or states[0][16] != states[1][16]:
        raise ValueError("same current partial state/clock required before applying the wish")
    return dict(
        status="FACTUAL_PAIR",
        baseline_features=states[0],
        action_features=states[1],
        baseline_delta_velocity=labels[0],
        action_delta_velocity=labels[1],
        measured_side=side,
        source_and_physics_authenticated_here=False,
    )
