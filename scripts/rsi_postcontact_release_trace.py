"""Read-only causal release/actuator attribution checks, no data admission."""

import numpy as np

from rosclaw_soccer.rsi.impact_torque_screen import CAPS
from rosclaw_soccer.rsi.postcontact_release_control import ACTION_NAMES
from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS


def check_release_trace(trace, review, *, action_name):
    if action_name not in ACTION_NAMES:
        raise ValueError("declared release action required")
    if any(k.startswith(("early_impact_feedback_", "impact_feedback_")) for k in trace):
        raise ValueError("release study must not mix old impact profiles")
    for key, shape, dtype in (
        ("postcontact_release_requested_offset_nm", (300, 1, 29), np.float64),
        ("postcontact_release_applied_offset_nm", (300, 1, 10, 29), np.float64),
        ("postcontact_release_forecast_steps", (300, 1, 1), np.int64),
        ("pd_requested_torque_nm", (300, 1, 10, 29), np.float64),
        ("torque_nm", (300, 1, 10, 29), np.float64),
        ("actual_actuator_force_nm", (300, 1, 10, 29), np.float64),
        ("force_n", (300, 1, 6), np.float64),
        ("root_pose_xyzw_m", (300, 1, 7), np.float64),
    ):
        a = trace.get(key)
        if (
            not isinstance(a, np.ndarray)
            or a.shape != shape
            or a.dtype != dtype
            or not np.isfinite(a).all()
        ):
            raise ValueError("whole typed release/physical trace required: " + key)
    if np.any(trace["force_n"] < 0) or np.any(trace["postcontact_release_forecast_steps"]):
        raise ValueError("nonnegative forces and zero forecasts required")
    r = review["postcontact_release"]
    if (
        r["action_name"] != action_name
        or r["maximum_action_duration_frames"] != 3
        or r["selection_from_completed_contact_only"] is not True
        or r["future_source_trace_consumed"] is not False
    ):
        raise ValueError("same fixed causal release protocol required")
    trigger = side = None
    cancelled = False
    allowed = np.zeros((300, 29), dtype=bool)
    for f in range(300):
        previous = trace["force_n"][f - 1, 0] if f else np.zeros(6)
        if trace["root_pose_xyzw_m"][f, 0, 2] < 0.65 or np.any(previous[2:] > 1):
            cancelled = True
        if not cancelled and f >= 30 and trigger is None and np.any(previous[:2] > 1):
            trigger = f
            side = int(np.argmax(previous[:2]))
        if not cancelled and trigger is not None and f < trigger + 3:
            allowed[f, 6 * side : 6 * side + 6] = True
    if (
        r["trigger_frame"] != trigger
        or r["measured_contact_side"] != side
        or r["cancelled"] is not cancelled
        or (
            r["action_available"] is not None
            if trigger is None
            else type(r["action_available"]) is not bool
        )
    ):
        raise ValueError("trigger/side/cancellation must derive from actual past force bins")
    action = trace["postcontact_release_requested_offset_nm"][:, 0]
    if np.any(abs(action) > CAPS) or np.any(action[~allowed]):
        raise ValueError("only measured contacting leg within one bounded release lease allowed")
    requested = np.flatnonzero(np.any(action, axis=1))
    if (
        action_name != "baseline"
        and r["action_available"] is True
        and not np.array_equal(requested, np.flatnonzero(np.any(allowed, axis=1)))
    ):
        raise ValueError("all uncancelled periods of the available nonzero lease required")
    if len(requested) and (
        action_name == "baseline"
        or not r["action_available"]
        or trigger != int(requested[0])
        or len(requested) > 3
        or not np.array_equal(requested, np.arange(requested[0], requested[-1] + 1))
        or not np.array_equal(action[requested], np.tile(action[requested[0]], (len(requested), 1)))
    ):
        raise ValueError("one constant declared postcontact action required")
    if r["action_available"] is False and len(requested):
        raise ValueError("aliased action must remain zero")
    limits = np.asarray(G1_HARD_TORQUE_LIMITS)
    raw = trace["pd_requested_torque_nm"]
    torque = np.clip(raw + action[:, None, None, :], -limits, limits)
    applied = torque - np.clip(raw, -limits, limits)
    for key, expected in (
        ("torque_nm", torque),
        ("actual_actuator_force_nm", torque),
        ("postcontact_release_applied_offset_nm", applied),
    ):
        if not np.array_equal(trace[key], expected):
            raise ValueError("actual release control differs from clipped-PD attribution")
    actual = np.flatnonzero(np.any(applied != 0, axis=(1, 2, 3)))
    return dict(
        observed_trigger_frame=trigger,
        measured_contact_side=side,
        first_requested_correction_frame=None if not len(requested) else int(requested[0]),
        first_physically_applied_correction_frame=None if not len(actual) else int(actual[0]),
        nonzero_requested_frames=int(len(requested)),
        cancelled=cancelled,
        all_controls_attributed_to_clipped_pd_plus_declared_release=True,
    )
