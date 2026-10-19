"""Raw postcontact exit causality/actuator attribution, not search qualification."""

import numpy as np

from rosclaw_soccer.rsi.foot_heading_parity import original_arrays
from rosclaw_soccer.rsi.impact_torque_screen import CAPS
from rosclaw_soccer.rsi.postcontact_release_control import ACTION_NAMES
from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS


def check_exit_trace(trace, review):
    keys = {
        "postcontact_exit_requested_offset_nm",
        "postcontact_exit_applied_offset_nm",
        "postcontact_exit_forecast_steps",
    }
    if {k for k in trace if k.startswith("postcontact_exit_")} != keys:
        raise ValueError("three actual distinct postcontact exit fields required")
    original_arrays({k: v for k, v in trace.items() if k not in keys})
    for key, shape, dtype in (
        ("postcontact_exit_requested_offset_nm", (300, 1, 29), np.float64),
        ("postcontact_exit_applied_offset_nm", (300, 1, 10, 29), np.float64),
        ("postcontact_exit_forecast_steps", (300, 1, 1), np.int64),
    ):
        a = trace[key]
        if (
            type(a) is not np.ndarray
            or a.shape != shape
            or a.dtype != dtype
            or not np.isfinite(a).all()
        ):
            raise ValueError("complete typed finite exit physical capture required")
    r = review["postcontact_exit"]
    if np.any(trace["force_n"] < 0) or r["selected_name"] not in (None, *ACTION_NAMES):
        raise ValueError("nonnegative measured forces and declared selected exit action required")
    trigger = side = None
    cancelled = False
    allowed = np.zeros((300, 29), dtype=bool)
    for f in range(300):
        prior = trace["force_n"][f - 1, 0] if f else np.zeros(6)
        if trace["canonical_qpos"][f, 0, 2] < 0.65 or np.any(prior[2:] > 1):
            cancelled = True
        if not cancelled and trigger is None and f >= 30 and np.any(prior[:2] > 1):
            trigger, side = f, int(np.argmax(prior[:2]))
        if (
            not cancelled
            and trigger is not None
            and trigger <= 294
            and trigger // 20 == (trigger + 3) // 20
            and f < trigger + 3
        ):
            allowed[f, 6 * side : 6 * side + 6] = True
    if (
        r["trigger_frame"] != trigger
        or r["measured_contact_side"] != side
        or r["cancelled"] is not cancelled
        or r["maximum_forecast_steps"] != 560
    ):
        raise ValueError("exit trigger/cancellation must derive from actual own past forces/state")
    counts = trace["postcontact_exit_forecast_steps"][:, 0, 0]
    active = np.flatnonzero(counts)
    expected_search = (
        trigger is not None and trigger <= 294 and trigger // 20 == (trigger + 3) // 20
    )
    if (
        np.any(counts < 0)
        or np.any(counts % 40)
        or counts.sum() > 560
        or (
            expected_search
            and (not np.array_equal(active, [trigger]) or not 80 <= counts[trigger] <= 560)
        )
        or (not expected_search and len(active))
        or int(counts.sum()) != review["forecast_integrated_steps"]
        or r["forecast_integrated_steps"] != review["forecast_integrated_steps"]
        or review["forecast_attempted_steps"] != review["forecast_integrated_steps"]
        or (expected_search != (r["selected_name"] is not None))
    ):
        raise ValueError("one bounded actual current-pose exit candidate bank required")
    action = trace["postcontact_exit_requested_offset_nm"][:, 0]
    if np.any(abs(action) > CAPS) or np.any(action[~allowed]):
        raise ValueError("exit action exceeds measured contacting-leg lease/caps")
    frames = np.flatnonzero(np.any(action, axis=1))
    if len(frames) and (
        r["selected_name"] in (None, "baseline", "repeat_zero")
        or not np.array_equal(frames, np.flatnonzero(np.any(allowed, axis=1)))
        or not np.array_equal(action[frames], np.tile(action[frames[0]], (len(frames), 1)))
    ):
        raise ValueError("constant selected exit action required for the uncancelled lease")
    if expected_search and r["selected_name"] not in ("baseline", None) and not len(frames):
        raise ValueError("selected nonzero exit action missing from actual requested control")
    limits = np.asarray(G1_HARD_TORQUE_LIMITS)
    raw = trace["pd_requested_torque_nm"]
    torque = np.clip(raw + action[:, None, None, :], -limits, limits)
    applied = torque - np.clip(raw, -limits, limits)
    for k, expected in (
        ("torque_nm", torque),
        ("actual_actuator_force_nm", torque),
        ("postcontact_exit_applied_offset_nm", applied),
    ):
        if not np.array_equal(trace[k], expected):
            raise ValueError("actual exit torques differ from clipped PD plus named correction")
    actual = np.flatnonzero(np.any(applied, axis=(1, 2, 3)))
    return dict(
        observed_trigger_frame=trigger,
        measured_contact_side=side,
        cancelled=cancelled,
        selected_name=r["selected_name"],
        first_physically_applied_correction_frame=None if not len(actual) else int(actual[0]),
        nonzero_requested_frames=len(frames),
        forecast_integrated_steps=int(counts.sum()),
        all_controls_attributed_to_clipped_pd_plus_declared_exit=True,
        independent_search_qualified=False,
    )
