"""Read-only current-state navigation/lease audit, no physics or NN execution.

Rebuild braking algebra directly instead of calling the candidate lease.
Recorded future arrays are audit inputs only, never controller observations.
"""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.bounded_first_touch_control import BoundedFirstTouchControl
from rosclaw_soccer.rsi.foot_heading_evidence import FIELDS, PREFIX
from rosclaw_soccer.rsi.foot_heading_parity import original_arrays
from rosclaw_soccer.rsi.original_ik_planner_pulse import OriginalIkPlannerPulse


def review_navigation_recovery(
    trace: dict[str, Any], baseline: dict[str, Any], *, action_name: str
) -> dict[str, Any]:
    if type(action_name) is not str or action_name not in (
        "baseline",
        "nav_brake_0_7",
        "nav_brake_0",
    ):
        raise ValueError("fixed declared navigation action required")
    stripped = original_arrays(trace)
    original_arrays(baseline)
    if (
        set(trace) != set(stripped) | {PREFIX + k for k in FIELDS}
        or trace.keys() != baseline.keys()
        or any(
            trace[k].shape != baseline[k].shape or trace[k].dtype != baseline[k].dtype
            for k in trace
        )
    ):
        raise ValueError("same complete body/control/NN layout required")
    foot = np.flatnonzero(np.any(trace["force_n"][:, 0, :2] > 1, axis=1))
    if not len(foot) or not 30 <= int(foot[0]) < 299:
        raise ValueError("actual completed measured-foot birth required")
    start = int(foot[0]) + 1
    for k, a in trace.items():
        if a[:start].tobytes() != baseline[k][:start].tobytes():
            raise ValueError("whole measured pre-lease prefix differs: " + k)
        if action_name == "baseline" and a.tobytes() != baseline[k].tobytes():
            raise ValueError("whole baseline must match original gold: " + k)
    # The first altered command acts on the same pre-action body. Its effects
    # on later body/NN states need not equal the baseline and are not hidden.
    for k in (
        "canonical_qpos",
        "canonical_qvel",
        "ball_position_before_step_m",
        "ball_linear_velocity_before_step_m_s",
    ):
        if trace[k][start].tobytes() != baseline[k][start].tobytes():
            raise ValueError("same current body at lease birth required: " + k)
    contact = False
    profile, pulse = BoundedFirstTouchControl(), OriginalIkPlannerPulse(0.08)
    for f in range(300):
        gap = trace["ball_position_before_step_m"][f, 0] - trace["root_pose_xyzw_m"][f, 0, :3]
        raw, projected = profile.navigation(
            gap_x=float(gap[0]),
            gap_y=float(gap[1]),
            ball_vx=float(trace["ball_linear_velocity_before_step_m_s"][f, 0, 0]),
            before_contact=not contact,
        )
        if not contact:
            raw = (1.4, *raw[1:])
            projected = (1.4, *projected[1:])
        if f >= start and action_name != "baseline":
            cap = 0.7 if action_name == "nav_brake_0_7" else 0.0
            raw = (min(raw[0], cap), *raw[1:])
            projected = (min(projected[0], cap), *projected[1:])
        residual, command = pulse.compose(f, projected)
        for key, expected in (
            ("approach_requested_navigation_command", raw),
            ("ik_planner_pulse_baseline_command", projected),
            ("ik_planner_pulse_requested_residual", residual),
            ("ik_planner_pulse_applied_command", command),
            ("navigation_command", command),
        ):
            if trace[key][f, 0].tobytes() != np.asarray(expected).tobytes():
                raise ValueError("independent current-state navigation algebra differs: " + key)
        contact = contact or bool(np.any(trace["force_n"][f] > 1))
    return dict(
        start_frame=start,
        action_name=action_name,
        complete_prelease43_arrays_match=True,
        first_changed_command_current_body_matches=True,
        all300_current_state_navigation_commands_rebuilt=True,
        new_physical_steps=0,
        new_nn_inferences=0,
        teacher_qualified=False,
        hardware_authorized=False,
    )
