"""Synthetic command capture audits only, not real physical effects."""

import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.bounded_first_touch_control import BoundedFirstTouchControl
from rosclaw_soccer.rsi.navigation_recovery_parity import review_navigation_recovery
from rosclaw_soccer.rsi.original_ik_planner_pulse import OriginalIkPlannerPulse
from tests.rsi.test_full_task_zero_readback import fixture


def command_capture(action):
    _, _, data = fixture()
    trace = data["main"]
    profile, pulse = BoundedFirstTouchControl(), OriginalIkPlannerPulse(0.08)
    for f in range(300):
        raw, projected = profile.navigation(
            gap_x=0.0, gap_y=0.0, ball_vx=0.0, before_contact=f < 186
        )
        if f < 186:
            raw = (1.4, *raw[1:])
            projected = (1.4, *projected[1:])
        if f >= 186 and action != "baseline":
            cap = 0.7 if action == "nav_brake_0_7" else 0.0
            raw = (min(raw[0], cap), *raw[1:])
            projected = (min(projected[0], cap), *projected[1:])
        residual, command = pulse.compose(f, projected)
        for k, v in [
            ("approach_requested_navigation_command", raw),
            ("ik_planner_pulse_baseline_command", projected),
            ("ik_planner_pulse_requested_residual", residual),
            ("ik_planner_pulse_applied_command", command),
            ("navigation_command", command),
        ]:
            trace[k][f, 0] = np.asarray(v)
    return trace


@pytest.mark.parametrize("action", ["baseline", "nav_brake_0_7", "nav_brake_0"])
def test_every_command_rebuilt_with_same_measured_prefix(action):
    result = review_navigation_recovery(
        command_capture(action), command_capture("baseline"), action_name=action
    )
    assert result["start_frame"] == 186 and result["new_physical_steps"] == 0


@pytest.mark.parametrize(
    "change", ["prefix", "birth", "yaw", "request", "missing", "baseline_future"]
)
def test_wrong_actuation_or_hidden_prefix_change_rejected(change):
    base = command_capture("baseline")
    trace = command_capture("nav_brake_0_7")
    if change == "prefix":
        trace["joint_target_rad"][30, 0, 0] = 1
    elif change == "birth":
        trace["canonical_qpos"][186, 0, 0] = 1
    elif change == "yaw":
        trace["navigation_command"][250, 0, 2] = 0.1
    elif change == "request":
        trace["approach_requested_navigation_command"][250, 0, 0] = 1.4
    elif change == "missing":
        del trace["force_n"]
    elif change == "baseline_future":
        trace = copy.deepcopy(base)
        trace["joint_target_rad"][250, 0, 0] = 1
    with pytest.raises(ValueError):
        review_navigation_recovery(
            trace, base, action_name="baseline" if change == "baseline_future" else "nav_brake_0_7"
        )
