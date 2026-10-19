"""Independent raw80ms branch validation on synthetic data, no new physics."""

import numpy as np
import pytest

from rosclaw_soccer.rsi.early_feedback_impact_score import early_feedback_impact_score
from scripts.rsi_read_early_feedback_screen_arrays import verify_early_prediction
from scripts.rsi_read_feedback_screen import require_totals
from tests.rsi.test_early_feedback_impact_score import fixture as prediction
from tests.rsi.test_early_feedback_readback import fixture as raw
from tests.rsi.test_feedback_screen_readback import totals as old_totals


def packet(side=0, first=29):
    a, _, _ = raw()
    offset = np.zeros(29)
    offset[6 * side] = 2
    forecast, _ = prediction(first=first, offset=offset)
    a.update(forecast.arrays)
    a["applied_control_nm"][:30] += offset
    a["actual_actuator_force_nm"][:] = a["applied_control_nm"]
    for j in range(1, 4):
        a[f"next_{j}_canonical_qvel"] = a["canonical_qvel_after_substep"][j * 10 - 1].copy()
    metadata = dict(
        offset_nm=offset.tolist(),
        is_zero=False,
        donor_hash=forecast.donor_integration_state_hash,
        score=early_feedback_impact_score(forecast, offset),
    )
    return a, metadata


@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize("first", [None, 29, 31, 39])
def test_all_vetoed_and_complete_branches_are_checked_not_dropped(side, first):
    a, w = packet(side, first)
    assert verify_early_prediction(a, w, side) == w["score"]


@pytest.mark.parametrize("j", [1, 2, 3])
@pytest.mark.parametrize("bad", ["body", "target", "neural"])
def test_all_three_actual_feedback_updates_are_required(j, bad):
    a, w = packet()
    if bad == "body":
        a[f"next_{j}_canonical_qpos"][0] = 1
    elif bad == "target":
        a[f"next_{j}_joint_target_rad"][0] = 1
    else:
        a[f"neural_next_{j}_decoder_input"][0, 0] = np.nan
    with pytest.raises(ValueError):
        verify_early_prediction(a, w, 0)


@pytest.mark.parametrize(
    "bad",
    ["late_action", "lost_action", "action_schedule", "actuator", "support_leg", "score", "zero"],
)
def test_action_duration_actual_force_and_ranking_cannot_be_forged(bad):
    a, w = packet()
    if bad == "late_action":
        a["applied_control_nm"][30, 0] = 2
    elif bad == "lost_action":
        a["applied_control_nm"][20, 0] = 0
    elif bad == "action_schedule":
        a["requested_offset_nm"][20, 0] = 1
    elif bad == "actuator":
        a["actual_actuator_force_nm"][0, 0] = 1
    elif bad == "support_leg":
        a["requested_offset_nm"][:30, 6] = w["offset_nm"][6] = 1
    elif bad == "score":
        w["score"]["proxy_cost"] = -1
    else:
        w["is_zero"] = True
    with pytest.raises(ValueError):
        verify_early_prediction(a, w, 0)


def test_whole80ms_counts_include_all_three_neural_calls():
    d = old_totals()
    d.update(
        forecast_attempted_steps=3360,
        forecast_integrated_steps=3360,
        combined_attempted_steps=21360,
        combined_integrated_steps=21360,
        branch_target_attempts=252,
        branch_target_completed=252,
    )
    require_totals(d, 84, 72, 3, early80=True)
    for key in d:
        changed = dict(d)
        changed[key] += 1
        with pytest.raises(ValueError):
            require_totals(changed, 84, 72, 3, early80=True)
