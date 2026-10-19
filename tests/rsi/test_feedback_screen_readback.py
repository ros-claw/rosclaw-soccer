"""Synthetic branch verification, not new physical or policy success evidence."""

import copy

import numpy as np
import pytest

from rosclaw_soccer.physics.g1_feedback_forecast import G1FeedbackForecast
from rosclaw_soccer.rsi.feedback_impact_score import feedback_impact_score
from scripts.rsi_read_feedback_screen import regenerate_bank, require_totals, verify_prediction
from tests.rsi.test_feedback_impact_score import prediction
from tests.rsi.test_zero_feedback_readback import fixture


def packet(side=0, first=9):
    a, _, _ = fixture()
    a.update(prediction(first).arrays)
    offset = np.zeros(29)
    offset[6 * side] = 2
    a["offset_nm"] = offset
    a["applied_control_nm"][:10] += offset
    a["actual_actuator_force_nm"][:] = a["applied_control_nm"]
    a["next_canonical_qvel"] = a["canonical_qvel_after_substep"][9].copy()
    f = G1FeedbackForecast("synthetic", a["end_integration_state"], a, {})
    metadata = dict(
        offset_nm=offset.tolist(),
        is_zero=False,
        donor_hash="synthetic",
        score=feedback_impact_score(f, offset),
    )
    return a, metadata


@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize("first", [None, 9, 11])
def test_complete_censored_and_nofoot_predictions_all_retained(side, first):
    a, w = packet(side, first)
    result = verify_prediction(a, w, side)
    assert result == w["score"]
    assert result["feasible_predicted_impact"] is (first == 9)


@pytest.mark.parametrize(
    "bad",
    [
        "score",
        "metadata_offset",
        "offset_cap",
        "support_leg",
        "upper_body",
        "wrong_side",
        "extend_offset",
        "lost_offset",
        "actuator",
        "next_body",
        "next_target",
        "neural",
        "end",
        "zero_flag",
    ],
)
def test_every_action_and_feedback_stage_rejects_tampering(bad):
    a, w = packet()
    side = 0
    if bad == "score":
        w["score"]["proxy_cost"] = -1
    elif bad == "metadata_offset":
        w["offset_nm"][0] = 1
    elif bad == "offset_cap":
        a["offset_nm"][0] = w["offset_nm"][0] = 100
    elif bad in ("support_leg", "upper_body"):
        index = 6 if bad == "support_leg" else 15
        a["offset_nm"][index] = w["offset_nm"][index] = 1
    elif bad == "wrong_side":
        side = 1
    elif bad == "extend_offset":
        a["applied_control_nm"][10, 0] = 2
    elif bad == "lost_offset":
        a["applied_control_nm"][0, 0] = 0
    elif bad == "actuator":
        a["actual_actuator_force_nm"][0, 0] = 1
    elif bad == "next_body":
        a["next_canonical_qvel"][0] = 1
    elif bad == "next_target":
        a["next_joint_target_rad"][0] = 1
    elif bad == "neural":
        a["neural_next_encoder_features"][0, 0] = np.nan
    elif bad == "end":
        a["end_integration_state"][0] = np.inf
    else:
        w["is_zero"] = True
    with pytest.raises(ValueError):
        verify_prediction(a, w, side)


def totals():
    return dict(
        recorded_attempted_steps=18000,
        recorded_integrated_steps=18000,
        forecast_attempted_steps=1680,
        forecast_integrated_steps=1680,
        combined_attempted_steps=19680,
        combined_integrated_steps=19680,
        planner_runs_attempted=60,
        current_pairs_completed=6,
        original_residual_decisions_completed=1800,
        original_recurrent_updates_completed=1620,
        branch_target_attempts=84,
        branch_target_completed=84,
        new_closed_loop_policy_trials=0,
        new_optimizer_updates=0,
        candidate_actions_optimized=72,
        local_proxy_improved_source_cases=3,
    )


def test_actual_raw_bank_sizes_bound_every_execution_counter():
    require_totals(totals(), 84, 72, 3)
    for key in totals():
        bad = copy.deepcopy(totals())
        bad[key] += 1
        with pytest.raises(ValueError):
            require_totals(bad, 84, 72, 3)
        bad[key] = float(totals()[key])
        with pytest.raises(ValueError):
            require_totals(bad, 84, 72, 3)


@pytest.mark.parametrize("count", [0, 11, 85])
def test_no_unbounded_or_incomplete_bank(count):
    with pytest.raises(ValueError):
        require_totals(totals(), count, 72, 3)


def test_regenerate_from_current_canonical_state_does_not_step_physics(monkeypatch):
    import mujoco

    from tests.physics.test_g1_feedback_forecast import world
    from tests.rsi.test_impact_torque_screen import sample

    model, _, _ = world()
    q = np.arange(43, dtype=np.float64) * 0.001
    v = np.arange(41, dtype=np.float64) * 0.002
    seen = []

    class Sampler:
        def sample(self, d):
            np.testing.assert_array_equal(d.qpos, q)
            np.testing.assert_array_equal(d.qvel, v)
            seen.append(True)
            return sample()

    def forbidden(*a, **k):
        pytest.fail("geometry readback must not integrate or refresh dynamics")

    monkeypatch.setattr(mujoco, "mj_step", forbidden)
    monkeypatch.setattr(mujoco, "mj_forward", forbidden)
    bank = regenerate_bank(model, Sampler(), q, v, 0, True)
    assert seen == [True] and bank[0][0] == "baseline" and bank[-1][0] == "repeat_zero"
    bank = regenerate_bank(None, None, None, None, -1, False)
    assert [n for n, _ in bank] == ["baseline", "repeat_zero"]


@pytest.mark.parametrize("bad", ["qshape", "vshape", "qdtype", "nan"])
def test_regeneration_rejects_malformed_current_body_before_allocation(bad):
    q, v = np.zeros(43), np.zeros(41)
    if bad == "qshape":
        q = q[:-1]
    elif bad == "vshape":
        v = v[:-1]
    elif bad == "qdtype":
        q = q.astype(np.float32)
    else:
        v[-1] = np.nan
    with pytest.raises(ValueError):
        regenerate_bank(None, None, q, v, 0, True)
