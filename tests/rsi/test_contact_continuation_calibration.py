"""Calibration admission and bias arithmetic; no robot success claims."""

import copy

import numpy as np
import pytest

from scripts.rsi_calibrate_contact_continuation import BIAS_LIMITS, model_bias, require_scope


def scope():
    return dict(
        mode="SIM_ONLY",
        profile="whole-six-held-target40ms-bias-v1",
        frame_selection="RETROSPECTIVE_FIRST_FOOT_CONTACT_CONTROL_FRAME",
        prediction_input="CURRENT_STATE_AND_CURRENT_TARGET_ONLY",
        oracle_input="FUTURE_TARGETS_OFFLINE_VALIDATION_ONLY",
        bias_limits=dict(BIAS_LIMITS),
        source_cases=6,
        horizon_substeps=20,
        recorded_replay_substep_budget=18000,
        held_target_forecast_substep_budget=240,
        oracle_validation_substep_budget=120,
        combined_substep_budget=18360,
        new_closed_loop_policy_trials=0,
        new_neural_inferences=0,
        new_optimizer_updates=0,
        candidate_actions_optimized=0,
        hardware_authorized=False,
        promotion_authorized=False,
        private_fresh_accessed=False,
        training_authorized=False,
        teacher_qualified=False,
        actor_future_event_input_allowed=False,
    )


def arrays():
    return dict(
        canonical_qpos_after_substep=np.zeros((20, 43)),
        canonical_qvel_after_substep=np.zeros((20, 41)),
        pelvis_height_solver_epoch_m=np.full(20, 0.7),
        robot_force_on_ball_world_n=np.zeros((20, 3)),
    )


def test_exact_scope_and_zero_bias():
    require_scope(scope())
    assert model_bias(arrays(), arrays(), first_contact_substep=9)["frozen_limits_satisfied"]


@pytest.mark.parametrize("key", list(scope()))
def test_no_missing_scope_fields(key):
    cfg = scope()
    del cfg[key]
    with pytest.raises(ValueError):
        require_scope(cfg)


@pytest.mark.parametrize("key", list(scope()))
def test_no_widening_or_silent_type_coercion(key):
    cfg = scope()
    original = cfg[key]
    cfg[key] = (
        int(original)
        if type(original) is bool
        else float(original)
        if type(original) is int
        else {}
    )
    with pytest.raises(ValueError):
        require_scope(cfg)


def test_bias_preserves_full_contact_window_even_when_touch_at_last_substep():
    held, oracle = arrays(), arrays()
    oracle["robot_force_on_ball_world_n"][9:19, 0] = 10
    oracle["robot_force_on_ball_world_n"][19, 0] = 1000  # Outside the exact first10 contact bins.
    result = model_bias(held, oracle, first_contact_substep=9)
    assert result["oracle_robot_impulse_first10_contact_substeps_n_s"] == [0.2, 0, 0]
    assert result["errors"]["maximum_initial_contact_impulse_error_n_s"] == 0.2
    assert not result["frozen_limits_satisfied"]


@pytest.mark.parametrize(
    "field,slot,value",
    [
        ("canonical_qpos_after_substep", (15, 36), 0.011),
        ("canonical_qvel_after_substep", (15, 35), 0.11),
        ("canonical_qpos_after_substep", (15, 7), 0.051),
        ("pelvis_height_solver_epoch_m", 15, 0.711),
    ],
)
def test_every_prespecified_model_bias_guard_can_veto(field, slot, value):
    held, oracle = arrays(), arrays()
    before = copy.deepcopy(oracle)
    held[field][slot] = value
    assert not model_bias(held, oracle, first_contact_substep=0)["frozen_limits_satisfied"]
    for name in before:
        np.testing.assert_array_equal(oracle[name], before[name])


@pytest.mark.parametrize("slot", [-1, 10, True, 1.0])
def test_no_shifted_or_clipped_contact_window(slot):
    with pytest.raises(ValueError):
        model_bias(arrays(), arrays(), first_contact_substep=slot)


def test_nonfinite_prediction_not_accepted_as_small_bias():
    a = arrays()
    a["canonical_qpos_after_substep"][19, 36] = np.nan
    with pytest.raises(ValueError):
        model_bias(a, arrays(), first_contact_substep=0)
