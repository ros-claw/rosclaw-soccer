"""Diagnostic contracts only; these tests do not constitute G1 task improvement."""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.physics.g1_pd_torque_forecast import G1PDTorqueForecast
from scripts.rsi_impact_screen_inputs import require_scope
from scripts.rsi_impact_torque_authority_screen import equal, repeat_equal


def scope():
    return dict(
        mode="SIM_ONLY",
        profile="retrospective-impact-torque-six-v1",
        frame_selection="RETROSPECTIVE_FIRST_FOOT_CONTACT_CONTROL_FRAME",
        leg_eligibility="ORIGINAL_CAUSAL_SWING_EQUALS_FIRST_CONTACT_FOOT",
        repeat_zero_after_candidates=True,
        source_cases=6,
        maximum_candidate_actions_per_case=13,
        substeps_per_forecast=10,
        recorded_replay_substep_budget=18000,
        counterfactual_substep_budget=840,
        combined_substep_budget=18840,
        new_closed_loop_policy_trials=0,
        new_neural_inferences=0,
        new_optimizer_updates=0,
        hardware_authorized=False,
        promotion_authorized=False,
        private_fresh_accessed=False,
        training_authorized=False,
        teacher_qualified=False,
        actor_future_event_input_allowed=False,
    )


def test_exact_retrospective_scope():
    require_scope(scope())


@pytest.mark.parametrize("key", list(scope()))
def test_scope_cannot_be_partial(key):
    cfg = scope()
    del cfg[key]
    with pytest.raises(ValueError):
        require_scope(cfg)


@pytest.mark.parametrize("key", list(scope()))
def test_scope_cannot_widen_or_coerce_truthiness(key):
    cfg = scope()
    original = cfg[key]
    cfg[key] = (
        int(original)
        if type(original) is bool
        else float(original)
        if type(original) is int
        else "REAL"
    )
    with pytest.raises(ValueError):
        require_scope(cfg)


def make_forecast():
    values = {f.name: np.zeros((10, 3)) for f in dataclasses.fields(G1PDTorqueForecast)}
    values.update(donor_integration_state_hash="synthetic", nonfoot_robot_contact_over_1n=False)
    return G1PDTorqueForecast(**values)


def test_repeat_all_fields_must_be_bit_exact():
    a, b = make_forecast(), make_forecast()
    repeat_equal(a, b)
    b.canonical_qpos_after_substep[0, 0] = -0.0
    with pytest.raises(ValueError, match="contaminated"):
        repeat_equal(a, b)


def test_replay_parity_uses_original_tolerance_and_rejects_nan():
    equal(np.zeros(2), np.full(2, 1e-9), "synthetic")
    for bad in (np.nan, np.inf, 1e-7):
        with pytest.raises(ValueError, match="differs"):
            equal(np.zeros(2), np.full(2, bad), "synthetic")


def test_no_online_actor_or_neural_allocation_in_screen():
    source = Path("scripts/rsi_impact_torque_authority_screen.py").read_text()
    for prohibited in ("InferenceSession", "torch.", "request_action", "Runtime(", "mj_resetData"):
        assert prohibited not in source
    assert "prepare(args.config)" in source
    assert source.index("if args.preflight:") < source.index("MjModel.from_binary_path")
