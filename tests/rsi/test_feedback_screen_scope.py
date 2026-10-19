"""Fixed screen budgets and action-bank logic, not G1 task improvement."""

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.rsi_feedback_screen_inputs import best_proxy_index, branch_bank, require_scope
from tests.rsi.test_impact_torque_screen import sample


def scope():
    return dict(
        mode="SIM_ONLY",
        profile="six-feedback-impact-screen-v1",
        frame_selection="RETROSPECTIVE_SOURCE_CONTACT_FRAME_DIAGNOSTIC_ONLY",
        repeat_zero_after_candidates=True,
        source_cases=6,
        maximum_candidate_actions_per_case=13,
        horizon_substeps=20,
        recorded_replay_substep_budget=18000,
        feedback_forecast_substep_budget=1680,
        combined_substep_budget=19680,
        maximum_planner_run_calls=60,
        current_encoder_decoder_pairs=6,
        maximum_branch_encoder_decoder_pairs=84,
        original_residual_decisions=1800,
        original_recurrent_updates=1620,
        new_closed_loop_policy_trials=0,
        new_optimizer_updates=0,
        hardware_authorized=False,
        promotion_authorized=False,
        private_fresh_accessed=False,
        training_authorized=False,
        teacher_qualified=False,
        actor_future_event_input_allowed=False,
    )


def test_exact_scope():
    require_scope(scope())


@pytest.mark.parametrize("key", list(scope()))
def test_no_missing_or_widened_scope(key):
    cfg = scope()
    del cfg[key]
    with pytest.raises(ValueError):
        require_scope(cfg)
    cfg = scope()
    value = cfg[key]
    cfg[key] = int(value) if type(value) is bool else float(value) if type(value) is int else "REAL"
    with pytest.raises(ValueError):
        require_scope(cfg)


@pytest.mark.parametrize(
    "ik",
    [
        SimpleNamespace(side=-1, contact=None),
        SimpleNamespace(side=0, contact=30),
        SimpleNamespace(side=True, contact=None),
    ],
)
def test_ineligible_case_retained_as_baseline_repeat_without_geometry_allocation(monkeypatch, ik):
    def forbidden(*a, **k):
        pytest.fail("no geometry allocation without a current precontact swing foot")

    monkeypatch.setattr("scripts.rsi_feedback_screen_inputs.G1JointTaskKinematicSampler", forbidden)
    bank, eligible = branch_bank(None, None, ik, "unused")
    assert not eligible and [n for n, _ in bank] == ["baseline", "repeat_zero"]
    assert all(np.all(a == 0) for _, a in bank)


@pytest.mark.parametrize("side", [0, 1])
def test_current_jacobian_bank_stays_on_causal_selected_leg_with_zero_repeat(monkeypatch, side):
    measured = sample()

    class Sampler:
        def __init__(self, model, *, expected_world_hash):
            assert model == "model" and expected_world_hash == "world"

        def sample(self, data):
            assert data == "current"
            return measured

    monkeypatch.setattr("scripts.rsi_feedback_screen_inputs.G1JointTaskKinematicSampler", Sampler)
    bank, eligible = branch_bank(
        "model", "current", SimpleNamespace(side=side, contact=None), "world"
    )
    assert eligible and 2 <= len(bank) <= 14
    assert bank[0][0] == "baseline" and bank[-1][0] == "repeat_zero"
    for _, a in bank:
        allowed = np.zeros(29, dtype=bool)
        allowed[6 * side : 6 * side + 6] = True
        assert np.all(a[~allowed] == 0)
    assert np.all(bank[0][1] == 0) and np.all(bank[-1][1] == 0)


def test_best_proxy_is_stable_baseline_first_and_excludes_repeat_and_vetoes():
    rows = [
        dict(name="baseline", score=dict(feasible_predicted_impact=True, proxy_cost=0.1)),
        dict(name="candidate", score=dict(feasible_predicted_impact=True, proxy_cost=0.1)),
        dict(name="veto", score=dict(feasible_predicted_impact=False, proxy_cost=None)),
        dict(name="repeat_zero", score=dict(feasible_predicted_impact=True, proxy_cost=-1)),
    ]
    assert best_proxy_index(rows) == 0
    rows[1]["score"]["proxy_cost"] = 0.01
    assert best_proxy_index(rows) == 1
    for row in rows:
        row["score"]["feasible_predicted_impact"] = False
    assert best_proxy_index(rows) is None
