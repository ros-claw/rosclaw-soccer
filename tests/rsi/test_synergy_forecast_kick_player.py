from types import SimpleNamespace

import numpy as np
import pytest
from test_predicted_touch_window import prediction

import rosclaw_soccer.rsi.synergy_forecast_kick_player as synergy
import rosclaw_soccer.rsi.synergy_response_model as response_model
from scripts.rsi_review_motion_timing_teacher import (
    verify_stable_plastic_search,
    verify_synergy_search,
)


@pytest.mark.parametrize(
    "mode,expected_calls",
    [
        ("parent", 1),
        ("rescue", 37),
        ("fail", 37),
        ("response", 38),
        ("long", 1),
        ("stable", 37),
        ("plastic", 73),
        ("stable-parent", 1),
    ],
)
def test_cem_is_bounded_repeatable_and_adopts_only_qualified_reference(
    monkeypatch, mode, expected_calls
):
    p = object.__new__(synergy.SynergyForecastKickPlayer)
    p.world = SimpleNamespace(_require_current_boundary=lambda: None, control_steps=100)
    p.policy = SimpleNamespace(time_step=20, WARMUP_STEPS=0, motion_total_steps=100)
    p.pending, p.measured_entry_receipt = None, "actual"
    p.layout = SimpleNamespace(bq=0, rq=7)
    p.decided = p.plan_applied = False
    p.selected_prediction = None
    p.events, p.candidate_predictions = [], []
    p.basis_isaac = tuple(tuple(float(i == j) for j in range(29)) for i in range(6))
    p.basis_hg = np.eye(6, 29, dtype=np.float64)
    p.synergy_response_enabled = mode == "response"
    p.forecast_controls = 128 if mode in ("long", "stable", "plastic", "stable-parent") else 96
    if mode in ("stable", "plastic", "stable-parent"):
        p.plastic_basis_isaac = tuple(tuple(float(i + 6 == j) for j in range(29)) for i in range(6))
    calls, installed = [], []

    def forecast(*, controls, before_proposal, reference_plan):
        assert controls == p.forecast_controls and before_proposal
        calls.append(reference_plan)
        r = prediction(n=controls)
        r["donor_control_frame"] = 100
        passes = mode in ("parent", "long", "stable-parent") or (
            mode in ("rescue", "stable")
            and reference_plan is not None
            and reference_plan.coefficients[0][0] > 0
        )
        passes = passes or (mode == "response" and len(calls) == 38)
        passes = passes or (mode == "plastic" and len(calls) > 37)
        if not passes:
            r["qpos"][:, 0] *= 0.1
        return r

    def install(policy, plan):
        installed.append(plan)
        return {"applied": True}

    p.forecaster = SimpleNamespace(forecast=forecast)
    monkeypatch.setattr(synergy, "condition_synergy_reference", install)
    monkeypatch.setattr(
        response_model,
        "propose_synergy_response",
        lambda basis, samples: (np.full((2, 6), 0.001), {"physics_verified": False}),
    )
    monkeypatch.setattr(synergy.SharedRoboNaldoPlayer, "propose", lambda self: (np.zeros(29), {}))
    p.propose()
    assert len(calls) == expected_calls
    event = p.events[0]
    audit_choice = verify_synergy_search(
        event["candidate_windows"][:37], [0.0] * min(37, expected_calls), event["generations"]
    )
    if mode == "response":
        assert audit_choice == 0 and event["selected"] == "native-synergy-response"
    elif mode == "plastic":
        assert audit_choice == 0 and event["selected"].startswith("plastic-")
        assert installed[0].basis_isaac == p.plastic_basis_isaac
    else:
        assert event["candidate_windows"][audit_choice]["name"] == event["selected"]
    assert bool(installed) == (mode in ("rescue", "response", "stable", "plastic"))
    if mode in ("stable", "plastic", "stable-parent"):
        rebuilt = verify_stable_plastic_search(
            event["candidate_windows"],
            [0.0] * expected_calls,
            event["generations"],
            event["plastic_generations"],
        )
        assert event["candidate_windows"][rebuilt]["name"] == event["selected"]
    if installed:
        assert installed[0] in calls
        assert p.events[0]["selected_window"]["window_pass"]
    if mode not in ("parent", "long", "stable-parent"):
        assert len(p.events[0]["generations"]) == 3
        rng = np.random.default_rng(7001)
        noise = rng.normal(size=(6, 2, 6))
        first_generation = np.clip(np.concatenate((noise, -noise)) * 0.025, -0.06, 0.06)
        for call, expected in zip(calls[1:13], first_generation, strict=True):
            np.testing.assert_array_equal(np.array(call.coefficients), expected)
    p.propose()
    assert len(calls) == expected_calls


def test_independent_audit_rejects_incomplete_search_and_forged_parent_plan():
    window = dict(name="parent", coefficients=None, outcome={"window_pass": False})
    with pytest.raises(ValueError, match="allocation"):
        verify_synergy_search([window], [0.0], [])
    with pytest.raises(ValueError, match="parent"):
        verify_synergy_search([dict(window, coefficients=[[0.0]])], [0.0], [])


def test_unsafe_high_travel_never_outranks_clean_complete_qualified_window():
    good = prediction(n=96)
    safe = dict(
        window_pass=True,
        predicted_body_healthy=True,
        predicted_nonfoot_clean=True,
        quality_observed=True,
        reward=3.0,
    )
    bad = dict(safe, window_pass=False, predicted_body_healthy=False, reward=100.0)
    assert synergy.synergy_search_rank(good, safe) > synergy.synergy_search_rank(good, bad)
