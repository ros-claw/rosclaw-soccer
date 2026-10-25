from types import SimpleNamespace

import numpy as np
import pytest
from test_predicted_touch_window import prediction

import rosclaw_soccer.rsi.executed_synergy_kick_player as executed_controller
import rosclaw_soccer.rsi.joint_task_response as joint_response
import rosclaw_soccer.rsi.synergy_forecast_kick_player as synergy
import rosclaw_soccer.rsi.synergy_response_model as response_model
from rosclaw_soccer.rsi.aim_command_kick_player import AimCommandKickPlayer
from rosclaw_soccer.rsi.aim_goal_command import AimGoalCommand
from rosclaw_soccer.rsi.partial_memory_kick_player import PartialMemoryKickPlayer
from rosclaw_soccer.rsi.partial_motor_memory import decode_motor_options
from scripts.rsi_review_motion_timing_teacher import (
    verify_stable_plastic_search,
    verify_synergy_search,
)


@pytest.mark.parametrize("projection", ["whole_tick", "per_axis"])
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
        ("joint", 96),
        ("joint-retains", 37),
        ("sequential", 95),
        ("sequential-retains", 37),
        ("aim", 57),
        ("aim-retains", 37),
        ("executed", 73),
        ("executed-retains", 37),
        ("memory-aim", 39),
        ("memory-target", 39),
        ("memory-retains", 37),
    ],
)
def test_cem_is_bounded_repeatable_and_adopts_only_qualified_reference(
    monkeypatch, mode, expected_calls, projection
):
    p = object.__new__(
        PartialMemoryKickPlayer
        if mode.startswith("memory")
        else executed_controller.ExecutedSynergyKickPlayer
        if mode.startswith("executed")
        else AimCommandKickPlayer
        if mode.startswith("aim")
        else synergy.SynergyForecastKickPlayer
    )
    p.world = SimpleNamespace(_require_current_boundary=lambda: None, control_steps=100)
    p.policy = SimpleNamespace(time_step=20, WARMUP_STEPS=0, motion_total_steps=100)
    p.pending, p.measured_entry_receipt = None, "actual"
    p.layout = SimpleNamespace(bq=0, rq=7)
    p.decided = p.plan_applied = False
    p.selected_prediction = None
    p.events, p.candidate_predictions = [], []
    p.target_edits, p.executed_target_plan = [], None
    p.target_projection_mode = projection
    p.basis_isaac = tuple(tuple(float(i == j) for j in range(29)) for i in range(6))
    p.basis_hg = np.eye(6, 29, dtype=np.float64)
    p.memory_options = decode_motor_options(
        [
            dict(kind="aim_command", world_point=[8.0, -2.0, 0.11]),
            dict(
                kind="executed_target_synergy",
                coefficients=[[0.01] * 6, [-0.01] * 6],
                projection_mode="per_axis",
            ),
        ],
        p.basis_hg,
    )
    p.synergy_response_enabled = mode == "response"
    p.forecast_controls = (
        128
        if mode
        in (
            "long",
            "stable",
            "plastic",
            "stable-parent",
            "joint",
            "joint-retains",
            "sequential",
            "sequential-retains",
            "aim",
            "aim-retains",
            "executed",
            "executed-retains",
            "memory-aim",
            "memory-target",
            "memory-retains",
        )
        else 96
    )
    p.joint_task_response_enabled = mode in (
        "joint",
        "joint-retains",
        "sequential",
        "sequential-retains",
    )
    p.joint_response_iterations = 4 if mode.startswith("sequential") else 1
    p.hg_to_isaac = tuple(range(29))
    if mode in ("stable", "plastic", "stable-parent"):
        p.plastic_basis_isaac = tuple(tuple(float(i + 6 == j) for j in range(29)) for i in range(6))
    calls, installed = [], []

    def forecast(
        *, controls, before_proposal, reference_plan=None, aim_goal=None, target_plan=None
    ):
        assert controls == p.forecast_controls and before_proposal
        calls.append(
            target_plan
            if target_plan is not None
            else reference_plan
            if aim_goal is None
            else aim_goal
        )
        r = prediction(n=controls)
        r["donor_control_frame"] = 100
        passes = mode in ("parent", "long", "stable-parent") or (
            mode
            in (
                "rescue",
                "stable",
                "joint-retains",
                "sequential-retains",
                "aim-retains",
                "executed-retains",
                "memory-retains",
            )
            and reference_plan is not None
            and reference_plan.coefficients[0][0] > 0
        )
        passes = passes or (mode == "response" and len(calls) == 38)
        passes = passes or (mode == "plastic" and len(calls) > 37)
        passes = passes or (mode in ("joint", "sequential") and len(calls) > 37)
        passes = passes or (mode == "aim" and len(calls) > 37)
        passes = passes or (mode == "executed" and len(calls) > 37)
        passes = passes or (mode == "memory-aim" and len(calls) == 38)
        passes = passes or (mode == "memory-target" and len(calls) == 39)
        if not passes:
            r["qpos"][:, 0] *= 0.1
        return r

    def install(policy, plan):
        installed.append(plan)
        if isinstance(plan, AimGoalCommand):
            policy.target_pos_w = plan.target()
        return {"applied": True}

    p.forecaster = SimpleNamespace(forecast=forecast)
    monkeypatch.setattr(synergy, "condition_synergy_reference", install)
    monkeypatch.setattr(synergy, "condition_whole_joint_reference", install)
    monkeypatch.setattr(synergy, "condition_aim_goal", install)
    monkeypatch.setattr(
        joint_response,
        "propose_joint_task_response",
        lambda baseline, probes: (np.full((2, 29), 0.001), {"verified": False}),
    )
    monkeypatch.setattr(
        response_model,
        "propose_synergy_response",
        lambda basis, samples: (np.full((2, 6), 0.001), {"physics_verified": False}),
    )
    monkeypatch.setattr(
        synergy.SharedRoboNaldoPlayer,
        "propose",
        lambda self: (np.zeros(29), {"proposed_target": np.zeros(29)}),
    )
    monkeypatch.setattr(
        executed_controller,
        "apply_executed_target_synergy",
        lambda self, plan: (np.zeros(29), {"applied": True}),
    )
    p.propose()
    assert len(calls) == expected_calls
    event = p.events[0]
    audit_choice = verify_synergy_search(
        event["candidate_windows"][:37], [0.0] * min(37, expected_calls), event["generations"]
    )
    if mode == "memory-aim":
        assert event["selected"] == "memory-00" and event["selected_basis"] == "aim-command"
        assert not p.target_edits
        assert event["candidate_windows"][37]["memory_option"]["kind"] == "aim_command"
    elif mode == "memory-target":
        assert event["selected"] == "memory-01" and event["selected_basis"] == "executed-target"
        assert p.executed_target_plan.start_control == 100
        assert p.target_edits == [{"applied": True}]
        assert event["candidate_windows"][38]["memory_option"]["projection_mode"] == "per_axis"
    elif mode == "memory-retains":
        assert audit_choice > 0 and not p.target_edits
        assert event["additional_candidate_search"] is None
    elif mode == "executed":
        assert audit_choice == 0 and event["selected"].startswith("target-")
        assert event["selected_basis"] == "executed-target"
        assert not event["original_actual_target_authority_unchanged"]
        assert event["installation"]["projection_mode"] == projection
        assert all(c.projection_mode == projection for c in calls[37:])
        assert p.target_edits == [{"applied": True}]
        additional = event["additional_candidate_search"]
        synthetic = [event["candidate_windows"][0]] + [
            dict(w, name=w["name"][7:]) for w in event["candidate_windows"][37:]
        ]
        assert verify_synergy_search(synthetic, [0.0] * 37, additional["generations"]) > 0
    elif mode == "executed-retains":
        assert audit_choice > 0 and not p.target_edits
        assert event["additional_candidate_search"] is None
        assert event["original_actual_target_authority_unchanged"]
    elif mode == "response":
        assert audit_choice == 0 and event["selected"] == "native-synergy-response"
    elif mode == "plastic":
        assert audit_choice == 0 and event["selected"].startswith("plastic-")
        assert installed[0].basis_isaac == p.plastic_basis_isaac
    elif mode in ("joint", "sequential"):
        assert audit_choice == 0 and event["selected"].startswith("joint-")
        assert event["selected_basis"] == "joint-reference"
        assert len(installed[0].coefficients[0]) == 29
    elif mode == "aim":
        assert audit_choice == 0 and event["selected"].startswith("aim-goal-")
        assert event["selected_basis"] == "aim-command"
        np.testing.assert_array_equal(p.goal, installed[0].target())
    else:
        assert event["candidate_windows"][audit_choice]["name"] == event["selected"]
    assert bool(installed) == (
        mode
        in (
            "rescue",
            "response",
            "stable",
            "plastic",
            "joint",
            "joint-retains",
            "sequential",
            "sequential-retains",
            "aim",
            "aim-retains",
            "executed-retains",
            "memory-aim",
            "memory-retains",
        )
    )
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
