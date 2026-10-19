"""Prospective gate/selection logic only, not full native G1 qualification."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi import impact_feedback_teacher as mod
from tests.physics.test_g1_feedback_forecast import world
from tests.rsi.test_feedback_impact_score import prediction
from tests.rsi.test_impact_torque_screen import sample


def trigger(**changes):
    args = dict(
        side=0,
        contact_frame=None,
        gap_m=0.02,
        pelvis_z_m=0.7,
        already_screened=False,
        baseline_windows=0,
    )
    args.update(changes)
    return mod.trigger_status(30, **args)


def test_trigger_uses_only_current_body_and_phase():
    assert trigger() is None
    assert trigger(gap_m=-0.02) is None and trigger(gap_m=0.06) is None


@pytest.mark.parametrize(
    "key,value,reason",
    [
        ("side", -1, "NO_CURRENT_PRECONTACT_SWING_FOOT"),
        ("contact_frame", 29, "NO_CURRENT_PRECONTACT_SWING_FOOT"),
        ("gap_m", 0.061, "OUTSIDE_FIXED_CURRENT_SURFACE_GAP"),
        ("gap_m", -0.021, "OUTSIDE_FIXED_CURRENT_SURFACE_GAP"),
        ("pelvis_z_m", 0.649, "CURRENT_PELVIS_BELOW_TASK_FLOOR"),
        ("already_screened", True, "ALREADY_SCREENED_THIS_EPISODE"),
        ("baseline_windows", 32, "BASELINE_SEARCH_BUDGET_EXHAUSTED"),
    ],
)
def test_causal_gate_reasons(key, value, reason):
    assert trigger(**{key: value}) == reason


@pytest.mark.parametrize("frame", [0, 29, 39, 199, 298, 299])
def test_do_not_cross_uncalibrated_refresh_or_terminal_boundary(frame):
    assert (
        mod.trigger_status(
            frame,
            side=0,
            contact_frame=None,
            gap_m=0.02,
            pelvis_z_m=0.7,
            already_screened=False,
            baseline_windows=0,
        )
        == "OUTSIDE_CALIBRATED_NO_REPLAN_WINDOW"
    )


@pytest.mark.parametrize(
    "bad",
    [
        dict(side=True),
        dict(side=0.0),
        dict(contact_frame=30),
        dict(gap_m=np.nan),
        dict(pelvis_z_m=np.inf),
        dict(baseline_windows=33),
        dict(baseline_windows=True),
        dict(already_screened=1),
    ],
)
def test_malformed_trigger_inputs_are_contract_errors(bad):
    with pytest.raises(ValueError):
        trigger(**bad)


def fixture(monkeypatch, *, first=9, gain=True, bad_repeat=False):
    model, d, _ = world()
    geometry = replace(sample(), foot_ball_signed_gap_m=np.zeros(2))

    class Sampler:
        def __init__(self, *a, **k):
            pass

        def sample(self, data):
            assert data is d
            return geometry

    class Capture:
        def cpu(self):
            return self

        def numpy(self):
            return np.zeros((1, 2), dtype=np.float32)

    class Control:
        def __init__(self, *a, **k):
            assert k["frame"] == round(d.time / 0.02) and len(k["body_prefix"]) == 0
            self.tracker = SimpleNamespace(neural_observation=lambda: {"synthetic": Capture()})

    class Forecaster:
        def __init__(self, *a, **k):
            assert k["max_forecast_steps"] == 900
            self._scratch = object()
            self.attempted_steps = 0
            self.zero_calls = 0

        def forecast(self, data, target, *, control, first_offset_nm):
            self.attempted_steps += 20
            p = prediction(first)
            p.arrays["canonical_qvel_after_substep"][:, 36] = (
                0.1 if gain and np.any(first_offset_nm) else 1.0
            )
            if not np.any(first_offset_nm):
                self.zero_calls += 1
                if bad_repeat and self.zero_calls > 1:
                    p.arrays["canonical_qvel_after_substep"][19, 0] = 1
            return p

    monkeypatch.setattr(mod, "G1JointTaskKinematicSampler", Sampler)
    monkeypatch.setattr(mod, "G1FeedbackBranchControl", Control)
    monkeypatch.setattr(mod, "G1FeedbackForecaster", Forecaster)
    teacher = mod.ImpactFeedbackTeacher(
        model, world_hash="synthetic", kp=np.ones(29), kd=np.ones(29)
    )
    kwargs = dict(
        frame=30,
        tracker=None,
        decoder=None,
        ik=SimpleNamespace(side=0, contact=None),
        policy={},
        body_prefix={},
        previous_motor_delta_rad=np.zeros(12),
    )
    return teacher, d, kwargs


def test_one_bank_screen_and_then_no_more_actions_with_failed_candidates_preserved(monkeypatch):
    teacher, d, kw = fixture(monkeypatch)
    plan = teacher.plan(d, np.zeros(29), **kw)
    assert np.any(plan.offset_nm) and len(plan.candidates) == 14
    assert plan.report["forecast_substeps"] == 280
    assert not plan.report["teacher_qualified"] and not plan.report["full_h300_gain_proven"]
    assert not plan.report["future_source_frame_or_target_used"]
    d.time = 0.62
    second = teacher.plan(d, np.zeros(29), **dict(kw, frame=31))
    assert (
        not np.any(second.offset_nm)
        and not second.candidates
        and second.report["forecast_substeps"] == 0
    )
    assert second.report["status"] == "ALREADY_SCREENED_THIS_EPISODE"


def test_no_gain_keeps_baseline(monkeypatch):
    teacher, d, kw = fixture(monkeypatch, gain=False)
    plan = teacher.plan(d, np.zeros(29), **kw)
    assert not np.any(plan.offset_nm) and plan.report["selected_name"] == "baseline"


@pytest.mark.parametrize("first", [None, 11])
def test_incomplete_or_nofoot_forecast_retained_without_screening_others(monkeypatch, first):
    teacher, d, kw = fixture(monkeypatch, first=first)
    plan = teacher.plan(d, np.zeros(29), **kw)
    assert not np.any(plan.offset_nm) and len(plan.candidates) == 1
    assert plan.report["forecast_substeps"] == 20 and not teacher.screened
    assert plan.candidates[0].score["proxy_cost"] is None


def test_repeat_mismatch_blocks_entire_plan_and_retains_all_completed_predictions(monkeypatch):
    teacher, d, kw = fixture(monkeypatch, bad_repeat=True)
    with pytest.raises(ValueError, match="zero repeat"):
        teacher.plan(d, np.zeros(29), **kw)
    assert len(teacher.last_candidates) == 14


def test_no_same_frame_retry_or_clock_regression(monkeypatch):
    teacher, d, kw = fixture(monkeypatch)
    teacher.plan(d, np.zeros(29), **kw)
    with pytest.raises(ValueError, match="strictly increasing"):
        teacher.plan(d, np.zeros(29), **kw)
    with pytest.raises(ValueError, match="strictly increasing"):
        teacher.plan(d, np.zeros(29), **dict(kw, frame=29))


def test_nofoot_search_cannot_extend_baseline_budget(monkeypatch):
    teacher, d, kw = fixture(monkeypatch, first=None)
    for frame in range(30, 85):
        d.time = frame * 0.02
        plan = teacher.plan(d, np.zeros(29), **dict(kw, frame=frame))
        assert not np.any(plan.offset_nm)
    assert teacher.baseline_windows == 32 and teacher.forecaster.attempted_steps == 640
    assert plan.report["status"] == "BASELINE_SEARCH_BUDGET_EXHAUSTED"
