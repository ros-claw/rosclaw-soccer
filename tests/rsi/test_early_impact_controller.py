"""Prospective trigger/action lease tests only, no G1 skill qualification."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi import early_impact_controller as mod
from tests.physics.test_g1_feedback_forecast import world
from tests.rsi.test_early_feedback_impact_score import fixture as prediction
from tests.rsi.test_impact_torque_screen import sample


def trigger(frame=30, **changes):
    kw = dict(
        side=0,
        contact_frame=None,
        gap_m=0.12,
        pelvis_z_m=0.7,
        screened=False,
        baseline_windows=0,
    )
    kw.update(changes)
    return mod.early_trigger_status(frame, **kw)


def test_causal_early_gate_accepts_gaps_larger_than_old_last20ms_gate():
    assert trigger() is None
    assert trigger(gap_m=0.18) is None
    assert trigger(gap_m=-0.02) is None
    assert trigger(gap_m=0.181) == "OUTSIDE_FIXED_CURRENT_SURFACE_GAP"


@pytest.mark.parametrize("frame", [0, 29, 37, 38, 39, 57, 58, 59, 295, 299])
def test_no_replan_or_terminal_crossing(frame):
    assert trigger(frame) == "OUTSIDE_NO_REPLAN80MS_WINDOW"


@pytest.mark.parametrize(
    "changes",
    [
        dict(side=True),
        dict(side=2),
        dict(contact_frame=30),
        dict(gap_m=np.nan),
        dict(pelvis_z_m=np.inf),
        dict(baseline_windows=True),
        dict(baseline_windows=33),
        dict(screened=1),
    ],
)
def test_bad_trigger_is_not_a_silent_noop(changes):
    with pytest.raises(ValueError):
        trigger(**changes)


def fixture(monkeypatch, *, first=29, gain=True, bad_repeat=False):
    model, data, _ = world()
    geometry = replace(sample(), foot_ball_signed_gap_m=np.full(2, 0.12))

    class Sampler:
        def __init__(self, *a, **kw):
            pass

        def sample(self, d):
            assert d is data
            return geometry

    class Control:
        def __init__(self, *a, **kw):
            assert kw["frame"] == round(data.time / 0.02)

    class Forecaster:
        def __init__(self, *a, **kw):
            assert kw["max_forecast_steps"] == 1800
            self._scratch = object()
            self.attempted_steps = self.integrated_steps = 0
            self.zero_calls = 0

        def forecast(self, d, target, *, control, early_offset_nm):
            self.attempted_steps += 40
            self.integrated_steps += 40
            p, _ = prediction(first=first, offset=early_offset_nm)
            p.arrays["canonical_qvel_after_substep"][:, 36] = (
                0.1 if gain and np.any(early_offset_nm) else 1.0
            )
            p = replace(
                p,
                next_target_records=tuple({"target": np.zeros(29)} for _ in range(3)),
                next_neural_records=tuple(
                    {"synthetic": np.zeros((1, 2), dtype=np.float32)} for _ in range(3)
                ),
            )
            if not np.any(early_offset_nm):
                self.zero_calls += 1
                if bad_repeat and self.zero_calls > 1:
                    p.next_neural_records[1]["synthetic"][0, 0] = -0.0
            return p

    monkeypatch.setattr(mod, "G1JointTaskKinematicSampler", Sampler)
    monkeypatch.setattr(mod, "G1FeedbackBranchControl", Control)
    monkeypatch.setattr(mod, "G1EarlyFeedbackForecaster", Forecaster)
    controller = mod.EarlyImpactController(
        model, world_hash="synthetic", kp=np.ones(29), kd=np.ones(29)
    )

    def step(frame):
        data.time = frame * 0.02
        return controller.plan(
            data,
            np.zeros(29),
            frame=frame,
            tracker=None,
            decoder=None,
            ik=SimpleNamespace(side=0, contact=None),
            policy={},
            body_prefix={},
            previous_motor_delta_rad=np.zeros(12),
        )

    return controller, data, step


def reach_start(step):
    for frame in range(30):
        assert not np.any(step(frame).offset_nm)
    return step(30)


def test_action_lasts_exactly_three_contiguous_frames_then_zero(monkeypatch):
    c, _, step = fixture(monkeypatch)
    first = reach_start(step)
    assert np.any(first.offset_nm) and len(first.candidates) == 14
    assert first.report["forecast_substeps"] == 560
    assert first.report["action_frames_remaining_including_current"] == 3
    assert not first.report["teacher_qualified"] and not first.report["full_h300_gain_proven"]
    for frame, remaining in ((31, 2), (32, 1)):
        held = step(frame)
        np.testing.assert_array_equal(held.offset_nm, first.offset_nm)
        assert held.report["action_frames_remaining_including_current"] == remaining
        assert held.report["status"] == "HELD_BOUNDED60MS_ACTION"
        assert not held.candidates and held.report["forecast_substeps"] == 0
    for frame in range(33, 300):
        assert not np.any(step(frame).offset_nm)
    assert c.baseline_windows == 1 and c.forecaster.attempted_steps == 560


def test_skip_or_repeat_during_lease_is_terminal_not_silent_continuation(monkeypatch):
    c, _, step = fixture(monkeypatch)
    reach_start(step)
    with pytest.raises(ValueError, match="contiguous"):
        step(32)
    with pytest.raises(ValueError, match="contiguous"):
        step(31)
    assert c.failed


def test_actual_low_pelvis_cancels_held_action_permanently(monkeypatch):
    c, d, step = fixture(monkeypatch)
    reach_start(step)
    d.qpos[2] = 0.64
    cancelled = step(31)
    assert not np.any(cancelled.offset_nm)
    assert cancelled.report["status"] == "CANCELLED_CURRENT_UNSAFE_CONTACT_OR_PELVIS"
    d.qpos[2] = 0.7
    assert not np.any(step(32).offset_nm)
    assert c.action_end_frame == 31 and c.screened


def test_actual_nonfoot_contact_cancels_held_action_permanently(monkeypatch):
    c, d, step = fixture(monkeypatch)
    reach_start(step)
    pelvis = int(c.model.body("pelvis").id)
    geom = int(np.flatnonzero(c.model.geom_bodyid == pelvis)[0])
    monkeypatch.setattr(
        mod,
        "read_geom_contact_wrenches",
        lambda *a, **kw: [
            SimpleNamespace(other_geom_id=geom, force_world_n=np.array([2.0, 0.0, 0.0]))
        ],
    )
    assert not np.any(step(31).offset_nm)
    monkeypatch.setattr(mod, "read_geom_contact_wrenches", lambda *a, **kw: [])
    assert not np.any(step(32).offset_nm)
    assert c.screened and c.action_end_frame == 31


def test_no_gain_selects_zero_and_never_holds_action(monkeypatch):
    c, _, step = fixture(monkeypatch, gain=False)
    plan = reach_start(step)
    assert not np.any(plan.offset_nm) and plan.report["selected_name"] == "baseline"
    assert c.action_end_frame == 0 and c.screened


@pytest.mark.parametrize("first", [None, 31, 39])
def test_no_complete_impact_cannot_start_action_or_exhaust_unbounded_search(monkeypatch, first):
    c, _, step = fixture(monkeypatch, first=first)
    for frame in range(300):
        assert not np.any(step(frame).offset_nm)
    assert c.baseline_windows == 32 and c.forecaster.attempted_steps == 1280
    assert not c.screened


def test_mismatch_of_middle_neural_capture_is_terminal_and_preserves_all_candidates(monkeypatch):
    c, _, step = fixture(monkeypatch, bad_repeat=True)
    with pytest.raises(ValueError, match="zero repeat"):
        reach_start(step)
    assert c.failed and len(c.last_candidates) == 14
    with pytest.raises(ValueError, match="contiguous"):
        step(31)
