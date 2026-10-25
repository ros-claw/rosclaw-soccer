from types import SimpleNamespace

import numpy as np
import pytest
from test_predicted_touch_window import prediction

import rosclaw_soccer.rsi.motion_timing_kick_player as timing


@pytest.mark.parametrize(
    "mode,expected,calls",
    [("parent", "parent", 1), ("rescue", "timing-+3", 3), ("fail", "parent", 3)],
)
def test_finite_reference_selection_uses_only_complete_window_rescue(
    monkeypatch, mode, expected, calls
):
    p = object.__new__(timing.MotionTimingKickPlayer)
    p.world = SimpleNamespace(_require_current_boundary=lambda: None, control_steps=100)
    p.policy = SimpleNamespace(time_step=20, WARMUP_STEPS=0, motion_total_steps=100)
    p.pending = None
    p.measured_entry_receipt = "actual"
    p.layout = SimpleNamespace(bq=0, rq=7)
    p.decided = p.plan_applied = False
    p.events, p.candidate_predictions = [], []
    p.selected_prediction = None
    count = []

    def forecast(*, controls, before_proposal, reference_plan):
        count.append(reference_plan)
        assert controls == 96 and before_proposal
        r = prediction(n=96)
        r["donor_control_frame"] = 100
        passes = mode == "parent" or (
            mode == "rescue"
            and reference_plan is not None
            and reference_plan.peak_shift_frames == 3
        )
        if not passes:
            r["qpos"][:, 0] *= 0.1
        return r

    p.forecaster = SimpleNamespace(forecast=forecast)
    installed = []

    def install(policy, plan):
        installed.append(plan)
        return {"applied": True}

    monkeypatch.setattr(timing, "condition_motion_reference", install)
    monkeypatch.setattr(timing.SharedRoboNaldoPlayer, "propose", lambda self: (np.zeros(29), {}))
    p.propose()
    assert len(count) == calls
    assert p.events[0]["selected"] == expected
    assert bool(installed) == (mode == "rescue")
    p.propose()
    assert len(count) == calls  # No second search/retry after seeing actual consequences.
