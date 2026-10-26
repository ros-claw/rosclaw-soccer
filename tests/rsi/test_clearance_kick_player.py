from types import SimpleNamespace

import pytest
from test_predicted_touch_window import prediction

from rosclaw_soccer.rsi.ball_clearance_control import BallClearancePlan
from rosclaw_soccer.rsi.clearance_kick_player import ClearanceKickPlayer
from rosclaw_soccer.rsi.synergy_forecast_kick_player import SynergyForecastKickPlayer


@pytest.mark.parametrize("mode", ["old-pass", "new-pass", "new-fail", "terminated", "exception"])
def test_only_complete_safe_prediction_can_enable_feedback_and_old_pass_never_searched(
    monkeypatch, mode
):
    player = object.__new__(ClearanceKickPlayer)
    player.clearance_screened = player.clearance_enabled = player.failed = False
    player.clearance_plan = BallClearancePlan()
    player.clearance_events, player.candidate_predictions = [], []
    player.selected_prediction, player.plan_applied = None, False
    player.layout = SimpleNamespace(bq=0, rq=7)
    player.events = [dict(frame=63, selected_window=dict(window_pass=mode == "old-pass"))]
    monkeypatch.setattr(SynergyForecastKickPlayer, "plan_entry", lambda self: None)
    calls = []

    def forecast(**kwargs):
        calls.append(kwargs)
        if mode == "exception":
            raise RuntimeError("preserved forecast failure")
        p = prediction(n=10 if mode == "terminated" else 128)
        p["donor_control_frame"], p["body_failed"] = 63, mode == "terminated"
        if mode == "new-fail":
            p["ball_forces"][:, 0, 2] = 2
        return p

    player.clearance_forecaster = SimpleNamespace(forecast=forecast)
    if mode == "exception":
        with pytest.raises(RuntimeError):
            player.plan_entry()
        assert player.failed
        return
    player.plan_entry()
    player.plan_entry()
    assert len(calls) == (0 if mode == "old-pass" else 1)
    assert player.clearance_enabled == (mode == "new-pass")
    assert player.plan_applied == (mode == "new-pass")
    if mode == "old-pass":
        assert player.clearance_events[0]["skipped"] == "OLD_PASSING_BRANCH_RETAINED"
