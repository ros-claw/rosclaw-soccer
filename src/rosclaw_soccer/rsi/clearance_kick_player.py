"""One measured-geometry feedback candidate after the old teacher abstains."""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.ball_clearance_control import BallClearanceController, BallClearancePlan
from rosclaw_soccer.rsi.predicted_touch_window import predicted_touch_window
from rosclaw_soccer.rsi.shared_nn_forecast import SharedNNForecaster
from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash
from rosclaw_soccer.rsi.synergy_forecast_kick_player import SynergyForecastKickPlayer


class ClearanceKickPlayer(SynergyForecastKickPlayer):
    forecast_controls = 128

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.clearance_plan = BallClearancePlan()
        self.clearance_controller = BallClearanceController(self, self.clearance_plan)
        self.clearance_forecaster = SharedNNForecaster(
            self,
            expected_policy_type=type(self.policy),
            expected_model_hash=compiled_model_hash(self.world.model),
            native_step_budget=1280,
        )
        self.clearance_screened = self.clearance_enabled = False
        self.clearance_events: list[dict[str, Any]] = []
        self.clearance_edits: list[dict[str, Any]] = []

    def plan_entry(self) -> None:
        super().plan_entry()
        if self.clearance_screened:
            return
        self.clearance_screened = True
        if self.events[-1].get("selected_window", {}).get("window_pass", False):
            self.clearance_events.append(dict(skipped="OLD_PASSING_BRANCH_RETAINED", enabled=False))
            return
        if self.events[-1].get("skipped") or self.events[-1]["frame"] > 232:
            self.clearance_events.append(dict(skipped="NO_COMPLETE_DECLARED_WINDOW", enabled=False))
            return
        try:
            predicted = self.clearance_forecaster.forecast(
                controls=128, before_proposal=True, clearance_plan=self.clearance_plan
            )
            # A physically terminated future is not a complete128 window.
            window = None
            if predicted["controls"] == 128:
                window = predicted_touch_window(
                    predicted,
                    ball_address=self.layout.bq,
                    root_address=self.layout.rq,
                    start_control=predicted["donor_control_frame"],
                    diagnostic_tail=True,
                )
            self.clearance_enabled = window is not None and bool(window["window_pass"])
            self.candidate_predictions.append(("measured-clearance", predicted))
            self.clearance_events.append(
                dict(
                    enabled=self.clearance_enabled,
                    window=window,
                    body_failed=predicted["body_failed"],
                    actual_execution_required_for_task_gain=True,
                    hardware_authorized=False,
                )
            )
            if self.clearance_enabled:
                self.selected_prediction = predicted
                self.plan_applied = True
        except BaseException:
            self.failed = True
            raise

    def propose(self) -> tuple[np.ndarray, dict[str, Any]]:
        target, receipt = super().propose()
        if not self.clearance_enabled:
            return target, receipt
        try:
            target, edit = self.clearance_controller.apply()
        except BaseException:
            self.failed = True
            raise
        self.clearance_edits.append(edit)
        return target, dict(
            receipt,
            original_nn_target=receipt["proposed_target"],
            proposed_target=target.copy(),
            clearance_edit=edit,
        )
