"""Finite pre-contact timing teacher, original measured NN and unmodified PD.

One search at measured skill entry, parent/±3 frames over24 intervals. No
course-label lookup, target-delta combination, root edits or neural training.
"""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.motion_reference_timing import MotionTimingPlan, condition_motion_reference
from rosclaw_soccer.rsi.predicted_touch_window import predicted_touch_window
from rosclaw_soccer.rsi.shared_nn_forecast import SharedNNForecaster
from rosclaw_soccer.rsi.shared_robonaldo_player import SharedRoboNaldoPlayer
from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash


class MotionTimingKickPlayer(SharedRoboNaldoPlayer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.forecaster = SharedNNForecaster(
            self,
            expected_policy_type=type(self.policy),
            expected_model_hash=compiled_model_hash(self.world.model),
            native_step_budget=2880,
        )
        self.decided = False
        self.events: list[dict[str, Any]] = []
        self.candidate_predictions: list[tuple[str, dict[str, Any]]] = []
        self.selected_prediction: dict[str, Any] | None = None
        self.plan_applied = False

    @property
    def changed_controls(self) -> int:
        # One reference installation, not a claim that only one NN target changes.
        return int(self.plan_applied)

    @property
    def selected_records(self) -> list[tuple[int, dict[str, Any]]]:
        if self.selected_prediction is None:
            return []
        return [(self.selected_prediction["donor_control_frame"], self.selected_prediction)]

    def propose(self) -> tuple[np.ndarray, dict[str, Any]]:
        if not self.decided:
            self.world._require_current_boundary()
            if self.pending is not None or self.measured_entry_receipt is None:
                raise ValueError("one actual measured pre-proposal entry required")
            self.decided = True
            frame = self.world.control_steps
            reference = int(self.policy.time_step) - int(self.policy.WARMUP_STEPS)
            if frame > 239 or reference + 24 >= self.policy.motion_total_steps:
                self.events.append(
                    dict(frame=frame, selected="parent", skipped="no_complete_window")
                )
                return super().propose()
            horizon = min(96, 360 - frame)
            plans = [
                None,
                MotionTimingPlan(reference, 24, -3.0),
                MotionTimingPlan(reference, 24, 3.0),
            ]
            candidates = []
            for plan in plans:
                predicted = self.forecaster.forecast(
                    controls=horizon, before_proposal=True, reference_plan=plan
                )
                outcome = predicted_touch_window(
                    predicted,
                    ball_address=self.layout.bq,
                    root_address=self.layout.rq,
                    start_control=frame,
                    diagnostic_tail=True,
                )
                name = "parent" if plan is None else f"timing-{plan.peak_shift_frames:+.0f}"
                candidates.append((name, plan, predicted, outcome))
                if plan is None and outcome["window_pass"]:
                    break  # Never optimize an already-passing original branch.
            chosen = 0
            eligible = [i for i, c in enumerate(candidates) if i and c[3]["window_pass"]]
            if not candidates[0][3]["window_pass"] and eligible:
                chosen = max(eligible, key=lambda i: candidates[i][3]["reward"])
            name, plan, prediction, outcome = candidates[chosen]
            install = None
            if plan is not None:
                install = condition_motion_reference(self.policy, plan)
                self.plan_applied = True
                self.selected_prediction = prediction
            self.candidate_predictions = [(c[0], c[2]) for c in candidates]
            self.events.append(
                dict(
                    frame=frame,
                    reference_start=reference,
                    selected=name,
                    changed=chosen != 0,
                    installation=install,
                    parent_window=candidates[0][3],
                    selected_window=outcome,
                    candidate_windows=[dict(name=c[0], outcome=c[3]) for c in candidates],
                    clean_prefix_required_by_episode_runner=True,
                )
            )
        return super().propose()
