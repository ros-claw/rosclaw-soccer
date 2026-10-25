"""One intervention selected by copied-NN contact/recovery forecasts, SIM teacher.

No actor updates, old-case label lookup, or short geometric distance objective.
After the one intervention, the actual original NN follows the chosen branch.
"""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.ball_surface_tasks import measure_ball_surfaces
from rosclaw_soccer.rsi.predicted_touch_window import predicted_touch_window
from rosclaw_soccer.rsi.shared_nn_forecast import SharedNNForecaster
from rosclaw_soccer.rsi.shared_robonaldo_player import SharedRoboNaldoPlayer
from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash
from rosclaw_soccer.rsi.task_space_kick_player import encode_modified_target

PITCH_AXES = (0, 3, 4, 6, 9, 10)


def candidate_native_budget(axes: tuple[int, ...]) -> int:
    if (
        type(axes) is not tuple
        or not axes
        or len(set(axes)) != len(axes)
        or any(type(i) is not int or not 0 <= i < 12 for i in axes)
    ):
        raise ValueError("explicit unique candidate leg axes required")
    return (1 + 2 * len(axes)) * 640


def choose_window_rescue(
    candidates: list[tuple[str, np.ndarray, np.ndarray | None, dict[str, Any], dict[str, Any]]],
) -> int:
    if not candidates or candidates[0][0] != "parent":
        raise ValueError("explicit unchanged parent first required")
    if candidates[0][4]["window_pass"]:
        return 0
    eligible = [
        i
        for i, c in enumerate(candidates)
        if i
        and c[4]["window_pass"]
        and c[4]["quality_observed"]
        and c[4]["predicted_body_healthy"]
        and c[4]["predicted_nonfoot_clean"]
    ]
    nominal = candidates[0][1]
    return (
        max(
            eligible,
            key=lambda i: (
                candidates[i][4]["reward"],
                -float(np.linalg.norm(candidates[i][1] - nominal)),
            ),
        )
        if eligible
        else 0
    )


class OutcomeForecastKickPlayer(SharedRoboNaldoPlayer):
    candidate_axes = PITCH_AXES

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        from importlib import import_module

        self.mapping = np.array(import_module(type(self.policy).__module__).MUJOCO_TO_ISAAC)
        self.predictor = SharedNNForecaster(
            self,
            expected_policy_type=type(self.policy),
            expected_model_hash=compiled_model_hash(self.world.model),
            native_step_budget=candidate_native_budget(self.candidate_axes),
        )
        self.forecaster = self.predictor
        self.decided = self.foot_seen = self.nonfoot_seen = False
        self.selected_prediction: dict[str, Any] | None = None
        self.selected_history: np.ndarray | None = None
        self.intervention_frame: int | None = None
        self.events: list[dict[str, Any]] = []
        self.candidate_predictions: list[tuple[str, dict[str, Any]]] = []

    @property
    def changed_controls(self) -> int:
        return int(self.intervention_frame is not None)

    @property
    def selected_records(self) -> list[tuple[int, dict[str, Any]]]:
        if self.selected_prediction is None:
            return []
        return [(self.selected_prediction["donor_control_frame"], self.selected_prediction)]

    def propose(self) -> tuple[np.ndarray, dict[str, Any]]:
        nominal, receipt = super().propose()
        frame = self.world.control_steps
        if self.decided or self.foot_seen or self.nonfoot_seen or frame > 239:
            return nominal, receipt
        surfaces = measure_ball_surfaces(self.world.model, self.world.data, self.layout)["surfaces"]
        gaps = [
            s["gap_m"] for name, s in surfaces.items() if name.endswith("foot") and s["available"]
        ]
        if not gaps or min(gaps) > 0.06:
            return nominal, receipt
        self.decided = (
            True  # Exactly one finite search per actual world, never an adaptive restart.
        )
        horizon = min(64, 300 - frame)
        parent = self.predictor.forecast(controls=horizon)
        baseline = predicted_touch_window(
            parent, ball_address=self.layout.bq, root_address=self.layout.rq, start_control=frame
        )
        candidates: list[
            tuple[str, np.ndarray, np.ndarray | None, dict[str, Any], dict[str, Any]]
        ] = [("parent", nominal, None, parent, baseline)]
        skipped = []
        if not baseline["window_pass"]:
            for axis in self.candidate_axes:
                for sign in (-1.0, 1.0):
                    target = nominal.copy()
                    target[axis] += sign * 0.03
                    encoded = encode_modified_target(
                        self.policy, target, self.mapping, modified_indices=(axis,)
                    )
                    if encoded is None:
                        skipped.append(dict(axis=axis, sign=sign, reason="original_action_bound"))
                        continue
                    target, history = encoded
                    lo, hi = self.world.model.jnt_range[self.layout.joints[axis]]
                    if not lo <= target[axis] <= hi:
                        skipped.append(
                            dict(axis=axis, sign=sign, reason="physical_modified_joint_bound")
                        )
                        continue
                    predicted = self.predictor.forecast(
                        controls=horizon, first_target=target, first_action_history=history
                    )
                    outcome = predicted_touch_window(
                        predicted,
                        ball_address=self.layout.bq,
                        root_address=self.layout.rq,
                        start_control=frame,
                    )
                    candidates.append(
                        (f"axis-{axis}-{sign:+.0f}", target, history, predicted, outcome)
                    )
        chosen = choose_window_rescue(candidates)
        name, target, chosen_history, prediction, outcome = candidates[chosen]
        self.candidate_predictions = [(c[0], c[3]) for c in candidates]
        self.events.append(
            dict(
                frame=frame,
                selected=name,
                parent_window=baseline,
                selected_window=outcome,
                candidate_windows=[dict(name=c[0], outcome=c[4]) for c in candidates],
                skipped=skipped,
                changed=chosen != 0,
                actual_prefix_has_no_foot_or_nonfoot_contact=True,
                requested_delta_rad=(target - nominal).tolist(),
            )
        )
        if chosen:
            if chosen_history is None:
                raise ValueError("selected intervention lacks exact composite history")
            self.pending = target.copy()
            self.selected_history = chosen_history.copy()
            self.intervention_frame = frame
        self.selected_prediction = prediction
        return target.copy(), receipt

    def commit(self) -> None:
        frame = self.world.control_steps - 1
        super().commit()
        if self.intervention_frame == frame:
            assert self.selected_history is not None
            self.policy.last_action_il = self.selected_history.copy()
        feet = {self.prefix + side + "_ankle_roll_link" for side in ("left", "right")}
        bodies = set(self.world.last_ball_contact_bodies[self.prefix])
        self.foot_seen |= bool(feet & bodies)
        self.nonfoot_seen |= bool(bodies - feet)
