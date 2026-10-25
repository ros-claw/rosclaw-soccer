"""Finite data-subspace CEM teacher with complete NN physical counterfactuals.

No NN weight updates and no motion authority outside the owned SIM reference.
Search outcomes cannot change the original exam or authorize promotion.
"""

from importlib import import_module
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.motion_synergy_reference import (
    MotionSynergyPlan,
    condition_synergy_reference,
)
from rosclaw_soccer.rsi.predicted_touch_window import predicted_touch_window
from rosclaw_soccer.rsi.shared_nn_forecast import SharedNNForecaster
from rosclaw_soccer.rsi.shared_robonaldo_player import SharedRoboNaldoPlayer
from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash


def synergy_search_rank(
    prediction: dict[str, Any], window: dict[str, Any]
) -> tuple[int, int, int, int, float, float]:
    """Feasibility first; contact-force severity breaks unsafe/censored ties only."""
    return (
        int(window["window_pass"]),
        int(window["predicted_body_healthy"]),
        int(window["predicted_nonfoot_clean"]),
        int(window["quality_observed"]),
        float(window["reward"]),
        -float(prediction["ball_forces"][:, 0, 2].max()),
    )


class SynergyForecastKickPlayer(SharedRoboNaldoPlayer):
    basis_hg: np.ndarray | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.basis_hg is None or self.basis_hg.shape != (6, 29):
            raise ValueError("explicit fitted HG29 rank6 motion subspace required")
        mapping = np.array(import_module(type(self.policy).__module__).MUJOCO_TO_ISAAC)
        if sorted(mapping.tolist()) != list(range(29)):
            raise ValueError("qualified HG-to-Isaac bijection required")
        basis_il = np.zeros_like(self.basis_hg)
        basis_il[:, mapping] = self.basis_hg
        self.basis_isaac = tuple(tuple(float(x) for x in r) for r in basis_il)
        self.forecaster = SharedNNForecaster(
            self,
            expected_policy_type=type(self.policy),
            expected_model_hash=compiled_model_hash(self.world.model),
            native_step_budget=37 * 960,
        )
        self.decided = self.plan_applied = False
        self.events: list[dict[str, Any]] = []
        self.candidate_predictions: list[tuple[str, dict[str, Any]]] = []
        self.selected_prediction: dict[str, Any] | None = None

    @property
    def changed_controls(self) -> int:
        return int(self.plan_applied)  # Reference installations, not changed PD controls.

    @property
    def selected_records(self) -> list[tuple[int, dict[str, Any]]]:
        if self.selected_prediction is None:
            return []
        return [(self.selected_prediction["donor_control_frame"], self.selected_prediction)]

    def propose(self) -> tuple[np.ndarray, dict[str, Any]]:
        if self.decided:
            return super().propose()
        self.world._require_current_boundary()
        if self.pending is not None or self.measured_entry_receipt is None:
            raise ValueError("single actual measured pre-proposal skill entry required")
        self.decided = True
        frame = self.world.control_steps
        reference = int(self.policy.time_step) - int(self.policy.WARMUP_STEPS)
        if frame > 239 or reference + 48 >= self.policy.motion_total_steps:
            self.events.append(
                dict(
                    frame=frame,
                    selected="parent",
                    changed=False,
                    skipped="no_complete_reference_window",
                )
            )
            return super().propose()
        horizon = min(96, 360 - frame)
        candidates: list[tuple[str, MotionSynergyPlan | None, dict[str, Any], dict[str, Any]]] = []

        def preview(
            name: str, coefficients: np.ndarray | None
        ) -> tuple[int, int, int, int, float, float]:
            plan = (
                None
                if coefficients is None
                else MotionSynergyPlan(
                    reference,
                    self.basis_isaac,
                    (
                        tuple(float(v) for v in coefficients[0]),
                        tuple(float(v) for v in coefficients[1]),
                    ),
                )
            )
            predicted = self.forecaster.forecast(
                controls=horizon, before_proposal=True, reference_plan=plan
            )
            score = predicted_touch_window(
                predicted,
                ball_address=self.layout.bq,
                root_address=self.layout.rq,
                start_control=frame,
                diagnostic_tail=True,
            )
            candidates.append((name, plan, predicted, score))
            return synergy_search_rank(predicted, score)

        preview("parent", None)
        generations = []
        if not candidates[0][3]["window_pass"]:
            rng = np.random.default_rng(7001)  # SAME seed per state; no course-label lookup.
            mean, std = np.zeros((2, 6)), np.full((2, 6), 0.025)
            for generation in range(3):
                noise = rng.normal(size=(6, 2, 6))
                population = np.clip(mean + np.concatenate((noise, -noise)) * std, -0.06, 0.06)
                ranks = [preview(f"g{generation}-p{i:02d}", c) for i, c in enumerate(population)]
                order = sorted(range(12), key=lambda i: ranks[i], reverse=True)
                elite = population[order[:3]]
                generations.append(
                    dict(
                        generation=generation,
                        elite_indices=order[:3],
                        mean=mean.tolist(),
                        std=std.tolist(),
                    )
                )
                mean = 0.5 * mean + 0.5 * elite.mean(axis=0)
                std = np.maximum(0.006, 0.5 * std + 0.5 * elite.std(axis=0))
        eligible = [i for i, c in enumerate(candidates) if i and c[3]["window_pass"]]
        chosen = (
            max(eligible, key=lambda i: synergy_search_rank(candidates[i][2], candidates[i][3]))
            if eligible
            else 0
        )
        name, plan, prediction, outcome = candidates[chosen]
        install = None
        if plan is not None:
            install = condition_synergy_reference(self.policy, plan)
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
                candidate_windows=[
                    dict(
                        name=c[0],
                        outcome=c[3],
                        coefficients=None if c[1] is None else c[1].coefficients,
                    )
                    for c in candidates
                ],
                generations=generations,
                original_actual_target_authority_unchanged=True,
            )
        )
        return super().propose()
