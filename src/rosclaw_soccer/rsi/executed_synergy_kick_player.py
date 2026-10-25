"""Finite actual-target CEM only after the frozen reference teacher abstains."""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.executed_target_synergy import (
    ExecutedTargetSynergy,
    apply_executed_target_synergy,
)
from rosclaw_soccer.rsi.predicted_touch_window import predicted_touch_window
from rosclaw_soccer.rsi.synergy_forecast_kick_player import (
    Candidate,
    SynergyForecastKickPlayer,
    synergy_search_rank,
)


class ExecutedSynergyKickPlayer(SynergyForecastKickPlayer):
    extra_candidate_ceiling = 36
    forecast_controls = 128

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.forecast_controls != 128:
            raise ValueError("explicit qualified128 executed-target teacher required")
        self.target_edits: list[dict[str, Any]] = []

    def _search_additional_candidates(
        self, candidates: list[Candidate], horizon: int
    ) -> dict[str, Any]:
        assert self.basis_hg is not None
        basis = tuple(tuple(float(x) for x in row) for row in self.basis_hg)
        rng = np.random.default_rng(7001)
        mean, std = np.zeros((2, 6)), np.full((2, 6), 0.025)
        generations = []
        for generation in range(3):
            noise = rng.normal(size=(6, 2, 6))
            population = np.clip(mean + np.concatenate((noise, -noise)) * std, -0.06, 0.06)
            ranks = []
            for i, coeff in enumerate(population):
                plan = ExecutedTargetSynergy(
                    self.world.control_steps,
                    basis,
                    (
                        tuple(float(x) for x in coeff[0]),
                        tuple(float(x) for x in coeff[1]),
                    ),
                )
                predicted = self.forecaster.forecast(
                    controls=horizon, before_proposal=True, target_plan=plan
                )
                window = predicted_touch_window(
                    predicted,
                    ball_address=self.layout.bq,
                    root_address=self.layout.rq,
                    start_control=self.world.control_steps,
                    diagnostic_tail=True,
                )
                candidates.append((f"target-g{generation}-p{i:02d}", plan, predicted, window))
                ranks.append(synergy_search_rank(predicted, window))
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
        return dict(
            schema="soccer.executed_target_synergy_search.v1",
            candidate_count=36,
            generations=generations,
            seed=7001,
            reference_modified_by_new_plans=False,
            actual_target_authority="original_float32_codec_plus_bounded_HG29_synergy",
            subsequent_nn_history="actual_composite_action",
            task_exam_modified=False,
            neural_optimizer_updates=0,
            hardware_authorized=False,
        )

    def propose(self) -> tuple[np.ndarray, dict[str, Any]]:
        target, receipt = super().propose()
        if self.executed_target_plan is not None:
            target, edit = apply_executed_target_synergy(self, self.executed_target_plan)
            self.target_edits.append(edit)
            receipt = dict(receipt, original_nn_target=receipt["proposed_target"], target_edit=edit)
            receipt["proposed_target"] = target.copy()
        return target, receipt
