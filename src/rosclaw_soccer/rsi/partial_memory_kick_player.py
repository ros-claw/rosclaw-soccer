"""Retrieve two frozen motor options with current-body NN physical previews.

No new goal grid, coefficient search, outcome lookup or neural update. The
unchanged backbone's passing branch always wins before memory is consulted.
"""

from dataclasses import replace
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.aim_goal_command import AimGoalCommand
from rosclaw_soccer.rsi.executed_synergy_kick_player import ExecutedSynergyKickPlayer
from rosclaw_soccer.rsi.executed_target_synergy import ExecutedTargetSynergy
from rosclaw_soccer.rsi.partial_motor_memory import MotorOption, encode_motor_option
from rosclaw_soccer.rsi.predicted_touch_window import predicted_touch_window
from rosclaw_soccer.rsi.shared_episode_clock import absolute_control_frame
from rosclaw_soccer.rsi.synergy_forecast_kick_player import Candidate


class PartialMemoryKickPlayer(ExecutedSynergyKickPlayer):
    extra_candidate_ceiling = 2
    target_projection_mode = "per_axis"
    memory_options: tuple[MotorOption, ...] = ()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if (
            type(self.memory_options) is not tuple
            or len(self.memory_options) != 2
            or type(self.memory_options[0]) is not AimGoalCommand
            or type(self.memory_options[1]) is not ExecutedTargetSynergy
            or self.memory_options[1].projection_mode != "per_axis"
            or self.basis_hg is None
            or not np.array_equal(np.asarray(self.memory_options[1].basis_hg), self.basis_hg)
            or not np.array_equal(self.goal, np.array([8.0, 0.0, 0.11], np.float32))
        ):
            raise ValueError(
                "exact frozen two-option memory on the same HG29 motor subspace required"
            )
        for option in self.memory_options:
            encode_motor_option(option)

    def _search_additional_candidates(
        self, candidates: list[Candidate], horizon: int
    ) -> dict[str, Any]:
        frame = absolute_control_frame(
            self.world, control_origin=getattr(self, "_forecast_control_origin", 0)
        )
        for i, remembered in enumerate(self.memory_options):
            option: MotorOption
            if type(remembered) is AimGoalCommand:
                option = remembered
                predicted = self.forecaster.forecast(
                    controls=horizon, before_proposal=True, aim_goal=remembered
                )
            else:
                assert type(remembered) is ExecutedTargetSynergy
                option = replace(remembered, start_control=frame)
                predicted = self.forecaster.forecast(
                    controls=horizon, before_proposal=True, target_plan=option
                )
            window = predicted_touch_window(
                predicted,
                ball_address=self.layout.bq,
                root_address=self.layout.rq,
                start_control=frame,
                diagnostic_tail=True,
            )
            candidates.append((f"memory-{i:02d}", option, predicted, window))
        return dict(
            schema="soccer.two_partial_motor_memory_preview.v1",
            candidate_count=2,
            runtime_options=[encode_motor_option(option) for option in self.memory_options],
            source_case_ids_used_at_runtime=False,
            source_outcomes_used_at_runtime=False,
            new_parameter_search=False,
            task_exam_modified=False,
            neural_optimizer_updates=0,
            hardware_authorized=False,
        )
