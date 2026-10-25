"""A task-command teacher extension; the original measured-body NN remains the motor controller."""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.aim_goal_command import AimGoalCommand
from rosclaw_soccer.rsi.predicted_touch_window import predicted_touch_window
from rosclaw_soccer.rsi.synergy_forecast_kick_player import Candidate, SynergyForecastKickPlayer


def declared_aim_commands() -> tuple[AimGoalCommand, ...]:
    return tuple(
        AimGoalCommand((8.0, y, z))
        for y in (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
        for z in (0.11, 0.61, 1.11)
        if (y, z) != (0.0, 0.11)
    )


class AimCommandKickPlayer(SynergyForecastKickPlayer):
    extra_candidate_ceiling = 20
    forecast_controls = 128

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.forecast_controls != 128 or not np.array_equal(
            self.goal, np.array([8.0, 0.0, 0.11], np.float32)
        ):
            raise ValueError(
                "qualified128 aim study with separate frozen nominal task goal required"
            )

    def _search_additional_candidates(
        self, candidates: list[Candidate], horizon: int
    ) -> dict[str, Any]:
        commands = declared_aim_commands()
        for i, command in enumerate(commands):
            predicted = self.forecaster.forecast(
                controls=horizon, before_proposal=True, aim_goal=command
            )
            window = predicted_touch_window(
                predicted,
                ball_address=self.layout.bq,
                root_address=self.layout.rq,
                start_control=self.world.control_steps,
                diagnostic_tail=True,
            )
            candidates.append((f"aim-goal-{i:02d}", command, predicted, window))
        return dict(
            schema="soccer.bounded_neural_aim_command_search.v1",
            candidate_count=len(commands),
            task_scoring_goal_world_point=[8.0, 0.0, 0.11],
            command_world_points=[list(c.world_point) for c in commands],
            measured_ball_modified=False,
            past_target_history_modified=False,
            task_exam_modified=False,
            reference_arrays_modified_by_new_commands=False,
            neural_optimizer_updates=0,
            hardware_authorized=False,
        )
