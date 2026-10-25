"""One finite actual-target response study, not the failed reference-space FD."""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.actuated_task_response import actuated_task_plan
from rosclaw_soccer.rsi.executed_synergy_kick_player import ExecutedSynergyKickPlayer
from rosclaw_soccer.rsi.joint_task_response import (
    propose_joint_task_response,
    response_features,
)
from rosclaw_soccer.rsi.predicted_touch_window import predicted_touch_window
from rosclaw_soccer.rsi.synergy_forecast_kick_player import Candidate


class ActuatedResponseKickPlayer(ExecutedSynergyKickPlayer):
    extra_candidate_ceiling = 59  # 29 signed pairs, at most one bounded proposal.
    target_projection_mode = "per_axis"

    def _search_additional_candidates(
        self, candidates: list[Candidate], horizon: int
    ) -> dict[str, Any]:
        plans = []

        def preview(name: str, delta: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
            plan = actuated_task_plan(self.world.control_steps, delta)
            prediction = self.forecaster.forecast(
                controls=horizon, before_proposal=True, target_plan=plan
            )
            window = predicted_touch_window(
                prediction,
                ball_address=self.layout.bq,
                root_address=self.layout.rq,
                start_control=self.world.control_steps,
                diagnostic_tail=True,
            )
            candidates.append((name, plan, prediction, window))
            plans.append(
                dict(
                    name=name,
                    first_lobe_offset_hg_rad=delta.tolist(),
                    basis_hg=plan.basis_hg,
                    coefficients=plan.coefficients,
                    start_control=plan.start_control,
                    projection_mode=plan.projection_mode,
                )
            )
            return prediction, window

        probes = []
        for axis in range(29):
            for sign in (-1, 1):
                delta = np.zeros(29, np.float64)
                delta[axis] = sign * 0.03
                prediction, window = preview(f"actual-axis-{axis:02d}-{sign:+d}", delta)
                probes.append((axis, sign, response_features(prediction, window)))
        proposal, solver = propose_joint_task_response(
            response_features(candidates[0][2], candidates[0][3]), probes
        )
        if proposal is not None:
            if np.any(proposal[1]):
                raise ValueError("no second-lobe recovery action in this task-response protocol")
            preview("actual-task-response", proposal[0].copy())
        return dict(
            schema="soccer.full_nn_actuated_task_response.v1",
            candidate_plans=plans,
            solver_report=solver,
            solver_math_reused_from_reference_study=True,
            fitted_pca_basis_used_for_new_actions=False,
            policy_reference_modified=False,
            actual_target_codec="existing_per_axis_qualified_history",
            new_action_or_torque_limits=False,
            neural_optimizer_updates=0,
            physical_teacher_qualified=False,
            hardware_authorized=False,
        )
