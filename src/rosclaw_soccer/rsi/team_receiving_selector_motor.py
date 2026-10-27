"""Optional SIM-only measured-state selector under existing receiving guards."""

from dataclasses import dataclass, field, replace

import numpy as np

from rosclaw_soccer.rsi.receiving_foot_servo_phase_motor import ReceivingFootServoPhaseMotor
from rosclaw_soccer.rsi.receiving_smooth_residual_motor import ReceivingSmoothResidualMotor
from rosclaw_soccer.rsi.receiving_temporal_motor_expert import ReceivingTemporalMotorExpert
from rosclaw_soccer.rsi.team_receiving_selector import TeamReceivingSelector
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.receiving_feedback import ReceivingFeedbackObservation


@dataclass
class TeamReceivingSelectorMotor(ReceivingSmoothResidualMotor):
    """Choose once at frame 19, retaining the full parent safety envelope.

    Action zero/out-of-support uses the exact smooth parent, not the older
    training-data baseline. Nonzero actions are learned output-bias offsets,
    never direct joint torques and never hardware authorization.
    """

    selector: TeamReceivingSelector | None = None
    selected_action: int = field(init=False, default=0)
    selector_out_of_support: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if type(self.selector) is not TeamReceivingSelector:
            raise ValueError("explicit qualified SIM_ONLY receiving selector required")
        self.contract_hash = hash_json(
            dict(
                schema="soccer.team_receiving_selector_motor.v1",
                parent_contract_hash=self.contract_hash,
                selector=dict(
                    center=self.selector.center.tolist(),
                    scale=self.selector.scale.tolist(),
                    weight1=self.selector.weight1.tolist(),
                    bias1=self.selector.bias1.tolist(),
                    weight2=self.selector.weight2.tolist(),
                    bias2=self.selector.bias2.tolist(),
                    offsets=self.selector.offsets.tolist(),
                ),
                parent_fallback_on_zero_or_outside_support=True,
                activation_ceiling="SIM_ONLY",
            )
        )

    def propose(self, observation: ReceivingFeedbackObservation) -> tuple[float, ...]:
        self.ensure_protection_decided(observation)
        if not self.activation_decided and observation.frame >= 19:
            measured = ReceivingTemporalMotorExpert.features(observation)
            self.activation_features = measured
            self.activation_enabled = (
                measured[1] <= self.maximum_relative_lateral_feature
                and measured[3] >= self.minimum_relative_vx_feature
                and any(
                    np.linalg.norm(np.asarray(measured) - np.asarray(row)) <= self.activation_radius
                    for row in self.activation_states
                )
            )
            self.activation_decided = True
            if self.activation_enabled:
                self.residual_gain = float(
                    np.clip(
                        (measured[1] - self.gain_lateral_start)
                        / (self.gain_lateral_full - self.gain_lateral_start),
                        0.0,
                        1.0,
                    )
                )
                base = np.asarray(self.candidate_neural_weights.output_bias)
                refined = np.asarray(self.refined_neural_weights.output_bias)
                assert self.selector is not None
                if self.protected_episode is False:
                    self.selected_action, self.selector_out_of_support = self.selector.select(
                        measured
                    )
                if self.selected_action == 0 and self.residual_gain == 0.0:
                    self.neural_weights = self.candidate_neural_weights
                elif self.selected_action == 0 and self.residual_gain == 1.0:
                    self.neural_weights = self.refined_neural_weights
                else:
                    chosen_bias = (
                        base + self.residual_gain * (refined - base)
                        if self.selected_action == 0
                        else base + self.selector.offsets[self.selected_action]
                    )
                    self.neural_weights = replace(
                        self.candidate_neural_weights,
                        output_bias=tuple(float(value) for value in chosen_bias),
                    )
        return ReceivingFootServoPhaseMotor.propose(self, observation)
