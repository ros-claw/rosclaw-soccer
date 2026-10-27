"""Experimental offline actor around an unchanged SIM-only parent motor."""

from dataclasses import dataclass, field, replace

import numpy as np

from rosclaw_soccer.rsi.receiving_foot_servo_phase_motor import ReceivingFootServoPhaseMotor
from rosclaw_soccer.rsi.receiving_smooth_residual_motor import ReceivingSmoothResidualMotor
from rosclaw_soccer.rsi.receiving_temporal_motor_expert import ReceivingTemporalMotorExpert
from rosclaw_soccer.rsi.team_coverage_actor import TeamCoverageActor
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.receiving_feedback import ReceivingFeedbackObservation


@dataclass
class TeamCoverageMotor(ReceivingSmoothResidualMotor):
    """Intervene only outside the original parent gate and near audited states."""

    coverage_actor: TeamCoverageActor | None = None
    coverage_action: int = field(init=False, default=16)
    coverage_fallback: bool = field(init=False, default=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        if type(self.coverage_actor) is not TeamCoverageActor:
            raise ValueError("explicit SIM_ONLY coverage actor required")
        self.contract_hash = hash_json(
            dict(
                schema="soccer.team_coverage_motor.v1",
                parent=self.contract_hash,
                actor={
                    k: getattr(self.coverage_actor, k).tolist()
                    for k in (
                        "center",
                        "scale",
                        "weight1",
                        "bias1",
                        "weight2",
                        "bias2",
                        "support",
                        "allowed",
                        "offsets",
                    )
                },
                radius=self.coverage_actor.radius,
                original_parent_unchanged=True,
                activation_ceiling="SIM_ONLY",
            )
        )

    def propose(self, observation: ReceivingFeedbackObservation) -> tuple[float, ...]:
        self.ensure_protection_decided(observation)
        if (
            not self.activation_decided
            and observation.frame >= 19
            and self.protected_episode is False
        ):
            measured = ReceivingTemporalMotorExpert.features(observation)
            distances = [
                np.linalg.norm(np.asarray(measured) - np.asarray(row))
                for row in self.activation_states
            ]
            parent_active = (
                measured[1] <= self.maximum_relative_lateral_feature
                and measured[3] >= self.minimum_relative_vx_feature
                and any(d <= self.activation_radius for d in distances)
            )
            expanded_active = (
                measured[1] <= 0.775 and measured[3] >= -0.55 and any(d <= 0.022 for d in distances)
            )
            if not parent_active and expanded_active:
                assert self.coverage_actor is not None
                self.coverage_action, self.coverage_fallback = self.coverage_actor.select(measured)
                if self.coverage_action != 16:
                    self.activation_features = measured
                    self.activation_enabled = True
                    self.activation_decided = True
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
                    if self.coverage_action == 0 and self.residual_gain == 0.0:
                        self.neural_weights = self.candidate_neural_weights
                    elif self.coverage_action == 0 and self.residual_gain == 1.0:
                        self.neural_weights = self.refined_neural_weights
                    else:
                        chosen = (
                            base + self.residual_gain * (refined - base)
                            if self.coverage_action == 0
                            else base + self.coverage_actor.offsets[self.coverage_action]
                        )
                        self.neural_weights = replace(
                            self.candidate_neural_weights,
                            output_bias=tuple(float(v) for v in chosen),
                        )
                    return ReceivingFootServoPhaseMotor.propose(self, observation)
        return super().propose(observation)
