"""SIM_ONLY proprioceptive neural receiving with measured feet and shin clearance."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from rosclaw_soccer.rsi.receiving_lateral_piecewise_expert import ReceivingLateralPiecewiseExpert
from rosclaw_soccer.rsi.receiving_temporal_motor_expert import (
    ReceivingTemporalMotorExpert,
    TemporalMotorWeights,
)
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.receiving_feedback import ReceivingFeedbackObservation

FEATURE_COUNT = 48
HIDDEN_COUNT = 32


@dataclass(frozen=True)
class KinematicMotorWeights:
    input_matrix: tuple[float, ...] = (0.0,) * (FEATURE_COUNT * HIDDEN_COUNT)
    input_bias: tuple[float, ...] = (0.0,) * HIDDEN_COUNT
    output_matrix: tuple[float, ...] = (0.0,) * (HIDDEN_COUNT * 12)
    output_bias: tuple[float, ...] = (0.0,) * 12

    @classmethod
    def from_legacy(cls, legacy: TemporalMotorWeights) -> KinematicMotorWeights:
        """Embed a pinned 10D actor without granting the new sensors any initial effect."""
        if not isinstance(legacy, TemporalMotorWeights):
            raise ValueError("typed legacy temporal motor weights required")
        first = np.zeros((HIDDEN_COUNT, FEATURE_COUNT), dtype=np.float64)
        first[:, :10] = np.asarray(legacy.input_matrix, dtype=np.float64).reshape(HIDDEN_COUNT, 10)
        return cls(
            input_matrix=tuple(float(value) for value in first.reshape(-1)),
            input_bias=legacy.input_bias,
            output_matrix=legacy.output_matrix,
            output_bias=legacy.output_bias,
        )

    def __post_init__(self) -> None:
        for values, size in (
            (self.input_matrix, FEATURE_COUNT * HIDDEN_COUNT),
            (self.input_bias, HIDDEN_COUNT),
            (self.output_matrix, HIDDEN_COUNT * 12),
            (self.output_bias, 12),
        ):
            if (
                type(values) is not tuple
                or len(values) != size
                or any(
                    type(value) is not float or not math.isfinite(value) or abs(value) > 10
                    for value in values
                )
            ):
                raise ValueError("finite bounded kinematic motor weights required")

    @property
    def contract_hash(self) -> str:
        return str(
            hash_json(
                {
                    "schema": "rosclaw_soccer.rsi.kinematic_motor_weights.v1",
                    "input_matrix": self.input_matrix,
                    "input_bias": self.input_bias,
                    "output_matrix": self.output_matrix,
                    "output_bias": self.output_bias,
                }
            )
        )


@dataclass
class ReceivingKinematicTemporalExpert(ReceivingTemporalMotorExpert):
    kinematic_policy: KinematicMotorWeights = field(default_factory=KinematicMotorWeights)
    requires_shin_clearance: bool = field(init=False, default=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.kinematic_policy, KinematicMotorWeights):
            raise ValueError("typed kinematic motor policy required")
        self.contract_hash = hash_json(
            {
                "schema": "rosclaw_soccer.rsi.receiving_kinematic_temporal_expert.v1",
                "parent_contract_hash": self.contract_hash,
                "kinematic_policy_hash": self.kinematic_policy.contract_hash,
                "requires_shin_clearance": True,
                "activation_ceiling": "SIM_ONLY",
            }
        )

    @staticmethod
    def features(observation: ReceivingFeedbackObservation) -> tuple[float, ...]:
        feet = observation.foot_kinematics
        shin = observation.shin_clearance
        if feet is None or shin is None:
            raise ValueError("same-frame measured feet and shin clearance required")
        ball_position = np.asarray(observation.qpos[36:39], dtype=np.float64)
        ball_velocity = np.asarray(observation.qvel[35:38], dtype=np.float64)
        foot_position = np.asarray(feet.foot_position_world_m, dtype=np.float64)
        foot_velocity = np.asarray(feet.foot_linear_velocity_world_mps, dtype=np.float64)
        vector = np.concatenate(
            (
                np.asarray(ReceivingTemporalMotorExpert.features(observation)),
                ((foot_position - ball_position) / 0.5).reshape(-1),
                ((foot_velocity - ball_velocity) / 2.0).reshape(-1),
                np.asarray(shin.clearance_m, dtype=np.float64) / 0.1,
                np.asarray(observation.qpos[7:19], dtype=np.float64) / 1.5,
                np.asarray(observation.qvel[6:18], dtype=np.float64) / 10.0,
            )
        )
        if vector.shape != (FEATURE_COUNT,) or not np.isfinite(vector).all():
            raise ValueError("finite 48-dimensional proprioceptive observation required")
        return tuple(float(value) for value in np.clip(vector, -3.0, 3.0))

    def _motor_logits(self, values: np.ndarray) -> np.ndarray:
        first = np.asarray(self.kinematic_policy.input_matrix).reshape(HIDDEN_COUNT, FEATURE_COUNT)
        if not np.any(first[:, 10:]):
            # The embedded legacy actor must use the same 32x10 contiguous GEMV,
            # not a 32x48 GEMV with zero columns: roundoff changes contact timing.
            projected = np.ascontiguousarray(first[:, :10]) @ values[:10]
        else:
            projected = first @ values
        hidden = np.tanh(projected + np.asarray(self.kinematic_policy.input_bias))
        return np.asarray(self.kinematic_policy.output_matrix).reshape(
            12, HIDDEN_COUNT
        ) @ hidden + np.asarray(self.kinematic_policy.output_bias)


@dataclass
class ReceivingProtectedKinematicExpert(ReceivingKinematicTemporalExpert):
    """Freeze the parent path for two measured old-skill initial states."""

    protected_initial_features: tuple[tuple[float, ...], ...] = ()
    protection_radius: float = 0.006
    protected_episode: bool | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            type(self.protected_initial_features) is not tuple
            or len(self.protected_initial_features) != 2
            or any(
                type(row) is not tuple
                or len(row) != 10
                or any(
                    type(value) is not float or not math.isfinite(value) or abs(value) > 2
                    for value in row
                )
                for row in self.protected_initial_features
            )
            or type(self.protection_radius) is not float
            or not 0.001 <= self.protection_radius <= 0.02
        ):
            raise ValueError("two finite old-skill retention states required")
        self.contract_hash = hash_json(
            {
                "schema": "rosclaw_soccer.rsi.receiving_protected_kinematic_expert.v1",
                "parent_contract_hash": self.contract_hash,
                "protected_initial_features": self.protected_initial_features,
                "protection_radius": self.protection_radius,
                "activation_ceiling": "SIM_ONLY",
            }
        )

    def propose(self, observation: ReceivingFeedbackObservation) -> tuple[float, ...]:
        if self.protected_episode is None and observation.frame >= 15:
            measured = np.asarray(ReceivingTemporalMotorExpert.features(observation))
            self.protected_episode = any(
                np.linalg.norm(measured - np.asarray(row)) <= self.protection_radius
                for row in self.protected_initial_features
            )
        if self.protected_episode:
            return ReceivingLateralPiecewiseExpert.propose(self, observation)
        return super().propose(observation)
