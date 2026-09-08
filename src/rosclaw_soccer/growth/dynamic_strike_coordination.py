"""Bounded runtime actor for continuous receive-to-strike coordination.

The actor owns only two dimensionless blends during the ORIENT phase: how
quickly locomotion converges from ball pacing to the predicted support stance,
and how quickly yaw turns from the incoming ball path toward the shot line.  It
cannot output joint targets, forces, torques, simulator state, or phase changes.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.sim.contracts import hash_json

STRIKE_COORDINATION_FEATURES = (
    "phase_progress",
    "stance_depth_m",
    "stance_lateral_error_m",
    "stance_yaw_error_rad",
    "approach_yaw_error_rad",
    "ball_speed_mps",
)
_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_MINIMUM_SIGMOID_PROBABILITY = 1.0 / (1.0 + math.exp(12.0))


@dataclass(frozen=True)
class StrikeCoordinationObservation:
    """Measured phase context; no intent label can substitute for body state."""

    phase_progress: float
    stance_depth_m: float
    stance_lateral_error_m: float
    stance_yaw_error_rad: float
    approach_yaw_error_rad: float
    ball_speed_mps: float

    def __post_init__(self) -> None:
        values = tuple(asdict(self).values())
        if (
            any(not isinstance(value, int | float) or not math.isfinite(value) for value in values)
            or not 0.0 <= self.phase_progress <= 2.0
            or not -1.0 <= self.stance_depth_m <= 3.0
            or not 0.0 <= self.stance_lateral_error_m <= 3.0
            or not 0.0 <= self.stance_yaw_error_rad <= math.pi
            or not 0.0 <= self.approach_yaw_error_rad <= math.pi
            or not 0.0 <= self.ball_speed_mps <= 20.0
        ):
            raise ValueError("strike coordination observation is outside its envelope")

    def vector(self) -> NDArray[np.float64]:
        return np.asarray(tuple(asdict(self).values()), dtype=np.float64)


@dataclass(frozen=True)
class StrikeCoordinationAction:
    """Two bounded high-level blends; neither value is an actuator command."""

    stance_blend: float
    goal_yaw_blend: float

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.stance_blend)
            or not math.isfinite(self.goal_yaw_blend)
            or not 0.0 <= self.stance_blend <= 0.80
            or not 0.0 <= self.goal_yaw_blend <= 1.0
        ):
            raise ValueError("strike coordination action violates its bounded authority")


@dataclass(frozen=True)
class DynamicStrikeCoordinationActor:
    """NumPy-only linear actor with explicit data provenance and SIM ceiling."""

    weights: tuple[float, ...]
    biases: tuple[float, float]
    feature_mean: tuple[float, ...] = (0.0,) * len(STRIKE_COORDINATION_FEATURES)
    feature_std: tuple[float, ...] = (1.0,) * len(STRIKE_COORDINATION_FEATURES)
    maximum_stance_blend: float = 0.80
    maximum_goal_yaw_blend: float = 1.0
    policy_type: str = "parameter"
    dataset_snapshot_hash: str | None = None
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.dynamic_strike_coordination_actor.v1"

    def __post_init__(self) -> None:
        feature_count = len(STRIKE_COORDINATION_FEATURES)
        numeric = (
            *self.weights,
            *self.biases,
            *self.feature_mean,
            *self.feature_std,
            self.maximum_stance_blend,
            self.maximum_goal_yaw_blend,
        )
        if (
            len(self.weights) != 2 * feature_count
            or len(self.feature_mean) != feature_count
            or len(self.feature_std) != feature_count
            or any(not math.isfinite(value) for value in numeric)
            or any(value <= 0.0 for value in self.feature_std)
            or any(abs(value) > 12.0 for value in (*self.weights, *self.biases))
            or not 0.05 <= self.maximum_stance_blend <= 0.80
            or not 0.05 <= self.maximum_goal_yaw_blend <= 1.0
            or self.policy_type not in {"parameter", "learned_linear"}
            or (
                self.policy_type == "learned_linear"
                and (
                    self.dataset_snapshot_hash is None
                    or not _HASH.fullmatch(self.dataset_snapshot_hash)
                )
            )
            or (self.policy_type == "parameter" and self.dataset_snapshot_hash is not None)
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("dynamic strike actor violates its SIM-only contract")

    @property
    def actor_hash(self) -> str:
        return str(hash_json(asdict(self)))

    def act(self, observation: StrikeCoordinationObservation) -> StrikeCoordinationAction:
        vector = observation.vector()
        mean = np.asarray(self.feature_mean, dtype=np.float64)
        std = np.asarray(self.feature_std, dtype=np.float64)
        weights = np.asarray(self.weights, dtype=np.float64).reshape(2, len(vector))
        logits = weights @ ((vector - mean) / std) + np.asarray(self.biases, dtype=np.float64)
        if not np.all(np.isfinite(logits)):
            raise ValueError("dynamic strike actor produced non-finite logits")
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
        return StrikeCoordinationAction(
            stance_blend=float(self.maximum_stance_blend * probabilities[0]),
            goal_yaw_blend=float(self.maximum_goal_yaw_blend * probabilities[1]),
        )

    @classmethod
    def constant(
        cls, *, stance_blend: float, goal_yaw_blend: float
    ) -> DynamicStrikeCoordinationActor:
        """Build an auditable parameter probe, never a learned-policy claim."""

        action = StrikeCoordinationAction(stance_blend, goal_yaw_blend)

        return cls(
            weights=(0.0,) * (2 * len(STRIKE_COORDINATION_FEATURES)),
            biases=(
                _bounded_logit(action.stance_blend / 0.80),
                _bounded_logit(action.goal_yaw_blend),
            ),
        )

    @classmethod
    def phase_ramp(
        cls,
        *,
        stance_start_blend: float,
        stance_end_blend: float,
        goal_yaw_start_blend: float,
        goal_yaw_end_blend: float,
    ) -> DynamicStrikeCoordinationActor:
        """Build a phase-progress teacher whose authority remains two blends.

        The start and end values are the requested outputs at normalized ORIENT
        progress zero and one.  This is an auditable parameter teacher used to
        collect labelled physics rollouts; it is not represented as learned.
        """

        start = StrikeCoordinationAction(stance_start_blend, goal_yaw_start_blend)
        end = StrikeCoordinationAction(stance_end_blend, goal_yaw_end_blend)
        start_logits = (
            _bounded_logit(start.stance_blend / 0.80),
            _bounded_logit(start.goal_yaw_blend),
        )
        end_logits = (
            _bounded_logit(end.stance_blend / 0.80),
            _bounded_logit(end.goal_yaw_blend),
        )
        feature_count = len(STRIKE_COORDINATION_FEATURES)
        weights = np.zeros((2, feature_count), dtype=np.float64)
        weights[:, 0] = np.subtract(end_logits, start_logits)
        return cls(
            weights=tuple(float(value) for value in weights.reshape(-1)),
            biases=start_logits,
        )


def _bounded_logit(probability: float) -> float:
    clipped = min(
        1.0 - _MINIMUM_SIGMOID_PROBABILITY,
        max(_MINIMUM_SIGMOID_PROBABILITY, probability),
    )
    return math.log(clipped / (1.0 - clipped))


__all__ = [
    "DynamicStrikeCoordinationActor",
    "STRIKE_COORDINATION_FEATURES",
    "StrikeCoordinationAction",
    "StrikeCoordinationObservation",
]
