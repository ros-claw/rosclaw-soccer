"""Causal actor observations with a separately privileged training critic."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.sim.contracts import hash_json

_FORBIDDEN_ACTOR_TOKENS = ("policy_frame", "policy_phase", "shooter_phase", "future_")


@dataclass(frozen=True)
class GoalkeeperObservationSpec:
    ball_history_steps: int = 8
    joint_count: int = 29
    ball_history_frame: str = "keeper_anchor"
    schema_version: str = "rosclaw_soccer.goalkeeper_observation_spec.v3"

    def __post_init__(self) -> None:
        if not 5 <= self.ball_history_steps <= 10:
            raise ValueError("goalkeeper ball history must contain 5-10 frames")
        if not 1 <= self.joint_count <= 64:
            raise ValueError("goalkeeper joint count must be in [1, 64]")
        if self.ball_history_frame != "keeper_anchor":
            raise ValueError("goalkeeper ball history must use the stable keeper anchor frame")
        if any(token in name for name in self.actor_names for token in _FORBIDDEN_ACTOR_TOKENS):
            raise ValueError("goalkeeper actor observation leaks a hidden policy phase")

    @property
    def actor_names(self) -> tuple[str, ...]:
        return (
            *(
                f"ball_relative_history.{step}.{axis}"
                for step in range(self.ball_history_steps)
                for axis in "xyz"
            ),
            *(f"causal_ball_velocity_estimate.{axis}" for axis in "xyz"),
            "causal_intercept_estimate.time",
            "causal_intercept_estimate.y",
            "causal_intercept_estimate.z",
            "causal_intercept_confidence",
            *(f"causal_target_region.{index}" for index in range(6)),
            *(f"gravity_orientation.{axis}" for axis in "xyz"),
            *(f"root_linear_velocity.{axis}" for axis in "xyz"),
            *(f"angular_velocity.{axis}" for axis in "xyz"),
            *(f"joint_position.{index}" for index in range(self.joint_count)),
            *(f"joint_velocity.{index}" for index in range(self.joint_count)),
            *(f"previous_action.{index}" for index in range(self.joint_count)),
        )

    @property
    def privileged_critic_names(self) -> tuple[str, ...]:
        return (
            *self.actor_names,
            *(f"ball_velocity.{axis}" for axis in "xyz"),
            "intercept_time",
            "intercept_y",
            "intercept_z",
            *(f"target_region.{index}" for index in range(6)),
            "ball_contact",
            *(f"left_hand_position.{axis}" for axis in "xyz"),
            *(f"right_hand_position.{axis}" for axis in "xyz"),
        )

    @property
    def actor_contract_hash(self) -> str:
        return str(
            hash_json(
                {
                    "schema_version": self.schema_version,
                    "names": self.actor_names,
                    "ball_history_frame": self.ball_history_frame,
                    "ball_history_steps": self.ball_history_steps,
                }
            )
        )

    @property
    def critic_contract_hash(self) -> str:
        return str(
            hash_json(
                {
                    "schema_version": self.schema_version,
                    "names": self.privileged_critic_names,
                    "ball_history_frame": self.ball_history_frame,
                    "ball_history_steps": self.ball_history_steps,
                }
            )
        )


@dataclass(frozen=True)
class GoalkeeperActorObservation:
    values: tuple[float, ...]
    actor_contract_hash: str
    ball_history_ready: bool
    estimated_ball_velocity_mps: tuple[float, float, float]
    estimated_intercept: tuple[float, float, float]
    intercept_confidence: float
    estimated_target_region: tuple[float, float, float, float, float, float]
    observed_flight_start_sec: float | None
    schema_version: str = "rosclaw_soccer.goalkeeper_actor_observation.v3"

    def __post_init__(self) -> None:
        if not self.values or any(not math.isfinite(value) for value in self.values):
            raise ValueError("goalkeeper actor observation must be non-empty and finite")
        if any(not math.isfinite(value) for value in self.estimated_ball_velocity_mps):
            raise ValueError("estimated ball velocity must be finite")
        if any(not math.isfinite(value) for value in self.estimated_intercept):
            raise ValueError("estimated ball intercept must be finite")
        if not math.isfinite(self.intercept_confidence) or not (
            0.0 <= self.intercept_confidence <= 1.0
        ):
            raise ValueError("intercept confidence must be in [0, 1]")
        if (
            len(self.estimated_target_region) != 6
            or any(not math.isfinite(value) for value in self.estimated_target_region)
            or not math.isclose(sum(self.estimated_target_region), 1.0, abs_tol=1e-9)
        ):
            raise ValueError("estimated target region must be a finite one-hot vector")
        if not self.actor_contract_hash.startswith("sha256:"):
            raise ValueError("actor observation requires a content-addressed contract")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "values": list(self.values),
            "actor_contract_hash": self.actor_contract_hash,
            "ball_history_ready": self.ball_history_ready,
            "estimated_ball_velocity_mps": list(self.estimated_ball_velocity_mps),
            "estimated_intercept": list(self.estimated_intercept),
            "intercept_confidence": self.intercept_confidence,
            "estimated_target_region": list(self.estimated_target_region),
            "observed_flight_start_sec": self.observed_flight_start_sec,
        }


class GoalkeeperActorObserver:
    """Build observations using only ball history and the keeper's own body."""

    def __init__(
        self,
        spec: GoalkeeperObservationSpec | None = None,
        *,
        control_dt_sec: float = 0.02,
        flight_velocity_threshold_mps: float = 0.10,
    ) -> None:
        self.spec = spec or GoalkeeperObservationSpec()
        if not math.isfinite(control_dt_sec) or control_dt_sec <= 0.0:
            raise ValueError("goalkeeper observation dt must be finite and positive")
        if not math.isfinite(flight_velocity_threshold_mps) or flight_velocity_threshold_mps <= 0.0:
            raise ValueError("goalkeeper flight threshold must be finite and positive")
        self.control_dt_sec = control_dt_sec
        self.flight_velocity_threshold_mps = flight_velocity_threshold_mps
        self._ball_history: deque[NDArray[np.float64]] = deque(maxlen=self.spec.ball_history_steps)
        self._flight_start_sec: float | None = None

    def observe(
        self,
        *,
        timestamp_sec: float,
        ball_relative_position_m: np.ndarray,
        gravity_orientation: np.ndarray,
        root_linear_velocity_mps: np.ndarray,
        angular_velocity_rad_s: np.ndarray,
        joint_position_rad: np.ndarray,
        joint_velocity_rad_s: np.ndarray,
        previous_action_rad: np.ndarray,
    ) -> GoalkeeperActorObservation:
        if not math.isfinite(timestamp_sec) or timestamp_sec < 0.0:
            raise ValueError("goalkeeper observation timestamp must be finite and non-negative")
        ball = _vector(ball_relative_position_m, 3, "ball relative position")
        gravity = _vector(gravity_orientation, 3, "gravity orientation")
        root_velocity = _vector(root_linear_velocity_mps, 3, "root linear velocity")
        angular = _vector(angular_velocity_rad_s, 3, "angular velocity")
        q = _vector(joint_position_rad, self.spec.joint_count, "joint position")
        dq = _vector(joint_velocity_rad_s, self.spec.joint_count, "joint velocity")
        previous = _vector(previous_action_rad, self.spec.joint_count, "previous action")
        self._ball_history.append(ball.copy())
        padded = [self._ball_history[0]] * (self.spec.ball_history_steps - len(self._ball_history))
        padded.extend(self._ball_history)
        velocity = self.estimated_relative_ball_velocity_mps
        intercept, confidence, region = _estimate_intercept(
            ball=ball,
            velocity=velocity,
            history_count=len(self._ball_history),
            history_capacity=self.spec.ball_history_steps,
            approach_threshold_mps=self.flight_velocity_threshold_mps,
        )
        distance = float(np.linalg.norm(ball[:2]))
        radial_velocity = (
            0.0 if distance <= 1e-9 else float(np.dot(velocity[:2], ball[:2]) / distance)
        )
        # Negative radial velocity means that the visible ball is approaching
        # the keeper.  This is orientation-independent and, unlike a fixed
        # local-x test, does not mistake ground settling or a pass moving away
        # from goal for a shot.
        new_threat = bool(
            self._flight_start_sec is None and radial_velocity < -self.flight_velocity_threshold_mps
        )
        if new_threat:
            self._flight_start_sec = timestamp_sec
        values = np.concatenate(
            (
                *padded,
                velocity,
                intercept,
                np.asarray((confidence,), dtype=np.float64),
                region,
                gravity,
                root_velocity,
                angular,
                q,
                dq,
                previous,
            )
        )
        velocity_values = (float(velocity[0]), float(velocity[1]), float(velocity[2]))
        intercept_values = (
            float(intercept[0]),
            float(intercept[1]),
            float(intercept[2]),
        )
        observation = GoalkeeperActorObservation(
            values=tuple(float(value) for value in values),
            actor_contract_hash=self.spec.actor_contract_hash,
            ball_history_ready=len(self._ball_history) == self.spec.ball_history_steps,
            estimated_ball_velocity_mps=velocity_values,
            estimated_intercept=intercept_values,
            intercept_confidence=confidence,
            estimated_target_region=tuple(float(value) for value in region),  # type: ignore[arg-type]
            observed_flight_start_sec=self._flight_start_sec,
        )
        if new_threat:
            # The detection frame may still contain a preceding pass or a
            # stationary-ball history.  Preserve its causal detection output
            # for this frame, then segment the history so subsequent velocity
            # and ballistic intercept estimates use only the new threat.
            latest = self._ball_history[-1].copy()
            self._ball_history.clear()
            self._ball_history.append(latest)
        return observation

    @property
    def estimated_relative_ball_velocity_mps(self) -> NDArray[np.float64]:
        if len(self._ball_history) < 2:
            return np.zeros(3, dtype=np.float64)
        span = (len(self._ball_history) - 1) * self.control_dt_sec
        return np.asarray((self._ball_history[-1] - self._ball_history[0]) / span, dtype=np.float64)


def _vector(value: np.ndarray, size: int, label: str) -> NDArray[np.float64]:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"goalkeeper {label} must contain {size} finite values")
    return result


def _estimate_intercept(
    *,
    ball: NDArray[np.float64],
    velocity: NDArray[np.float64],
    history_count: int,
    history_capacity: int,
    approach_threshold_mps: float,
) -> tuple[NDArray[np.float64], float, NDArray[np.float64]]:
    """Predict the keeper-plane crossing from past and present ball samples only.

    The stable goalkeeper anchor is the local ``x=0`` plane.  A ballistic
    vertical estimate is used so high shots are not confused with rising
    balls that will already be descending at the keeper.  Region index five
    means that the visible history does not yet support an approaching-ball
    estimate; it is deliberately distinct from a center shot.
    """

    confidence = min(1.0, max(0.0, (history_count - 1) / max(1, history_capacity - 1)))
    approaching = bool(ball[0] >= 0.0 and velocity[0] < -approach_threshold_mps)
    if not approaching:
        region = np.zeros(6, dtype=np.float64)
        region[5] = 1.0
        return np.zeros(3, dtype=np.float64), 0.0, region
    horizon = float(np.clip(-ball[0] / velocity[0], 0.0, 3.0))
    intercept_y = float(ball[1] + velocity[1] * horizon)
    intercept_z = float(max(0.0, ball[2] + velocity[2] * horizon - 4.905 * horizon**2))
    intercept = np.asarray((horizon, intercept_y, intercept_z), dtype=np.float64)
    region = np.zeros(6, dtype=np.float64)
    lateral_threshold = 0.25
    high = intercept_z >= 1.0
    # Local +y is world -y because the keeper faces the shooter.  Indices are
    # semantic only inside this frozen observation contract.
    if intercept_y < -lateral_threshold:
        region[0 if high else 2] = 1.0
    elif intercept_y > lateral_threshold:
        region[1 if high else 3] = 1.0
    else:
        region[4] = 1.0
    return intercept, confidence, region


__all__ = [
    "GoalkeeperActorObservation",
    "GoalkeeperActorObserver",
    "GoalkeeperObservationSpec",
]
