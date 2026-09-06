"""Fail-closed phase controller for receive-to-strike skill composition.

The controller contains no actuator or simulator access.  It converts measured
whole-body/ball readiness into a monotonic skill phase so a locomotion policy,
a strike option, and a recovery controller are never blended implicitly.  The
caller remains responsible for producing motion and for proving physical
football contact.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import IntEnum, StrEnum

from rosclaw_soccer.sim.contracts import hash_json


class StrikePhase(StrEnum):
    IDLE = "idle"
    CAPTURE = "capture"
    ORIENT = "orient"
    PLANT = "plant"
    STRIKE = "strike"
    RECOVER = "recover"
    COMPLETE = "complete"
    ABORTED = "aborted"


class StrikePhaseCode(IntEnum):
    IDLE = 0
    CAPTURE = 1
    ORIENT = 2
    PLANT = 3
    STRIKE = 4
    RECOVER = 5
    COMPLETE = 6
    ABORTED = 7


_PHASE_CODES = {
    StrikePhase.IDLE: StrikePhaseCode.IDLE,
    StrikePhase.CAPTURE: StrikePhaseCode.CAPTURE,
    StrikePhase.ORIENT: StrikePhaseCode.ORIENT,
    StrikePhase.PLANT: StrikePhaseCode.PLANT,
    StrikePhase.STRIKE: StrikePhaseCode.STRIKE,
    StrikePhase.RECOVER: StrikePhaseCode.RECOVER,
    StrikePhase.COMPLETE: StrikePhaseCode.COMPLETE,
    StrikePhase.ABORTED: StrikePhaseCode.ABORTED,
}


@dataclass(frozen=True)
class StrikePhaseConfig:
    """SIM-only transition and stance envelope for one receive-to-shot chain."""

    capture_duration_sec: float = 0.22
    orient_timeout_sec: float = 3.40
    plant_timeout_sec: float = 3.40
    strike_timeout_sec: float = 2.80
    recover_duration_sec: float = 0.70
    recover_timeout_sec: float = 2.00
    target_stance_depth_m: float = 0.36
    target_stance_lateral_m: float = -0.16
    orient_goalward_lag_mps: float = 0.20
    orient_lateral_gain_per_sec: float = 0.90
    orient_minimum_depth_m: float = 0.18
    strike_contact_horizon_sec: float = 0.36
    strike_aim_lateral_bias_m: float = 0.0
    maximum_yaw_rate_radps: float = 1.20
    orient_yaw_error_rad: float = 0.42
    ready_minimum_depth_m: float = 0.30
    ready_maximum_depth_m: float = 1.20
    ready_maximum_lateral_error_m: float = 0.60
    ready_maximum_yaw_error_rad: float = 0.35
    ready_maximum_ball_speed_mps: float = 0.60
    maximum_ball_distance_m: float = 2.40
    support_lane_clearance_m: float = 1.35
    activation_ceiling: str = "SIM_ONLY"
    training_only: bool = True
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.strike_phase_config.v1"

    def __post_init__(self) -> None:
        values = (
            self.capture_duration_sec,
            self.orient_timeout_sec,
            self.plant_timeout_sec,
            self.strike_timeout_sec,
            self.recover_duration_sec,
            self.recover_timeout_sec,
            self.target_stance_depth_m,
            self.target_stance_lateral_m,
            self.orient_goalward_lag_mps,
            self.orient_lateral_gain_per_sec,
            self.orient_minimum_depth_m,
            self.strike_contact_horizon_sec,
            self.strike_aim_lateral_bias_m,
            self.maximum_yaw_rate_radps,
            self.orient_yaw_error_rad,
            self.ready_minimum_depth_m,
            self.ready_maximum_depth_m,
            self.ready_maximum_lateral_error_m,
            self.ready_maximum_yaw_error_rad,
            self.ready_maximum_ball_speed_mps,
            self.maximum_ball_distance_m,
            self.support_lane_clearance_m,
        )
        if (
            any(not math.isfinite(value) for value in values)
            or not 0.08 <= self.capture_duration_sec <= 0.60
            or not 0.40 <= self.orient_timeout_sec <= 4.00
            or not 0.80 <= self.plant_timeout_sec <= 5.00
            or not 0.80 <= self.strike_timeout_sec <= 4.00
            or not 0.20 <= self.recover_duration_sec <= 2.00
            or not self.recover_duration_sec <= self.recover_timeout_sec <= 4.00
            or not 0.30 <= self.target_stance_depth_m <= 0.80
            or not -0.35 <= self.target_stance_lateral_m <= 0.35
            or not 0.08 <= self.orient_goalward_lag_mps <= 0.40
            or not 0.20 <= self.orient_lateral_gain_per_sec <= 1.50
            or not 0.15 <= self.orient_minimum_depth_m <= 0.60
            or not 0.20 <= self.strike_contact_horizon_sec <= 1.20
            or not -0.75 <= self.strike_aim_lateral_bias_m <= 0.75
            or not 0.80 <= self.maximum_yaw_rate_radps <= 1.50
            or not 0.20 <= self.orient_yaw_error_rad <= 0.60
            or not 0.25 <= self.ready_minimum_depth_m <= 0.80
            or not 0.80 <= self.ready_maximum_depth_m <= 1.50
            or self.ready_minimum_depth_m >= self.ready_maximum_depth_m
            or not 0.20 <= self.ready_maximum_lateral_error_m <= 0.60
            or not 0.15 <= self.ready_maximum_yaw_error_rad <= 0.60
            or not 0.10 <= self.ready_maximum_ball_speed_mps <= 1.00
            or not 0.80 <= self.maximum_ball_distance_m <= 3.00
            or not 1.00 <= self.support_lane_clearance_m <= 2.00
            or self.activation_ceiling != "SIM_ONLY"
            or not self.training_only
            or self.hardware_authorized
        ):
            raise ValueError("strike phase config violates its SIM-only envelope")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass
class StrikePhaseState:
    phase: StrikePhase = StrikePhase.IDLE
    phase_enter_time_sec: float = 0.0
    capture_time_sec: float | None = None
    transition_count: int = 0
    abort_reason: str | None = None
    contact_observed: bool = False

    @property
    def code(self) -> int:
        return int(_PHASE_CODES[self.phase])

    @property
    def active(self) -> bool:
        return self.phase not in {
            StrikePhase.IDLE,
            StrikePhase.COMPLETE,
            StrikePhase.ABORTED,
        }

    def begin_capture(self, time_sec: float) -> None:
        if not math.isfinite(time_sec) or time_sec < 0.0:
            raise ValueError("strike capture time is invalid")
        if self.phase is not StrikePhase.IDLE:
            return
        self.phase = StrikePhase.CAPTURE
        self.phase_enter_time_sec = time_sec
        self.capture_time_sec = time_sec
        self.transition_count = 1
        self.abort_reason = None
        self.contact_observed = False

    def advance(
        self,
        *,
        time_sec: float,
        stable: bool,
        stance_depth_m: float,
        stance_lateral_error_m: float,
        stance_yaw_error_rad: float,
        approach_yaw_error_rad: float,
        ball_speed_mps: float,
        ball_distance_m: float,
        option_active: bool,
        option_contact_observed: bool,
        option_completed: bool,
        config: StrikePhaseConfig,
    ) -> StrikePhase:
        if self.phase in {StrikePhase.IDLE, StrikePhase.COMPLETE, StrikePhase.ABORTED}:
            return self.phase
        values = (
            time_sec,
            stance_depth_m,
            stance_lateral_error_m,
            stance_yaw_error_rad,
            approach_yaw_error_rad,
            ball_speed_mps,
            ball_distance_m,
        )
        if any(not math.isfinite(value) for value in values) or time_sec < 0.0:
            self._abort(time_sec=self.phase_enter_time_sec, reason="NONFINITE_STATE")
            return self.phase
        if time_sec < self.phase_enter_time_sec:
            self._abort(time_sec=self.phase_enter_time_sec, reason="CLOCK_ROLLBACK")
            return self.phase
        elapsed = time_sec - self.phase_enter_time_sec
        if self.phase is StrikePhase.RECOVER:
            # A struck football is expected to leave the player.  Ball escape
            # is therefore only a pre-contact failure, never a recovery
            # failure.  Likewise, transient post-impact motion is exactly
            # what RECOVER owns: wait for a measured stable body before
            # completing, and fail closed only when that recovery times out.
            if stable and elapsed >= config.recover_duration_sec:
                self._transition(StrikePhase.COMPLETE, time_sec)
            elif elapsed > config.recover_timeout_sec:
                self._abort(time_sec=time_sec, reason="RECOVERY_TIMEOUT")
            return self.phase
        if not stable:
            self._abort(time_sec=time_sec, reason="BODY_UNSTABLE")
            return self.phase
        if ball_distance_m > config.maximum_ball_distance_m:
            self._abort(time_sec=time_sec, reason="BALL_ESCAPED")
            return self.phase

        if self.phase is StrikePhase.CAPTURE:
            if elapsed >= config.capture_duration_sec:
                self._transition(StrikePhase.ORIENT, time_sec)
        elif self.phase is StrikePhase.ORIENT:
            if (
                approach_yaw_error_rad <= config.orient_yaw_error_rad
                and stance_depth_m >= config.orient_minimum_depth_m
                and stance_lateral_error_m <= config.ready_maximum_lateral_error_m
            ):
                self._transition(StrikePhase.PLANT, time_sec)
            elif elapsed > config.orient_timeout_sec:
                self._abort(time_sec=time_sec, reason="ORIENT_TIMEOUT")
        elif self.phase is StrikePhase.PLANT:
            ready = bool(
                config.ready_minimum_depth_m <= stance_depth_m <= config.ready_maximum_depth_m
                and stance_lateral_error_m <= config.ready_maximum_lateral_error_m
                and stance_yaw_error_rad <= config.ready_maximum_yaw_error_rad
                and ball_speed_mps <= config.ready_maximum_ball_speed_mps
            )
            if ready:
                self._transition(StrikePhase.STRIKE, time_sec)
            elif elapsed > config.plant_timeout_sec:
                self._abort(time_sec=time_sec, reason="PLANT_TIMEOUT")
        elif self.phase is StrikePhase.STRIKE:
            self.contact_observed = bool(self.contact_observed or option_contact_observed)
            if self.contact_observed:
                self._transition(StrikePhase.RECOVER, time_sec)
            elif elapsed > config.strike_timeout_sec:
                self._abort(time_sec=time_sec, reason="STRIKE_TIMEOUT")
            elif option_completed and not option_active:
                self._abort(time_sec=time_sec, reason="OPTION_ENDED_WITHOUT_CONTACT")
        return self.phase

    def _transition(self, phase: StrikePhase, time_sec: float) -> None:
        self.phase = phase
        self.phase_enter_time_sec = time_sec
        self.transition_count += 1

    def _abort(self, *, time_sec: float, reason: str) -> None:
        self.phase = StrikePhase.ABORTED
        self.phase_enter_time_sec = time_sec
        self.transition_count += 1
        self.abort_reason = reason


__all__ = [
    "StrikePhase",
    "StrikePhaseCode",
    "StrikePhaseConfig",
    "StrikePhaseState",
]
