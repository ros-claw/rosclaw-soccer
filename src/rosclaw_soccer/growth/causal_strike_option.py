"""Causal state machine for a delayed, short-horizon G1 strike option.

The controller owns no pose, joint, torque or ball command.  It observes a
predecessor skill phase and receiver proprioception, then emits one bounded
event: begin a velocity-matched bridge into a frozen strike motion.  Once the
bridge begins the decision is latched, preventing the stop/go phase fighting
seen when two clocks independently modulated the same long motion.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import IntEnum

from rosclaw_soccer.sim.contracts import hash_json


class CausalStrikeOptionPhase(IntEnum):
    PREPARE = 0
    TRACK = 1
    COMMIT = 2
    RECOVER = 3
    ABORTED = 4


@dataclass(frozen=True)
class G1CausalStrikeOptionConfig:
    """SIM-only authority envelope for the first dynamic strike seed."""

    predecessor_track_policy_frame: int = 155
    predecessor_commit_policy_frame: int = 170
    predecessor_abort_policy_frame: int = 185
    strike_phase_start_frame: int = 100
    bridge_duration_sec: float = 0.60
    bridge_entry_velocity_scale: float = 0.20
    bridge_exit_velocity_scale: float = 1.0
    bridge_boundary_velocity_limit_rad_s: float = 2.0
    history_prime_frames: int = 5
    minimum_pelvis_height_m: float = 0.64
    maximum_tilt_rad: float = 0.45
    maximum_joint_velocity_rms_rad_s: float = 0.80
    minimum_incoming_ball_speed_mps: float = 0.05
    ball_contact_policy_frame: int = 253
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    direct_joint_torque_output: bool = False
    schema_version: str = "rosclaw.growth.g1_causal_strike_option_config.v1"

    def __post_init__(self) -> None:
        integer_values = (
            self.predecessor_track_policy_frame,
            self.predecessor_commit_policy_frame,
            self.predecessor_abort_policy_frame,
            self.strike_phase_start_frame,
            self.history_prime_frames,
            self.ball_contact_policy_frame,
        )
        real_values = (
            self.bridge_duration_sec,
            self.bridge_entry_velocity_scale,
            self.bridge_exit_velocity_scale,
            self.bridge_boundary_velocity_limit_rad_s,
            self.minimum_pelvis_height_m,
            self.maximum_tilt_rad,
            self.maximum_joint_velocity_rms_rad_s,
            self.minimum_incoming_ball_speed_mps,
        )
        if any(isinstance(value, bool) for value in integer_values) or not all(
            math.isfinite(value) for value in real_values
        ):
            raise ValueError("causal strike option config must be finite and typed")
        if not (
            150
            <= self.predecessor_track_policy_frame
            < self.predecessor_commit_policy_frame
            < self.predecessor_abort_policy_frame
            <= 280
            and 100 <= self.strike_phase_start_frame <= 200
            and self.strike_phase_start_frame < self.ball_contact_policy_frame <= 300
            and 0.20 <= self.bridge_duration_sec <= 0.80
            and 0.0 <= self.bridge_entry_velocity_scale <= 1.0
            and 0.0 <= self.bridge_exit_velocity_scale <= 1.0
            and 0.5 <= self.bridge_boundary_velocity_limit_rad_s <= 4.0
            and 3 <= self.history_prime_frames <= 5
            and 0.55 <= self.minimum_pelvis_height_m <= 0.75
            and 0.20 <= self.maximum_tilt_rad <= 0.70
            and 0.20 <= self.maximum_joint_velocity_rms_rad_s <= 2.0
            and 0.01 <= self.minimum_incoming_ball_speed_mps <= 0.50
            and self.activation_ceiling == "SIM_ONLY"
            and not self.hardware_authorized
            and not self.direct_joint_torque_output
            and self.schema_version == "rosclaw.growth.g1_causal_strike_option_config.v1"
        ):
            raise ValueError("causal strike option config violates its SIM-only envelope")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass(frozen=True)
class CausalStrikeOptionObservation:
    timestamp_sec: float
    predecessor_policy_frame: int
    receiver_pelvis_height_m: float
    receiver_roll_rad: float
    receiver_pitch_rad: float
    receiver_joint_velocity_rms_rad_s: float
    receiver_ball_local_x_m: float
    receiver_ball_local_vx_mps: float
    schema_version: str = "rosclaw.growth.causal_strike_option_observation.v1"

    def __post_init__(self) -> None:
        values = (
            self.timestamp_sec,
            self.receiver_pelvis_height_m,
            self.receiver_roll_rad,
            self.receiver_pitch_rad,
            self.receiver_joint_velocity_rms_rad_s,
            self.receiver_ball_local_x_m,
            self.receiver_ball_local_vx_mps,
        )
        if (
            isinstance(self.predecessor_policy_frame, bool)
            or not 0 <= self.predecessor_policy_frame <= 1_000
            or not all(math.isfinite(value) for value in values)
            or self.timestamp_sec < 0.0
            or self.receiver_joint_velocity_rms_rad_s < 0.0
            or self.schema_version != "rosclaw.growth.causal_strike_option_observation.v1"
        ):
            raise ValueError("causal strike option observation is malformed")


@dataclass(frozen=True)
class CausalStrikeOptionDecision:
    phase: CausalStrikeOptionPhase
    begin_bridge: bool
    ready: bool
    reason: str
    incoming_ball: bool
    ball_arrival_eta_sec: float | None
    config_hash: str


class G1CausalStrikeOptionController:
    """Monotonic PREPARE→TRACK→COMMIT→RECOVER/ABORT controller."""

    def __init__(self, config: G1CausalStrikeOptionConfig) -> None:
        self.config = config
        self.phase = CausalStrikeOptionPhase.PREPARE
        self._last_timestamp_sec = -math.inf
        self._last_predecessor_policy_frame = 0
        self._terminal_reason: str | None = None

    def step(self, observation: CausalStrikeOptionObservation) -> CausalStrikeOptionDecision:
        if (
            observation.timestamp_sec + 1.0e-12 < self._last_timestamp_sec
            or observation.predecessor_policy_frame < self._last_predecessor_policy_frame
        ):
            raise ValueError("causal strike option observations must be monotonic")
        self._last_timestamp_sec = observation.timestamp_sec
        self._last_predecessor_policy_frame = observation.predecessor_policy_frame
        incoming = observation.receiver_ball_local_vx_mps <= (
            -self.config.minimum_incoming_ball_speed_mps
        )
        eta = None
        if incoming:
            eta = max(
                0.0,
                observation.receiver_ball_local_x_m
                / max(-observation.receiver_ball_local_vx_mps, 1.0e-9),
            )
        ready, reason = self._readiness(observation)
        begin_bridge = False
        if (
            self.phase == CausalStrikeOptionPhase.PREPARE
            and observation.predecessor_policy_frame >= (self.config.predecessor_track_policy_frame)
        ):
            self.phase = CausalStrikeOptionPhase.TRACK
        if self.phase == CausalStrikeOptionPhase.TRACK:
            if observation.predecessor_policy_frame >= self.config.predecessor_abort_policy_frame:
                self.phase = CausalStrikeOptionPhase.ABORTED
                self._terminal_reason = "readiness_deadline_missed"
            elif ready and observation.predecessor_policy_frame >= (
                self.config.predecessor_commit_policy_frame
            ):
                self.phase = CausalStrikeOptionPhase.COMMIT
                begin_bridge = True
                self._terminal_reason = "bounded_bridge_committed"
        if self._terminal_reason is not None:
            reason = self._terminal_reason
        return CausalStrikeOptionDecision(
            phase=self.phase,
            begin_bridge=begin_bridge,
            ready=ready,
            reason=reason,
            incoming_ball=incoming,
            ball_arrival_eta_sec=eta,
            config_hash=self.config.config_hash,
        )

    def observe_contact(self) -> None:
        if self.phase == CausalStrikeOptionPhase.COMMIT:
            self.phase = CausalStrikeOptionPhase.RECOVER
            self._terminal_reason = "ball_contact_observed"

    def _readiness(self, observation: CausalStrikeOptionObservation) -> tuple[bool, str]:
        if observation.receiver_pelvis_height_m < self.config.minimum_pelvis_height_m:
            return False, "pelvis_below_ready_envelope"
        if max(abs(observation.receiver_roll_rad), abs(observation.receiver_pitch_rad)) > (
            self.config.maximum_tilt_rad
        ):
            return False, "tilt_outside_ready_envelope"
        if observation.receiver_joint_velocity_rms_rad_s > (
            self.config.maximum_joint_velocity_rms_rad_s
        ):
            return False, "joint_velocity_outside_ready_envelope"
        return True, "ready"


__all__ = [
    "CausalStrikeOptionDecision",
    "CausalStrikeOptionObservation",
    "CausalStrikeOptionPhase",
    "G1CausalStrikeOptionConfig",
    "G1CausalStrikeOptionController",
]
