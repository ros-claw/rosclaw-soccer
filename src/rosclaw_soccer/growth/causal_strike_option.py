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
    """SIM-only authority envelope for an arrival-conditioned strike option."""

    predecessor_track_policy_frame: int = 155
    predecessor_commit_policy_frame: int = 170
    predecessor_abort_policy_frame: int = 185
    minimum_strike_phase_start_frame: int = 100
    maximum_strike_phase_start_frame: int = 100
    ball_contact_policy_frame: int = 253
    policy_dt_sec: float = 0.02
    bridge_duration_sec: float = 0.60
    bridge_entry_velocity_scale: float = 0.20
    bridge_exit_velocity_scale: float = 1.0
    bridge_boundary_velocity_limit_rad_s: float = 2.0
    history_prime_frames: int = 5
    minimum_pelvis_height_m: float = 0.64
    maximum_tilt_rad: float = 0.45
    maximum_joint_velocity_rms_rad_s: float = 0.80
    minimum_incoming_ball_speed_mps: float = 0.05
    minimum_incoming_observations: int = 5
    receiver_contact_pocket_x_m: float = 1.25
    minimum_ball_arrival_eta_sec: float = 0.55
    maximum_ball_arrival_eta_sec: float = 2.50
    arrival_alignment_start_policy_frame: int = 184
    arrival_alignment_tolerance_sec: float = 0.08
    maximum_arrival_hold_frames: int = 12
    maximum_arrival_advance_frames: int = 12
    missing_ball_abort_predecessor_policy_frame: int = 270
    abort_recovery_blend_frames: int = 12
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    direct_joint_torque_output: bool = False
    schema_version: str = "rosclaw.growth.g1_causal_strike_option_config.v2"

    def __post_init__(self) -> None:
        integer_values = (
            self.predecessor_track_policy_frame,
            self.predecessor_commit_policy_frame,
            self.predecessor_abort_policy_frame,
            self.minimum_strike_phase_start_frame,
            self.maximum_strike_phase_start_frame,
            self.history_prime_frames,
            self.ball_contact_policy_frame,
            self.minimum_incoming_observations,
            self.arrival_alignment_start_policy_frame,
            self.maximum_arrival_hold_frames,
            self.maximum_arrival_advance_frames,
            self.missing_ball_abort_predecessor_policy_frame,
            self.abort_recovery_blend_frames,
        )
        real_values = (
            self.policy_dt_sec,
            self.bridge_duration_sec,
            self.bridge_entry_velocity_scale,
            self.bridge_exit_velocity_scale,
            self.bridge_boundary_velocity_limit_rad_s,
            self.minimum_pelvis_height_m,
            self.maximum_tilt_rad,
            self.maximum_joint_velocity_rms_rad_s,
            self.minimum_incoming_ball_speed_mps,
            self.receiver_contact_pocket_x_m,
            self.minimum_ball_arrival_eta_sec,
            self.maximum_ball_arrival_eta_sec,
            self.arrival_alignment_tolerance_sec,
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
            <= 400
            and 80
            <= self.minimum_strike_phase_start_frame
            <= self.maximum_strike_phase_start_frame
            < self.ball_contact_policy_frame
            <= 300
            and self.policy_dt_sec == 0.02
            and 0.20 <= self.bridge_duration_sec <= 0.80
            and 0.0 <= self.bridge_entry_velocity_scale <= 1.0
            and 0.0 <= self.bridge_exit_velocity_scale <= 1.0
            and 0.5 <= self.bridge_boundary_velocity_limit_rad_s <= 4.0
            and 3 <= self.history_prime_frames <= 5
            and 0.55 <= self.minimum_pelvis_height_m <= 0.75
            and 0.20 <= self.maximum_tilt_rad <= 0.70
            and 0.20 <= self.maximum_joint_velocity_rms_rad_s <= 2.0
            and 0.01 <= self.minimum_incoming_ball_speed_mps <= 0.50
            and 2 <= self.minimum_incoming_observations <= 10
            and 0.80 <= self.receiver_contact_pocket_x_m <= 1.60
            and 0.20 <= self.minimum_ball_arrival_eta_sec < self.maximum_ball_arrival_eta_sec
            and self.maximum_ball_arrival_eta_sec <= 4.0
            and self.minimum_strike_phase_start_frame
            < self.arrival_alignment_start_policy_frame
            < self.ball_contact_policy_frame
            and 0.02 <= self.arrival_alignment_tolerance_sec <= 0.30
            and 0 <= self.maximum_arrival_hold_frames <= 30
            and 0 <= self.maximum_arrival_advance_frames <= 30
            and self.predecessor_commit_policy_frame
            < self.missing_ball_abort_predecessor_policy_frame
            <= 400
            and 5 <= self.abort_recovery_blend_frames <= 30
            and self.activation_ceiling == "SIM_ONLY"
            and not self.hardware_authorized
            and not self.direct_joint_torque_output
            and self.schema_version == "rosclaw.growth.g1_causal_strike_option_config.v2"
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
    strike_phase_start_frame: int | None
    incoming_observation_count: int
    config_hash: str


class G1CausalStrikeOptionController:
    """Monotonic PREPARE→TRACK→COMMIT→RECOVER/ABORT controller."""

    def __init__(self, config: G1CausalStrikeOptionConfig) -> None:
        self.config = config
        self.phase = CausalStrikeOptionPhase.PREPARE
        self._last_timestamp_sec = -math.inf
        self._last_predecessor_policy_frame = 0
        self._terminal_reason: str | None = None
        self._incoming_observation_count = 0
        self._last_ball_arrival_eta_sec: float | None = None
        self._arrival_hold_count = 0
        self._arrival_advance_count = 0
        self._stable_incoming_observed = False

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
        self._incoming_observation_count = self._incoming_observation_count + 1 if incoming else 0
        self._stable_incoming_observed = self._stable_incoming_observed or (
            self._incoming_observation_count >= self.config.minimum_incoming_observations
        )
        eta = None
        if incoming:
            eta = max(
                0.0,
                (observation.receiver_ball_local_x_m - self.config.receiver_contact_pocket_x_m)
                / max(-observation.receiver_ball_local_vx_mps, 1.0e-9),
            )
        self._last_ball_arrival_eta_sec = eta
        ready, reason = self._readiness(observation, eta=eta)
        begin_bridge = False
        strike_phase_start_frame = None
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
                strike_phase_start_frame = self.config.minimum_strike_phase_start_frame
        elif (
            self.phase == CausalStrikeOptionPhase.COMMIT
            and not self._stable_incoming_observed
            and observation.predecessor_policy_frame
            >= self.config.missing_ball_abort_predecessor_policy_frame
        ):
            self.phase = CausalStrikeOptionPhase.ABORTED
            self._terminal_reason = "incoming_ball_deadline_missed"
        if self._terminal_reason is not None:
            reason = self._terminal_reason
        return CausalStrikeOptionDecision(
            phase=self.phase,
            begin_bridge=begin_bridge,
            ready=ready,
            reason=reason,
            incoming_ball=incoming,
            ball_arrival_eta_sec=eta,
            strike_phase_start_frame=strike_phase_start_frame,
            incoming_observation_count=self._incoming_observation_count,
            config_hash=self.config.config_hash,
        )

    def observe_contact(self) -> None:
        if self.phase == CausalStrikeOptionPhase.COMMIT:
            self.phase = CausalStrikeOptionPhase.RECOVER
            self._terminal_reason = "ball_contact_observed"

    @property
    def stable_incoming_observed(self) -> bool:
        return self._stable_incoming_observed

    def align_repeat_count(self, *, policy_frame: int, nominal_repeat: int) -> tuple[int, int]:
        """Return a bounded causal phase correction and its {-1,0,+1} direction."""

        if isinstance(policy_frame, bool) or isinstance(nominal_repeat, bool):
            raise ValueError("arrival alignment frames must be integers")
        if policy_frame < 0 or nominal_repeat < 0:
            raise ValueError("arrival alignment frames cannot be negative")
        eta = self._last_ball_arrival_eta_sec
        if (
            self.phase != CausalStrikeOptionPhase.COMMIT
            or eta is None
            or self._incoming_observation_count < self.config.minimum_incoming_observations
            or not self.config.arrival_alignment_start_policy_frame
            <= policy_frame
            < self.config.ball_contact_policy_frame
        ):
            return nominal_repeat, 0
        motion_eta = (
            self.config.ball_contact_policy_frame - policy_frame
        ) * self.config.policy_dt_sec
        if (
            eta > motion_eta + self.config.arrival_alignment_tolerance_sec
            and self._arrival_hold_count < self.config.maximum_arrival_hold_frames
        ):
            self._arrival_hold_count += 1
            return 0, -1
        if (
            eta < motion_eta - self.config.arrival_alignment_tolerance_sec
            and self._arrival_advance_count < self.config.maximum_arrival_advance_frames
        ):
            self._arrival_advance_count += 1
            return max(2, nominal_repeat + 1), 1
        return nominal_repeat, 0

    def _readiness(
        self,
        observation: CausalStrikeOptionObservation,
        *,
        eta: float | None,
    ) -> tuple[bool, str]:
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
