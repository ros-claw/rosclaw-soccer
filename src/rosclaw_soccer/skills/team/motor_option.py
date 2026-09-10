"""Read-only, per-player motor proposal boundary inside shared simulation.

This is a Soccer adapter contract, not a robot driver or policy promotion gate.
The shared world retains all physics stepping and torque/position guards.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


def motor_blocks_residual(
    *, registered: bool, proposed: bool, faulted: bool, allow_idle_fallback: bool
) -> bool:
    """Exclusive control ownership; a failed motor never grants learning fallback.

    Idle fallback is an explicit simulation experiment, not a safety recovery.
    The caller must also clear residual filter history whenever this blocks.
    """
    if any(type(v) is not bool for v in (registered, proposed, faulted, allow_idle_fallback)):
        raise ValueError("explicit boolean motor ownership required")
    if not registered and (proposed or faulted):
        raise ValueError("unregistered motor cannot propose or fault")
    return registered and (proposed or faulted or not allow_idle_fallback)


@dataclass(frozen=True)
class TeamMotorObservation:
    agent_id: str
    frame: int
    time_sec: float
    intent: str
    prospective_owner: bool
    qpos: tuple[float, ...]
    qvel: tuple[float, ...]
    target_position_m: tuple[float, float, float]
    # Post-clearance world vx, vy and yaw rate, not an unguarded goal vector.
    navigation_command: tuple[float, float, float] | None = None
    # Team handshake context, never a claim that reception already happened.
    committed_receiver: bool = False
    # Read-only current frozen inference, not a simulator or recurrent-state handle.
    foundation: TeamMotorFoundation | None = None

    def __post_init__(self) -> None:
        if (
            re.fullmatch(r"[a-z][a-z0-9_.:-]{0,127}", self.agent_id) is None
            or type(self.frame) is not int
            or self.frame < 0
            or type(self.prospective_owner) is not bool
            or type(self.committed_receiver) is not bool
            or self.intent not in {"shoot", "pass", "carry", "other"}
            or type(self.qpos) is not tuple
            or type(self.qvel) is not tuple
            or type(self.target_position_m) is not tuple
            or len(self.qpos) != 43
            or len(self.qvel) != 41
            or len(self.target_position_m) != 3
            or any(
                type(v) not in (float, int) or not math.isfinite(v)
                for v in (*self.qpos, *self.qvel, *self.target_position_m, self.time_sec)
            )
            or self.time_sec < 0
        ):
            raise ValueError("finite immutable shared motor observation required")
        if self.navigation_command is not None:
            command = self.navigation_command
            if (
                type(command) is not tuple
                or len(command) != 3
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in command)
                or math.hypot(*command[:2]) > 0.700000001
                or abs(command[2]) > 1.500000001
            ):
                raise ValueError("bounded post-clearance navigation command required")
        if self.foundation is not None and (
            not isinstance(self.foundation, TeamMotorFoundation)
            or self.foundation.agent_id != self.agent_id
            or self.foundation.frame != self.frame
        ):
            raise ValueError("foundation proposal belongs to another player or frame")


@dataclass(frozen=True)
class TeamMotorTarget:
    target_rad: tuple[float, ...]
    kp: tuple[float, ...]
    kd: tuple[float, ...]

    def __post_init__(self) -> None:
        for values, bound in ((self.target_rad, 10), (self.kp, 300), (self.kd, 30)):
            if (
                type(values) is not tuple
                or len(values) != 29
                or any(type(v) not in (float, int) or not math.isfinite(v) for v in values)
                or any(abs(v) > bound for v in values)
            ):
                raise ValueError("bounded immutable joint target/gain proposal required")
        if any(v < 0 for v in (*self.kp, *self.kd)):
            raise ValueError("motor gains cannot be negative")


@dataclass(frozen=True)
class TeamMotorFoundation:
    """Immutable same-player/tick fallback proposal; no LSTM reset or mutation.

    The hash identifies the frozen policy artifact, not a hardware permit or
    a proof that a residual trained against it transfers to this world.
    """

    agent_id: str
    frame: int
    target: TeamMotorTarget
    default_angles: tuple[float, ...]
    policy_hash: str
    configuration_hash: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.agent_id, str)
            or re.fullmatch(r"[a-z][a-z0-9_.:-]{0,127}", self.agent_id) is None
            or type(self.frame) is not int
            or self.frame < 0
            or not isinstance(self.target, TeamMotorTarget)
            or type(self.default_angles) is not tuple
            or len(self.default_angles) != 29
            or any(
                type(v) not in (float, int) or not math.isfinite(v) or abs(v) > 10
                for v in self.default_angles
            )
            or not isinstance(self.policy_hash, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", self.policy_hash) is None
            or not isinstance(self.configuration_hash, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", self.configuration_hash) is None
        ):
            raise ValueError("finite immutable player-bound foundation proposal required")


class TeamMotorOption(Protocol):
    """One instance per player; no model/data handles cross this boundary."""

    @property
    def contract_hash(self) -> str: ...

    def propose(self, observation: TeamMotorObservation) -> TeamMotorTarget | None: ...


@dataclass(frozen=True)
class TeamMotorReadiness:
    """Deny new ball commitments while a local motor skill still owns control.

    Ready is not permission to act: possession, handshake and physical guards
    remain mandatory. The snapshot neither chooses a teammate nor an action.
    """

    agent_id: str
    frame: int
    time_sec: float
    ball_action_ready: bool

    def __post_init__(self) -> None:
        if (
            not isinstance(self.agent_id, str)
            or re.fullmatch(r"[a-z][a-z0-9_.:-]{0,127}", self.agent_id) is None
            or type(self.frame) is not int
            or self.frame < 0
            or type(self.time_sec) not in (int, float)
            or not math.isfinite(self.time_sec)
            or self.time_sec < 0
            or type(self.ball_action_ready) is not bool
        ):
            raise ValueError("finite player-bound motor readiness required")


@runtime_checkable
class TeamMotorReadinessProvider(Protocol):
    def readiness(self, *, frame: int, time_sec: float) -> TeamMotorReadiness: ...


def motor_ball_action_ready(
    motor: TeamMotorReadinessProvider, *, agent_id: str, frame: int, time_sec: float
) -> bool:
    """Read current readiness; a stale/foreign/malformed reply must fail closed."""
    expected = TeamMotorReadiness(agent_id, frame, time_sec, False)
    reply = motor.readiness(frame=frame, time_sec=time_sec)
    if (
        not isinstance(reply, TeamMotorReadiness)
        or reply.agent_id != expected.agent_id
        or reply.frame != expected.frame
        or abs(reply.time_sec - expected.time_sec) > 1e-9
    ):
        raise ValueError("motor readiness belongs to another player, frame or clock")
    return reply.ball_action_ready


@dataclass(frozen=True)
class TeamBallContact:
    """Measured non-ground ball counterpart, not possession or handoff approval."""

    geometry_id: int
    agent_id: str | None
    effector: str
    normal_force_n: float

    def __post_init__(self) -> None:
        if (
            type(self.geometry_id) is not int
            or self.geometry_id < 0
            or type(self.effector) is not str
            or self.effector
            not in {"left_foot", "right_foot", "left_hand", "right_hand", "body", "environment"}
            or type(self.normal_force_n) not in (int, float)
            or not math.isfinite(self.normal_force_n)
            or self.normal_force_n < 0
            or (self.agent_id is None) != (self.effector == "environment")
            or (
                self.agent_id is not None
                and (
                    type(self.agent_id) is not str
                    or re.fullmatch(r"[a-z][a-z0-9_.:-]{0,127}", self.agent_id) is None
                )
            )
        ):
            raise ValueError("finite immutable ball-contact identity required")

    @property
    def is_foot(self) -> bool:
        return self.effector in {"left_foot", "right_foot"}


@dataclass(frozen=True)
class TeamMotorPhysicsObservation:
    """One measured physics step, with this player's feet as contact source.

    Other means any non-floor counterpart, including another player's feet.
    Body safety covers every body in the shared world, not just the proposer.
    """

    time_sec: float
    qpos: tuple[float, ...]
    qvel: tuple[float, ...]
    world_bodies_safe: bool
    foot_normal_force_n: float
    other_non_ground_normal_force_n: float
    # Optional complete attribution; legacy aggregates keep their old meaning.
    observer_agent_id: str | None = None
    contacts_complete: bool = False
    ball_contacts: tuple[TeamBallContact, ...] = ()

    def __post_init__(self) -> None:
        if (
            type(self.qpos) is not tuple
            or type(self.qvel) is not tuple
            or len(self.qpos) != 43
            or len(self.qvel) != 41
            or type(self.world_bodies_safe) is not bool
            or any(
                type(v) not in (float, int) or not math.isfinite(v)
                for v in (
                    *self.qpos,
                    *self.qvel,
                    self.time_sec,
                    self.foot_normal_force_n,
                    self.other_non_ground_normal_force_n,
                )
            )
            or min(self.time_sec, self.foot_normal_force_n, self.other_non_ground_normal_force_n)
            < 0
        ):
            raise ValueError("finite immutable physical motor evidence required")
        if (
            type(self.contacts_complete) is not bool
            or type(self.ball_contacts) is not tuple
            or len(self.ball_contacts) > 1024
            or any(not isinstance(c, TeamBallContact) for c in self.ball_contacts)
        ):
            raise ValueError("immutable complete ball-contact records required")
        if not self.contacts_complete:
            if self.observer_agent_id is not None or self.ball_contacts:
                raise ValueError("partial contact attribution cannot imply completeness")
            return
        if (
            type(self.observer_agent_id) is not str
            or re.fullmatch(r"[a-z][a-z0-9_.:-]{0,127}", self.observer_agent_id) is None
        ):
            raise ValueError("contact attribution requires the observing player identity")
        own_foot = max(
            (
                c.normal_force_n
                for c in self.ball_contacts
                if c.agent_id == self.observer_agent_id and c.is_foot
            ),
            default=0.0,
        )
        other = max(
            (
                c.normal_force_n
                for c in self.ball_contacts
                if not (c.agent_id == self.observer_agent_id and c.is_foot)
            ),
            default=0.0,
        )
        if own_foot != self.foot_normal_force_n or other != self.other_non_ground_normal_force_n:
            raise ValueError("contact identities disagree with unchanged force aggregates")


@runtime_checkable
class TeamMotorPhysicsObserver(Protocol):
    """Optional evidence consumer; never receives writable simulator handles."""

    def observe_physics(self, observation: TeamMotorPhysicsObservation) -> None: ...
