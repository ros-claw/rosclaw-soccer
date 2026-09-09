"""Read-only, per-player motor proposal boundary inside shared simulation.

This is a Soccer adapter contract, not a robot driver or policy promotion gate.
The shared world retains all physics stepping and torque/position guards.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


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

    def __post_init__(self) -> None:
        if (
            re.fullmatch(r"[a-z][a-z0-9_.:-]{0,127}", self.agent_id) is None
            or type(self.frame) is not int
            or self.frame < 0
            or type(self.prospective_owner) is not bool
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


class TeamMotorOption(Protocol):
    """One instance per player; no model/data handles cross this boundary."""

    @property
    def contract_hash(self) -> str: ...

    def propose(self, observation: TeamMotorObservation) -> TeamMotorTarget | None: ...


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


@runtime_checkable
class TeamMotorPhysicsObserver(Protocol):
    """Optional evidence consumer; never receives writable simulator handles."""

    def observe_physics(self, observation: TeamMotorPhysicsObservation) -> None: ...
