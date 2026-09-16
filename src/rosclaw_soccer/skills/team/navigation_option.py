"""Read-only local navigation proposals upstream of the simulation's guards.

Providers receive values, not a world, motor, robot or recurrent-state handle.
No proposal is permission to move hardware or evidence of football success.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


def _identity(value: str) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[a-z][a-z0-9_.:-]{0,127}", value) is not None


def _vector(value: tuple[float, ...], size: int) -> bool:
    return (
        type(value) is tuple
        and len(value) == size
        and all(type(v) in (int, float) and math.isfinite(v) and abs(v) <= 1e4 for v in value)
    )


@dataclass(frozen=True)
class NavigationObservation:
    agent_id: str
    frame: int
    time_sec: float
    role: str
    intent: str
    body_pose: tuple[float, ...]
    body_velocity: tuple[float, float, float]
    ball_position: tuple[float, float, float]
    ball_velocity: tuple[float, float, float]
    task_target: tuple[float, float, float]
    steering_target: tuple[float, float]
    baseline_command: tuple[float, float, float]
    previous_command: tuple[float, float, float]
    neighbors: tuple[tuple[str, float, float], ...]
    # Optional measured world-frame end-effectors, never inferred from root pose.
    effector_positions: tuple[tuple[str, float, float, float], ...] = ()
    committed_receiver: bool = False
    # Runtime lifecycle only: retirement is not evidence of skill success.
    motor_option_retired: bool = False

    def __post_init__(self) -> None:
        if (
            not _identity(self.agent_id)
            or type(self.frame) is not int
            or self.frame < 0
            or type(self.time_sec) not in (int, float)
            or not math.isfinite(self.time_sec)
            or self.time_sec < 0
            or not _identity(self.role)
            or not _identity(self.intent)
            or not _vector(self.body_pose, 7)
            or abs(sum(v * v for v in self.body_pose[3:7]) - 1) > 2e-4
            or any(
                not _vector(v, 3)
                for v in (
                    self.body_velocity,
                    self.ball_position,
                    self.ball_velocity,
                    self.task_target,
                    self.baseline_command,
                    self.previous_command,
                )
            )
            or not _vector(self.steering_target, 2)
            or type(self.neighbors) is not tuple
            or type(self.committed_receiver) is not bool
            or type(self.motor_option_retired) is not bool
            or type(self.effector_positions) is not tuple
            or len(self.effector_positions) > 16
            or any(
                type(v) is not tuple or len(v) != 4 or not _identity(v[0]) or not _vector(v[1:], 3)
                for v in self.effector_positions
            )
            or len(self.neighbors) > 31
            or any(
                type(v) is not tuple
                or len(v) != 3
                or not _identity(v[0])
                or v[0] == self.agent_id
                or not _vector(v[1:], 2)
                for v in self.neighbors
            )
        ):
            raise ValueError("finite immutable local navigation observation required")
        ids = tuple(v[0] for v in self.neighbors)
        if tuple(sorted(set(ids))) != ids:
            raise ValueError("unique sorted navigation neighbors required")
        effectors = tuple(v[0] for v in self.effector_positions)
        if tuple(sorted(set(effectors))) != effectors:
            raise ValueError("unique sorted measured end-effectors required")


@dataclass(frozen=True)
class NavigationDelta:
    agent_id: str
    frame: int
    time_sec: float
    velocity_delta: tuple[float, float, float]

    def __post_init__(self) -> None:
        if (
            not _identity(self.agent_id)
            or type(self.frame) is not int
            or self.frame < 0
            or type(self.time_sec) not in (int, float)
            or not math.isfinite(self.time_sec)
            or self.time_sec < 0
            or not _vector(self.velocity_delta, 3)
            or math.hypot(*self.velocity_delta[:2]) > 0.250000001
            or abs(self.velocity_delta[2]) > 0.400000001
        ):
            raise ValueError("bounded same-frame navigation delta required")


@runtime_checkable
class TeamNavigationPolicy(Protocol):
    agent_id: str
    contract_hash: str
    activation_ceiling: str
    foundation_hash: str
    foundation_config_hash: str

    def propose(self, observation: NavigationObservation) -> NavigationDelta | None: ...


class NavigationSlot:
    """Per-episode fault latch; zero, idle and fault remain distinguishable."""

    def __init__(self, policy: TeamNavigationPolicy) -> None:
        if (
            not isinstance(policy, TeamNavigationPolicy)
            or not _identity(policy.agent_id)
            or not isinstance(policy.contract_hash, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", policy.contract_hash) is None
            or policy.activation_ceiling != "SIM_ONLY"
            or any(
                not isinstance(v, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", v) is None
                for v in (policy.foundation_hash, policy.foundation_config_hash)
            )
        ):
            raise ValueError("content-bound private SIM navigation policy required")
        self.policy = policy
        self.agent_id, self.contract_hash = policy.agent_id, policy.contract_hash
        self.foundation_hash = policy.foundation_hash
        self.foundation_config_hash = policy.foundation_config_hash
        self.faulted = False
        self.fault_reason: str | None = None
        self.active = False
        self.delta = (0.0, 0.0, 0.0)
        self._frame: int | None = None
        self._time: float | None = None

    def propose(self, observation: NavigationObservation) -> tuple[float, float, float]:
        self.active = False
        self.delta = (0.0, 0.0, 0.0)
        if self.faulted:
            return self.delta
        try:
            if (
                not isinstance(observation, NavigationObservation)
                or observation.agent_id != self.agent_id
                or self.policy.agent_id != self.agent_id
                or self.policy.contract_hash != self.contract_hash
                or self.policy.activation_ceiling != "SIM_ONLY"
                or self.policy.foundation_hash != self.foundation_hash
                or self.policy.foundation_config_hash != self.foundation_config_hash
                or self._frame is not None
                and observation.frame != self._frame + 1
                or self._time is not None
                and abs(observation.time_sec - self._time - 0.02) > 1e-6
            ):
                raise ValueError("navigation identity, clock or content binding changed")
            observation.__post_init__()
            self._frame, self._time = observation.frame, observation.time_sec
            proposal = self.policy.propose(observation)
            if proposal is None:
                return self.delta
            if not isinstance(proposal, NavigationDelta):
                raise ValueError("typed navigation proposal required")
            proposal = NavigationDelta(
                proposal.agent_id, proposal.frame, proposal.time_sec, proposal.velocity_delta
            )
            if (
                proposal.agent_id != observation.agent_id
                or proposal.frame != observation.frame
                or abs(proposal.time_sec - observation.time_sec) > 1e-9
            ):
                raise ValueError("navigation proposal is stale or foreign")
            self.active = True
            self.delta = proposal.velocity_delta
            return self.delta
        except Exception as exc:
            self.faulted = True
            self.fault_reason = type(exc).__name__
            return self.delta
