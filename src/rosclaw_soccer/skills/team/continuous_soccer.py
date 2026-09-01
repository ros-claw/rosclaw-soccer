"""Fail-closed event contracts for continuous Soccer Academy matches.

The contract intentionally sits above a simulator and below a learner.  It
does not decide how a G1 moves and it cannot authorize hardware.  Its job is
to prevent a sequence of independently reset highlights from being presented
as one continuous match.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from rosclaw_soccer.sim.contracts import hash_json

_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.:-]{0,127}$")


class SoccerEventKind(StrEnum):
    POSSESSION = "possession"
    TOUCH = "touch"
    PASS = "pass"
    INTERCEPTION = "interception"
    SHOT = "shot"
    SAVE = "save"
    GOAL = "goal"
    RESTART = "restart"
    RECOVERY_READY = "recovery_ready"


@dataclass(frozen=True)
class ContinuousSoccerConfig:
    """Frozen hierarchy and clock contract for one match."""

    duration_sec: float = 60.0
    tactical_hz: float = 10.0
    skill_hz: float = 50.0
    maximum_activity_gap_sec: float = 10.0
    terminate_on_goal: bool = False
    reset_clock_on_goal: bool = False
    minimum_distinct_agents: int = 2
    minimum_distinct_teams: int = 2
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.continuous_soccer_config.v1"

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.duration_sec)
            or not 30.0 <= self.duration_sec <= 180.0
            or not math.isfinite(self.tactical_hz)
            or not 5.0 <= self.tactical_hz <= 10.0
            or not math.isfinite(self.skill_hz)
            or not 25.0 <= self.skill_hz <= 50.0
            or self.tactical_hz >= self.skill_hz
            or not math.isfinite(self.maximum_activity_gap_sec)
            or not 1.0 <= self.maximum_activity_gap_sec <= 30.0
            or self.terminate_on_goal
            or self.reset_clock_on_goal
            or not 2 <= self.minimum_distinct_agents <= 22
            or not 2 <= self.minimum_distinct_teams <= 11
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("continuous soccer config violates the hierarchy or safety boundary")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass(frozen=True)
class SoccerMatchEvent:
    """One physics-derived event on a single monotonic match clock."""

    event_id: str
    kind: SoccerEventKind
    time_sec: float
    actor_id: str
    team_id: str
    ball_state_hash: str
    self_state_hash: str
    world_state_hash: str
    source_evidence_hash: str
    target_agent_id: str | None = None
    physics_derived: bool = True
    pixels_used_for_scoring: bool = False
    schema_version: str = "rosclaw_soccer.match_event.v1"

    def __post_init__(self) -> None:
        for label, value in (
            ("event_id", self.event_id),
            ("actor_id", self.actor_id),
            ("team_id", self.team_id),
        ):
            if not _IDENTIFIER.fullmatch(value):
                raise ValueError(f"{label} is not a normalized identifier")
        if self.target_agent_id is not None and not _IDENTIFIER.fullmatch(self.target_agent_id):
            raise ValueError("target_agent_id is not a normalized identifier")
        if not isinstance(self.kind, SoccerEventKind):
            raise ValueError("soccer match event kind is invalid")
        if not math.isfinite(self.time_sec) or self.time_sec < 0.0:
            raise ValueError("soccer match event time is invalid")
        for label, value in (
            ("ball_state_hash", self.ball_state_hash),
            ("self_state_hash", self.self_state_hash),
            ("world_state_hash", self.world_state_hash),
            ("source_evidence_hash", self.source_evidence_hash),
        ):
            if not _HASH.fullmatch(value):
                raise ValueError(f"{label} must be a sha256 content hash")
        if not self.physics_derived or self.pixels_used_for_scoring:
            raise ValueError("match events require physics telemetry and cannot use pixels")

    @property
    def event_hash(self) -> str:
        return str(hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["kind"] = self.kind.value
        return value


@dataclass(frozen=True)
class ContinuousSoccerTrace:
    """Content-bound proof that one match remained continuous across events."""

    match_id: str
    config: ContinuousSoccerConfig
    body_bundle_hash: str
    environment_hash: str
    policy_bundle_hash: str
    trajectory_hash: str
    clock_id: str
    observed_duration_sec: float
    physics_step_count: int
    events: tuple[SoccerMatchEvent, ...]
    strict_replay: bool = True
    physics_authority: str = "CPU_MUJOCO"
    activation_ceiling: str = "SIM_ONLY"
    hardware_command_sent: bool = False
    schema_version: str = "rosclaw_soccer.continuous_soccer_trace.v1"

    def __post_init__(self) -> None:
        if not _IDENTIFIER.fullmatch(self.match_id) or not _IDENTIFIER.fullmatch(self.clock_id):
            raise ValueError("continuous match identity or clock is invalid")
        for label, value in (
            ("body_bundle_hash", self.body_bundle_hash),
            ("environment_hash", self.environment_hash),
            ("policy_bundle_hash", self.policy_bundle_hash),
            ("trajectory_hash", self.trajectory_hash),
        ):
            if not _HASH.fullmatch(value):
                raise ValueError(f"{label} must be a sha256 content hash")
        if (
            not math.isfinite(self.observed_duration_sec)
            or self.observed_duration_sec < self.config.duration_sec
            or self.physics_step_count <= 0
            or not self.events
            or not self.strict_replay
            or self.physics_authority != "CPU_MUJOCO"
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_command_sent
        ):
            raise ValueError("continuous match evidence boundary is invalid")
        event_ids = tuple(event.event_id for event in self.events)
        event_times = tuple(event.time_sec for event in self.events)
        if len(event_ids) != len(set(event_ids)) or event_times != tuple(sorted(event_times)):
            raise ValueError("continuous match events must be unique and monotonic")
        if event_times[-1] > self.observed_duration_sec:
            raise ValueError("continuous match event exceeds the observed clock")
        clock_points = (0.0, *event_times, self.observed_duration_sec)
        if any(
            later - earlier > self.config.maximum_activity_gap_sec
            for earlier, later in zip(clock_points, clock_points[1:], strict=False)
        ):
            raise ValueError("continuous match contains an unobserved or idle activity gap")
        actors = {event.actor_id for event in self.events}
        if len(actors) < self.config.minimum_distinct_agents:
            raise ValueError("continuous match has insufficient participating agents")
        teams = {event.team_id for event in self.events}
        if len(teams) < self.config.minimum_distinct_teams:
            raise ValueError("continuous match has insufficient participating teams")
        self._validate_goal_continuity()

    def _validate_goal_continuity(self) -> None:
        """A goal must be followed by a restart and later play on the same clock."""

        for index, event in enumerate(self.events):
            if event.kind is not SoccerEventKind.GOAL:
                continue
            later = self.events[index + 1 :]
            if not later or later[0].kind is not SoccerEventKind.RESTART:
                raise ValueError("a continuous match goal is missing its in-episode restart")
            after_restart = later[1:]
            if not after_restart or not any(
                item.kind not in {SoccerEventKind.GOAL, SoccerEventKind.RESTART}
                for item in after_restart
            ):
                raise ValueError("a continuous match ended at a highlight instead of resuming play")

    @property
    def trace_hash(self) -> str:
        return str(hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "match_id": self.match_id,
            "config": asdict(self.config),
            "config_hash": self.config.config_hash,
            "body_bundle_hash": self.body_bundle_hash,
            "environment_hash": self.environment_hash,
            "policy_bundle_hash": self.policy_bundle_hash,
            "trajectory_hash": self.trajectory_hash,
            "clock_id": self.clock_id,
            "observed_duration_sec": self.observed_duration_sec,
            "physics_step_count": self.physics_step_count,
            "events": [event.to_dict() for event in self.events],
            "strict_replay": self.strict_replay,
            "physics_authority": self.physics_authority,
            "activation_ceiling": self.activation_ceiling,
            "hardware_command_sent": self.hardware_command_sent,
        }


__all__ = [
    "ContinuousSoccerConfig",
    "ContinuousSoccerTrace",
    "SoccerEventKind",
    "SoccerMatchEvent",
]
