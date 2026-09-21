"""Value-only receiving reference proposals, separate from motor ownership.

Providers see current measured values, never writable simulator handles. The
world's oracle cursor retains filtering, amplitude/rate limits and safety.
This opt-in research boundary neither qualifies a teacher nor permits hardware.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from rosclaw_soccer.providers.g1.locomotion_memory import LocomotionMemory
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule
from rosclaw_soccer.training.receiving_scene import ReceivingSceneContext

if TYPE_CHECKING:
    from rosclaw_soccer.skills.team.motor_option import TeamMotorTarget


@dataclass(frozen=True)
class ReceivingCaptureContext:
    """Measured capture event retained by the executor, not a success label."""

    start_time_sec: float
    duration_sec: float
    foot: int
    direction_xy: tuple[float, float]

    def __post_init__(self) -> None:
        if (
            type(self.start_time_sec) not in (int, float)
            or not math.isfinite(self.start_time_sec)
            or self.start_time_sec < 0
            or type(self.duration_sec) not in (int, float)
            or not math.isfinite(self.duration_sec)
            or not 0.2 <= self.duration_sec <= 1.0
            or type(self.foot) is not int
            or self.foot not in (1, 2)
            or type(self.direction_xy) is not tuple
            or len(self.direction_xy) != 2
            or any(
                type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 1e4
                for v in self.direction_xy
            )
            or math.hypot(*self.direction_xy) <= 1e-9
        ):
            raise ValueError("bounded measured capture context required")


@dataclass(frozen=True)
class ReceivingLocomotionContext:
    """Post-inference frozen memory and command, never a writable policy."""

    frame: int
    memory: LocomotionMemory
    configuration_hash: str
    raw_action: tuple[float, ...]
    world_command: tuple[float, ...]
    reflected: bool
    scene: ReceivingSceneContext | None = None

    def __post_init__(self) -> None:
        if (
            type(self.frame) is not int
            or not 0 <= self.frame < 1000
            or not isinstance(self.memory, LocomotionMemory)
            or type(self.configuration_hash) is not str
            or re.fullmatch(r"sha256:[0-9a-f]{64}", self.configuration_hash) is None
            or type(self.reflected) is not bool
        ):
            raise ValueError("bound current locomotion memory required")
        self.memory.__post_init__()
        if self.scene is not None:
            if not isinstance(self.scene, ReceivingSceneContext):
                raise ValueError("typed current receiving scene required")
            self.scene.__post_init__()
            if self.scene.frame != self.frame:
                raise ValueError("scene must share the locomotion clock")
        for values, size in ((self.raw_action, 29), (self.world_command, 3)):
            if (
                type(values) is not tuple
                or len(values) != size
                or any(
                    type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 100
                    for v in values
                )
            ):
                raise ValueError("bounded immutable locomotion values required")


@dataclass(frozen=True)
class ReceivingContactHistory:
    """Completed native contact history, not possession or successor readiness.

    Interruption means force-positive foreign-player or non-foot ball contact,
    or robot-robot contact. Ground/net contact is not classified as a player.
    A same-time own-foot touch and interruption must not count as clean control.
    """

    observed_through_time_sec: float
    last_interruption_time_sec: float | None

    def __post_init__(self) -> None:
        now = self.observed_through_time_sec
        interruption = self.last_interruption_time_sec
        if type(now) not in (int, float) or not math.isfinite(now) or not 0 <= now <= 20:
            raise ValueError("bounded completed contact observation clock required")
        if interruption is not None and (
            type(interruption) not in (int, float)
            or not math.isfinite(interruption)
            or not 0 <= interruption <= now + 1e-9
        ):
            raise ValueError("interruption must be a measured past contact event")


@dataclass(frozen=True)
class ReceivingFeedbackObservation:
    agent_id: str
    frame: int
    time_sec: float
    qpos: tuple[float, ...]
    qvel: tuple[float, ...]
    foundation_target: TeamMotorTarget
    native_actor_raw: tuple[float, ...]
    last_own_foot_contact_time_sec: float | None
    last_own_contact_foot: int | None
    capture_context: ReceivingCaptureContext | None = None
    committed_receive: bool = False
    previous_filtered_residual_rad: tuple[float, ...] = (0.0,) * 12
    residual_admitted: bool = False
    locomotion: ReceivingLocomotionContext | None = None
    action_substrate: str = "A0_leg12"
    previous_body_residual_rad: tuple[float, ...] | None = None
    contact_history: ReceivingContactHistory | None = None

    def __post_init__(self) -> None:
        if (
            type(self.agent_id) is not str
            or re.fullmatch(r"[a-z][a-z0-9_.:-]{0,127}", self.agent_id) is None
            or type(self.frame) is not int
            or not 0 <= self.frame < 1000
            or type(self.time_sec) not in (int, float)
            or not math.isfinite(self.time_sec)
            or abs(self.time_sec - self.frame * 0.02) > 1e-6
        ):
            raise ValueError("named current 50 Hz receiving observation required")
        for values, size in ((self.qpos, 43), (self.qvel, 41), (self.native_actor_raw, 12)):
            if (
                type(values) is not tuple
                or len(values) != size
                or any(
                    type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 1e4
                    for v in values
                )
            ):
                raise ValueError("bounded immutable measured vectors required")
        if any(abs(sum(v * v for v in self.qpos[a:b]) - 1) > 2e-4 for a, b in ((3, 7), (39, 43))):
            raise ValueError("measured unit body and ball quaternions required")
        # Team package eagerly imports the world, which imports this contract.
        # Defer the runtime type check until construction, not module import.
        from rosclaw_soccer.skills.team.motor_option import TeamMotorTarget

        if not isinstance(self.foundation_target, TeamMotorTarget):
            raise ValueError("typed current foundation target required")
        self.foundation_target.__post_init__()
        if self.locomotion is not None:
            if not isinstance(self.locomotion, ReceivingLocomotionContext):
                raise ValueError("typed locomotion context required")
            self.locomotion.__post_init__()
            if self.locomotion.frame != self.frame:
                raise ValueError("locomotion memory must share the current observation clock")
            if (
                self.locomotion.scene is not None
                and self.locomotion.scene.agent_id != self.agent_id
            ):
                raise ValueError("scene must bind the observed receiving agent")
        if type(self.committed_receive) is not bool:
            raise ValueError("explicit current receiving commitment required")
        if self.contact_history is not None:
            if not isinstance(self.contact_history, ReceivingContactHistory):
                raise ValueError("typed completed contact history required")
            self.contact_history.__post_init__()
            if abs(self.contact_history.observed_through_time_sec - self.time_sec) > 1e-9:
                raise ValueError("contact history must cover the current observation clock")
        if type(self.action_substrate) is not str or self.action_substrate not in (
            "A0_leg12",
            "A1_body29",
        ):
            raise ValueError("explicit supported feedback action substrate required")
        body = self.previous_body_residual_rad
        if self.action_substrate == "A0_leg12":
            if body is not None:
                raise ValueError("leg feedback cannot carry whole-body filter state")
        elif (
            type(body) is not tuple
            or len(body) != 29
            or any(
                type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 0.1 for v in body
            )
            or body[:12] != self.previous_filtered_residual_rad
        ):
            raise ValueError("whole-body feedback requires the complete consistent filter state")
        if (
            type(self.residual_admitted) is not bool
            or type(self.previous_filtered_residual_rad) is not tuple
            or len(self.previous_filtered_residual_rad) != 12
            or any(
                type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 0.1
                for v in self.previous_filtered_residual_rad
            )
        ):
            raise ValueError("bounded prior filter output and explicit admission required")
        if self.capture_context is not None:
            if not isinstance(self.capture_context, ReceivingCaptureContext):
                raise ValueError("typed measured capture context required")
            self.capture_context.__post_init__()
            age = self.time_sec - self.capture_context.start_time_sec
            if not -1e-9 <= age <= self.capture_context.duration_sec + 1e-9:
                raise ValueError("capture event must be current and not from the future")
        contact = self.last_own_foot_contact_time_sec
        if contact is None:
            if self.last_own_contact_foot is not None:
                raise ValueError("contact foot without measured contact time")
        elif (
            type(contact) not in (int, float)
            or not math.isfinite(contact)
            or not 0 <= contact <= self.time_sec + 1e-9
            or type(self.last_own_contact_foot) is not int
            or self.last_own_contact_foot not in (1, 2)
        ):
            raise ValueError("past measured own-foot contact required")

    @property
    def observation_hash(self) -> str:
        value = asdict(self)
        if self.contact_history is None:
            value.pop("contact_history")
        if self.action_substrate == "A0_leg12" and self.previous_body_residual_rad is None:
            value.pop("action_substrate")
            value.pop("previous_body_residual_rad")
        if self.locomotion is None:
            value.pop("locomotion")  # Preserve existing providers' observation identities.
        else:
            value["locomotion"]["memory"] = self.locomotion.memory.state_hash
            if self.locomotion.scene is None:
                value["locomotion"].pop("scene")  # Preserve memory-only observation identities.
        return str(hash_json(value))


@runtime_checkable
class ReceivingFeedbackProvider(Protocol):
    agent_id: str
    schedule_hash: str
    contract_hash: str
    activation_ceiling: str

    def propose(self, observation: ReceivingFeedbackObservation) -> tuple[float, ...]: ...


class ReceivingFeedbackSlot:
    """Fault-latched proposal boundary; failure aborts this offline experiment.

    No automatic fallback, retry, stepping, or candidate promotion occurs here.
    The execution cursor applies its unchanged low-pass filter and rate bound.
    """

    def __init__(self, provider: ReceivingFeedbackProvider, schedule: ReceivingOracleSchedule):
        if not isinstance(schedule, ReceivingOracleSchedule):
            raise ValueError("typed receiving schedule required")
        schedule.__post_init__()
        if (
            schedule.substrate not in ("A0_leg12", "A1_body29")
            or getattr(provider, "action_substrate", "A0_leg12") != schedule.substrate
            or not isinstance(provider, ReceivingFeedbackProvider)
            or provider.agent_id != schedule.agent_id
            or provider.schedule_hash != schedule.contract_hash
            or provider.activation_ceiling != "SIM_ONLY"
            or type(provider.contract_hash) is not str
            or re.fullmatch(r"sha256:[0-9a-f]{64}", provider.contract_hash) is None
        ):
            raise ValueError("SIM-only feedback must bind the exact action schedule")
        self.provider = provider
        self.requires_contact_history = getattr(provider, "requires_contact_history", False)
        if type(self.requires_contact_history) is not bool:
            raise ValueError("explicit completed-contact history requirement required")
        self.requires_locomotion_memory = getattr(provider, "requires_locomotion_memory", False)
        if type(self.requires_locomotion_memory) is not bool:
            raise ValueError("explicit locomotion memory requirement required")
        self.requires_navigation_context = getattr(provider, "requires_navigation_context", False)
        if type(self.requires_navigation_context) is not bool or (
            self.requires_navigation_context and not self.requires_locomotion_memory
        ):
            raise ValueError("navigation context requires explicit current locomotion memory")
        self.agent_id = schedule.agent_id
        self.action_substrate = schedule.substrate
        self.action_dimension = len(schedule.knots[0])
        self.schedule_hash = schedule.contract_hash
        self.contract_hash = provider.contract_hash
        self.start_frame = schedule.start_frame
        self.next_frame = 0
        self.faulted = False
        self._last_interruption_time: float | None = None

    def step(self, observation: ReceivingFeedbackObservation) -> tuple[float, ...] | None:
        if self.faulted:
            raise ValueError("receiving feedback fault is latched")
        try:
            if not isinstance(observation, ReceivingFeedbackObservation):
                raise ValueError("typed current receiving observation required")
            observation.__post_init__()
            if (
                observation.agent_id != self.agent_id
                or type(getattr(self.provider, "requires_contact_history", False)) is not bool
                or getattr(self.provider, "requires_contact_history", False)
                != self.requires_contact_history
                or (self.requires_contact_history and observation.contact_history is None)
                or observation.action_substrate != self.action_substrate
                or getattr(self.provider, "action_substrate", "A0_leg12") != self.action_substrate
                or observation.frame != self.next_frame
                or self.provider.agent_id != self.agent_id
                or self.provider.schedule_hash != self.schedule_hash
                or self.provider.contract_hash != self.contract_hash
                or self.provider.activation_ceiling != "SIM_ONLY"
                or getattr(self.provider, "requires_locomotion_memory", False)
                != self.requires_locomotion_memory
                or type(getattr(self.provider, "requires_locomotion_memory", False)) is not bool
                or (self.requires_locomotion_memory and observation.locomotion is None)
                or type(getattr(self.provider, "requires_navigation_context", False)) is not bool
                or getattr(self.provider, "requires_navigation_context", False)
                != self.requires_navigation_context
                or (
                    self.requires_navigation_context
                    and (observation.locomotion is None or observation.locomotion.scene is None)
                )
            ):
                raise ValueError("feedback identity, clock or source binding changed")
            if observation.contact_history is not None:
                interruption = observation.contact_history.last_interruption_time_sec
                if self._last_interruption_time is not None and (
                    interruption is None or interruption < self._last_interruption_time
                ):
                    raise ValueError("completed contact history cannot forget an interruption")
                self._last_interruption_time = interruption
            self.next_frame += 1
            if observation.frame < self.start_frame:
                return None
            observation_hash = observation.observation_hash
            proposal = self.provider.propose(observation)
            if (
                observation.observation_hash != observation_hash
                or type(getattr(self.provider, "requires_contact_history", False)) is not bool
                or getattr(self.provider, "requires_contact_history", False)
                != self.requires_contact_history
                or getattr(self.provider, "action_substrate", "A0_leg12") != self.action_substrate
                or self.provider.agent_id != self.agent_id
                or self.provider.schedule_hash != self.schedule_hash
                or self.provider.contract_hash != self.contract_hash
                or self.provider.activation_ceiling != "SIM_ONLY"
                or getattr(self.provider, "requires_locomotion_memory", False)
                != self.requires_locomotion_memory
                or type(getattr(self.provider, "requires_locomotion_memory", False)) is not bool
                or type(getattr(self.provider, "requires_navigation_context", False)) is not bool
                or getattr(self.provider, "requires_navigation_context", False)
                != self.requires_navigation_context
            ):
                raise ValueError("feedback changed its input or binding during proposal")
            if (
                type(proposal) is not tuple
                or len(proposal) != self.action_dimension
                or any(
                    type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 0.1
                    for v in proposal
                )
            ):
                raise ValueError(
                    "finite immutable action-bound desired residual limited to 0.1 rad required"
                )
            return proposal
        except Exception as error:
            self.faulted = True
            raise ValueError("receiving feedback rejected; experiment must stop") from error
