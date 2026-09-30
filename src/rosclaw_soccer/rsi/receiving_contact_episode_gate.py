"""Read-only, event-relative receiving phase clock for SIM_ONLY team play."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

from rosclaw_soccer.sim.contracts import hash_json


@dataclass(frozen=True)
class ReceivingContactEpisode:
    arm_frame: int
    arm_time_sec: float
    contact_frame: int | None
    contact_time_sec: float | None
    end_frame: int
    end_time_sec: float
    outcome: str


@dataclass
class ReceivingContactEpisodeGate:
    """Track causal approach/foot-contact/release without owning a motor."""

    agent_id: str
    arm_radius_m: float = 0.42
    minimum_closing_speed_mps: float = 0.25
    maximum_approach_sec: float = 0.80
    recovery_sec: float = 0.32
    state: str = field(init=False, default="WAITING")
    last_frame: int | None = field(init=False, default=None)
    arm_frame: int | None = field(init=False, default=None)
    arm_time_sec: float | None = field(init=False, default=None)
    contact_frame: int | None = field(init=False, default=None)
    contact_time_sec: float | None = field(init=False, default=None)
    episodes: list[ReceivingContactEpisode] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        if (
            type(self.agent_id) is not str
            or re.fullmatch(r"[a-z][a-z0-9_.:-]{0,127}", self.agent_id) is None
            or any(
                type(value) is not float or not math.isfinite(value)
                for value in (
                    self.arm_radius_m,
                    self.minimum_closing_speed_mps,
                    self.maximum_approach_sec,
                    self.recovery_sec,
                )
            )
            or not 0.25 <= self.arm_radius_m <= 0.70
            or not 0.10 <= self.minimum_closing_speed_mps <= 1.0
            or not 0.3 <= self.maximum_approach_sec <= 1.2
            or not 0.1 <= self.recovery_sec <= 0.5
        ):
            raise ValueError("bounded measured receiving episode gate required")

    @property
    def contract_hash(self) -> str:
        return str(
            hash_json(
                {
                    "schema": "rosclaw_soccer.rsi.receiving_contact_episode_gate.v1",
                    "agent_id": self.agent_id,
                    "arm_radius_m": self.arm_radius_m,
                    "minimum_closing_speed_mps": self.minimum_closing_speed_mps,
                    "maximum_approach_sec": self.maximum_approach_sec,
                    "recovery_sec": self.recovery_sec,
                    "authority": "READ_ONLY_SIM_ONLY",
                }
            )
        )

    def observe(
        self,
        *,
        frame: int,
        time_sec: float,
        ball_relative_position_xy_m: tuple[float, float],
        ball_relative_velocity_xy_mps: tuple[float, float],
        own_foot_contact: bool,
    ) -> str:
        """Advance one 50 Hz measured frame; never predict or command motion."""
        if (
            type(frame) is not int
            or frame < 0
            or self.last_frame is not None
            and frame != self.last_frame + 1
            or type(time_sec) is not float
            or not math.isfinite(time_sec)
            or abs(time_sec - 0.02 * (frame + 1)) > 0.001
            or type(own_foot_contact) is not bool
            or any(
                type(vector) is not tuple
                or len(vector) != 2
                or any(type(value) is not float or not math.isfinite(value) for value in vector)
                for vector in (
                    ball_relative_position_xy_m,
                    ball_relative_velocity_xy_mps,
                )
            )
        ):
            raise ValueError("sequential same-frame finite receiving measurements required")
        self.last_frame = frame
        relative = np.asarray(ball_relative_position_xy_m)
        velocity = np.asarray(ball_relative_velocity_xy_mps)
        distance = float(np.linalg.norm(relative))
        closing_speed = -float(np.dot(relative, velocity)) / distance if distance > 1e-9 else 0.0
        if self.state == "WAITING":
            if distance <= self.arm_radius_m and closing_speed >= self.minimum_closing_speed_mps:
                self.state = "APPROACH"
                self.arm_frame = frame
                self.arm_time_sec = time_sec
        elif self.state == "APPROACH":
            if self.arm_time_sec is None or self.arm_frame is None:
                raise ValueError("unbound receiving approach state")
            if own_foot_contact:
                self.state = "CONTACT"
                self.contact_frame = frame
                self.contact_time_sec = time_sec
            elif time_sec - self.arm_time_sec > self.maximum_approach_sec:
                self.episodes.append(
                    ReceivingContactEpisode(
                        self.arm_frame,
                        self.arm_time_sec,
                        None,
                        None,
                        frame,
                        time_sec,
                        "TIMED_OUT_WITHOUT_FOOT_CONTACT",
                    )
                )
                self.state = "WAITING"
                self.arm_frame = None
                self.arm_time_sec = None
        elif self.state == "CONTACT":
            if (
                self.arm_time_sec is None
                or self.arm_frame is None
                or self.contact_time_sec is None
                or self.contact_frame is None
            ):
                raise ValueError("unbound measured contact state")
            if time_sec - self.contact_time_sec >= self.recovery_sec:
                self.episodes.append(
                    ReceivingContactEpisode(
                        self.arm_frame,
                        self.arm_time_sec,
                        self.contact_frame,
                        self.contact_time_sec,
                        frame,
                        time_sec,
                        "MEASURED_FOOT_CONTACT_AND_RECOVERY",
                    )
                )
                self.state = "WAITING"
                self.arm_frame = None
                self.arm_time_sec = None
                self.contact_frame = None
                self.contact_time_sec = None
        return self.state
