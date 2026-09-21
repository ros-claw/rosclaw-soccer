"""Causal measured control streak for a successor *probe*, never READY authority.

Foot origins and joint margin must be synchronously measured from the observed
state by the caller. This value-only monitor does not authenticate geometry,
prove support/COM stability, choose a motor action or replace the receiving exam.
"""

import math
import re
from dataclasses import dataclass

from rosclaw_soccer.training.receiving_feedback import ReceivingFeedbackObservation


@dataclass(frozen=True)
class ReceivingControlWitnessResult:
    observation_hash: str
    clean_own_contact: bool
    control_streak_sec: float
    handoff_probe_admissible: bool
    successor_ready_verified: bool = False


class ReceivingControlWitness:
    """One ordered, agent-bound stream; a gap or mutation cannot preserve a streak."""

    def __init__(self, agent_id: str, *, start_frame: int, minimum_control_sec: float = 0.5):
        if (
            type(agent_id) is not str
            or re.fullmatch(r"[a-z][a-z0-9_.:-]{0,127}", agent_id) is None
            or type(start_frame) is not int
            or not 0 <= start_frame < 1000
            or type(minimum_control_sec) not in (int, float)
            or not math.isfinite(minimum_control_sec)
            or not 0.5 <= minimum_control_sec <= 2.0
        ):
            raise ValueError("bound receiving stream and conservative hold interval required")
        self._agent_id = agent_id
        self._next_frame = start_frame
        self._minimum_control_sec = minimum_control_sec
        self.faulted = False
        self._control_since: float | None = None
        self._last_interruption: float | None = None

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def next_frame(self) -> int:
        return self._next_frame

    @property
    def minimum_control_sec(self) -> float:
        return self._minimum_control_sec

    def observe(
        self,
        observation: ReceivingFeedbackObservation,
        *,
        foot_positions: tuple[tuple[float, float, float], tuple[float, float, float]],
        minimum_joint_margin_rad: float,
    ) -> ReceivingControlWitnessResult:
        if self.faulted:
            raise ValueError("receiving control witness fault is latched")
        try:
            if not isinstance(observation, ReceivingFeedbackObservation):
                raise ValueError("typed current receiving state required")
            observation.__post_init__()
            if (
                observation.agent_id != self.agent_id
                or observation.frame != self.next_frame
                or observation.contact_history is None
            ):
                raise ValueError("ordered own-state stream with complete contact history required")
            if (
                type(foot_positions) is not tuple
                or len(foot_positions) != 2
                or any(type(p) is not tuple or len(p) != 3 for p in foot_positions)
                or any(
                    type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 1e4
                    for p in foot_positions
                    for v in p
                )
                or type(minimum_joint_margin_rad) not in (int, float)
                or not math.isfinite(minimum_joint_margin_rad)
                or abs(minimum_joint_margin_rad) > 1e4
            ):
                raise ValueError("finite synchronous foot origins and joint margin required")
            interruption = observation.contact_history.last_interruption_time_sec
            if self._last_interruption is not None and (
                interruption is None or interruption < self._last_interruption
            ):
                raise ValueError("completed interruption history cannot regress")
            if interruption != self._last_interruption:
                self._control_since = None
            self._last_interruption = interruption
            own_contact = observation.last_own_foot_contact_time_sec
            clean = own_contact is not None and (
                interruption is None or own_contact > interruption + 1e-9
            )
            q = observation.qpos
            ball = q[36:39]
            distance = min(math.dist(p, ball) for p in foot_positions)
            speed = math.sqrt(sum(v * v for v in observation.qvel[35:38]))
            measured_safe = (
                q[2] >= 0.55
                and 1 - 2 * (q[4] ** 2 + q[5] ** 2) >= math.cos(0.8)
                and minimum_joint_margin_rad >= 0
            )
            controlled = (
                clean
                and measured_safe
                and distance <= 0.35
                and speed <= 0.35
                and 0 <= ball[2] <= 0.2
            )
            if not controlled:
                self._control_since = None
            elif self._control_since is None:
                self._control_since = observation.time_sec
            duration = (
                0.0 if self._control_since is None else observation.time_sec - self._control_since
            )
            self._next_frame += 1
            return ReceivingControlWitnessResult(
                observation.observation_hash,
                clean,
                duration,
                controlled and duration + 1e-9 >= self.minimum_control_sec,
            )
        except (ValueError, TypeError, OverflowError):
            self.faulted = True
            self._control_since = None
            raise
