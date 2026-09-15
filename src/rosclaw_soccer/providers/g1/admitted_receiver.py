"""One-shot causal admission of an existing learned receive policy in simulation.

This adapter does not learn, navigate, reset a recurrent foundation, or claim
reception. Its 100-tick proposal must still be assessed using physical contacts.
"""

from __future__ import annotations

import math

import numpy as np

from rosclaw_soccer.providers.g1.recurrent_receiver import G1RecurrentReceiver
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.skills.team.motor_option import (
    TeamMotorObservation,
    TeamMotorReadiness,
    TeamMotorTarget,
)


def _validate_minimum_speed(value: float) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or not 0.05 <= value <= 1.0:
        raise ValueError("bounded explicit incoming speed threshold required")


def incoming_receive_admissible(
    observation: TeamMotorObservation, *, minimum_speed_mps: float = 0.4
) -> bool:
    """Accepted handshake plus measured incoming ball and upright body."""
    if not isinstance(observation, TeamMotorObservation):
        raise ValueError("immutable measured motor observation required")
    _validate_minimum_speed(minimum_speed_mps)
    q, v = np.asarray(observation.qpos), np.asarray(observation.qvel)
    relative = q[36:38] - q[:2]
    velocity = v[35:37]
    return bool(
        observation.committed_receiver
        and 0.15 < np.linalg.norm(relative) <= 1.2
        and np.linalg.norm(velocity) > minimum_speed_mps
        and relative @ velocity < 0
        and q[2] > 0.65
        and abs(np.linalg.norm(q[3:7]) - 1) <= 1e-4
        and 1 - 2 * (q[4] ** 2 + q[5] ** 2) > 0.9
    )


class AdmittedRecurrentReceiver:
    """Private model instance; no automatic retry or rearm after completion/fault."""

    def __init__(
        self,
        receiver: G1RecurrentReceiver,
        *,
        admission_enabled: bool = True,
        minimum_speed_mps: float = 0.4,
    ) -> None:
        if not isinstance(receiver, G1RecurrentReceiver) or type(admission_enabled) is not bool:
            raise ValueError("content-bound recurrent receiver required")
        self.receiver = receiver
        _validate_minimum_speed(minimum_speed_mps)
        self.minimum_speed_mps = minimum_speed_mps
        self.admission_enabled = admission_enabled
        self.agent_id = receiver.agent_id
        self.contract_hash = str(
            hash_json(
                {
                    "schema": "soccer.admitted_recurrent_receiver.v1",
                    "receiver": receiver.contract_hash,
                    "admission_enabled": admission_enabled,
                    **(
                        {"minimum_speed_mps": minimum_speed_mps} if minimum_speed_mps != 0.4 else {}
                    ),
                    "admission": (
                        "accepted_incoming_0.15_1.2_speed_0.4_height_0.65_upright_0.9"
                        if minimum_speed_mps == 0.4
                        else (
                            f"accepted_incoming_0.15_1.2_speed_{minimum_speed_mps}"
                            "_height_0.65_upright_0.9"
                        )
                    ),
                    "duration_frames": 100,
                    "automatic_rearm": False,
                    "navigation_override": False,
                    "activation_ceiling": "SIM_ONLY",
                }
            )
        )
        self.start_frame: int | None = None
        self.completed = False
        self.faulted = False
        self._last_frame: int | None = None

    def readiness(self, *, frame: int, time_sec: float) -> TeamMotorReadiness:
        return TeamMotorReadiness(
            self.agent_id,
            frame,
            time_sec,
            not self.faulted and (self.start_frame is None or self.completed),
        )

    def propose(self, observation: TeamMotorObservation) -> TeamMotorTarget | None:
        if self.faulted:
            raise ValueError("receiver admission remains faulted")
        try:
            if (
                observation.agent_id != self.agent_id
                or self._last_frame is not None
                and observation.frame != self._last_frame + 1
            ):
                raise ValueError("consecutive same-player admission observations required")
            # Validate the immutable foundation even while idle; no stale model
            # identity silently accepted until a ball happens to arrive.
            self.receiver._validate(observation)
            self._last_frame = observation.frame
            if self.completed:
                return None
            if self.start_frame is None:
                if not self.admission_enabled or not incoming_receive_admissible(
                    observation, minimum_speed_mps=self.minimum_speed_mps
                ):
                    return None
                self.receiver.begin_skill(observation)
                self.start_frame = observation.frame
            if observation.frame - self.start_frame == 100:
                self.completed = True
                return None
            return self.receiver.propose(observation)
        except (ValueError, TypeError, FloatingPointError):
            self.faulted = True
            raise
