"""Sparse, read-only SIM contact windows with adjacent actuator evidence.

Selection is conditioned on actual future contact: resulting windows are not
unbiased samples of a whole episode and never certify a football skill.
"""

import math
from collections import deque
from dataclasses import asdict
from typing import Any

from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.skills.team.motor_option import TeamMotorPhysicsObservation


class ContactEventActuationRecorder:
    """Keep each own-foot impact and bounded context without issuing actions."""

    needs_actuator_evidence = True
    needs_contact_velocity = True

    def __init__(self, agent_id: str, *, context_steps: int = 25, maximum_substeps: int = 5000):
        if (
            type(agent_id) is not str
            or not agent_id
            or type(context_steps) is not int
            or not 1 <= context_steps <= 100
            or type(maximum_substeps) is not int
            or not 1 <= maximum_substeps <= 10000
        ):
            raise ValueError("explicit bounded SIM contact-window configuration required")
        self.agent_id = agent_id
        self.context_steps = context_steps
        self.maximum_substeps = maximum_substeps
        self.contract_hash = hash_json(
            {
                "schema": "soccer.contact_event_actuation_recorder.v1",
                "agent_id": agent_id,
                "context_steps": context_steps,
                "maximum_substeps": maximum_substeps,
                "step_sec": 0.002,
                "minimum_own_foot_force_n_exclusive": 1.0,
                "actuator_evidence": True,
                "contact_velocity": True,
                "hardware_authorized": False,
            }
        )
        self._history: deque[tuple[int, TeamMotorPhysicsObservation]] = deque(
            maxlen=context_steps + 1
        )
        self._captured: list[tuple[int, TeamMotorPhysicsObservation]] = []
        self._step = 0
        self._through = 0
        self._events = 0

    def observe_physics(self, observation: TeamMotorPhysicsObservation) -> None:
        if not isinstance(observation, TeamMotorPhysicsObservation):
            raise ValueError("typed immutable SIM observation required")
        observation.__post_init__()
        step = self._step + 1
        if (
            not observation.contacts_complete
            or observation.observer_agent_id != self.agent_id
            or observation.actuator_evidence is None
            or step > self.maximum_substeps
            or abs(observation.time_sec - step * 0.002) > 1e-6
            or any(
                c.counterpart_minus_ball_velocity_world_mps is None
                for c in observation.ball_contacts
            )
        ):
            raise ValueError("complete ordered contact and actuator evidence required")
        self._step = step
        self._history.append((step, observation))
        if observation.foot_normal_force_n > 1.0:
            self._events += 1
            for candidate in self._history:
                if not self._captured or candidate[0] > self._captured[-1][0]:
                    self._captured.append(candidate)
            self._through = step + self.context_steps
        elif step <= self._through:
            if not self._captured or step > self._captured[-1][0]:
                self._captured.append((step, observation))

    def verified_observations(self, *, control_frame_count: int, terminal_time_sec: float) -> None:
        if (
            type(control_frame_count) is not int
            or not 1 <= control_frame_count <= self.maximum_substeps // 10
            or type(terminal_time_sec) not in (int, float)
            or not math.isfinite(terminal_time_sec)
            or abs(terminal_time_sec - control_frame_count * 0.02) > 1e-6
            or self._step != control_frame_count * 10
        ):
            raise ValueError("complete authenticated SIM observer horizon required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "soccer.contact_event_actuation_evidence.v1",
            "contract_hash": self.contract_hash,
            "agent_id": self.agent_id,
            "observed_steps": self._step,
            "event_substeps": self._events,
            "context_steps": self.context_steps,
            "rows": [{"step": step, "observation": asdict(obs)} for step, obs in self._captured],
            "selection": "own_foot_force_gt_1n_with_bounded_context",
            "sim_only": True,
            "hardware_authorized": False,
            "promotion_authorized": False,
        }
