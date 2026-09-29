"""Read-only 500 Hz dynamic receiver-contact window for SIM_ONLY learning evidence."""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.skills.team.motor_option import TeamMotorPhysicsObservation


class B6MicrophysicsObserver:
    """Observe one real moving-ball foot collision without motor authority."""

    needs_contact_velocity = True

    def __init__(self, agent_id: str, *, pre_steps: int = 75, post_steps: int = 250) -> None:
        if (
            agent_id != "red.finisher"
            or type(pre_steps) is not int
            or type(post_steps) is not int
            or not 25 <= pre_steps <= 100
            or not 100 <= post_steps <= 300
        ):
            raise ValueError("bounded receiver microphysics observer required")
        self.agent_id = agent_id
        self.pre_steps = pre_steps
        self.post_steps = post_steps
        self.contract_hash = hash_json(
            {
                "schema": "rosclaw_soccer.rsi.b6_microphysics_observer.v1",
                "agent_id": agent_id,
                "pre_steps": pre_steps,
                "post_steps": post_steps,
                "activation_ceiling": "SIM_ONLY",
                "motor_authority": False,
                "contact_relative_velocity_requested": True,
            }
        )
        self._ring: deque[TeamMotorPhysicsObservation] = deque(maxlen=pre_steps)
        self._window: list[TeamMotorPhysicsObservation] = []
        self._remaining = 0
        self.first_foot_time_sec: float | None = None
        self.incoming_ball_speed_mps: float | None = None

    def observe_physics(self, observation: TeamMotorPhysicsObservation) -> None:
        if (
            not isinstance(observation, TeamMotorPhysicsObservation)
            or observation.observer_agent_id != self.agent_id
            or not observation.contacts_complete
        ):
            raise ValueError("complete same-player read-only microphysics required")
        if self.first_foot_time_sec is not None:
            if self._remaining > 0:
                self._window.append(observation)
                self._remaining -= 1
            return
        own_foot = any(
            contact.agent_id == self.agent_id and contact.is_foot
            for contact in observation.ball_contacts
        )
        previous_speed = (
            math.hypot(self._ring[-1].qvel[35], self._ring[-1].qvel[36]) if self._ring else 0.0
        )
        if own_foot and previous_speed >= 0.30:
            self.first_foot_time_sec = observation.time_sec
            self.incoming_ball_speed_mps = previous_speed
            self._window = [*self._ring, observation]
            self._remaining = self.post_steps
        else:
            self._ring.append(observation)

    @property
    def complete(self) -> bool:
        return (
            self.first_foot_time_sec is not None
            and self._remaining == 0
            and len(self._window) == self.pre_steps + 1 + self.post_steps
        )

    def arrays(self) -> dict[str, NDArray[Any]]:
        if not self.complete:
            raise ValueError("complete measured dynamic receiver window required")
        rows = self._window
        time = np.asarray([row.time_sec for row in rows], dtype=np.float64)
        if not np.allclose(np.diff(time), 0.002, atol=1.0e-6, rtol=0.0):
            raise ValueError("consecutive 500 Hz microphysics window required")
        foot_force = []
        contact_normal = []
        counterpart_minus_ball_velocity = []
        complete_geometry = []
        for row in rows:
            own = [
                contact
                for contact in row.ball_contacts
                if contact.agent_id == self.agent_id and contact.is_foot
            ]
            contact = max(own, key=lambda value: value.normal_force_n) if own else None
            foot_force.append(0.0 if contact is None else contact.normal_force_n)
            has_geometry = bool(
                contact is not None
                and contact.normal_ball_to_counterpart_world is not None
                and contact.counterpart_minus_ball_velocity_world_mps is not None
            )
            complete_geometry.append(has_geometry)
            if (
                contact is None
                or contact.normal_ball_to_counterpart_world is None
                or contact.counterpart_minus_ball_velocity_world_mps is None
            ):
                contact_normal.append((0.0, 0.0, 0.0))
                counterpart_minus_ball_velocity.append((0.0, 0.0, 0.0))
            else:
                contact_normal.append(contact.normal_ball_to_counterpart_world)
                counterpart_minus_ball_velocity.append(
                    contact.counterpart_minus_ball_velocity_world_mps
                )
        return {
            "time_sec": time,
            "qpos": np.asarray([row.qpos for row in rows], dtype=np.float64),
            "qvel": np.asarray([row.qvel for row in rows], dtype=np.float64),
            "world_bodies_safe": np.asarray(
                [row.world_bodies_safe for row in rows], dtype=np.bool_
            ),
            "own_foot_normal_force_n": np.asarray(foot_force, dtype=np.float64),
            "other_non_ground_normal_force_n": np.asarray(
                [row.other_non_ground_normal_force_n for row in rows], dtype=np.float64
            ),
            "contact_geometry_complete": np.asarray(complete_geometry, dtype=np.bool_),
            "ball_to_foot_normal_world": np.asarray(contact_normal, dtype=np.float64),
            "counterpart_minus_ball_velocity_world_mps": np.asarray(
                counterpart_minus_ball_velocity, dtype=np.float64
            ),
        }


__all__ = ["B6MicrophysicsObserver"]
