"""Read-only, substep-resolved ball-contact normal impulse evidence.

This deliberately excludes tangential force and ground. It is not a complete
ball impulse, successful pass, motion command or hardware authorization.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.skills.team.motor_option import TeamMotorPhysicsObservation

LOADED_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass(frozen=True)
class ContactImpulseFrame:
    time_sec: float
    # World-frame force on the ball, integrated over 10 observed 2 ms steps.
    foot_normal_impulse_ns: tuple[tuple[str, float, float, float], ...]
    foot_contact_substeps: tuple[tuple[str, int], ...]
    # True when any non-foot/another agent contacts the ball in this frame.
    ambiguous_agents: tuple[str, ...]
    world_bodies_safe: bool


class ContactNormalImpulseRecorder:
    """Attribute normal force to each robot foot, never alter physics state."""

    def __init__(self, agent_id: str, roster: tuple[str, ...], *, maximum_frames: int = 1000):
        if (
            type(agent_id) is not str
            or type(roster) is not tuple
            or not roster
            or tuple(sorted(set(roster))) != roster
            or agent_id not in roster
            or type(maximum_frames) is not int
            or not 1 <= maximum_frames <= 1000
        ):
            raise ValueError("bounded canonical roster and observer required")
        self.agent_id = agent_id
        self.roster = roster
        self.maximum_frames = maximum_frames
        self.contract_hash = hash_json(
            {
                "schema": "soccer.contact-normal-impulse.v2",
                "agent": agent_id,
                "roster": roster,
                "maximum_frames": maximum_frames,
                "dt": 0.002,
                "steps_per_frame": 10,
                "sim_only": True,
            }
        )
        self._steps = 0
        self._rows: list[ContactImpulseFrame] = []
        self._impulses = {a: [0.0, 0.0, 0.0] for a in roster}
        self._contacts = {a: 0 for a in roster}
        self._counterparts: set[tuple[str | None, bool]] = set()
        self._safe = True

    def observe_physics(self, observation: TeamMotorPhysicsObservation) -> None:
        if (
            not isinstance(observation, TeamMotorPhysicsObservation)
            or not observation.contacts_complete
            or observation.observer_agent_id != self.agent_id
            or self._steps >= self.maximum_frames * 10
            or abs(observation.time_sec - 0.002 * (self._steps + 1)) > 1e-7
        ):
            raise ValueError("complete contiguous SIM-only substep observations required")
        observation.__post_init__()
        contacts = tuple(c for c in observation.ball_contacts if c.normal_force_n > 0.0)
        if any(c.agent_id is not None and c.agent_id not in self.roster for c in contacts):
            raise ValueError("unknown contact owner")
        if any(c.normal_ball_to_counterpart_world is None for c in contacts):
            raise ValueError("contact normal required")
        owners = {c.agent_id for c in contacts if c.is_foot}
        for owner in owners:
            assert owner is not None
            self._contacts[owner] += 1
        self._counterparts.update((c.agent_id, c.is_foot) for c in contacts)
        for contact in contacts:
            if not contact.is_foot:
                continue
            owner = contact.agent_id
            normal = contact.normal_ball_to_counterpart_world
            assert owner is not None and normal is not None
            for i in range(3):
                # Normal points from ball toward counterpart, so the ball's
                # normal force points in the opposite direction.
                self._impulses[owner][i] -= contact.normal_force_n * normal[i] * 0.002
        self._steps += 1
        self._safe = self._safe and observation.world_bodies_safe
        if self._steps % 10 == 0:
            vectors = tuple(
                (a, self._impulses[a][0], self._impulses[a][1], self._impulses[a][2])
                for a in self.roster
            )
            if any(not math.isfinite(v) for row in vectors for v in row[1:]):
                raise ValueError("nonfinite integrated impulse")
            self._rows.append(
                ContactImpulseFrame(
                    observation.time_sec,
                    vectors,
                    tuple((a, self._contacts[a]) for a in self.roster),
                    tuple(
                        agent
                        for agent in self.roster
                        if self._contacts[agent] > 0
                        and any(owner != agent or not foot for owner, foot in self._counterparts)
                    ),
                    self._safe,
                )
            )
            self._impulses = {a: [0.0, 0.0, 0.0] for a in self.roster}
            self._contacts = {a: 0 for a in self.roster}
            self._counterparts.clear()
            self._safe = True

    def verified_frames(self, *, control_frames: int) -> tuple[ContactImpulseFrame, ...]:
        if (
            type(control_frames) is not int
            or not 1 <= control_frames <= self.maximum_frames
            or self._steps != control_frames * 10
            or len(self._rows) != control_frames
        ):
            raise ValueError("normal impulse evidence does not cover the entire control trace")
        return tuple(self._rows)
