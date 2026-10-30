"""Bounded SIM-only consumer for physical football contact supervision."""

from dataclasses import asdict
from typing import Any

from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.skills.team.motor_option import TeamMotorPhysicsObservation


class DenseTeamContactRecorder:
    """Keep an uninterrupted observed stream, without issuing motor actions.

    Registration and timing are checked independently by PhysicsEvidenceSlot;
    the recorder checks them again so direct use cannot silently skip substeps.
    Rows are immutable observations of the original world, not a reset state.
    """

    def __init__(self, agent_id: str, *, maximum_substeps: int = 10000) -> None:
        if type(agent_id) is not str or not agent_id or type(maximum_substeps) is not int:
            raise ValueError("explicit agent and integral bounded capacity required")
        if not 1 <= maximum_substeps <= 10000:
            raise ValueError("maximum 20 simulated seconds of evidence required")
        self.agent_id = agent_id
        self.maximum_substeps = maximum_substeps
        self.contract_hash = hash_json(
            {
                "schema": "soccer.dense_team_contact_recorder.v1",
                "agent_id": agent_id,
                "maximum_substeps": maximum_substeps,
                "step_sec": 0.002,
                "sim_only": True,
                "hardware_authorized": False,
            }
        )
        self._rows: list[TeamMotorPhysicsObservation] = []

    def observe_physics(self, observation: TeamMotorPhysicsObservation) -> None:
        if not isinstance(observation, TeamMotorPhysicsObservation):
            raise ValueError("typed physical observation required")
        observation.__post_init__()
        if (
            not observation.contacts_complete
            or observation.observer_agent_id != self.agent_id
            or len(self._rows) >= self.maximum_substeps
            or abs(observation.time_sec - 0.002 * (len(self._rows) + 1)) > 1e-6
        ):
            raise ValueError("complete sequential same-agent SIM stream required")
        self._rows.append(observation)

    def observations(self) -> tuple[TeamMotorPhysicsObservation, ...]:
        return tuple(self._rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "soccer.dense_team_contact_evidence.v1",
            "contract_hash": self.contract_hash,
            "agent_id": self.agent_id,
            "steps": len(self._rows),
            "rows": [asdict(row) for row in self._rows],
            "sim_only": True,
            "hardware_authorized": False,
            "promotion_authorized": False,
        }
