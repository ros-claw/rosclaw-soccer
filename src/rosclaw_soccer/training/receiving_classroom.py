"""Explicit R0 receiving classroom; no imported script patches or match defaults."""

from dataclasses import asdict, dataclass, fields, replace
from typing import Any

from rosclaw_soccer.growth.independent_agent_cell import (
    AgentCellDecision,
    AgentCellObservation,
    RosclawSoccerAgentCell,
)
from rosclaw_soccer.growth.locomotion_contact_teacher import G1LocomotionContactTeacherConfig
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.skills.team.independent_team_world import IndependentTeamWorldConfig
from rosclaw_soccer.training.continuous_match_residual_ppo import collection_world
from rosclaw_soccer.training.role_receiving_courses import ROSTER, receiving_practice_task

R0_WORLD_HASH = "sha256:7d213ea5bb1f94720c962f3a2b673ab7b71be636f426372846bd20efe396a1f9"
R0_TEACHER_HASH = "sha256:4f29c10c29a2076e1dec5bdf5025715857edf53ecde1f3552f75c61fda881d67"


def r0_receiving_configuration() -> tuple[
    IndependentTeamWorldConfig, G1LocomotionContactTeacherConfig
]:
    """Reconstruct and check the historical protocol before running physics.

    Drift is a new protocol, not permission to silently redefine R0. These
    hashes bind configuration, not checkpoint weights or full experiment code.
    """
    world = replace(
        collection_world(0.24, True),
        simulation_duration_sec=6.0,
        loose_ball_capture_hold=True,
        loose_ball_capture_control=True,
        loose_ball_capture_hold_sec=0.6,
        loose_ball_capture_live_foundation=True,
        rotation_equivariant_receive_heading=True,
        rotation_equivariant_duel_side=True,
        stop_on_ball_exit=True,
        receive_pacing_ratio=0.20,
        loose_ball_capture_follow_navigation=True,
    )
    teacher = replace(
        G1LocomotionContactTeacherConfig(),
        committed_receive_ankle_lateral_offset_m=0.12,
        one_touch_finish_enabled=False,
    )
    if world.config_hash != R0_WORLD_HASH or hash_json(asdict(teacher)) != R0_TEACHER_HASH:
        raise ValueError("R0 classroom configuration drift; define a new protocol explicitly")
    return world, teacher


@dataclass(frozen=True)
class ReceivingPracticeCell(RosclawSoccerAgentCell):
    """Instance-scoped assignment, with a different cell identity from a match."""

    focal_agent_id: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.focal_agent_id != self.agent_id or self.focal_agent_id not in ROSTER:
            raise ValueError("receiving coach must bind exactly its own roster agent")

    def decide(self, observation: AgentCellObservation) -> AgentCellDecision:
        return receiving_practice_task(
            self, observation, super().decide(observation), focal_agent_id=self.focal_agent_id
        )

    def to_dict(self) -> dict[str, Any]:
        return {**super().to_dict(), "receiving_coach": self.focal_agent_id}


def coached_receiving_cells(
    cells: tuple[RosclawSoccerAgentCell, ...], *, focal_agent_id: str
) -> tuple[RosclawSoccerAgentCell, ...]:
    if tuple(sorted(c.agent_id for c in cells)) != ROSTER or focal_agent_id not in ROSTER:
        raise ValueError("complete frozen receiving roster required")
    if any(type(c) is not RosclawSoccerAgentCell for c in cells):
        raise ValueError("do not silently wrap or replace an existing classroom cell")
    return tuple(
        ReceivingPracticeCell(
            **{f.name: getattr(c, f.name) for f in fields(RosclawSoccerAgentCell)},
            focal_agent_id=focal_agent_id,
        )
        if c.agent_id == focal_agent_id
        else c
        for c in cells
    )
