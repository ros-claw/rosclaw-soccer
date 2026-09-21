"""Balanced single-ball receiving classrooms, not autonomous match evidence."""

import math
from dataclasses import dataclass

from rosclaw_soccer.growth.independent_agent_cell import (
    AgentCellDecision,
    AgentCellObservation,
    RosclawSoccerAgentCell,
)
from rosclaw_soccer.growth.role_self_model import SoccerSkill, TacticalIntent

ROSTER = tuple(
    f"{team}.{role}"
    for team in ("blue", "red")
    for role in ("defender", "finisher", "goalkeeper", "playmaker")
)


@dataclass(frozen=True)
class ReceivingCourse:
    agent_id: str
    seed: int
    speed_mps: float
    lateral_m: float


def receiving_courses(*, repetitions: int, first_seed: int) -> tuple[ReceivingCourse, ...]:
    """Equal opportunities per player/speed with unique run identifiers.

    Distinct seeds alone do not establish distinct physical situations or
    statistically independent evidence; audit measured states separately.
    """
    if (
        type(repetitions) is not int
        or not 1 <= repetitions <= 128
        or type(first_seed) is not int
        or not 0 <= first_seed < 2**32 - repetitions * 16
    ):
        raise ValueError("bounded repetitions and unique uint32 seeds required")
    return tuple(
        ReceivingCourse(agent, first_seed + i, speed, (-0.08, 0.08)[r % 2])
        for i, (r, speed, agent) in enumerate(
            (r, speed, agent)
            for r in range(repetitions)
            for speed in (0.75, 1.25)
            for agent in ROSTER
        )
    )


def receiving_ball_launch(
    course: ReceivingCourse, *, origin: tuple[float, float, float], radius_m: float
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """One explicit initial reset, 0.9 m in front; no in-flight correction."""
    if (
        course.agent_id not in ROSTER
        or not math.isfinite(course.speed_mps)
        or not 0.5 <= course.speed_mps <= 3.0
        or not math.isfinite(course.lateral_m)
        or abs(course.lateral_m) > 0.3
        or len(origin) != 3
        or not all(math.isfinite(x) for x in origin)
        or not math.isfinite(radius_m)
        or not 0.03 <= radius_m <= 0.15
    ):
        raise ValueError("finite bounded physical receiving launch required")
    sign = 1.0 if course.agent_id.startswith("red.") else -1.0
    return (
        (origin[0] + sign * 0.9, origin[1] + sign * course.lateral_m, radius_m),
        (-sign * course.speed_mps, 0.0, 0.0),
    )


def receiving_practice_task(
    cell: RosclawSoccerAgentCell,
    observation: AgentCellObservation,
    proposed: AgentCellDecision,
    *,
    focal_agent_id: str,
) -> AgentCellDecision:
    """Coach assigns FIRST_TOUCH, without granting motor authority or possession.

    All other players retain their ordinary decisions. Recovery wins. This must
    be explicitly enabled by a classroom caller, never a match default.
    """
    if (
        focal_agent_id not in ROSTER
        or observation.observer_agent_id != cell.agent_id
        or proposed.agent_id != cell.agent_id
        or proposed.observation_hash != observation.observation_hash
        or proposed.policy_artifact_hash != cell.self_model.policy_artifact_hash
    ):
        raise ValueError("receiving assignment must bind this cell and observation")
    if cell.agent_id != focal_agent_id:
        return proposed
    if not observation.self_state.stable or proposed.intent is TacticalIntent.RECOVER:
        return proposed
    if not cell.self_model.authorizes(TacticalIntent.RECEIVE, SoccerSkill.FIRST_TOUCH):
        raise ValueError("role lacks receiving curriculum capability")
    return cell._decision(
        observation,
        TacticalIntent.RECEIVE,
        SoccerSkill.FIRST_TOUCH,
        observation.ball_position_m,
        None,
        0.9,
    )
