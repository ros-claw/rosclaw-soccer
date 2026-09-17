"""Opt-in ball-relative role proposals; not learned motor skills or possession."""

from __future__ import annotations

import math

import numpy as np

from rosclaw_soccer.growth.independent_agent_cell import (
    AgentCellDecision,
    AgentCellObservation,
    RosclawSoccerAgentCell,
)
from rosclaw_soccer.growth.role_self_model import MatchRole, SoccerSkill, TacticalIntent


def ball_role_task(
    cell: RosclawSoccerAgentCell,
    observation: AgentCellObservation,
    proposed: AgentCellDecision,
) -> tuple[AgentCellDecision, str]:
    """Keep one team challenger, offer distinct outlets and protect own goal.

    Uses only the cell's current observation. Recovery, keeper decisions,
    accepted receptions and existing ball actions keep priority. This explicit
    experimental caller opt-in grants no motor permission and changes no state.
    Target clearance is a position heuristic, not collision certification.
    """
    if (
        cell.agent_id != observation.observer_agent_id
        or proposed.agent_id != cell.agent_id
        or proposed.observation_hash != observation.observation_hash
        or proposed.policy_artifact_hash != cell.self_model.policy_artifact_hash
        or not cell.self_model.authorizes(proposed.intent, proposed.skill)
        or set(cell.self_model.teammate_ids) != {s.agent_id for s in observation.teammate_states}
        or set(cell.self_model.opponent_ids) != {s.agent_id for s in observation.opponent_states}
    ):
        raise ValueError("role task must bind this cell's observation, roster and proposal")
    role = cell.self_model.primary_role
    if not observation.self_state.stable or proposed.intent is TacticalIntent.RECOVER:
        return proposed, "recover_before_ball_task"
    if role is MatchRole.GOALKEEPER:
        return proposed, "protect_goal_or_distribute"
    if observation.active_receive_source_agent_id is not None:
        return proposed, "honor_live_reception"
    if proposed.intent in {TacticalIntent.PASS, TacticalIntent.SHOOT, TacticalIntent.CARRY}:
        return proposed, "execute_existing_ball_task"
    if observation.possession_agent_id == cell.agent_id:
        return proposed, "retain_measured_owner_task"
    teammates = (observation.self_state, *observation.teammate_states)
    candidates = tuple(
        s
        for s in teammates
        if s.stable
        and cell._role_for_agent(s.agent_id) is not MatchRole.GOALKEEPER
        and (
            cell.self_model.basic_ball_play
            or cell._role_for_agent(s.agent_id) is not MatchRole.DEFENDER
        )
    )
    if not candidates:
        return proposed, "no_stable_outfielder"
    ball = np.asarray(observation.ball_position_m[:2], dtype=float)
    selected = next(
        (s for s in candidates if s.agent_id == observation.ball_chaser_agent_id),
        min(candidates, key=lambda s: (math.dist(s.position_m[:2], ball), s.agent_id)),
    )
    team_owns = observation.possession_agent_id in cell.self_model.teammate_ids
    if not team_owns and selected.agent_id == cell.agent_id:
        if not cell.self_model.authorizes(TacticalIntent.RECEIVE, SoccerSkill.FIRST_TOUCH):
            return proposed, "ball_challenge_not_authorized"
        return cell._decision(
            observation,
            TacticalIntent.RECEIVE,
            SoccerSkill.FIRST_TOUCH,
            observation.ball_position_m,
            observation.possession_agent_id,
            0.90,
        ), "challenge_observed_ball"
    if role is MatchRole.DEFENDER:
        intent, skill, reason = TacticalIntent.COVER, SoccerSkill.BLOCKING, "protect_goal_side"
        base = 0.55 * ball + 0.45 * np.asarray(observation.own_goal_m[:2])
        offset = 0.75
    else:
        intent, skill = TacticalIntent.SUPPORT, SoccerSkill.OFF_BALL_RUN
        reason = "offer_forward_outlet" if role is MatchRole.FINISHER else "offer_return_outlet"
        sign = math.copysign(1.0, observation.opponent_goal_m[0] - observation.own_goal_m[0])
        depth = 1.10 if role is MatchRole.FINISHER else -0.85
        base = ball + np.array((sign * depth, 0.0))
        offset = 1.35
    if not cell.self_model.authorizes(intent, skill):
        return proposed, "support_task_not_authorized"
    lo, hi = sorted((observation.own_goal_m[0], observation.opponent_goal_m[0]))
    positions = [
        s.position_m[:2] for s in (*observation.teammate_states, *observation.opponent_states)
    ]
    side = math.copysign(1.0, cell.tactical_profile.home_position_m[1] or 1.0)
    targets = [
        (
            float(np.clip(base[0], lo + 0.25, hi - 0.25)),
            float(np.clip(base[1] + lateral * offset, -2.8, 2.8)),
            0.0,
        )
        for lateral in (side, -side)
    ]
    target = max(
        targets, key=lambda p: min((math.dist(p[:2], q) for q in positions), default=math.inf)
    )
    if min((math.dist(target[:2], q) for q in positions), default=math.inf) < 0.9:
        return proposed, "no_clear_support_pocket"
    return cell._decision(
        observation,
        intent,
        skill,
        target,
        observation.possession_agent_id if team_owns else selected.agent_id,
        0.88,
    ), reason
