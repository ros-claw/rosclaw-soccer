"""Withdraw declined pass proposals without inventing a receiver's consent."""

from rosclaw_soccer.growth.independent_agent_cell import (
    AgentCellDecision,
    AgentCellObservation,
    RosclawSoccerAgentCell,
)
from rosclaw_soccer.growth.role_self_model import TacticalIntent


def reconcile_pass_readiness(
    *,
    cells: tuple[RosclawSoccerAgentCell, ...],
    observations: tuple[AgentCellObservation, ...],
    decisions: tuple[AgentCellDecision, ...],
) -> tuple[AgentCellDecision, ...]:
    """Keep accepted proposals identical; turn a legitimate declined PASS into HOLD.

    Inspect the original simultaneous proposals, not earlier replacements, so
    ordering cannot manufacture consent. Invalid identities/bindings still raise.
    HOLD is a bounded tactical proposal, not a physical stop or new motor permit.
    The coordination builder must still validate the resulting complete frame.
    """
    by_cell = {c.agent_id: c for c in cells}
    by_obs = {o.observer_agent_id: o for o in observations}
    by_decision = {d.agent_id: d for d in decisions}
    expected = set(by_cell)
    if (
        not expected
        or not len(cells) == len(observations) == len(decisions) == len(expected)
        or set(by_obs) != expected
        or set(by_decision) != expected
    ):
        raise ValueError("one bound proposal per independent player required")
    for agent, decision in by_decision.items():
        cell, obs = by_cell[agent], by_obs[agent]
        if (
            decision.observation_hash != obs.observation_hash
            or decision.policy_artifact_hash != cell.self_model.policy_artifact_hash
            or not cell.self_model.authorizes(decision.intent, decision.skill)
        ):
            raise ValueError("pass readiness cannot repair invalid proposal authority")
    result = []
    for decision in decisions:
        cell = by_cell[decision.agent_id]
        if decision.intent is not TacticalIntent.PASS:
            result.append(decision)
            continue
        receiver = decision.target_agent_id
        if receiver not in expected or receiver not in cell.self_model.teammate_ids:
            raise ValueError("pass proposal requires an actual teammate receiver")
        if (
            by_decision[receiver].intent
            in {
                TacticalIntent.RECEIVE,
                TacticalIntent.SUPPORT,
                TacticalIntent.RUN_IN_BEHIND,
            }
            and by_obs[receiver].self_state.stable
        ):
            result.append(decision)
            continue
        if not cell.self_model.authorizes(TacticalIntent.HOLD, decision.skill):
            raise ValueError("declined pass has no authorized hold fallback")
        observation = by_obs[decision.agent_id]
        result.append(
            cell._decision(
                observation,
                TacticalIntent.HOLD,
                decision.skill,
                observation.self_state.position_m,
                None,
                0.0,
            )
        )
    return tuple(result)
