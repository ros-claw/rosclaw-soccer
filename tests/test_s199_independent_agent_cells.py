from __future__ import annotations

from dataclasses import replace

import pytest
from rosclaw.continual.plasticity_lease import audit_plasticity_lease

from rosclaw_soccer.growth.independent_agent_cell import (
    AgentCellObservation,
    AgentPhysicalState,
    RosclawSoccerAgentCell,
    build_agent_plasticity_lease,
    build_independent_agent_cell,
    build_team_coordination_frame,
)
from rosclaw_soccer.growth.role_self_model import MatchRole, TacticalIntent, TeamRoleRoster
from rosclaw_soccer.sim.contracts import hash_json

_LAYOUT = (
    ("red.goalkeeper", "red", MatchRole.GOALKEEPER, (0.2, 0.0, 0.0)),
    ("red.playmaker", "red", MatchRole.PLAYMAKER, (2.0, -0.4, 0.0)),
    ("red.finisher", "red", MatchRole.FINISHER, (3.4, 0.8, 0.0)),
    ("blue.goalkeeper", "blue", MatchRole.GOALKEEPER, (7.1, 0.0, 0.0)),
    ("blue.playmaker", "blue", MatchRole.PLAYMAKER, (5.2, -0.8, 0.0)),
    ("blue.finisher", "blue", MatchRole.FINISHER, (4.2, 0.8, 0.0)),
)


def _hash(label: str) -> str:
    return str(hash_json({"fixture": label}))


def _cells() -> tuple[RosclawSoccerAgentCell, ...]:
    team_ids = {
        team: tuple(agent_id for agent_id, value, _, _ in _LAYOUT if value == team)
        for team in ("red", "blue")
    }
    return tuple(
        build_independent_agent_cell(
            agent_id=agent_id,
            team_id=team,
            primary_role=role,
            teammate_ids=tuple(value for value in team_ids[team] if value != agent_id),
            opponent_ids=team_ids["blue" if team == "red" else "red"],
            body_hash=_hash("body"),
            foundation_policy_hash=_hash("foundation"),
            home_position_m=home,
        )
        for agent_id, team, role, home in _LAYOUT
    )


def _states() -> dict[str, AgentPhysicalState]:
    return {
        agent_id: AgentPhysicalState(
            agent_id=agent_id,
            position_m=(home[0], home[1], 0.78),
            velocity_mps=(0.0, 0.0, 0.0),
            pelvis_height_m=0.78,
            tilt_rad=0.04,
            stable=True,
        )
        for agent_id, _, _, home in _LAYOUT
    }


def _observations(
    cells: tuple[RosclawSoccerAgentCell, ...],
    *,
    possession_agent_id: str = "red.playmaker",
) -> tuple[AgentCellObservation, ...]:
    states = _states()
    return tuple(
        AgentCellObservation(
            observer_agent_id=cell.agent_id,
            time_sec=1.2,
            ball_position_m=(2.15, -0.35, 0.115),
            ball_velocity_mps=(0.1, 0.0, 0.0),
            own_goal_m=(-1.5, 0.0, 0.0) if cell.self_model.team_id == "red" else (7.5, 0.0, 0.0),
            opponent_goal_m=(7.5, 0.0, 0.0)
            if cell.self_model.team_id == "red"
            else (-1.5, 0.0, 0.0),
            possession_agent_id=possession_agent_id,
            self_state=states[cell.agent_id],
            teammate_states=tuple(states[value] for value in cell.self_model.teammate_ids),
            opponent_states=tuple(states[value] for value in cell.self_model.opponent_ids),
        )
        for cell in cells
    )


def test_each_g1_has_an_independent_rosclaw_growth_cell() -> None:
    cells = _cells()

    assert len({cell.cell_hash for cell in cells}) == 6
    assert len({cell.growth_scope.scope_hash for cell in cells}) == 6
    assert len({cell.growth_scope.personal_memory_namespace for cell in cells}) == 6
    assert len({cell.growth_scope.failure_memory_namespace for cell in cells}) == 6
    assert len({cell.self_model.policy_artifact_hash for cell in cells}) == 6
    assert len({cell.growth_scope.foundation_policy_hash for cell in cells}) == 1
    assert all(cell.activation_ceiling == "SIM_ONLY" for cell in cells)
    assert all(not cell.direct_joint_torque_output for cell in cells)


def test_cells_decide_separately_then_form_a_pass_handshake() -> None:
    cells = _cells()
    observations = _observations(cells)
    decisions = tuple(
        next(cell for cell in cells if cell.agent_id == value.observer_agent_id).decide(value)
        for value in observations
    )
    decision_by_id = {decision.agent_id: decision for decision in decisions}
    roster = TeamRoleRoster("s199.unit.match", tuple(cell.self_model for cell in cells))

    frame = build_team_coordination_frame(
        roster=roster,
        cells=cells,
        observations=observations,
        decisions=decisions,
        frame_index=12,
    )

    assert decision_by_id["red.playmaker"].intent is TacticalIntent.PASS
    assert decision_by_id["red.finisher"].intent is TacticalIntent.RUN_IN_BEHIND
    assert decision_by_id["red.goalkeeper"].intent is TacticalIntent.COVER
    assert decision_by_id["blue.goalkeeper"].intent is TacticalIntent.COVER
    assert len(frame.observations) == len(frame.intents) == 6
    assert len(frame.pass_receive_handshakes) == 1
    assert frame.pass_receive_handshakes[0].passer_agent_id == "red.playmaker"
    assert frame.pass_receive_handshakes[0].receiver_agent_id == "red.finisher"


def test_cell_rejects_another_players_observation() -> None:
    cells = _cells()
    observations = _observations(cells)

    with pytest.raises(ValueError, match="another player's observation"):
        cells[0].decide(observations[1])


def test_coordination_rejects_a_forged_agent_policy_identity() -> None:
    cells = _cells()
    observations = _observations(cells)
    decisions = tuple(
        next(cell for cell in cells if cell.agent_id == value.observer_agent_id).decide(value)
        for value in observations
    )
    forged = replace(decisions[0], policy_artifact_hash=_hash("forged-policy"))

    with pytest.raises(ValueError, match="observation and policy"):
        build_team_coordination_frame(
            roster=TeamRoleRoster("s199.unit.forged", tuple(cell.self_model for cell in cells)),
            cells=cells,
            observations=observations,
            decisions=(forged, *decisions[1:]),
            frame_index=13,
        )


def test_blue_playmaker_selects_a_forward_receiver_in_its_attack_direction() -> None:
    cells = _cells()
    observations = _observations(cells, possession_agent_id="blue.playmaker")
    decisions = {
        cell.agent_id: cell.decide(
            next(value for value in observations if value.observer_agent_id == cell.agent_id)
        )
        for cell in cells
    }

    assert decisions["blue.playmaker"].intent is TacticalIntent.PASS
    assert decisions["blue.playmaker"].target_agent_id == "blue.finisher"
    assert decisions["blue.finisher"].intent is TacticalIntent.RUN_IN_BEHIND


def test_blue_goalkeeper_distributes_toward_the_opponent_goal() -> None:
    cells = _cells()
    observations = _observations(cells, possession_agent_id="blue.goalkeeper")
    goalkeeper = next(cell for cell in cells if cell.agent_id == "blue.goalkeeper")
    observation = next(
        value for value in observations if value.observer_agent_id == goalkeeper.agent_id
    )

    decision = goalkeeper.decide(observation)

    assert decision.intent is TacticalIntent.DISTRIBUTE
    assert decision.target_agent_id == "blue.finisher"


def test_plasticity_lease_freezes_the_other_five_rosclaw_cells() -> None:
    cells = _cells()
    lease = build_agent_plasticity_lease(
        cells=cells,
        focal_agent_id="blue.goalkeeper",
        dataset_manifest_hash=_hash("failure-memory"),
        scenario_contract_hash=_hash("hard-save-curriculum"),
        maximum_optimizer_steps=2_000,
    )
    before = {cell.agent_id: cell.growth_scope.champion_policy.version_hash for cell in cells}
    accepted_after = {**before, "blue.goalkeeper": _hash("keeper-candidate")}
    rejected_after = {**accepted_after, "red.finisher": _hash("leaked-update")}

    accepted = audit_plasticity_lease(
        lease=lease,
        optimizer_steps=1_000,
        before_policy_hashes=before,
        after_policy_hashes=accepted_after,
    )
    rejected = audit_plasticity_lease(
        lease=lease,
        optimizer_steps=1_000,
        before_policy_hashes=before,
        after_policy_hashes=rejected_after,
    )

    assert lease.focal_agent_id == "blue.goalkeeper"
    assert accepted.passed
    assert accepted.changed_agent_ids == ("blue.goalkeeper",)
    assert not rejected.passed
    assert rejected.reasons == ("FROZEN_AGENT_CHANGED:red.finisher",)


def test_bootstrap_cannot_invent_a_parent_for_a_later_generation() -> None:
    with pytest.raises(ValueError, match="generation zero only"):
        build_independent_agent_cell(
            agent_id="red.playmaker",
            team_id="red",
            primary_role=MatchRole.PLAYMAKER,
            teammate_ids=("red.finisher",),
            opponent_ids=("blue.goalkeeper",),
            body_hash=_hash("body"),
            foundation_policy_hash=_hash("foundation"),
            home_position_m=(2.0, 0.0, 0.0),
            generation=1,
        )
