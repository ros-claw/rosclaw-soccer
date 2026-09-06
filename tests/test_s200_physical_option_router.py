from __future__ import annotations

from dataclasses import replace

import pytest

from rosclaw_soccer.growth.independent_agent_cell import (
    AgentCellObservation,
    AgentPhysicalState,
    RosclawSoccerAgentCell,
    build_independent_agent_cell,
    build_team_coordination_frame,
)
from rosclaw_soccer.growth.physical_option_router import (
    PhysicalOptionOutcome,
    PhysicalOptionTerminal,
    PhysicalSoccerOption,
    build_physical_option_request,
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


def _frame(*, possession: str = "red.playmaker"):
    cells = _cells()
    states = {
        agent_id: AgentPhysicalState(
            agent_id=agent_id,
            position_m=(home[0], home[1], 0.78),
            velocity_mps=(0.0, 0.0, 0.0),
            pelvis_height_m=0.78,
            tilt_rad=0.02,
            stable=True,
        )
        for agent_id, _, _, home in _LAYOUT
    }
    observations = tuple(
        AgentCellObservation(
            observer_agent_id=cell.agent_id,
            time_sec=1.2,
            ball_position_m=(2.15, -0.35, 0.115),
            ball_velocity_mps=(0.1, 0.0, 0.0),
            own_goal_m=(-1.5, 0.0, 0.0) if cell.self_model.team_id == "red" else (7.5, 0.0, 0.0),
            opponent_goal_m=(7.5, 0.0, 0.0)
            if cell.self_model.team_id == "red"
            else (-1.5, 0.0, 0.0),
            possession_agent_id=possession,
            self_state=states[cell.agent_id],
            teammate_states=tuple(states[value] for value in cell.self_model.teammate_ids),
            opponent_states=tuple(states[value] for value in cell.self_model.opponent_ids),
        )
        for cell in cells
    )
    decisions = tuple(
        cell.decide(
            next(value for value in observations if value.observer_agent_id == cell.agent_id)
        )
        for cell in cells
    )
    coordination = build_team_coordination_frame(
        roster=TeamRoleRoster("s200.unit.match", tuple(cell.self_model for cell in cells)),
        cells=cells,
        observations=observations,
        decisions=decisions,
        frame_index=20,
    )
    return cells, decisions, coordination


def test_pass_request_binds_cell_policy_decision_frame_ball_and_receiver() -> None:
    cells, decisions, frame = _frame()
    cell = next(value for value in cells if value.agent_id == "red.playmaker")
    decision = next(value for value in decisions if value.agent_id == cell.agent_id)

    request = build_physical_option_request(
        cell=cell,
        decision=decision,
        coordination=frame,
        option_policy_hash=_hash("contact-policy"),
        timeout_sec=7.0,
    )

    assert request.option is PhysicalSoccerOption.PASS
    assert request.tactical_intent is TacticalIntent.PASS
    assert request.target_agent_id == "red.finisher"
    assert request.pass_handshake_hash == frame.pass_receive_handshakes[0].handshake_hash
    assert request.coordination_frame_hash == frame.frame_hash
    assert request.ball_state_hash in {
        observation.ball_state_hash for observation in frame.observations
    }
    assert frame.frame_hash.removeprefix("sha256:")[:12] in request.request_id
    assert not request.hardware_authorized
    assert not request.direct_joint_torque_output


def test_router_rejects_stale_or_forged_commitment() -> None:
    cells, decisions, frame = _frame()
    cell = next(value for value in cells if value.agent_id == "red.playmaker")
    decision = next(value for value in decisions if value.agent_id == cell.agent_id)

    with pytest.raises(ValueError, match="current agent commitment"):
        build_physical_option_request(
            cell=cell,
            decision=replace(decision, confidence=0.51),
            coordination=frame,
            option_policy_hash=_hash("contact-policy"),
            timeout_sec=7.0,
        )
    with pytest.raises(ValueError, match="timing handshake"):
        build_physical_option_request(
            cell=cell,
            decision=decision,
            coordination=replace(frame, pass_receive_handshakes=()),
            option_policy_hash=_hash("contact-policy"),
            timeout_sec=7.0,
        )


def test_non_contact_tactical_intent_has_no_physical_option_authority() -> None:
    cells, decisions, frame = _frame(possession="blue.playmaker")
    cell = next(value for value in cells if value.agent_id == "red.playmaker")
    decision = next(value for value in decisions if value.agent_id == cell.agent_id)
    assert decision.intent is TacticalIntent.RECEIVE

    with pytest.raises(ValueError, match="no physical option route"):
        build_physical_option_request(
            cell=cell,
            decision=decision,
            coordination=frame,
            option_policy_hash=_hash("contact-policy"),
            timeout_sec=7.0,
        )


def test_measured_outcome_fails_closed_on_safety_or_missing_contact() -> None:
    outcome = PhysicalOptionOutcome(
        request_hash=_hash("request"),
        terminal=PhysicalOptionTerminal.COMPLETE,
        contact_observed=True,
        contact_time_sec=5.2,
        contact_link="right_foot",
        pre_contact_ball_speed_mps=0.1,
        post_contact_peak_ball_speed_mps=2.0,
        target_delivery_distance_m=0.25,
        target_agent_contact_observed=False,
        goal_crossed=False,
        goalkeeper_contact_observed=False,
        finite_state=True,
        minimum_pelvis_height_m=0.68,
        maximum_tilt_rad=0.32,
        required_minimum_pelvis_height_m=0.55,
        allowed_maximum_tilt_rad=0.80,
        joint_limit_violation=False,
        torque_limit_violation=False,
        robot_robot_contact_count=0,
        option_started_from_current_commitment=True,
        recovery_handoff_completed=True,
    )

    assert outcome.safe
    assert outcome.physical_contact_success
    assert not replace(outcome, ball_state_write_after_start=True).physical_contact_success
    assert not replace(outcome, recovery_handoff_completed=False).physical_contact_success
    assert not replace(outcome, minimum_pelvis_height_m=0.40).safe
    assert not replace(
        outcome,
        contact_observed=False,
        contact_time_sec=None,
        contact_link=None,
    ).physical_contact_success


def test_outcome_thresholds_are_explicit_and_content_bound() -> None:
    with pytest.raises(ValueError, match="invalid"):
        PhysicalOptionOutcome(
            request_hash=_hash("request"),
            terminal=PhysicalOptionTerminal.COMPLETE,
            contact_observed=False,
            contact_time_sec=None,
            contact_link=None,
            pre_contact_ball_speed_mps=0.0,
            post_contact_peak_ball_speed_mps=0.0,
            target_delivery_distance_m=None,
            target_agent_contact_observed=False,
            goal_crossed=False,
            goalkeeper_contact_observed=False,
            finite_state=True,
            minimum_pelvis_height_m=0.70,
            maximum_tilt_rad=0.20,
            required_minimum_pelvis_height_m=0.0,
            allowed_maximum_tilt_rad=0.80,
            joint_limit_violation=False,
            torque_limit_violation=False,
            robot_robot_contact_count=0,
            option_started_from_current_commitment=True,
            recovery_handoff_completed=True,
        )
