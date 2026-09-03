from __future__ import annotations

from dataclasses import replace

import pytest

from rosclaw_soccer.growth.role_self_model import (
    EgocentricTeamObservation,
    MatchRole,
    PassReceiveHandshake,
    PossessionOwner,
    RoleIntentCommitment,
    RoleSelfModel,
    RoleSkillBinding,
    SoccerSkill,
    TacticalIntent,
    TeamCoordinationFrame,
    TeamRoleRoster,
)
from rosclaw_soccer.sim.contracts import hash_json


def _hash(label: str) -> str:
    return str(hash_json({"fixture": label}))


def _skill(skill: SoccerSkill) -> RoleSkillBinding:
    return RoleSkillBinding(skill, _hash(f"actor-{skill}"), _hash(f"evidence-{skill}"), 1, 0.6, 0.4)


def _model(
    agent_id: str,
    team_id: str,
    role: MatchRole,
    teammates: tuple[str, ...],
    opponents: tuple[str, ...],
    skills: tuple[SoccerSkill, ...],
) -> RoleSelfModel:
    return RoleSelfModel(
        agent_id=agent_id,
        team_id=team_id,
        primary_role=role,
        teammate_ids=teammates,
        opponent_ids=opponents,
        skills=tuple(_skill(skill) for skill in skills),
        observation_contract_hash=_hash("observation-v1"),
        action_contract_hash=_hash("action-v1"),
        policy_artifact_hash=_hash(f"policy-{agent_id}"),
        failure_memory_namespace=f"memory.{agent_id}",
        generation=1,
    )


def _roster() -> TeamRoleRoster:
    return TeamRoleRoster(
        "s132.match",
        (
            _model(
                "red.playmaker",
                "red",
                MatchRole.PLAYMAKER,
                ("red.finisher",),
                ("blue.goalkeeper",),
                (
                    SoccerSkill.LEAD_PASS,
                    SoccerSkill.FIRST_TOUCH,
                    SoccerSkill.DRIBBLE,
                    SoccerSkill.RECOVERY,
                ),
            ),
            _model(
                "red.finisher",
                "red",
                MatchRole.FINISHER,
                ("red.playmaker",),
                ("blue.goalkeeper",),
                (
                    SoccerSkill.FIRST_TOUCH,
                    SoccerSkill.OFF_BALL_RUN,
                    SoccerSkill.FINISHING,
                    SoccerSkill.RECOVERY,
                ),
            ),
            _model(
                "blue.goalkeeper",
                "blue",
                MatchRole.GOALKEEPER,
                (),
                ("red.playmaker", "red.finisher"),
                (
                    SoccerSkill.POSITIONING,
                    SoccerSkill.SAVE,
                    SoccerSkill.DISTRIBUTION,
                    SoccerSkill.RECOVERY,
                ),
            ),
        ),
    )


def _observation(
    model: RoleSelfModel,
    possession: PossessionOwner,
    possession_agent_id: str | None,
) -> EgocentricTeamObservation:
    return EgocentricTeamObservation(
        observer_agent_id=model.agent_id,
        self_model_hash=model.self_model_hash,
        match_state_hash=_hash("match-state"),
        ball_state_hash=_hash("ball-state"),
        self_state_hash=_hash(f"state-{model.agent_id}"),
        teammate_state_hashes=tuple(
            (value, _hash(f"state-{value}")) for value in model.teammate_ids
        ),
        opponent_state_hashes=tuple(
            (value, _hash(f"state-{value}")) for value in model.opponent_ids
        ),
        possession_owner=possession,
        possession_agent_id=possession_agent_id,
        self_to_ball_m=(1.0, 0.0, 0.0),
        own_goal_to_ball_m=(4.0, 0.0, 0.0),
        opponent_goal_to_ball_m=(7.0, 0.0, 0.0),
        time_sec=2.0,
    )


def _intent(
    model: RoleSelfModel,
    observation: EgocentricTeamObservation,
    intent: TacticalIntent,
    skill: SoccerSkill,
    target: str | None,
) -> RoleIntentCommitment:
    return RoleIntentCommitment(
        agent_id=model.agent_id,
        self_model_hash=model.self_model_hash,
        observation_hash=observation.observation_hash,
        policy_artifact_hash=model.policy_artifact_hash,
        intent=intent,
        skill=skill,
        target_agent_id=target,
        confidence=0.8,
        coordination_id="s132.clock.001",
    )


def test_role_roster_knows_complete_team_and_opposition() -> None:
    roster = _roster()
    assert roster.agent("red.playmaker").primary_role is MatchRole.PLAYMAKER
    assert roster.agent("red.finisher").allowed_intents
    with pytest.raises(ValueError, match="relationships"):
        TeamRoleRoster(
            roster.match_id,
            (replace(roster.agents[0], opponent_ids=()), *roster.agents[1:]),
        )


def test_role_authority_rejects_goalkeeper_shooting_with_save_skill() -> None:
    model = _roster().agent("blue.goalkeeper")
    observation = _observation(model, PossessionOwner.OPPONENT, "red.playmaker")
    commitment = _intent(model, observation, TacticalIntent.SHOOT, SoccerSkill.SAVE, None)
    with pytest.raises(ValueError, match="not authorized"):
        commitment.validate_for(model, observation)


def test_team_frame_requires_pass_receiver_handshake() -> None:
    roster = _roster()
    playmaker, finisher, goalkeeper = roster.agents
    observations = (
        _observation(playmaker, PossessionOwner.SELF, playmaker.agent_id),
        _observation(finisher, PossessionOwner.TEAMMATE, playmaker.agent_id),
        _observation(goalkeeper, PossessionOwner.OPPONENT, playmaker.agent_id),
    )
    intents = (
        _intent(
            playmaker,
            observations[0],
            TacticalIntent.PASS,
            SoccerSkill.LEAD_PASS,
            finisher.agent_id,
        ),
        _intent(
            finisher,
            observations[1],
            TacticalIntent.RECEIVE,
            SoccerSkill.FIRST_TOUCH,
            None,
        ),
        _intent(
            goalkeeper,
            observations[2],
            TacticalIntent.COVER,
            SoccerSkill.POSITIONING,
            None,
        ),
    )
    handshake = PassReceiveHandshake(
        handshake_id="s132.handshake.001",
        coordination_id="s132.clock.001",
        passer_agent_id=playmaker.agent_id,
        receiver_agent_id=finisher.agent_id,
        passer_self_model_hash=playmaker.self_model_hash,
        receiver_self_model_hash=finisher.self_model_hash,
        passer_observation_hash=observations[0].observation_hash,
        receiver_observation_hash=observations[1].observation_hash,
        pass_target_m=(1.30, -0.08, 0.115),
        predicted_arrival_time_sec=3.60,
        receiver_ready_start_sec=3.50,
        receiver_ready_end_sec=3.70,
        predicted_ball_speed_mps=2.0,
        accepted_by_receiver=True,
    )
    valid = TeamCoordinationFrame(
        roster,
        observations,
        intents,
        (handshake,),
    )
    assert valid.frame_hash.startswith("sha256:")
    invalid_receiver = _intent(
        finisher,
        observations[1],
        TacticalIntent.HOLD,
        SoccerSkill.RECOVERY,
        None,
    )
    with pytest.raises(ValueError, match="receiver handshake"):
        TeamCoordinationFrame(
            roster,
            observations,
            (valid.intents[0], invalid_receiver, valid.intents[2]),
            (handshake,),
        )


def test_team_frame_rejects_stale_pass_arrival_handshake() -> None:
    roster = _roster()
    playmaker, finisher, goalkeeper = roster.agents
    observations = (
        _observation(playmaker, PossessionOwner.SELF, playmaker.agent_id),
        _observation(finisher, PossessionOwner.TEAMMATE, playmaker.agent_id),
        _observation(goalkeeper, PossessionOwner.OPPONENT, playmaker.agent_id),
    )
    with pytest.raises(ValueError, match="timing handshake"):
        PassReceiveHandshake(
            "s132.handshake.stale",
            "s132.clock.001",
            playmaker.agent_id,
            finisher.agent_id,
            playmaker.self_model_hash,
            finisher.self_model_hash,
            observations[0].observation_hash,
            observations[1].observation_hash,
            (1.30, -0.08, 0.115),
            4.10,
            3.50,
            3.70,
            2.0,
            True,
        )


def test_pass_handshake_rejects_non_xyz_target() -> None:
    roster = _roster()
    playmaker, finisher, _ = roster.agents
    passer_observation = _observation(playmaker, PossessionOwner.SELF, playmaker.agent_id)
    receiver_observation = _observation(finisher, PossessionOwner.TEAMMATE, playmaker.agent_id)
    with pytest.raises(ValueError, match="must be xyz"):
        PassReceiveHandshake(
            "s132.handshake.bad-target",
            "s132.clock.001",
            playmaker.agent_id,
            finisher.agent_id,
            playmaker.self_model_hash,
            finisher.self_model_hash,
            passer_observation.observation_hash,
            receiver_observation.observation_hash,
            (1.30, -0.08),  # type: ignore[arg-type]
            3.60,
            3.50,
            3.70,
            2.0,
            True,
        )


def test_role_observation_rejects_teammate_opponent_swap() -> None:
    model = _roster().agent("red.playmaker")
    observation = _observation(model, PossessionOwner.TEAMMATE, "red.finisher")
    swapped = replace(
        observation,
        teammate_state_hashes=observation.opponent_state_hashes,
        opponent_state_hashes=observation.teammate_state_hashes,
    )
    with pytest.raises(ValueError, match="does not match"):
        swapped.validate_for(model)
