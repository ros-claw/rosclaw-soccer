"""Role-aware identity, skill authority, and team/opponent contracts.

These contracts sit above whole-body policies.  They make the agent that owns
an action explicit, distinguish team-mates from opponents, and prevent a
generic tactical policy from silently executing a skill outside its role.
They never authorize hardware or direct joint torque.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from rosclaw_soccer.sim.contracts import hash_json

_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.:-]{0,127}$")


class MatchRole(StrEnum):
    PLAYMAKER = "playmaker"
    FINISHER = "finisher"
    GOALKEEPER = "goalkeeper"
    DEFENDER = "defender"


class SoccerSkill(StrEnum):
    LEAD_PASS = "lead_pass"
    FIRST_TOUCH = "first_touch"
    DRIBBLE = "dribble"
    OFF_BALL_RUN = "off_ball_run"
    FINISHING = "finishing"
    WEAK_FOOT = "weak_foot"
    POSITIONING = "positioning"
    SAVE = "save"
    DISTRIBUTION = "distribution"
    MARKING = "marking"
    INTERCEPTION = "interception"
    BLOCKING = "blocking"
    RECOVERY = "recovery"


class TacticalIntent(StrEnum):
    PASS = "pass"
    RECEIVE = "receive"
    CARRY = "carry"
    SUPPORT = "support"
    RUN_IN_BEHIND = "run_in_behind"
    SHOOT = "shoot"
    PRESS = "press"
    COVER = "cover"
    INTERCEPT = "intercept"
    SAVE = "save"
    DISTRIBUTE = "distribute"
    RECOVER = "recover"
    HOLD = "hold"


class PossessionOwner(StrEnum):
    SELF = "self"
    TEAMMATE = "teammate"
    OPPONENT = "opponent"
    LOOSE = "loose"


_REQUIRED_SKILLS: dict[MatchRole, frozenset[SoccerSkill]] = {
    MatchRole.PLAYMAKER: frozenset(
        {SoccerSkill.LEAD_PASS, SoccerSkill.FIRST_TOUCH, SoccerSkill.RECOVERY}
    ),
    MatchRole.FINISHER: frozenset(
        {SoccerSkill.FIRST_TOUCH, SoccerSkill.FINISHING, SoccerSkill.RECOVERY}
    ),
    MatchRole.GOALKEEPER: frozenset(
        {SoccerSkill.POSITIONING, SoccerSkill.SAVE, SoccerSkill.RECOVERY}
    ),
    MatchRole.DEFENDER: frozenset(
        {SoccerSkill.MARKING, SoccerSkill.INTERCEPTION, SoccerSkill.RECOVERY}
    ),
}

_ROLE_INTENTS: dict[MatchRole, frozenset[TacticalIntent]] = {
    MatchRole.PLAYMAKER: frozenset(
        {
            TacticalIntent.PASS,
            TacticalIntent.RECEIVE,
            TacticalIntent.CARRY,
            TacticalIntent.SUPPORT,
            TacticalIntent.RECOVER,
            TacticalIntent.HOLD,
        }
    ),
    MatchRole.FINISHER: frozenset(
        {
            TacticalIntent.RECEIVE,
            TacticalIntent.SUPPORT,
            TacticalIntent.RUN_IN_BEHIND,
            TacticalIntent.SHOOT,
            TacticalIntent.RECOVER,
            TacticalIntent.HOLD,
        }
    ),
    MatchRole.GOALKEEPER: frozenset(
        {
            TacticalIntent.COVER,
            TacticalIntent.SAVE,
            TacticalIntent.DISTRIBUTE,
            TacticalIntent.RECOVER,
            TacticalIntent.HOLD,
        }
    ),
    MatchRole.DEFENDER: frozenset(
        {
            TacticalIntent.PRESS,
            TacticalIntent.COVER,
            TacticalIntent.INTERCEPT,
            TacticalIntent.RECOVER,
            TacticalIntent.HOLD,
        }
    ),
}

_INTENT_SKILLS: dict[TacticalIntent, frozenset[SoccerSkill]] = {
    TacticalIntent.PASS: frozenset({SoccerSkill.LEAD_PASS}),
    TacticalIntent.RECEIVE: frozenset({SoccerSkill.FIRST_TOUCH}),
    TacticalIntent.CARRY: frozenset({SoccerSkill.DRIBBLE}),
    TacticalIntent.SUPPORT: frozenset({SoccerSkill.LEAD_PASS, SoccerSkill.OFF_BALL_RUN}),
    TacticalIntent.RUN_IN_BEHIND: frozenset({SoccerSkill.OFF_BALL_RUN}),
    TacticalIntent.SHOOT: frozenset({SoccerSkill.FINISHING, SoccerSkill.WEAK_FOOT}),
    TacticalIntent.PRESS: frozenset({SoccerSkill.INTERCEPTION}),
    TacticalIntent.COVER: frozenset({SoccerSkill.POSITIONING, SoccerSkill.BLOCKING}),
    TacticalIntent.INTERCEPT: frozenset({SoccerSkill.INTERCEPTION}),
    TacticalIntent.SAVE: frozenset({SoccerSkill.SAVE}),
    TacticalIntent.DISTRIBUTE: frozenset({SoccerSkill.DISTRIBUTION}),
    TacticalIntent.RECOVER: frozenset({SoccerSkill.RECOVERY}),
    TacticalIntent.HOLD: frozenset(SoccerSkill),
}


@dataclass(frozen=True)
class RoleSkillBinding:
    """One role-owned champion skill and its current learning priority."""

    skill: SoccerSkill
    champion_artifact_hash: str
    evidence_hash: str
    generation: int
    proficiency: float
    training_priority: float
    schema_version: str = "rosclaw_soccer.role_skill_binding.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.skill, SoccerSkill):
            raise ValueError("role skill is unknown")
        _content_hash("champion_artifact_hash", self.champion_artifact_hash)
        _content_hash("evidence_hash", self.evidence_hash)
        if not 0 <= self.generation <= 1_000_000:
            raise ValueError("skill generation must be in [0, 1000000]")
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in (self.proficiency, self.training_priority)
        ):
            raise ValueError("skill proficiency and priority must be in [0, 1]")

    @property
    def binding_hash(self) -> str:
        return str(hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["skill"] = self.skill.value
        return value


@dataclass(frozen=True)
class RoleSelfModel:
    """A player's immutable answer to who it is and what it may learn/do."""

    agent_id: str
    team_id: str
    primary_role: MatchRole
    teammate_ids: tuple[str, ...]
    opponent_ids: tuple[str, ...]
    skills: tuple[RoleSkillBinding, ...]
    observation_contract_hash: str
    action_contract_hash: str
    policy_artifact_hash: str
    failure_memory_namespace: str
    generation: int
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    direct_joint_torque_output: bool = False
    schema_version: str = "rosclaw_soccer.role_self_model.v1"

    def __post_init__(self) -> None:
        for label, value in (
            ("agent_id", self.agent_id),
            ("team_id", self.team_id),
            ("failure_memory_namespace", self.failure_memory_namespace),
        ):
            _identifier(label, value)
        if not isinstance(self.primary_role, MatchRole):
            raise ValueError("primary role is unknown")
        teammates = tuple(self.teammate_ids)
        opponents = tuple(self.opponent_ids)
        relations = (*teammates, *opponents)
        if any(not _IDENTIFIER.fullmatch(value) for value in relations):
            raise ValueError("role relationships contain an invalid identity")
        if (
            self.agent_id in relations
            or len(set(relations)) != len(relations)
            or not opponents
            or set(teammates) & set(opponents)
        ):
            raise ValueError("role relationships must separate self, team-mates, and opponents")
        bindings = tuple(self.skills)
        skill_names = {binding.skill for binding in bindings}
        if (
            len(skill_names) != len(bindings)
            or not _REQUIRED_SKILLS[self.primary_role] <= skill_names
        ):
            raise ValueError("role self model lacks distinct mandatory skills")
        for label in (
            "observation_contract_hash",
            "action_contract_hash",
            "policy_artifact_hash",
        ):
            _content_hash(label, getattr(self, label))
        if not 0 <= self.generation <= 1_000_000:
            raise ValueError("role generation must be in [0, 1000000]")
        if (
            self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
            or self.direct_joint_torque_output
        ):
            raise ValueError("role self model must remain high-level and SIM_ONLY")
        object.__setattr__(self, "teammate_ids", teammates)
        object.__setattr__(self, "opponent_ids", opponents)
        object.__setattr__(self, "skills", bindings)

    @property
    def self_model_hash(self) -> str:
        return str(hash_json(self.to_dict()))

    @property
    def allowed_intents(self) -> tuple[TacticalIntent, ...]:
        return tuple(sorted(_ROLE_INTENTS[self.primary_role], key=lambda value: value.value))

    def skill(self, skill: SoccerSkill) -> RoleSkillBinding:
        try:
            return next(binding for binding in self.skills if binding.skill is skill)
        except StopIteration as exc:
            raise ValueError(f"{self.agent_id} does not own skill {skill.value}") from exc

    def authorizes(self, intent: TacticalIntent, skill: SoccerSkill) -> bool:
        return bool(
            intent in _ROLE_INTENTS[self.primary_role]
            and skill in _INTENT_SKILLS[intent]
            and any(binding.skill is skill for binding in self.skills)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "agent_id": self.agent_id,
            "team_id": self.team_id,
            "primary_role": self.primary_role.value,
            "teammate_ids": list(self.teammate_ids),
            "opponent_ids": list(self.opponent_ids),
            "skills": [binding.to_dict() for binding in self.skills],
            "allowed_intents": [intent.value for intent in self.allowed_intents],
            "observation_contract_hash": self.observation_contract_hash,
            "action_contract_hash": self.action_contract_hash,
            "policy_artifact_hash": self.policy_artifact_hash,
            "failure_memory_namespace": self.failure_memory_namespace,
            "generation": self.generation,
            "activation_ceiling": self.activation_ceiling,
            "hardware_authorized": self.hardware_authorized,
            "direct_joint_torque_output": self.direct_joint_torque_output,
        }


@dataclass(frozen=True)
class TeamRoleRoster:
    """A complete cross-referenced roster for all agents in one match."""

    match_id: str
    agents: tuple[RoleSelfModel, ...]
    schema_version: str = "rosclaw_soccer.team_role_roster.v1"

    def __post_init__(self) -> None:
        _identifier("match_id", self.match_id)
        agents = tuple(self.agents)
        if len(agents) < 3 or len({agent.agent_id for agent in agents}) != len(agents):
            raise ValueError("role roster requires at least three unique agents")
        teams = {agent.team_id for agent in agents}
        if len(teams) < 2:
            raise ValueError("role roster requires cooperation and opposition")
        for agent in agents:
            expected_teammates = {
                other.agent_id
                for other in agents
                if other.team_id == agent.team_id and other.agent_id != agent.agent_id
            }
            expected_opponents = {
                other.agent_id for other in agents if other.team_id != agent.team_id
            }
            if set(agent.teammate_ids) != expected_teammates:
                raise ValueError(f"{agent.agent_id} has an incomplete team-mate model")
            if set(agent.opponent_ids) != expected_opponents:
                raise ValueError(f"{agent.agent_id} has an incomplete opponent model")
        object.__setattr__(self, "agents", agents)

    @property
    def roster_hash(self) -> str:
        return str(hash_json(self.to_dict()))

    def agent(self, agent_id: str) -> RoleSelfModel:
        try:
            return next(agent for agent in self.agents if agent.agent_id == agent_id)
        except StopIteration as exc:
            raise ValueError(f"unknown roster agent: {agent_id}") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "match_id": self.match_id,
            "agents": [agent.to_dict() for agent in self.agents],
        }


@dataclass(frozen=True)
class EgocentricTeamObservation:
    """One agent's teammate/opponent-labelled, non-privileged world view."""

    observer_agent_id: str
    self_model_hash: str
    match_state_hash: str
    ball_state_hash: str
    self_state_hash: str
    teammate_state_hashes: tuple[tuple[str, str], ...]
    opponent_state_hashes: tuple[tuple[str, str], ...]
    possession_owner: PossessionOwner
    possession_agent_id: str | None
    self_to_ball_m: tuple[float, float, float]
    own_goal_to_ball_m: tuple[float, float, float]
    opponent_goal_to_ball_m: tuple[float, float, float]
    time_sec: float
    pixels_used: bool = False
    privileged_labels_used: bool = False
    schema_version: str = "rosclaw_soccer.egocentric_team_observation.v1"

    def __post_init__(self) -> None:
        _identifier("observer_agent_id", self.observer_agent_id)
        for label in (
            "self_model_hash",
            "match_state_hash",
            "ball_state_hash",
            "self_state_hash",
        ):
            _content_hash(label, getattr(self, label))
        teammate_states = tuple(self.teammate_state_hashes)
        opponent_states = tuple(self.opponent_state_hashes)
        identities = tuple(item[0] for item in (*teammate_states, *opponent_states))
        if (
            len(set(identities)) != len(identities)
            or self.observer_agent_id in identities
            or any(
                not _IDENTIFIER.fullmatch(agent_id) or not _HASH.fullmatch(state_hash)
                for agent_id, state_hash in (*teammate_states, *opponent_states)
            )
        ):
            raise ValueError("egocentric relationship states are invalid")
        vectors = (
            *self.self_to_ball_m,
            *self.own_goal_to_ball_m,
            *self.opponent_goal_to_ball_m,
        )
        if any(not math.isfinite(value) for value in vectors):
            raise ValueError("egocentric geometry must be finite")
        if not math.isfinite(self.time_sec) or self.time_sec < 0.0:
            raise ValueError("egocentric observation time is invalid")
        if not isinstance(self.possession_owner, PossessionOwner):
            raise ValueError("possession owner is invalid")
        if self.possession_owner in {PossessionOwner.SELF, PossessionOwner.LOOSE}:
            expected = (
                self.observer_agent_id if self.possession_owner is PossessionOwner.SELF else None
            )
            if self.possession_agent_id != expected:
                raise ValueError("self/loose possession identity is inconsistent")
        elif self.possession_agent_id not in identities:
            raise ValueError("possession agent is absent from the observed world")
        if self.pixels_used or self.privileged_labels_used:
            raise ValueError("role decisions require non-privileged physics observations")
        object.__setattr__(self, "teammate_state_hashes", teammate_states)
        object.__setattr__(self, "opponent_state_hashes", opponent_states)

    @property
    def observation_hash(self) -> str:
        return str(hash_json(self.to_dict()))

    def validate_for(self, model: RoleSelfModel) -> None:
        if (
            self.observer_agent_id != model.agent_id
            or self.self_model_hash != model.self_model_hash
            or {item[0] for item in self.teammate_state_hashes} != set(model.teammate_ids)
            or {item[0] for item in self.opponent_state_hashes} != set(model.opponent_ids)
        ):
            raise ValueError("egocentric observation does not match the role self model")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["possession_owner"] = self.possession_owner.value
        return value


@dataclass(frozen=True)
class RoleIntentCommitment:
    """One role-authorized high-level intent; never a joint command."""

    agent_id: str
    self_model_hash: str
    observation_hash: str
    policy_artifact_hash: str
    intent: TacticalIntent
    skill: SoccerSkill
    target_agent_id: str | None
    confidence: float
    coordination_id: str
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    direct_joint_torque_output: bool = False
    schema_version: str = "rosclaw_soccer.role_intent_commitment.v1"

    def __post_init__(self) -> None:
        _identifier("agent_id", self.agent_id)
        _identifier("coordination_id", self.coordination_id)
        for label in ("self_model_hash", "observation_hash", "policy_artifact_hash"):
            _content_hash(label, getattr(self, label))
        if not isinstance(self.intent, TacticalIntent) or not isinstance(self.skill, SoccerSkill):
            raise ValueError("role intent or selected skill is invalid")
        if self.target_agent_id is not None:
            _identifier("target_agent_id", self.target_agent_id)
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("role intent confidence must be in [0, 1]")
        if (
            self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
            or self.direct_joint_torque_output
        ):
            raise ValueError("role intent must remain high-level and SIM_ONLY")

    @property
    def commitment_hash(self) -> str:
        return str(hash_json(self.to_dict()))

    def validate_for(
        self,
        model: RoleSelfModel,
        observation: EgocentricTeamObservation,
    ) -> None:
        observation.validate_for(model)
        if (
            self.agent_id != model.agent_id
            or self.self_model_hash != model.self_model_hash
            or self.observation_hash != observation.observation_hash
            or self.policy_artifact_hash != model.policy_artifact_hash
            or not model.authorizes(self.intent, self.skill)
        ):
            raise ValueError("intent is not authorized by this agent's role and skill model")
        if self.intent is TacticalIntent.PASS and self.target_agent_id not in model.teammate_ids:
            raise ValueError("a pass must target a declared team-mate")
        if self.intent in {TacticalIntent.PRESS, TacticalIntent.INTERCEPT} and (
            self.target_agent_id not in model.opponent_ids
        ):
            raise ValueError("press/intercept must target a declared opponent")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["intent"] = self.intent.value
        value["skill"] = self.skill.value
        return value


@dataclass(frozen=True)
class PassReceiveHandshake:
    """A jointly committed pass-arrival and receiver-readiness window."""

    handshake_id: str
    coordination_id: str
    passer_agent_id: str
    receiver_agent_id: str
    passer_self_model_hash: str
    receiver_self_model_hash: str
    passer_observation_hash: str
    receiver_observation_hash: str
    pass_target_m: tuple[float, float, float]
    predicted_arrival_time_sec: float
    receiver_ready_start_sec: float
    receiver_ready_end_sec: float
    predicted_ball_speed_mps: float
    accepted_by_receiver: bool
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.pass_receive_handshake.v1"

    def __post_init__(self) -> None:
        if len(self.pass_target_m) != 3:
            raise ValueError("pass-receive target must be xyz")
        for label, value in (
            ("handshake_id", self.handshake_id),
            ("coordination_id", self.coordination_id),
            ("passer_agent_id", self.passer_agent_id),
            ("receiver_agent_id", self.receiver_agent_id),
        ):
            _identifier(label, value)
        for label, value in (
            ("passer_self_model_hash", self.passer_self_model_hash),
            ("receiver_self_model_hash", self.receiver_self_model_hash),
            ("passer_observation_hash", self.passer_observation_hash),
            ("receiver_observation_hash", self.receiver_observation_hash),
        ):
            _content_hash(label, value)
        values = (
            *self.pass_target_m,
            self.predicted_arrival_time_sec,
            self.receiver_ready_start_sec,
            self.receiver_ready_end_sec,
            self.predicted_ball_speed_mps,
        )
        if (
            any(not math.isfinite(value) for value in values)
            or self.passer_agent_id == self.receiver_agent_id
            or self.predicted_arrival_time_sec < 0.0
            or self.receiver_ready_start_sec < 0.0
            or self.receiver_ready_end_sec <= self.receiver_ready_start_sec
            or not self.receiver_ready_start_sec
            <= self.predicted_arrival_time_sec
            <= self.receiver_ready_end_sec
            or not 0.1 <= self.predicted_ball_speed_mps <= 15.0
            or not self.accepted_by_receiver
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("pass-receive timing handshake is invalid")

    @property
    def handshake_hash(self) -> str:
        return str(hash_json(self.to_dict()))

    def validate_for(
        self,
        *,
        roster: TeamRoleRoster,
        passer_observation: EgocentricTeamObservation,
        receiver_observation: EgocentricTeamObservation,
        passer_intent: RoleIntentCommitment,
        receiver_intent: RoleIntentCommitment,
    ) -> None:
        passer = roster.agent(self.passer_agent_id)
        receiver = roster.agent(self.receiver_agent_id)
        if (
            passer.team_id != receiver.team_id
            or self.receiver_agent_id not in passer.teammate_ids
            or self.passer_agent_id not in receiver.teammate_ids
            or self.passer_self_model_hash != passer.self_model_hash
            or self.receiver_self_model_hash != receiver.self_model_hash
            or self.passer_observation_hash != passer_observation.observation_hash
            or self.receiver_observation_hash != receiver_observation.observation_hash
            or passer_intent.agent_id != self.passer_agent_id
            or passer_intent.intent is not TacticalIntent.PASS
            or passer_intent.target_agent_id != self.receiver_agent_id
            or receiver_intent.agent_id != self.receiver_agent_id
            or receiver_intent.intent
            not in {
                TacticalIntent.RECEIVE,
                TacticalIntent.SUPPORT,
                TacticalIntent.RUN_IN_BEHIND,
            }
            or passer_intent.coordination_id != self.coordination_id
            or receiver_intent.coordination_id != self.coordination_id
        ):
            raise ValueError("pass-receive handshake does not bind the coordinated roles")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TeamCoordinationFrame:
    """Simultaneous role intents with pass handshakes and adversarial presence."""

    roster: TeamRoleRoster
    observations: tuple[EgocentricTeamObservation, ...]
    intents: tuple[RoleIntentCommitment, ...]
    pass_receive_handshakes: tuple[PassReceiveHandshake, ...] = ()
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.team_coordination_frame.v1"

    def __post_init__(self) -> None:
        observations = tuple(self.observations)
        intents = tuple(self.intents)
        handshakes = tuple(self.pass_receive_handshakes)
        agent_ids = {agent.agent_id for agent in self.roster.agents}
        if (
            {value.observer_agent_id for value in observations} != agent_ids
            or len(observations) != len(agent_ids)
            or {value.agent_id for value in intents} != agent_ids
            or len(intents) != len(agent_ids)
        ):
            raise ValueError("coordination frame requires one observation and intent per agent")
        if len({intent.coordination_id for intent in intents}) != 1:
            raise ValueError("coordination frame intents do not share one clock commitment")
        observations_by_agent = {value.observer_agent_id: value for value in observations}
        intents_by_agent = {value.agent_id: value for value in intents}
        for agent_id in agent_ids:
            intents_by_agent[agent_id].validate_for(
                self.roster.agent(agent_id), observations_by_agent[agent_id]
            )
        for team_id in {agent.team_id for agent in self.roster.agents}:
            team_intents = [
                value for value in intents if self.roster.agent(value.agent_id).team_id == team_id
            ]
            ball_actions = [
                value
                for value in team_intents
                if value.intent in {TacticalIntent.PASS, TacticalIntent.CARRY, TacticalIntent.SHOOT}
            ]
            if len(ball_actions) > 1:
                raise ValueError("one team cannot commit multiple simultaneous ball owners")
        for commitment in intents:
            if commitment.intent is not TacticalIntent.PASS:
                continue
            assert commitment.target_agent_id is not None
            receiver = intents_by_agent[commitment.target_agent_id]
            if receiver.intent not in {
                TacticalIntent.RECEIVE,
                TacticalIntent.SUPPORT,
                TacticalIntent.RUN_IN_BEHIND,
            }:
                raise ValueError("a pass requires an explicit receiver handshake")
            matching = [
                handshake
                for handshake in handshakes
                if handshake.passer_agent_id == commitment.agent_id
                and handshake.receiver_agent_id == commitment.target_agent_id
                and handshake.coordination_id == commitment.coordination_id
            ]
            if len(matching) != 1:
                raise ValueError("a pass requires one timing handshake")
            matching[0].validate_for(
                roster=self.roster,
                passer_observation=observations_by_agent[commitment.agent_id],
                receiver_observation=observations_by_agent[commitment.target_agent_id],
                passer_intent=commitment,
                receiver_intent=receiver,
            )
        pass_intents = [intent for intent in intents if intent.intent is TacticalIntent.PASS]
        if len(handshakes) != len(pass_intents):
            raise ValueError("coordination frame contains an orphan timing handshake")
        if self.activation_ceiling != "SIM_ONLY" or self.hardware_authorized:
            raise ValueError("coordination frames must remain SIM_ONLY")
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "intents", intents)
        object.__setattr__(self, "pass_receive_handshakes", handshakes)

    @property
    def frame_hash(self) -> str:
        return str(hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "roster": self.roster.to_dict(),
            "roster_hash": self.roster.roster_hash,
            "observations": [value.to_dict() for value in self.observations],
            "intents": [value.to_dict() for value in self.intents],
            "pass_receive_handshakes": [value.to_dict() for value in self.pass_receive_handshakes],
            "activation_ceiling": self.activation_ceiling,
            "hardware_authorized": self.hardware_authorized,
        }


def _content_hash(label: str, value: str) -> None:
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise ValueError(f"{label} must be a sha256 content hash")


def _identifier(label: str, value: str) -> None:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{label} must be a normalized identifier")


__all__ = [
    "EgocentricTeamObservation",
    "MatchRole",
    "PassReceiveHandshake",
    "PossessionOwner",
    "RoleIntentCommitment",
    "RoleSelfModel",
    "RoleSkillBinding",
    "SoccerSkill",
    "TacticalIntent",
    "TeamCoordinationFrame",
    "TeamRoleRoster",
]
