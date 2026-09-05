"""Independent ROSClaw development cells for multi-player soccer.

Each cell owns a role self-model, a ROSClaw Core ``IndividualGrowthScope``,
private memory namespaces, and one high-level tactical policy identity.  Cells
receive egocentric physics observations and submit role-authorized intentions;
they never own poses, joints, torques, football state, ROS, or hardware.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from rosclaw.continual.contracts import PolicyVersion
from rosclaw.continual.individual_scope import IndividualGrowthScope
from rosclaw.continual.plasticity_lease import (
    AgentPolicyBinding,
    AgentUpdateMode,
    PlasticityLease,
)

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

_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.:-]{0,127}$")


@dataclass(frozen=True)
class AgentPhysicalState:
    agent_id: str
    position_m: tuple[float, float, float]
    velocity_mps: tuple[float, float, float]
    pelvis_height_m: float
    tilt_rad: float
    stable: bool
    schema_version: str = "rosclaw_soccer.agent_physical_state.v1"

    def __post_init__(self) -> None:
        values = (*self.position_m, *self.velocity_mps, self.pelvis_height_m, self.tilt_rad)
        if (
            not _IDENTIFIER.fullmatch(self.agent_id)
            or len(self.position_m) != 3
            or len(self.velocity_mps) != 3
            or any(not math.isfinite(value) for value in values)
            or self.pelvis_height_m < 0.0
            or self.tilt_rad < 0.0
            or not isinstance(self.stable, bool)
        ):
            raise ValueError("agent physical state is invalid")

    @property
    def state_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass(frozen=True)
class AgentCellObservation:
    observer_agent_id: str
    time_sec: float
    ball_position_m: tuple[float, float, float]
    ball_velocity_mps: tuple[float, float, float]
    own_goal_m: tuple[float, float, float]
    opponent_goal_m: tuple[float, float, float]
    possession_agent_id: str | None
    self_state: AgentPhysicalState
    teammate_states: tuple[AgentPhysicalState, ...]
    opponent_states: tuple[AgentPhysicalState, ...]
    pixels_used: bool = False
    privileged_labels_used: bool = False
    schema_version: str = "rosclaw_soccer.agent_cell_observation.v1"

    def __post_init__(self) -> None:
        vectors = (
            *self.ball_position_m,
            *self.ball_velocity_mps,
            *self.own_goal_m,
            *self.opponent_goal_m,
        )
        related = (*self.teammate_states, *self.opponent_states)
        identities = {state.agent_id for state in related}
        if (
            not _IDENTIFIER.fullmatch(self.observer_agent_id)
            or any(
                len(value) != 3
                for value in (
                    self.ball_position_m,
                    self.ball_velocity_mps,
                    self.own_goal_m,
                    self.opponent_goal_m,
                )
            )
            or self.self_state.agent_id != self.observer_agent_id
            or not math.isfinite(self.time_sec)
            or self.time_sec < 0.0
            or any(not math.isfinite(value) for value in vectors)
            or len(identities) != len(related)
            or self.observer_agent_id in identities
            or (
                self.possession_agent_id is not None
                and self.possession_agent_id not in identities | {self.observer_agent_id}
            )
            or self.pixels_used
            or self.privileged_labels_used
        ):
            raise ValueError("agent-cell observation is invalid")

    @property
    def observation_hash(self) -> str:
        return str(hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "observer_agent_id": self.observer_agent_id,
            "time_sec": self.time_sec,
            "ball_position_m": list(self.ball_position_m),
            "ball_velocity_mps": list(self.ball_velocity_mps),
            "own_goal_m": list(self.own_goal_m),
            "opponent_goal_m": list(self.opponent_goal_m),
            "possession_agent_id": self.possession_agent_id,
            "self_state": asdict(self.self_state),
            "teammate_states": [asdict(value) for value in self.teammate_states],
            "opponent_states": [asdict(value) for value in self.opponent_states],
            "pixels_used": self.pixels_used,
            "privileged_labels_used": self.privileged_labels_used,
        }


@dataclass(frozen=True)
class AgentTacticalProfile:
    home_position_m: tuple[float, float, float]
    maximum_target_shift_m: float = 2.0
    decision_period_sec: float = 0.10
    intent_hysteresis_sec: float = 0.20
    pass_lane_clearance_m: float = 0.55
    goalkeeper_depth_m: float = 0.48
    schema_version: str = "rosclaw_soccer.agent_tactical_profile.v1"

    def __post_init__(self) -> None:
        values = (
            *self.home_position_m,
            self.maximum_target_shift_m,
            self.decision_period_sec,
            self.intent_hysteresis_sec,
            self.pass_lane_clearance_m,
            self.goalkeeper_depth_m,
        )
        if (
            len(self.home_position_m) != 3
            or any(not math.isfinite(value) for value in values)
            or abs(self.home_position_m[2]) > 1.0e-12
            or not 0.25 <= self.maximum_target_shift_m <= 4.0
            or not 0.08 <= self.decision_period_sec <= 0.20
            or not 0.10 <= self.intent_hysteresis_sec <= 0.60
            or not 0.30 <= self.pass_lane_clearance_m <= 1.20
            or not 0.20 <= self.goalkeeper_depth_m <= 0.90
        ):
            raise ValueError("agent tactical profile is invalid")

    @property
    def profile_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass(frozen=True)
class AgentCellDecision:
    agent_id: str
    intent: TacticalIntent
    skill: SoccerSkill
    target_position_m: tuple[float, float, float]
    target_agent_id: str | None
    confidence: float
    observation_hash: str
    policy_artifact_hash: str
    schema_version: str = "rosclaw_soccer.agent_cell_decision.v1"

    def __post_init__(self) -> None:
        if (
            not _IDENTIFIER.fullmatch(self.agent_id)
            or not isinstance(self.intent, TacticalIntent)
            or not isinstance(self.skill, SoccerSkill)
            or len(self.target_position_m) != 3
            or any(not math.isfinite(value) for value in self.target_position_m)
            or not math.isfinite(self.confidence)
            or not 0.0 <= self.confidence <= 1.0
            or (
                self.target_agent_id is not None and not _IDENTIFIER.fullmatch(self.target_agent_id)
            )
            or not _HASH.fullmatch(self.observation_hash)
            or not _HASH.fullmatch(self.policy_artifact_hash)
        ):
            raise ValueError("agent-cell decision is invalid")

    @property
    def decision_hash(self) -> str:
        return str(hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["intent"] = self.intent.value
        value["skill"] = self.skill.value
        return value


@dataclass(frozen=True)
class RosclawSoccerAgentCell:
    """One independently versioned ROSClaw learner behind one G1 body."""

    self_model: RoleSelfModel
    growth_scope: IndividualGrowthScope
    tactical_profile: AgentTacticalProfile
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    direct_joint_torque_output: bool = False
    schema_version: str = "rosclaw_soccer.independent_agent_cell.v1"

    def __post_init__(self) -> None:
        if (
            self.self_model.agent_id != self.growth_scope.agent_id
            or self.self_model.failure_memory_namespace
            != self.growth_scope.failure_memory_namespace
            or self.self_model.policy_artifact_hash
            != self.growth_scope.champion_policy.artifact_hash
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
            or self.direct_joint_torque_output
        ):
            raise ValueError("agent cell identity, lineage, or authority is inconsistent")

    @property
    def agent_id(self) -> str:
        return self.self_model.agent_id

    @property
    def cell_hash(self) -> str:
        return str(hash_json(self.to_dict()))

    def decide(self, observation: AgentCellObservation) -> AgentCellDecision:
        """Choose a role-authorized intent from this cell's own observation."""

        if observation.observer_agent_id != self.agent_id:
            raise ValueError("agent cell cannot consume another player's observation")
        if {state.agent_id for state in observation.teammate_states} != set(
            self.self_model.teammate_ids
        ) or {state.agent_id for state in observation.opponent_states} != set(
            self.self_model.opponent_ids
        ):
            raise ValueError("agent cell observation has a mismatched social graph")
        role = self.self_model.primary_role
        if not observation.self_state.stable:
            return self._decision(
                observation,
                TacticalIntent.RECOVER,
                SoccerSkill.RECOVERY,
                self.tactical_profile.home_position_m,
                None,
                1.0,
            )
        if role is MatchRole.GOALKEEPER:
            return self._goalkeeper_decision(observation)
        if role is MatchRole.DEFENDER:
            return self._defender_decision(observation)
        if role is MatchRole.PLAYMAKER:
            return self._playmaker_decision(observation)
        return self._finisher_decision(observation)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "self_model": self.self_model.to_dict(),
            "self_model_hash": self.self_model.self_model_hash,
            "growth_scope": self.growth_scope.to_dict(),
            "growth_scope_hash": self.growth_scope.scope_hash,
            "tactical_profile": asdict(self.tactical_profile),
            "tactical_profile_hash": self.tactical_profile.profile_hash,
            "activation_ceiling": self.activation_ceiling,
            "hardware_authorized": self.hardware_authorized,
            "direct_joint_torque_output": self.direct_joint_torque_output,
        }

    def _goalkeeper_decision(self, value: AgentCellObservation) -> AgentCellDecision:
        ball = np.asarray(value.ball_position_m, dtype=np.float64)
        velocity = np.asarray(value.ball_velocity_mps, dtype=np.float64)
        own_goal = np.asarray(value.own_goal_m, dtype=np.float64)
        if value.possession_agent_id == self.agent_id:
            available = tuple(state for state in value.teammate_states if state.stable)
            if not available:
                return self._decision(
                    value,
                    TacticalIntent.COVER,
                    SoccerSkill.POSITIONING,
                    self.tactical_profile.home_position_m,
                    None,
                    0.98,
                )
            opponent_goal = np.asarray(value.opponent_goal_m[:2], dtype=np.float64)
            teammate = min(
                available,
                key=lambda state: float(
                    np.linalg.norm(
                        np.asarray(state.position_m[:2], dtype=np.float64) - opponent_goal
                    )
                ),
            )
            return self._decision(
                value,
                TacticalIntent.DISTRIBUTE,
                SoccerSkill.DISTRIBUTION,
                teammate.position_m,
                teammate.agent_id,
                0.90,
            )
        toward_goal = float(np.dot(velocity[:2], own_goal[:2] - ball[:2])) > 0.10
        danger = float(np.linalg.norm(own_goal[:2] - ball[:2])) < 3.5
        predicted_y = float(ball[1])
        if abs(float(velocity[0])) > 0.10:
            arrival = max(0.0, float((own_goal[0] - ball[0]) / velocity[0]))
            predicted_y += min(1.5, arrival) * float(velocity[1])
        target = (
            float(
                own_goal[0] - math.copysign(self.tactical_profile.goalkeeper_depth_m, own_goal[0])
            ),
            float(np.clip(predicted_y, own_goal[1] - 1.25, own_goal[1] + 1.25)),
            0.0,
        )
        return self._decision(
            value,
            TacticalIntent.SAVE if toward_goal and danger else TacticalIntent.COVER,
            SoccerSkill.SAVE if toward_goal and danger else SoccerSkill.POSITIONING,
            target,
            value.possession_agent_id,
            0.96 if toward_goal and danger else 0.82,
        )

    def _defender_decision(self, value: AgentCellObservation) -> AgentCellDecision:
        ball = np.asarray(value.ball_position_m, dtype=np.float64)
        self_xy = np.asarray(value.self_state.position_m[:2], dtype=np.float64)
        defenders = (
            value.self_state,
            *(state for state in value.teammate_states if state.agent_id != self.agent_id),
        )
        nearest = min(
            defenders,
            key=lambda state: float(
                np.linalg.norm(np.asarray(state.position_m[:2], dtype=np.float64) - ball[:2])
            ),
        )
        opponent_has_ball = value.possession_agent_id in self.self_model.opponent_ids
        if opponent_has_ball and nearest.agent_id == self.agent_id:
            carrier = next(
                state
                for state in value.opponent_states
                if state.agent_id == value.possession_agent_id
            )
            intent = (
                TacticalIntent.INTERCEPT
                if float(np.linalg.norm(self_xy - ball[:2])) < 1.0
                else TacticalIntent.PRESS
            )
            return self._decision(
                value,
                intent,
                SoccerSkill.INTERCEPTION,
                carrier.position_m,
                carrier.agent_id,
                0.91,
            )
        own_goal = np.asarray(value.own_goal_m, dtype=np.float64)
        cover = 0.58 * ball[:2] + 0.42 * own_goal[:2]
        home_y = self.tactical_profile.home_position_m[1]
        target = (float(cover[0]), float(0.72 * cover[1] + 0.28 * home_y), 0.0)
        return self._decision(
            value,
            TacticalIntent.COVER,
            SoccerSkill.BLOCKING,
            target,
            value.possession_agent_id,
            0.84,
        )

    def _playmaker_decision(self, value: AgentCellObservation) -> AgentCellDecision:
        if value.possession_agent_id == self.agent_id:
            receiver = self._best_receiver(value)
            if receiver is not None and self._lane_clear(value, receiver):
                lead = np.asarray(receiver.position_m, dtype=np.float64) + 0.35 * np.asarray(
                    receiver.velocity_mps, dtype=np.float64
                )
                return self._decision(
                    value,
                    TacticalIntent.PASS,
                    SoccerSkill.LEAD_PASS,
                    (float(lead[0]), float(lead[1]), 0.0),
                    receiver.agent_id,
                    0.94,
                )
            return self._decision(
                value,
                TacticalIntent.CARRY,
                SoccerSkill.DRIBBLE,
                self._toward_goal(value, distance_m=1.1),
                None,
                0.76,
            )
        if value.possession_agent_id in self.self_model.teammate_ids:
            return self._decision(
                value,
                TacticalIntent.SUPPORT,
                SoccerSkill.OFF_BALL_RUN,
                self._support_target(value, depth_m=-0.65),
                value.possession_agent_id,
                0.88,
            )
        return self._decision(
            value,
            TacticalIntent.RECEIVE,
            SoccerSkill.FIRST_TOUCH,
            value.ball_position_m,
            None,
            0.80,
        )

    def _finisher_decision(self, value: AgentCellObservation) -> AgentCellDecision:
        if value.possession_agent_id == self.agent_id:
            return self._decision(
                value,
                TacticalIntent.SHOOT,
                SoccerSkill.FINISHING,
                value.opponent_goal_m,
                None,
                0.97,
            )
        if value.possession_agent_id in self.self_model.teammate_ids:
            return self._decision(
                value,
                TacticalIntent.RUN_IN_BEHIND,
                SoccerSkill.OFF_BALL_RUN,
                self._support_target(value, depth_m=0.95),
                value.possession_agent_id,
                0.91,
            )
        return self._decision(
            value,
            TacticalIntent.RECEIVE,
            SoccerSkill.FIRST_TOUCH,
            value.ball_position_m,
            None,
            0.82,
        )

    def _decision(
        self,
        observation: AgentCellObservation,
        intent: TacticalIntent,
        skill: SoccerSkill,
        target: tuple[float, float, float],
        target_agent_id: str | None,
        confidence: float,
    ) -> AgentCellDecision:
        if not self.self_model.authorizes(intent, skill):
            raise ValueError(f"{self.agent_id} attempted an unauthorized tactical option")
        home = np.asarray(self.tactical_profile.home_position_m, dtype=np.float64)
        requested = np.asarray(target, dtype=np.float64)
        delta = requested[:2] - home[:2]
        distance = float(np.linalg.norm(delta))
        if distance > self.tactical_profile.maximum_target_shift_m:
            requested[:2] = home[:2] + delta * (
                self.tactical_profile.maximum_target_shift_m / distance
            )
        requested[2] = 0.0
        return AgentCellDecision(
            agent_id=self.agent_id,
            intent=intent,
            skill=skill,
            target_position_m=(float(requested[0]), float(requested[1]), 0.0),
            target_agent_id=target_agent_id,
            confidence=confidence,
            observation_hash=observation.observation_hash,
            policy_artifact_hash=self.self_model.policy_artifact_hash,
        )

    def _best_receiver(self, value: AgentCellObservation) -> AgentPhysicalState | None:
        attack_direction = float(
            np.sign(value.opponent_goal_m[0] - value.self_state.position_m[0]) or 1.0
        )
        candidates = [
            state
            for state in value.teammate_states
            if state.stable
            and attack_direction * (state.position_m[0] - value.self_state.position_m[0]) > -0.20
        ]
        if not candidates:
            return None
        goal = np.asarray(value.opponent_goal_m[:2], dtype=np.float64)
        return min(
            candidates,
            key=lambda state: float(
                np.linalg.norm(np.asarray(state.position_m[:2], dtype=np.float64) - goal)
            ),
        )

    def _lane_clear(self, value: AgentCellObservation, receiver: AgentPhysicalState) -> bool:
        start = np.asarray(value.self_state.position_m[:2], dtype=np.float64)
        end = np.asarray(receiver.position_m[:2], dtype=np.float64)
        segment = end - start
        denominator = float(np.dot(segment, segment))
        if denominator <= 1.0e-9:
            return False
        for opponent in value.opponent_states:
            point = np.asarray(opponent.position_m[:2], dtype=np.float64)
            phase = float(np.clip(np.dot(point - start, segment) / denominator, 0.0, 1.0))
            if float(np.linalg.norm(point - (start + phase * segment))) < (
                self.tactical_profile.pass_lane_clearance_m
            ):
                return False
        return True

    def _toward_goal(
        self, value: AgentCellObservation, *, distance_m: float
    ) -> tuple[float, float, float]:
        current = np.asarray(value.self_state.position_m[:2], dtype=np.float64)
        goal = np.asarray(value.opponent_goal_m[:2], dtype=np.float64)
        direction = goal - current
        direction /= max(float(np.linalg.norm(direction)), 1.0e-9)
        target = current + distance_m * direction
        return (float(target[0]), float(target[1]), 0.0)

    def _support_target(
        self, value: AgentCellObservation, *, depth_m: float
    ) -> tuple[float, float, float]:
        ball = np.asarray(value.ball_position_m[:2], dtype=np.float64)
        goal = np.asarray(value.opponent_goal_m[:2], dtype=np.float64)
        direction = goal - ball
        direction /= max(float(np.linalg.norm(direction)), 1.0e-9)
        lateral = np.asarray((-direction[1], direction[0]), dtype=np.float64)
        home_side = float(np.sign(self.tactical_profile.home_position_m[1]) or 1.0)
        target = ball + depth_m * direction + 0.65 * home_side * lateral
        return (float(target[0]), float(target[1]), 0.0)


def build_independent_agent_cell(
    *,
    agent_id: str,
    team_id: str,
    primary_role: MatchRole,
    teammate_ids: tuple[str, ...],
    opponent_ids: tuple[str, ...],
    body_hash: str,
    foundation_policy_hash: str,
    home_position_m: tuple[float, float, float],
    generation: int = 0,
) -> RosclawSoccerAgentCell:
    """Build one fully isolated cell from content-addressed ROSClaw state."""

    if generation != 0:
        raise ValueError(
            "bootstrap agent-cell construction is generation zero only; "
            "later generations require an evidence-bound parent policy"
        )
    if not _HASH.fullmatch(body_hash) or not _HASH.fullmatch(foundation_policy_hash):
        raise ValueError("agent cell requires content-addressed body and foundation")
    policy_artifact_hash = str(
        hash_json(
            {
                "agent_id": agent_id,
                "team_id": team_id,
                "role": primary_role.value,
                "generation": generation,
                "foundation_policy_hash": foundation_policy_hash,
            }
        )
    )
    skills = _skills_for_role(
        agent_id=agent_id,
        role=primary_role,
        generation=generation,
        foundation_policy_hash=foundation_policy_hash,
    )
    observation_contract_hash = str(
        hash_json({"schema": AgentCellObservation.schema_version, "agent_id": agent_id})
    )
    action_contract_hash = str(
        hash_json({"schema": AgentCellDecision.schema_version, "agent_id": agent_id})
    )
    failure_namespace = f"soccer.failure.{agent_id}"
    self_model = RoleSelfModel(
        agent_id=agent_id,
        team_id=team_id,
        primary_role=primary_role,
        teammate_ids=tuple(sorted(teammate_ids)),
        opponent_ids=tuple(sorted(opponent_ids)),
        skills=skills,
        observation_contract_hash=observation_contract_hash,
        action_contract_hash=action_contract_hash,
        policy_artifact_hash=policy_artifact_hash,
        failure_memory_namespace=failure_namespace,
        generation=generation,
    )
    controller_hash = str(hash_json({"controller": "bounded_soccer_option_v1"}))
    safety_hash = str(hash_json({"safety": "sim_only_no_joint_authority_v1"}))
    policy = PolicyVersion(
        version=generation,
        artifact_hash=policy_artifact_hash,
        parent_version_hash=None,
        controller_snapshot_hash=controller_hash,
        body_hash=body_hash,
        safety_kernel_hash=safety_hash,
        observation_names=(
            "self_to_ball_xyz",
            "ball_velocity_xyz",
            "own_goal_to_ball_xyz",
            "opponent_goal_to_ball_xyz",
            "teammate_states",
            "opponent_states",
            "possession_owner",
        ),
        residual_action_names=("tactical_intent", "target_x", "target_y"),
    )
    scope = IndividualGrowthScope(
        agent_id=agent_id,
        body_hash=body_hash,
        body_state_hash=str(hash_json({"agent_id": agent_id, "body": body_hash})),
        foundation_policy_hash=foundation_policy_hash,
        personal_adapter_hash=str(hash_json({"agent_id": agent_id, "adapter": generation})),
        role_policy_hash=policy_artifact_hash,
        residual_policy_hash=str(hash_json({"agent_id": agent_id, "residual": generation})),
        capability_profile_hash=str(
            hash_json({"agent_id": agent_id, "role": primary_role.value, "generation": generation})
        ),
        career_lineage_hash=str(hash_json({"agent_id": agent_id, "career": generation})),
        personal_memory_namespace=f"soccer.memory.{agent_id}",
        failure_memory_namespace=failure_namespace,
        parent_policy=policy,
        champion_policy=policy,
        generation=generation,
    )
    return RosclawSoccerAgentCell(
        self_model=self_model,
        growth_scope=scope,
        tactical_profile=AgentTacticalProfile(home_position_m=home_position_m),
    )


def build_team_coordination_frame(
    *,
    roster: TeamRoleRoster,
    cells: tuple[RosclawSoccerAgentCell, ...],
    observations: tuple[AgentCellObservation, ...],
    decisions: tuple[AgentCellDecision, ...],
    frame_index: int,
) -> TeamCoordinationFrame:
    """Bind independently produced decisions into one fail-closed team frame."""

    if frame_index < 0:
        raise ValueError("coordination frame index must be non-negative")
    cell_by_id = {cell.agent_id: cell for cell in cells}
    observation_by_id = {value.observer_agent_id: value for value in observations}
    decision_by_id = {value.agent_id: value for value in decisions}
    expected = {agent.agent_id for agent in roster.agents}
    if (
        set(cell_by_id) != expected
        or set(observation_by_id) != expected
        or set(decision_by_id) != expected
        or len(cells) != len(expected)
        or len(observations) != len(expected)
        or len(decisions) != len(expected)
    ):
        raise ValueError("coordination requires exactly one independent cell result per player")
    coordination_id = f"team.frame.{frame_index:06d}"
    role_observations: list[EgocentricTeamObservation] = []
    commitments: list[RoleIntentCommitment] = []
    for agent_id in sorted(expected):
        cell = cell_by_id[agent_id]
        observation = observation_by_id[agent_id]
        decision = decision_by_id[agent_id]
        if (
            decision.observation_hash != observation.observation_hash
            or decision.policy_artifact_hash != cell.self_model.policy_artifact_hash
        ):
            raise ValueError("decision is not bound to the current agent observation and policy")
        owner, owner_id = _possession_for(cell.self_model, observation.possession_agent_id)
        self_position = np.asarray(observation.self_state.position_m, dtype=np.float64)
        ball = np.asarray(observation.ball_position_m, dtype=np.float64)
        own_goal = np.asarray(observation.own_goal_m, dtype=np.float64)
        opponent_goal = np.asarray(observation.opponent_goal_m, dtype=np.float64)
        role_observation = EgocentricTeamObservation(
            observer_agent_id=agent_id,
            self_model_hash=cell.self_model.self_model_hash,
            match_state_hash=str(
                hash_json(
                    {
                        "frame": frame_index,
                        "time_sec": observation.time_sec,
                        "observation_hash": observation.observation_hash,
                    }
                )
            ),
            ball_state_hash=str(
                hash_json(
                    {
                        "position": observation.ball_position_m,
                        "velocity": observation.ball_velocity_mps,
                    }
                )
            ),
            self_state_hash=observation.self_state.state_hash,
            teammate_state_hashes=tuple(
                sorted((state.agent_id, state.state_hash) for state in observation.teammate_states)
            ),
            opponent_state_hashes=tuple(
                sorted((state.agent_id, state.state_hash) for state in observation.opponent_states)
            ),
            possession_owner=owner,
            possession_agent_id=owner_id,
            self_to_ball_m=_xyz_tuple(ball - self_position),
            own_goal_to_ball_m=_xyz_tuple(ball - own_goal),
            opponent_goal_to_ball_m=_xyz_tuple(ball - opponent_goal),
            time_sec=observation.time_sec,
        )
        role_observations.append(role_observation)
        commitments.append(
            RoleIntentCommitment(
                agent_id=agent_id,
                self_model_hash=cell.self_model.self_model_hash,
                observation_hash=role_observation.observation_hash,
                policy_artifact_hash=cell.self_model.policy_artifact_hash,
                intent=decision.intent,
                skill=decision.skill,
                target_agent_id=decision.target_agent_id,
                confidence=decision.confidence,
                coordination_id=coordination_id,
            )
        )
    role_observation_by_id = {value.observer_agent_id: value for value in role_observations}
    handshakes: list[PassReceiveHandshake] = []
    for decision in decisions:
        if decision.intent is not TacticalIntent.PASS:
            continue
        receiver_id = decision.target_agent_id
        if receiver_id is None:
            raise ValueError("pass decision lacks its receiver")
        receiver_decision = decision_by_id[receiver_id]
        if receiver_decision.intent not in {
            TacticalIntent.RECEIVE,
            TacticalIntent.SUPPORT,
            TacticalIntent.RUN_IN_BEHIND,
        }:
            raise ValueError("independent receiver did not accept the pass")
        passer_observation = observation_by_id[decision.agent_id]
        receiver_observation = observation_by_id[receiver_id]
        distance = float(
            np.linalg.norm(
                np.asarray(decision.target_position_m[:2], dtype=np.float64)
                - np.asarray(passer_observation.ball_position_m[:2], dtype=np.float64)
            )
        )
        arrival = passer_observation.time_sec + max(0.20, distance / 1.30)
        handshakes.append(
            PassReceiveHandshake(
                handshake_id=f"pass.frame.{frame_index:06d}.{decision.agent_id}",
                coordination_id=coordination_id,
                passer_agent_id=decision.agent_id,
                receiver_agent_id=receiver_id,
                passer_self_model_hash=cell_by_id[decision.agent_id].self_model.self_model_hash,
                receiver_self_model_hash=cell_by_id[receiver_id].self_model.self_model_hash,
                passer_observation_hash=role_observation_by_id[decision.agent_id].observation_hash,
                receiver_observation_hash=role_observation_by_id[receiver_id].observation_hash,
                pass_target_m=decision.target_position_m,
                predicted_arrival_time_sec=arrival,
                receiver_ready_start_sec=max(receiver_observation.time_sec, arrival - 0.35),
                receiver_ready_end_sec=arrival + 0.45,
                predicted_ball_speed_mps=1.30,
                accepted_by_receiver=True,
            )
        )
    return TeamCoordinationFrame(
        roster=roster,
        observations=tuple(role_observations),
        intents=tuple(commitments),
        pass_receive_handshakes=tuple(handshakes),
    )


def build_agent_plasticity_lease(
    *,
    cells: tuple[RosclawSoccerAgentCell, ...],
    focal_agent_id: str,
    dataset_manifest_hash: str,
    scenario_contract_hash: str,
    maximum_optimizer_steps: int,
) -> PlasticityLease:
    """Grant bounded plasticity to one cell while freezing every teammate/opponent."""

    identities = tuple(cell.agent_id for cell in cells)
    if (
        len(cells) < 2
        or len(set(identities)) != len(identities)
        or focal_agent_id not in identities
    ):
        raise ValueError("plasticity lease requires one focal cell in a unique roster")
    return PlasticityLease(
        lease_id=f"soccer.lease.{focal_agent_id}",
        bindings=tuple(
            AgentPolicyBinding(
                agent_id=cell.agent_id,
                policy_hash=cell.growth_scope.champion_policy.version_hash,
                mode=(
                    AgentUpdateMode.PLASTIC
                    if cell.agent_id == focal_agent_id
                    else AgentUpdateMode.FROZEN
                ),
            )
            for cell in sorted(cells, key=lambda item: item.agent_id)
        ),
        dataset_manifest_hash=dataset_manifest_hash,
        scenario_contract_hash=scenario_contract_hash,
        maximum_optimizer_steps=maximum_optimizer_steps,
    )


def _skills_for_role(
    *,
    agent_id: str,
    role: MatchRole,
    generation: int,
    foundation_policy_hash: str,
) -> tuple[RoleSkillBinding, ...]:
    skills = {
        MatchRole.PLAYMAKER: (
            SoccerSkill.LEAD_PASS,
            SoccerSkill.FIRST_TOUCH,
            SoccerSkill.DRIBBLE,
            SoccerSkill.OFF_BALL_RUN,
            SoccerSkill.RECOVERY,
        ),
        MatchRole.FINISHER: (
            SoccerSkill.FIRST_TOUCH,
            SoccerSkill.OFF_BALL_RUN,
            SoccerSkill.FINISHING,
            SoccerSkill.WEAK_FOOT,
            SoccerSkill.RECOVERY,
        ),
        MatchRole.DEFENDER: (
            SoccerSkill.MARKING,
            SoccerSkill.INTERCEPTION,
            SoccerSkill.BLOCKING,
            SoccerSkill.RECOVERY,
        ),
        MatchRole.GOALKEEPER: (
            SoccerSkill.POSITIONING,
            SoccerSkill.SAVE,
            SoccerSkill.DISTRIBUTION,
            SoccerSkill.RECOVERY,
        ),
    }[role]
    return tuple(
        RoleSkillBinding(
            skill=skill,
            champion_artifact_hash=str(
                hash_json(
                    {
                        "agent_id": agent_id,
                        "skill": skill.value,
                        "generation": generation,
                        "foundation": foundation_policy_hash,
                    }
                )
            ),
            evidence_hash=str(
                hash_json({"agent_id": agent_id, "skill": skill.value, "evidence": generation})
            ),
            generation=generation,
            proficiency=0.50,
            training_priority=0.50,
        )
        for skill in skills
    )


def _possession_for(
    model: RoleSelfModel, possession_agent_id: str | None
) -> tuple[PossessionOwner, str | None]:
    if possession_agent_id is None:
        return PossessionOwner.LOOSE, None
    if possession_agent_id == model.agent_id:
        return PossessionOwner.SELF, model.agent_id
    if possession_agent_id in model.teammate_ids:
        return PossessionOwner.TEAMMATE, possession_agent_id
    if possession_agent_id in model.opponent_ids:
        return PossessionOwner.OPPONENT, possession_agent_id
    raise ValueError("possession identity is outside the agent social graph")


def _xyz_tuple(value: np.ndarray) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError("agent vector must be finite xyz")
    return (float(array[0]), float(array[1]), float(array[2]))


__all__ = [
    "AgentCellDecision",
    "AgentCellObservation",
    "AgentPhysicalState",
    "AgentTacticalProfile",
    "RosclawSoccerAgentCell",
    "build_agent_plasticity_lease",
    "build_independent_agent_cell",
    "build_team_coordination_frame",
]
