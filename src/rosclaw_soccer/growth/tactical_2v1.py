"""High-level 2v1 tactical evidence with matched counterfactual credit.

This module scores *what to do* at 5--10 Hz.  It deliberately carries hashes
for the frozen athlete foundation and skill bundle, so a tactical candidate
cannot silently obtain credit by changing low-level G1 motion.
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


class TacticalAction(StrEnum):
    PASS = "pass"
    SHOOT = "shoot"
    DRIBBLE_LEFT = "dribble_left"
    DRIBBLE_RIGHT = "dribble_right"
    HOLD = "hold"


@dataclass(frozen=True)
class TacticalRewardWeights:
    team: float = 1.0
    role: float = 0.25
    counterfactual: float = 0.50
    progress: float = 0.20
    schema_version: str = "rosclaw_soccer.tactical_reward_weights.v1"

    def __post_init__(self) -> None:
        values = (self.team, self.role, self.counterfactual, self.progress)
        if any(not math.isfinite(value) or value < 0.0 for value in values) or not any(values):
            raise ValueError("tactical reward weights must be finite and non-negative")

    @property
    def weights_hash(self) -> str:
        return str(hash_json(asdict(self)))

    def score(
        self,
        *,
        team_reward: float,
        role_reward: float,
        ablated_team_reward: float,
        possession_progress: float,
    ) -> float:
        values = (team_reward, role_reward, ablated_team_reward, possession_progress)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("tactical reward inputs must be finite")
        difference_reward = team_reward - ablated_team_reward
        return (
            self.team * team_reward
            + self.role * role_reward
            + self.counterfactual * difference_reward
            + self.progress * possession_progress
        )


@dataclass(frozen=True)
class TwoVsOneState:
    """Normalized tactical observation; no privileged phase label is allowed."""

    state_id: str
    seed: int
    self_state_hash: str
    world_state_hash: str
    scenario_hash: str
    environment_hash: str
    frozen_foundation_hash: str
    frozen_skill_bundle_hash: str
    frozen_defender_hash: str
    carrier_pressure: float
    teammate_lane_openness: float
    shot_lane_openness: float
    goal_progress: float
    teammate_progress: float
    schema_version: str = "rosclaw_soccer.two_vs_one_state.v1"

    def __post_init__(self) -> None:
        if not _IDENTIFIER.fullmatch(self.state_id):
            raise ValueError("2v1 state identity is invalid")
        if isinstance(self.seed, bool) or not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("2v1 state seed is invalid")
        for value in (
            self.self_state_hash,
            self.world_state_hash,
            self.scenario_hash,
            self.environment_hash,
            self.frozen_foundation_hash,
            self.frozen_skill_bundle_hash,
            self.frozen_defender_hash,
        ):
            if not _HASH.fullmatch(value):
                raise ValueError("2v1 state identities must be sha256 content hashes")
        observations = (
            self.carrier_pressure,
            self.teammate_lane_openness,
            self.shot_lane_openness,
            self.goal_progress,
            self.teammate_progress,
        )
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in observations):
            raise ValueError("2v1 tactical observations must be normalized to [0, 1]")

    @property
    def state_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass(frozen=True)
class MatchedTacticalRollout:
    """One selected-action rollout and its focal-player ablation."""

    state_hash: str
    policy_hash: str
    action: TacticalAction
    action_trace_hash: str
    trajectory_hash: str
    ablation_action_trace_hash: str
    ablated_trajectory_hash: str
    team_reward: float
    role_reward: float
    ablated_team_reward: float
    possession_progress: float
    safety_cost: float
    ablation_mode: str = "focal_agent_removed"
    matched_seed: bool = True
    matched_environment: bool = True
    matched_start_state: bool = True
    strict_replay: bool = True
    physics_authority: str = "CPU_MUJOCO"
    pixels_used_for_scoring: bool = False
    activation_ceiling: str = "SIM_ONLY"
    hardware_command_sent: bool = False
    schema_version: str = "rosclaw_soccer.matched_tactical_rollout.v1"

    def __post_init__(self) -> None:
        for value in (
            self.state_hash,
            self.policy_hash,
            self.action_trace_hash,
            self.trajectory_hash,
            self.ablation_action_trace_hash,
            self.ablated_trajectory_hash,
        ):
            if not _HASH.fullmatch(value):
                raise ValueError("tactical rollout identities must be sha256 content hashes")
        if self.trajectory_hash == self.ablated_trajectory_hash:
            raise ValueError("counterfactual rollout must be a distinct physical replay")
        if self.action_trace_hash == self.ablation_action_trace_hash:
            raise ValueError("counterfactual rollout must use an explicit ablation trace")
        values = (
            self.team_reward,
            self.role_reward,
            self.ablated_team_reward,
            self.possession_progress,
            self.safety_cost,
        )
        if any(not math.isfinite(value) for value in values) or self.safety_cost < 0.0:
            raise ValueError("tactical rollout metrics are invalid")
        if (
            not isinstance(self.action, TacticalAction)
            or not self.strict_replay
            or self.physics_authority != "CPU_MUJOCO"
            or self.pixels_used_for_scoring
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_command_sent
            or self.ablation_mode != "focal_agent_removed"
            or not self.matched_seed
            or not self.matched_environment
            or not self.matched_start_state
        ):
            raise ValueError("tactical rollout violates the evidence boundary")

    @property
    def difference_reward(self) -> float:
        return self.team_reward - self.ablated_team_reward

    def score(self, weights: TacticalRewardWeights) -> float:
        return weights.score(
            team_reward=self.team_reward,
            role_reward=self.role_reward,
            ablated_team_reward=self.ablated_team_reward,
            possession_progress=self.possession_progress,
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["action"] = self.action.value
        value["difference_reward"] = self.difference_reward
        return value


@dataclass(frozen=True)
class TwoVsOneDecisionEvidence:
    """Decision evidence bound to one state and all frozen low-level policies."""

    state: TwoVsOneState
    rollout: MatchedTacticalRollout
    weights: TacticalRewardWeights = TacticalRewardWeights()
    schema_version: str = "rosclaw_soccer.two_vs_one_decision_evidence.v1"

    def __post_init__(self) -> None:
        if self.rollout.state_hash != self.state.state_hash:
            raise ValueError("2v1 rollout belongs to another tactical state")

    @property
    def weighted_score(self) -> float:
        return self.rollout.score(self.weights)

    @property
    def promotion_eligible(self) -> bool:
        return self.rollout.safety_cost == 0.0 and self.rollout.difference_reward >= 0.0

    @property
    def evidence_hash(self) -> str:
        return str(hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "state": asdict(self.state),
            "state_hash": self.state.state_hash,
            "rollout": self.rollout.to_dict(),
            "weights": asdict(self.weights),
            "weights_hash": self.weights.weights_hash,
            "weighted_score": self.weighted_score,
            "promotion_eligible": self.promotion_eligible,
        }


__all__ = [
    "MatchedTacticalRollout",
    "TacticalAction",
    "TacticalRewardWeights",
    "TwoVsOneDecisionEvidence",
    "TwoVsOneState",
]
