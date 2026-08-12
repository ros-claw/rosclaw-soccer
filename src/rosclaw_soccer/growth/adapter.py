"""ROSClaw Growth adapter for football experience semantics.

The Core entry-point registry is intentionally imported only when an
experience is normalized.  This keeps the downstream package importable while
the stacked Core PRs are reviewed and gives older Core installations an honest
feature boundary instead of a partial fallback contract.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rosclaw.growth.experience import ExperienceSegment, FailureSignature


@dataclass(frozen=True)
class SoccerGrowthAdapter:
    """Translate a signed football envelope into task-neutral Core records."""

    adapter_id: str = "soccer.growth"
    skill_ids: tuple[str, ...] = (
        "soccer.first_touch",
        "soccer.free_kick",
        "soccer.passing",
        "soccer.shooting",
        "soccer.goalkeeping",
    )

    def normalize_experience(self, payload: Mapping[str, Any]) -> ExperienceSegment:
        """Parse one strict adapter envelope without obtaining execution authority."""

        from rosclaw.growth.contracts import EvidenceLevel
        from rosclaw.growth.experience import (
            ActionTraceCommitment,
            DerivedExperienceLineage,
            ExperienceSegment,
            PhysicalAdvantageLabel,
        )

        value = _mapping(payload, "payload")
        skill_id = _string(value, "skill_id")
        if skill_id not in self.skill_ids:
            raise ValueError("soccer experience uses an undeclared skill_id")
        lineage_value = _mapping(value.get("lineage"), "lineage")
        action_value = _mapping(value.get("action"), "action")
        reward = _metric_mapping(value.get("reward_vector"), "reward_vector")
        cost = _metric_mapping(value.get("cost_vector"), "cost_vector", non_negative=True)
        if any(not key.startswith("soccer.") for key in (*reward, *cost)):
            raise ValueError("soccer reward and cost metrics must use the soccer namespace")
        advantage = PhysicalAdvantageLabel(_string(value, "advantage_label"))
        signature = self._diagnose_values(
            skill_id=skill_id,
            reward=reward,
            cost=cost,
            advantage=advantage,
        )
        segment = ExperienceSegment(
            segment_id=_string(value, "segment_id"),
            episode_id=_string(value, "episode_id"),
            skill_id=skill_id,
            phase=_string(value, "phase"),
            start_time_sec=_number(value, "start_time_sec"),
            end_time_sec=_number(value, "end_time_sec"),
            body_hash=_string(value, "body_hash"),
            regime_hash=_string(value, "regime_hash"),
            source_evidence_level=EvidenceLevel(_string(value, "source_evidence_level")),
            lineage=DerivedExperienceLineage(
                source_artifact_hash=_string(lineage_value, "source_artifact_hash"),
                source_event_hashes=_string_tuple(lineage_value, "source_event_hashes"),
                transform_hash=_string(lineage_value, "transform_hash"),
                clock_id=_string(lineage_value, "clock_id"),
                maximum_skew_sec=_number(lineage_value, "maximum_skew_sec"),
                observed_skew_sec=_number(lineage_value, "observed_skew_sec"),
                synchronization_receipt_hash=_optional_string(
                    lineage_value,
                    "synchronization_receipt_hash",
                ),
            ),
            base_policy_version=_string(value, "base_policy_version"),
            residual_policy_version=_optional_string(value, "residual_policy_version"),
            state_start_hash=_string(value, "state_start_hash"),
            observation_sequence_hash=_string(value, "observation_sequence_hash"),
            self_state_hash=_string(value, "self_state_hash"),
            world_state_hash=_string(value, "world_state_hash"),
            action=ActionTraceCommitment(
                commanded_action_hash=_string(action_value, "commanded_action_hash"),
                executed_action_hash=_string(action_value, "executed_action_hash"),
                safety_projected_action_hash=_string(
                    action_value,
                    "safety_projected_action_hash",
                ),
                policy_version=_string(action_value, "policy_version"),
                controller_hash=_string(action_value, "controller_hash"),
                projection_applied=_boolean(action_value, "projection_applied"),
            ),
            reward_vector=reward,
            cost_vector=cost,
            terminal_state_hash=_string(value, "terminal_state_hash"),
            advantage_label=advantage,
            label_confidence=_number(value, "label_confidence"),
            failure_signature=signature,
        )
        return segment

    def diagnose(self, segment: ExperienceSegment) -> FailureSignature | None:
        """Return football failure semantics without changing the Core record."""

        if segment.skill_id not in self.skill_ids:
            raise ValueError("segment is outside the soccer adapter skill set")
        return self._diagnose_values(
            skill_id=segment.skill_id,
            reward=segment.reward_vector,
            cost=segment.cost_vector,
            advantage=segment.advantage_label,
        )

    def _diagnose_values(
        self,
        *,
        skill_id: str,
        reward: Mapping[str, float],
        cost: Mapping[str, float],
        advantage: Any,
    ) -> FailureSignature | None:
        from rosclaw.growth.experience import FailureSignature, PhysicalAdvantageLabel

        negative = advantage in {
            PhysicalAdvantageLabel.ADVANTAGE_NEGATIVE,
            PhysicalAdvantageLabel.UNSAFE_NEGATIVE,
        }
        if not negative:
            return None
        if any(value > 0.0 for value in cost.values()):
            primary = "soccer.post_contact_instability"
            contributors = tuple(sorted(key for key, value in cost.items() if value > 0.0))
            learners = ("residual_sac", "system_identification")
        elif reward.get("soccer.contact", 0.0) <= 0.0:
            primary = "soccer.contact_miss"
            contributors = ("soccer.contact",)
            learners = ("motion_tracking", "ilc")
        else:
            primary = "soccer.target_error"
            contributors = tuple(sorted(key for key, value in reward.items() if value < 0.0))
            if not contributors:
                contributors = ("soccer.precision",)
            learners = ("iql", "world_model_observer")
        return FailureSignature(
            primary_type=primary,
            contributors=contributors,
            confidence=1.0,
            affected_capability_ids=(skill_id,),
            reusable_evidence_ids=(),
            recommended_learner_ids=learners,
        )


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _string(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise ValueError(f"{key} must be a non-empty string")
    return item


def _optional_string(value: Mapping[str, Any], key: str) -> str | None:
    item = value.get(key)
    if item is None:
        return None
    if not isinstance(item, str) or not item:
        raise ValueError(f"{key} must be null or a non-empty string")
    return item


def _string_tuple(value: Mapping[str, Any], key: str) -> tuple[str, ...]:
    item = value.get(key)
    if not isinstance(item, (list, tuple)) or any(
        not isinstance(part, str) or not part for part in item
    ):
        raise ValueError(f"{key} must be a sequence of non-empty strings")
    return tuple(item)


def _number(value: Mapping[str, Any], key: str) -> float:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
        raise ValueError(f"{key} must be a finite number")
    return float(item)


def _boolean(value: Mapping[str, Any], key: str) -> bool:
    item = value.get(key)
    if not isinstance(item, bool):
        raise ValueError(f"{key} must be boolean")
    return item


def _metric_mapping(
    value: object,
    label: str,
    *,
    non_negative: bool = False,
) -> dict[str, float]:
    mapping = _mapping(value, label)
    result: dict[str, float] = {}
    for key, raw in mapping.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{label} keys must be non-empty strings")
        number = _number({key: raw}, key)
        if non_negative and number < 0.0:
            raise ValueError(f"{label} values must be non-negative")
        result[key] = number
    if not result:
        raise ValueError(f"{label} must not be empty")
    return result


SOCCER_GROWTH_ADAPTER = SoccerGrowthAdapter()

__all__ = ["SOCCER_GROWTH_ADAPTER", "SoccerGrowthAdapter"]
