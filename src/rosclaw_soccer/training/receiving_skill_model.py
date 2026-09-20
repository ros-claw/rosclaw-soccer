"""Measured-neighbour receiving capability pilot for the tactical research line.

This is not a motor learner or calibrated transition model. It cannot provide
unmeasured duration/readiness. Skill execution and promotion remain outside.
"""

import re
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.receiving_context import ReceivingContext

_HASH = re.compile(r"sha256:[0-9a-f]{64}")
_ROLE = re.compile(r"(?:red|blue)\.(?:defender|finisher|goalkeeper|playmaker)")
_FIELDS = (
    "ball_forward_m",
    "ball_lateral_m",
    "ball_height_m",
    "ball_speed_mps",
    "relative_forward_velocity_mps",
    "relative_lateral_velocity_mps",
    "player_speed_mps",
)
# Explicit observation neighbourhood, not hand-entered outcome probabilities.
_SCALE = np.array([0.5, 0.12, 0.05, 0.5, 0.5, 0.5, 0.5])


@dataclass(frozen=True)
class ReceivingSkillSample:
    course_hash: str
    evidence_hash: str
    policy_hash: str
    physics_hash: str
    agent_id: str
    context: ReceivingContext
    safe_capture: bool

    def __post_init__(self) -> None:
        if any(
            type(v) is not str or _HASH.fullmatch(v) is None
            for v in (self.course_hash, self.evidence_hash, self.policy_hash, self.physics_hash)
        ):
            raise ValueError("content-bound course, evidence, policy and physics required")
        if type(self.agent_id) is not str or _ROLE.fullmatch(self.agent_id) is None:
            raise ValueError("named receiving agent required")
        if not isinstance(self.context, ReceivingContext) or type(self.safe_capture) is not bool:
            raise ValueError("typed observed context and measured success required")
        self.context.__post_init__()


@dataclass(frozen=True)
class ReceivingSkillModel:
    """Fixed 16-neighbour pilot; independent validation must assess calibration.

    Every query is policy/body-physics and agent-specific. Neighbours must all
    fit the declared max-coordinate radius. Missing coverage returns UNKNOWN,
    not zero skill, and no query can create a full tactical transition.
    """

    samples: tuple[ReceivingSkillSample, ...]

    def __post_init__(self) -> None:
        if type(self.samples) is not tuple or not 16 <= len(self.samples) <= 10000:
            raise ValueError("bounded immutable measured sample bank required")
        for sample in self.samples:
            if not isinstance(sample, ReceivingSkillSample):
                raise ValueError("typed measured skill samples required")
            sample.__post_init__()
        for name in ("course_hash", "evidence_hash"):
            if len({getattr(s, name) for s in self.samples}) != len(self.samples):
                raise ValueError("course repeats and replay receipts are not independent samples")
        if len({(s.policy_hash, s.physics_hash, s.context.time_sec) for s in self.samples}) != 1:
            raise ValueError("one policy, physics and fixed observation time per model")

    @property
    def model_hash(self) -> str:
        return str(
            hash_json(
                dict(
                    schema="soccer.receiving_measured_neighbour_pilot.v1",
                    samples=[asdict(s) for s in self.samples],
                    fields=_FIELDS,
                    scales=_SCALE.tolist(),
                    neighbours=16,
                    max_coordinate_radius=1.0,
                )
            )
        )

    def query(
        self,
        context: ReceivingContext,
        *,
        agent_id: str,
        policy_hash: str,
        physics_hash: str,
    ) -> dict[str, Any]:
        if not isinstance(context, ReceivingContext):
            raise ValueError("typed causal observation required")
        context.__post_init__()
        if type(agent_id) is not str or _ROLE.fullmatch(agent_id) is None:
            raise ValueError("explicit named agent required")
        for value in (policy_hash, physics_hash):
            if type(value) is not str or _HASH.fullmatch(value) is None:
                raise ValueError("explicit query policy and physics required")
        result: dict[str, Any] = dict(
            model_hash=self.model_hash,
            status="UNKNOWN",
            observed_fraction=None,
            samples=0,
            evidence_hashes=[],
            skill_duration_sec=None,
            readiness_after=None,
            calibrated_probability_model=False,
            transition_model_ready=False,
            promotion_authorized=False,
        )
        reference = self.samples[0]
        if (policy_hash, physics_hash, context.time_sec) != (
            reference.policy_hash,
            reference.physics_hash,
            reference.context.time_sec,
        ):
            return {**result, "reason": "policy_physics_or_observation_time_mismatch"}
        values = np.array([getattr(context, name) for name in _FIELDS])
        neighbours = []
        for sample in self.samples:
            if sample.agent_id != agent_id:
                continue
            difference = (np.array([getattr(sample.context, n) for n in _FIELDS]) - values) / _SCALE
            radius = float(np.max(abs(difference)))
            if radius <= 1:
                neighbours.append(
                    (float(np.dot(difference, difference)), sample.course_hash, sample)
                )
        neighbours.sort(key=lambda row: (row[0], row[1]))
        if len(neighbours) < 16:
            return {
                **result,
                "reason": "insufficient_measured_neighbourhood",
                "samples": len(neighbours),
            }
        chosen = [row[2] for row in neighbours[:16]]
        return {
            **result,
            "status": "OBSERVED",
            "reason": "empirical_neighbour_frequency_not_deployment_probability",
            "observed_fraction": sum(s.safe_capture for s in chosen) / 16,
            "samples": 16,
            "evidence_hashes": [s.evidence_hash for s in chosen],
        }
