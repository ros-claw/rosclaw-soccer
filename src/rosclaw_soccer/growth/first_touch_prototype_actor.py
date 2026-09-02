"""Failure-aware prototype actor for discontinuous First Touch contacts.

Linear interpolation is unsafe near a hybrid contact boundary: two successful
stance parameters can average into a body-first collision.  This actor keeps
successful CPU-MuJoCo contact modes intact and learns only a deterministic
context router between them.  Unknown contexts fall back to the frozen parent.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from rosclaw_soccer.sim.contracts import hash_json

if TYPE_CHECKING:
    from rosclaw_soccer.training.first_touch_physics import (
        FirstTouchCandidate,
        FirstTouchPhysicsScenario,
    )

_SCHEMA = "rosclaw_soccer.first_touch_failure_aware_prototype_actor.v1"
_FEATURE_COUNT = 4
_CANDIDATE_VALUE_COUNT = 19


def _hash(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
        raise ValueError(f"{label} must be a SHA-256 commitment")
    return value


def first_touch_candidate_vector(candidate: FirstTouchCandidate) -> tuple[float, ...]:
    """Serialize one already-validated candidate to a numeric prototype."""

    return (
        candidate.receiver_start_delay_sec,
        candidate.stance_offset_x,
        candidate.stance_offset_y,
        candidate.swing_amplitude,
        candidate.swing_speed_scale,
        candidate.com_shift_y,
        candidate.pelvis_yaw_offset,
        candidate.foot_yaw_offset,
        candidate.foot_pitch_offset,
        candidate.loft_synergy,
        *candidate.contact_residual_rad,
        float(candidate.contact_policy_frame),
        candidate.contact_lead_duration_sec,
        candidate.contact_trail_duration_sec,
    )


def _candidate_from_vector(
    values: tuple[float, ...],
    *,
    candidate_id: str,
    kick_foot: str,
) -> FirstTouchCandidate:
    from rosclaw_soccer.training.first_touch_physics import FirstTouchCandidate

    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (_CANDIDATE_VALUE_COUNT,) or not np.all(np.isfinite(vector)):
        raise ValueError("First Touch prototype candidate vector is invalid")
    frame = int(round(float(vector[16])))
    if abs(float(vector[16]) - frame) > 1.0e-12:
        raise ValueError("First Touch prototype contact frame must be integral")
    return FirstTouchCandidate(
        candidate_id=candidate_id,
        kick_foot=kick_foot,
        receiver_start_delay_sec=float(vector[0]),
        stance_offset_x=float(vector[1]),
        stance_offset_y=float(vector[2]),
        swing_amplitude=float(vector[3]),
        swing_speed_scale=float(vector[4]),
        com_shift_y=float(vector[5]),
        pelvis_yaw_offset=float(vector[6]),
        foot_yaw_offset=float(vector[7]),
        foot_pitch_offset=float(vector[8]),
        loft_synergy=float(vector[9]),
        contact_residual_rad=tuple(float(value) for value in vector[10:16]),
        contact_policy_frame=frame,
        contact_lead_duration_sec=float(vector[17]),
        contact_trail_duration_sec=float(vector[18]),
    )


@dataclass(frozen=True)
class FirstTouchPrototypeDecision:
    accepted: bool
    route: str
    confidence: float
    nearest_support_distance: float
    selected_prototype_report_hash: str | None
    candidate: FirstTouchCandidate | None


@dataclass(frozen=True)
class FirstTouchFailureAwarePrototypeActor:
    """Nearest-success router with explicit failed-contact memory."""

    body_hash: str
    kick_prior_hash: str
    implementation_hash: str
    training_snapshot_hash: str
    kick_foot: str
    feature_center: tuple[float, ...]
    feature_scale: tuple[float, ...]
    prototype_features: tuple[tuple[float, ...], ...]
    prototype_candidate_vectors: tuple[tuple[float, ...], ...]
    prototype_report_hashes: tuple[str, ...]
    prototype_scenario_hashes: tuple[str, ...]
    failed_report_hashes: tuple[str, ...]
    failed_scenario_hashes: tuple[str, ...]
    maximum_support_distance: float = 2.50
    activation_ceiling: str = "SIM_ONLY"
    promotion_authorized: bool = False
    hardware_authorized: bool = False
    online_hot_swap_allowed: bool = False
    schema_version: str = _SCHEMA

    def __post_init__(self) -> None:
        for value, label in (
            (self.body_hash, "body_hash"),
            (self.kick_prior_hash, "kick_prior_hash"),
            (self.implementation_hash, "implementation_hash"),
            (self.training_snapshot_hash, "training_snapshot_hash"),
        ):
            _hash(value, label)
        if self.schema_version != _SCHEMA or self.kick_foot not in {"left", "right"}:
            raise ValueError("First Touch prototype actor identity is invalid")
        center = np.asarray(self.feature_center, dtype=np.float64)
        scale = np.asarray(self.feature_scale, dtype=np.float64)
        features = np.asarray(self.prototype_features, dtype=np.float64)
        candidates = np.asarray(self.prototype_candidate_vectors, dtype=np.float64)
        count = len(self.prototype_report_hashes)
        if (
            center.shape != (_FEATURE_COUNT,)
            or scale.shape != (_FEATURE_COUNT,)
            or not np.all(np.isfinite(center))
            or not np.all(np.isfinite(scale))
            or np.any(scale <= 0.0)
            or count < 4
            or features.shape != (count, _FEATURE_COUNT)
            or candidates.shape != (count, _CANDIDATE_VALUE_COUNT)
            or not np.all(np.isfinite(features))
            or not np.all(np.isfinite(candidates))
            or len(self.prototype_scenario_hashes) != count
        ):
            raise ValueError("First Touch prototype support is invalid")
        for vector in self.prototype_candidate_vectors:
            _candidate_from_vector(
                vector, candidate_id="prototype.validation", kick_foot=self.kick_foot
            )
        commitments = (
            *self.prototype_report_hashes,
            *self.prototype_scenario_hashes,
            *self.failed_report_hashes,
            *self.failed_scenario_hashes,
        )
        if any(_hash(value, "prototype commitment") != value for value in commitments):
            raise ValueError("First Touch prototype commitment is invalid")
        if (
            len(set(self.prototype_report_hashes)) != count
            or len(set(self.prototype_scenario_hashes)) != count
            or len(self.failed_report_hashes) < 2
            or len(self.failed_report_hashes) != len(self.failed_scenario_hashes)
        ):
            raise ValueError("First Touch prototype success/failure memory is insufficient")
        if set(self.prototype_report_hashes) & set(self.failed_report_hashes):
            raise ValueError("First Touch prototype success/failure memory overlaps")
        if not math.isfinite(self.maximum_support_distance) or not (
            0.25 <= self.maximum_support_distance <= 4.0
        ):
            raise ValueError("First Touch prototype support radius is invalid")
        if (
            self.activation_ceiling != "SIM_ONLY"
            or self.promotion_authorized
            or self.hardware_authorized
            or self.online_hot_swap_allowed
        ):
            raise ValueError("First Touch prototype actor must remain frozen and SIM_ONLY")

    @property
    def actor_hash(self) -> str:
        return str(hash_json(self.to_dict(include_hash=False)))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value = {
            **asdict(self),
            "feature_names": [
                "incoming_speed_mps",
                "incoming_lateral_m",
                "target_direction_deg",
                "target_outgoing_speed_mps",
            ],
            "candidate_value_names": [
                "receiver_start_delay_sec",
                "stance_offset_x",
                "stance_offset_y",
                "swing_amplitude",
                "swing_speed_scale",
                "com_shift_y",
                "pelvis_yaw_offset",
                "foot_yaw_offset",
                "foot_pitch_offset",
                "loft_synergy",
                "contact_hip_pitch_rad",
                "contact_hip_roll_rad",
                "contact_hip_yaw_rad",
                "contact_knee_rad",
                "contact_ankle_pitch_rad",
                "contact_ankle_roll_rad",
                "contact_policy_frame",
                "contact_lead_duration_sec",
                "contact_trail_duration_sec",
            ],
            "algorithm": "failure_aware_nearest_success_contact_mode_routing",
            "pixels_used_for_training": False,
            "retention_evidence_used_for_training": False,
            "direct_joint_torque_output": False,
            "stability_plasticity_contract": {
                "stability": "never interpolate across a hybrid contact boundary",
                "plasticity": "append verified successful modes while retaining failures",
            },
        }
        if include_hash:
            value["actor_hash"] = self.actor_hash
        return value

    def decide(
        self,
        scenario: FirstTouchPhysicsScenario,
        *,
        candidate_id: str,
    ) -> FirstTouchPrototypeDecision:
        raw = np.asarray(
            (
                scenario.incoming_speed_mps,
                scenario.incoming_lateral_m,
                scenario.target_direction_deg,
                scenario.target_outgoing_speed_mps,
            ),
            dtype=np.float64,
        )
        center = np.asarray(self.feature_center, dtype=np.float64)
        scale = np.asarray(self.feature_scale, dtype=np.float64)
        normalized = (raw - center) / scale
        support = (np.asarray(self.prototype_features, dtype=np.float64) - center) / scale
        distances = np.linalg.norm(support - normalized, axis=1)
        order = sorted(
            range(distances.size),
            key=lambda index: (float(distances[index]), self.prototype_report_hashes[index]),
        )
        selected = order[0]
        distance = float(distances[selected])
        if distance > self.maximum_support_distance:
            return FirstTouchPrototypeDecision(
                accepted=False,
                route="FROZEN_PARENT_OOD_FALLBACK",
                confidence=0.0,
                nearest_support_distance=distance,
                selected_prototype_report_hash=None,
                candidate=None,
            )
        confidence = max(0.0, min(1.0, 1.0 - distance / self.maximum_support_distance))
        return FirstTouchPrototypeDecision(
            accepted=True,
            route="VERIFIED_CONTACT_MODE_PROTOTYPE",
            confidence=confidence,
            nearest_support_distance=distance,
            selected_prototype_report_hash=self.prototype_report_hashes[selected],
            candidate=_candidate_from_vector(
                self.prototype_candidate_vectors[selected],
                candidate_id=candidate_id,
                kick_foot=self.kick_foot,
            ),
        )


def save_first_touch_prototype_actor(
    actor: FirstTouchFailureAwarePrototypeActor,
    path: Path,
) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(actor.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def load_first_touch_prototype_actor(path: Path) -> FirstTouchFailureAwarePrototypeActor:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("First Touch prototype artifact must be a mapping")
    claimed_hash = payload.pop("actor_hash", None)
    for key in (
        "feature_names",
        "candidate_value_names",
        "algorithm",
        "pixels_used_for_training",
        "retention_evidence_used_for_training",
        "direct_joint_torque_output",
        "stability_plasticity_contract",
    ):
        payload.pop(key, None)
    for key in (
        "feature_center",
        "feature_scale",
        "prototype_report_hashes",
        "prototype_scenario_hashes",
        "failed_report_hashes",
        "failed_scenario_hashes",
    ):
        payload[key] = tuple(payload[key])
    for key in ("prototype_features", "prototype_candidate_vectors"):
        payload[key] = tuple(tuple(row) for row in payload[key])
    actor = FirstTouchFailureAwarePrototypeActor(**payload)
    if claimed_hash != actor.actor_hash:
        raise ValueError("First Touch prototype actor hash does not match its payload")
    return actor


__all__ = [
    "FirstTouchFailureAwarePrototypeActor",
    "FirstTouchPrototypeDecision",
    "first_touch_candidate_vector",
    "load_first_touch_prototype_actor",
    "save_first_touch_prototype_actor",
]
