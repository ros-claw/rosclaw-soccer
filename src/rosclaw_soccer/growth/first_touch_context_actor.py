"""Content-bound contextual residual actor for moving-ball First Touch.

The actor predicts bounded parameters around the qualified whole-body kick
prior.  It is deliberately low dimensional: MuJoCo evidence, not a rendered
video, teaches the residual; an out-of-distribution context is rejected
instead of extrapolated; and the artifact can never authorize hardware or a
runtime hot swap.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.sim.contracts import hash_json

if TYPE_CHECKING:
    from rosclaw_soccer.training.first_touch_physics import (
        FirstTouchCandidate,
        FirstTouchPhysicsScenario,
    )

_SCHEMA = "rosclaw_soccer.first_touch_context_residual_actor.v1"
_FEATURE_NAMES = (
    "incoming_speed_mps",
    "incoming_lateral_m",
    "target_direction_deg",
    "target_outgoing_speed_mps",
)
_OUTPUT_NAMES = (
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
)
_OUTPUT_BOUNDS = np.asarray(
    (
        (0.0, 1.0),
        (-0.12, 0.12),
        (-0.12, 0.12),
        (0.40, 1.15),
        (0.80, 1.50),
        (-0.08, 0.08),
        (-0.20, 0.20),
        (-0.12, 0.12),
        (-0.18, 0.18),
        (0.0, 0.30),
    ),
    dtype=np.float64,
)


def _sha256(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
        raise ValueError(f"{label} must be a SHA-256 commitment")
    return value


def _finite_tuple(
    value: tuple[float, ...],
    *,
    length: int,
    label: str,
) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain {length} finite values")
    return array


@dataclass(frozen=True)
class FirstTouchActorDecision:
    """Fail-closed result of one contextual policy query."""

    accepted: bool
    route: str
    confidence: float
    nearest_support_distance: float
    candidate: FirstTouchCandidate | None
    safety_projection_applied: bool


@dataclass(frozen=True)
class FirstTouchContextResidualActor:
    """A ridge-distilled parameter actor with an immutable support envelope."""

    body_hash: str
    kick_prior_hash: str
    implementation_hash: str
    training_snapshot_hash: str
    source_report_hashes: tuple[str, ...]
    training_scenario_hashes: tuple[str, ...]
    kick_foot: str
    feature_center: tuple[float, ...]
    feature_scale: tuple[float, ...]
    support_minimum: tuple[float, ...]
    support_maximum: tuple[float, ...]
    normalized_support_points: tuple[tuple[float, ...], ...]
    coefficient_matrix: tuple[tuple[float, ...], ...]
    successful_teacher_count: int
    rejected_teacher_count: int
    fit_rmse: float
    ridge_regularization: float
    maximum_support_distance: float = 2.25
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
            _sha256(value, label)
        if self.schema_version != _SCHEMA:
            raise ValueError("First Touch context actor schema is unsupported")
        if self.kick_foot not in {"left", "right"}:
            raise ValueError("First Touch context actor kick foot is invalid")
        if len(self.source_report_hashes) < 6 or any(
            _sha256(value, "source_report_hash") != value for value in self.source_report_hashes
        ):
            raise ValueError("First Touch actor requires six bound teacher reports")
        if len(set(self.source_report_hashes)) != len(self.source_report_hashes):
            raise ValueError("First Touch actor report commitments must be unique")
        if len(self.training_scenario_hashes) < 4 or any(
            _sha256(value, "training_scenario_hash") != value
            for value in self.training_scenario_hashes
        ):
            raise ValueError("First Touch actor requires four training contexts")
        center = _finite_tuple(
            self.feature_center,
            length=len(_FEATURE_NAMES),
            label="feature_center",
        )
        scale = _finite_tuple(
            self.feature_scale,
            length=len(_FEATURE_NAMES),
            label="feature_scale",
        )
        minimum = _finite_tuple(
            self.support_minimum,
            length=len(_FEATURE_NAMES),
            label="support_minimum",
        )
        maximum = _finite_tuple(
            self.support_maximum,
            length=len(_FEATURE_NAMES),
            label="support_maximum",
        )
        if (
            np.any(scale <= 0.0)
            or np.any(minimum > maximum)
            or not np.all((center >= minimum) & (center <= maximum))
        ):
            raise ValueError("First Touch actor feature envelope is invalid")
        support = np.asarray(self.normalized_support_points, dtype=np.float64)
        if (
            support.ndim != 2
            or support.shape[0] != self.successful_teacher_count
            or support.shape[1] != len(_FEATURE_NAMES)
            or not np.all(np.isfinite(support))
        ):
            raise ValueError("First Touch actor support points are invalid")
        coefficients = np.asarray(self.coefficient_matrix, dtype=np.float64)
        if coefficients.shape != (len(_OUTPUT_NAMES), len(_FEATURE_NAMES) + 1) or not np.all(
            np.isfinite(coefficients)
        ):
            raise ValueError("First Touch actor coefficient matrix is invalid")
        if self.successful_teacher_count < 4 or self.rejected_teacher_count < 2:
            raise ValueError("First Touch actor needs successful and rejected support")
        if not math.isfinite(self.fit_rmse) or not 0.0 <= self.fit_rmse <= 0.50:
            raise ValueError("First Touch actor fit error is invalid")
        if not math.isfinite(self.ridge_regularization) or not (
            0.0 < self.ridge_regularization <= 10.0
        ):
            raise ValueError("First Touch actor ridge regularization is invalid")
        if not math.isfinite(self.maximum_support_distance) or not (
            0.25 <= self.maximum_support_distance <= 4.0
        ):
            raise ValueError("First Touch actor support radius is invalid")
        if (
            self.activation_ceiling != "SIM_ONLY"
            or self.promotion_authorized
            or self.hardware_authorized
            or self.online_hot_swap_allowed
        ):
            raise ValueError("First Touch actor must remain frozen and SIM_ONLY")

    @property
    def actor_hash(self) -> str:
        return str(hash_json(self.to_dict(include_hash=False)))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value = {
            **asdict(self),
            "source_report_hashes": list(self.source_report_hashes),
            "training_scenario_hashes": list(self.training_scenario_hashes),
            "feature_names": list(_FEATURE_NAMES),
            "output_names": list(_OUTPUT_NAMES),
            "algorithm": "bounded_ridge_teacher_distillation",
            "direct_joint_torque_output": False,
            "pixels_used_for_training": False,
            "retention_evidence_used_for_training": False,
            "stability_plasticity_contract": {
                "stability": "immutable successful support plus OOD rejection and clipping",
                "plasticity": "ridge fit over content-bound CPU MuJoCo teacher outcomes",
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
    ) -> FirstTouchActorDecision:
        """Predict a bounded candidate only inside measured teacher support."""

        from rosclaw_soccer.training.first_touch_physics import FirstTouchCandidate

        raw = np.asarray(
            (
                scenario.incoming_speed_mps,
                scenario.incoming_lateral_m,
                scenario.target_direction_deg,
                scenario.target_outgoing_speed_mps,
            ),
            dtype=np.float64,
        )
        minimum = np.asarray(self.support_minimum, dtype=np.float64)
        maximum = np.asarray(self.support_maximum, dtype=np.float64)
        center = np.asarray(self.feature_center, dtype=np.float64)
        scale = np.asarray(self.feature_scale, dtype=np.float64)
        normalized = (raw - center) / scale
        support = np.asarray(self.normalized_support_points, dtype=np.float64)
        nearest = float(np.linalg.norm(support - normalized, axis=1).min())
        inside_box = bool(np.all((raw >= minimum - 1.0e-12) & (raw <= maximum + 1.0e-12)))
        if not inside_box or nearest > self.maximum_support_distance:
            return FirstTouchActorDecision(
                accepted=False,
                route="FROZEN_PARENT_OOD_FALLBACK",
                confidence=0.0,
                nearest_support_distance=nearest,
                candidate=None,
                safety_projection_applied=False,
            )
        design = np.concatenate((np.ones(1, dtype=np.float64), normalized))
        predicted = np.asarray(self.coefficient_matrix, dtype=np.float64) @ design
        projected = np.clip(predicted, _OUTPUT_BOUNDS[:, 0], _OUTPUT_BOUNDS[:, 1])
        projection_applied = not np.allclose(predicted, projected, rtol=0.0, atol=1.0e-12)
        confidence = max(0.0, min(1.0, 1.0 - nearest / self.maximum_support_distance))
        candidate = FirstTouchCandidate(
            candidate_id=candidate_id,
            kick_foot=self.kick_foot,
            receiver_start_delay_sec=float(projected[0]),
            stance_offset_x=float(projected[1]),
            stance_offset_y=float(projected[2]),
            swing_amplitude=float(projected[3]),
            swing_speed_scale=float(projected[4]),
            com_shift_y=float(projected[5]),
            pelvis_yaw_offset=float(projected[6]),
            foot_yaw_offset=float(projected[7]),
            foot_pitch_offset=float(projected[8]),
            loft_synergy=float(projected[9]),
        )
        return FirstTouchActorDecision(
            accepted=True,
            route="CONTEXTUAL_RESIDUAL_CANDIDATE",
            confidence=confidence,
            nearest_support_distance=nearest,
            candidate=candidate,
            safety_projection_applied=projection_applied,
        )


def save_first_touch_context_actor(
    actor: FirstTouchContextResidualActor,
    path: Path,
) -> None:
    """Atomically persist one self-hashing actor artifact."""

    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(actor.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def load_first_touch_context_actor(path: Path) -> FirstTouchContextResidualActor:
    """Load and integrity-check a contextual actor."""

    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("First Touch actor artifact must be a mapping")
    claimed_hash = payload.pop("actor_hash", None)
    for derived in (
        "feature_names",
        "output_names",
        "algorithm",
        "direct_joint_torque_output",
        "pixels_used_for_training",
        "retention_evidence_used_for_training",
        "stability_plasticity_contract",
    ):
        payload.pop(derived, None)
    actor = FirstTouchContextResidualActor(
        **{
            **payload,
            "source_report_hashes": tuple(payload["source_report_hashes"]),
            "training_scenario_hashes": tuple(payload["training_scenario_hashes"]),
            "feature_center": tuple(payload["feature_center"]),
            "feature_scale": tuple(payload["feature_scale"]),
            "support_minimum": tuple(payload["support_minimum"]),
            "support_maximum": tuple(payload["support_maximum"]),
            "normalized_support_points": tuple(
                tuple(row) for row in payload["normalized_support_points"]
            ),
            "coefficient_matrix": tuple(tuple(row) for row in payload["coefficient_matrix"]),
        }
    )
    if claimed_hash != actor.actor_hash:
        raise ValueError("First Touch actor hash does not match its payload")
    return actor


__all__ = [
    "FirstTouchActorDecision",
    "FirstTouchContextResidualActor",
    "load_first_touch_context_actor",
    "save_first_touch_context_actor",
]
