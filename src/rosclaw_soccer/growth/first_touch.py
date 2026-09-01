"""First-touch measurement, failure attribution, and Dream routing.

The evaluator is deterministic and simulator-independent.  A backend must
provide physics telemetry; rendered pixels never determine success.  Failure
Dreams use ROSClaw Core's task-neutral contract and remain training-only.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from rosclaw_soccer.sim.contracts import hash_json

if TYPE_CHECKING:
    from rosclaw.continual.failure_curriculum import FailureConditionedDream

_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.:-]{0,127}$")


class FirstTouchFailure(StrEnum):
    TOUCH_TOO_HARD = "soccer.touch_too_hard"
    TOUCH_TOO_SOFT = "soccer.touch_too_soft"
    TOUCH_WRONG_DIRECTION = "soccer.touch_wrong_direction"
    WRONG_FOOT = "soccer.wrong_foot"
    LOST_BALANCE = "soccer.lost_balance"
    TOO_SLOW_TO_NEXT_ACTION = "soccer.too_slow_to_next_action"


@dataclass(frozen=True)
class FirstTouchGateConfig:
    minimum_incoming_speed_mps: float = 0.5
    maximum_incoming_speed_mps: float = 6.0
    minimum_outgoing_speed_mps: float = 0.15
    maximum_outgoing_speed_mps: float = 2.5
    maximum_target_error_m: float = 0.35
    maximum_direction_error_deg: float = 20.0
    maximum_next_action_latency_sec: float = 0.70
    minimum_pelvis_height_m: float = 0.62
    maximum_torso_tilt_deg: float = 28.0
    maximum_root_speed_mps: float = 1.8
    schema_version: str = "rosclaw_soccer.first_touch_gate_config.v1"

    def __post_init__(self) -> None:
        values = tuple(value for key, value in asdict(self).items() if key != "schema_version")
        if (
            any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in values)
            or self.minimum_incoming_speed_mps >= self.maximum_incoming_speed_mps
            or self.minimum_outgoing_speed_mps >= self.maximum_outgoing_speed_mps
            or self.maximum_direction_error_deg > 90.0
            or self.maximum_torso_tilt_deg > 60.0
        ):
            raise ValueError("first-touch gate config is invalid")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass(frozen=True)
class FirstTouchMeasurement:
    """One causally measured incoming-ball → touch → next-action transition."""

    sample_id: str
    actor_id: str
    source_snapshot_hash: str
    body_hash: str
    scenario_hash: str
    incoming_speed_mps: float
    outgoing_speed_mps: float
    target_error_m: float
    direction_error_deg: float
    next_action_latency_sec: float
    minimum_pelvis_height_m: float
    maximum_torso_tilt_deg: float
    maximum_root_speed_mps: float
    contact_detected: bool
    selected_foot: str
    required_foot: str
    strict_replay: bool = True
    physics_derived: bool = True
    pixels_used_for_scoring: bool = False
    activation_ceiling: str = "SIM_ONLY"
    hardware_command_sent: bool = False
    schema_version: str = "rosclaw_soccer.first_touch_measurement.v1"

    def __post_init__(self) -> None:
        for value in (self.sample_id, self.actor_id):
            if not _IDENTIFIER.fullmatch(value):
                raise ValueError("first-touch sample identity is invalid")
        for value in (self.source_snapshot_hash, self.body_hash, self.scenario_hash):
            if not _HASH.fullmatch(value):
                raise ValueError("first-touch identities must be sha256 content hashes")
        metrics = (
            self.incoming_speed_mps,
            self.outgoing_speed_mps,
            self.target_error_m,
            self.direction_error_deg,
            self.next_action_latency_sec,
            self.minimum_pelvis_height_m,
            self.maximum_torso_tilt_deg,
            self.maximum_root_speed_mps,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in metrics):
            raise ValueError("first-touch telemetry must be finite and non-negative")
        if self.selected_foot not in {"left", "right"} or self.required_foot not in {
            "left",
            "right",
        }:
            raise ValueError("first-touch foot selection is invalid")
        if (
            not self.strict_replay
            or not self.physics_derived
            or self.pixels_used_for_scoring
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_command_sent
        ):
            raise ValueError("first-touch measurement violates the evidence boundary")

    @property
    def measurement_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass(frozen=True)
class FirstTouchEvaluation:
    measurement_hash: str
    gate_config_hash: str
    passed: bool
    primary_failure: FirstTouchFailure | None
    all_failures: tuple[FirstTouchFailure, ...]
    controlled_first_touch: bool
    successor_action_ready: bool
    safety_passed: bool
    schema_version: str = "rosclaw_soccer.first_touch_evaluation.v1"

    def __post_init__(self) -> None:
        if not _HASH.fullmatch(self.measurement_hash) or not _HASH.fullmatch(self.gate_config_hash):
            raise ValueError("first-touch evaluation is not content bound")
        if self.passed != (not self.all_failures):
            raise ValueError("first-touch pass flag disagrees with failures")
        if self.passed != (
            self.controlled_first_touch and self.successor_action_ready and self.safety_passed
        ):
            raise ValueError("first-touch pass flag disagrees with its three gates")
        if (self.primary_failure is None) != self.passed:
            raise ValueError("first-touch primary failure is inconsistent")
        if self.primary_failure is not None and (
            not self.all_failures or self.primary_failure is not self.all_failures[0]
        ):
            raise ValueError("first-touch primary failure must lead the failure tuple")

    @property
    def evaluation_hash(self) -> str:
        return str(hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "measurement_hash": self.measurement_hash,
            "gate_config_hash": self.gate_config_hash,
            "passed": self.passed,
            "primary_failure": (
                None if self.primary_failure is None else self.primary_failure.value
            ),
            "all_failures": [item.value for item in self.all_failures],
            "controlled_first_touch": self.controlled_first_touch,
            "successor_action_ready": self.successor_action_ready,
            "safety_passed": self.safety_passed,
        }


def evaluate_first_touch(
    measurement: FirstTouchMeasurement,
    config: FirstTouchGateConfig | None = None,
) -> FirstTouchEvaluation:
    """Attribute every failed gate in stable causal-priority order."""

    gate = config or FirstTouchGateConfig()
    failures: list[FirstTouchFailure] = []
    if not measurement.contact_detected:
        failures.append(FirstTouchFailure.TOUCH_TOO_SOFT)
    if measurement.selected_foot != measurement.required_foot:
        failures.append(FirstTouchFailure.WRONG_FOOT)
    safety_passed = bool(
        measurement.minimum_pelvis_height_m >= gate.minimum_pelvis_height_m
        and measurement.maximum_torso_tilt_deg <= gate.maximum_torso_tilt_deg
        and measurement.maximum_root_speed_mps <= gate.maximum_root_speed_mps
    )
    if not safety_passed:
        failures.append(FirstTouchFailure.LOST_BALANCE)
    if measurement.direction_error_deg > gate.maximum_direction_error_deg:
        failures.append(FirstTouchFailure.TOUCH_WRONG_DIRECTION)
    if (
        measurement.outgoing_speed_mps > gate.maximum_outgoing_speed_mps
        or measurement.target_error_m > gate.maximum_target_error_m
    ):
        failures.append(FirstTouchFailure.TOUCH_TOO_HARD)
    elif (
        measurement.contact_detected
        and measurement.outgoing_speed_mps < gate.minimum_outgoing_speed_mps
    ):
        failures.append(FirstTouchFailure.TOUCH_TOO_SOFT)
    if measurement.next_action_latency_sec > gate.maximum_next_action_latency_sec:
        failures.append(FirstTouchFailure.TOO_SLOW_TO_NEXT_ACTION)
    if (
        not gate.minimum_incoming_speed_mps
        <= measurement.incoming_speed_mps
        <= (gate.maximum_incoming_speed_mps)
    ):
        raise ValueError("first-touch measurement is outside the declared incoming-ball suite")

    ordered = tuple(dict.fromkeys(failures))
    controlled = not any(
        item
        in {
            FirstTouchFailure.TOUCH_TOO_HARD,
            FirstTouchFailure.TOUCH_TOO_SOFT,
            FirstTouchFailure.TOUCH_WRONG_DIRECTION,
            FirstTouchFailure.WRONG_FOOT,
        }
        for item in ordered
    )
    successor_ready = FirstTouchFailure.TOO_SLOW_TO_NEXT_ACTION not in ordered
    return FirstTouchEvaluation(
        measurement_hash=measurement.measurement_hash,
        gate_config_hash=gate.config_hash,
        passed=not ordered,
        primary_failure=None if not ordered else ordered[0],
        all_failures=ordered,
        controlled_first_touch=controlled,
        successor_action_ready=successor_ready,
        safety_passed=safety_passed,
    )


def build_first_touch_dream(
    measurement: FirstTouchMeasurement,
    evaluation: FirstTouchEvaluation,
    *,
    maximum_variants: int = 128,
) -> FailureConditionedDream:
    """Route a failed touch into ROSClaw Core without importing it at discovery."""

    if evaluation.measurement_hash != measurement.measurement_hash:
        raise ValueError("first-touch Dream evaluation belongs to another measurement")
    if evaluation.passed or evaluation.primary_failure is None:
        raise ValueError("a passing first touch cannot create a failure-conditioned Dream")
    from rosclaw.continual.failure_curriculum import (
        DreamPerturbation,
        FailureConditionedDream,
        PerturbationDistribution,
    )

    perturbations = (
        DreamPerturbation("incoming_speed_scale", -0.10, 0.10),
        DreamPerturbation("incoming_angle_deg", -10.0, 10.0),
        DreamPerturbation("ground_friction_delta", -0.08, 0.08),
        DreamPerturbation("foot_selector", -1.0, 1.0),
        DreamPerturbation(
            "body_velocity_delta_mps",
            -0.25,
            0.25,
            PerturbationDistribution.NORMAL_CLIPPED,
        ),
    )
    return FailureConditionedDream(
        failure_code=evaluation.primary_failure.value,
        source_snapshot_hash=measurement.source_snapshot_hash,
        body_hash=measurement.body_hash,
        scenario_hash=measurement.scenario_hash,
        perturbations=perturbations,
        maximum_variants=maximum_variants,
    )


__all__ = [
    "FirstTouchEvaluation",
    "FirstTouchFailure",
    "FirstTouchGateConfig",
    "FirstTouchMeasurement",
    "build_first_touch_dream",
    "evaluate_first_touch",
]
