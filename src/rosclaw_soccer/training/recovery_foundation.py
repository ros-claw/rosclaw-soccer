"""Successor-state recovery foundation for a full-body G1 goalkeeper.

The deployable route is deliberately reference-free: a short proprioceptive
history produces continuous weights over impact absorption, get-up and
athletic-ready experts.  Reference motion phase, simulator truth and task
stage may train the experts or gate, but are not accepted by this API.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray
from rosclaw.continual import (
    GrowthSafetyProfile,
    GrowthSafetyUse,
    SkillSuccessorState,
    SuccessorMetricSpec,
    validate_profile_pair,
)

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.recovery_snapshot import load_recovery_snapshot_corpus


class RecoveryExpert(StrEnum):
    """Deployable full-body recovery experts."""

    ABSORB = "ABSORB"
    GET_UP = "GET_UP"
    ATHLETE = "ATHLETE"


class RecoveryResetSource(StrEnum):
    """Frozen training distribution required by Discussion 4."""

    TRUE_POST_SKILL = "TRUE_POST_SKILL"
    PHYSICS_PERTURBATION = "PHYSICS_PERTURBATION"
    RANDOMIZED_RESET = "RANDOMIZED_RESET"
    DIVE_INTERMEDIATE = "DIVE_INTERMEDIATE"
    HARDEST_FAILURE_MEMORY = "HARDEST_FAILURE_MEMORY"
    NIGHTMARE = "NIGHTMARE"


@dataclass(frozen=True)
class RecoveryProprioFrame:
    """One deployable frame; all quantities are locally observable."""

    projected_gravity_z: float
    pelvis_height_m: float
    root_linear_speed_mps: float
    root_angular_speed_rad_s: float
    left_foot_load_normalized: float
    right_foot_load_normalized: float
    nonfoot_contact_load_normalized: float
    mean_joint_speed_rad_s: float
    previous_action_delta_rms: float

    def __post_init__(self) -> None:
        values = tuple(asdict(self).values())
        if any(not math.isfinite(value) for value in values):
            raise ValueError("recovery proprio frame must be finite")
        if not -1.0 <= self.projected_gravity_z <= 1.0:
            raise ValueError("projected gravity must be in [-1, 1]")
        if not 0.0 <= self.pelvis_height_m <= 2.0:
            raise ValueError("pelvis height is outside the G1 envelope")
        non_negative = values[2:]
        if any(value < 0.0 for value in non_negative):
            raise ValueError("recovery proprio magnitudes must be non-negative")

    def vector(self) -> NDArray[np.float64]:
        return np.asarray(tuple(asdict(self).values()), dtype=np.float64)


@dataclass(frozen=True)
class RecoveryGateObservation:
    """Four-frame history matching the StableMimic-style deployable boundary."""

    history: tuple[RecoveryProprioFrame, ...]
    schema_version: str = "rosclaw_soccer.recovery_gate_observation.v1"

    def __post_init__(self) -> None:
        if len(self.history) != 4:
            raise ValueError("recovery gate requires exactly four proprioceptive frames")

    @property
    def current(self) -> RecoveryProprioFrame:
        return self.history[-1]

    def matrix(self) -> NDArray[np.float64]:
        return np.stack(tuple(frame.vector() for frame in self.history))


@dataclass(frozen=True)
class RecoveryGateParameters:
    temperature: float = 0.35
    temporal_smoothing: float = 0.25
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.temperature)
            or not 0.05 <= self.temperature <= 2.0
            or not math.isfinite(self.temporal_smoothing)
            or not 0.0 <= self.temporal_smoothing <= 0.75
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("recovery gate parameters are invalid")

    @property
    def parameter_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass(frozen=True)
class RecoveryGateOutput:
    expert_weights: Mapping[RecoveryExpert, float]
    parameter_hash: str
    schema_version: str = "rosclaw_soccer.recovery_gate_output.v1"

    def __post_init__(self) -> None:
        expected = tuple(RecoveryExpert)
        normalized = {expert: float(self.expert_weights[expert]) for expert in expected}
        if (
            set(self.expert_weights) != set(expected)
            or any(not math.isfinite(value) or value < 0.0 for value in normalized.values())
            or not math.isclose(sum(normalized.values()), 1.0, rel_tol=0.0, abs_tol=1e-9)
        ):
            raise ValueError("recovery gate output must be one finite simplex")
        object.__setattr__(self, "expert_weights", MappingProxyType(normalized))

    @property
    def dominant_expert(self) -> RecoveryExpert:
        return max(self.expert_weights, key=self.expert_weights.__getitem__)


class ProprioceptiveRecoveryGate:
    """Deterministic seed gate that can later be distilled into a neural gate.

    This seed supplies a safe, testable interface before learned weights exist.
    It is not represented as an RL result or a physics-qualified controller.
    """

    def __init__(self, parameters: RecoveryGateParameters | None = None) -> None:
        self.parameters = parameters or RecoveryGateParameters()
        self._previous = np.full(3, 1.0 / 3.0, dtype=np.float64)

    def reset(self) -> None:
        self._previous[:] = 1.0 / 3.0

    def infer(self, observation: RecoveryGateObservation) -> RecoveryGateOutput:
        history = observation.matrix()
        current = history[-1]
        projected_gravity = current[0]
        pelvis_height = current[1]
        linear_speed = current[2]
        angular_speed = current[3]
        bilateral_support = min(current[4], current[5])
        nonfoot_contact = current[6]
        joint_speed = current[7]
        action_delta = current[8]
        history_motion = float(np.mean(history[:, 2] + 0.35 * history[:, 3]))
        standing_height_readiness = 1.0 / (1.0 + math.exp(-30.0 * (pelvis_height - 0.70)))
        settled_readiness = math.exp(-1.5 * linear_speed - 0.8 * angular_speed)

        # Scores encode broad physical envelopes, not a task-phase switch.
        absorb = (
            2.0 * linear_speed
            + 0.95 * angular_speed
            + 0.8 * nonfoot_contact
            + 0.15 * joint_speed
            + 0.25 * action_delta
            + 0.25 * history_motion
        )
        fallen = max(0.0, 0.68 - pelvis_height) * 4.0 + max(0.0, 0.65 - projected_gravity) * 2.0
        get_up = (
            fallen
            + 4.0 * (1.0 - standing_height_readiness) * settled_readiness
            + 0.8 * nonfoot_contact
            - 0.65 * linear_speed
            - 0.22 * angular_speed
        )
        athlete = (
            2.4 * max(0.0, projected_gravity)
            + 1.8 * min(1.0, bilateral_support)
            + 1.5 * min(1.0, pelvis_height / 0.78)
            - 1.6 * linear_speed
            - 0.50 * angular_speed
            - 0.4 * nonfoot_contact
            - 5.0 * (1.0 - standing_height_readiness)
        )
        logits = np.asarray((absorb, get_up, athlete), dtype=np.float64)
        logits = (logits - float(np.max(logits))) / self.parameters.temperature
        raw = np.exp(logits)
        raw /= raw.sum()
        smoothing = self.parameters.temporal_smoothing
        weights = (1.0 - smoothing) * raw + smoothing * self._previous
        weights /= weights.sum()
        self._previous = weights
        return RecoveryGateOutput(
            expert_weights=MappingProxyType(
                {expert: float(weights[index]) for index, expert in enumerate(RecoveryExpert)}
            ),
            parameter_hash=self.parameters.parameter_hash,
        )


def blend_recovery_actions(
    *,
    actions: Mapping[RecoveryExpert, Sequence[float] | NDArray[np.floating[Any]]],
    gate: RecoveryGateOutput,
    maximum_absolute_action: float = 1.0,
) -> NDArray[np.float64]:
    """Blend three normalized 29-DoF actions without adding authority."""

    if not math.isfinite(maximum_absolute_action) or not 0.0 < maximum_absolute_action <= 1.0:
        raise ValueError("recovery blend action bound must be in (0, 1]")
    if set(actions) != set(RecoveryExpert):
        raise ValueError("recovery blend requires exactly three experts")
    result = np.zeros(len(G1_DDS_JOINT_NAMES), dtype=np.float64)
    for expert in RecoveryExpert:
        action = np.asarray(actions[expert], dtype=np.float64)
        if action.shape != result.shape or not np.all(np.isfinite(action)):
            raise ValueError("each recovery expert must emit 29 finite actions")
        if np.any(np.abs(action) > maximum_absolute_action + 1e-12):
            raise ValueError("recovery expert action exceeds its normalized authority")
        result += gate.expert_weights[expert] * action
    return np.clip(result, -maximum_absolute_action, maximum_absolute_action)


@dataclass(frozen=True)
class RecoveryFoundationContract:
    """Content-bound three-expert deployment boundary."""

    absorb_policy_hash: str
    get_up_policy_hash: str
    athlete_policy_hash: str
    gate_parameter_hash: str
    body_hash: str
    action_names: tuple[str, ...] = G1_DDS_JOINT_NAMES
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.recovery_foundation_contract.v1"

    def __post_init__(self) -> None:
        hashes = (
            self.absorb_policy_hash,
            self.get_up_policy_hash,
            self.athlete_policy_hash,
            self.gate_parameter_hash,
            self.body_hash,
        )
        if (
            any(not value.startswith("sha256:") or len(value) != 71 for value in hashes)
            or self.action_names != G1_DDS_JOINT_NAMES
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("recovery foundation binding is invalid")

    @property
    def contract_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass(frozen=True)
class RecoveryTrainingDistribution:
    """Exact reset mixture for the first Recovery Foundation campaign."""

    true_post_skill: float = 0.20
    physics_perturbation: float = 0.30
    randomized_reset: float = 0.20
    dive_intermediate: float = 0.15
    hardest_failure_memory: float = 0.10
    nightmare: float = 0.05

    def __post_init__(self) -> None:
        values = tuple(self.as_mapping().values())
        if any(not math.isfinite(value) or value < 0.0 for value in values) or not math.isclose(
            sum(values), 1.0, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError("recovery training distribution must sum to one")

    def as_mapping(self) -> Mapping[RecoveryResetSource, float]:
        return MappingProxyType(
            {
                RecoveryResetSource.TRUE_POST_SKILL: self.true_post_skill,
                RecoveryResetSource.PHYSICS_PERTURBATION: self.physics_perturbation,
                RecoveryResetSource.RANDOMIZED_RESET: self.randomized_reset,
                RecoveryResetSource.DIVE_INTERMEDIATE: self.dive_intermediate,
                RecoveryResetSource.HARDEST_FAILURE_MEMORY: self.hardest_failure_memory,
                RecoveryResetSource.NIGHTMARE: self.nightmare,
            }
        )

    def allocate(self, batch_size: int) -> Mapping[RecoveryResetSource, int]:
        if not 1 <= batch_size <= 1_000_000:
            raise ValueError("recovery training batch size is invalid")
        weights = self.as_mapping()
        raw = {source: batch_size * weight for source, weight in weights.items()}
        counts = {source: math.floor(value) for source, value in raw.items()}
        remainder = batch_size - sum(counts.values())
        order = sorted(
            weights,
            key=lambda source: (raw[source] - counts[source], weights[source], source.value),
            reverse=True,
        )
        for source in order[:remainder]:
            counts[source] += 1
        return MappingProxyType(counts)


def build_goalkeeper_ready_successor_contract(
    *,
    save_policy_hash: str,
    recovery_policy_hash: str,
    body_hash: str,
) -> SkillSuccessorState:
    """Require one second of continuous goalkeeper readiness at 50 Hz."""

    return SkillSuccessorState(
        contract_id="soccer.goalkeeper.save-to-ready.v1",
        source_skill_id="soccer.goalkeeper.save",
        successor_skill_id="soccer.goalkeeper.ready",
        source_policy_hash=save_policy_hash,
        successor_policy_hash=recovery_policy_hash,
        body_hash=body_hash,
        metrics=(
            SuccessorMetricSpec("pelvis_height_m", minimum=0.70),
            SuccessorMetricSpec("upright_projection", minimum=0.90),
            SuccessorMetricSpec("root_linear_speed_mps", maximum=0.25),
            SuccessorMetricSpec("root_angular_speed_rad_s", maximum=0.50),
            SuccessorMetricSpec("bilateral_support", minimum=1.0),
            SuccessorMetricSpec("facing_field_cos", minimum=0.90),
            SuccessorMetricSpec("inside_keeper_region", minimum=1.0),
            SuccessorMetricSpec("hand_ready_error_rad", maximum=0.18),
            SuccessorMetricSpec("lateral_acceleration_capacity_mps2", minimum=0.50),
        ),
        hold_steps=50,
        maximum_transition_steps=500,
        control_period_s=0.02,
    )


def build_controlled_fall_safety_profiles() -> tuple[GrowthSafetyProfile, GrowthSafetyProfile]:
    """Permit learnable contact in exploration while promotion stays strict."""

    always = ("left_foot", "right_foot")
    hard = ("head", "neck")
    exploration = GrowthSafetyProfile(
        profile_id="soccer.g1.recovery.exploration.v1",
        use=GrowthSafetyUse.EXPLORATION_SIM,
        maximum_joint_limit_excess_rad=0.08,
        maximum_normalized_actuator_command=1.0,
        maximum_head_impact_speed_mps=0.0,
        maximum_root_angular_speed_rad_s=9.0,
        maximum_self_penetration_m=0.012,
        always_allowed_contacts=always,
        hard_fail_contacts=hard,
        phase_contact_permissions={
            "ABSORB": ("left_hand", "right_hand", "left_forearm", "right_forearm"),
            "GET_UP": (
                "left_hand",
                "right_hand",
                "left_forearm",
                "right_forearm",
                "left_upper_arm",
                "right_upper_arm",
                "left_lateral_thigh",
                "right_lateral_thigh",
                "side_torso",
            ),
            "ATHLETE": (),
        },
    )
    promotion = GrowthSafetyProfile(
        profile_id="soccer.g1.recovery.promotion.v1",
        use=GrowthSafetyUse.PROMOTION_SIM,
        maximum_joint_limit_excess_rad=0.02,
        maximum_normalized_actuator_command=0.95,
        maximum_head_impact_speed_mps=0.0,
        maximum_root_angular_speed_rad_s=7.0,
        maximum_self_penetration_m=0.004,
        always_allowed_contacts=always,
        hard_fail_contacts=hard,
        phase_contact_permissions={
            "ABSORB": ("left_hand", "right_hand", "left_forearm", "right_forearm"),
            "GET_UP": ("left_hand", "right_hand", "left_forearm", "right_forearm"),
            "ATHLETE": (),
        },
    )
    validate_profile_pair(exploration, promotion)
    return exploration, promotion


def audit_recovery_gate_on_snapshot_corpus(manifest_path: Path) -> dict[str, Any]:
    """Run the seed gate on causal S49 states without making a rollout claim."""

    snapshots = load_recovery_snapshot_corpus(manifest_path)
    rows = []
    routes: Counter[str] = Counter()
    confusion: Counter[str] = Counter()
    for snapshot in snapshots:
        quaternion = np.asarray(snapshot.qpos[3:7], dtype=np.float64)
        quaternion /= np.linalg.norm(quaternion)
        _, x, y, _ = quaternion
        gravity_z = 1.0 - 2.0 * (x * x + y * y)
        frame = RecoveryProprioFrame(
            projected_gravity_z=float(gravity_z),
            pelvis_height_m=float(snapshot.qpos[2]),
            root_linear_speed_mps=float(np.linalg.norm(snapshot.qvel[:3])),
            root_angular_speed_rad_s=float(np.linalg.norm(snapshot.qvel[3:6])),
            left_foot_load_normalized=float(snapshot.left_foot_supported),
            right_foot_load_normalized=float(snapshot.right_foot_supported),
            # The S49 archive records support booleans but not body-contact load.
            nonfoot_contact_load_normalized=0.0,
            mean_joint_speed_rad_s=float(np.mean(np.abs(snapshot.qvel[6:]))),
            # There is one action per snapshot, so causal action delta is unavailable.
            previous_action_delta_rms=0.0,
        )
        gate = ProprioceptiveRecoveryGate()
        output = gate.infer(RecoveryGateObservation(history=(frame,) * 4))
        route = output.dominant_expert.value
        routes[route] += 1
        confusion[f"{snapshot.posture_cluster}->{route}"] += 1
        rows.append(
            {
                "snapshot_hash": snapshot.snapshot_hash,
                "source_posture_cluster": snapshot.posture_cluster,
                "dominant_expert": route,
                "proprio_frame": asdict(frame),
                "expert_weights": {
                    expert.value: weight for expert, weight in output.expert_weights.items()
                },
            }
        )
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.recovery_gate_snapshot_audit.v1",
        "snapshot_count": len(rows),
        "route_counts": dict(sorted(routes.items())),
        "posture_route_counts": dict(sorted(confusion.items())),
        "rows": rows,
        "input_boundary": "PROPRIOCEPTION_ONLY",
        "history_boundary": "STATIC_DUPLICATE_NO_TEMPORAL_CLAIM",
        "contact_boundary": "FOOT_SUPPORT_ONLY_NO_NONFOOT_FORCE_IN_S49_ARCHIVE",
        "claim_boundary": "OFFLINE_ROUTING_AUDIT_NOT_PHYSICS_ROLLOUT",
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
    }
    report["report_hash"] = hash_json(report)
    return report


def write_recovery_gate_snapshot_audit(*, manifest_path: Path, output_path: Path) -> dict[str, Any]:
    report = audit_recovery_gate_on_snapshot_corpus(manifest_path)
    _write_json_atomic(output_path, report)
    return report


def _write_json_atomic(output_path: Path, report: Mapping[str, Any]) -> None:
    destination = output_path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


__all__ = [
    "ProprioceptiveRecoveryGate",
    "RecoveryExpert",
    "RecoveryFoundationContract",
    "RecoveryGateObservation",
    "RecoveryGateOutput",
    "RecoveryGateParameters",
    "RecoveryProprioFrame",
    "RecoveryResetSource",
    "RecoveryTrainingDistribution",
    "blend_recovery_actions",
    "audit_recovery_gate_on_snapshot_corpus",
    "build_controlled_fall_safety_profiles",
    "build_goalkeeper_ready_successor_contract",
    "write_recovery_gate_snapshot_audit",
]
