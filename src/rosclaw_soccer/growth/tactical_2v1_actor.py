"""Content-bound high-level actor for the 2v1 pass-or-shoot decision."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from rosclaw_soccer.growth.tactical_2v1 import (
    TacticalAction,
    TwoVsOneDecisionEvidence,
    TwoVsOneState,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

_BASE_FEATURES = (
    "carrier_pressure",
    "teammate_lane_openness",
    "shot_lane_openness",
    "goal_progress",
    "teammate_progress",
)
_EXPANDED_FEATURES = (
    "bias",
    "carrier_pressure",
    "teammate_lane_openness",
    "shot_lane_openness",
    "goal_progress",
    "teammate_progress",
    "pressure_x_teammate_lane",
    "inverse_pressure_x_shot_lane",
    "lane_openness_difference",
    "pressure_squared",
)
_ACTIONS = (TacticalAction.PASS, TacticalAction.SHOOT)


def _state_features(state: TwoVsOneState) -> np.ndarray:
    return np.asarray(
        (
            state.carrier_pressure,
            state.teammate_lane_openness,
            state.shot_lane_openness,
            state.goal_progress,
            state.teammate_progress,
        ),
        dtype=np.float64,
    )


def _expanded(normalized: np.ndarray) -> np.ndarray:
    if normalized.shape != (5,) or not np.all(np.isfinite(normalized)):
        raise ValueError("2v1 actor features must contain five finite values")
    pressure, teammate_lane, shot_lane, goal_progress, teammate_progress = normalized
    return np.asarray(
        (
            1.0,
            pressure,
            teammate_lane,
            shot_lane,
            goal_progress,
            teammate_progress,
            pressure * teammate_lane,
            (1.0 - pressure) * shot_lane,
            teammate_lane - shot_lane,
            pressure * pressure,
        ),
        dtype=np.float64,
    )


@dataclass(frozen=True)
class TwoVsOneTacticalDecision:
    accepted: bool
    action: TacticalAction
    route: str
    confidence: float
    q_pass: float
    q_shoot: float
    support_distance: float
    actor_hash: str


@dataclass(frozen=True)
class TwoVsOneTacticalActor:
    """Failure-reweighted ridge Q actor with explicit OOD fallback."""

    frozen_foundation_hash: str
    frozen_skill_bundle_hash: str
    training_snapshot_hash: str
    implementation_hash: str
    feature_center: tuple[float, ...]
    feature_scale: tuple[float, ...]
    feature_minimum: tuple[float, ...]
    feature_maximum: tuple[float, ...]
    action_weights: tuple[tuple[float, ...], ...]
    training_state_hashes: tuple[str, ...]
    training_evidence_hashes: tuple[str, ...]
    hard_example_state_hashes: tuple[str, ...]
    ridge_l2: float = 1.0e-3
    maximum_ood_distance: float = 0.45
    activation_ceiling: str = "SIM_ONLY"
    promotion_authorized: bool = False
    hardware_authorized: bool = False
    direct_joint_torque_output: bool = False
    schema_version: str = "rosclaw_soccer.two_vs_one_tactical_actor.v1"

    def __post_init__(self) -> None:
        commitments = (
            self.frozen_foundation_hash,
            self.frozen_skill_bundle_hash,
            self.training_snapshot_hash,
            self.implementation_hash,
            *self.training_state_hashes,
            *self.training_evidence_hashes,
            *self.hard_example_state_hashes,
        )
        if any(not value.startswith("sha256:") or len(value) != 71 for value in commitments):
            raise ValueError("2v1 actor commitments must be SHA-256 hashes")
        center = np.asarray(self.feature_center, dtype=np.float64)
        scale = np.asarray(self.feature_scale, dtype=np.float64)
        minimum = np.asarray(self.feature_minimum, dtype=np.float64)
        maximum = np.asarray(self.feature_maximum, dtype=np.float64)
        weights = np.asarray(self.action_weights, dtype=np.float64)
        if (
            center.shape != (5,)
            or scale.shape != (5,)
            or minimum.shape != (5,)
            or maximum.shape != (5,)
            or weights.shape != (2, len(_EXPANDED_FEATURES))
            or not all(
                np.all(np.isfinite(value)) for value in (center, scale, minimum, maximum, weights)
            )
            or np.any(scale <= 0.0)
            or np.any(minimum > maximum)
            or len(self.training_state_hashes) < 8
            or len(self.training_evidence_hashes) != 2 * len(self.training_state_hashes)
            or len(set(self.training_state_hashes)) != len(self.training_state_hashes)
            or len(set(self.training_evidence_hashes)) != len(self.training_evidence_hashes)
        ):
            raise ValueError("2v1 actor support or weights are invalid")
        if (
            not math.isfinite(self.ridge_l2)
            or not 0.0 < self.ridge_l2 <= 1.0
            or not math.isfinite(self.maximum_ood_distance)
            or not 0.05 <= self.maximum_ood_distance <= 1.0
        ):
            raise ValueError("2v1 actor regularization or support bound is invalid")
        if (
            self.activation_ceiling != "SIM_ONLY"
            or self.promotion_authorized
            or self.hardware_authorized
            or self.direct_joint_torque_output
        ):
            raise ValueError("2v1 actor must remain high-level and SIM_ONLY")

    @property
    def actor_hash(self) -> str:
        return str(hash_json(self.to_dict(include_hash=False)))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            **asdict(self),
            "base_feature_names": list(_BASE_FEATURES),
            "expanded_feature_names": list(_EXPANDED_FEATURES),
            "actions": [action.value for action in _ACTIONS],
            "algorithm": "failure_reweighted_ridge_q_v1",
            "pixels_used_for_training": False,
            "retention_evidence_used_for_training": False,
            "decision_frequency_hz": 10.0,
        }
        if include_hash:
            value["actor_hash"] = self.actor_hash
        return value

    def decide(self, state: TwoVsOneState) -> TwoVsOneTacticalDecision:
        if (
            state.frozen_foundation_hash != self.frozen_foundation_hash
            or state.frozen_skill_bundle_hash != self.frozen_skill_bundle_hash
        ):
            raise ValueError("2v1 actor cannot run with a changed low-level bundle")
        raw = _state_features(state)
        minimum = np.asarray(self.feature_minimum, dtype=np.float64)
        maximum = np.asarray(self.feature_maximum, dtype=np.float64)
        below = np.maximum(minimum - raw, 0.0)
        above = np.maximum(raw - maximum, 0.0)
        span = np.maximum(maximum - minimum, 0.05)
        support_distance = float(np.linalg.norm((below + above) / span))
        if support_distance > self.maximum_ood_distance:
            return TwoVsOneTacticalDecision(
                accepted=False,
                action=TacticalAction.HOLD,
                route="FROZEN_HOLD_OOD_FALLBACK",
                confidence=0.0,
                q_pass=0.0,
                q_shoot=0.0,
                support_distance=support_distance,
                actor_hash=self.actor_hash,
            )
        center = np.asarray(self.feature_center, dtype=np.float64)
        scale = np.asarray(self.feature_scale, dtype=np.float64)
        features = _expanded((raw - center) / scale)
        q_values = np.asarray(self.action_weights, dtype=np.float64) @ features
        selected = int(np.argmax(q_values))
        margin = abs(float(q_values[0] - q_values[1]))
        confidence = float(1.0 - math.exp(-max(0.0, margin)))
        return TwoVsOneTacticalDecision(
            accepted=True,
            action=_ACTIONS[selected],
            route="LEARNED_TACTICAL_Q",
            confidence=confidence,
            q_pass=float(q_values[0]),
            q_shoot=float(q_values[1]),
            support_distance=support_distance,
            actor_hash=self.actor_hash,
        )


def _ridge(
    design: np.ndarray,
    targets: np.ndarray,
    *,
    weights: np.ndarray,
    l2: float,
) -> np.ndarray:
    root = np.sqrt(weights)
    weighted_design = design * root[:, None]
    weighted_targets = targets * root
    penalty = np.eye(design.shape[1], dtype=np.float64) * l2
    penalty[0, 0] = 0.0
    return np.linalg.solve(
        weighted_design.T @ weighted_design + penalty,
        weighted_design.T @ weighted_targets,
    )


def fit_two_vs_one_tactical_actor(
    evidence: Iterable[TwoVsOneDecisionEvidence],
    *,
    ridge_l2: float = 1.0e-3,
    hard_example_weight: float = 4.0,
    maximum_ood_distance: float = 0.45,
) -> TwoVsOneTacticalActor:
    """Fit both action values, then replay initial mistakes at higher weight."""

    rows = tuple(evidence)
    if len(rows) < 16 or len(rows) % 2:
        raise ValueError("2v1 actor needs paired PASS/SHOOT evidence for at least eight states")
    grouped: dict[str, dict[TacticalAction, TwoVsOneDecisionEvidence]] = {}
    for item in rows:
        if item.rollout.action not in _ACTIONS:
            raise ValueError("2v1 actor training accepts only PASS/SHOOT evidence")
        grouped.setdefault(item.state.state_hash, {})[item.rollout.action] = item
    ordered_states = sorted(grouped)
    if len(ordered_states) < 8 or any(set(grouped[key]) != set(_ACTIONS) for key in ordered_states):
        raise ValueError("2v1 actor training states must have both physical actions")
    first = grouped[ordered_states[0]][TacticalAction.PASS]
    foundation_hash = first.state.frozen_foundation_hash
    bundle_hash = first.state.frozen_skill_bundle_hash
    if any(
        item.state.frozen_foundation_hash != foundation_hash
        or item.state.frozen_skill_bundle_hash != bundle_hash
        or item.state.state_hash != state_hash
        for state_hash in ordered_states
        for item in grouped[state_hash].values()
    ):
        raise ValueError("2v1 actor training changed the frozen low-level bundle or state")
    raw = np.stack(
        [_state_features(grouped[key][TacticalAction.PASS].state) for key in ordered_states]
    )
    center = np.mean(raw, axis=0)
    scale = np.std(raw, axis=0)
    scale = np.where(scale < 1.0e-6, 1.0, scale)
    design = np.stack([_expanded((row - center) / scale) for row in raw])
    targets = np.asarray(
        [[grouped[key][action].weighted_score for action in _ACTIONS] for key in ordered_states],
        dtype=np.float64,
    )
    unit_weights = np.ones(len(ordered_states), dtype=np.float64)
    initial = np.stack(
        [_ridge(design, targets[:, index], weights=unit_weights, l2=ridge_l2) for index in range(2)]
    )
    predicted = np.argmax(design @ initial.T, axis=1)
    wanted = np.argmax(targets, axis=1)
    target_margin = np.abs(targets[:, 0] - targets[:, 1])
    hard = (predicted != wanted) | (target_margin < 0.20)
    replay_weights = np.where(hard, hard_example_weight, 1.0)
    final = np.stack(
        [
            _ridge(design, targets[:, index], weights=replay_weights, l2=ridge_l2)
            for index in range(2)
        ]
    )
    evidence_hashes = tuple(
        grouped[key][action].evidence_hash for key in ordered_states for action in _ACTIONS
    )
    implementation_hash = hash_bytes(Path(__file__).read_bytes())
    training_snapshot_hash = hash_json(
        {
            "state_hashes": ordered_states,
            "evidence_hashes": evidence_hashes,
            "ridge_l2": ridge_l2,
            "hard_example_weight": hard_example_weight,
            "implementation_hash": implementation_hash,
        }
    )
    return TwoVsOneTacticalActor(
        frozen_foundation_hash=foundation_hash,
        frozen_skill_bundle_hash=bundle_hash,
        training_snapshot_hash=training_snapshot_hash,
        implementation_hash=implementation_hash,
        feature_center=tuple(float(value) for value in center),
        feature_scale=tuple(float(value) for value in scale),
        feature_minimum=tuple(float(value) for value in np.min(raw, axis=0)),
        feature_maximum=tuple(float(value) for value in np.max(raw, axis=0)),
        action_weights=tuple(tuple(float(value) for value in row) for row in final),
        training_state_hashes=tuple(ordered_states),
        training_evidence_hashes=evidence_hashes,
        hard_example_state_hashes=tuple(ordered_states[index] for index in np.flatnonzero(hard)),
        ridge_l2=ridge_l2,
        maximum_ood_distance=maximum_ood_distance,
    )


def save_two_vs_one_tactical_actor(actor: TwoVsOneTacticalActor, path: Path) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(actor.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def load_two_vs_one_tactical_actor(path: Path) -> TwoVsOneTacticalActor:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("2v1 tactical actor artifact must be an object")
    claimed = payload.pop("actor_hash", None)
    for key in (
        "base_feature_names",
        "expanded_feature_names",
        "actions",
        "algorithm",
        "pixels_used_for_training",
        "retention_evidence_used_for_training",
        "decision_frequency_hz",
    ):
        payload.pop(key, None)
    for key in (
        "feature_center",
        "feature_scale",
        "feature_minimum",
        "feature_maximum",
        "training_state_hashes",
        "training_evidence_hashes",
        "hard_example_state_hashes",
    ):
        payload[key] = tuple(payload[key])
    payload["action_weights"] = tuple(tuple(row) for row in payload["action_weights"])
    actor = TwoVsOneTacticalActor(**payload)
    if claimed != actor.actor_hash:
        raise ValueError("2v1 tactical actor hash does not match its payload")
    return actor


__all__ = [
    "TwoVsOneTacticalActor",
    "TwoVsOneTacticalDecision",
    "fit_two_vs_one_tactical_actor",
    "load_two_vs_one_tactical_actor",
    "save_two_vs_one_tactical_actor",
]
