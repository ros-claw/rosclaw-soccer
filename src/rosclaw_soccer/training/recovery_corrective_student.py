"""Integrity contracts for a conservative corrective muscle-memory student.

The student is a small deployable-proprioception MLP trained from two equally
weighted streams: multi-step corrective teacher traces and normal-route
examples whose desired increment is exactly zero.  This module intentionally
contains no simulator imports so reports can be validated without JAX/MuJoCo.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_JOINT_COUNT = 29
_MODEL_ARRAYS = (
    "observation_mean",
    "observation_scale",
    "hidden_0_weight",
    "hidden_0_bias",
    "hidden_1_weight",
    "hidden_1_bias",
    "output_weight",
    "output_bias",
)
_GATE_MODEL_ARRAYS = (
    "gate_weight",
    "gate_bias",
    "gate_ood_center",
    "gate_ood_scale",
    "gate_ood_radius",
)
_VETO_GATE_ARRAYS = tuple(f"veto_{name}" for name in _GATE_MODEL_ARRAYS)
_VETO_TRIGGER_SEMANTICS = "veto_gate_primary_trigger_amplitude_only"
_VETO_MINIMUM_AUTHORITY = "veto_gate_minimum_authority"
_CHANNEL_VETO_ARRAYS = (
    "channel_veto_weight",
    "channel_veto_bias",
    "channel_veto_ood_center",
    "channel_veto_ood_scale",
    "channel_veto_ood_radius",
)
_CHANNEL_VETO_CALIBRATION_SOURCE_ARRAYS = (
    "channel_veto_uncalibrated_weight",
    "channel_veto_uncalibrated_bias",
)
_CHANNEL_VETO_TEMPORAL_TRIGGER = "channel_veto_temporal_trigger_mean_authority"
_TEMPORAL_GATE_ARRAYS = (
    "temporal_gate_open_threshold",
    "temporal_gate_exit_threshold",
    "temporal_gate_required_open_steps",
    "temporal_gate_maximum_lease_steps",
    "temporal_gate_cooldown_steps",
    "temporal_gate_maximum_slew",
)
_CORPUS_ARRAYS = (
    "failure_observation",
    "failure_parent_action",
    "failure_target_increment",
    "failure_state_index",
    "failure_control_step",
    "normal_observation",
    "normal_parent_action",
    "train_source_mask",
    "holdout_source_mask",
)
_DAGGER_AUDIT_CORPUS_ARRAYS = (
    "dagger_current_applied_increment",
    "dagger_current_selected_index",
    "dagger_frozen_applied_increment",
    "dagger_frozen_selected_index",
    "dagger_frozen_replay_source_mask",
)


@dataclass(frozen=True)
class RecoveryCorrectiveStudentConfig:
    """Bounded four-GPU distillation and paired-physics exam contract."""

    hidden_sizes: tuple[int, int] = (128, 64)
    maximum_action_increment: float = 0.50
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    training_steps: int = 1_500
    holdout_states_per_window: int = 4
    normal_rollout_steps: int = 600
    trace_steps: int = 20
    normal_sample_count_per_route: int = 20
    minimum_holdout_cost_improvement_fraction: float = 0.01
    maximum_holdout_directional_regression_fraction: float = 0.02
    maximum_holdout_directional_regression_absolute: float = 0.002
    maximum_normal_increment_rms: float = 0.02
    maximum_normal_cost_regression_fraction: float = 0.01
    required_gpu_count: int = 4
    random_seed: int = 5_600
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.recovery_corrective_student_config.v1"

    def __post_init__(self) -> None:
        finite = (
            self.maximum_action_increment,
            self.learning_rate,
            self.weight_decay,
            self.minimum_holdout_cost_improvement_fraction,
            self.maximum_holdout_directional_regression_fraction,
            self.maximum_holdout_directional_regression_absolute,
            self.maximum_normal_increment_rms,
            self.maximum_normal_cost_regression_fraction,
        )
        if (
            len(self.hidden_sizes) != 2
            or any(isinstance(value, bool) or not 16 <= value <= 512 for value in self.hidden_sizes)
            or any(not math.isfinite(value) for value in finite)
            or not 0.01 <= self.maximum_action_increment <= 1.0
            or not 1.0e-6 <= self.learning_rate <= 0.1
            or not 0.0 <= self.weight_decay <= 0.1
            or not 10 <= self.training_steps <= 100_000
            or not 1 <= self.holdout_states_per_window <= 32
            or not 20 <= self.normal_rollout_steps <= 3_000
            or not 4 <= self.trace_steps <= 80
            or not 2 <= self.normal_sample_count_per_route <= self.normal_rollout_steps
            or not 0.0 <= self.minimum_holdout_cost_improvement_fraction <= 0.5
            or not 0.0 <= self.maximum_holdout_directional_regression_fraction <= 0.25
            or not 0.0 <= self.maximum_holdout_directional_regression_absolute <= 0.1
            or not 0.0 < self.maximum_normal_increment_rms <= self.maximum_action_increment
            or not 0.0 <= self.maximum_normal_cost_regression_fraction <= 0.25
            or self.required_gpu_count != 4
            or not 0 <= self.random_seed < 2**31
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
            or self.schema_version != "rosclaw_soccer.recovery_corrective_student_config.v1"
        ):
            raise ValueError("recovery corrective student config is invalid")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass(frozen=True)
class CorrectiveTemporalLeaseConfig:
    """Bounded stateful authority for a learned corrective confidence gate."""

    open_threshold: float = 0.50
    exit_threshold: float = 0.05
    required_open_steps: int = 3
    maximum_lease_steps: int = 20
    cooldown_steps: int = 600
    maximum_slew: float = 0.50
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.corrective_temporal_lease_config.v1"

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.open_threshold)
            or not 0.05 <= self.open_threshold <= 0.99
            or not math.isfinite(self.exit_threshold)
            or not 0.0 <= self.exit_threshold < self.open_threshold
            or isinstance(self.required_open_steps, bool)
            or not 1 <= self.required_open_steps <= 32
            or isinstance(self.maximum_lease_steps, bool)
            or not 1 <= self.maximum_lease_steps <= 256
            or isinstance(self.cooldown_steps, bool)
            or not 0 <= self.cooldown_steps <= 10_000
            or not math.isfinite(self.maximum_slew)
            or not 0.01 <= self.maximum_slew <= 1.0
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
            or self.schema_version != "rosclaw_soccer.corrective_temporal_lease_config.v1"
        ):
            raise ValueError("corrective temporal lease config is invalid")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


def stratified_source_split(
    control_steps: np.ndarray[Any, Any],
    *,
    holdout_per_window: int,
    random_seed: int,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Return source-disjoint masks with holdouts from every failure window."""

    steps = np.asarray(control_steps, dtype=np.int32)
    if steps.ndim != 1 or steps.size < 8 or np.any(steps < 0):
        raise ValueError("corrective student control steps are invalid")
    holdout = np.zeros(steps.shape, dtype=np.bool_)
    random = np.random.default_rng(random_seed)
    for window in sorted({int(value) for value in steps.tolist()}):
        indices = np.flatnonzero(steps == window)
        if indices.size <= holdout_per_window:
            raise ValueError("corrective student window cannot provide disjoint sources")
        holdout[random.permutation(indices)[:holdout_per_window]] = True
    return ~holdout, holdout


def mix_corrective_normal_dagger_replay(
    *,
    parent_observation: np.ndarray[Any, Any],
    parent_action: np.ndarray[Any, Any],
    candidate_observation: np.ndarray[Any, Any],
    candidate_parent_action: np.ndarray[Any, Any],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Interleave parent and candidate-visited zero-correction replay.

    DAgger samples are kept source aligned: every route contributes the same
    number of parent-policy and candidate-drift states, and no frame is moved
    across a train/holdout source boundary by this operation.
    """

    parent_obs = np.asarray(parent_observation, dtype=np.float32)
    parent_act = np.asarray(parent_action, dtype=np.float32)
    candidate_obs = np.asarray(candidate_observation, dtype=np.float32)
    candidate_act = np.asarray(candidate_parent_action, dtype=np.float32)
    if (
        parent_obs.ndim != 3
        or parent_act.ndim != 3
        or parent_obs.shape != candidate_obs.shape
        or parent_act.shape != candidate_act.shape
        or parent_obs.shape[:2] != parent_act.shape[:2]
        or parent_act.shape[2] != _JOINT_COUNT
        or parent_obs.shape[1] < 2
        or parent_obs.shape[1] % 2
        or not all(
            np.all(np.isfinite(value))
            for value in (parent_obs, parent_act, candidate_obs, candidate_act)
        )
        or np.any(np.abs(parent_act) > 1.0 + 1.0e-6)
        or np.any(np.abs(candidate_act) > 1.0 + 1.0e-6)
    ):
        raise ValueError("corrective student DAgger replay is invalid")
    parent_index = np.arange(0, parent_obs.shape[1], 2, dtype=np.int32)
    candidate_index = np.arange(1, parent_obs.shape[1], 2, dtype=np.int32)
    mixed_observation = np.empty_like(parent_obs)
    mixed_action = np.empty_like(parent_act)
    candidate_mask = np.zeros((parent_obs.shape[1],), dtype=np.bool_)
    mixed_observation[:, 0::2] = parent_obs[:, parent_index]
    mixed_action[:, 0::2] = parent_act[:, parent_index]
    mixed_observation[:, 1::2] = candidate_obs[:, candidate_index]
    mixed_action[:, 1::2] = candidate_act[:, candidate_index]
    candidate_mask[1::2] = True
    return mixed_observation, mixed_action, candidate_mask


def mine_corrective_temporal_hard_negatives(
    *,
    observation: np.ndarray[Any, Any],
    parent_action: np.ndarray[Any, Any],
    confidence: np.ndarray[Any, Any],
    sample_count_per_source: int,
    consecutive_window_steps: int = 2,
    excluded_index_mask: np.ndarray[Any, Any] | None = None,
) -> tuple[
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    dict[str, Any],
]:
    """Mine non-overlapping high-confidence windows from normal trajectories.

    The selection rule is source-local and deterministic.  It therefore keeps
    the train/holdout source boundary intact while focusing the silence gate on
    the normal states most likely to create a consecutive false activation.
    """

    obs = np.asarray(observation, dtype=np.float32)
    action = np.asarray(parent_action, dtype=np.float32)
    score = np.asarray(confidence, dtype=np.float32)
    if score.ndim == 3 and score.shape[-1] == 1:
        score = score[..., 0]
    excluded = (
        np.zeros(score.shape, dtype=np.bool_)
        if excluded_index_mask is None
        else np.asarray(excluded_index_mask, dtype=np.bool_)
    )
    if (
        obs.ndim != 3
        or action.ndim != 3
        or score.ndim != 2
        or obs.shape[:2] != action.shape[:2]
        or obs.shape[:2] != score.shape
        or excluded.shape != score.shape
        or action.shape[-1] != _JOINT_COUNT
        or isinstance(sample_count_per_source, bool)
        or sample_count_per_source < 2
        or isinstance(consecutive_window_steps, bool)
        or consecutive_window_steps < 2
        or sample_count_per_source % consecutive_window_steps
        or sample_count_per_source > obs.shape[1]
        or not all(np.all(np.isfinite(value)) for value in (obs, action, score))
        or np.any(score < 0.0)
        or np.any(score > 1.0)
        or np.any(np.abs(action) > 1.0 + 1.0e-6)
    ):
        raise ValueError("corrective temporal hard-negative trace is invalid")
    windows_per_source = sample_count_per_source // consecutive_window_steps
    if (
        windows_per_source * consecutive_window_steps
        > (obs.shape[1] // consecutive_window_steps) * consecutive_window_steps
    ):
        raise ValueError("corrective temporal hard-negative trace is too short")

    selected_index, selected_window_score = _mine_corrective_temporal_window_indices(
        score=score,
        sample_count_per_source=sample_count_per_source,
        consecutive_window_steps=consecutive_window_steps,
        excluded=excluded,
    )

    source_index = np.arange(obs.shape[0], dtype=np.int32)[:, None]
    selected_observation = obs[source_index, selected_index]
    selected_action = action[source_index, selected_index]
    selected_confidence = score[source_index, selected_index]
    return (
        selected_observation.astype(np.float32),
        selected_action.astype(np.float32),
        selected_index,
        {
            "algorithm": "SOURCE_LOCAL_NON_OVERLAPPING_TOP_MIN_CONFIDENCE_WINDOWS",
            "source_count": int(obs.shape[0]),
            "rollout_steps": int(obs.shape[1]),
            "sample_count_per_source": int(sample_count_per_source),
            "consecutive_window_steps": int(consecutive_window_steps),
            "selected_confidence_mean": float(np.mean(selected_confidence)),
            "selected_confidence_minimum": float(np.min(selected_confidence)),
            "selected_window_score_mean": float(np.mean(selected_window_score)),
            "selected_window_score_minimum": float(np.min(selected_window_score)),
        },
    )


def _mine_corrective_temporal_window_indices(
    *,
    score: np.ndarray[Any, Any],
    sample_count_per_source: int,
    consecutive_window_steps: int,
    excluded: np.ndarray[Any, Any],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Return deterministic non-overlapping top-min-score window indices."""

    windows_per_source = sample_count_per_source // consecutive_window_steps
    selected_index = np.empty((score.shape[0], sample_count_per_source), dtype=np.int32)
    selected_window_score = np.empty((score.shape[0], windows_per_source), dtype=np.float32)
    for source in range(score.shape[0]):
        window_score = np.asarray(
            [
                np.min(score[source, start : start + consecutive_window_steps])
                for start in range(score.shape[1] - consecutive_window_steps + 1)
            ],
            dtype=np.float32,
        )
        occupied = excluded[source].copy()
        starts: list[int] = []
        values: list[float] = []
        for start in np.argsort(-window_score, kind="stable"):
            stop = int(start) + consecutive_window_steps
            if np.any(occupied[int(start) : stop]):
                continue
            occupied[int(start) : stop] = True
            starts.append(int(start))
            values.append(float(window_score[int(start)]))
            if len(starts) == windows_per_source:
                break
        if len(starts) != windows_per_source:
            raise RuntimeError("corrective temporal hard-negative mining was incomplete")
        indices = np.asarray(
            [step for start in starts for step in range(start, start + consecutive_window_steps)],
            dtype=np.int32,
        )
        selected_index[source] = np.sort(indices)
        selected_window_score[source] = np.asarray(values, dtype=np.float32)
    return selected_index, selected_window_score


def mix_corrective_cross_domain_normal_replay(
    *,
    current_observation: np.ndarray[Any, Any],
    current_parent_action: np.ndarray[Any, Any],
    frozen_observation: np.ndarray[Any, Any],
    frozen_parent_action: np.ndarray[Any, Any],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Interleave current and frozen normal domains without duplicating sources.

    The frozen domain must contain exactly half as many sources as the current
    domain.  Every frozen source is inserted once, while the remaining rows
    retain current-domain hard negatives.  This makes the anti-forgetting
    mixture deterministic and auditable instead of silently resampling a small
    frozen set until it dominates the gate fit.
    """

    current_obs = np.asarray(current_observation, dtype=np.float32)
    current_action = np.asarray(current_parent_action, dtype=np.float32)
    frozen_obs = np.asarray(frozen_observation, dtype=np.float32)
    frozen_action = np.asarray(frozen_parent_action, dtype=np.float32)
    if (
        current_obs.ndim != 3
        or current_action.ndim != 3
        or frozen_obs.ndim != 3
        or frozen_action.ndim != 3
        or current_obs.shape[:2] != current_action.shape[:2]
        or frozen_obs.shape[:2] != frozen_action.shape[:2]
        or current_obs.shape[1:] != frozen_obs.shape[1:]
        or current_action.shape[1:] != frozen_action.shape[1:]
        or current_obs.shape[0] != 2 * frozen_obs.shape[0]
        or current_action.shape[-1] != _JOINT_COUNT
        or not all(
            np.all(np.isfinite(value))
            for value in (current_obs, current_action, frozen_obs, frozen_action)
        )
        or np.any(np.abs(current_action) > 1.0 + 1.0e-6)
        or np.any(np.abs(frozen_action) > 1.0 + 1.0e-6)
    ):
        raise ValueError("corrective cross-domain normal replay is invalid")
    frozen_mask = np.zeros((current_obs.shape[0],), dtype=np.bool_)
    frozen_mask[0::2] = True
    mixed_observation = current_obs.copy()
    mixed_parent_action = current_action.copy()
    mixed_observation[frozen_mask] = frozen_obs
    mixed_parent_action[frozen_mask] = frozen_action
    return mixed_observation, mixed_parent_action, frozen_mask


def mix_corrective_training_normal_sources(
    *,
    current_observation: np.ndarray[Any, Any],
    current_parent_action: np.ndarray[Any, Any],
    current_train_source_mask: np.ndarray[Any, Any],
    frozen_training_observation: np.ndarray[Any, Any],
    frozen_training_parent_action: np.ndarray[Any, Any],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Replace only current training rows with frozen-domain training rows.

    The current holdout rows are copied byte-for-byte.  Frozen inputs are
    already sliced to their training split, making old holdout consumption
    impossible inside this primitive.
    """

    current_obs = np.asarray(current_observation, dtype=np.float32)
    current_action = np.asarray(current_parent_action, dtype=np.float32)
    train = np.asarray(current_train_source_mask, dtype=np.bool_)
    frozen_obs = np.asarray(frozen_training_observation, dtype=np.float32)
    frozen_action = np.asarray(frozen_training_parent_action, dtype=np.float32)
    if (
        current_obs.ndim != 3
        or current_action.ndim != 3
        or frozen_obs.ndim != 3
        or frozen_action.ndim != 3
        or current_obs.shape[:2] != current_action.shape[:2]
        or frozen_obs.shape != frozen_action.shape[:2] + (current_obs.shape[-1],)
        or current_obs.shape[1:] != frozen_obs.shape[1:]
        or current_action.shape[-1] != _JOINT_COUNT
        or frozen_action.shape[-1] != _JOINT_COUNT
        or train.shape != current_obs.shape[:1]
        or frozen_obs.shape[0] < 1
        or frozen_obs.shape[0] > int(np.sum(train))
        or not all(
            np.all(np.isfinite(value))
            for value in (current_obs, current_action, frozen_obs, frozen_action)
        )
        or np.any(np.abs(current_action) > 1.0 + 1.0e-6)
        or np.any(np.abs(frozen_action) > 1.0 + 1.0e-6)
    ):
        raise ValueError("corrective training-only normal replay is invalid")
    train_index = np.flatnonzero(train)
    replacement_offset = np.linspace(
        0, train_index.size - 1, num=frozen_obs.shape[0], dtype=np.int32
    )
    replacement_index = train_index[replacement_offset]
    if np.unique(replacement_index).size != frozen_obs.shape[0]:
        raise RuntimeError("corrective training-only normal replay reused a source")
    mixed_observation = current_obs.copy()
    mixed_parent_action = current_action.copy()
    mixed_observation[replacement_index] = frozen_obs
    mixed_parent_action[replacement_index] = frozen_action
    frozen_source_mask = np.zeros(train.shape, dtype=np.bool_)
    frozen_source_mask[replacement_index] = True
    if np.any(frozen_source_mask & ~train):
        raise RuntimeError("corrective training-only normal replay touched holdout")
    return mixed_observation, mixed_parent_action, frozen_source_mask


def corrective_stability_retention(
    *,
    parent_effect: np.ndarray[Any, Any],
    candidate_effect: np.ndarray[Any, Any],
    config: RecoveryCorrectiveStudentConfig,
    allow_configured_tolerance: bool,
) -> tuple[bool, float]:
    """Evaluate stability as an independent fail-closed Pareto gate."""

    parent = np.asarray(parent_effect, dtype=np.float64)
    candidate = np.asarray(candidate_effect, dtype=np.float64)
    if (
        parent.ndim < 2
        or parent.shape != candidate.shape
        or parent.shape[-1] != 4
        or not np.all(np.isfinite(parent))
        or not np.all(np.isfinite(candidate))
    ):
        raise ValueError("corrective student stability evidence is invalid")
    mean_parent = float(np.mean(parent, axis=tuple(range(parent.ndim - 1)))[3])
    mean_candidate = float(np.mean(candidate, axis=tuple(range(candidate.ndim - 1)))[3])
    tolerance = (
        max(
            abs(mean_parent) * config.maximum_holdout_directional_regression_fraction,
            config.maximum_holdout_directional_regression_absolute,
        )
        if allow_configured_tolerance
        else 0.0
    )
    return bool(mean_candidate <= mean_parent + tolerance), float(tolerance)


def derive_corrective_channel_gain(
    *,
    action_effect_jacobian: np.ndarray[Any, Any],
    failure_prediction: np.ndarray[Any, Any],
    failure_trace_prediction: np.ndarray[Any, Any],
    normal_trace_prediction: np.ndarray[Any, Any],
    channel_count: int,
    active_gain: float,
    effect_weights: tuple[float, float, float, float] = (3.0, 1.0, 0.5, 0.35),
    allow_fewer_beneficial: bool = False,
) -> tuple[np.ndarray[Any, Any], dict[str, Any]]:
    """Select a sparse actuator subspace from benefit and interference evidence."""

    jacobian = np.asarray(action_effect_jacobian, dtype=np.float64)
    initial = np.asarray(failure_prediction, dtype=np.float64)
    failure_trace = np.asarray(failure_trace_prediction, dtype=np.float64)
    normal_trace = np.asarray(normal_trace_prediction, dtype=np.float64)
    weights = np.asarray(effect_weights, dtype=np.float64)
    if (
        jacobian.ndim != 3
        or jacobian.shape[1:] != (4, _JOINT_COUNT)
        or initial.shape != (jacobian.shape[0], _JOINT_COUNT)
        or failure_trace.ndim != 3
        or normal_trace.ndim != 3
        or failure_trace.shape[0] != jacobian.shape[0]
        or normal_trace.shape[0] != jacobian.shape[0]
        or failure_trace.shape[2] != _JOINT_COUNT
        or normal_trace.shape[2] != _JOINT_COUNT
        or not 1 <= channel_count <= _JOINT_COUNT
        or not math.isfinite(active_gain)
        or not 0.0 < active_gain <= 1.0
        or weights.shape != (4,)
        or np.any(weights < 0.0)
        or not all(
            np.all(np.isfinite(value))
            for value in (jacobian, initial, failure_trace, normal_trace, weights)
        )
    ):
        raise ValueError("corrective student channel-gain evidence is invalid")
    effect_contribution = np.mean(jacobian * initial[:, None, :], axis=0)
    weighted_contribution = np.sum(effect_contribution * weights[:, None], axis=0)
    failure_rms = np.sqrt(np.mean(np.square(failure_trace), axis=(0, 1)))
    normal_rms = np.sqrt(np.mean(np.square(normal_trace), axis=(0, 1)))
    discrimination_ratio = failure_rms / np.maximum(normal_rms, 1.0e-5)
    selection_score = weighted_contribution * discrimination_ratio
    beneficial = np.flatnonzero(weighted_contribution < 0.0)
    if beneficial.size < channel_count and not allow_fewer_beneficial:
        raise ValueError("corrective student has too few beneficial actuator channels")
    selected_count = min(channel_count, int(beneficial.size))
    if selected_count < 1:
        raise ValueError("corrective student has no beneficial actuator channels")
    order = beneficial[np.argsort(selection_score[beneficial], kind="stable")]
    selected = np.sort(order[:selected_count])
    gain = np.zeros((_JOINT_COUNT,), dtype=np.float32)
    gain[selected] = np.float32(active_gain)
    return gain, {
        "algorithm": "WEIGHTED_LOCAL_EFFECT_TIMES_FAILURE_NORMAL_DISCRIMINATION",
        "requested_maximum_channel_count": int(channel_count),
        "selected_channel_count": int(selected.size),
        "selected_joint_indices": [int(value) for value in selected],
        "active_gain": float(active_gain),
        "effect_weights": [float(value) for value in weights],
        "weighted_effect_contribution": [float(value) for value in weighted_contribution],
        "failure_normal_discrimination_ratio": [float(value) for value in discrimination_ratio],
        "selection_score": [float(value) for value in selection_score],
    }


def derive_corrective_effect_budget_gain(
    *,
    action_effect_jacobian: np.ndarray[Any, Any],
    failure_prediction: np.ndarray[Any, Any],
    historical_normal_prediction: np.ndarray[Any, Any],
    base_gain: float | np.ndarray[Any, Any],
    mirrored_channel_pairs: tuple[tuple[int, int], ...] = (),
    minimum_gain_fraction: float = 0.50,
    maximum_gain_fraction: float | None = None,
    effect_weights: tuple[float, float, float, float] = (3.0, 1.0, 0.5, 0.35),
) -> tuple[np.ndarray[Any, Any], dict[str, Any]]:
    """Attenuate locally harmful channels without raising source authority.

    The failure-state Jacobian supplies the causal sign.  Historical normal
    predictions supply an interference audit, but never labels from a frozen
    holdout.  Mirrored channels are budgeted together so a one-sided estimate
    cannot introduce a new asymmetry into a humanoid controller.
    """

    jacobian = np.asarray(action_effect_jacobian, dtype=np.float64)
    failure = np.asarray(failure_prediction, dtype=np.float64)
    historical = np.asarray(historical_normal_prediction, dtype=np.float64)
    weights = np.asarray(effect_weights, dtype=np.float64)
    gain = np.asarray(base_gain, dtype=np.float64)
    maximum_fraction = (
        minimum_gain_fraction if maximum_gain_fraction is None else maximum_gain_fraction
    )
    if gain.shape == ():
        gain = np.full((_JOINT_COUNT,), float(gain), dtype=np.float64)
    elif gain.shape == (1,):
        gain = np.full((_JOINT_COUNT,), float(gain[0]), dtype=np.float64)
    if (
        jacobian.ndim != 3
        or jacobian.shape[1:] != (4, _JOINT_COUNT)
        or failure.shape != (jacobian.shape[0], _JOINT_COUNT)
        or historical.ndim < 2
        or historical.shape[-1] != _JOINT_COUNT
        or gain.shape != (_JOINT_COUNT,)
        or weights.shape != (4,)
        or not math.isfinite(minimum_gain_fraction)
        or not math.isfinite(maximum_fraction)
        or not 0.0 < minimum_gain_fraction <= 1.0
        or not minimum_gain_fraction <= maximum_fraction <= 1.0
        or np.any(weights < 0.0)
        or np.any(gain < 0.0)
        or np.any(gain > 1.0)
        or not np.any(gain > 0.0)
        or not all(
            np.all(np.isfinite(value)) for value in (jacobian, failure, historical, gain, weights)
        )
    ):
        raise ValueError("corrective effect-channel budget evidence is invalid")
    paired_indices: set[int] = set()
    normalized_pairs: list[tuple[int, int]] = []
    for pair in mirrored_channel_pairs:
        if (
            len(pair) != 2
            or pair[0] == pair[1]
            or any(index < 0 or index >= _JOINT_COUNT for index in pair)
            or any(index in paired_indices for index in pair)
        ):
            raise ValueError("corrective effect-channel mirror pairs are invalid")
        normalized = (min(pair), max(pair))
        paired_indices.update(normalized)
        normalized_pairs.append(normalized)

    effect_contribution = np.mean(jacobian * failure[:, None, :], axis=0)
    weighted_contribution = np.sum(effect_contribution * weights[:, None], axis=0)
    failure_rms = np.sqrt(np.mean(np.square(failure), axis=0))
    historical_axes = tuple(range(historical.ndim - 1))
    historical_rms = np.sqrt(np.mean(np.square(historical), axis=historical_axes))
    interference_ratio = historical_rms / np.maximum(failure_rms, 1.0e-5)
    interference_score = weighted_contribution * interference_ratio
    attenuated = np.zeros((_JOINT_COUNT,), dtype=np.bool_)
    attenuation_units: list[tuple[int, ...]] = []
    pair_contribution: list[dict[str, Any]] = []
    for left, right in normalized_pairs:
        contribution = float(weighted_contribution[left] + weighted_contribution[right])
        pair_is_harmful = contribution > 0.0
        if pair_is_harmful:
            attenuated[[left, right]] = True
            attenuation_units.append((left, right))
        pair_contribution.append(
            {
                "indices": [left, right],
                "weighted_effect_contribution": contribution,
                "attenuated": pair_is_harmful,
            }
        )
    unpaired = np.ones((_JOINT_COUNT,), dtype=np.bool_)
    if paired_indices:
        unpaired[np.fromiter(sorted(paired_indices), dtype=np.int64)] = False
    harmful_unpaired = np.flatnonzero(unpaired & (weighted_contribution > 0.0))
    attenuated[harmful_unpaired] = True
    attenuation_units.extend((int(index),) for index in harmful_unpaired)
    derived = gain.copy()
    unit_risk = np.asarray(
        [max(float(interference_ratio[index]) for index in unit) for unit in attenuation_units],
        dtype=np.float64,
    )
    if maximum_fraction == minimum_gain_fraction or unit_risk.size < 2:
        unit_fraction = np.full(unit_risk.shape, minimum_gain_fraction, dtype=np.float64)
    else:
        risk_span = float(np.max(unit_risk) - np.min(unit_risk))
        if risk_span <= 1.0e-12:
            unit_fraction = np.full(unit_risk.shape, minimum_gain_fraction, dtype=np.float64)
        else:
            normalized_risk = (unit_risk - np.min(unit_risk)) / risk_span
            unit_fraction = (
                maximum_fraction - (maximum_fraction - minimum_gain_fraction) * normalized_risk
            )
    channel_fraction = np.ones((_JOINT_COUNT,), dtype=np.float64)
    for unit, fraction in zip(attenuation_units, unit_fraction, strict=True):
        derived[np.asarray(unit, dtype=np.int64)] *= fraction
        channel_fraction[np.asarray(unit, dtype=np.int64)] = fraction
    if np.any(derived > gain + 1.0e-12) or not np.any(derived > 0.0):
        raise ValueError("corrective effect-channel budget raised authority")
    return derived.astype(np.float32), {
        "algorithm": (
            "MIRROR_PRESERVING_RISK_WEIGHTED_FAILURE_EFFECT_BUDGET_WITH_"
            "HISTORICAL_INTERFERENCE_AUDIT"
            if maximum_fraction > minimum_gain_fraction
            else "MIRROR_PRESERVING_FAILURE_EFFECT_BUDGET_WITH_HISTORICAL_INTERFERENCE_AUDIT"
        ),
        "authority_monotonicity": "DERIVED_GAIN_CAN_ONLY_RETAIN_OR_REDUCE_SOURCE_GAIN",
        "minimum_gain_fraction": float(minimum_gain_fraction),
        "maximum_gain_fraction": float(maximum_fraction),
        "channel_gain_fraction": [float(value) for value in channel_fraction],
        "attenuation_unit_historical_risk": [float(value) for value in unit_risk],
        "effect_weights": [float(value) for value in weights],
        "mirrored_channel_pairs": [[left, right] for left, right in normalized_pairs],
        "mirrored_pair_effect_contribution": pair_contribution,
        "attenuated_joint_indices": [int(value) for value in np.flatnonzero(attenuated)],
        "weighted_effect_contribution": [float(value) for value in weighted_contribution],
        "failure_prediction_rms": [float(value) for value in failure_rms],
        "historical_normal_prediction_rms": [float(value) for value in historical_rms],
        "historical_failure_interference_ratio": [float(value) for value in interference_ratio],
        "interference_score": [float(value) for value in interference_score],
        "source_gain": [float(value) for value in gain],
        "derived_gain": [float(value) for value in derived],
    }


def _corrective_student_latent_numpy(
    model: Mapping[str, Any], observation: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    arrays = {name: np.asarray(model[name], dtype=np.float32) for name in _MODEL_ARRAYS}
    value = (np.asarray(observation, dtype=np.float32) - arrays["observation_mean"]) / arrays[
        "observation_scale"
    ]
    value = np.tanh(value @ arrays["hidden_0_weight"] + arrays["hidden_0_bias"])
    return np.asarray(
        np.tanh(value @ arrays["hidden_1_weight"] + arrays["hidden_1_bias"]),
        dtype=np.float32,
    )


def predict_corrective_confidence_numpy(
    model: Mapping[str, Any], observation: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    """Evaluate the learned correction-needed gate with OOD fail-closed semantics."""

    present = {name for name in _GATE_MODEL_ARRAYS if name in model}
    veto_present = {name for name in _VETO_GATE_ARRAYS if name in model}
    if not present:
        if veto_present:
            raise ValueError("corrective student historical veto requires confidence")
        return np.ones(np.asarray(observation).shape[:-1] + (1,), dtype=np.float32)
    if present != set(_GATE_MODEL_ARRAYS):
        raise ValueError("corrective student confidence gate is incomplete")
    latent = _corrective_student_latent_numpy(model, observation)
    confidence = _predict_corrective_named_gate_numpy(model, latent, prefix="")
    if veto_present:
        if veto_present != set(_VETO_GATE_ARRAYS):
            raise ValueError("corrective student historical veto gate is incomplete")
        confidence = confidence * _predict_corrective_named_gate_numpy(
            model, latent, prefix="veto_"
        )
    return confidence.astype(np.float32)


def predict_corrective_primary_confidence_numpy(
    model: Mapping[str, Any], observation: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    present = {name for name in _GATE_MODEL_ARRAYS if name in model}
    if not present:
        return np.ones(np.asarray(observation).shape[:-1] + (1,), dtype=np.float32)
    if present != set(_GATE_MODEL_ARRAYS):
        raise ValueError("corrective student confidence gate is incomplete")
    confidence = _predict_corrective_named_gate_numpy(
        model, _corrective_student_latent_numpy(model, observation), prefix=""
    )
    if "veto_gate_weight" in model and _veto_uses_primary_trigger(model) is False:
        confidence = confidence * _predict_corrective_veto_numpy(model, observation)
    return confidence.astype(np.float32)


def _predict_corrective_veto_numpy(
    model: Mapping[str, Any], observation: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    present = {name for name in _VETO_GATE_ARRAYS if name in model}
    if not present:
        return np.ones(np.asarray(observation).shape[:-1] + (1,), dtype=np.float32)
    if present != set(_VETO_GATE_ARRAYS):
        raise ValueError("corrective student historical veto gate is incomplete")
    confidence = _predict_corrective_named_gate_numpy(
        model, _corrective_student_latent_numpy(model, observation), prefix="veto_"
    )
    if _VETO_MINIMUM_AUTHORITY in model:
        floor = np.asarray(model[_VETO_MINIMUM_AUTHORITY], dtype=np.float32)
        if floor.shape != (1,) or not np.all(np.isfinite(floor)) or not 0.0 <= floor[0] < 1.0:
            raise ValueError("corrective historical veto authority floor is invalid")
        confidence = floor + (1.0 - floor) * confidence
    return confidence.astype(np.float32)


def _veto_uses_primary_trigger(model: Mapping[str, Any]) -> bool:
    if _VETO_TRIGGER_SEMANTICS not in model:
        return False
    value = np.asarray(model[_VETO_TRIGGER_SEMANTICS], dtype=np.float32)
    if value.shape != (1,) or not np.all(np.isfinite(value)) or value[0] != 1.0:
        raise ValueError("corrective historical veto trigger semantics are invalid")
    return True


def _predict_corrective_named_gate_numpy(
    model: Mapping[str, Any], latent: np.ndarray[Any, Any], *, prefix: str
) -> np.ndarray[Any, Any]:
    weight = np.asarray(model[f"{prefix}gate_weight"], dtype=np.float32)
    bias = np.asarray(model[f"{prefix}gate_bias"], dtype=np.float32)
    center = np.asarray(model[f"{prefix}gate_ood_center"], dtype=np.float32)
    scale = np.asarray(model[f"{prefix}gate_ood_scale"], dtype=np.float32)
    radius = np.asarray(model[f"{prefix}gate_ood_radius"], dtype=np.float32)
    if (
        weight.shape != latent.shape[-1:]
        or bias.shape != (1,)
        or center.shape != latent.shape[-1:]
        or scale.shape != latent.shape[-1:]
        or radius.shape != (1,)
        or np.any(scale <= 0.0)
        or radius[0] <= 0.0
        or not all(np.all(np.isfinite(value)) for value in (weight, bias, center, scale, radius))
    ):
        raise ValueError("corrective student confidence gate is invalid")
    logit = np.clip(latent @ weight + bias, -30.0, 30.0)
    probability = 1.0 / (1.0 + np.exp(-logit))
    distance = np.sqrt(np.mean(np.square((latent - center) / scale), axis=-1))
    return np.asarray(
        np.where(distance[..., None] <= radius, probability[..., None], 0.0),
        dtype=np.float32,
    )


def predict_corrective_channel_veto_numpy(
    model: Mapping[str, Any], observation: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    """Predict a fail-closed per-channel authority mask in ``[0, 1]``."""

    present = {name for name in _CHANNEL_VETO_ARRAYS if name in model}
    if present != set(_CHANNEL_VETO_ARRAYS):
        raise ValueError("corrective channel veto is incomplete")
    latent = _corrective_student_latent_numpy(model, observation)
    weight = np.asarray(model["channel_veto_weight"], dtype=np.float32)
    bias = np.asarray(model["channel_veto_bias"], dtype=np.float32)
    center = np.asarray(model["channel_veto_ood_center"], dtype=np.float32)
    scale = np.asarray(model["channel_veto_ood_scale"], dtype=np.float32)
    radius = np.asarray(model["channel_veto_ood_radius"], dtype=np.float32)
    if (
        weight.shape != (latent.shape[-1], _JOINT_COUNT)
        or bias.shape != (_JOINT_COUNT,)
        or center.shape != latent.shape[-1:]
        or scale.shape != latent.shape[-1:]
        or radius.shape != (1,)
        or np.any(scale <= 0.0)
        or radius[0] <= 0.0
        or not all(np.all(np.isfinite(value)) for value in (weight, bias, center, scale, radius))
    ):
        raise ValueError("corrective channel veto is invalid")
    logit = np.clip(latent @ weight + bias, -30.0, 30.0)
    authority = 1.0 / (1.0 + np.exp(-logit))
    distance = np.sqrt(np.mean(np.square((latent - center) / scale), axis=-1))
    return np.asarray(np.where(distance[..., None] <= radius, authority, 0.0), dtype=np.float32)


def attach_corrective_veto_aware_temporal_trigger(
    model: Mapping[str, Any],
    *,
    scale_amplitude_by_consensus: bool = False,
) -> dict[str, np.ndarray[Any, Any]]:
    """Require vector-veto agreement before a temporal reflex lease can open.

    This is an opt-in, archive-visible behavior version. Mode one changes only
    lease eligibility. Mode two also scales the lease target by squared mean
    channel consensus, retaining near-full failure authority while damping
    ambiguous closed-loop entries.
    """

    if {name for name in _CHANNEL_VETO_ARRAYS if name in model} != set(_CHANNEL_VETO_ARRAYS) or {
        name for name in _TEMPORAL_GATE_ARRAYS if name in model
    } != set(_TEMPORAL_GATE_ARRAYS):
        raise ValueError("veto-aware temporal trigger requires complete veto and lease")
    mode = 2.0 if scale_amplitude_by_consensus else 1.0
    return {_CHANNEL_VETO_TEMPORAL_TRIGGER: np.asarray((mode,), dtype=np.float32)}


def predict_corrective_temporal_trigger_confidence_numpy(
    model: Mapping[str, Any], observation: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    """Return the explicitly versioned confidence used to open/hold a lease."""

    confidence = predict_corrective_primary_confidence_numpy(model, observation)
    if _CHANNEL_VETO_TEMPORAL_TRIGGER not in model:
        return confidence
    marker = np.asarray(model[_CHANNEL_VETO_TEMPORAL_TRIGGER], dtype=np.float32)
    if marker.shape != (1,) or marker[0] not in (1.0, 2.0):
        raise ValueError("corrective veto-aware temporal trigger is invalid")
    authority = predict_corrective_channel_veto_numpy(model, observation)
    mean_authority = np.mean(authority, axis=-1, keepdims=True)
    if marker[0] == 2.0:
        mean_authority = np.square(mean_authority)
    return np.asarray(confidence * mean_authority, dtype=np.float32)


def calibrate_corrective_channel_veto(
    model: Mapping[str, Any],
    *,
    logit_temperature: float,
    failure_recall_logit_margin: float = 0.0,
) -> dict[str, np.ndarray[Any, Any]]:
    """Sharpen a vector veto without moving its learned half-authority boundary.

    A temperature greater than one moves confident per-channel decisions toward
    zero or one. An optional positive margin trades a measured amount of normal
    silence for failure recall without changing the learned direction. The veto
    remains in ``[0, 1]`` and cannot exceed archived static output gain.
    """

    present = {name for name in _CHANNEL_VETO_ARRAYS if name in model}
    if (
        present != set(_CHANNEL_VETO_ARRAYS)
        or not math.isfinite(logit_temperature)
        or not 1.0 <= logit_temperature <= 32.0
        or not math.isfinite(failure_recall_logit_margin)
        or not 0.0 <= failure_recall_logit_margin <= 8.0
    ):
        raise ValueError("corrective channel-veto calibration is invalid")
    source_present = {name for name in _CHANNEL_VETO_CALIBRATION_SOURCE_ARRAYS if name in model}
    if source_present and source_present != set(_CHANNEL_VETO_CALIBRATION_SOURCE_ARRAYS):
        raise ValueError("corrective channel-veto calibration source is incomplete")
    calibrated = {
        name: np.asarray(model[name], dtype=np.float32).copy() for name in _CHANNEL_VETO_ARRAYS
    }
    base_weight = np.asarray(
        model.get("channel_veto_uncalibrated_weight", calibrated["channel_veto_weight"]),
        dtype=np.float32,
    )
    base_bias = np.asarray(
        model.get("channel_veto_uncalibrated_bias", calibrated["channel_veto_bias"]),
        dtype=np.float32,
    )
    calibrated["channel_veto_weight"] = np.asarray(
        base_weight * logit_temperature, dtype=np.float32
    )
    calibrated["channel_veto_bias"] = np.asarray(
        base_bias * logit_temperature + failure_recall_logit_margin,
        dtype=np.float32,
    )
    return calibrated


def fit_corrective_channel_veto(
    *,
    model: Mapping[str, Any],
    failure_observation: np.ndarray[Any, Any],
    normal_observation: np.ndarray[Any, Any],
    mirrored_channel_pairs: tuple[tuple[int, int], ...] = (),
    training_steps: int = 1_500,
    batch_size: int = 512,
    learning_rate: float = 0.003,
    weight_decay: float = 0.001,
    ood_quantile: float = 0.999,
    minimum_sample_weight: float = 0.05,
    logit_temperature: float = 1.0,
    random_seed: int = 5_610,
) -> tuple[dict[str, np.ndarray[Any, Any]], dict[str, Any]]:
    """Fit a mirrored vector veto from balanced failure/normal proprioception.

    The head is weighted by the source student's absolute raw action per
    channel.  This makes the 29 classifiers specialize even though all failure
    frames carry an authority-retention label and all normal frames carry a
    suppression label.  Runtime use is multiplicative, so the head cannot
    increase the source model's static action authority.
    """

    failure = np.asarray(failure_observation, dtype=np.float32)
    normal = np.asarray(normal_observation, dtype=np.float32)
    if (
        failure.shape != normal.shape
        or failure.ndim < 2
        or failure.shape[-1] != np.asarray(model["observation_mean"]).size
        or failure.size == 0
        or not all(np.all(np.isfinite(value)) for value in (failure, normal))
        or isinstance(training_steps, bool)
        or not 100 <= training_steps <= 100_000
        or isinstance(batch_size, bool)
        or not 32 <= batch_size <= 16_384
        or not math.isfinite(learning_rate)
        or not 1.0e-5 <= learning_rate <= 0.1
        or not math.isfinite(weight_decay)
        or not 0.0 <= weight_decay <= 0.1
        or not math.isfinite(ood_quantile)
        or not 0.95 <= ood_quantile < 1.0
        or not math.isfinite(minimum_sample_weight)
        or not 0.0 < minimum_sample_weight <= 1.0
        or not math.isfinite(logit_temperature)
        or not 1.0 <= logit_temperature <= 32.0
        or isinstance(random_seed, bool)
        or not 0 <= random_seed < 2**31
    ):
        raise ValueError("corrective channel-veto corpus is invalid")
    paired_indices: set[int] = set()
    normalized_pairs: list[tuple[int, int]] = []
    for pair in mirrored_channel_pairs:
        if (
            len(pair) != 2
            or pair[0] == pair[1]
            or any(index < 0 or index >= _JOINT_COUNT for index in pair)
            or any(index in paired_indices for index in pair)
        ):
            raise ValueError("corrective channel-veto mirror pairs are invalid")
        normalized = (min(pair), max(pair))
        paired_indices.update(normalized)
        normalized_pairs.append(normalized)

    failure_latent = _corrective_student_latent_numpy(model, failure).reshape(
        (-1, np.asarray(model["hidden_1_bias"]).size)
    )
    normal_latent = _corrective_student_latent_numpy(model, normal).reshape(
        (-1, failure_latent.shape[1])
    )
    if failure_latent.shape != normal_latent.shape or failure_latent.shape[0] < 16:
        raise ValueError("corrective channel-veto latent corpus is invalid")
    output_weight = np.asarray(model["output_weight"], dtype=np.float32)
    output_bias = np.asarray(model["output_bias"], dtype=np.float32)
    output_gain = np.asarray(
        model.get("output_gain", np.ones((1,), dtype=np.float32)), dtype=np.float32
    )
    if output_gain.shape == (1,):
        output_gain = np.broadcast_to(output_gain, (_JOINT_COUNT,))
    if output_gain.shape != (_JOINT_COUNT,) or np.any(output_gain < 0.0):
        raise ValueError("corrective channel-veto source gain is invalid")
    failure_activity = np.abs(
        np.tanh(failure_latent @ output_weight + output_bias) * output_gain
    ).astype(np.float64)
    normal_activity = np.abs(
        np.tanh(normal_latent @ output_weight + output_bias) * output_gain
    ).astype(np.float64)
    activity_scale = np.maximum(
        np.quantile(np.concatenate((failure_activity, normal_activity), axis=0), 0.95, axis=0),
        1.0e-5,
    )
    failure_sample_weight = minimum_sample_weight + (1.0 - minimum_sample_weight) * np.clip(
        failure_activity / activity_scale, 0.0, 1.0
    )
    normal_sample_weight = minimum_sample_weight + (1.0 - minimum_sample_weight) * np.clip(
        normal_activity / activity_scale, 0.0, 1.0
    )
    feature = np.concatenate((failure_latent, normal_latent), axis=0).astype(np.float64)
    label = np.concatenate(
        (
            np.ones((failure_latent.shape[0], 1), dtype=np.float64),
            np.zeros((normal_latent.shape[0], 1), dtype=np.float64),
        ),
        axis=0,
    )
    sample_weight = np.concatenate((failure_sample_weight, normal_sample_weight), axis=0)
    center = np.mean(feature, axis=0)
    scale = np.maximum(np.std(feature, axis=0), 1.0e-3)
    normalized = (feature - center) / scale
    weight = np.zeros((normalized.shape[1], _JOINT_COUNT), dtype=np.float64)
    bias = np.zeros((_JOINT_COUNT,), dtype=np.float64)
    first_weight = np.zeros_like(weight)
    second_weight = np.zeros_like(weight)
    first_bias = np.zeros_like(bias)
    second_bias = np.zeros_like(bias)
    random = np.random.default_rng(random_seed)
    order = random.permutation(normalized.shape[0])
    cursor = 0
    effective_batch = min(batch_size, normalized.shape[0])
    for step in range(1, training_steps + 1):
        if cursor + effective_batch > order.size:
            order = random.permutation(normalized.shape[0])
            cursor = 0
        index = order[cursor : cursor + effective_batch]
        cursor += effective_batch
        x = normalized[index]
        y = label[index]
        active_weight = sample_weight[index]
        probability = 1.0 / (1.0 + np.exp(-np.clip(x @ weight + bias, -30.0, 30.0)))
        error = (probability - y) * active_weight
        denominator = np.maximum(np.sum(active_weight, axis=0), 1.0e-8)
        gradient_weight = x.T @ error / denominator + weight_decay * weight
        gradient_bias = np.sum(error, axis=0) / denominator
        first_weight = 0.9 * first_weight + 0.1 * gradient_weight
        second_weight = 0.999 * second_weight + 0.001 * np.square(gradient_weight)
        first_bias = 0.9 * first_bias + 0.1 * gradient_bias
        second_bias = 0.999 * second_bias + 0.001 * np.square(gradient_bias)
        corrected_first_weight = first_weight / (1.0 - 0.9**step)
        corrected_second_weight = second_weight / (1.0 - 0.999**step)
        corrected_first_bias = first_bias / (1.0 - 0.9**step)
        corrected_second_bias = second_bias / (1.0 - 0.999**step)
        weight -= (
            learning_rate * corrected_first_weight / (np.sqrt(corrected_second_weight) + 1.0e-8)
        )
        bias -= learning_rate * corrected_first_bias / (np.sqrt(corrected_second_bias) + 1.0e-8)
        for left, right in normalized_pairs:
            shared_weight = 0.5 * (weight[:, left] + weight[:, right])
            shared_bias = 0.5 * (bias[left] + bias[right])
            weight[:, left] = shared_weight
            weight[:, right] = shared_weight
            bias[left] = shared_bias
            bias[right] = shared_bias

    raw_weight = weight / scale[:, None]
    raw_bias = bias - center @ raw_weight
    distance = np.sqrt(np.mean(np.square((feature - center) / scale), axis=-1))
    radius = float(np.quantile(distance, ood_quantile))
    uncalibrated_weight = raw_weight.astype(np.float32)
    uncalibrated_bias = raw_bias.astype(np.float32)
    veto = {
        "channel_veto_weight": np.asarray(
            uncalibrated_weight * logit_temperature, dtype=np.float32
        ),
        "channel_veto_bias": np.asarray(uncalibrated_bias * logit_temperature, dtype=np.float32),
        "channel_veto_ood_center": center.astype(np.float32),
        "channel_veto_ood_scale": scale.astype(np.float32),
        "channel_veto_ood_radius": np.asarray((radius,), dtype=np.float32),
    }
    if logit_temperature > 1.0:
        veto.update(
            {
                "channel_veto_uncalibrated_weight": uncalibrated_weight,
                "channel_veto_uncalibrated_bias": uncalibrated_bias,
            }
        )
    vetoed_model = {**dict(model), **veto}
    failure_authority = predict_corrective_channel_veto_numpy(vetoed_model, failure)
    normal_authority = predict_corrective_channel_veto_numpy(vetoed_model, normal)
    if any(
        not np.array_equal(failure_authority[..., left], failure_authority[..., right])
        or not np.array_equal(normal_authority[..., left], normal_authority[..., right])
        for left, right in normalized_pairs
    ):
        raise RuntimeError("corrective channel-veto mirror binding failed")
    report = {
        "algorithm": "RAW_ACTIVITY_WEIGHTED_MIRRORED_LATENT_VECTOR_VETO",
        "combination": "STATIC_OUTPUT_GAIN_TIMES_STATE_CONDITIONED_CHANNEL_VETO",
        "authority_monotonicity": "CHANNEL_VETO_CAN_ONLY_RETAIN_OR_REDUCE_STATIC_GAIN",
        "training_steps": int(training_steps),
        "batch_size": int(effective_batch),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "ood_quantile": float(ood_quantile),
        "ood_radius": radius,
        "minimum_sample_weight": float(minimum_sample_weight),
        "random_seed": int(random_seed),
        "mirrored_channel_pairs": [[left, right] for left, right in normalized_pairs],
        "failure_sample_count": int(failure_latent.shape[0]),
        "normal_sample_count": int(normal_latent.shape[0]),
        "failure_mean_authority": [
            float(value)
            for value in np.mean(failure_authority, axis=tuple(range(failure.ndim - 1)))
        ],
        "normal_mean_authority": [
            float(value) for value in np.mean(normal_authority, axis=tuple(range(normal.ndim - 1)))
        ],
        "failure_ood_fraction": float(np.mean(np.all(failure_authority == 0.0, axis=-1))),
        "normal_ood_fraction": float(np.mean(np.all(normal_authority == 0.0, axis=-1))),
    }
    if logit_temperature > 1.0:
        report.update(
            {
                "calibration": (
                    "IN_PROCESS_LOGIT_TEMPERATURE_AROUND_UNCHANGED_HALF_AUTHORITY_BOUNDARY"
                ),
                "calibration_logit_temperature": float(logit_temperature),
            }
        )
    return veto, report


def fit_corrective_historical_veto_gate(
    *,
    model: Mapping[str, Any],
    failure_observation: np.ndarray[Any, Any],
    frozen_normal_observation: np.ndarray[Any, Any],
    training_steps: int = 2_000,
    learning_rate: float = 0.01,
    weight_decay: float = 0.001,
    ood_quantile: float = 0.999,
    minimum_authority: float = 0.0,
) -> tuple[dict[str, np.ndarray[Any, Any]], dict[str, Any]]:
    """Fit a multiplicative historical-domain veto with no authority to amplify.

    The returned arrays use a separate namespace.  Runtime confidence is the
    product of the primary intervention gate and this veto, so adding the veto
    can only retain or reduce corrective authority.
    """

    if not math.isfinite(minimum_authority) or not 0.0 <= minimum_authority < 1.0:
        raise ValueError("corrective historical veto minimum authority is invalid")
    gate, training = fit_corrective_confidence_gate(
        model=model,
        failure_observation=failure_observation,
        normal_observation=frozen_normal_observation,
        training_steps=training_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        ood_quantile=ood_quantile,
    )
    veto = {f"veto_{name}": value for name, value in gate.items()}
    veto[_VETO_TRIGGER_SEMANTICS] = np.asarray((1.0,), dtype=np.float32)
    veto[_VETO_MINIMUM_AUTHORITY] = np.asarray((minimum_authority,), dtype=np.float32)
    return veto, {
        **training,
        "algorithm": "BALANCED_LATENT_HISTORICAL_VETO_WITH_DIAGONAL_OOD_ENVELOPE",
        "combination": "PRIMARY_CONFIDENCE_TIMES_VETO_CONFIDENCE",
        "authority_monotonicity": "VETO_CAN_ONLY_RETAIN_OR_REDUCE_PRIMARY_AUTHORITY",
        "minimum_authority": float(minimum_authority),
    }


def calibrate_corrective_confidence_gate(
    model: Mapping[str, Any], *, threshold: float, logit_temperature: float
) -> dict[str, np.ndarray[Any, Any]]:
    """Sharpen a fitted gate around a conservative intervention threshold.

    The transformation stays smooth and keeps the density envelope unchanged.
    A confidence below ``threshold`` is pushed toward silence, while a value
    above it is pushed toward full corrective authority.  This is a calibration
    operation only; it neither changes the corrective network nor adds actuator
    authority.
    """

    present = {name for name in _GATE_MODEL_ARRAYS if name in model}
    if (
        present != set(_GATE_MODEL_ARRAYS)
        or not math.isfinite(threshold)
        or not 0.5 <= threshold <= 0.99
        or not math.isfinite(logit_temperature)
        or not 1.0 <= logit_temperature <= 32.0
    ):
        raise ValueError("corrective student confidence calibration is invalid")
    weight = np.asarray(model["gate_weight"], dtype=np.float32)
    bias = np.asarray(model["gate_bias"], dtype=np.float32)
    if weight.ndim != 1 or bias.shape != (1,):
        raise ValueError("corrective student confidence calibration is invalid")
    threshold_logit = math.log(threshold / (1.0 - threshold))
    calibrated = {
        name: np.asarray(model[name], dtype=np.float32).copy() for name in _GATE_MODEL_ARRAYS
    }
    calibrated["gate_weight"] = np.asarray(weight * logit_temperature, dtype=np.float32)
    calibrated["gate_bias"] = np.asarray(
        (bias - np.float32(threshold_logit)) * logit_temperature, dtype=np.float32
    )
    return calibrated


def attach_corrective_temporal_lease(
    model: Mapping[str, Any], config: CorrectiveTemporalLeaseConfig | None = None
) -> dict[str, np.ndarray[Any, Any]]:
    """Attach a fail-closed temporal intervention lease to a fitted gate."""

    active = config or CorrectiveTemporalLeaseConfig()
    if {name for name in _GATE_MODEL_ARRAYS if name in model} != set(_GATE_MODEL_ARRAYS):
        raise ValueError("corrective temporal lease requires a complete confidence gate")
    return {
        "temporal_gate_open_threshold": np.asarray((active.open_threshold,), dtype=np.float32),
        "temporal_gate_exit_threshold": np.asarray((active.exit_threshold,), dtype=np.float32),
        "temporal_gate_required_open_steps": np.asarray(
            (active.required_open_steps,), dtype=np.float32
        ),
        "temporal_gate_maximum_lease_steps": np.asarray(
            (active.maximum_lease_steps,), dtype=np.float32
        ),
        "temporal_gate_cooldown_steps": np.asarray((active.cooldown_steps,), dtype=np.float32),
        "temporal_gate_maximum_slew": np.asarray((active.maximum_slew,), dtype=np.float32),
    }


def initial_corrective_temporal_gate_state(
    model: Mapping[str, Any], leading_shape: tuple[int, ...]
) -> np.ndarray[Any, Any]:
    """Create the zero-authority OPEN/HOLD/EXIT state for one rollout batch."""

    present = {name for name in _TEMPORAL_GATE_ARRAYS if name in model}
    if present != set(_TEMPORAL_GATE_ARRAYS):
        raise ValueError("corrective temporal gate is incomplete")
    return np.zeros(leading_shape + (4,), dtype=np.float32)


def step_corrective_temporal_gate_numpy(
    model: Mapping[str, Any],
    confidence: np.ndarray[Any, Any],
    state: np.ndarray[Any, Any],
    *,
    trigger_confidence: np.ndarray[Any, Any] | None = None,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Advance a bounded evidence-accumulating intervention lease by one step."""

    values = {name: np.asarray(model[name], dtype=np.float32) for name in _TEMPORAL_GATE_ARRAYS}
    if any(value.shape != (1,) or not np.all(np.isfinite(value)) for value in values.values()):
        raise ValueError("corrective temporal gate is invalid")
    confidence_array = np.asarray(confidence, dtype=np.float32)
    trigger_array = np.asarray(
        confidence if trigger_confidence is None else trigger_confidence, dtype=np.float32
    )
    state_array = np.asarray(state, dtype=np.float32)
    if (
        confidence_array.shape[-1:] != (1,)
        or trigger_array.shape != confidence_array.shape
        or state_array.shape != confidence_array.shape[:-1] + (4,)
        or not np.all(np.isfinite(confidence_array))
        or not np.all(np.isfinite(trigger_array))
        or not np.all(np.isfinite(state_array))
    ):
        raise ValueError("corrective temporal gate state is invalid")
    open_threshold = float(values["temporal_gate_open_threshold"][0])
    exit_threshold = float(values["temporal_gate_exit_threshold"][0])
    required_open_steps = float(values["temporal_gate_required_open_steps"][0])
    maximum_lease_steps = float(values["temporal_gate_maximum_lease_steps"][0])
    cooldown_steps = float(values["temporal_gate_cooldown_steps"][0])
    maximum_slew = float(values["temporal_gate_maximum_slew"][0])
    if (
        not 0.05 <= open_threshold <= 0.99
        or not 0.0 <= exit_threshold < open_threshold
        or required_open_steps != round(required_open_steps)
        or not 1 <= required_open_steps <= 32
        or maximum_lease_steps != round(maximum_lease_steps)
        or not 1 <= maximum_lease_steps <= 256
        or cooldown_steps != round(cooldown_steps)
        or not 0 <= cooldown_steps <= 10_000
        or not 0.01 <= maximum_slew <= 1.0
    ):
        raise ValueError("corrective temporal gate is invalid")
    confidence_scalar = np.clip(confidence_array[..., 0], 0.0, 1.0)
    trigger_scalar = np.clip(trigger_array[..., 0], 0.0, 1.0)
    gate_value, open_streak, lease_remaining, cooldown_remaining = np.moveaxis(state_array, -1, 0)
    cooldown_remaining = np.maximum(cooldown_remaining - 1.0, 0.0)
    eligible = (lease_remaining <= 0.0) & (cooldown_remaining <= 0.0) & (gate_value <= 1e-6)
    open_streak = np.where(eligible & (trigger_scalar >= open_threshold), open_streak + 1.0, 0.0)
    starting = eligible & (open_streak >= required_open_steps)
    lease_remaining = np.where(starting, maximum_lease_steps, lease_remaining)
    open_streak = np.where(starting, 0.0, open_streak)
    active = lease_remaining > 0.0
    target = np.where(active & (trigger_scalar >= exit_threshold), confidence_scalar, 0.0)
    gate_value = gate_value + np.clip(target - gate_value, -maximum_slew, maximum_slew)
    next_lease = np.maximum(lease_remaining - active.astype(np.float32), 0.0)
    ended = active & (next_lease <= 0.0)
    cooldown_remaining = np.where(ended, cooldown_steps, cooldown_remaining)
    next_state = np.stack(
        (gate_value, open_streak, next_lease, cooldown_remaining), axis=-1
    ).astype(np.float32)
    return gate_value[..., None].astype(np.float32), next_state


def fit_corrective_confidence_gate(
    *,
    model: Mapping[str, Any],
    failure_observation: np.ndarray[Any, Any],
    normal_observation: np.ndarray[Any, Any],
    training_steps: int = 2_000,
    learning_rate: float = 0.01,
    weight_decay: float = 0.001,
    ood_quantile: float = 0.999,
    failure_sample_weight: np.ndarray[Any, Any] | None = None,
    normal_sample_weight: np.ndarray[Any, Any] | None = None,
) -> tuple[dict[str, np.ndarray[Any, Any]], dict[str, Any]]:
    """Fit a balanced latent logistic gate and a fail-closed density envelope.

    Optional weights redistribute attention *within* each class.  Each class is
    normalized back to unit mean before fitting, so temporal-prefix emphasis
    cannot silently turn the balanced failure/normal gate into a class-prior
    hack.
    """

    failure_input = np.asarray(failure_observation, dtype=np.float32)
    normal_input = np.asarray(normal_observation, dtype=np.float32)
    failure = _corrective_student_latent_numpy(model, failure_input).reshape(
        (-1, np.asarray(model["hidden_1_bias"]).size)
    )
    normal = _corrective_student_latent_numpy(model, normal_input).reshape((-1, failure.shape[1]))
    failure_weight = (
        np.ones(failure_input.shape[:-1], dtype=np.float64)
        if failure_sample_weight is None
        else np.asarray(failure_sample_weight, dtype=np.float64)
    )
    normal_weight = (
        np.ones(normal_input.shape[:-1], dtype=np.float64)
        if normal_sample_weight is None
        else np.asarray(normal_sample_weight, dtype=np.float64)
    )
    if (
        failure.shape != normal.shape
        or failure.shape[0] < 16
        or failure_weight.shape != failure_input.shape[:-1]
        or normal_weight.shape != normal_input.shape[:-1]
        or not np.all(np.isfinite(failure_weight))
        or not np.all(np.isfinite(normal_weight))
        or np.any(failure_weight <= 0.0)
        or np.any(normal_weight <= 0.0)
        or not 100 <= training_steps <= 100_000
        or not math.isfinite(learning_rate)
        or not 1.0e-5 <= learning_rate <= 0.1
        or not math.isfinite(weight_decay)
        or not 0.0 <= weight_decay <= 0.1
        or not math.isfinite(ood_quantile)
        or not 0.95 <= ood_quantile < 1.0
    ):
        raise ValueError("corrective student confidence-gate corpus is invalid")
    feature = np.concatenate((failure, normal), axis=0).astype(np.float64)
    label = np.concatenate((np.ones(failure.shape[0]), np.zeros(normal.shape[0])), axis=0).astype(
        np.float64
    )
    failure_weight = failure_weight.reshape((-1,)) / float(np.mean(failure_weight))
    normal_weight = normal_weight.reshape((-1,)) / float(np.mean(normal_weight))
    sample_weight = np.concatenate((failure_weight, normal_weight), axis=0)
    mean = np.mean(feature, axis=0)
    scale = np.maximum(np.std(feature, axis=0), 1.0e-3)
    normalized = (feature - mean) / scale
    weight = np.zeros((feature.shape[1],), dtype=np.float64)
    bias = 0.0
    first_weight = np.zeros_like(weight)
    second_weight = np.zeros_like(weight)
    first_bias = 0.0
    second_bias = 0.0
    for step in range(1, training_steps + 1):
        logit = np.clip(normalized @ weight + bias, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-logit))
        error = (probability - label) * sample_weight
        gradient_weight = normalized.T @ error / np.sum(sample_weight) + weight_decay * weight
        gradient_bias = float(np.sum(error) / np.sum(sample_weight))
        first_weight = 0.9 * first_weight + 0.1 * gradient_weight
        second_weight = 0.999 * second_weight + 0.001 * np.square(gradient_weight)
        first_bias = 0.9 * first_bias + 0.1 * gradient_bias
        second_bias = 0.999 * second_bias + 0.001 * gradient_bias * gradient_bias
        corrected_first_weight = first_weight / (1.0 - 0.9**step)
        corrected_second_weight = second_weight / (1.0 - 0.999**step)
        corrected_first_bias = first_bias / (1.0 - 0.9**step)
        corrected_second_bias = second_bias / (1.0 - 0.999**step)
        weight -= (
            learning_rate * corrected_first_weight / (np.sqrt(corrected_second_weight) + 1.0e-8)
        )
        bias -= learning_rate * corrected_first_bias / (math.sqrt(corrected_second_bias) + 1.0e-8)
    raw_weight = weight / scale
    raw_bias = bias - float(mean @ raw_weight)
    distance = np.sqrt(np.mean(np.square((feature - mean) / scale), axis=-1))
    radius = float(np.quantile(distance, ood_quantile))
    gate = {
        "gate_weight": raw_weight.astype(np.float32),
        "gate_bias": np.asarray((raw_bias,), dtype=np.float32),
        "gate_ood_center": mean.astype(np.float32),
        "gate_ood_scale": scale.astype(np.float32),
        "gate_ood_radius": np.asarray((radius,), dtype=np.float32),
    }
    gated_model = {**dict(model), **gate}
    failure_confidence = predict_corrective_confidence_numpy(gated_model, failure_observation)
    normal_confidence = predict_corrective_confidence_numpy(gated_model, normal_observation)
    return gate, {
        "algorithm": "BALANCED_LATENT_LOGISTIC_GATE_WITH_DIAGONAL_OOD_ENVELOPE",
        "training_steps": int(training_steps),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "ood_quantile": float(ood_quantile),
        "ood_radius": radius,
        "failure_mean_confidence": float(np.mean(failure_confidence)),
        "failure_confidence_rms": float(np.sqrt(np.mean(np.square(failure_confidence)))),
        "normal_mean_confidence": float(np.mean(normal_confidence)),
        "normal_confidence_rms": float(np.sqrt(np.mean(np.square(normal_confidence)))),
        "failure_ood_fraction": float(np.mean(failure_confidence == 0.0)),
        "normal_ood_fraction": float(np.mean(normal_confidence == 0.0)),
        "sample_weight_semantics": "WITHIN_CLASS_UNIT_MEAN_BALANCED",
        "failure_sample_weight_minimum": float(np.min(failure_weight)),
        "failure_sample_weight_maximum": float(np.max(failure_weight)),
        "normal_sample_weight_minimum": float(np.min(normal_weight)),
        "normal_sample_weight_maximum": float(np.max(normal_weight)),
    }


def predict_corrective_raw_numpy(
    model: Mapping[str, Any], observation: np.ndarray[Any, Any], *, maximum_increment: float
) -> np.ndarray[Any, Any]:
    """Predict bounded pre-lease residuals for attribution on training sources."""

    arrays = {name: np.asarray(model[name], dtype=np.float32) for name in _MODEL_ARRAYS}
    value = _corrective_student_latent_numpy(model, observation)
    output_gain = np.asarray(
        model.get("output_gain", np.ones((1,), dtype=np.float32)), dtype=np.float32
    )
    raw = np.asarray(
        output_gain
        * maximum_increment
        * np.tanh(value @ arrays["output_weight"] + arrays["output_bias"]),
        dtype=np.float32,
    )
    if "veto_gate_weight" in model and _veto_uses_primary_trigger(model):
        raw = raw * _predict_corrective_veto_numpy(model, observation)
    channel_veto_present = {name for name in _CHANNEL_VETO_ARRAYS if name in model}
    if channel_veto_present:
        if channel_veto_present != set(_CHANNEL_VETO_ARRAYS):
            raise ValueError("corrective channel veto is incomplete")
        raw = raw * predict_corrective_channel_veto_numpy(model, observation)
    return np.asarray(raw, dtype=np.float32)


def predict_corrective_temporal_sequence_numpy(
    model: Mapping[str, Any], observation: np.ndarray[Any, Any], *, maximum_increment: float
) -> np.ndarray[Any, Any]:
    """Evaluate a temporal lease over explicit ``[..., time, observation]`` sequences."""

    observation_array = np.asarray(observation, dtype=np.float32)
    if observation_array.ndim < 3:
        raise ValueError("corrective temporal prediction requires explicit batched sequences")
    raw = predict_corrective_raw_numpy(
        model, observation_array, maximum_increment=maximum_increment
    )
    confidence = predict_corrective_primary_confidence_numpy(model, observation_array)
    trigger_confidence = predict_corrective_temporal_trigger_confidence_numpy(
        model, observation_array
    )
    lease_confidence = confidence
    marker = np.asarray(model.get(_CHANNEL_VETO_TEMPORAL_TRIGGER, (0.0,)), dtype=np.float32)
    if marker.shape == (1,) and marker[0] == 2.0:
        lease_confidence = trigger_confidence
    state = initial_corrective_temporal_gate_state(model, observation_array.shape[:-2])
    outputs: list[np.ndarray[Any, Any]] = []
    for step in range(observation_array.shape[-2]):
        gate, state = step_corrective_temporal_gate_numpy(
            model,
            lease_confidence[..., step, :],
            state,
            trigger_confidence=trigger_confidence[..., step, :],
        )
        outputs.append(gate * raw[..., step, :])
    return np.stack(outputs, axis=-2).astype(np.float32)


def predict_corrective_student_numpy(
    model: Mapping[str, Any], observation: np.ndarray[Any, Any], *, maximum_increment: float
) -> np.ndarray[Any, Any]:
    """Evaluate the exact corrective policy used by the JAX physics runner."""

    temporal_present = {name for name in _TEMPORAL_GATE_ARRAYS if name in model}
    if temporal_present:
        if temporal_present != set(_TEMPORAL_GATE_ARRAYS):
            raise ValueError("corrective temporal gate is incomplete")
        return predict_corrective_temporal_sequence_numpy(
            model, observation, maximum_increment=maximum_increment
        )
    return np.asarray(
        predict_corrective_primary_confidence_numpy(model, observation)
        * predict_corrective_raw_numpy(model, observation, maximum_increment=maximum_increment),
        dtype=np.float32,
    )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray[Any, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w+b", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **arrays)  # type: ignore[arg-type]
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _array_mapping_content_hash(arrays: Mapping[str, Any]) -> str:
    rows = []
    for name in sorted(arrays):
        value = np.ascontiguousarray(np.asarray(arrays[name]))
        rows.append(
            {
                "name": name,
                "dtype": value.dtype.str,
                "shape": list(value.shape),
                "content_hash": hash_bytes(value.tobytes()),
            }
        )
    return hash_json(rows)


def _normalized_arrays(
    *, corpus: Mapping[str, Any], model: Mapping[str, Any], config: RecoveryCorrectiveStudentConfig
) -> tuple[dict[str, np.ndarray[Any, Any]], dict[str, np.ndarray[Any, Any]]]:
    corpus_arrays = {name: np.asarray(corpus[name]) for name in _CORPUS_ARRAYS if name in corpus}
    dagger_audit_present = {name for name in _DAGGER_AUDIT_CORPUS_ARRAYS if name in corpus}
    if dagger_audit_present and dagger_audit_present != set(_DAGGER_AUDIT_CORPUS_ARRAYS):
        raise ValueError("recovery corrective student DAgger audit corpus is incomplete")
    model_arrays = {
        name: np.asarray(model[name], dtype=np.float32) for name in _MODEL_ARRAYS if name in model
    }
    if set(corpus_arrays) != set(_CORPUS_ARRAYS) or set(model_arrays) != set(_MODEL_ARRAYS):
        raise ValueError("recovery corrective student arrays are incomplete")
    model_arrays["output_gain"] = np.asarray(
        model.get("output_gain", np.ones((1,), dtype=np.float32)), dtype=np.float32
    )
    gate_present = {name for name in _GATE_MODEL_ARRAYS if name in model}
    if gate_present and gate_present != set(_GATE_MODEL_ARRAYS):
        raise ValueError("recovery corrective student confidence gate is incomplete")
    for name in gate_present:
        model_arrays[name] = np.asarray(model[name], dtype=np.float32)
    veto_present = {name for name in _VETO_GATE_ARRAYS if name in model}
    if veto_present and veto_present != set(_VETO_GATE_ARRAYS):
        raise ValueError("recovery corrective student historical veto gate is incomplete")
    if veto_present and not gate_present:
        raise ValueError("recovery corrective student historical veto requires confidence")
    for name in veto_present:
        model_arrays[name] = np.asarray(model[name], dtype=np.float32)
    veto_semantics_present = _VETO_TRIGGER_SEMANTICS in model
    if veto_semantics_present:
        if not veto_present:
            raise ValueError("recovery corrective veto semantics require a veto gate")
        model_arrays[_VETO_TRIGGER_SEMANTICS] = np.asarray(
            model[_VETO_TRIGGER_SEMANTICS], dtype=np.float32
        )
    veto_floor_present = _VETO_MINIMUM_AUTHORITY in model
    if veto_floor_present:
        if not veto_present:
            raise ValueError("recovery corrective veto authority floor requires a veto gate")
        model_arrays[_VETO_MINIMUM_AUTHORITY] = np.asarray(
            model[_VETO_MINIMUM_AUTHORITY], dtype=np.float32
        )
    channel_veto_present = {name for name in _CHANNEL_VETO_ARRAYS if name in model}
    if channel_veto_present and channel_veto_present != set(_CHANNEL_VETO_ARRAYS):
        raise ValueError("recovery corrective student channel veto is incomplete")
    for name in channel_veto_present:
        model_arrays[name] = np.asarray(model[name], dtype=np.float32)
    channel_veto_source_present = {
        name for name in _CHANNEL_VETO_CALIBRATION_SOURCE_ARRAYS if name in model
    }
    if channel_veto_source_present:
        if (
            channel_veto_source_present != set(_CHANNEL_VETO_CALIBRATION_SOURCE_ARRAYS)
            or not channel_veto_present
        ):
            raise ValueError("recovery corrective channel-veto calibration source is incomplete")
        for name in channel_veto_source_present:
            model_arrays[name] = np.asarray(model[name], dtype=np.float32)
    veto_aware_temporal_trigger_present = _CHANNEL_VETO_TEMPORAL_TRIGGER in model
    if veto_aware_temporal_trigger_present:
        model_arrays[_CHANNEL_VETO_TEMPORAL_TRIGGER] = np.asarray(
            model[_CHANNEL_VETO_TEMPORAL_TRIGGER], dtype=np.float32
        )
    temporal_present = {name for name in _TEMPORAL_GATE_ARRAYS if name in model}
    if temporal_present and temporal_present != set(_TEMPORAL_GATE_ARRAYS):
        raise ValueError("recovery corrective student temporal gate is incomplete")
    if temporal_present and not gate_present:
        raise ValueError("recovery corrective student temporal gate requires confidence")
    for name in temporal_present:
        model_arrays[name] = np.asarray(model[name], dtype=np.float32)
    failure_obs = np.asarray(corpus_arrays["failure_observation"], dtype=np.float32)
    normal_obs = np.asarray(corpus_arrays["normal_observation"], dtype=np.float32)
    failure_parent = np.asarray(corpus_arrays["failure_parent_action"], dtype=np.float32)
    normal_parent = np.asarray(corpus_arrays["normal_parent_action"], dtype=np.float32)
    targets = np.asarray(corpus_arrays["failure_target_increment"], dtype=np.float32)
    state_index = np.asarray(corpus_arrays["failure_state_index"], dtype=np.int32)
    control_step = np.asarray(corpus_arrays["failure_control_step"], dtype=np.int32)
    train = np.asarray(corpus_arrays["train_source_mask"], dtype=np.bool_)
    holdout = np.asarray(corpus_arrays["holdout_source_mask"], dtype=np.bool_)
    source_count, trace_steps, observation_dim = failure_obs.shape
    hidden_0, hidden_1 = config.hidden_sizes
    if (
        source_count < 8
        or trace_steps != config.trace_steps
        or normal_obs.shape != (source_count, config.normal_sample_count_per_route, observation_dim)
        or failure_parent.shape != (source_count, trace_steps, _JOINT_COUNT)
        or normal_parent.shape != (source_count, config.normal_sample_count_per_route, _JOINT_COUNT)
        or targets.shape != failure_parent.shape
        or any(
            value.shape != (source_count,) for value in (state_index, control_step, train, holdout)
        )
        or not np.array_equal(train, ~holdout)
        or not np.any(train)
        or not np.any(holdout)
        or np.unique(state_index).size != source_count
        or np.any(np.abs(targets) > config.maximum_action_increment + 1.0e-6)
        or np.any(np.abs(failure_parent) > 1.0 + 1.0e-6)
        or np.any(np.abs(normal_parent) > 1.0 + 1.0e-6)
        or model_arrays["observation_mean"].shape != (observation_dim,)
        or model_arrays["observation_scale"].shape != (observation_dim,)
        or model_arrays["hidden_0_weight"].shape != (observation_dim, hidden_0)
        or model_arrays["hidden_0_bias"].shape != (hidden_0,)
        or model_arrays["hidden_1_weight"].shape != (hidden_0, hidden_1)
        or model_arrays["hidden_1_bias"].shape != (hidden_1,)
        or model_arrays["output_weight"].shape != (hidden_1, _JOINT_COUNT)
        or model_arrays["output_bias"].shape != (_JOINT_COUNT,)
        or model_arrays["output_gain"].shape not in {(1,), (_JOINT_COUNT,)}
        or np.any(model_arrays["output_gain"] < 0.0)
        or not np.any(model_arrays["output_gain"] > 0.0)
        or np.any(model_arrays["output_gain"] > 1.0)
        or np.any(model_arrays["observation_scale"] <= 0.0)
        or (
            bool(gate_present)
            and (
                model_arrays["gate_weight"].shape != (hidden_1,)
                or model_arrays["gate_bias"].shape != (1,)
                or model_arrays["gate_ood_center"].shape != (hidden_1,)
                or model_arrays["gate_ood_scale"].shape != (hidden_1,)
                or model_arrays["gate_ood_radius"].shape != (1,)
                or np.any(model_arrays["gate_ood_scale"] <= 0.0)
                or model_arrays["gate_ood_radius"][0] <= 0.0
            )
        )
        or (
            bool(veto_present)
            and (
                model_arrays["veto_gate_weight"].shape != (hidden_1,)
                or model_arrays["veto_gate_bias"].shape != (1,)
                or model_arrays["veto_gate_ood_center"].shape != (hidden_1,)
                or model_arrays["veto_gate_ood_scale"].shape != (hidden_1,)
                or model_arrays["veto_gate_ood_radius"].shape != (1,)
                or np.any(model_arrays["veto_gate_ood_scale"] <= 0.0)
                or model_arrays["veto_gate_ood_radius"][0] <= 0.0
            )
        )
        or (
            veto_semantics_present
            and (
                model_arrays[_VETO_TRIGGER_SEMANTICS].shape != (1,)
                or model_arrays[_VETO_TRIGGER_SEMANTICS][0] != 1.0
            )
        )
        or (
            veto_floor_present
            and (
                model_arrays[_VETO_MINIMUM_AUTHORITY].shape != (1,)
                or not 0.0 <= model_arrays[_VETO_MINIMUM_AUTHORITY][0] < 1.0
            )
        )
        or (
            bool(channel_veto_present)
            and (
                model_arrays["channel_veto_weight"].shape != (hidden_1, _JOINT_COUNT)
                or model_arrays["channel_veto_bias"].shape != (_JOINT_COUNT,)
                or model_arrays["channel_veto_ood_center"].shape != (hidden_1,)
                or model_arrays["channel_veto_ood_scale"].shape != (hidden_1,)
                or model_arrays["channel_veto_ood_radius"].shape != (1,)
                or np.any(model_arrays["channel_veto_ood_scale"] <= 0.0)
                or model_arrays["channel_veto_ood_radius"][0] <= 0.0
            )
        )
        or (
            bool(channel_veto_source_present)
            and (
                model_arrays["channel_veto_uncalibrated_weight"].shape != (hidden_1, _JOINT_COUNT)
                or model_arrays["channel_veto_uncalibrated_bias"].shape != (_JOINT_COUNT,)
            )
        )
        or (
            veto_aware_temporal_trigger_present
            and (
                not channel_veto_present
                or not temporal_present
                or model_arrays[_CHANNEL_VETO_TEMPORAL_TRIGGER].shape != (1,)
                or model_arrays[_CHANNEL_VETO_TEMPORAL_TRIGGER][0] not in (1.0, 2.0)
            )
        )
        or (
            bool(temporal_present)
            and (
                any(model_arrays[name].shape != (1,) for name in _TEMPORAL_GATE_ARRAYS)
                or not 0.05 <= model_arrays["temporal_gate_open_threshold"][0] <= 0.99
                or not 0.0
                <= model_arrays["temporal_gate_exit_threshold"][0]
                < model_arrays["temporal_gate_open_threshold"][0]
                or model_arrays["temporal_gate_required_open_steps"][0]
                != round(float(model_arrays["temporal_gate_required_open_steps"][0]))
                or not 1 <= model_arrays["temporal_gate_required_open_steps"][0] <= 32
                or model_arrays["temporal_gate_maximum_lease_steps"][0]
                != round(float(model_arrays["temporal_gate_maximum_lease_steps"][0]))
                or not 1 <= model_arrays["temporal_gate_maximum_lease_steps"][0] <= 256
                or model_arrays["temporal_gate_cooldown_steps"][0]
                != round(float(model_arrays["temporal_gate_cooldown_steps"][0]))
                or not 0 <= model_arrays["temporal_gate_cooldown_steps"][0] <= 10_000
                or not 0.01 <= model_arrays["temporal_gate_maximum_slew"][0] <= 1.0
            )
        )
    ):
        raise ValueError("recovery corrective student array contract is invalid")
    dagger_audit: dict[str, np.ndarray[Any, Any]] = {}
    if dagger_audit_present:
        current_applied = np.asarray(corpus["dagger_current_applied_increment"], dtype=np.float32)
        current_selected = np.asarray(corpus["dagger_current_selected_index"], dtype=np.int32)
        frozen_applied = np.asarray(corpus["dagger_frozen_applied_increment"], dtype=np.float32)
        frozen_selected = np.asarray(corpus["dagger_frozen_selected_index"], dtype=np.int32)
        frozen_replay = np.asarray(corpus["dagger_frozen_replay_source_mask"], dtype=np.bool_)
        train_count = int(np.sum(train))
        frozen_count = int(np.sum(frozen_replay))
        required_open_steps = (
            int(model_arrays["temporal_gate_required_open_steps"][0]) if temporal_present else 0
        )
        if (
            not temporal_present
            or required_open_steps < 2
            or config.normal_sample_count_per_route % required_open_steps
            or current_applied.shape != (train_count, config.normal_rollout_steps, _JOINT_COUNT)
            or frozen_applied.shape != (frozen_count, config.normal_rollout_steps, _JOINT_COUNT)
            or current_selected.shape != (train_count, config.normal_sample_count_per_route)
            or frozen_selected.shape != (frozen_count, config.normal_sample_count_per_route)
            or frozen_replay.shape != (source_count,)
            or frozen_count <= 0
            or frozen_count > train_count
            or np.any(frozen_replay & holdout)
            or not all(np.all(np.isfinite(value)) for value in (current_applied, frozen_applied))
        ):
            raise ValueError("recovery corrective student DAgger audit contract is invalid")
        current_score = np.clip(
            np.sqrt(np.mean(np.square(current_applied), axis=-1)) / config.maximum_action_increment,
            0.0,
            1.0,
        ).astype(np.float32)
        frozen_score = np.clip(
            np.sqrt(np.mean(np.square(frozen_applied), axis=-1)) / config.maximum_action_increment,
            0.0,
            1.0,
        ).astype(np.float32)
        expected_current, _ = _mine_corrective_temporal_window_indices(
            score=current_score,
            sample_count_per_source=config.normal_sample_count_per_route,
            consecutive_window_steps=required_open_steps,
            excluded=np.zeros(current_score.shape, dtype=np.bool_),
        )
        expected_frozen, _ = _mine_corrective_temporal_window_indices(
            score=frozen_score,
            sample_count_per_source=config.normal_sample_count_per_route,
            consecutive_window_steps=required_open_steps,
            excluded=np.zeros(frozen_score.shape, dtype=np.bool_),
        )
        if not np.array_equal(current_selected, expected_current) or not np.array_equal(
            frozen_selected, expected_frozen
        ):
            raise ValueError("recovery corrective student DAgger audit contract is invalid")
        dagger_audit = {
            "dagger_current_applied_increment": current_applied,
            "dagger_current_selected_index": current_selected,
            "dagger_frozen_applied_increment": frozen_applied,
            "dagger_frozen_selected_index": frozen_selected,
            "dagger_frozen_replay_source_mask": frozen_replay,
        }
    normalized_corpus = {
        "failure_observation": failure_obs,
        "failure_parent_action": failure_parent,
        "failure_target_increment": targets,
        "failure_state_index": state_index,
        "failure_control_step": control_step,
        "normal_observation": normal_obs,
        "normal_parent_action": normal_parent,
        "train_source_mask": train,
        "holdout_source_mask": holdout,
        **dagger_audit,
    }
    if any(
        not np.all(np.isfinite(value))
        for value in (*normalized_corpus.values(), *model_arrays.values())
    ):
        raise ValueError("recovery corrective student array is non-finite")
    return normalized_corpus, model_arrays


def write_recovery_corrective_student_evidence(
    *,
    output_dir: Path,
    config: RecoveryCorrectiveStudentConfig,
    corpus: Mapping[str, Any],
    model: Mapping[str, Any],
    lineage: Mapping[str, Any],
    devices: tuple[str, ...],
    training: Mapping[str, Any],
    failure_exam: Mapping[str, Any],
    normal_exam: Mapping[str, Any],
) -> dict[str, Any]:
    """Write a byte-bound student, balanced corpus, and paired exam report."""

    destination = output_dir.expanduser().resolve()
    if destination.exists():
        raise ValueError("recovery corrective student refuses to overwrite evidence")
    required_lineage = {
        "teacher_report_hash",
        "teacher_report_file_hash",
        "teacher_corpus_hash",
        "failure_state_manifest_hash",
        "parent_checkpoint_hash",
        "teacher_checkpoint_hash",
        "snapshot_manifest_hash",
        "route_manifest_hash",
        "route_group_hash",
    }
    if set(lineage) != required_lineage or any(
        not isinstance(lineage[name], str) or not _HASH.fullmatch(str(lineage[name]))
        for name in required_lineage
    ):
        raise ValueError("recovery corrective student lineage is invalid")
    if len(devices) != config.required_gpu_count or len(set(devices)) != len(devices):
        raise ValueError("recovery corrective student device map is invalid")
    normalized_corpus, normalized_model = _normalized_arrays(
        corpus=corpus, model=model, config=config
    )
    destination.mkdir(parents=True)
    corpus_path = destination / "corrective-student-corpus.npz"
    model_path = destination / "corrective-student-model.npz"
    _atomic_npz(corpus_path, normalized_corpus)
    _atomic_npz(model_path, normalized_model)
    holdout = normalized_corpus["holdout_source_mask"]
    prediction = predict_corrective_student_numpy(
        normalized_model,
        normalized_corpus["failure_observation"][holdout],
        maximum_increment=config.maximum_action_increment,
    )
    target = normalized_corpus["failure_target_increment"][holdout]
    normal_prediction = predict_corrective_student_numpy(
        normalized_model,
        normalized_corpus["normal_observation"],
        maximum_increment=config.maximum_action_increment,
    )
    holdout_rmse = float(np.sqrt(np.mean(np.square(prediction - target))))
    target_rms = float(np.sqrt(np.mean(np.square(target))))
    normal_rms = float(np.sqrt(np.mean(np.square(normal_prediction))))
    failure_passed = bool(
        failure_exam.get("passed") is True
        and failure_exam.get("stability_retention_passed", True) is True
    )
    normal_passed = bool(
        normal_exam.get("passed") is True
        and normal_exam.get("stability_retention_passed", True) is True
    )
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.recovery_corrective_student_evidence.v1",
        "config": asdict(config),
        "config_hash": config.config_hash,
        **dict(lineage),
        "actor_observation": "DEPLOYABLE_PROPRIOCEPTION_HISTORY_ONLY",
        "label_semantics": "MULTISTEP_TEACHER_INCREMENT_AND_EQUAL_NORMAL_ZERO_REPLAY",
        "model_architecture": (
            "ZERO_OUTPUT_INITIALIZED_TANH_MLP_WITH_STATE_CONDITIONED_CHANNEL_VETO_"
            "AND_STATEFUL_TEMPORAL_INTERVENTION_LEASE"
            if "channel_veto_weight" in normalized_model
            and "temporal_gate_open_threshold" in normalized_model
            else (
                "ZERO_OUTPUT_INITIALIZED_TANH_MLP_WITH_HISTORICAL_VETO_"
                "AND_STATEFUL_TEMPORAL_INTERVENTION_LEASE"
            )
            if "veto_gate_weight" in normalized_model
            and "temporal_gate_open_threshold" in normalized_model
            else "ZERO_OUTPUT_INITIALIZED_TANH_MLP_WITH_STATEFUL_TEMPORAL_INTERVENTION_LEASE"
            if "temporal_gate_open_threshold" in normalized_model
            else (
                "ZERO_OUTPUT_INITIALIZED_TANH_MLP_WITH_FAIL_CLOSED_CONFIDENCE_GATE"
                if "gate_weight" in normalized_model
                else "ZERO_OUTPUT_INITIALIZED_TANH_MLP"
            )
        ),
        "corpus_archive": corpus_path.name,
        "corpus_archive_hash": hash_bytes(corpus_path.read_bytes()),
        "corpus_content_hash": _array_mapping_content_hash(normalized_corpus),
        "model_archive": model_path.name,
        "model_archive_hash": hash_bytes(model_path.read_bytes()),
        "model_content_hash": _array_mapping_content_hash(normalized_model),
        "source_count": int(holdout.size),
        "train_source_count": int(np.sum(~holdout)),
        "holdout_source_count": int(np.sum(holdout)),
        "train_holdout_source_disjoint": True,
        "failure_training_sample_count": int(np.sum(~holdout) * config.trace_steps),
        "normal_training_sample_count": int(
            np.sum(~holdout) * config.normal_sample_count_per_route
        ),
        "balanced_failure_normal_training": bool(
            config.trace_steps == config.normal_sample_count_per_route
        ),
        "supervised_metrics": {
            "holdout_increment_rmse": holdout_rmse,
            "holdout_target_rms": target_rms,
            "normal_predicted_increment_rms": normal_rms,
        },
        "training": dict(training),
        "failure_state_paired_physics_exam": dict(failure_exam),
        "normal_route_paired_physics_exam": dict(normal_exam),
        "four_gpu_training": True,
        "devices": list(devices),
        "student_development_retained": bool(
            failure_passed and normal_passed and normal_rms <= config.maximum_normal_increment_rms
        ),
        "deployment_candidate": False,
        "promotion_eligible": False,
        "promotion_authority": "NONE",
        "cpu_mujoco_exam_required_for_promotion": True,
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    report_path = destination / "student-report.json"
    _atomic_json(report_path, report)
    return validate_recovery_corrective_student_evidence(report_path)


def _validate_corrective_exam_source_diagnostics(
    exam: Mapping[str, Any],
    *,
    config: RecoveryCorrectiveStudentConfig,
    normal_route: bool,
) -> bool:
    """Recompute a paired exam summary when per-source diagnostics are present."""

    diagnostics = exam.get("source_diagnostics")
    if diagnostics is None:
        return True
    if not isinstance(diagnostics, dict):
        return False
    # The MJX summarizer operates on float32 device results.  Reconstruct the
    # same arithmetic here instead of silently promoting JSON scalars to
    # float64: long-route ratios can otherwise drift just over the 1e-7
    # evidence tolerance even though every archived per-source value is exact.
    parent_cost = np.asarray(diagnostics.get("parent_cost"), dtype=np.float32)
    candidate_cost = np.asarray(diagnostics.get("candidate_cost"), dtype=np.float32)
    parent_effect = np.asarray(diagnostics.get("parent_effect_metrics"), dtype=np.float32)
    candidate_effect = np.asarray(diagnostics.get("candidate_effect_metrics"), dtype=np.float32)
    action_rms = np.asarray(diagnostics.get("action_increment_rms"), dtype=np.float32)
    finite = np.asarray(diagnostics.get("finite"), dtype=np.bool_)
    state_count = int(exam.get("state_count", 0))
    if (
        state_count < 1
        or parent_cost.shape != (state_count,)
        or candidate_cost.shape != (state_count,)
        or parent_effect.shape != (state_count, 4)
        or candidate_effect.shape != (state_count, 4)
        or action_rms.shape != (state_count,)
        or finite.shape != (state_count,)
        or not all(
            np.all(np.isfinite(value))
            for value in (parent_cost, candidate_cost, parent_effect, candidate_effect, action_rms)
        )
        or np.any(action_rms < 0.0)
    ):
        return False
    paired_execution = exam.get("paired_execution_semantics")
    causal_identity_enforced = exam.get("exact_zero_intervention_causal_identity_enforced")
    lockstep_contract_valid = bool(paired_execution is None and causal_identity_enforced is None)
    if paired_execution is not None or causal_identity_enforced is not None:
        zero_intervention = action_rms == 0.0
        lockstep_contract_valid = bool(
            paired_execution
            == "LOCKSTEP_SINGLE_GRAPH_EXACT_ZERO_CAUSAL_COUPLING_SHARED_RESET_AND_ACTION_RNG"
            and causal_identity_enforced is True
            and np.array_equal(parent_cost[zero_intervention], candidate_cost[zero_intervention])
            and np.array_equal(
                parent_effect[zero_intervention], candidate_effect[zero_intervention]
            )
        )
    if not lockstep_contract_valid:
        return False
    improvement = (parent_cost - candidate_cost) / np.maximum(np.abs(parent_cost), 1.0e-12)
    mean_parent_effect = np.mean(parent_effect, axis=0)
    mean_candidate_effect = np.mean(candidate_effect, axis=0)
    directional_tolerance = np.maximum(
        np.abs(mean_parent_effect[:3]) * config.maximum_holdout_directional_regression_fraction,
        config.maximum_holdout_directional_regression_absolute,
    )
    directional_passed = bool(
        np.all(mean_candidate_effect[:3] <= mean_parent_effect[:3] + directional_tolerance)
    )
    stability_passed, stability_tolerance = corrective_stability_retention(
        parent_effect=parent_effect,
        candidate_effect=candidate_effect,
        config=config,
        allow_configured_tolerance=normal_route,
    )
    finite_fraction = float(np.mean(finite))
    action_rms_mean = float(np.sqrt(np.mean(np.square(action_rms))))
    if normal_route:
        computed_normal_regression = float(
            (np.mean(candidate_cost) - np.mean(parent_cost))
            / max(abs(float(np.mean(parent_cost))), 1.0e-12)
        )
        normal_regression: float | None = computed_normal_regression
        passed = bool(
            np.all(finite)
            and computed_normal_regression <= config.maximum_normal_cost_regression_fraction
            and action_rms_mean <= config.maximum_normal_increment_rms
            and directional_passed
            and stability_passed
        )
    else:
        normal_regression = None
        passed = bool(
            np.all(finite)
            and float(np.mean(improvement)) >= config.minimum_holdout_cost_improvement_fraction
            and directional_passed
            and stability_passed
        )
    reported_normal_regression = exam.get("normal_cost_regression_fraction")
    normal_regression_valid = bool(
        reported_normal_regression is None
        and normal_regression is None
        or (
            isinstance(reported_normal_regression, (int, float))
            and normal_regression is not None
            and math.isclose(float(reported_normal_regression), normal_regression, abs_tol=1.0e-7)
        )
    )
    return bool(
        math.isclose(
            float(exam.get("mean_parent_cost", -1.0)),
            float(np.mean(parent_cost)),
            abs_tol=1e-7,
        )
        and math.isclose(
            float(exam.get("mean_candidate_cost", -1.0)),
            float(np.mean(candidate_cost)),
            abs_tol=1e-7,
        )
        and math.isclose(
            float(exam.get("mean_cost_improvement_fraction", -1.0)),
            float(np.mean(improvement)),
            abs_tol=1e-7,
        )
        and math.isclose(
            float(exam.get("median_cost_improvement_fraction", -1.0)),
            float(np.median(improvement)),
            abs_tol=1e-7,
        )
        and math.isclose(
            float(exam.get("minimum_cost_improvement_fraction", -1.0)),
            float(np.min(improvement)),
            abs_tol=1e-7,
        )
        and math.isclose(
            float(exam.get("mean_action_increment_rms", -1.0)),
            action_rms_mean,
            abs_tol=1e-7,
        )
        and np.allclose(
            np.asarray(exam.get("mean_parent_effect_metrics"), dtype=np.float64),
            mean_parent_effect,
            rtol=0.0,
            atol=1e-7,
        )
        and np.allclose(
            np.asarray(exam.get("mean_candidate_effect_metrics"), dtype=np.float64),
            mean_candidate_effect,
            rtol=0.0,
            atol=1e-7,
        )
        and exam.get("directional_retention_passed") is directional_passed
        and exam.get("stability_retention_passed") is stability_passed
        and math.isclose(
            float(exam.get("stability_retention_tolerance", -1.0)),
            stability_tolerance,
            abs_tol=1e-7,
        )
        and math.isclose(float(exam.get("finite_fraction", -1.0)), finite_fraction, abs_tol=1e-7)
        and normal_regression_valid
        and exam.get("passed") is passed
    )


def _validate_corrective_dagger_mining(
    mining: Any,
    *,
    applied_increment: np.ndarray[Any, Any],
    selected_index: np.ndarray[Any, Any],
    config: RecoveryCorrectiveStudentConfig,
    consecutive_window_steps: int,
) -> bool:
    """Recompute a closed-loop DAgger mining summary from archived actions."""

    if not isinstance(mining, dict):
        return False
    score = np.clip(
        np.sqrt(np.mean(np.square(applied_increment), axis=-1)) / config.maximum_action_increment,
        0.0,
        1.0,
    ).astype(np.float32)
    expected_index, selected_window_score = _mine_corrective_temporal_window_indices(
        score=score,
        sample_count_per_source=config.normal_sample_count_per_route,
        consecutive_window_steps=consecutive_window_steps,
        excluded=np.zeros(score.shape, dtype=np.bool_),
    )
    source_index = np.arange(score.shape[0], dtype=np.int32)[:, None]
    selected_score = score[source_index, expected_index]
    return bool(
        np.array_equal(selected_index, expected_index)
        and mining.get("algorithm") == "SOURCE_LOCAL_NON_OVERLAPPING_TOP_MIN_CONFIDENCE_WINDOWS"
        and mining.get("score_semantics")
        == "ACTUAL_APPLIED_INCREMENT_RMS_DIVIDED_BY_MAXIMUM_INCREMENT"
        and mining.get("source_count") == score.shape[0]
        and mining.get("rollout_steps") == config.normal_rollout_steps
        and mining.get("sample_count_per_source") == config.normal_sample_count_per_route
        and mining.get("consecutive_window_steps") == consecutive_window_steps
        and math.isclose(
            float(mining.get("selected_confidence_mean", -1.0)),
            float(np.mean(selected_score)),
            abs_tol=1.0e-7,
        )
        and math.isclose(
            float(mining.get("selected_confidence_minimum", -1.0)),
            float(np.min(selected_score)),
            abs_tol=1.0e-7,
        )
        and math.isclose(
            float(mining.get("selected_window_score_mean", -1.0)),
            float(np.mean(selected_window_score)),
            abs_tol=1.0e-7,
        )
        and math.isclose(
            float(mining.get("selected_window_score_minimum", -1.0)),
            float(np.min(selected_window_score)),
            abs_tol=1.0e-7,
        )
        and math.isclose(
            float(mining.get("full_trace_applied_increment_rms", -1.0)),
            float(np.sqrt(np.mean(np.square(applied_increment)))),
            abs_tol=1.0e-7,
        )
    )


def validate_recovery_corrective_student_evidence(path: Path) -> dict[str, Any]:
    """Fail closed on tampering, malformed arrays, or raised authority."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("recovery corrective student evidence is invalid")
    report_hash = payload.pop("report_hash", None)
    valid_hash = report_hash == hash_json(payload)
    payload["report_hash"] = report_hash
    config_payload = payload.get("config")
    try:
        config = (
            RecoveryCorrectiveStudentConfig(**config_payload)
            if isinstance(config_payload, dict)
            else None
        )
    except (TypeError, ValueError):
        config = None
    corpus_path = path.parent / str(payload.get("corpus_archive"))
    model_path = path.parent / str(payload.get("model_archive"))
    archives_valid = bool(
        config is not None
        and payload.get("corpus_archive") == "corrective-student-corpus.npz"
        and payload.get("model_archive") == "corrective-student-model.npz"
        and corpus_path.is_file()
        and model_path.is_file()
        and payload.get("corpus_archive_hash") == hash_bytes(corpus_path.read_bytes())
        and payload.get("model_archive_hash") == hash_bytes(model_path.read_bytes())
    )
    arrays_valid = False
    summaries_valid = False
    effect_budget_valid = False
    channel_veto_valid = False
    content_hashes_valid = False
    if archives_valid and config is not None:
        try:
            with np.load(corpus_path, allow_pickle=False) as archive:
                corpus = {name: np.array(archive[name], copy=True) for name in archive.files}
            with np.load(model_path, allow_pickle=False) as archive:
                model = {name: np.array(archive[name], copy=True) for name in archive.files}
            normalized_corpus, normalized_model = _normalized_arrays(
                corpus=corpus, model=model, config=config
            )
            reported_corpus_content_hash = payload.get("corpus_content_hash")
            reported_model_content_hash = payload.get("model_content_hash")
            content_hashes_valid = bool(
                reported_corpus_content_hash is None
                and reported_model_content_hash is None
                or (
                    reported_corpus_content_hash == _array_mapping_content_hash(normalized_corpus)
                    and reported_model_content_hash == _array_mapping_content_hash(normalized_model)
                )
            )
            arrays_valid = True
            holdout = normalized_corpus["holdout_source_mask"]
            prediction = predict_corrective_student_numpy(
                normalized_model,
                normalized_corpus["failure_observation"][holdout],
                maximum_increment=config.maximum_action_increment,
            )
            target = normalized_corpus["failure_target_increment"][holdout]
            normal_prediction = predict_corrective_student_numpy(
                normalized_model,
                normalized_corpus["normal_observation"],
                maximum_increment=config.maximum_action_increment,
            )
            metrics = payload.get("supervised_metrics")
            failure_exam = payload.get("failure_state_paired_physics_exam")
            normal_exam = payload.get("normal_route_paired_physics_exam")
            normal_prediction_rms = float(np.sqrt(np.mean(np.square(normal_prediction))))
            training = payload.get("training")
            effect_budget = (
                training.get("effect_channel_budget") if isinstance(training, dict) else None
            )
            if effect_budget is None:
                effect_budget_valid = True
            elif isinstance(effect_budget, dict):
                source_gain = np.asarray(effect_budget.get("source_gain"), dtype=np.float32)
                derived_gain = np.asarray(effect_budget.get("derived_gain"), dtype=np.float32)
                archived_gain = np.asarray(normalized_model.get("output_gain"), dtype=np.float32)
                effect_budget_valid = bool(
                    source_gain.shape == (_JOINT_COUNT,)
                    and derived_gain.shape == (_JOINT_COUNT,)
                    and archived_gain.shape == (_JOINT_COUNT,)
                    and np.all(np.isfinite(source_gain))
                    and np.all(np.isfinite(derived_gain))
                    and np.all(source_gain >= 0.0)
                    and np.all(source_gain <= 1.0)
                    and np.all(derived_gain >= 0.0)
                    and np.all(derived_gain <= source_gain + 1e-8)
                    and np.array_equal(archived_gain, derived_gain)
                    and effect_budget.get("authority_monotonicity")
                    == "DERIVED_GAIN_CAN_ONLY_RETAIN_OR_REDUCE_SOURCE_GAIN"
                    and effect_budget.get("selection_split")
                    == "CURRENT_FAILURE_TRAIN_AND_FROZEN_NORMAL_TRAIN_ONLY"
                    and effect_budget.get("frozen_holdout_consumed_for_selection") is False
                    and effect_budget.get("failure_training_source_count") == int(np.sum(~holdout))
                    and isinstance(
                        effect_budget.get("historical_normal_training_source_count"), int
                    )
                    and effect_budget["historical_normal_training_source_count"] > 0
                    and isinstance(effect_budget.get("failure_prediction_content_hash"), str)
                    and _HASH.fullmatch(effect_budget["failure_prediction_content_hash"])
                    and isinstance(
                        effect_budget.get("historical_normal_prediction_content_hash"),
                        str,
                    )
                    and _HASH.fullmatch(effect_budget["historical_normal_prediction_content_hash"])
                )
            channel_veto = training.get("channel_veto") if isinstance(training, dict) else None
            channel_veto_model_present = all(
                name in normalized_model for name in _CHANNEL_VETO_ARRAYS
            )
            if channel_veto is None:
                channel_veto_valid = not channel_veto_model_present
            elif isinstance(channel_veto, dict) and channel_veto_model_present:
                mirrored_pairs = channel_veto.get("mirrored_channel_pairs")
                pairs_valid = isinstance(mirrored_pairs, list)
                mirrored_pair_items = mirrored_pairs if isinstance(mirrored_pairs, list) else []
                normalized_pairs: list[tuple[int, int]] = []
                paired_indices: set[int] = set()
                if pairs_valid:
                    for pair in mirrored_pair_items:
                        if (
                            not isinstance(pair, list)
                            or len(pair) != 2
                            or any(
                                isinstance(index, bool) or not isinstance(index, int)
                                for index in pair
                            )
                            or pair[0] == pair[1]
                            or any(index < 0 or index >= _JOINT_COUNT for index in pair)
                            or any(index in paired_indices for index in pair)
                        ):
                            pairs_valid = False
                            break
                        normalized_pair = (min(pair), max(pair))
                        normalized_pairs.append(normalized_pair)
                        paired_indices.update(normalized_pair)
                archived_gain = np.asarray(normalized_model["output_gain"], dtype=np.float32)
                if archived_gain.shape == (1,):
                    archived_gain = np.broadcast_to(archived_gain, (_JOINT_COUNT,))
                static_gain = np.asarray(channel_veto.get("static_output_gain"), dtype=np.float32)
                failure_authority = np.asarray(
                    channel_veto.get("failure_mean_authority"), dtype=np.float32
                )
                normal_authority = np.asarray(
                    channel_veto.get("normal_mean_authority"), dtype=np.float32
                )
                archived_failure_authority = predict_corrective_channel_veto_numpy(
                    normalized_model,
                    normalized_corpus["failure_observation"][~holdout],
                )
                archived_normal_authority = predict_corrective_channel_veto_numpy(
                    normalized_model,
                    normalized_corpus["normal_observation"][~holdout],
                )
                authority_reduction_axes = tuple(range(archived_failure_authority.ndim - 1))
                archived_failure_mean_authority = np.mean(
                    archived_failure_authority, axis=authority_reduction_axes
                )
                archived_normal_mean_authority = np.mean(
                    archived_normal_authority, axis=authority_reduction_axes
                )
                calibration = channel_veto.get("calibration")
                calibration_source_model_present = all(
                    name in normalized_model for name in _CHANNEL_VETO_CALIBRATION_SOURCE_ARRAYS
                )
                calibration_valid = bool(
                    calibration is None and not calibration_source_model_present
                )
                if calibration == "LOGIT_TEMPERATURE_AROUND_UNCHANGED_HALF_AUTHORITY_BOUNDARY":
                    calibration_temperature = channel_veto.get("calibration_logit_temperature")
                    calibration_valid = bool(
                        not calibration_source_model_present
                        and isinstance(calibration_temperature, (int, float))
                        and not isinstance(calibration_temperature, bool)
                        and math.isfinite(float(calibration_temperature))
                        and 1.0 <= float(calibration_temperature) <= 32.0
                        and "calibration_failure_recall_logit_margin" not in channel_veto
                        and isinstance(
                            channel_veto.get("calibration_source_student_report_hash"), str
                        )
                        and _HASH.fullmatch(channel_veto["calibration_source_student_report_hash"])
                        and isinstance(
                            channel_veto.get("calibration_source_student_report_file_hash"),
                            str,
                        )
                        and _HASH.fullmatch(
                            channel_veto["calibration_source_student_report_file_hash"]
                        )
                    )
                elif calibration == (
                    "IN_PROCESS_LOGIT_TEMPERATURE_AROUND_UNCHANGED_HALF_AUTHORITY_BOUNDARY"
                ):
                    calibration_temperature = channel_veto.get("calibration_logit_temperature")
                    calibration_valid = bool(
                        calibration_source_model_present
                        and isinstance(calibration_temperature, (int, float))
                        and not isinstance(calibration_temperature, bool)
                        and math.isfinite(float(calibration_temperature))
                        and 1.0 < float(calibration_temperature) <= 32.0
                        and "calibration_failure_recall_logit_margin" not in channel_veto
                        and np.array_equal(
                            normalized_model["channel_veto_weight"],
                            np.asarray(
                                normalized_model["channel_veto_uncalibrated_weight"]
                                * float(calibration_temperature),
                                dtype=np.float32,
                            ),
                        )
                        and np.array_equal(
                            normalized_model["channel_veto_bias"],
                            np.asarray(
                                normalized_model["channel_veto_uncalibrated_bias"]
                                * float(calibration_temperature),
                                dtype=np.float32,
                            ),
                        )
                    )
                elif calibration == ("IN_PROCESS_LOGIT_TEMPERATURE_WITH_FAILURE_RECALL_MARGIN"):
                    calibration_temperature = channel_veto.get("calibration_logit_temperature")
                    recall_margin = channel_veto.get("calibration_failure_recall_logit_margin")
                    calibration_valid = bool(
                        calibration_source_model_present
                        and isinstance(calibration_temperature, (int, float))
                        and not isinstance(calibration_temperature, bool)
                        and math.isfinite(float(calibration_temperature))
                        and 1.0 <= float(calibration_temperature) <= 32.0
                        and isinstance(recall_margin, (int, float))
                        and not isinstance(recall_margin, bool)
                        and math.isfinite(float(recall_margin))
                        and 0.0 < float(recall_margin) <= 8.0
                        and np.array_equal(
                            normalized_model["channel_veto_weight"],
                            np.asarray(
                                normalized_model["channel_veto_uncalibrated_weight"]
                                * float(calibration_temperature),
                                dtype=np.float32,
                            ),
                        )
                        and np.array_equal(
                            normalized_model["channel_veto_bias"],
                            np.asarray(
                                normalized_model["channel_veto_uncalibrated_bias"]
                                * float(calibration_temperature)
                                + float(recall_margin),
                                dtype=np.float32,
                            ),
                        )
                    )
                veto_aware_trigger_model_present = (
                    _CHANNEL_VETO_TEMPORAL_TRIGGER in normalized_model
                )
                veto_aware_trigger_report_fields = (
                    "temporal_trigger",
                    "temporal_trigger_amplitude_semantics",
                    "temporal_trigger_source_student_report_hash",
                    "temporal_trigger_source_student_report_file_hash",
                )
                veto_aware_trigger_valid = bool(
                    not veto_aware_trigger_model_present
                    and all(field not in channel_veto for field in veto_aware_trigger_report_fields)
                )
                if veto_aware_trigger_model_present:
                    trigger_mode = float(normalized_model[_CHANNEL_VETO_TEMPORAL_TRIGGER][0])
                    expected_trigger = (
                        "PRIMARY_CONFIDENCE_TIMES_MEAN_CHANNEL_AUTHORITY"
                        if trigger_mode == 1.0
                        else "PRIMARY_CONFIDENCE_TIMES_SQUARED_MEAN_CHANNEL_AUTHORITY"
                    )
                    expected_amplitude = (
                        "PRIMARY_CONFIDENCE_UNCHANGED_AFTER_TRIGGER_QUALIFIES"
                        if trigger_mode == 1.0
                        else "TRIGGER_CONFIDENCE_SETS_LEASE_AMPLITUDE"
                    )
                    veto_aware_trigger_valid = bool(
                        trigger_mode in (1.0, 2.0)
                        and channel_veto.get("temporal_trigger") == expected_trigger
                        and channel_veto.get("temporal_trigger_amplitude_semantics")
                        == expected_amplitude
                        and isinstance(
                            channel_veto.get("temporal_trigger_source_student_report_hash"),
                            str,
                        )
                        and _HASH.fullmatch(
                            channel_veto["temporal_trigger_source_student_report_hash"]
                        )
                        and isinstance(
                            channel_veto.get("temporal_trigger_source_student_report_file_hash"),
                            str,
                        )
                        and _HASH.fullmatch(
                            channel_veto["temporal_trigger_source_student_report_file_hash"]
                        )
                    )
                hash_fields = (
                    "frozen_closed_loop_observation_content_hash",
                    "frozen_closed_loop_parent_action_content_hash",
                    "frozen_selected_index_content_hash",
                    "frozen_replay_source_mask_content_hash",
                    "source_student_report_hash",
                    "source_student_report_file_hash",
                    "current_normal_student_report_hash",
                    "current_normal_student_report_file_hash",
                    "current_normal_corpus_hash",
                    "frozen_normal_student_report_hash",
                    "frozen_normal_student_report_file_hash",
                    "frozen_normal_corpus_hash",
                    "frozen_failure_state_manifest_hash",
                )
                dagger_selection = channel_veto.get("selection_split") == (
                    "CURRENT_AND_FROZEN_CANDIDATE_CLOSED_LOOP_NORMAL_TRAIN_ONLY"
                )
                dagger_hash_fields = (
                    "current_closed_loop_observation_content_hash",
                    "current_closed_loop_parent_action_content_hash",
                    "current_closed_loop_applied_increment_content_hash",
                    "current_selected_index_content_hash",
                    "frozen_closed_loop_applied_increment_content_hash",
                    "dagger_candidate_student_report_hash",
                    "dagger_candidate_student_report_file_hash",
                )
                dagger_audit_present = all(
                    name in normalized_corpus for name in _DAGGER_AUDIT_CORPUS_ARRAYS
                )
                dagger_provenance_valid = not dagger_selection
                if dagger_selection and dagger_audit_present:
                    current_applied = normalized_corpus["dagger_current_applied_increment"]
                    current_selected = normalized_corpus["dagger_current_selected_index"]
                    frozen_applied = normalized_corpus["dagger_frozen_applied_increment"]
                    frozen_selected = normalized_corpus["dagger_frozen_selected_index"]
                    frozen_replay = normalized_corpus["dagger_frozen_replay_source_mask"]
                    mining = channel_veto.get("hard_negative_mining")
                    required_open_steps = int(
                        normalized_model["temporal_gate_required_open_steps"][0]
                    )
                    dagger_provenance_valid = bool(
                        channel_veto.get("current_closed_loop_rollout_steps")
                        == config.normal_rollout_steps
                        and channel_veto.get("frozen_training_source_count")
                        == int(np.sum(frozen_replay))
                        and all(
                            isinstance(channel_veto.get(name), str)
                            and _HASH.fullmatch(channel_veto[name])
                            for name in dagger_hash_fields
                        )
                        and channel_veto.get("current_closed_loop_applied_increment_content_hash")
                        == hash_bytes(np.ascontiguousarray(current_applied).tobytes())
                        and channel_veto.get("current_selected_index_content_hash")
                        == hash_bytes(np.ascontiguousarray(current_selected).tobytes())
                        and channel_veto.get("frozen_closed_loop_applied_increment_content_hash")
                        == hash_bytes(np.ascontiguousarray(frozen_applied).tobytes())
                        and channel_veto.get("frozen_selected_index_content_hash")
                        == hash_bytes(np.ascontiguousarray(frozen_selected).tobytes())
                        and channel_veto.get("frozen_replay_source_mask_content_hash")
                        == hash_bytes(np.ascontiguousarray(frozen_replay).tobytes())
                        and isinstance(mining, dict)
                        and _validate_corrective_dagger_mining(
                            mining.get("current"),
                            applied_increment=current_applied,
                            selected_index=current_selected,
                            config=config,
                            consecutive_window_steps=required_open_steps,
                        )
                        and _validate_corrective_dagger_mining(
                            mining.get("frozen"),
                            applied_increment=frozen_applied,
                            selected_index=frozen_selected,
                            config=config,
                            consecutive_window_steps=required_open_steps,
                        )
                    )
                repair_contract_fields = (
                    "source_repair_mode",
                    "source_failure_gate_passed",
                    "current_normal_on_policy_child_bound",
                )
                repair_contract_present = any(
                    name in channel_veto for name in repair_contract_fields
                )
                repair_contract_valid = not repair_contract_present
                if repair_contract_present:
                    repair_mode = channel_veto.get("source_repair_mode")
                    mining = channel_veto.get("hard_negative_mining")
                    repair_contract_valid = bool(
                        all(
                            isinstance(channel_veto.get(name), bool)
                            for name in repair_contract_fields
                        )
                        and (
                            repair_mode is False
                            or (
                                channel_veto.get("source_failure_gate_passed") is True
                                and channel_veto.get("current_normal_on_policy_child_bound") is True
                                and isinstance(mining, dict)
                                and mining.get("score_semantics")
                                == "SOURCE_RAW_INCREMENT_RMS_DIVIDED_BY_MAXIMUM_INCREMENT"
                            )
                        )
                    )
                mirrored_arrays_valid = bool(
                    pairs_valid
                    and all(
                        np.array_equal(
                            normalized_model["channel_veto_weight"][:, left],
                            normalized_model["channel_veto_weight"][:, right],
                        )
                        and normalized_model["channel_veto_bias"][left]
                        == normalized_model["channel_veto_bias"][right]
                        for left, right in normalized_pairs
                    )
                )
                channel_veto_valid = bool(
                    channel_veto.get("algorithm")
                    == "RAW_ACTIVITY_WEIGHTED_MIRRORED_LATENT_VECTOR_VETO"
                    and channel_veto.get("combination")
                    == "STATIC_OUTPUT_GAIN_TIMES_STATE_CONDITIONED_CHANNEL_VETO"
                    and channel_veto.get("authority_monotonicity")
                    == "CHANNEL_VETO_CAN_ONLY_RETAIN_OR_REDUCE_STATIC_GAIN"
                    and channel_veto.get("selection_split")
                    in {
                        (
                            "CURRENT_FAILURE_TRAIN_CURRENT_NORMAL_TRAIN_"
                            "AND_FROZEN_CLOSED_LOOP_NORMAL_TRAIN_ONLY"
                        ),
                        "CURRENT_AND_FROZEN_CANDIDATE_CLOSED_LOOP_NORMAL_TRAIN_ONLY",
                    }
                    and channel_veto.get("current_holdout_consumed_for_selection") is False
                    and channel_veto.get("frozen_holdout_consumed_for_selection") is False
                    and channel_veto.get("current_training_source_count") == int(np.sum(~holdout))
                    and channel_veto.get("failure_sample_count")
                    == int(np.sum(~holdout) * config.trace_steps)
                    and channel_veto.get("normal_sample_count")
                    == int(np.sum(~holdout) * config.normal_sample_count_per_route)
                    and isinstance(channel_veto.get("frozen_training_source_count"), int)
                    and not isinstance(channel_veto.get("frozen_training_source_count"), bool)
                    and 0 < channel_veto["frozen_training_source_count"] <= int(np.sum(~holdout))
                    and channel_veto.get("frozen_closed_loop_rollout_steps")
                    == config.normal_rollout_steps
                    and static_gain.shape == (_JOINT_COUNT,)
                    and np.array_equal(static_gain, archived_gain)
                    and failure_authority.shape == (_JOINT_COUNT,)
                    and normal_authority.shape == (_JOINT_COUNT,)
                    and np.all(np.isfinite(failure_authority))
                    and np.all(np.isfinite(normal_authority))
                    and np.all((failure_authority >= 0.0) & (failure_authority <= 1.0))
                    and np.all((normal_authority >= 0.0) & (normal_authority <= 1.0))
                    and np.allclose(
                        failure_authority,
                        archived_failure_mean_authority,
                        rtol=0.0,
                        atol=1.0e-7,
                    )
                    and np.allclose(
                        normal_authority,
                        archived_normal_mean_authority,
                        rtol=0.0,
                        atol=1.0e-7,
                    )
                    and math.isclose(
                        float(channel_veto.get("failure_ood_fraction", -1.0)),
                        float(np.mean(np.all(archived_failure_authority == 0.0, axis=-1))),
                        abs_tol=1.0e-7,
                    )
                    and math.isclose(
                        float(channel_veto.get("normal_ood_fraction", -1.0)),
                        float(np.mean(np.all(archived_normal_authority == 0.0, axis=-1))),
                        abs_tol=1.0e-7,
                    )
                    and math.isclose(
                        float(channel_veto.get("ood_radius", -1.0)),
                        float(normalized_model["channel_veto_ood_radius"][0]),
                        abs_tol=1.0e-6,
                    )
                    and mirrored_arrays_valid
                    and calibration_valid
                    and veto_aware_trigger_valid
                    and dagger_provenance_valid
                    and repair_contract_valid
                    and isinstance(channel_veto.get("hard_negative_mining"), dict)
                    and all(
                        isinstance(channel_veto.get(name), str)
                        and _HASH.fullmatch(channel_veto[name])
                        for name in hash_fields
                    )
                )
            summaries_valid = bool(
                isinstance(metrics, dict)
                and isinstance(failure_exam, dict)
                and isinstance(normal_exam, dict)
                and _validate_corrective_exam_source_diagnostics(
                    failure_exam, config=config, normal_route=False
                )
                and _validate_corrective_exam_source_diagnostics(
                    normal_exam, config=config, normal_route=True
                )
                and math.isclose(
                    float(metrics.get("holdout_increment_rmse", -1)),
                    float(np.sqrt(np.mean(np.square(prediction - target)))),
                    abs_tol=1e-7,
                )
                and math.isclose(
                    float(metrics.get("holdout_target_rms", -1)),
                    float(np.sqrt(np.mean(np.square(target)))),
                    abs_tol=1e-7,
                )
                and math.isclose(
                    float(metrics.get("normal_predicted_increment_rms", -1)),
                    normal_prediction_rms,
                    abs_tol=1e-7,
                )
                and payload.get("train_source_count") == int(np.sum(~holdout))
                and payload.get("holdout_source_count") == int(np.sum(holdout))
                and payload.get("student_development_retained")
                is bool(
                    failure_exam.get("passed") is True
                    and failure_exam.get("stability_retention_passed", True) is True
                    and normal_exam.get("passed") is True
                    and normal_exam.get("stability_retention_passed", True) is True
                    and normal_prediction_rms <= config.maximum_normal_increment_rms
                )
            )
        except (KeyError, OSError, TypeError, ValueError):
            arrays_valid = False
            summaries_valid = False
            effect_budget_valid = False
            channel_veto_valid = False
            content_hashes_valid = False
    authority_valid = bool(
        payload.get("schema_version") == "rosclaw_soccer.recovery_corrective_student_evidence.v1"
        and payload.get("config_hash") == (config.config_hash if config is not None else None)
        and payload.get("activation_ceiling") == "SIM_ONLY"
        and payload.get("hardware_authorized") is False
        and payload.get("hardware_command_sent") is False
        and payload.get("deployment_candidate") is False
        and payload.get("promotion_eligible") is False
        and payload.get("promotion_authority") == "NONE"
        and payload.get("four_gpu_training") is True
        and isinstance(payload.get("devices"), list)
        and len(payload["devices"]) == 4
        and len(set(payload["devices"])) == 4
    )
    lineage_valid = all(
        isinstance(payload.get(name), str) and _HASH.fullmatch(str(payload[name]))
        for name in (
            "teacher_report_hash",
            "teacher_report_file_hash",
            "teacher_corpus_hash",
            "failure_state_manifest_hash",
            "parent_checkpoint_hash",
            "teacher_checkpoint_hash",
            "snapshot_manifest_hash",
            "route_manifest_hash",
            "route_group_hash",
        )
    )
    if not all(
        (
            valid_hash,
            archives_valid,
            arrays_valid,
            content_hashes_valid,
            summaries_valid,
            effect_budget_valid,
            channel_veto_valid,
            authority_valid,
            lineage_valid,
        )
    ):
        raise ValueError("recovery corrective student evidence is invalid")
    return payload


def _recovery_corrective_repeat_payload(report_paths: tuple[Path, ...]) -> dict[str, Any]:
    if len(report_paths) < 2:
        raise ValueError("corrective repeat gate requires at least two reports")
    resolved = tuple(path.expanduser().resolve() for path in report_paths)
    if len(set(resolved)) != len(resolved):
        raise ValueError("corrective repeat gate reports are not unique")
    reports = tuple(validate_recovery_corrective_student_evidence(path) for path in resolved)
    content_rows: list[tuple[str, str]] = []
    for path, report in zip(resolved, reports, strict=True):
        with np.load(path.parent / str(report["corpus_archive"]), allow_pickle=False) as archive:
            corpus_content_hash = _array_mapping_content_hash(
                {name: np.array(archive[name], copy=True) for name in archive.files}
            )
        with np.load(path.parent / str(report["model_archive"]), allow_pickle=False) as archive:
            model_content_hash = _array_mapping_content_hash(
                {name: np.array(archive[name], copy=True) for name in archive.files}
            )
        content_rows.append((corpus_content_hash, model_content_hash))
    shared_fields = (
        "config_hash",
        "failure_state_manifest_hash",
        "parent_checkpoint_hash",
        "teacher_checkpoint_hash",
        "snapshot_manifest_hash",
        "route_manifest_hash",
        "route_group_hash",
    )
    if any(
        report.get(name) != reports[0].get(name) for report in reports[1:] for name in shared_fields
    ) or any(row != content_rows[0] for row in content_rows[1:]):
        raise ValueError("corrective repeat reports do not bind the same candidate")
    entries = []
    for path, report in zip(resolved, reports, strict=True):
        failure_exam = report["failure_state_paired_physics_exam"]
        normal_exam = report["normal_route_paired_physics_exam"]
        entries.append(
            {
                "student_report_path": str(path),
                "student_report_hash": report["report_hash"],
                "student_report_file_hash": hash_bytes(path.read_bytes()),
                "corpus_archive_hash": report["corpus_archive_hash"],
                "model_archive_hash": report["model_archive_hash"],
                "failure_improvement_fraction": failure_exam["mean_cost_improvement_fraction"],
                "failure_passed": failure_exam["passed"],
                "failure_directional_passed": failure_exam["directional_retention_passed"],
                "failure_stability_passed": failure_exam["stability_retention_passed"],
                "failure_finite_fraction": failure_exam["finite_fraction"],
                "normal_cost_regression_fraction": normal_exam["normal_cost_regression_fraction"],
                "normal_passed": normal_exam["passed"],
                "normal_directional_passed": normal_exam["directional_retention_passed"],
                "normal_stability_passed": normal_exam["stability_retention_passed"],
                "normal_finite_fraction": normal_exam["finite_fraction"],
                "student_development_retained": report["student_development_retained"],
            }
        )
    all_retained = all(entry["student_development_retained"] is True for entry in entries)
    return {
        "schema_version": "rosclaw_soccer.recovery_corrective_repeat_gate.v1",
        **{name: reports[0][name] for name in shared_fields},
        "corpus_content_hash": content_rows[0][0],
        "model_content_hash": content_rows[0][1],
        "repeat_count": len(entries),
        "report_entries": entries,
        "worst_failure_improvement_fraction": min(
            float(entry["failure_improvement_fraction"]) for entry in entries
        ),
        "worst_normal_cost_regression_fraction": max(
            float(entry["normal_cost_regression_fraction"]) for entry in entries
        ),
        "all_failure_directional_passed": all(
            entry["failure_directional_passed"] is True for entry in entries
        ),
        "all_failure_stability_passed": all(
            entry["failure_stability_passed"] is True for entry in entries
        ),
        "all_normal_directional_passed": all(
            entry["normal_directional_passed"] is True for entry in entries
        ),
        "all_normal_stability_passed": all(
            entry["normal_stability_passed"] is True for entry in entries
        ),
        "all_finite": all(
            entry["failure_finite_fraction"] == 1.0 and entry["normal_finite_fraction"] == 1.0
            for entry in entries
        ),
        "all_repeats_retained": all_retained,
        "repeat_gate_passed": all_retained,
        "activation_ceiling": "SIM_ONLY",
        "deployment_candidate": False,
        "promotion_eligible": False,
        "promotion_authority": "NONE",
        "hardware_authorized": False,
        "hardware_command_sent": False,
    }


def write_recovery_corrective_repeat_evidence(
    *, report_paths: tuple[Path, ...], output_path: Path
) -> dict[str, Any]:
    """Write a worst-repeat gate over content-identical corrective candidates."""

    destination = output_path.expanduser().resolve()
    if destination.exists():
        raise ValueError("corrective repeat evidence refuses to overwrite output")
    payload = _recovery_corrective_repeat_payload(report_paths)
    payload["report_hash"] = hash_json(payload)
    _atomic_json(destination, payload)
    return validate_recovery_corrective_repeat_evidence(destination)


def validate_recovery_corrective_repeat_evidence(path: Path) -> dict[str, Any]:
    """Fail closed unless every bound repeat and worst-case aggregate is exact."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("corrective repeat evidence is invalid")
    report_hash = payload.pop("report_hash", None)
    entries = payload.get("report_entries")
    entry_items = entries if isinstance(entries, list) else []
    try:
        report_paths = tuple(
            Path(str(entry["student_report_path"]))
            for entry in entry_items
            if isinstance(entry, dict)
        )
    except (KeyError, TypeError):
        report_paths = ()
    expected = _recovery_corrective_repeat_payload(report_paths)
    if (
        report_hash != hash_json(payload)
        or payload != expected
        or payload.get("activation_ceiling") != "SIM_ONLY"
        or payload.get("deployment_candidate") is not False
        or payload.get("promotion_eligible") is not False
        or payload.get("promotion_authority") != "NONE"
        or payload.get("hardware_authorized") is not False
        or payload.get("hardware_command_sent") is not False
    ):
        raise ValueError("corrective repeat evidence is invalid")
    payload["report_hash"] = report_hash
    return payload


__all__ = [
    "CorrectiveTemporalLeaseConfig",
    "RecoveryCorrectiveStudentConfig",
    "attach_corrective_temporal_lease",
    "attach_corrective_veto_aware_temporal_trigger",
    "calibrate_corrective_channel_veto",
    "calibrate_corrective_confidence_gate",
    "corrective_stability_retention",
    "derive_corrective_channel_gain",
    "derive_corrective_effect_budget_gain",
    "fit_corrective_channel_veto",
    "fit_corrective_confidence_gate",
    "fit_corrective_historical_veto_gate",
    "initial_corrective_temporal_gate_state",
    "mine_corrective_temporal_hard_negatives",
    "mix_corrective_cross_domain_normal_replay",
    "mix_corrective_normal_dagger_replay",
    "mix_corrective_training_normal_sources",
    "predict_corrective_confidence_numpy",
    "predict_corrective_channel_veto_numpy",
    "predict_corrective_primary_confidence_numpy",
    "predict_corrective_raw_numpy",
    "predict_corrective_student_numpy",
    "predict_corrective_temporal_trigger_confidence_numpy",
    "predict_corrective_temporal_sequence_numpy",
    "step_corrective_temporal_gate_numpy",
    "stratified_source_split",
    "validate_recovery_corrective_student_evidence",
    "validate_recovery_corrective_repeat_evidence",
    "write_recovery_corrective_repeat_evidence",
    "write_recovery_corrective_student_evidence",
]
