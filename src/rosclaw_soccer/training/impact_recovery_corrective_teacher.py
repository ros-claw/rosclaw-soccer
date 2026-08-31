"""Four-GPU counterfactual corrective teacher for impact-recovery failures."""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
from brax.envs.wrappers import training as brax_wrappers

from rosclaw_soccer.providers.g1.asset_qualification import g1_body_hash
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.impact_recovery_curriculum import (
    validate_impact_recovery_curriculum,
)
from rosclaw_soccer.training.impact_recovery_mjx import (
    ImpactRecoveryMJXConfig,
    ImpactRecoveryMJXEnv,
    ImpactRecoveryMJXEvaluationConfig,
)
from rosclaw_soccer.training.recovery_mjx import _atomic_json, compiled_mujoco_model_contract

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_JOINT_COUNT = 29
_EFFECT_NAMES = (
    "backward_speed_cost",
    "lateral_speed_cost",
    "angular_speed_cost",
    "height_deficit",
    "upright_deficit",
    "support_deficit",
)
_ARCHIVE_NAMES = (
    "actor_observation",
    "teacher_plan",
    "corrective_action",
    "baseline_cost",
    "teacher_cost",
    "cost_improvement_fraction",
    "teacher_accepted",
    "finite_rollout",
    "curriculum_index",
    "elapsed_since_contact_sec",
    "baseline_effect_metrics",
    "teacher_effect_metrics",
)
_STABILITY_ARCHIVE_NAMES = _ARCHIVE_NAMES + ("maximum_stability_regression",)
_ROBUST_ARCHIVE_NAMES = _STABILITY_ARCHIVE_NAMES + (
    "baseline_success",
    "teacher_success",
    "baseline_maximum_stable_streak",
    "teacher_maximum_stable_streak",
)


@dataclass(frozen=True)
class ImpactRecoveryCorrectiveTeacherConfig:
    """Bounded CEM search that produces labels, never deployment authority."""

    state_count: int = 16
    robust_variants_per_state: int = 1
    robust_worst_case_weight: float = 0.0
    horizon_steps: int = 40
    action_chunk_steps: int = 5
    plan_knot_count: int = 0
    candidate_count: int = 128
    elite_fraction: float = 0.125
    cem_iterations: int = 4
    initial_action_std: float = 0.20
    minimum_action_std: float = 0.02
    maximum_action: float = 0.60
    action_magnitude_cost_weight: float = 0.02
    action_slew_cost_weight: float = 0.03
    minimum_cost_improvement_fraction: float = 0.02
    maximum_stability_deficit_regression: float = 0.01
    objective_mode: Literal["AVERAGE_STABILITY", "SUCCESSOR_STREAK"] = "AVERAGE_STABILITY"
    ready_fraction_deficit_weight: float = 2.0
    stable_streak_deficit_weight: float = 4.0
    success_deficit_weight: float = 10.0
    minimum_accepted_fraction: float = 0.25
    minimum_teacher_action_rms: float = 0.001
    minimum_successor_success_fraction: float = 0.0
    required_gpu_count: int = 4
    random_seed: int = 57_117
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.impact_recovery_corrective_teacher_config.v1"

    def __post_init__(self) -> None:
        finite = (
            self.elite_fraction,
            self.robust_worst_case_weight,
            self.initial_action_std,
            self.minimum_action_std,
            self.maximum_action,
            self.action_magnitude_cost_weight,
            self.action_slew_cost_weight,
            self.minimum_cost_improvement_fraction,
            self.maximum_stability_deficit_regression,
            self.ready_fraction_deficit_weight,
            self.stable_streak_deficit_weight,
            self.success_deficit_weight,
            self.minimum_accepted_fraction,
            self.minimum_teacher_action_rms,
            self.minimum_successor_success_fraction,
        )
        if (
            not 4 <= self.state_count <= 384
            or self.state_count % self.required_gpu_count
            or not 1 <= self.robust_variants_per_state <= 8
            or self.state_count * self.robust_variants_per_state > 384
            or not 0.0 <= self.robust_worst_case_weight <= 1.0
            or not 10 <= self.horizon_steps <= 100
            or not 1 <= self.action_chunk_steps <= self.horizon_steps
            or self.horizon_steps % self.action_chunk_steps
            or not (
                self.plan_knot_count == 0 or 2 <= self.plan_knot_count <= self.action_chunk_count
            )
            or not 64 <= self.candidate_count <= 512
            or self.candidate_count % 4
            or not 0.02 <= self.elite_fraction <= 0.5
            or int(self.candidate_count * self.elite_fraction) < 2
            or not 1 <= self.cem_iterations <= 8
            or any(not math.isfinite(value) for value in finite)
            or not 0.01 <= self.minimum_action_std <= self.initial_action_std <= 0.5
            or not self.initial_action_std <= self.maximum_action <= 1.0
            or not 0.0 <= self.action_magnitude_cost_weight <= 1.0
            or not 0.0 <= self.action_slew_cost_weight <= 1.0
            or not 0.0 <= self.minimum_cost_improvement_fraction <= 0.5
            or not 0.0 <= self.maximum_stability_deficit_regression <= 0.25
            or self.objective_mode not in {"AVERAGE_STABILITY", "SUCCESSOR_STREAK"}
            or not 0.0 <= self.ready_fraction_deficit_weight <= 20.0
            or not 0.0 <= self.stable_streak_deficit_weight <= 20.0
            or not 0.0 <= self.success_deficit_weight <= 100.0
            or not 0.0 <= self.minimum_accepted_fraction <= 1.0
            or not 0.0 < self.minimum_teacher_action_rms <= self.maximum_action
            or not 0.0 <= self.minimum_successor_success_fraction <= 1.0
            or self.required_gpu_count != 4
            or not 0 <= self.random_seed < 2**31
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("impact-recovery corrective teacher config is invalid")

    @property
    def action_chunk_count(self) -> int:
        return self.horizon_steps // self.action_chunk_steps

    @property
    def elite_count(self) -> int:
        return int(self.candidate_count * self.elite_fraction)

    @property
    def search_knot_count(self) -> int:
        return self.plan_knot_count or self.action_chunk_count

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


def _pseudo_huber(value: jax.Array, delta: float = 0.25) -> jax.Array:
    scaled = value / delta
    return delta * delta * (jnp.sqrt(1.0 + jnp.square(scaled)) - 1.0)


def _repeat_state_candidates(state: Any, candidate_count: int) -> Any:
    return jax.tree_util.tree_map(
        lambda value: jnp.repeat(value, candidate_count, axis=0),
        state,
    )


def _aggregate_robust_variants(
    *,
    cost: jax.Array,
    effect: jax.Array,
    finite: jax.Array,
    success: jax.Array,
    maximum_streak: jax.Array,
    state_count: int,
    config: ImpactRecoveryCorrectiveTeacherConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Reduce perturbations conservatively while retaining candidate axes."""

    variant_count = config.robust_variants_per_state
    candidate_count = cost.size // (state_count * variant_count)
    variant_cost = cost.reshape((state_count, variant_count, candidate_count))
    mean_cost = jnp.mean(variant_cost, axis=1)
    worst_cost = jnp.max(variant_cost, axis=1)
    robust_cost = (
        1.0 - config.robust_worst_case_weight
    ) * mean_cost + config.robust_worst_case_weight * worst_cost
    variant_effect = effect.reshape(
        (state_count, variant_count, candidate_count, len(_EFFECT_NAMES))
    )
    stability_regression = jnp.max(
        variant_effect[..., 3:5] - variant_effect[:, :, :1, 3:5],
        axis=(1, 3),
    )
    return (
        robust_cost,
        jnp.max(variant_effect, axis=1),
        jnp.all(finite.reshape((state_count, variant_count, candidate_count)), axis=1),
        jnp.all(success.reshape((state_count, variant_count, candidate_count)), axis=1),
        jnp.min(maximum_streak.reshape((state_count, variant_count, candidate_count)), axis=1),
        stability_regression,
    )


def _selected_reset_keys(
    *,
    rng: jax.Array,
    state_count: int,
    acquisition_indexes: np.ndarray[Any, Any],
    elapsed: np.ndarray[Any, Any],
    variants_per_state: int = 1,
) -> jax.Array:
    """Find unique failure rows and grouped perturbation keys for each row."""

    if not 1 <= variants_per_state <= 8:
        raise ValueError("impact-recovery reset variants are invalid")

    pool_rng, permutation_rng = jax.random.split(rng)
    pool = jax.random.split(pool_rng, max(4_096, state_count * variants_per_state * 128))
    acquisition = jnp.asarray(acquisition_indexes, dtype=jnp.int32)

    def selected_index(reset_key: jax.Array) -> jax.Array:
        _, index_rng, _, _ = jax.random.split(reset_key, 4)
        _, _, acquisition_rng = jax.random.split(index_rng, 3)
        offset = jax.random.randint(acquisition_rng, (), 0, acquisition.shape[0])
        return acquisition[offset]

    indexes = np.asarray(jax.jit(jax.vmap(selected_index))(pool), dtype=np.int32)
    bins = np.floor(np.asarray(elapsed, dtype=np.float64)).astype(np.int32)
    bin_order = sorted({int(bins[index]) for index in acquisition_indexes.tolist()})
    selected_positions: list[int] = []
    selected_indexes: set[int] = set()
    for target_bin in (bin_order * state_count)[:state_count]:
        for position, index in enumerate(indexes.tolist()):
            if (
                position not in selected_positions
                and index not in selected_indexes
                and int(bins[index]) == target_bin
            ):
                selected_positions.append(position)
                selected_indexes.add(index)
                break
    if len(selected_positions) < state_count:
        for position, index in enumerate(indexes.tolist()):
            if index not in selected_indexes:
                selected_positions.append(position)
                selected_indexes.add(index)
            if len(selected_positions) == state_count:
                break
    if len(selected_positions) != state_count:
        raise RuntimeError("impact-recovery corrective reset-key pool is incomplete")
    base_positions = np.asarray(
        jax.random.permutation(
            permutation_rng,
            jnp.asarray(selected_positions, dtype=jnp.int32),
            axis=0,
            independent=False,
        ),
        dtype=np.int32,
    )
    grouped_positions: list[int] = []
    used_positions: set[int] = set()
    for base_position in base_positions.tolist():
        target_index = int(indexes[base_position])
        matches = [
            position
            for position, index in enumerate(indexes.tolist())
            if index == target_index and position not in used_positions
        ][:variants_per_state]
        if len(matches) != variants_per_state:
            raise RuntimeError("impact-recovery reset-key variants are incomplete")
        grouped_positions.extend(matches)
        used_positions.update(matches)
    return pool[jnp.asarray(grouped_positions, dtype=jnp.int32)]


def _make_plan_rollout(wrapped: Any, config: ImpactRecoveryCorrectiveTeacherConfig) -> Any:
    """Build one sharded counterfactual rollout function."""

    def rollout_device(
        initial_state: Any,
        plans: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        state_count, candidate_count = plans.shape[:2]
        variant_count = config.robust_variants_per_state
        rollout_state_count = state_count * variant_count
        variant_plans = jnp.repeat(plans, variant_count, axis=0)
        flat_plans = variant_plans.reshape(
            (rollout_state_count * candidate_count, config.action_chunk_count, _JOINT_COUNT)
        )
        state = _repeat_state_candidates(initial_state, candidate_count)
        effect = jnp.zeros((rollout_state_count * candidate_count, len(_EFFECT_NAMES)), jnp.float32)
        action_cost = jnp.zeros((rollout_state_count * candidate_count,), jnp.float32)
        slew_cost = jnp.zeros_like(action_cost)
        finite = jnp.ones((rollout_state_count * candidate_count,), jnp.bool_)
        previous = jnp.zeros((rollout_state_count * candidate_count, _JOINT_COUNT), jnp.float32)
        ready_deficit = jnp.zeros_like(action_cost)
        maximum_streak = jnp.zeros_like(action_cost)
        ever_success = jnp.zeros((rollout_state_count * candidate_count,), jnp.bool_)

        def step(
            carry: tuple[
                Any,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
            ],
            step_index: jax.Array,
        ) -> tuple[
            tuple[
                Any,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
            ],
            None,
        ]:
            (
                current,
                total_effect,
                total_action,
                total_slew,
                finite_so_far,
                prior,
                ready_total,
                streak_max,
                succeeded,
            ) = carry
            chunk = step_index // config.action_chunk_steps
            action = flat_plans[:, chunk]
            next_state = wrapped.step(current, action)
            metrics = next_state.metrics
            step_effect = jnp.stack(
                (
                    _pseudo_huber(metrics["root_body_backward_speed"] / 0.5),
                    _pseudo_huber(metrics["root_body_lateral_speed"] / 0.5),
                    _pseudo_huber(metrics["root_angular_speed"] / 1.0),
                    jnp.square(jnp.maximum(0.70 - metrics["pelvis_height"], 0.0) / 0.15),
                    jnp.square(jnp.maximum(0.95 - metrics["upright"], 0.0) / 0.20),
                    1.0 - metrics["bilateral_support"],
                ),
                axis=-1,
            )
            current_finite = jnp.all(
                jnp.isfinite(next_state.pipeline_state.qpos), axis=-1
            ) & jnp.all(jnp.isfinite(next_state.pipeline_state.qvel), axis=-1)
            return (
                next_state,
                total_effect + jnp.nan_to_num(step_effect, nan=100.0, posinf=100.0),
                total_action + jnp.mean(jnp.square(action), axis=-1),
                total_slew + jnp.mean(jnp.square(action - prior), axis=-1),
                finite_so_far & current_finite,
                action,
                ready_total + (1.0 - metrics["ready"]),
                jnp.maximum(streak_max, metrics["maximum_stable_streak"]),
                succeeded | (metrics["success"] > 0.0),
            ), None

        final, _ = jax.lax.scan(
            step,
            (
                state,
                effect,
                action_cost,
                slew_cost,
                finite,
                previous,
                ready_deficit,
                maximum_streak,
                ever_success,
            ),
            jnp.arange(config.horizon_steps, dtype=jnp.int32),
        )
        mean_effect = final[1] / config.horizon_steps
        weights = jnp.asarray((3.0, 1.0, 1.0, 2.0, 2.0, 0.5), jnp.float32)
        cost = (
            jnp.sum(mean_effect * weights, axis=-1)
            + config.action_magnitude_cost_weight * final[2] / config.horizon_steps
            + config.action_slew_cost_weight * final[3] / config.horizon_steps
            + (~final[4]).astype(jnp.float32) * 100.0
        )
        if config.objective_mode == "SUCCESSOR_STREAK":
            cost = (
                cost
                + config.ready_fraction_deficit_weight * final[6] / config.horizon_steps
                + config.stable_streak_deficit_weight * (1.0 - jnp.clip(final[7] / 25.0, 0.0, 1.0))
                + config.success_deficit_weight * (~final[8]).astype(jnp.float32)
            )
        return _aggregate_robust_variants(
            cost=cost,
            effect=mean_effect,
            finite=final[4],
            success=final[8],
            maximum_streak=final[7],
            state_count=state_count,
            config=config,
        )

    return jax.pmap(rollout_device, axis_name="impact_recovery_corrective_teacher_devices")


def _search(
    *,
    rollout: Any,
    initial_state: Any,
    config: ImpactRecoveryCorrectiveTeacherConfig,
) -> tuple[np.ndarray[Any, Any], ...]:
    device_count = config.required_gpu_count
    states_per_device = config.state_count // device_count
    plan_shape = (
        device_count,
        states_per_device,
        config.candidate_count,
        config.search_knot_count,
        _JOINT_COUNT,
    )
    mean = np.zeros(plan_shape[:2] + plan_shape[3:], dtype=np.float32)
    std = np.full_like(mean, config.initial_action_std)
    random = np.random.default_rng(config.random_seed)
    best_cost = np.full(plan_shape[:2], np.inf, dtype=np.float64)
    best_plan = np.zeros(
        plan_shape[:2] + (config.action_chunk_count, _JOINT_COUNT), dtype=np.float32
    )
    best_effect = np.zeros(plan_shape[:2] + (len(_EFFECT_NAMES),), dtype=np.float64)
    best_finite = np.zeros(plan_shape[:2], dtype=np.bool_)
    best_success = np.zeros(plan_shape[:2], dtype=np.bool_)
    best_streak = np.zeros(plan_shape[:2], dtype=np.float64)
    best_stability_regression = np.full(plan_shape[:2], np.inf, dtype=np.float64)
    baseline_cost = np.zeros(plan_shape[:2], dtype=np.float64)
    baseline_effect = np.zeros_like(best_effect)
    baseline_success = np.zeros(plan_shape[:2], dtype=np.bool_)
    baseline_streak = np.zeros(plan_shape[:2], dtype=np.float64)
    for iteration in range(config.cem_iterations):
        noise = random.standard_normal(plan_shape).astype(np.float32)
        plans = np.clip(
            mean[:, :, None] + std[:, :, None] * noise,
            -config.maximum_action,
            config.maximum_action,
        )
        plans[:, :, 0] = 0.0
        plans[:, :, 1] = mean
        if config.search_knot_count == config.action_chunk_count:
            expanded_plans = plans
        else:
            positions = np.linspace(
                0.0,
                config.search_knot_count - 1,
                config.action_chunk_count,
                dtype=np.float64,
            )
            left = np.floor(positions).astype(np.int32)
            right = np.minimum(left + 1, config.search_knot_count - 1)
            fraction = (positions - left).astype(np.float32)
            expanded_plans = (
                plans[..., left, :] * (1.0 - fraction)[None, None, None, :, None]
                + plans[..., right, :] * fraction[None, None, None, :, None]
            )
        (
            costs_device,
            effects_device,
            finite_device,
            success_device,
            streak_device,
            stability_regression_device,
        ) = rollout(initial_state, jnp.asarray(expanded_plans))
        costs = np.asarray(costs_device, dtype=np.float64)
        effects = np.asarray(effects_device, dtype=np.float64)
        finite = np.asarray(finite_device, dtype=np.bool_)
        success = np.asarray(success_device, dtype=np.bool_)
        streak = np.asarray(streak_device, dtype=np.float64)
        stability_regression = np.asarray(stability_regression_device, dtype=np.float64)
        if iteration == 0:
            baseline_cost = np.array(costs[:, :, 0], copy=True)
            baseline_effect = np.array(effects[:, :, 0], copy=True)
            baseline_success = np.array(success[:, :, 0], copy=True)
            baseline_streak = np.array(streak[:, :, 0], copy=True)
        stability_ok = stability_regression <= config.maximum_stability_deficit_regression
        constrained = np.where(stability_ok & finite, costs, costs + 100.0)
        selected_index = np.argmin(constrained, axis=2)
        selected_cost = np.take_along_axis(costs, selected_index[:, :, None], axis=2)[..., 0]
        selected_plan = np.take_along_axis(
            expanded_plans, selected_index[:, :, None, None, None], axis=2
        )[:, :, 0]
        selected_effect = np.take_along_axis(effects, selected_index[:, :, None, None], axis=2)[
            :, :, 0
        ]
        selected_finite = np.take_along_axis(finite, selected_index[:, :, None], axis=2)[:, :, 0]
        selected_success = np.take_along_axis(success, selected_index[:, :, None], axis=2)[:, :, 0]
        selected_streak = np.take_along_axis(streak, selected_index[:, :, None], axis=2)[:, :, 0]
        selected_stability_regression = np.take_along_axis(
            stability_regression, selected_index[:, :, None], axis=2
        )[:, :, 0]
        improved = selected_cost < best_cost
        best_cost = np.where(improved, selected_cost, best_cost)
        best_plan = np.where(improved[..., None, None], selected_plan, best_plan)
        best_effect = np.where(improved[..., None], selected_effect, best_effect)
        best_finite = np.where(improved, selected_finite, best_finite)
        best_success = np.where(improved, selected_success, best_success)
        best_streak = np.where(improved, selected_streak, best_streak)
        best_stability_regression = np.where(
            improved, selected_stability_regression, best_stability_regression
        )
        elite_index = np.argsort(constrained, axis=2)[..., : config.elite_count]
        elites = np.take_along_axis(plans, elite_index[..., None, None], axis=2)
        mean = np.asarray(np.mean(elites, axis=2, dtype=np.float64), dtype=np.float32)
        std = np.maximum(
            np.std(elites, axis=2, dtype=np.float64), config.minimum_action_std
        ).astype(np.float32)
    return tuple(
        value.reshape((config.state_count,) + value.shape[2:])
        for value in (
            best_plan,
            best_cost,
            best_effect,
            best_finite,
            baseline_cost,
            baseline_effect,
            best_success,
            best_streak,
            baseline_success,
            baseline_streak,
            best_stability_regression,
        )
    )


def _write_evidence(
    *,
    output_dir: Path,
    config: ImpactRecoveryCorrectiveTeacherConfig,
    arrays: dict[str, np.ndarray[Any, Any]],
    lineage: dict[str, str],
    devices: tuple[str, ...],
    compiled_model: dict[str, Any],
    objective_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    destination = output_dir.expanduser().resolve()
    if destination.exists() or frozenset(arrays) not in {
        frozenset(_ARCHIVE_NAMES),
        frozenset(_STABILITY_ARCHIVE_NAMES),
        frozenset(_ROBUST_ARCHIVE_NAMES),
    }:
        raise ValueError("impact-recovery corrective evidence output is invalid")
    if set(lineage) != {
        "curriculum_manifest_hash",
        "curriculum_manifest_file_hash",
        "curriculum_archive_hash",
        "body_hash",
        "training_model_hash",
    } or any(_SHA256.fullmatch(value) is None for value in lineage.values()):
        raise ValueError("impact-recovery corrective evidence lineage is invalid")
    state_count = config.state_count
    if (
        arrays["actor_observation"].shape[0] != state_count
        or arrays["teacher_plan"].shape != (state_count, config.action_chunk_count, _JOINT_COUNT)
        or arrays["corrective_action"].shape != (state_count, _JOINT_COUNT)
        or arrays["baseline_effect_metrics"].shape != (state_count, len(_EFFECT_NAMES))
        or arrays["teacher_effect_metrics"].shape != (state_count, len(_EFFECT_NAMES))
        or any(
            arrays[name].shape != (state_count,)
            for name in (
                "baseline_cost",
                "teacher_cost",
                "cost_improvement_fraction",
                "teacher_accepted",
                "finite_rollout",
                "curriculum_index",
                "elapsed_since_contact_sec",
            )
        )
        or (
            "maximum_stability_regression" in arrays
            and arrays["maximum_stability_regression"].shape != (state_count,)
        )
        or any(
            arrays[name].shape != (state_count,)
            for name in (
                "baseline_success",
                "teacher_success",
                "baseline_maximum_stable_streak",
                "teacher_maximum_stable_streak",
            )
            if name in arrays
        )
        or any(
            not np.all(np.isfinite(value))
            for name, value in arrays.items()
            if name not in {"teacher_accepted", "finite_rollout"}
        )
        or not np.allclose(arrays["corrective_action"], arrays["teacher_plan"][:, 0])
        or np.any(np.abs(arrays["teacher_plan"]) > config.maximum_action + 1.0e-6)
    ):
        raise ValueError("impact-recovery corrective evidence arrays are invalid")
    improvement = (arrays["baseline_cost"] - arrays["teacher_cost"]) / np.maximum(
        np.abs(arrays["baseline_cost"]), 1.0e-12
    )
    stability_ok = (
        arrays["maximum_stability_regression"] <= config.maximum_stability_deficit_regression
        if "maximum_stability_regression" in arrays
        else np.all(
            arrays["teacher_effect_metrics"][:, 3:5]
            <= arrays["baseline_effect_metrics"][:, 3:5]
            + config.maximum_stability_deficit_regression,
            axis=1,
        )
    )
    accepted = (
        arrays["finite_rollout"].astype(np.bool_)
        & stability_ok
        & (improvement >= config.minimum_cost_improvement_fraction)
    )
    if not np.allclose(arrays["cost_improvement_fraction"], improvement) or not np.array_equal(
        arrays["teacher_accepted"], accepted
    ):
        raise ValueError("impact-recovery corrective evidence labels changed")
    destination.mkdir(parents=True)
    archive_path = destination / "corrective-teacher-corpus.npz"
    temporary = destination / ".corrective-teacher-corpus.npz.tmp"
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)  # type: ignore[arg-type]
    os.replace(temporary, archive_path)
    accepted_count = int(np.sum(accepted))
    action_rms = float(np.sqrt(np.mean(np.square(arrays["corrective_action"]))))
    if frozenset(arrays) == frozenset(_ROBUST_ARCHIVE_NAMES):
        derived_objective_diagnostics = {
            "robust_variants_per_state": config.robust_variants_per_state,
            "robust_worst_case_weight": config.robust_worst_case_weight,
            "baseline_success_count": int(np.sum(arrays["baseline_success"])),
            "teacher_success_count": int(np.sum(arrays["teacher_success"])),
            "baseline_median_maximum_stable_streak": float(
                np.median(arrays["baseline_maximum_stable_streak"])
            ),
            "teacher_median_maximum_stable_streak": float(
                np.median(arrays["teacher_maximum_stable_streak"])
            ),
            "teacher_maximum_stable_streak": float(np.max(arrays["teacher_maximum_stable_streak"])),
        }
        if objective_diagnostics != derived_objective_diagnostics:
            raise ValueError("impact-recovery corrective objective diagnostics changed")
    teacher_success_count = (
        int(objective_diagnostics.get("teacher_success_count", 0))
        if isinstance(objective_diagnostics, dict)
        else 0
    )
    successor_success_fraction = teacher_success_count / state_count
    successor_gate = bool(
        config.objective_mode != "SUCCESSOR_STREAK"
        or successor_success_fraction >= config.minimum_successor_success_fraction
    )
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.impact_recovery_corrective_teacher.v1",
        "config": asdict(config),
        "config_hash": config.config_hash,
        **lineage,
        "compiled_model_contract": compiled_model,
        "rollout_backend": "MUJOCO_MJX",
        "search_algorithm": "SHORT_HORIZON_CHUNKED_CEM_WITH_ZERO_RESIDUAL_CONTROL",
        "effect_metric_names": list(_EFFECT_NAMES),
        "actor_observation": "DEPLOYABLE_PROPRIOCEPTION_HISTORY_AND_DYNAMIC_GAINS",
        "actor_observation_dim": int(arrays["actor_observation"].shape[1]),
        "devices": list(devices),
        "device_count": len(devices),
        "all_devices_used": len(devices) == config.required_gpu_count,
        "state_count": state_count,
        "unique_curriculum_state_count": int(np.unique(arrays["curriculum_index"]).size),
        "corpus_archive": archive_path.name,
        "corpus_archive_hash": hash_bytes(archive_path.read_bytes()),
        "accepted_count": accepted_count,
        "accepted_fraction": accepted_count / state_count,
        "median_cost_improvement_fraction": float(np.median(arrays["cost_improvement_fraction"])),
        "mean_corrective_action_rms": action_rms,
        "successor_success_fraction": successor_success_fraction,
        "supervised_warm_start_eligible": bool(
            accepted_count / state_count >= config.minimum_accepted_fraction
            and action_rms >= config.minimum_teacher_action_rms
            and successor_gate
        ),
        "objective_diagnostics": objective_diagnostics,
        "counterexamples_preserved": True,
        "deployment_candidate": False,
        "promotion_eligible": False,
        "promotion_authority": "NONE",
        "claim_boundary": "Counterfactual SIM labels only; no policy or full-chain claim",
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    report_path = destination / "teacher-report.json"
    _atomic_json(report_path, report)
    return validate_impact_recovery_corrective_teacher(report_path)


def validate_impact_recovery_corrective_teacher(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("impact-recovery corrective teacher report is invalid")
    report = cast(dict[str, Any], payload)
    declared = report.pop("report_hash", None)
    try:
        config_value = report.get("config")
        if not isinstance(config_value, dict):
            raise ValueError("impact-recovery corrective teacher config is missing")
        config = ImpactRecoveryCorrectiveTeacherConfig(**config_value)
        archive_path = resolved.parent / str(report.get("corpus_archive", ""))
        with np.load(archive_path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
        if frozenset(arrays) not in {
            frozenset(_ARCHIVE_NAMES),
            frozenset(_STABILITY_ARCHIVE_NAMES),
            frozenset(_ROBUST_ARCHIVE_NAMES),
        }:
            raise ValueError("impact-recovery corrective teacher archive changed")
        improvement = (arrays["baseline_cost"] - arrays["teacher_cost"]) / np.maximum(
            np.abs(arrays["baseline_cost"]), 1.0e-12
        )
        if "maximum_stability_regression" in arrays and arrays[
            "maximum_stability_regression"
        ].shape != (config.state_count,):
            raise ValueError("impact-recovery corrective stability evidence changed")
        if frozenset(arrays) == frozenset(_ROBUST_ARCHIVE_NAMES):
            robust_names = (
                "baseline_success",
                "teacher_success",
                "baseline_maximum_stable_streak",
                "teacher_maximum_stable_streak",
            )
            if any(arrays[name].shape != (config.state_count,) for name in robust_names):
                raise ValueError("impact-recovery corrective objective evidence changed")
            expected_objective_diagnostics: dict[str, Any] | None = {
                "robust_variants_per_state": config.robust_variants_per_state,
                "robust_worst_case_weight": config.robust_worst_case_weight,
                "baseline_success_count": int(np.sum(arrays["baseline_success"])),
                "teacher_success_count": int(np.sum(arrays["teacher_success"])),
                "baseline_median_maximum_stable_streak": float(
                    np.median(arrays["baseline_maximum_stable_streak"])
                ),
                "teacher_median_maximum_stable_streak": float(
                    np.median(arrays["teacher_maximum_stable_streak"])
                ),
                "teacher_maximum_stable_streak": float(
                    np.max(arrays["teacher_maximum_stable_streak"])
                ),
            }
        else:
            expected_objective_diagnostics = report.get("objective_diagnostics")
        stability_ok = (
            arrays["maximum_stability_regression"] <= config.maximum_stability_deficit_regression
            if "maximum_stability_regression" in arrays
            else np.all(
                arrays["teacher_effect_metrics"][:, 3:5]
                <= arrays["baseline_effect_metrics"][:, 3:5]
                + config.maximum_stability_deficit_regression,
                axis=1,
            )
        )
        accepted = (
            arrays["finite_rollout"].astype(np.bool_)
            & stability_ok
            & (improvement >= config.minimum_cost_improvement_fraction)
        )
        accepted_count = int(np.sum(accepted))
        action_rms = float(np.sqrt(np.mean(np.square(arrays["corrective_action"]))))
        teacher_success_count = (
            int(expected_objective_diagnostics.get("teacher_success_count", 0))
            if isinstance(expected_objective_diagnostics, dict)
            else 0
        )
        successor_success_fraction = teacher_success_count / config.state_count
        expected_warm_start_eligible = bool(
            accepted_count / config.state_count >= config.minimum_accepted_fraction
            and action_rms >= config.minimum_teacher_action_rms
            and (
                config.objective_mode != "SUCCESSOR_STREAK"
                or successor_success_fraction >= config.minimum_successor_success_fraction
            )
        )
        if (
            report.get("schema_version") != "rosclaw_soccer.impact_recovery_corrective_teacher.v1"
            or declared != hash_json(report)
            or report.get("config_hash") != hash_json(config_value)
            or report.get("corpus_archive") != "corrective-teacher-corpus.npz"
            or report.get("corpus_archive_hash") != hash_bytes(archive_path.read_bytes())
            or report.get("state_count") != config.state_count
            or report.get("accepted_count") != accepted_count
            or report.get("accepted_fraction") != accepted_count / config.state_count
            or report.get("mean_corrective_action_rms") != action_rms
            or report.get("objective_diagnostics") != expected_objective_diagnostics
            or report.get("supervised_warm_start_eligible") is not expected_warm_start_eligible
            or (
                "successor_success_fraction" in report
                and report.get("successor_success_fraction") != successor_success_fraction
            )
            or not np.allclose(arrays["cost_improvement_fraction"], improvement)
            or not np.array_equal(arrays["teacher_accepted"], accepted)
            or report.get("rollout_backend") != "MUJOCO_MJX"
            or report.get("device_count") != config.required_gpu_count
            or report.get("all_devices_used") is not True
            or report.get("deployment_candidate") is not False
            or report.get("promotion_eligible") is not False
            or report.get("promotion_authority") != "NONE"
            or report.get("activation_ceiling") != "SIM_ONLY"
            or report.get("hardware_authorized") is not False
            or report.get("hardware_command_sent") is not False
            or any(
                _SHA256.fullmatch(str(report.get(name, ""))) is None
                for name in (
                    "curriculum_manifest_hash",
                    "curriculum_manifest_file_hash",
                    "curriculum_archive_hash",
                    "body_hash",
                    "training_model_hash",
                )
            )
        ):
            raise ValueError("impact-recovery corrective teacher integrity changed")
        return report
    finally:
        if declared is not None:
            report["report_hash"] = declared


def run_impact_recovery_corrective_teacher(
    *,
    asset_root: Path,
    curriculum_manifest_path: Path,
    output_dir: Path,
    source_checkout_path: Path,
    config: ImpactRecoveryCorrectiveTeacherConfig | None = None,
) -> dict[str, Any]:
    """Search four-GPU short-horizon corrections from real failed resets."""

    active = config or ImpactRecoveryCorrectiveTeacherConfig()
    root = asset_root.expanduser().resolve()
    curriculum_path = curriculum_manifest_path.expanduser().resolve()
    destination = output_dir.expanduser().resolve()
    checkout = source_checkout_path.expanduser().resolve()
    model_path = root / "g1_description" / "g1_liao.xml"
    if not model_path.is_file() or not curriculum_path.is_file():
        raise FileNotFoundError("impact-recovery corrective teacher inputs are incomplete")
    if destination.exists() or destination == checkout or checkout in destination.parents:
        raise ValueError("impact-recovery corrective teacher output must be new and external")
    devices = tuple(jax.devices())
    if len(devices) != active.required_gpu_count or any(
        getattr(device, "platform", "") != "gpu" for device in devices
    ):
        raise RuntimeError("impact-recovery corrective teacher requires exactly four GPUs")
    manifest = validate_impact_recovery_curriculum(curriculum_path)
    archive_path = curriculum_path.parent / str(manifest["archive"])
    with np.load(archive_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    if (
        manifest.get("schema_version")
        not in {
            "rosclaw_soccer.impact_recovery_curriculum.v4",
            "rosclaw_soccer.impact_recovery_curriculum.v5",
            "rosclaw_soccer.impact_recovery_curriculum.v6",
        }
        or manifest.get("training_model_hash") != hash_bytes(model_path.read_bytes())
        or manifest.get("body_hash") != g1_body_hash(root)
    ):
        raise ValueError("impact-recovery corrective teacher body binding changed")
    controller = ImpactRecoveryMJXConfig(
        retention_memory_mode="DIRECT_REPLAY",
        gain_memory_mode="DYNAMIC",
    )
    environment = ImpactRecoveryMJXEnv(
        model_path=model_path,
        curriculum_arrays=arrays,
        desired_heading_rad=float(cast(dict[str, Any], manifest["config"])["desired_heading_rad"]),
        reset_population="ACQUISITION",
        config=controller,
    )
    wrapped = brax_wrappers.EpisodeWrapper(
        brax_wrappers.VmapWrapper(environment),
        episode_length=active.horizon_steps + 1,
        action_repeat=1,
    )
    acquisition_indexes = np.flatnonzero(~arrays["source_succeeded"].astype(np.bool_))
    if acquisition_indexes.size < active.state_count:
        raise ValueError("impact-recovery corrective teacher has too few unique failures")
    reset_rng = jax.random.PRNGKey(active.random_seed)
    reset_keys = _selected_reset_keys(
        rng=reset_rng,
        state_count=active.state_count,
        acquisition_indexes=acquisition_indexes,
        elapsed=arrays["elapsed_since_contact_sec"],
        variants_per_state=active.robust_variants_per_state,
    ).reshape(
        (
            active.required_gpu_count,
            active.state_count // active.required_gpu_count * active.robust_variants_per_state,
            2,
        )
    )
    initial_state = jax.pmap(wrapped.reset)(reset_keys)
    rollout = _make_plan_rollout(wrapped, active)
    (
        teacher_plan,
        teacher_cost,
        teacher_effect,
        finite,
        baseline_cost,
        baseline_effect,
        teacher_success,
        teacher_streak,
        baseline_success,
        baseline_streak,
        maximum_stability_regression,
    ) = _search(
        rollout=rollout,
        initial_state=initial_state,
        config=active,
    )
    curriculum_index_variants = np.asarray(
        initial_state.info["curriculum_index"], dtype=np.int32
    ).reshape((active.state_count, active.robust_variants_per_state))
    if not np.all(curriculum_index_variants == curriculum_index_variants[:, :1]):
        raise RuntimeError("impact-recovery robust reset grouping changed")
    curriculum_index = curriculum_index_variants[:, 0]
    observation = np.asarray(initial_state.obs, dtype=np.float32).reshape(
        (active.state_count, active.robust_variants_per_state, -1)
    )[:, 0]
    improvement = (baseline_cost - teacher_cost) / np.maximum(np.abs(baseline_cost), 1.0e-12)
    stability_ok = maximum_stability_regression <= active.maximum_stability_deficit_regression
    accepted = finite & stability_ok & (improvement >= active.minimum_cost_improvement_fraction)
    evidence_arrays = {
        "actor_observation": observation,
        "teacher_plan": teacher_plan.astype(np.float32),
        "corrective_action": teacher_plan[:, 0].astype(np.float32),
        "baseline_cost": baseline_cost.astype(np.float64),
        "teacher_cost": teacher_cost.astype(np.float64),
        "cost_improvement_fraction": improvement.astype(np.float64),
        "teacher_accepted": accepted.astype(np.bool_),
        "finite_rollout": finite.astype(np.bool_),
        "curriculum_index": curriculum_index,
        "elapsed_since_contact_sec": arrays["elapsed_since_contact_sec"][curriculum_index].astype(
            np.float32
        ),
        "baseline_effect_metrics": baseline_effect.astype(np.float64),
        "teacher_effect_metrics": teacher_effect.astype(np.float64),
        "maximum_stability_regression": maximum_stability_regression.astype(np.float64),
        "baseline_success": baseline_success.astype(np.bool_),
        "teacher_success": teacher_success.astype(np.bool_),
        "baseline_maximum_stable_streak": baseline_streak.astype(np.float64),
        "teacher_maximum_stable_streak": teacher_streak.astype(np.float64),
    }
    lineage = {
        "curriculum_manifest_hash": str(manifest["manifest_hash"]),
        "curriculum_manifest_file_hash": hash_bytes(curriculum_path.read_bytes()),
        "curriculum_archive_hash": str(manifest["archive_hash"]),
        "body_hash": str(manifest["body_hash"]),
        "training_model_hash": str(manifest["training_model_hash"]),
    }
    return _write_evidence(
        output_dir=destination,
        config=active,
        arrays=evidence_arrays,
        lineage=lineage,
        devices=tuple(str(device) for device in devices),
        compiled_model=compiled_mujoco_model_contract(environment.mj_model),
        objective_diagnostics={
            "robust_variants_per_state": active.robust_variants_per_state,
            "robust_worst_case_weight": active.robust_worst_case_weight,
            "baseline_success_count": int(np.sum(baseline_success)),
            "teacher_success_count": int(np.sum(teacher_success)),
            "baseline_median_maximum_stable_streak": float(np.median(baseline_streak)),
            "teacher_median_maximum_stable_streak": float(np.median(teacher_streak)),
            "teacher_maximum_stable_streak": float(np.max(teacher_streak)),
        },
    )


class _CorrectivePlanBankEnv(ImpactRecoveryMJXEnv):
    """Privileged plan replay used only to diagnose teacher objective alignment."""

    def __init__(
        self,
        *,
        plan_bank: np.ndarray[Any, Any],
        plan_available: np.ndarray[Any, Any],
        action_chunk_steps: int,
        gate_division_floor: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._plan_bank = jnp.asarray(plan_bank)
        self._plan_available = jnp.asarray(plan_available)
        self._plan_chunk_steps = action_chunk_steps
        self._gate_division_floor = gate_division_floor

    def step(self, state: Any, unused_action: jax.Array) -> Any:
        del unused_action
        index = state.info["curriculum_index"]
        memory_step = state.info["memory_step"]
        chunk = jnp.minimum(
            memory_step // self._plan_chunk_steps,
            self._plan_bank.shape[1] - 1,
        )
        desired = self._plan_bank[index, chunk]
        desired = jnp.where(self._plan_available[index], desired, jnp.zeros_like(desired))
        gate = self._residual_gate(state.pipeline_state, index, memory_step)
        command = jnp.where(
            gate >= self._gate_division_floor,
            jnp.clip(desired / jnp.maximum(gate, 1.0e-6), -1.0, 1.0),
            jnp.zeros_like(desired),
        )
        return super().step(state, command)


def evaluate_impact_recovery_corrective_plan_bank(
    *,
    asset_root: Path,
    curriculum_manifest_path: Path,
    teacher_report_path: Path,
    output_dir: Path,
    source_checkout_path: Path,
    population: str = "ACQUISITION",
    config: ImpactRecoveryMJXEvaluationConfig | None = None,
) -> dict[str, Any]:
    """Test whether privileged CEM plans improve the true full episode metric."""

    from brax.envs.wrappers import training as training_wrappers
    from brax.training import acting

    if population not in {"ACQUISITION", "RETENTION"}:
        raise ValueError("impact-recovery plan-bank population is invalid")
    active = config or ImpactRecoveryMJXEvaluationConfig(seeds=(57_151, 57_152, 57_153, 57_154))
    root = asset_root.expanduser().resolve()
    curriculum_path = curriculum_manifest_path.expanduser().resolve()
    teacher_path = teacher_report_path.expanduser().resolve()
    destination = output_dir.expanduser().resolve()
    checkout = source_checkout_path.expanduser().resolve()
    model_path = root / "g1_description" / "g1_liao.xml"
    if any(not path.is_file() for path in (model_path, curriculum_path, teacher_path)):
        raise FileNotFoundError("impact-recovery plan-bank inputs are incomplete")
    if destination.exists() or destination == checkout or checkout in destination.parents:
        raise ValueError("impact-recovery plan-bank output must be new and external")
    manifest = validate_impact_recovery_curriculum(curriculum_path)
    teacher = validate_impact_recovery_corrective_teacher(teacher_path)
    if (
        teacher.get("curriculum_manifest_hash") != manifest.get("manifest_hash")
        or manifest.get("training_model_hash") != hash_bytes(model_path.read_bytes())
        or manifest.get("body_hash") != g1_body_hash(root)
    ):
        raise ValueError("impact-recovery plan-bank lineage changed")
    curriculum_archive = curriculum_path.parent / str(manifest["archive"])
    teacher_archive = teacher_path.parent / str(teacher["corpus_archive"])
    with np.load(curriculum_archive, allow_pickle=False) as archive:
        curriculum_arrays = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(teacher_archive, allow_pickle=False) as archive:
        teacher_arrays = {name: np.asarray(archive[name]) for name in archive.files}
    teacher_config = ImpactRecoveryCorrectiveTeacherConfig(
        **cast(dict[str, Any], teacher["config"])
    )
    snapshot_count = int(manifest["snapshot_count"])
    plan_bank = np.zeros(
        (snapshot_count, teacher_config.action_chunk_count, _JOINT_COUNT), np.float32
    )
    plan_available = np.zeros((snapshot_count,), np.bool_)
    accepted = teacher_arrays["teacher_accepted"].astype(np.bool_)
    indexes = teacher_arrays["curriculum_index"].astype(np.int32)
    plan_bank[indexes[accepted]] = teacher_arrays["teacher_plan"][accepted]
    plan_available[indexes[accepted]] = True
    controller = ImpactRecoveryMJXConfig(
        retention_memory_mode="DIRECT_REPLAY",
        gain_memory_mode="DYNAMIC",
        residual_gate_mode="TEACHER_NOVELTY",
        residual_authority_steps=teacher_config.horizon_steps,
    )
    environment = _CorrectivePlanBankEnv(
        model_path=model_path,
        curriculum_arrays=curriculum_arrays,
        desired_heading_rad=float(cast(dict[str, Any], manifest["config"])["desired_heading_rad"]),
        reset_population=cast(Any, population),
        config=controller,
        plan_bank=plan_bank,
        plan_available=plan_available,
        action_chunk_steps=teacher_config.action_chunk_steps,
        gate_division_floor=0.05,
    )
    wrapped = training_wrappers.wrap(
        environment,
        episode_length=controller.episode_length,
        action_repeat=1,
    )

    def zero_policy(unused: Any) -> Any:
        del unused

        def policy(observation: jax.Array, rng: jax.Array) -> tuple[jax.Array, dict[str, Any]]:
            del rng
            return jnp.zeros(observation.shape[:-1] + (_JOINT_COUNT,), jnp.float32), {}

        return policy

    evaluator = acting.Evaluator(
        wrapped,
        zero_policy,
        num_eval_envs=active.num_envs,
        episode_length=controller.episode_length,
        action_repeat=1,
        key=jax.random.PRNGKey(active.seeds[0]),
    )
    repeats: list[dict[str, Any]] = []
    for seed in active.seeds:
        evaluator._key = jax.random.PRNGKey(seed)
        metrics = evaluator.run_evaluation(None, {}, aggregate_episodes=False)
        success = np.asarray(metrics["eval/episode_success"], dtype=np.float64)
        repeats.append(
            {
                "seed": seed,
                "success_count": int(np.count_nonzero(success > 0.0)),
                "success_rate": float(np.mean(success > 0.0)),
            }
        )
    episode_count = active.num_envs * len(active.seeds)
    success_count = sum(int(row["success_count"]) for row in repeats)
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.impact_recovery_corrective_plan_bank_exam.v1",
        "config": asdict(active),
        "config_hash": active.config_hash,
        "teacher_report_hash": teacher["report_hash"],
        "teacher_corpus_hash": teacher["corpus_archive_hash"],
        "curriculum_manifest_hash": manifest["manifest_hash"],
        "population": population,
        "episode_count": episode_count,
        "success_count": success_count,
        "success_rate": success_count / episode_count,
        "repeats": repeats,
        "accepted_plan_count": int(np.sum(plan_available)),
        "privileged_curriculum_index_used": True,
        "privileged_control_phase_used": True,
        "deployment_candidate": False,
        "promotion_eligible": False,
        "promotion_authority": "NONE",
        "claim_boundary": "Objective-alignment diagnostic only; privileged plan lookup",
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    destination.mkdir(parents=True)
    _atomic_json(destination / "plan-bank-exam.json", report)
    return report


__all__ = [
    "ImpactRecoveryCorrectiveTeacherConfig",
    "evaluate_impact_recovery_corrective_plan_bank",
    "run_impact_recovery_corrective_teacher",
    "validate_impact_recovery_corrective_teacher",
]
