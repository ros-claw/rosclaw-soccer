"""Distill counterfactual recovery plans into a gated proprioceptive student."""

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
from rosclaw_soccer.training.impact_recovery_corrective_teacher import (
    ImpactRecoveryCorrectiveTeacherConfig,
    _selected_reset_keys,
    validate_impact_recovery_corrective_teacher,
)
from rosclaw_soccer.training.impact_recovery_curriculum import (
    validate_impact_recovery_curriculum,
)
from rosclaw_soccer.training.impact_recovery_mjx import (
    ImpactRecoveryMJXConfig,
    ImpactRecoveryMJXEnv,
    ImpactRecoveryMJXEvaluationConfig,
    _teacher_novelty_gate,
)
from rosclaw_soccer.training.recovery_mjx import _atomic_json, compiled_mujoco_model_contract

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_JOINT_COUNT = 29
_EFFECT_WEIGHTS = np.asarray((3.0, 1.0, 1.0, 2.0, 2.0, 0.5), dtype=np.float64)


@dataclass(frozen=True)
class ImpactRecoveryDistillationConfig:
    """Reproducible collection and low-capacity student training contract."""

    hidden_width: int = 128
    student_model_type: Literal["MLP", "RIDGE_CURRENT_FRAME"] = "MLP"
    ridge_regularization: float = 100.0
    training_steps: int = 3_000
    batch_size: int = 256
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-5
    holdout_state_count: int = 8
    gate_division_floor: float = 0.05
    required_validation_loss_improvement_fraction: float = 0.20
    required_gpu_count: int = 4
    random_seed: int = 57_119
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.impact_recovery_distillation_config.v1"

    def __post_init__(self) -> None:
        finite = (
            self.learning_rate,
            self.weight_decay,
            self.ridge_regularization,
            self.gate_division_floor,
            self.required_validation_loss_improvement_fraction,
        )
        if (
            not 32 <= self.hidden_width <= 512
            or self.student_model_type not in {"MLP", "RIDGE_CURRENT_FRAME"}
            or not 1.0e-3 <= self.ridge_regularization <= 1.0e6
            or not 100 <= self.training_steps <= 100_000
            or not 32 <= self.batch_size <= 8_192
            or not 1.0e-6 <= self.learning_rate <= 1.0e-2
            or not 0.0 <= self.weight_decay <= 0.1
            or not 1 <= self.holdout_state_count <= 128
            or not 0.01 <= self.gate_division_floor <= 0.5
            or not 0.0 <= self.required_validation_loss_improvement_fraction <= 0.95
            or any(not math.isfinite(value) for value in finite)
            or self.required_gpu_count != 4
            or not 0 <= self.random_seed < 2**31
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("impact-recovery distillation config is invalid")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


def _effect(metrics: dict[str, jax.Array]) -> jax.Array:
    def pseudo_huber(value: jax.Array, delta: float = 0.25) -> jax.Array:
        scaled = value / delta
        return delta * delta * (jnp.sqrt(1.0 + jnp.square(scaled)) - 1.0)

    return jnp.stack(
        (
            pseudo_huber(metrics["root_body_backward_speed"] / 0.5),
            pseudo_huber(metrics["root_body_lateral_speed"] / 0.5),
            pseudo_huber(metrics["root_angular_speed"] / 1.0),
            jnp.square(jnp.maximum(0.70 - metrics["pelvis_height"], 0.0) / 0.15),
            jnp.square(jnp.maximum(0.95 - metrics["upright"], 0.0) / 0.20),
            1.0 - metrics["bilateral_support"],
        ),
        axis=-1,
    )


def _make_gated_plan_replay(
    *,
    wrapped: Any,
    teacher_qpos: jax.Array,
    teacher_qvel: jax.Array,
    controller: ImpactRecoveryMJXConfig,
    teacher_config: ImpactRecoveryCorrectiveTeacherConfig,
    distillation_config: ImpactRecoveryDistillationConfig,
) -> Any:
    weights = jnp.asarray(_EFFECT_WEIGHTS, dtype=jnp.float32)

    def replay_device(
        initial_state: Any,
        plans: jax.Array,
    ) -> tuple[jax.Array, ...]:
        baseline_state = initial_state
        teacher_state = initial_state
        baseline_effect = jnp.zeros((plans.shape[0], len(_EFFECT_WEIGHTS)), jnp.float32)
        teacher_effect = jnp.zeros_like(baseline_effect)
        command_cost = jnp.zeros((plans.shape[0],), jnp.float32)
        slew_cost = jnp.zeros_like(command_cost)
        previous_command = jnp.zeros((plans.shape[0], _JOINT_COUNT), jnp.float32)
        finite = jnp.ones((plans.shape[0],), jnp.bool_)
        baseline_ready_deficit = jnp.zeros_like(command_cost)
        teacher_ready_deficit = jnp.zeros_like(command_cost)
        baseline_maximum_streak = jnp.zeros_like(command_cost)
        teacher_maximum_streak = jnp.zeros_like(command_cost)
        baseline_ever_success = jnp.zeros((plans.shape[0],), jnp.bool_)
        teacher_ever_success = jnp.zeros((plans.shape[0],), jnp.bool_)

        def step(
            carry: tuple[
                Any,
                Any,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
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
                Any,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
                jax.Array,
            ],
            tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        ]:
            (
                baseline,
                teacher,
                baseline_total,
                teacher_total,
                action_total,
                slew_total,
                prior_command,
                finite_so_far,
                baseline_ready_total,
                teacher_ready_total,
                baseline_streak_max,
                teacher_streak_max,
                baseline_succeeded,
                teacher_succeeded,
            ) = carry
            indexes = teacher.info["curriculum_index"]
            memory_steps = jnp.minimum(teacher.info["memory_step"], teacher_qpos.shape[1] - 1)
            reference_qpos = teacher_qpos[indexes, memory_steps]
            reference_qvel = teacher_qvel[indexes, memory_steps]
            gates = jax.vmap(
                lambda qpos, qvel, ref_qpos, ref_qvel: _teacher_novelty_gate(
                    qpos[:36], qvel[:35], ref_qpos, ref_qvel, controller
                )
            )(
                teacher.pipeline_state.qpos,
                teacher.pipeline_state.qvel,
                reference_qpos,
                reference_qvel,
            )
            desired = plans[:, step_index // teacher_config.action_chunk_steps]
            command = jnp.where(
                gates[:, None] >= distillation_config.gate_division_floor,
                jnp.clip(desired / jnp.maximum(gates[:, None], 1.0e-6), -1.0, 1.0),
                jnp.zeros_like(desired),
            )
            next_teacher = wrapped.step(teacher, command)
            next_baseline = wrapped.step(baseline, jnp.zeros_like(command))
            current_finite = jnp.all(
                jnp.isfinite(next_teacher.pipeline_state.qpos), axis=-1
            ) & jnp.all(jnp.isfinite(next_teacher.pipeline_state.qvel), axis=-1)
            return (
                next_baseline,
                next_teacher,
                baseline_total + _effect(next_baseline.metrics),
                teacher_total + _effect(next_teacher.metrics),
                action_total + jnp.mean(jnp.square(command), axis=-1),
                slew_total + jnp.mean(jnp.square(command - prior_command), axis=-1),
                command,
                finite_so_far & current_finite,
                baseline_ready_total + (1.0 - next_baseline.metrics["ready"]),
                teacher_ready_total + (1.0 - next_teacher.metrics["ready"]),
                jnp.maximum(baseline_streak_max, next_baseline.metrics["maximum_stable_streak"]),
                jnp.maximum(teacher_streak_max, next_teacher.metrics["maximum_stable_streak"]),
                baseline_succeeded | (next_baseline.metrics["success"] > 0.0),
                teacher_succeeded | (next_teacher.metrics["success"] > 0.0),
            ), (teacher.obs, command, gates, desired)

        final, rows = jax.lax.scan(
            step,
            (
                baseline_state,
                teacher_state,
                baseline_effect,
                teacher_effect,
                command_cost,
                slew_cost,
                previous_command,
                finite,
                baseline_ready_deficit,
                teacher_ready_deficit,
                baseline_maximum_streak,
                teacher_maximum_streak,
                baseline_ever_success,
                teacher_ever_success,
            ),
            jnp.arange(teacher_config.horizon_steps, dtype=jnp.int32),
        )
        baseline_mean = final[2] / teacher_config.horizon_steps
        teacher_mean = final[3] / teacher_config.horizon_steps
        baseline_cost = jnp.sum(baseline_mean * weights, axis=-1)
        teacher_cost = (
            jnp.sum(teacher_mean * weights, axis=-1)
            + teacher_config.action_magnitude_cost_weight * final[4] / teacher_config.horizon_steps
            + teacher_config.action_slew_cost_weight * final[5] / teacher_config.horizon_steps
            + (~final[7]).astype(jnp.float32) * 100.0
        )
        if teacher_config.objective_mode == "SUCCESSOR_STREAK":
            baseline_cost = (
                baseline_cost
                + teacher_config.ready_fraction_deficit_weight
                * final[8]
                / teacher_config.horizon_steps
                + teacher_config.stable_streak_deficit_weight
                * (1.0 - jnp.clip(final[10] / 25.0, 0.0, 1.0))
                + teacher_config.success_deficit_weight * (~final[12]).astype(jnp.float32)
            )
            teacher_cost = (
                teacher_cost
                + teacher_config.ready_fraction_deficit_weight
                * final[9]
                / teacher_config.horizon_steps
                + teacher_config.stable_streak_deficit_weight
                * (1.0 - jnp.clip(final[11] / 25.0, 0.0, 1.0))
                + teacher_config.success_deficit_weight * (~final[13]).astype(jnp.float32)
            )
        return (
            rows[0],
            rows[1],
            rows[2],
            rows[3],
            baseline_cost,
            teacher_cost,
            baseline_mean,
            teacher_mean,
            final[7],
            final[12],
            final[13],
            final[10],
            final[11],
        )

    return jax.pmap(replay_device, axis_name="impact_recovery_distillation_devices")


def _flatten_time_state(value: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    axes = (0, 2, 1) + tuple(range(3, value.ndim))
    transposed = np.transpose(value, axes)
    return transposed.reshape((-1,) + value.shape[3:])


def _train_student(
    *,
    observation: np.ndarray[Any, Any],
    action: np.ndarray[Any, Any],
    state_index: np.ndarray[Any, Any],
    config: ImpactRecoveryDistillationConfig,
) -> tuple[dict[str, np.ndarray[Any, Any]], dict[str, Any]]:
    unique_states = np.unique(state_index)
    if unique_states.size <= config.holdout_state_count:
        raise ValueError("impact-recovery distillation has too few teacher states")
    random = np.random.default_rng(config.random_seed)
    shuffled_states = random.permutation(unique_states)
    validation_states = np.sort(shuffled_states[: config.holdout_state_count])
    validation_mask = np.isin(state_index, validation_states)
    training_mask = ~validation_mask
    selected_observation = (
        observation[:, -189:] if config.student_model_type == "RIDGE_CURRENT_FRAME" else observation
    )
    train_x = np.asarray(selected_observation[training_mask], dtype=np.float32)
    train_y = np.asarray(action[training_mask], dtype=np.float32)
    validation_x = np.asarray(selected_observation[validation_mask], dtype=np.float32)
    validation_y = np.asarray(action[validation_mask], dtype=np.float32)
    input_mean = np.mean(train_x, axis=0, dtype=np.float64).astype(np.float32)
    input_std = np.maximum(np.std(train_x, axis=0, dtype=np.float64), 1.0e-4).astype(np.float32)
    train_x = np.clip((train_x - input_mean) / input_std, -10.0, 10.0)
    validation_x = np.clip((validation_x - input_mean) / input_std, -10.0, 10.0)
    zero_validation_loss = float(np.mean(np.square(validation_y), dtype=np.float64))
    common_metrics = {
        "training_row_count": int(train_x.shape[0]),
        "validation_row_count": int(validation_x.shape[0]),
        "training_state_count": int(unique_states.size - validation_states.size),
        "validation_state_count": int(validation_states.size),
        "validation_state_indices": [int(value) for value in validation_states],
        "zero_action_validation_loss": zero_validation_loss,
    }
    if config.student_model_type == "RIDGE_CURRENT_FRAME":
        design = np.concatenate(
            (train_x.astype(np.float64), np.ones((train_x.shape[0], 1), dtype=np.float64)),
            axis=1,
        )
        regularizer = np.eye(design.shape[1], dtype=np.float64)
        regularizer[-1, -1] = 0.0
        coefficient = np.linalg.solve(
            design.T @ design + config.ridge_regularization * regularizer,
            design.T @ train_y.astype(np.float64),
        )
        validation_prediction = np.clip(
            validation_x.astype(np.float64) @ coefficient[:-1] + coefficient[-1],
            -1.0,
            1.0,
        )
        training_prediction = np.clip(
            train_x.astype(np.float64) @ coefficient[:-1] + coefficient[-1],
            -1.0,
            1.0,
        )
        validation_loss = float(np.mean(np.square(validation_prediction - validation_y)))
        training_loss = float(np.mean(np.square(training_prediction - train_y)))
        return (
            {
                "input_mean": input_mean,
                "input_std": input_std,
                "weight": coefficient[:-1].astype(np.float32),
                "bias": coefficient[-1].astype(np.float32),
            },
            {
                **common_metrics,
                "final_training_loss": training_loss,
                "best_validation_loss": validation_loss,
                "validation_loss_improvement_fraction": (
                    (zero_validation_loss - validation_loss) / max(zero_validation_loss, 1.0e-12)
                ),
            },
        )

    import optax

    key = jax.random.PRNGKey(config.random_seed)
    key1, key2, key3 = jax.random.split(key, 3)

    def initialize(rng: jax.Array, inputs: int, outputs: int) -> jax.Array:
        return jax.random.normal(rng, (inputs, outputs), dtype=jnp.float32) * jnp.sqrt(2.0 / inputs)

    params: tuple[jax.Array, ...] = (
        initialize(key1, train_x.shape[1], config.hidden_width),
        jnp.zeros((config.hidden_width,), jnp.float32),
        initialize(key2, config.hidden_width, config.hidden_width),
        jnp.zeros((config.hidden_width,), jnp.float32),
        initialize(key3, config.hidden_width, _JOINT_COUNT),
        jnp.zeros((_JOINT_COUNT,), jnp.float32),
    )

    def forward(active: tuple[jax.Array, ...], values: jax.Array) -> jax.Array:
        w1, b1, w2, b2, w3, b3 = active
        hidden1 = jnp.tanh(values @ w1 + b1)
        hidden2 = jnp.tanh(hidden1 @ w2 + b2)
        return jnp.tanh(hidden2 @ w3 + b3)

    optimizer = optax.adamw(config.learning_rate, weight_decay=config.weight_decay)
    optimizer_state = optimizer.init(params)

    @jax.jit  # type: ignore[untyped-decorator]
    def update(
        active: tuple[jax.Array, ...],
        state: Any,
        batch_x: jax.Array,
        batch_y: jax.Array,
    ) -> tuple[tuple[jax.Array, ...], Any, jax.Array]:
        def loss_fn(candidate: tuple[jax.Array, ...]) -> jax.Array:
            error = forward(candidate, batch_x) - batch_y
            return jnp.mean(jnp.square(error))

        loss, gradients = jax.value_and_grad(loss_fn)(active)
        updates, state = optimizer.update(gradients, state, active)
        return cast(tuple[jax.Array, ...], optax.apply_updates(active, updates)), state, loss

    train_x_jax = jnp.asarray(train_x)
    train_y_jax = jnp.asarray(train_y)
    validation_x_jax = jnp.asarray(validation_x)
    validation_y_jax = jnp.asarray(validation_y)
    batch_rng = np.random.default_rng(config.random_seed + 1)
    best_params = params
    best_validation_loss = math.inf
    final_training_loss = math.inf
    for step in range(config.training_steps):
        indexes = batch_rng.integers(0, train_x.shape[0], size=config.batch_size)
        params, optimizer_state, loss = update(
            params,
            optimizer_state,
            train_x_jax[indexes],
            train_y_jax[indexes],
        )
        final_training_loss = float(loss)
        if step % 25 == 0 or step + 1 == config.training_steps:
            validation_loss = float(
                jnp.mean(jnp.square(forward(params, validation_x_jax) - validation_y_jax))
            )
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_params = params
    model = {
        "input_mean": input_mean,
        "input_std": input_std,
        "w1": np.asarray(best_params[0], dtype=np.float32),
        "b1": np.asarray(best_params[1], dtype=np.float32),
        "w2": np.asarray(best_params[2], dtype=np.float32),
        "b2": np.asarray(best_params[3], dtype=np.float32),
        "w3": np.asarray(best_params[4], dtype=np.float32),
        "b3": np.asarray(best_params[5], dtype=np.float32),
    }
    metrics = {
        **common_metrics,
        "final_training_loss": final_training_loss,
        "best_validation_loss": best_validation_loss,
        "validation_loss_improvement_fraction": (
            (zero_validation_loss - best_validation_loss) / max(zero_validation_loss, 1.0e-12)
        ),
    }
    return model, metrics


def build_impact_recovery_distilled_student(
    *,
    asset_root: Path,
    curriculum_manifest_path: Path,
    teacher_report_path: Path,
    output_dir: Path,
    source_checkout_path: Path,
    config: ImpactRecoveryDistillationConfig | None = None,
) -> dict[str, Any]:
    """Replay accepted plans behind a novelty gate and train a feedback student."""

    active = config or ImpactRecoveryDistillationConfig()
    root = asset_root.expanduser().resolve()
    curriculum_path = curriculum_manifest_path.expanduser().resolve()
    teacher_path = teacher_report_path.expanduser().resolve()
    destination = output_dir.expanduser().resolve()
    checkout = source_checkout_path.expanduser().resolve()
    model_path = root / "g1_description" / "g1_liao.xml"
    if any(not path.is_file() for path in (model_path, curriculum_path, teacher_path)):
        raise FileNotFoundError("impact-recovery distillation inputs are incomplete")
    if destination.exists() or destination == checkout or checkout in destination.parents:
        raise ValueError("impact-recovery distillation output must be new and external")
    devices = tuple(jax.devices())
    if len(devices) != active.required_gpu_count or any(
        getattr(device, "platform", "") != "gpu" for device in devices
    ):
        raise RuntimeError("impact-recovery distillation requires exactly four GPUs")
    manifest = validate_impact_recovery_curriculum(curriculum_path)
    teacher = validate_impact_recovery_corrective_teacher(teacher_path)
    if (
        manifest.get("schema_version")
        not in {
            "rosclaw_soccer.impact_recovery_curriculum.v5",
            "rosclaw_soccer.impact_recovery_curriculum.v6",
        }
        or teacher.get("supervised_warm_start_eligible") is not True
        or teacher.get("curriculum_manifest_hash") != manifest.get("manifest_hash")
        or manifest.get("training_model_hash") != hash_bytes(model_path.read_bytes())
        or manifest.get("body_hash") != g1_body_hash(root)
    ):
        raise ValueError("impact-recovery distillation lineage changed")
    archive_path = curriculum_path.parent / str(manifest["archive"])
    teacher_archive_path = teacher_path.parent / str(teacher["corpus_archive"])
    with np.load(archive_path, allow_pickle=False) as archive:
        curriculum_arrays = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(teacher_archive_path, allow_pickle=False) as archive:
        teacher_arrays = {name: np.asarray(archive[name]) for name in archive.files}
    teacher_config = ImpactRecoveryCorrectiveTeacherConfig(
        **cast(dict[str, Any], teacher["config"])
    )
    controller = ImpactRecoveryMJXConfig(
        retention_memory_mode="DIRECT_REPLAY",
        gain_memory_mode="DYNAMIC",
        residual_gate_mode="TEACHER_NOVELTY",
        residual_authority_steps=teacher_config.horizon_steps,
    )
    environment = ImpactRecoveryMJXEnv(
        model_path=model_path,
        curriculum_arrays=curriculum_arrays,
        desired_heading_rad=float(cast(dict[str, Any], manifest["config"])["desired_heading_rad"]),
        reset_population="ACQUISITION",
        config=controller,
    )
    wrapped = brax_wrappers.EpisodeWrapper(
        brax_wrappers.VmapWrapper(environment),
        episode_length=teacher_config.horizon_steps + 1,
        action_repeat=1,
    )
    acquisition_indexes = np.flatnonzero(~curriculum_arrays["source_succeeded"].astype(np.bool_))
    reset_keys = _selected_reset_keys(
        rng=jax.random.PRNGKey(teacher_config.random_seed),
        state_count=teacher_config.state_count,
        acquisition_indexes=acquisition_indexes,
        elapsed=curriculum_arrays["elapsed_since_contact_sec"],
        variants_per_state=teacher_config.robust_variants_per_state,
    ).reshape(
        (
            teacher_config.required_gpu_count,
            teacher_config.state_count
            // teacher_config.required_gpu_count
            * teacher_config.robust_variants_per_state,
            2,
        )
    )
    initial_state = jax.pmap(wrapped.reset)(reset_keys)
    initial_index_groups = np.asarray(
        initial_state.info["curriculum_index"], dtype=np.int32
    ).reshape((teacher_config.state_count, teacher_config.robust_variants_per_state))
    if not np.all(initial_index_groups == initial_index_groups[:, :1]) or not np.array_equal(
        initial_index_groups[:, 0], teacher_arrays["curriculum_index"]
    ):
        raise ValueError("impact-recovery distillation reset reconstruction changed")
    initial_indexes = initial_index_groups.reshape((-1,))
    repeated_plans = np.repeat(
        teacher_arrays["teacher_plan"], teacher_config.robust_variants_per_state, axis=0
    )
    plans = jnp.asarray(repeated_plans).reshape(
        (
            teacher_config.required_gpu_count,
            teacher_config.state_count
            // teacher_config.required_gpu_count
            * teacher_config.robust_variants_per_state,
            teacher_config.action_chunk_count,
            _JOINT_COUNT,
        )
    )
    replay = _make_gated_plan_replay(
        wrapped=wrapped,
        teacher_qpos=jnp.asarray(curriculum_arrays["frozen_memory_qpos"]),
        teacher_qvel=jnp.asarray(curriculum_arrays["frozen_memory_qvel"]),
        controller=controller,
        teacher_config=teacher_config,
        distillation_config=active,
    )
    (
        observation_device,
        command_device,
        gate_device,
        desired_device,
        baseline_cost_device,
        teacher_cost_device,
        baseline_effect_device,
        teacher_effect_device,
        finite_device,
        baseline_success_device,
        teacher_success_device,
        baseline_streak_device,
        teacher_streak_device,
    ) = replay(initial_state, plans)
    observation = _flatten_time_state(np.asarray(observation_device, dtype=np.float32))
    command = _flatten_time_state(np.asarray(command_device, dtype=np.float32))
    gate = _flatten_time_state(np.asarray(gate_device, dtype=np.float32))
    desired = _flatten_time_state(np.asarray(desired_device, dtype=np.float32))
    state_rows = np.repeat(initial_indexes, teacher_config.horizon_steps)
    step_rows = np.tile(
        np.arange(teacher_config.horizon_steps, dtype=np.int32), len(initial_indexes)
    )
    baseline_cost = np.asarray(baseline_cost_device, dtype=np.float64).reshape((-1,))
    teacher_cost = np.asarray(teacher_cost_device, dtype=np.float64).reshape((-1,))
    baseline_effect = np.asarray(baseline_effect_device, dtype=np.float64).reshape(
        (-1, len(_EFFECT_WEIGHTS))
    )
    teacher_effect = np.asarray(teacher_effect_device, dtype=np.float64).reshape(
        (-1, len(_EFFECT_WEIGHTS))
    )
    finite = np.asarray(finite_device, dtype=np.bool_).reshape((-1,))
    baseline_success = np.asarray(baseline_success_device, dtype=np.bool_).reshape((-1,))
    teacher_success = np.asarray(teacher_success_device, dtype=np.bool_).reshape((-1,))
    baseline_streak = np.asarray(baseline_streak_device, dtype=np.float64).reshape((-1,))
    teacher_streak = np.asarray(teacher_streak_device, dtype=np.float64).reshape((-1,))
    improvement = (baseline_cost - teacher_cost) / np.maximum(np.abs(baseline_cost), 1.0e-12)
    stability_ok = np.all(
        teacher_effect[:, 3:5]
        <= baseline_effect[:, 3:5] + teacher_config.maximum_stability_deficit_regression,
        axis=1,
    )
    gated_accepted = (
        finite & stability_ok & (improvement >= teacher_config.minimum_cost_improvement_fraction)
    )
    state_instance_rows = np.repeat(
        np.arange(initial_indexes.size, dtype=np.int32), teacher_config.horizon_steps
    )
    row_mask = gated_accepted[state_instance_rows]
    if not np.any(row_mask):
        raise RuntimeError("impact-recovery gated teacher produced no distillation rows")
    model, training_metrics = _train_student(
        observation=observation[row_mask],
        action=command[row_mask],
        state_index=state_rows[row_mask],
        config=active,
    )
    destination.mkdir(parents=True)
    corpus_path = destination / "gated-distillation-corpus.npz"
    corpus_tmp = destination / ".gated-distillation-corpus.npz.tmp"
    with corpus_tmp.open("wb") as stream:
        np.savez_compressed(
            stream,
            actor_observation=observation,
            commanded_action=command,
            desired_applied_action=desired,
            novelty_gate=gate,
            curriculum_index=state_rows,
            control_step=step_rows,
            accepted_state_row=row_mask,
            gated_state_accepted=gated_accepted,
            baseline_cost=baseline_cost,
            teacher_cost=teacher_cost,
            cost_improvement_fraction=improvement,
            baseline_effect_metrics=baseline_effect,
            teacher_effect_metrics=teacher_effect,
            baseline_success=baseline_success,
            teacher_success=teacher_success,
            baseline_maximum_stable_streak=baseline_streak,
            teacher_maximum_stable_streak=teacher_streak,
        )
    os.replace(corpus_tmp, corpus_path)
    model_path_output = destination / "student-model.npz"
    model_tmp = destination / ".student-model.npz.tmp"
    with model_tmp.open("wb") as stream:
        np.savez_compressed(stream, **model)  # type: ignore[arg-type]
    os.replace(model_tmp, model_path_output)
    gated_accepted_count = int(np.sum(gated_accepted))
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.impact_recovery_distilled_student.v1",
        "config": asdict(active),
        "config_hash": active.config_hash,
        "teacher_report_hash": teacher["report_hash"],
        "teacher_report_file_hash": hash_bytes(teacher_path.read_bytes()),
        "teacher_corpus_hash": teacher["corpus_archive_hash"],
        "curriculum_manifest_hash": manifest["manifest_hash"],
        "curriculum_archive_hash": manifest["archive_hash"],
        "body_hash": manifest["body_hash"],
        "compiled_model_contract": compiled_mujoco_model_contract(environment.mj_model),
        "devices": [str(device) for device in devices],
        "device_count": len(devices),
        "all_devices_used": True,
        "teacher_state_count": teacher_config.state_count,
        "robust_variants_per_state": teacher_config.robust_variants_per_state,
        "replay_state_count": teacher_config.state_count * teacher_config.robust_variants_per_state,
        "gated_accepted_state_count": gated_accepted_count,
        "gated_accepted_fraction": gated_accepted_count
        / (teacher_config.state_count * teacher_config.robust_variants_per_state),
        "gated_median_cost_improvement_fraction": float(np.median(improvement)),
        "gated_median_novelty_permission": float(np.median(gate)),
        "gated_objective_diagnostics": {
            "baseline_success_count": int(np.sum(baseline_success)),
            "teacher_success_count": int(np.sum(teacher_success)),
            "baseline_median_maximum_stable_streak": float(np.median(baseline_streak)),
            "teacher_median_maximum_stable_streak": float(np.median(teacher_streak)),
        },
        "corpus": corpus_path.name,
        "corpus_hash": hash_bytes(corpus_path.read_bytes()),
        "student_model": model_path_output.name,
        "student_model_hash": hash_bytes(model_path_output.read_bytes()),
        "model_architecture": (
            "NORMALIZED_RIDGE_CURRENT_FRAME_TO_29_JOINT_RESIDUAL"
            if active.student_model_type == "RIDGE_CURRENT_FRAME"
            else "NORMALIZED_TANH_MLP_128X128_TO_29_JOINT_RESIDUAL"
        ),
        "action_semantics": "TEACHER_NOVELTY_GATED_BOUNDED_29_JOINT_PD_RESIDUAL",
        "residual_authority_steps": teacher_config.horizon_steps,
        "training_metrics": training_metrics,
        "student_exam_eligible": bool(
            training_metrics["validation_loss_improvement_fraction"]
            >= active.required_validation_loss_improvement_fraction
        ),
        "deployment_candidate": False,
        "promotion_eligible": False,
        "promotion_authority": "NONE",
        "claim_boundary": (
            "Supervised SIM candidate; matched acquisition and retention exam required"
        ),
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    report_path = destination / "distillation-report.json"
    _atomic_json(report_path, report)
    return validate_impact_recovery_distilled_student(report_path)


def validate_impact_recovery_distilled_student(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("impact-recovery distilled student report is invalid")
    report = cast(dict[str, Any], payload)
    declared = report.pop("report_hash", None)
    try:
        config_value = report.get("config")
        if not isinstance(config_value, dict):
            raise ValueError("impact-recovery distilled student config is missing")
        config = ImpactRecoveryDistillationConfig(**config_value)
        corpus_path = resolved.parent / str(report.get("corpus", ""))
        model_path = resolved.parent / str(report.get("student_model", ""))
        with np.load(model_path, allow_pickle=False) as archive:
            model = {name: np.asarray(archive[name]) for name in archive.files}
        metrics = report.get("training_metrics")
        validation_improvement = (
            metrics.get("validation_loss_improvement_fraction")
            if isinstance(metrics, dict)
            else None
        )
        expected_exam_eligible = bool(
            isinstance(validation_improvement, int | float)
            and not isinstance(validation_improvement, bool)
            and math.isfinite(float(validation_improvement))
            and float(validation_improvement)
            >= config.required_validation_loss_improvement_fraction
        )
        model_valid = (
            set(model) == {"input_mean", "input_std", "weight", "bias"}
            and model["weight"].shape == (model["input_mean"].size, _JOINT_COUNT)
            and model["bias"].shape == (_JOINT_COUNT,)
            if config.student_model_type == "RIDGE_CURRENT_FRAME"
            else set(model) == {"input_mean", "input_std", "w1", "b1", "w2", "b2", "w3", "b3"}
            and model["w1"].shape == (model["input_mean"].size, config.hidden_width)
            and model["b1"].shape == (config.hidden_width,)
            and model["w2"].shape == (config.hidden_width, config.hidden_width)
            and model["b2"].shape == (config.hidden_width,)
            and model["w3"].shape == (config.hidden_width, _JOINT_COUNT)
            and model["b3"].shape == (_JOINT_COUNT,)
        )
        model_valid = bool(
            model_valid
            and model["input_mean"].ndim == 1
            and model["input_mean"].size > 0
            and model["input_std"].shape == model["input_mean"].shape
            and np.all(model["input_std"] > 0.0)
        )
        if (
            not model_valid
            or any(not np.all(np.isfinite(value)) for value in model.values())
            or not isinstance(metrics, dict)
            or not isinstance(validation_improvement, int | float)
            or isinstance(validation_improvement, bool)
            or not math.isfinite(float(validation_improvement))
            or report.get("student_exam_eligible") is not expected_exam_eligible
            or report.get("schema_version") != "rosclaw_soccer.impact_recovery_distilled_student.v1"
            or declared != hash_json(report)
            or report.get("config_hash") != hash_json(config_value)
            or report.get("corpus") != "gated-distillation-corpus.npz"
            or report.get("student_model") != "student-model.npz"
            or report.get("corpus_hash") != hash_bytes(corpus_path.read_bytes())
            or report.get("student_model_hash") != hash_bytes(model_path.read_bytes())
            or report.get("device_count") != config.required_gpu_count
            or report.get("all_devices_used") is not True
            or report.get("deployment_candidate") is not False
            or report.get("promotion_eligible") is not False
            or report.get("promotion_authority") != "NONE"
            or report.get("activation_ceiling") != "SIM_ONLY"
            or report.get("hardware_authorized") is not False
            or report.get("hardware_command_sent") is not False
            or not isinstance(report.get("residual_authority_steps", 0), int)
            or not 0 <= int(report.get("residual_authority_steps", 0)) <= 100
            or any(
                _SHA256.fullmatch(str(report.get(name, ""))) is None
                for name in (
                    "teacher_report_hash",
                    "teacher_report_file_hash",
                    "teacher_corpus_hash",
                    "curriculum_manifest_hash",
                    "curriculum_archive_hash",
                    "body_hash",
                    "corpus_hash",
                    "student_model_hash",
                )
            )
        ):
            raise ValueError("impact-recovery distilled student integrity changed")
        return report
    finally:
        if declared is not None:
            report["report_hash"] = declared


def _student_policy(model: dict[str, np.ndarray[Any, Any]]) -> Any:
    mean = jnp.asarray(model["input_mean"])
    std = jnp.asarray(model["input_std"])
    ridge = set(model) == {"input_mean", "input_std", "weight", "bias"}
    if ridge:
        weight = jnp.asarray(model["weight"])
        bias = jnp.asarray(model["bias"])
    else:
        w1, b1 = jnp.asarray(model["w1"]), jnp.asarray(model["b1"])
        w2, b2 = jnp.asarray(model["w2"]), jnp.asarray(model["b2"])
        w3, b3 = jnp.asarray(model["w3"]), jnp.asarray(model["b3"])

    def make_policy(unused: Any) -> Any:
        del unused

        def policy(observation: jax.Array, rng: jax.Array) -> tuple[jax.Array, dict[str, Any]]:
            del rng
            selected = observation[..., -mean.size :]
            normalized = jnp.clip((selected - mean) / std, -10.0, 10.0)
            if ridge:
                return jnp.clip(normalized @ weight + bias, -1.0, 1.0), {}
            hidden1 = jnp.tanh(normalized @ w1 + b1)
            hidden2 = jnp.tanh(hidden1 @ w2 + b2)
            return jnp.tanh(hidden2 @ w3 + b3), {}

        return policy

    return make_policy


def evaluate_impact_recovery_distilled_student(
    *,
    asset_root: Path,
    curriculum_manifest_path: Path,
    student_report_path: Path,
    output_dir: Path,
    source_checkout_path: Path,
    config: ImpactRecoveryMJXEvaluationConfig | None = None,
) -> dict[str, Any]:
    """Run the fixed acquisition/retention exam for a distilled candidate."""

    from brax.envs.wrappers import training as training_wrappers
    from brax.training import acting

    active = config or ImpactRecoveryMJXEvaluationConfig(seeds=(57_151, 57_152, 57_153, 57_154))
    root = asset_root.expanduser().resolve()
    curriculum_path = curriculum_manifest_path.expanduser().resolve()
    student_path = student_report_path.expanduser().resolve()
    destination = output_dir.expanduser().resolve()
    checkout = source_checkout_path.expanduser().resolve()
    model_path = root / "g1_description" / "g1_liao.xml"
    if any(not path.is_file() for path in (model_path, curriculum_path, student_path)):
        raise FileNotFoundError("impact-recovery distilled evaluation inputs are incomplete")
    if destination.exists() or destination == checkout or checkout in destination.parents:
        raise ValueError("impact-recovery distilled evaluation output must be new and external")
    student = validate_impact_recovery_distilled_student(student_path)
    if student.get("student_exam_eligible") is not True:
        raise ValueError("impact-recovery distilled student did not pass its holdout exam")
    manifest = validate_impact_recovery_curriculum(curriculum_path)
    if (
        student.get("curriculum_manifest_hash") != manifest.get("manifest_hash")
        or manifest.get("training_model_hash") != hash_bytes(model_path.read_bytes())
        or manifest.get("body_hash") != g1_body_hash(root)
    ):
        raise ValueError("impact-recovery distilled evaluation lineage changed")
    archive_path = curriculum_path.parent / str(manifest["archive"])
    with np.load(archive_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(
        student_path.parent / str(student["student_model"]), allow_pickle=False
    ) as archive:
        model = {name: np.asarray(archive[name]) for name in archive.files}
    controller = ImpactRecoveryMJXConfig(
        retention_memory_mode="DIRECT_REPLAY",
        gain_memory_mode="DYNAMIC",
        residual_gate_mode="TEACHER_NOVELTY",
        residual_authority_steps=int(student.get("residual_authority_steps", 0)),
    )
    policy = _student_policy(model)
    populations: dict[str, Any] = {}
    for population in ("ACQUISITION", "RETENTION"):
        environment = ImpactRecoveryMJXEnv(
            model_path=model_path,
            curriculum_arrays=arrays,
            desired_heading_rad=float(
                cast(dict[str, Any], manifest["config"])["desired_heading_rad"]
            ),
            reset_population=cast(Any, population),
            config=controller,
        )
        wrapped = training_wrappers.wrap(
            environment,
            episode_length=controller.episode_length,
            action_repeat=1,
        )
        evaluator = acting.Evaluator(
            wrapped,
            policy,
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
        success_count = sum(int(row["success_count"]) for row in repeats)
        episode_count = active.num_envs * len(active.seeds)
        populations[population.lower()] = {
            "episode_count": episode_count,
            "success_count": success_count,
            "success_rate": success_count / episode_count,
            "repeats": repeats,
        }
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.impact_recovery_distilled_evaluation.v1",
        "config": asdict(active),
        "config_hash": active.config_hash,
        "student_report_hash": student["report_hash"],
        "student_model_hash": student["student_model_hash"],
        "curriculum_manifest_hash": manifest["manifest_hash"],
        "populations": populations,
        "physics_backend": "MUJOCO_MJX",
        "promotion_eligible": False,
        "promotion_authority": "NONE",
        "claim_boundary": "Fixed isolated exam; CPU-MuJoCo full-chain exam required",
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    destination.mkdir(parents=True)
    report_path = destination / "evaluation-report.json"
    _atomic_json(report_path, report)
    return validate_impact_recovery_distilled_evaluation(report_path)


def validate_impact_recovery_distilled_evaluation(path: Path) -> dict[str, Any]:
    """Validate a fixed acquisition/retention exam without trusting its producer."""

    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("impact-recovery distilled evaluation must be an object")
    report = cast(dict[str, Any], payload)
    declared = report.pop("report_hash", None)
    try:
        config_value = report.get("config")
        populations = report.get("populations")
        if not isinstance(config_value, dict) or not isinstance(populations, dict):
            raise ValueError("impact-recovery distilled evaluation is incomplete")
        config = ImpactRecoveryMJXEvaluationConfig(**config_value)
        if set(populations) != {"acquisition", "retention"}:
            raise ValueError("impact-recovery distilled evaluation populations changed")
        expected_episode_count = config.num_envs * len(config.seeds)
        for population in ("acquisition", "retention"):
            row = populations[population]
            if not isinstance(row, dict):
                raise ValueError("impact-recovery distilled evaluation population is invalid")
            repeats = row.get("repeats")
            success_count = row.get("success_count")
            if (
                not isinstance(repeats, list)
                or len(repeats) != len(config.seeds)
                or not isinstance(success_count, int)
                or not 0 <= success_count <= expected_episode_count
                or row.get("episode_count") != expected_episode_count
                or row.get("success_rate") != success_count / expected_episode_count
            ):
                raise ValueError("impact-recovery distilled evaluation totals changed")
            repeat_success_count = 0
            for expected_seed, repeat in zip(config.seeds, repeats, strict=True):
                if not isinstance(repeat, dict):
                    raise ValueError("impact-recovery distilled evaluation repeat is invalid")
                repeat_count = repeat.get("success_count")
                if (
                    repeat.get("seed") != expected_seed
                    or not isinstance(repeat_count, int)
                    or not 0 <= repeat_count <= config.num_envs
                    or repeat.get("success_rate") != repeat_count / config.num_envs
                ):
                    raise ValueError("impact-recovery distilled evaluation repeat changed")
                repeat_success_count += repeat_count
            if repeat_success_count != success_count:
                raise ValueError("impact-recovery distilled evaluation repeat totals changed")
        if (
            report.get("schema_version") != "rosclaw_soccer.impact_recovery_distilled_evaluation.v1"
            or declared != hash_json(report)
            or report.get("config_hash") != hash_json(config_value)
            or report.get("physics_backend") != "MUJOCO_MJX"
            or report.get("promotion_eligible") is not False
            or report.get("promotion_authority") != "NONE"
            or report.get("activation_ceiling") != "SIM_ONLY"
            or report.get("hardware_authorized") is not False
            or report.get("hardware_command_sent") is not False
            or any(
                _SHA256.fullmatch(str(report.get(name, ""))) is None
                for name in (
                    "student_report_hash",
                    "student_model_hash",
                    "curriculum_manifest_hash",
                )
            )
        ):
            raise ValueError("impact-recovery distilled evaluation integrity changed")
        return report
    finally:
        if declared is not None:
            report["report_hash"] = declared


__all__ = [
    "ImpactRecoveryDistillationConfig",
    "build_impact_recovery_distilled_student",
    "evaluate_impact_recovery_distilled_student",
    "validate_impact_recovery_distilled_evaluation",
    "validate_impact_recovery_distilled_student",
]
