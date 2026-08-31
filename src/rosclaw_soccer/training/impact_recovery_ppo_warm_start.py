"""Content-bound supervised warm start for the impact-recovery PPO actor."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax.training.agents.ppo import checkpoint

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.impact_recovery_distillation import (
    validate_impact_recovery_distilled_student,
)
from rosclaw_soccer.training.impact_recovery_mjx import (
    ImpactRecoveryMJXConfig,
    _tree_hash,
    validate_impact_recovery_mjx_report,
)
from rosclaw_soccer.training.opentrack_recovery_mjx_ppo import (
    _make_recovery_ppo_networks,
)
from rosclaw_soccer.training.recovery_mjx import _atomic_json

_JOINT_COUNT = 29
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True)
class ImpactRecoveryPPOWarmStartConfig:
    """Whole-state split and bounded actor pretraining settings."""

    training_steps: int = 2_000
    batch_size: int = 256
    learning_rate: float = 3.0e-4
    calibration_state_count: int = 4
    exam_state_count: int = 4
    selection_interval_steps: int = 50
    maximum_gradient_norm: float = 1.0
    required_calibration_improvement_fraction: float = 0.10
    required_exam_improvement_fraction: float = 0.0
    trainable_scope: Literal["ALL_POLICY", "TAIL_128_LOCATION", "LOCATION_HEAD"] = "ALL_POLICY"
    random_seed: int = 117_201
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.impact_recovery_ppo_warm_start_config.v1"

    def __post_init__(self) -> None:
        finite = (
            self.learning_rate,
            self.maximum_gradient_norm,
            self.required_calibration_improvement_fraction,
            self.required_exam_improvement_fraction,
        )
        if (
            any(not math.isfinite(value) for value in finite)
            or not 100 <= self.training_steps <= 100_000
            or not 16 <= self.batch_size <= 8_192
            or not 1.0e-6 <= self.learning_rate <= 1.0e-2
            or not 2 <= self.calibration_state_count <= 32
            or not 2 <= self.exam_state_count <= 32
            or self.calibration_state_count + self.exam_state_count >= 64
            or not 10 <= self.selection_interval_steps <= self.training_steps
            or self.training_steps % self.selection_interval_steps
            or not 0.0 < self.maximum_gradient_norm <= 10.0
            or not 0.0 <= self.required_calibration_improvement_fraction <= 1.0
            or not 0.0 <= self.required_exam_improvement_fraction <= 1.0
            or self.trainable_scope not in {"ALL_POLICY", "TAIL_128_LOCATION", "LOCATION_HEAD"}
            or not 0 <= self.random_seed < 2**31
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("impact-recovery PPO warm-start config is invalid")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


def _checkpoint_bound(
    *,
    training: dict[str, Any],
    training_path: Path,
    selected_checkpoint: Path,
) -> tuple[str, list[dict[str, Any]]]:
    checkpoint_root = (training_path.parent / "checkpoints").resolve()
    if checkpoint_root not in selected_checkpoint.parents:
        raise ValueError("impact-recovery warm-start checkpoint escaped its training tree")
    selected_hash, selected_rows = _tree_hash(selected_checkpoint)
    prefix = selected_checkpoint.relative_to(checkpoint_root).as_posix() + "/"
    declared = [
        {**row, "path": str(row["path"])[len(prefix) :]}
        for row in cast(list[dict[str, Any]], training["checkpoint_files"])
        if str(row.get("path", "")).startswith(prefix)
    ]
    if declared != selected_rows:
        raise ValueError("impact-recovery warm-start parent checkpoint bytes changed")
    return selected_hash, selected_rows


def _whole_state_split(
    state_index: np.ndarray[Any, Any],
    config: ImpactRecoveryPPOWarmStartConfig,
) -> dict[str, np.ndarray[Any, Any]]:
    unique = np.unique(state_index)
    required = config.calibration_state_count + config.exam_state_count + 2
    if unique.size < required:
        raise ValueError("impact-recovery warm-start has too few independent teacher states")
    permutation = np.random.default_rng(config.random_seed).permutation(unique)
    exam = np.sort(permutation[: config.exam_state_count])
    calibration = np.sort(
        permutation[
            config.exam_state_count : config.exam_state_count + config.calibration_state_count
        ]
    )
    training = np.sort(permutation[config.exam_state_count + config.calibration_state_count :])
    return {"training": training, "calibration": calibration, "exam": exam}


def _loss_metrics(
    *,
    policy_network: Any,
    normalizer: Any,
    policy_params: Any,
    observation: jax.Array,
    target: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    logits = policy_network.apply(normalizer, policy_params, observation)
    action = jnp.tanh(logits[..., :_JOINT_COUNT])
    return jnp.mean(jnp.square(action - target)), jnp.sqrt(jnp.mean(jnp.square(action)))


def build_impact_recovery_ppo_warm_start(
    *,
    parent_training_report_path: Path,
    parent_checkpoint_path: Path,
    distillation_report_path: Path,
    output_dir: Path,
    source_checkout_path: Path,
    config: ImpactRecoveryPPOWarmStartConfig | None = None,
) -> dict[str, Any]:
    """Fit the PPO actor to successor labels without touching critic or authority."""

    active = config or ImpactRecoveryPPOWarmStartConfig()
    training_path = parent_training_report_path.expanduser().resolve()
    selected_checkpoint = parent_checkpoint_path.expanduser().resolve()
    distillation_path = distillation_report_path.expanduser().resolve()
    destination = output_dir.expanduser().resolve()
    checkout = source_checkout_path.expanduser().resolve()
    if (
        not training_path.is_file()
        or not distillation_path.is_file()
        or not selected_checkpoint.is_dir()
        or not (selected_checkpoint / "ppo_network_config.json").is_file()
    ):
        raise FileNotFoundError("impact-recovery PPO warm-start inputs are incomplete")
    if (
        destination.exists()
        or destination == checkout
        or checkout in destination.parents
        or destination in checkout.parents
    ):
        raise ValueError("impact-recovery PPO warm-start output must be new and external")
    devices = tuple(jax.devices())
    if not devices or any(getattr(device, "platform", "") != "cpu" for device in devices):
        raise RuntimeError("impact-recovery PPO warm-start requires CPU-only JAX visibility")
    training = validate_impact_recovery_mjx_report(training_path)
    distillation = validate_impact_recovery_distilled_student(distillation_path)
    parent_hash, parent_files = _checkpoint_bound(
        training=training,
        training_path=training_path,
        selected_checkpoint=selected_checkpoint,
    )
    training_config = ImpactRecoveryMJXConfig(**cast(dict[str, Any], training["config"]))
    if (
        training_config.gain_memory_mode != "DYNAMIC"
        or training_config.residual_gate_mode != "TEACHER_NOVELTY"
        or training.get("curriculum_manifest_hash") != distillation.get("curriculum_manifest_hash")
        or training.get("body_hash") != distillation.get("body_hash")
        or distillation.get("student_exam_eligible") is not True
    ):
        raise ValueError("impact-recovery PPO warm-start lineage or actor contract changed")
    corpus_path = distillation_path.parent / str(distillation["corpus"])
    if not corpus_path.is_file() or hash_bytes(corpus_path.read_bytes()) != distillation.get(
        "corpus_hash"
    ):
        raise ValueError("impact-recovery PPO warm-start corpus bytes changed")
    with np.load(corpus_path, allow_pickle=False) as archive:
        required = {
            "actor_observation",
            "commanded_action",
            "curriculum_index",
            "accepted_state_row",
        }
        if not required.issubset(archive.files):
            raise ValueError("impact-recovery PPO warm-start corpus is incomplete")
        observation = np.asarray(archive["actor_observation"], dtype=np.float32)
        target = np.asarray(archive["commanded_action"], dtype=np.float32)
        state_index = np.asarray(archive["curriculum_index"], dtype=np.int32)
        accepted = np.asarray(archive["accepted_state_row"], dtype=np.bool_)
    if (
        observation.ndim != 2
        or observation.shape[1] != training_config.observation_dim
        or target.shape != (observation.shape[0], _JOINT_COUNT)
        or state_index.shape != (observation.shape[0],)
        or accepted.shape != (observation.shape[0],)
        or not np.any(accepted)
        or not np.all(np.isfinite(observation))
        or not np.all(np.isfinite(target))
        or np.any(np.abs(target) > 1.0 + 1.0e-6)
    ):
        raise ValueError("impact-recovery PPO warm-start samples are invalid")
    observation = observation[accepted]
    target = target[accepted]
    state_index = state_index[accepted]
    splits = _whole_state_split(state_index, active)
    masks = {name: np.isin(state_index, indexes) for name, indexes in splits.items()}
    if any(not np.any(mask) for mask in masks.values()):
        raise ValueError("impact-recovery PPO warm-start split is empty")

    parent_params = checkpoint.load(selected_checkpoint)
    if not isinstance(parent_params, list) or len(parent_params) != 3:
        raise ValueError("impact-recovery PPO warm-start parent parameter tree changed")
    network_config = checkpoint.load_config(selected_checkpoint)
    network = checkpoint._get_ppo_network(network_config, _make_recovery_ppo_networks)
    normalizer, initial_policy_params, critic_params = parent_params
    optimizer = optax.chain(
        optax.clip_by_global_norm(active.maximum_gradient_norm),
        optax.adam(active.learning_rate),
    )
    optimizer_state = optimizer.init(initial_policy_params)
    trainable_mask = jax.tree_util.tree_map(lambda unused: True, initial_policy_params)
    if active.trainable_scope != "ALL_POLICY":
        trainable_modules = (
            {"Dense_2", "location"}
            if active.trainable_scope == "TAIL_128_LOCATION"
            else {"location"}
        )
        trainable_mask = {
            "params": {
                name: jax.tree_util.tree_map(
                    lambda unused, module_name=name: module_name in trainable_modules, values
                )
                for name, values in cast(dict[str, Any], initial_policy_params["params"]).items()
            }
        }
    training_observation = jnp.asarray(observation[masks["training"]])
    training_target = jnp.asarray(target[masks["training"]])
    calibration_observation = jnp.asarray(observation[masks["calibration"]])
    calibration_target = jnp.asarray(target[masks["calibration"]])
    exam_observation = jnp.asarray(observation[masks["exam"]])
    exam_target = jnp.asarray(target[masks["exam"]])

    def loss_fn(policy_params: Any, batch_observation: jax.Array, batch_target: jax.Array) -> Any:
        return _loss_metrics(
            policy_network=network.policy_network,
            normalizer=normalizer,
            policy_params=policy_params,
            observation=batch_observation,
            target=batch_target,
        )[0]

    @jax.jit  # type: ignore[untyped-decorator]
    def update(
        policy_params: Any,
        current_optimizer_state: Any,
        batch_observation: jax.Array,
        batch_target: jax.Array,
    ) -> tuple[Any, Any, jax.Array]:
        loss, gradients = jax.value_and_grad(loss_fn)(
            policy_params, batch_observation, batch_target
        )
        gradients = jax.tree_util.tree_map(
            lambda gradient, trainable: gradient if trainable else jnp.zeros_like(gradient),
            gradients,
            trainable_mask,
        )
        updates, next_optimizer_state = optimizer.update(
            gradients, current_optimizer_state, policy_params
        )
        return optax.apply_updates(policy_params, updates), next_optimizer_state, loss

    zero_calibration_loss = float(jnp.mean(jnp.square(calibration_target)))
    parent_calibration_loss, parent_calibration_rms = _loss_metrics(
        policy_network=network.policy_network,
        normalizer=normalizer,
        policy_params=initial_policy_params,
        observation=calibration_observation,
        target=calibration_target,
    )
    params = initial_policy_params
    best_params = initial_policy_params
    best_step = 0
    best_calibration_loss = float(parent_calibration_loss)
    progress: list[dict[str, Any]] = []
    rng = np.random.default_rng(active.random_seed + 1)
    started = time.perf_counter()
    for step in range(1, active.training_steps + 1):
        batch_indexes = rng.integers(0, training_observation.shape[0], size=active.batch_size)
        params, optimizer_state, batch_loss = update(
            params,
            optimizer_state,
            training_observation[batch_indexes],
            training_target[batch_indexes],
        )
        if step % active.selection_interval_steps == 0:
            calibration_loss, calibration_rms = _loss_metrics(
                policy_network=network.policy_network,
                normalizer=normalizer,
                policy_params=params,
                observation=calibration_observation,
                target=calibration_target,
            )
            row = {
                "step": step,
                "training_batch_loss": float(batch_loss),
                "calibration_loss": float(calibration_loss),
                "calibration_action_rms": float(calibration_rms),
            }
            progress.append(row)
            if float(calibration_loss) < best_calibration_loss:
                best_calibration_loss = float(calibration_loss)
                best_step = step
                best_params = jax.tree_util.tree_map(lambda value: jnp.array(value), params)
    training_sec = time.perf_counter() - started
    exam_loss, exam_action_rms = _loss_metrics(
        policy_network=network.policy_network,
        normalizer=normalizer,
        policy_params=best_params,
        observation=exam_observation,
        target=exam_target,
    )
    parent_exam_loss, parent_exam_action_rms = _loss_metrics(
        policy_network=network.policy_network,
        normalizer=normalizer,
        policy_params=initial_policy_params,
        observation=exam_observation,
        target=exam_target,
    )
    zero_exam_loss = float(jnp.mean(jnp.square(exam_target)))
    calibration_improvement = (float(parent_calibration_loss) - best_calibration_loss) / max(
        float(parent_calibration_loss), 1.0e-12
    )
    exam_improvement = (float(parent_exam_loss) - float(exam_loss)) / max(
        float(parent_exam_loss), 1.0e-12
    )
    warm_start_eligible = bool(
        best_step > 0
        and calibration_improvement >= active.required_calibration_improvement_fraction
        and exam_improvement >= active.required_exam_improvement_fraction
    )
    destination.mkdir(parents=True)
    checkpoint_root = destination / "checkpoints"
    checkpoint.save(
        checkpoint_root,
        1,
        [normalizer, best_params, critic_params],
        network_config,
    )
    warm_checkpoint = checkpoint_root / "000000000001"
    warm_hash, warm_files = _tree_hash(warm_checkpoint)
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.impact_recovery_ppo_warm_start.v1",
        "config": asdict(active),
        "config_hash": active.config_hash,
        "parent_training_report_hash": training["report_hash"],
        "parent_training_report_file_hash": hash_bytes(training_path.read_bytes()),
        "parent_checkpoint_hash": parent_hash,
        "parent_checkpoint_files": parent_files,
        "distillation_report_hash": distillation["report_hash"],
        "distillation_report_file_hash": hash_bytes(distillation_path.read_bytes()),
        "teacher_report_hash": distillation["teacher_report_hash"],
        "corpus_hash": distillation["corpus_hash"],
        "curriculum_manifest_hash": distillation["curriculum_manifest_hash"],
        "body_hash": distillation["body_hash"],
        "split_semantics": "WHOLE_CURRICULUM_STATE_TRAIN_CALIBRATION_SEALED_EXAM",
        "split_states": {name: values.tolist() for name, values in splits.items()},
        "sample_counts": {name: int(np.sum(mask)) for name, mask in masks.items()},
        "training_backend": "JAX_CPU_SUPERVISED_ACTOR_ONLY",
        "trainable_scope": active.trainable_scope,
        "normalizer_frozen": True,
        "critic_frozen": True,
        "scale_head_frozen_by_zero_gradient": True,
        "selection_metric": "CALIBRATION_ACTION_MSE",
        "best_step": best_step,
        "training_sec": training_sec,
        "progress": progress,
        "metrics": {
            "zero_calibration_loss": zero_calibration_loss,
            "parent_calibration_loss": float(parent_calibration_loss),
            "parent_calibration_action_rms": float(parent_calibration_rms),
            "best_calibration_loss": best_calibration_loss,
            "calibration_improvement_fraction": calibration_improvement,
            "zero_exam_loss": zero_exam_loss,
            "parent_exam_loss": float(parent_exam_loss),
            "parent_exam_action_rms": float(parent_exam_action_rms),
            "warm_exam_loss": float(exam_loss),
            "warm_exam_action_rms": float(exam_action_rms),
            "exam_improvement_fraction": exam_improvement,
        },
        "warm_checkpoint": "checkpoints/000000000001",
        "warm_checkpoint_hash": warm_hash,
        "warm_checkpoint_files": warm_files,
        "warm_start_eligible": warm_start_eligible,
        "deployment_candidate": False,
        "promotion_eligible": False,
        "promotion_authority": "NONE",
        "claim_boundary": (
            "Supervised actor initialization; online RL and matched physics exams required"
        ),
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    report_path = destination / "warm-start-report.json"
    _atomic_json(report_path, report)
    return validate_impact_recovery_ppo_warm_start(report_path)


def validate_impact_recovery_ppo_warm_start(path: Path) -> dict[str, Any]:
    """Validate actor warm-start evidence and its local checkpoint bytes."""

    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("impact-recovery PPO warm-start report must be an object")
    report = cast(dict[str, Any], payload)
    declared = report.pop("report_hash", None)
    try:
        config_value = report.get("config")
        metrics = report.get("metrics")
        split_states = report.get("split_states")
        sample_counts = report.get("sample_counts")
        if (
            not isinstance(config_value, dict)
            or not isinstance(metrics, dict)
            or not isinstance(split_states, dict)
            or not isinstance(sample_counts, dict)
        ):
            raise ValueError("impact-recovery PPO warm-start report is incomplete")
        config = ImpactRecoveryPPOWarmStartConfig(**config_value)
        split_valid = bool(
            set(split_states) == {"training", "calibration", "exam"}
            and all(isinstance(values, list) and values for values in split_states.values())
            and all(
                isinstance(value, int) and not isinstance(value, bool)
                for values in split_states.values()
                for value in values
            )
            and len(
                set(split_states["training"])
                | set(split_states["calibration"])
                | set(split_states["exam"])
            )
            == sum(len(values) for values in split_states.values())
            and len(split_states["calibration"]) == config.calibration_state_count
            and len(split_states["exam"]) == config.exam_state_count
            and set(sample_counts) == set(split_states)
            and all(
                isinstance(value, int) and not isinstance(value, bool) and value > 0
                for value in sample_counts.values()
            )
        )
        checkpoint_name = report.get("warm_checkpoint")
        checkpoint_path = (resolved.parent / str(checkpoint_name)).resolve()
        checkpoint_files = report.get("warm_checkpoint_files")
        checkpoint_manifest_valid = False
        if (
            isinstance(checkpoint_name, str)
            and not Path(checkpoint_name).is_absolute()
            and ".." not in Path(checkpoint_name).parts
            and resolved.parent in checkpoint_path.parents
            and checkpoint_path.is_dir()
            and isinstance(checkpoint_files, list)
            and checkpoint_files
        ):
            try:
                checkpoint_hash, actual_files = _tree_hash(checkpoint_path)
            except ValueError:
                checkpoint_hash, actual_files = "", []
            checkpoint_manifest_valid = bool(
                actual_files == checkpoint_files
                and checkpoint_hash == report.get("warm_checkpoint_hash")
            )
        parent_files = report.get("parent_checkpoint_files")
        parent_manifest_valid = bool(
            isinstance(parent_files, list)
            and parent_files
            and report.get("parent_checkpoint_hash") == hash_json(parent_files)
        )
        best_step = report.get("best_step")
        calibration_improvement = metrics.get("calibration_improvement_fraction")
        exam_improvement = metrics.get("exam_improvement_fraction")
        expected_eligible = bool(
            isinstance(best_step, int)
            and not isinstance(best_step, bool)
            and 0 < best_step <= config.training_steps
            and best_step % config.selection_interval_steps == 0
            and isinstance(calibration_improvement, int | float)
            and not isinstance(calibration_improvement, bool)
            and math.isfinite(float(calibration_improvement))
            and float(calibration_improvement) >= config.required_calibration_improvement_fraction
            and isinstance(exam_improvement, int | float)
            and not isinstance(exam_improvement, bool)
            and math.isfinite(float(exam_improvement))
            and float(exam_improvement) >= config.required_exam_improvement_fraction
        )
        if (
            report.get("schema_version") != "rosclaw_soccer.impact_recovery_ppo_warm_start.v1"
            or declared != hash_json(report)
            or report.get("config_hash") != config.config_hash
            or not split_valid
            or not checkpoint_manifest_valid
            or not parent_manifest_valid
            or report.get("warm_start_eligible") is not expected_eligible
            or report.get("training_backend") != "JAX_CPU_SUPERVISED_ACTOR_ONLY"
            or report.get("trainable_scope") != config.trainable_scope
            or report.get("normalizer_frozen") is not True
            or report.get("critic_frozen") is not True
            or report.get("scale_head_frozen_by_zero_gradient") is not True
            or report.get("deployment_candidate") is not False
            or report.get("promotion_eligible") is not False
            or report.get("promotion_authority") != "NONE"
            or report.get("activation_ceiling") != "SIM_ONLY"
            or report.get("hardware_authorized") is not False
            or report.get("hardware_command_sent") is not False
            or any(
                _SHA256.fullmatch(str(report.get(name, ""))) is None
                for name in (
                    "parent_training_report_hash",
                    "parent_training_report_file_hash",
                    "parent_checkpoint_hash",
                    "distillation_report_hash",
                    "distillation_report_file_hash",
                    "teacher_report_hash",
                    "corpus_hash",
                    "curriculum_manifest_hash",
                    "body_hash",
                    "warm_checkpoint_hash",
                )
            )
        ):
            raise ValueError("impact-recovery PPO warm-start authority or integrity changed")
        return report
    finally:
        if declared is not None:
            report["report_hash"] = declared


def _main() -> None:
    parser = argparse.ArgumentParser(description="Build a CPU-supervised PPO actor warm start")
    parser.add_argument("--parent-training-report", required=True, type=Path)
    parser.add_argument("--parent-checkpoint", required=True, type=Path)
    parser.add_argument("--distillation-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-checkout", required=True, type=Path)
    parser.add_argument("--training-steps", default=2_000, type=int)
    args = parser.parse_args()
    report = build_impact_recovery_ppo_warm_start(
        parent_training_report_path=args.parent_training_report,
        parent_checkpoint_path=args.parent_checkpoint,
        distillation_report_path=args.distillation_report,
        output_dir=args.output_dir,
        source_checkout_path=args.source_checkout,
        config=ImpactRecoveryPPOWarmStartConfig(training_steps=args.training_steps),
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    _main()


__all__ = [
    "ImpactRecoveryPPOWarmStartConfig",
    "build_impact_recovery_ppo_warm_start",
    "validate_impact_recovery_ppo_warm_start",
]
