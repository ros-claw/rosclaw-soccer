"""Train a reference-free temporal recovery student on one or many GPUs.

Torch is imported only inside the training entry point so the default Soccer
Academy installation keeps its lightweight, torch-optional import boundary.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.recovery_student import (
    RecoveryDistillationCorpus,
    load_recovery_distillation_corpus,
    normalize_absolute_motor_targets,
)


@dataclass(frozen=True)
class RecoveryStudentTrainingConfig:
    """Frozen S52 distillation recipe."""

    epochs: int = 100
    minibatch_size_per_rank: int = 256
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-5
    frame_hidden_size: int = 128
    recurrent_hidden_size: int = 256
    huber_delta: float = 0.04
    action_delta_loss_weight: float = 0.15
    ready_frame_weight: float = 2.0
    initial_window_steps: int = 75
    initial_frame_weight: float = 6.0
    hard_joint_loss_weight: float = 0.20
    gradient_clip_norm: float = 1.0
    random_seed: int = 52_001
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.recovery_student_training_config.v1"

    def __post_init__(self) -> None:
        finite_positive = (
            self.learning_rate,
            self.weight_decay,
            self.huber_delta,
            self.ready_frame_weight,
            self.initial_frame_weight,
            self.gradient_clip_norm,
        )
        if (
            not 1 <= self.epochs <= 10_000
            or not 8 <= self.minibatch_size_per_rank <= 16_384
            or not 16 <= self.frame_hidden_size <= 2_048
            or not 32 <= self.recurrent_hidden_size <= 4_096
            or not 1 <= self.initial_window_steps <= 500
            or any(not math.isfinite(value) or value <= 0.0 for value in finite_positive)
            or not math.isfinite(self.action_delta_loss_weight)
            or not 0.0 <= self.action_delta_loss_weight <= 10.0
            or not math.isfinite(self.hard_joint_loss_weight)
            or not 0.0 <= self.hard_joint_loss_weight <= 10.0
            or not 0 <= self.random_seed < 2**31
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("recovery student training config is invalid")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass(frozen=True)
class RecoveryStudentSamples:
    history: NDArray[np.float32]
    normalized_target: NDArray[np.float32]
    normalized_previous_target: NDArray[np.float32]
    ready_handoff: NDArray[np.bool_]
    episode_index: NDArray[np.int32]
    control_step: NDArray[np.int32]

    def __post_init__(self) -> None:
        count = self.history.shape[0]
        if (
            self.history.ndim != 3
            or self.history.shape[2] != 93
            or self.normalized_target.shape != (count, 29)
            or self.normalized_previous_target.shape != (count, 29)
            or self.ready_handoff.shape != (count,)
            or self.episode_index.shape != (count,)
            or self.control_step.shape != (count,)
            or not np.all(np.isfinite(self.history))
            or not np.all(np.isfinite(self.normalized_target))
            or not np.all(np.isfinite(self.normalized_previous_target))
        ):
            raise ValueError("recovery student samples are invalid")


def split_recovery_student_episodes(
    corpus: RecoveryDistillationCorpus,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Hold out one entire route-neighborhood episode per base posture."""

    grouped: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for row in corpus.rows:
        index = int(row["episode_index"])
        base_hash = str(row["base_snapshot_hash"])
        initial_hash = str(row["initial_snapshot_hash"])
        grouped[base_hash][initial_hash].append(index)
    if not grouped or any(len(items) < 2 for items in grouped.values()):
        raise ValueError("student split needs at least two episodes per base posture")
    training: list[int] = []
    validation: list[int] = []
    for base_hash, initial_groups in sorted(grouped.items()):
        ranked = sorted(
            initial_groups,
            key=lambda initial_hash: hash_json(
                {"base_snapshot_hash": base_hash, "initial_snapshot_hash": initial_hash}
            ),
        )
        validation.extend(initial_groups[ranked[-1]])
        for initial_hash in ranked[:-1]:
            training.extend(initial_groups[initial_hash])
    if set(training) & set(validation) or set(training) | set(validation) != {
        int(row["episode_index"]) for row in corpus.rows
    }:
        raise ValueError("student episode split is incomplete")
    return tuple(sorted(training)), tuple(sorted(validation))


def build_recovery_student_samples(
    corpus: RecoveryDistillationCorpus,
    *,
    episode_indexes: tuple[int, ...],
) -> RecoveryStudentSamples:
    """Construct left-padded histories without crossing episode boundaries."""

    if not episode_indexes or len(set(episode_indexes)) != len(episode_indexes):
        raise ValueError("recovery student episode indexes are invalid")
    history_steps = corpus.proprioception_spec.history_steps
    histories: list[NDArray[np.float32]] = []
    targets: list[NDArray[np.float32]] = []
    previous_targets: list[NDArray[np.float32]] = []
    readiness: list[NDArray[np.bool_]] = []
    indexes: list[NDArray[np.int32]] = []
    steps: list[NDArray[np.int32]] = []
    row_by_index = {int(row["episode_index"]): row for row in corpus.rows}
    normalized = normalize_absolute_motor_targets(
        corpus.absolute_motor_targets_rad,
        joint_lower_rad=corpus.joint_lower_rad,
        joint_upper_rad=corpus.joint_upper_rad,
    )
    for episode_index in episode_indexes:
        row = row_by_index.get(episode_index)
        if row is None:
            raise ValueError("recovery student episode index is absent")
        start = int(row["start_row"])
        count = int(row["row_count"])
        stop = start + count
        frames = corpus.proprio[start:stop]
        if frames.shape != (count, corpus.proprioception_spec.observation_dim):
            raise ValueError("recovery student episode range is invalid")
        padded = np.concatenate(
            (np.repeat(frames[:1], history_steps - 1, axis=0), frames), axis=0
        )
        windows = np.lib.stride_tricks.sliding_window_view(
            padded,
            window_shape=history_steps,
            axis=0,
        ).transpose(0, 2, 1)
        episode_target = normalized[start:stop]
        previous = np.concatenate((episode_target[:1], episode_target[:-1]), axis=0)
        histories.append(np.asarray(windows, dtype=np.float32).copy())
        targets.append(np.asarray(episode_target, dtype=np.float32))
        previous_targets.append(np.asarray(previous, dtype=np.float32))
        readiness.append(np.asarray(corpus.ready_handoff[start:stop], dtype=np.bool_))
        indexes.append(np.full(count, episode_index, dtype=np.int32))
        steps.append(np.arange(count, dtype=np.int32))
    return RecoveryStudentSamples(
        history=np.concatenate(histories),
        normalized_target=np.concatenate(targets),
        normalized_previous_target=np.concatenate(previous_targets),
        ready_handoff=np.concatenate(readiness),
        episode_index=np.concatenate(indexes),
        control_step=np.concatenate(steps),
    )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def train_recovery_student(
    *,
    corpus_manifest_path: Path,
    output_dir: Path,
    config: RecoveryStudentTrainingConfig | None = None,
    device: str | None = None,
) -> dict[str, Any] | None:
    """Train with DDP when launched by torchrun; rank zero writes evidence."""

    import torch
    from safetensors.torch import save_file
    from torch import nn

    active = config or RecoveryStudentTrainingConfig()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("recovery student DDP requires CUDA")
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        active_device = torch.device(f"cuda:{local_rank}")
    else:
        active_device = torch.device(
            device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        )

    manifest_path = corpus_manifest_path.expanduser().resolve()
    target = output_dir.expanduser().resolve()
    creation_error: str | None = None
    if rank == 0:
        try:
            if target.exists():
                raise ValueError("recovery student trainer refuses to overwrite")
            target.mkdir(parents=True)
        except Exception as error:  # pragma: no cover - synchronized DDP path
            creation_error = f"{type(error).__name__}: {error}"
    if distributed:
        error_box = [creation_error]
        torch.distributed.broadcast_object_list(error_box, src=0)
        creation_error = error_box[0]
    if creation_error is not None:
        if distributed:
            torch.distributed.destroy_process_group()
        raise ValueError(creation_error)

    corpus = load_recovery_distillation_corpus(manifest_path)
    training_episodes, validation_episodes = split_recovery_student_episodes(corpus)
    train_samples = build_recovery_student_samples(
        corpus, episode_indexes=training_episodes
    )
    validation_samples = build_recovery_student_samples(
        corpus, episode_indexes=validation_episodes
    )
    observation_mean = np.mean(train_samples.history.reshape(-1, 93), axis=0).astype(
        np.float32
    )
    observation_std = np.std(train_samples.history.reshape(-1, 93), axis=0).astype(
        np.float32
    )
    observation_std = np.maximum(observation_std, 1.0e-4).astype(np.float32)

    class TemporalRecoveryStudent(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer(
                "observation_mean", torch.as_tensor(observation_mean)[None, None, :]
            )
            self.register_buffer(
                "observation_std", torch.as_tensor(observation_std)[None, None, :]
            )
            self.frame_encoder = nn.Sequential(
                nn.Linear(93, active.frame_hidden_size),
                nn.LayerNorm(active.frame_hidden_size),
                nn.SiLU(),
                nn.Linear(active.frame_hidden_size, active.frame_hidden_size),
                nn.SiLU(),
            )
            self.temporal = nn.GRU(
                input_size=active.frame_hidden_size,
                hidden_size=active.recurrent_hidden_size,
                batch_first=True,
            )
            self.motor_head = nn.Sequential(
                nn.Linear(active.recurrent_hidden_size, active.frame_hidden_size),
                nn.SiLU(),
                nn.Linear(active.frame_hidden_size, 29),
                nn.Tanh(),
            )

        def forward(self, proprio_history: Any) -> Any:
            normalized_history = (
                proprio_history - self.observation_mean
            ) / self.observation_std
            encoded = self.frame_encoder(normalized_history)
            temporal, _ = self.temporal(encoded)
            return self.motor_head(temporal[:, -1])

    random.seed(active.random_seed)
    np.random.seed(active.random_seed)
    torch.manual_seed(active.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(active.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    raw_model = TemporalRecoveryStudent().to(active_device)
    model: Any = raw_model
    if distributed:
        model = nn.parallel.DistributedDataParallel(raw_model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=active.learning_rate, weight_decay=active.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=active.epochs, eta_min=active.learning_rate * 0.05
    )

    train_history = torch.as_tensor(train_samples.history, device=active_device)
    train_target = torch.as_tensor(
        train_samples.normalized_target, device=active_device
    )
    train_previous = torch.as_tensor(
        train_samples.normalized_previous_target, device=active_device
    )
    train_ready = torch.as_tensor(
        train_samples.ready_handoff, dtype=torch.float32, device=active_device
    )
    train_step = torch.as_tensor(train_samples.control_step, device=active_device)
    validation_history = torch.as_tensor(
        validation_samples.history, device=active_device
    )
    validation_target = torch.as_tensor(
        validation_samples.normalized_target, device=active_device
    )
    lower = torch.as_tensor(corpus.joint_lower_rad, device=active_device)
    upper = torch.as_tensor(corpus.joint_upper_rad, device=active_device)
    radius = 0.5 * (upper - lower)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(active.random_seed)
    history: list[dict[str, Any]] = []
    sample_count = train_history.shape[0]
    padded_count = math.ceil(sample_count / world_size) * world_size
    for epoch in range(active.epochs):
        order = torch.randperm(sample_count, generator=generator)
        if padded_count > sample_count:
            order = torch.cat((order, order[: padded_count - sample_count]))
        shard = order[rank::world_size].to(active_device)
        model.train()
        loss_sum = torch.zeros((), dtype=torch.float64, device=active_device)
        seen = torch.zeros((), dtype=torch.float64, device=active_device)
        for start in range(0, shard.shape[0], active.minibatch_size_per_rank):
            indices = shard[start : start + active.minibatch_size_per_rank]
            predicted = model(train_history[indices])
            difference = predicted - train_target[indices]
            absolute = torch.abs(difference)
            huber = torch.where(
                absolute <= active.huber_delta,
                0.5 * torch.square(difference) / active.huber_delta,
                absolute - 0.5 * active.huber_delta,
            ).mean(dim=1)
            weights = 1.0 + train_ready[indices] * (active.ready_frame_weight - 1.0)
            weights += (train_step[indices] < active.initial_window_steps).float() * (
                active.initial_frame_weight - 1.0
            )
            position_loss = torch.mean(huber * weights)
            predicted_delta = predicted - train_previous[indices]
            target_delta = train_target[indices] - train_previous[indices]
            delta_loss = torch.mean(torch.square(predicted_delta - target_delta))
            hard_joint_loss = torch.mean(torch.max(absolute, dim=1).values * weights)
            loss = (
                position_loss
                + active.action_delta_loss_weight * delta_loss
                + active.hard_joint_loss_weight * hard_joint_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), active.gradient_clip_norm)
            optimizer.step()
            loss_sum += loss.detach().double() * indices.shape[0]
            seen += indices.shape[0]
        scheduler.step()
        if distributed:
            torch.distributed.all_reduce(loss_sum)
            torch.distributed.all_reduce(seen)
        if rank == 0 and (
            epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == active.epochs
        ):
            history.append(
                {
                    "epoch": epoch + 1,
                    "mean_training_loss": float(loss_sum / seen),
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                }
            )

    raw_model.eval()
    validation_ids = torch.arange(
        rank, validation_history.shape[0], world_size, device=active_device
    )
    with torch.inference_mode():
        validation_prediction = raw_model(validation_history[validation_ids])
        error_rad = torch.abs(
            (validation_prediction - validation_target[validation_ids]) * radius
        )
        absolute_sum = torch.sum(error_rad, dtype=torch.float64)
        absolute_count = torch.tensor(
            error_rad.numel(), dtype=torch.float64, device=active_device
        )
        maximum_error = torch.max(error_rad)
        squared_sum = torch.sum(torch.square(error_rad), dtype=torch.float64)
    if distributed:
        torch.distributed.all_reduce(absolute_sum)
        torch.distributed.all_reduce(absolute_count)
        torch.distributed.all_reduce(maximum_error, op=torch.distributed.ReduceOp.MAX)
        torch.distributed.all_reduce(squared_sum)

    device_evidence = {
        "rank": rank,
        "local_rank": local_rank,
        "device": str(active_device),
        "device_name": (
            torch.cuda.get_device_name(local_rank)
            if active_device.type == "cuda"
            else "CPU"
        ),
        "cuda_visible_device_count": torch.cuda.device_count(),
    }
    all_devices: list[Any] = [None] * world_size if rank == 0 else []
    if distributed:
        torch.distributed.gather_object(
            device_evidence, all_devices if rank == 0 else None, dst=0
        )
    else:
        all_devices = [device_evidence]

    report: dict[str, Any] | None = None
    if rank == 0:
        from onnx import checker, load

        checkpoint_path = target / "recovery-student-v1.safetensors"
        state = {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in raw_model.state_dict().items()
        }
        save_file(
            state,
            str(checkpoint_path),
            metadata={
                "activation_ceiling": "SIM_ONLY",
                "hardware_authorized": "false",
                "corpus_manifest_hash": corpus.manifest_hash,
                "training_config_hash": active.config_hash,
                "proprioception_spec_hash": corpus.proprioception_spec.spec_hash,
            },
        )
        onnx_path = target / "recovery-student-v1.onnx"
        example = torch.zeros(
            1,
            corpus.proprioception_spec.history_steps,
            corpus.proprioception_spec.observation_dim,
            dtype=torch.float32,
            device=active_device,
        )
        torch.onnx.export(
            raw_model,
            (example,),
            str(onnx_path),
            input_names=["proprio_history"],
            output_names=["normalized_absolute_motor_target"],
            dynamic_axes={
                "proprio_history": {0: "batch"},
                "normalized_absolute_motor_target": {0: "batch"},
            },
            opset_version=17,
            dynamo=False,
        )
        checker.check_model(load(str(onnx_path)))
        artifact_manifest: dict[str, Any] = {
            "schema_version": "rosclaw_soccer.recovery_student_artifact.v1",
            "architecture": "FRAME_MLP_GRU_ABSOLUTE_JOINT_TARGET_V1",
            "history_steps": corpus.proprioception_spec.history_steps,
            "observation_dim": corpus.proprioception_spec.observation_dim,
            "output_dim": 29,
            "output_semantics": corpus.proprioception_spec.output_semantics,
            "contains_reference_features": False,
            "proprioception_spec_hash": corpus.proprioception_spec.spec_hash,
            "corpus_manifest_hash": corpus.manifest_hash,
            "training_config": asdict(active),
            "training_config_hash": active.config_hash,
            "checkpoint": checkpoint_path.name,
            "checkpoint_hash": hash_bytes(checkpoint_path.read_bytes()),
            "onnx": onnx_path.name,
            "onnx_hash": hash_bytes(onnx_path.read_bytes()),
            "onnx_input": "proprio_history",
            "onnx_output": "normalized_absolute_motor_target",
            "activation_ceiling": "SIM_ONLY",
            "hardware_authorized": False,
        }
        artifact_manifest["manifest_hash"] = hash_json(artifact_manifest)
        _atomic_json(target / "recovery-student-v1.json", artifact_manifest)
        report = {
            "schema_version": "rosclaw_soccer.recovery_student_training_report.v1",
            "corpus_manifest_hash": corpus.manifest_hash,
            "proprioception_spec_hash": corpus.proprioception_spec.spec_hash,
            "training_config_hash": active.config_hash,
            "training_episode_indexes": list(training_episodes),
            "validation_episode_indexes": list(validation_episodes),
            "episode_overlap_count": 0,
            "training_sample_count": train_samples.history.shape[0],
            "validation_sample_count": validation_samples.history.shape[0],
            "world_size": world_size,
            "devices": all_devices,
            "parameter_count": sum(item.numel() for item in raw_model.parameters()),
            "history": history,
            "validation_mean_absolute_joint_error_rad": float(
                absolute_sum / absolute_count
            ),
            "validation_root_mean_square_joint_error_rad": float(
                torch.sqrt(squared_sum / absolute_count)
            ),
            "validation_maximum_joint_error_rad": float(maximum_error),
            "artifact_manifest": "recovery-student-v1.json",
            "artifact_manifest_hash": artifact_manifest["manifest_hash"],
            "student_reads_reference_phase": False,
            "student_reads_teacher_identity": False,
            "student_direct_motor_target_exam_completed": False,
            "promotion_eligible": False,
            "claim_boundary": "OFFLINE_DISTILLATION_NOT_PHYSICS_QUALIFICATION",
            "activation_ceiling": "SIM_ONLY",
            "hardware_command_sent": False,
        }
        report["report_hash"] = hash_json(report)
        _atomic_json(target / "training-report.json", report)
    if distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--minibatch-size-per-rank", default=256, type=int)
    parser.add_argument("--device")
    args = parser.parse_args()
    report = train_recovery_student(
        corpus_manifest_path=args.corpus_manifest,
        output_dir=args.output_dir,
        config=RecoveryStudentTrainingConfig(
            epochs=args.epochs,
            minibatch_size_per_rank=args.minibatch_size_per_rank,
        ),
        device=args.device,
    )
    if report is not None:
        print(
            json.dumps(
                {
                    "report_hash": report["report_hash"],
                    "world_size": report["world_size"],
                    "validation_mean_absolute_joint_error_rad": report[
                        "validation_mean_absolute_joint_error_rad"
                    ],
                    "artifact_manifest_hash": report["artifact_manifest_hash"],
                }
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RecoveryStudentSamples",
    "RecoveryStudentTrainingConfig",
    "build_recovery_student_samples",
    "split_recovery_student_episodes",
    "train_recovery_student",
]
