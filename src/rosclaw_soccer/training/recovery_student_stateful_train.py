"""Train a stateful proprio-only recovery student over complete episodes."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.recovery_student import (
    load_recovery_distillation_corpus,
    normalize_absolute_motor_targets,
)
from rosclaw_soccer.training.recovery_student_train import (
    _atomic_json,
    split_recovery_student_episodes,
)


@dataclass(frozen=True)
class StatefulRecoveryStudentTrainingConfig:
    epochs: int = 160
    episode_batch_size_per_rank: int = 3
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-5
    frame_hidden_size: int = 128
    memory_size: int = 256
    huber_delta: float = 0.04
    initial_window_steps: int = 75
    initial_frame_weight: float = 6.0
    ready_frame_weight: float = 2.0
    hard_joint_loss_weight: float = 0.20
    action_delta_loss_weight: float = 0.10
    gradient_clip_norm: float = 1.0
    random_seed: int = 52_101
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.stateful_recovery_student_training_config.v1"

    def __post_init__(self) -> None:
        positive = (
            self.learning_rate,
            self.weight_decay,
            self.huber_delta,
            self.initial_frame_weight,
            self.ready_frame_weight,
            self.gradient_clip_norm,
        )
        if (
            not 1 <= self.epochs <= 10_000
            or not 1 <= self.episode_batch_size_per_rank <= 64
            or not 16 <= self.frame_hidden_size <= 2_048
            or not 32 <= self.memory_size <= 4_096
            or not 1 <= self.initial_window_steps <= 500
            or any(not math.isfinite(value) or value <= 0.0 for value in positive)
            or not math.isfinite(self.hard_joint_loss_weight)
            or not 0.0 <= self.hard_joint_loss_weight <= 10.0
            or not math.isfinite(self.action_delta_loss_weight)
            or not 0.0 <= self.action_delta_loss_weight <= 10.0
            or not 0 <= self.random_seed < 2**31
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("stateful recovery student training config is invalid")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


def train_stateful_recovery_student(
    *,
    corpus_manifest_path: Path,
    output_dir: Path,
    config: StatefulRecoveryStudentTrainingConfig | None = None,
    device: str | None = None,
) -> dict[str, Any] | None:
    """Use full-episode BPTT so internal memory replaces privileged phase."""

    import torch
    from safetensors.torch import save_file
    from torch import nn

    active = config or StatefulRecoveryStudentTrainingConfig()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("stateful recovery student DDP requires CUDA")
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
                raise ValueError("stateful recovery trainer refuses to overwrite")
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
    row_by_index = {int(row["episode_index"]): row for row in corpus.rows}
    training_rows = np.concatenate(
        [
            np.arange(
                int(row_by_index[index]["start_row"]),
                int(row_by_index[index]["start_row"])
                + int(row_by_index[index]["row_count"]),
            )
            for index in training_episodes
        ]
    )
    target_lower = np.minimum(
        corpus.joint_lower_rad,
        np.min(corpus.absolute_motor_targets_rad[training_rows], axis=0),
    ).astype(np.float32)
    target_upper = np.maximum(
        corpus.joint_upper_rad,
        np.max(corpus.absolute_motor_targets_rad[training_rows], axis=0),
    ).astype(np.float32)
    normalized_targets = normalize_absolute_motor_targets(
        corpus.absolute_motor_targets_rad,
        joint_lower_rad=target_lower,
        joint_upper_rad=target_upper,
    )
    observation_mean = np.mean(corpus.proprio[training_rows], axis=0).astype(np.float32)
    observation_std = np.maximum(
        np.std(corpus.proprio[training_rows], axis=0), 1.0e-4
    ).astype(np.float32)

    class StatefulRecoveryStudent(nn.Module):  # type: ignore[misc]
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
            self.memory = nn.GRU(
                input_size=active.frame_hidden_size,
                hidden_size=active.memory_size,
                batch_first=True,
            )
            self.motor_head = nn.Sequential(
                nn.Linear(active.memory_size, active.frame_hidden_size),
                nn.SiLU(),
                nn.Linear(active.frame_hidden_size, 29),
                nn.Tanh(),
            )

        def forward(self, proprio_sequence: Any, memory_in: Any) -> tuple[Any, Any]:
            normalized = (
                proprio_sequence - self.observation_mean
            ) / self.observation_std
            encoded = self.frame_encoder(normalized)
            remembered, memory_out = self.memory(encoded, memory_in)
            return self.motor_head(remembered), memory_out

    random.seed(active.random_seed)
    np.random.seed(active.random_seed)
    torch.manual_seed(active.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(active.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    raw_model = StatefulRecoveryStudent().to(active_device)
    model: Any = raw_model
    if distributed:
        model = nn.parallel.DistributedDataParallel(raw_model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=active.learning_rate, weight_decay=active.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=active.epochs, eta_min=active.learning_rate * 0.05
    )

    def batch_tensors(indexes: list[int]) -> tuple[Any, Any, Any, Any]:
        lengths = [int(row_by_index[index]["row_count"]) for index in indexes]
        maximum = max(lengths)
        features: NDArray[np.float32] = np.zeros(
            (len(indexes), maximum, 93), dtype=np.float32
        )
        targets: NDArray[np.float32] = np.zeros(
            (len(indexes), maximum, 29), dtype=np.float32
        )
        weights: NDArray[np.float32] = np.zeros(
            (len(indexes), maximum), dtype=np.float32
        )
        mask: NDArray[np.float32] = np.zeros(
            (len(indexes), maximum), dtype=np.float32
        )
        for batch_index, (episode_index, length) in enumerate(zip(indexes, lengths, strict=True)):
            row = row_by_index[episode_index]
            start = int(row["start_row"])
            stop = start + length
            features[batch_index, :length] = corpus.proprio[start:stop]
            targets[batch_index, :length] = normalized_targets[start:stop]
            episode_weight: NDArray[np.float32] = np.ones(
                length, dtype=np.float32
            )
            episode_weight[: active.initial_window_steps] += (
                active.initial_frame_weight - 1.0
            )
            episode_weight += corpus.ready_handoff[start:stop].astype(np.float32) * (
                active.ready_frame_weight - 1.0
            )
            weights[batch_index, :length] = episode_weight
            mask[batch_index, :length] = 1.0
        return (
            torch.as_tensor(features, device=active_device),
            torch.as_tensor(targets, device=active_device),
            torch.as_tensor(weights, device=active_device),
            torch.as_tensor(mask, device=active_device),
        )

    history: list[dict[str, Any]] = []
    generator = np.random.default_rng(active.random_seed)
    padded_episode_count = (
        math.ceil(
            len(training_episodes)
            / (world_size * active.episode_batch_size_per_rank)
        )
        * world_size
        * active.episode_batch_size_per_rank
    )
    for epoch in range(active.epochs):
        order = generator.permutation(training_episodes).tolist()
        if padded_episode_count > len(order):
            order.extend(order[: padded_episode_count - len(order)])
        shard = order[rank::world_size]
        model.train()
        loss_sum = torch.zeros((), dtype=torch.float64, device=active_device)
        weighted_count = torch.zeros((), dtype=torch.float64, device=active_device)
        for start in range(0, len(shard), active.episode_batch_size_per_rank):
            episode_batch = shard[start : start + active.episode_batch_size_per_rank]
            features, expected, weights, mask = batch_tensors(episode_batch)
            memory = torch.zeros(
                1,
                len(episode_batch),
                active.memory_size,
                dtype=torch.float32,
                device=active_device,
            )
            predicted, _ = model(features, memory)
            difference = predicted - expected
            absolute = torch.abs(difference)
            huber = torch.where(
                absolute <= active.huber_delta,
                0.5 * torch.square(difference) / active.huber_delta,
                absolute - 0.5 * active.huber_delta,
            ).mean(dim=2)
            denominator = torch.sum(weights * mask)
            position_loss = torch.sum(huber * weights * mask) / denominator
            hard_loss = torch.sum(
                torch.max(absolute, dim=2).values * weights * mask
            ) / denominator
            delta_mask = mask[:, 1:] * mask[:, :-1]
            predicted_delta = predicted[:, 1:] - predicted[:, :-1]
            expected_delta = expected[:, 1:] - expected[:, :-1]
            delta_loss = torch.sum(
                torch.mean(torch.square(predicted_delta - expected_delta), dim=2)
                * delta_mask
            ) / torch.clamp(torch.sum(delta_mask), min=1.0)
            loss = (
                position_loss
                + active.hard_joint_loss_weight * hard_loss
                + active.action_delta_loss_weight * delta_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), active.gradient_clip_norm)
            optimizer.step()
            loss_sum += loss.detach().double() * denominator.detach().double()
            weighted_count += denominator.detach().double()
        scheduler.step()
        if distributed:
            torch.distributed.all_reduce(loss_sum)
            torch.distributed.all_reduce(weighted_count)
        if rank == 0 and (
            epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == active.epochs
        ):
            history.append(
                {
                    "epoch": epoch + 1,
                    "mean_weighted_training_loss": float(loss_sum / weighted_count),
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                }
            )

    raw_model.eval()
    validation_shard = list(validation_episodes[rank::world_size])
    validation_sum = torch.zeros((), dtype=torch.float64, device=active_device)
    validation_squared = torch.zeros((), dtype=torch.float64, device=active_device)
    validation_count = torch.zeros((), dtype=torch.float64, device=active_device)
    maximum_error = torch.zeros((), device=active_device)
    radius = torch.as_tensor(0.5 * (target_upper - target_lower), device=active_device)
    with torch.inference_mode():
        for episode_index in validation_shard:
            features, expected, _, mask = batch_tensors([episode_index])
            memory = torch.zeros(
                1, 1, active.memory_size, dtype=torch.float32, device=active_device
            )
            predicted, _ = raw_model(features, memory)
            validation_error = torch.abs(
                (predicted - expected) * radius[None, None, :]
            )
            valid = mask[:, :, None]
            validation_sum += torch.sum(
                validation_error * valid, dtype=torch.float64
            )
            validation_squared += torch.sum(
                torch.square(validation_error) * valid, dtype=torch.float64
            )
            validation_count += torch.sum(valid, dtype=torch.float64) * 29
            maximum_error = torch.maximum(
                maximum_error, torch.max(validation_error * valid)
            )
    if distributed:
        for value in (validation_sum, validation_squared, validation_count):
            torch.distributed.all_reduce(value)
        torch.distributed.all_reduce(
            maximum_error, op=torch.distributed.ReduceOp.MAX
        )
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

        checkpoint_path = target / "recovery-student-stateful-v1.safetensors"
        save_file(
            {
                name: tensor.detach().cpu().contiguous()
                for name, tensor in raw_model.state_dict().items()
            },
            str(checkpoint_path),
            metadata={
                "activation_ceiling": "SIM_ONLY",
                "hardware_authorized": "false",
                "corpus_manifest_hash": corpus.manifest_hash,
                "training_config_hash": active.config_hash,
                "proprioception_spec_hash": corpus.proprioception_spec.spec_hash,
                "memory_semantics": "INTERNAL_PROPRIOCEPTION_UPDATED_ONLY",
            },
        )
        onnx_path = target / "recovery-student-stateful-v1.onnx"
        example_frame = torch.zeros(1, 1, 93, device=active_device)
        example_memory = torch.zeros(1, 1, active.memory_size, device=active_device)
        torch.onnx.export(
            raw_model,
            (example_frame, example_memory),
            str(onnx_path),
            input_names=["proprio_sequence", "memory_in"],
            output_names=["normalized_absolute_motor_target", "memory_out"],
            dynamic_axes={
                "proprio_sequence": {0: "batch", 1: "sequence"},
                "memory_in": {1: "batch"},
                "normalized_absolute_motor_target": {0: "batch", 1: "sequence"},
                "memory_out": {1: "batch"},
            },
            opset_version=17,
            dynamo=False,
        )
        checker.check_model(load(str(onnx_path)))
        artifact: dict[str, Any] = {
            "schema_version": "rosclaw_soccer.recovery_student_artifact.v1",
            "architecture": "STATEFUL_FRAME_MLP_GRU_ABSOLUTE_JOINT_TARGET_V1",
            "history_steps": 1,
            "observation_dim": corpus.proprioception_spec.observation_dim,
            "output_dim": 29,
            "memory_size": active.memory_size,
            "memory_semantics": "INTERNAL_PROPRIOCEPTION_UPDATED_ONLY",
            "output_semantics": corpus.proprioception_spec.output_semantics,
            "motor_target_authority": "TRAINING_TEACHER_PD_TARGET_ENVELOPE",
            "motor_target_lower_rad": target_lower.tolist(),
            "motor_target_upper_rad": target_upper.tolist(),
            "physical_joint_lower_rad": corpus.joint_lower_rad.tolist(),
            "physical_joint_upper_rad": corpus.joint_upper_rad.tolist(),
            "maximum_target_overreach_rad": float(
                max(
                    np.max(corpus.joint_lower_rad - target_lower),
                    np.max(target_upper - corpus.joint_upper_rad),
                )
            ),
            "torque_limit_required": True,
            "contains_reference_features": False,
            "proprioception_spec_hash": corpus.proprioception_spec.spec_hash,
            "corpus_manifest_hash": corpus.manifest_hash,
            "training_config": asdict(active),
            "training_config_hash": active.config_hash,
            "checkpoint": checkpoint_path.name,
            "checkpoint_hash": hash_bytes(checkpoint_path.read_bytes()),
            "onnx": onnx_path.name,
            "onnx_hash": hash_bytes(onnx_path.read_bytes()),
            "onnx_inputs": ["proprio_sequence", "memory_in"],
            "onnx_outputs": [
                "normalized_absolute_motor_target",
                "memory_out",
            ],
            "activation_ceiling": "SIM_ONLY",
            "hardware_authorized": False,
        }
        artifact["manifest_hash"] = hash_json(artifact)
        _atomic_json(target / "recovery-student-stateful-v1.json", artifact)
        report = {
            "schema_version": "rosclaw_soccer.recovery_student_training_report.v1",
            "corpus_manifest_hash": corpus.manifest_hash,
            "proprioception_spec_hash": corpus.proprioception_spec.spec_hash,
            "training_config_hash": active.config_hash,
            "training_episode_indexes": list(training_episodes),
            "validation_episode_indexes": list(validation_episodes),
            "episode_overlap_count": 0,
            "training_episode_count": len(training_episodes),
            "validation_episode_count": len(validation_episodes),
            "world_size": world_size,
            "devices": all_devices,
            "parameter_count": sum(item.numel() for item in raw_model.parameters()),
            "history": history,
            "validation_mean_absolute_joint_error_rad": float(
                validation_sum / validation_count
            ),
            "validation_root_mean_square_joint_error_rad": float(
                torch.sqrt(validation_squared / validation_count)
            ),
            "validation_maximum_joint_error_rad": float(maximum_error),
            "artifact_manifest": "recovery-student-stateful-v1.json",
            "artifact_manifest_hash": artifact["manifest_hash"],
            "student_reads_reference_phase": False,
            "student_reads_teacher_identity": False,
            "student_uses_internal_memory": True,
            "internal_memory_updated_from_proprioception_only": True,
            "student_direct_motor_target_exam_completed": False,
            "promotion_eligible": False,
            "claim_boundary": "STATEFUL_OFFLINE_DISTILLATION_NOT_PHYSICS_QUALIFICATION",
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
    parser.add_argument("--epochs", default=160, type=int)
    parser.add_argument("--episode-batch-size-per-rank", default=3, type=int)
    parser.add_argument("--frame-hidden-size", default=128, type=int)
    parser.add_argument("--memory-size", default=256, type=int)
    parser.add_argument("--device")
    args = parser.parse_args()
    report = train_stateful_recovery_student(
        corpus_manifest_path=args.corpus_manifest,
        output_dir=args.output_dir,
        config=StatefulRecoveryStudentTrainingConfig(
            epochs=args.epochs,
            episode_batch_size_per_rank=args.episode_batch_size_per_rank,
            frame_hidden_size=args.frame_hidden_size,
            memory_size=args.memory_size,
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
    "StatefulRecoveryStudentTrainingConfig",
    "train_stateful_recovery_student",
]
