"""Data-driven high-level lateral athlete expert for the G1 locomotion prior.

S100 showed that a goalkeeper residual cannot manufacture missing footwork by
perturbing arms and waist.  This module therefore learns a separate, bilateral
velocity-command expert.  It does not replace the qualified 29-DoF locomotion
prior: the neural expert owns acceleration/braking intent while the frozen
prior owns joint targets and torque stabilization.

The artifact is content-bound to that locomotion prior, strictly ``SIM_ONLY``,
and has no hardware or promotion authority.  CPU MuJoCo remains the only
qualification path.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

LATERAL_ATHLETE_INPUT_SIZE = 11
LATERAL_ATHLETE_OUTPUT_SIZE = 1
_MIRRORED_FEATURE_INDICES = (0, 1, 7, 9, 10)


@dataclass(frozen=True)
class LateralAthleteExpertConfig:
    """Training and authority contract for the lateral command expert."""

    hidden_size: int = 96
    samples_per_epoch: int = 65_536
    epochs: int = 400
    minibatch_size: int = 1_024
    learning_rate: float = 1.5e-3
    weight_decay: float = 1.0e-5
    lateral_speed_limit_mps: float = 0.40
    position_gain: float = 1.40
    velocity_gain: float = 0.75
    minimum_effective_speed_mps: float = 0.06
    arrival_deadband_m: float = 0.025
    maximum_fit_rmse: float = 0.020
    maximum_fit_error: float = 0.080
    random_seed: int = 101_421
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    commercial_use_allowed: bool = False
    schema_version: str = "rosclaw_soccer.lateral_athlete_expert_config.v1"

    def __post_init__(self) -> None:
        if not 32 <= self.hidden_size <= 512:
            raise ValueError("lateral athlete hidden size is invalid")
        if not 4_096 <= self.samples_per_epoch <= 2_097_152:
            raise ValueError("lateral athlete sample count is invalid")
        if self.samples_per_epoch % 2:
            raise ValueError("lateral athlete samples must form mirrored pairs")
        if not 10 <= self.epochs <= 20_000:
            raise ValueError("lateral athlete epoch count is invalid")
        if not 128 <= self.minibatch_size <= self.samples_per_epoch:
            raise ValueError("lateral athlete minibatch size is invalid")
        positive = (
            self.learning_rate,
            self.lateral_speed_limit_mps,
            self.position_gain,
            self.velocity_gain,
            self.minimum_effective_speed_mps,
            self.arrival_deadband_m,
            self.maximum_fit_rmse,
            self.maximum_fit_error,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("lateral athlete settings must be finite and positive")
        if not math.isfinite(self.weight_decay) or not 0.0 <= self.weight_decay <= 0.01:
            raise ValueError("lateral athlete weight decay is invalid")
        if not 0.20 <= self.lateral_speed_limit_mps <= 0.40:
            raise ValueError("lateral athlete speed exceeds qualified locomotion range")
        if not 0.02 <= self.minimum_effective_speed_mps <= 0.12:
            raise ValueError("lateral athlete minimum effective speed is invalid")
        if not 0.005 <= self.arrival_deadband_m <= 0.05:
            raise ValueError("lateral athlete arrival deadband is invalid")
        if (
            self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
            or self.commercial_use_allowed
        ):
            raise ValueError("lateral athlete expert must remain non-commercial SIM_ONLY")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


def lateral_athlete_features_numpy(
    *,
    lateral_error_m: NDArray[np.float64],
    lateral_velocity_mps: NDArray[np.float64],
    time_remaining_sec: NDArray[np.float64],
    pelvis_height_m: NDArray[np.float64],
    upright_projection: NDArray[np.float64],
    root_angular_velocity_rad_s: NDArray[np.float64],
    previous_command: NDArray[np.float64],
) -> NDArray[np.float32]:
    """Build the normalized, causal feature vector shared by train and replay."""

    error = np.asarray(lateral_error_m, dtype=np.float64)
    velocity = np.asarray(lateral_velocity_mps, dtype=np.float64)
    remaining = np.asarray(time_remaining_sec, dtype=np.float64)
    pelvis = np.asarray(pelvis_height_m, dtype=np.float64)
    upright = np.asarray(upright_projection, dtype=np.float64)
    angular = np.asarray(root_angular_velocity_rad_s, dtype=np.float64)
    previous = np.asarray(previous_command, dtype=np.float64)
    shape = error.shape
    if (
        velocity.shape != shape
        or remaining.shape != shape
        or pelvis.shape != shape
        or upright.shape != shape
        or previous.shape != shape
        or angular.shape != (*shape, 3)
    ):
        raise ValueError("lateral athlete feature arrays are not shape-aligned")
    values = (error, velocity, remaining, pelvis, upright, angular, previous)
    if any(not np.all(np.isfinite(value)) for value in values):
        raise ValueError("lateral athlete features must be finite")
    return np.asarray(
        np.stack(
            (
                np.clip(error / 2.0, -1.25, 1.25),
                np.clip(velocity / 0.80, -1.5, 1.5),
                np.clip(np.abs(error) / 2.0, 0.0, 1.25),
                np.clip(np.abs(velocity) / 0.80, 0.0, 1.5),
                np.clip(remaining / 10.0, 0.0, 1.0),
                np.clip((pelvis - 0.793) / 0.20, -1.5, 1.5),
                np.clip(upright, -1.0, 1.0),
                np.clip(angular[..., 0] / 3.0, -1.5, 1.5),
                np.clip(angular[..., 1] / 3.0, -1.5, 1.5),
                np.clip(angular[..., 2] / 3.0, -1.5, 1.5),
                np.clip(previous, -1.0, 1.0),
            ),
            axis=-1,
        ),
        dtype=np.float32,
    )


def lateral_athlete_features_torch(
    *,
    torch: Any,
    lateral_error_m: Any,
    lateral_velocity_mps: Any,
    time_remaining_sec: Any,
    pelvis_height_m: Any,
    upright_projection: Any,
    root_angular_velocity_rad_s: Any,
    previous_command: Any,
) -> Any:
    """GPU feature path with the same normalization as CPU replay."""

    return torch.stack(
        (
            torch.clamp(lateral_error_m / 2.0, -1.25, 1.25),
            torch.clamp(lateral_velocity_mps / 0.80, -1.5, 1.5),
            torch.clamp(torch.abs(lateral_error_m) / 2.0, 0.0, 1.25),
            torch.clamp(torch.abs(lateral_velocity_mps) / 0.80, 0.0, 1.5),
            torch.clamp(time_remaining_sec / 10.0, 0.0, 1.0),
            torch.clamp((pelvis_height_m - 0.793) / 0.20, -1.5, 1.5),
            torch.clamp(upright_projection, -1.0, 1.0),
            torch.clamp(root_angular_velocity_rad_s[..., 0] / 3.0, -1.5, 1.5),
            torch.clamp(root_angular_velocity_rad_s[..., 1] / 3.0, -1.5, 1.5),
            torch.clamp(root_angular_velocity_rad_s[..., 2] / 3.0, -1.5, 1.5),
            torch.clamp(previous_command, -1.0, 1.0),
        ),
        dim=-1,
    )


def capture_point_teacher_numpy(
    *,
    lateral_error_m: NDArray[np.float64],
    lateral_velocity_mps: NDArray[np.float64],
    config: LateralAthleteExpertConfig,
) -> NDArray[np.float32]:
    """Return a smooth accelerate/brake target in locomotion-local coordinates.

    The goalkeeper faces pi yaw, so positive world-y velocity requires a
    negative local-y command.  A small identified minimum speed compensates
    the frozen locomotion policy's command dead zone without sign switching.
    """

    error = np.asarray(lateral_error_m, dtype=np.float64)
    velocity = np.asarray(lateral_velocity_mps, dtype=np.float64)
    if (
        error.shape != velocity.shape
        or not np.all(np.isfinite(error))
        or not np.all(np.isfinite(velocity))
    ):
        raise ValueError("lateral athlete teacher state is invalid")
    # A hard dead-band/minimum-command switch created a discontinuous teacher
    # that the student could only memorize poorly.  This smooth identified
    # dead-zone compensation preserves the same physical authority while
    # remaining differentiable.  Velocity damping is continuous through zero;
    # a sharp near-arrival gain switch would teach an avoidable discontinuity.
    desired_world_velocity = (
        config.position_gain * error
        + config.minimum_effective_speed_mps * np.tanh(error / config.arrival_deadband_m)
        - config.velocity_gain * velocity
    )
    command = -np.clip(
        desired_world_velocity / config.lateral_speed_limit_mps,
        -1.0,
        1.0,
    )
    return np.asarray(command, dtype=np.float32)


def build_lateral_athlete_actor(torch: Any, nn: Any, *, hidden_size: int) -> Any:
    """Construct the compact command model used by DDP and CPU replay."""

    return nn.Sequential(
        nn.Linear(LATERAL_ATHLETE_INPUT_SIZE, hidden_size),
        nn.SiLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.SiLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.SiLU(),
        nn.Linear(hidden_size, LATERAL_ATHLETE_OUTPUT_SIZE),
    )


def _mirror_features_torch(torch: Any, features: Any) -> Any:
    mirrored = features.clone()
    mirrored[:, list(_MIRRORED_FEATURE_INDICES)] *= -1.0
    return mirrored


def _equivariant_command_torch(torch: Any, model: Any, features: Any) -> Any:
    """Enforce exact left/right odd symmetry by construction."""

    raw = model(features)
    mirrored = model(_mirror_features_torch(torch, features))
    # The teacher and runtime authority both have a hard physical speed cap.
    # A tanh output cannot reproduce that plateau without driving its latent
    # value toward infinity, leaving systematic under-braking at the tail.
    # Projecting the exactly odd latent command matches the bounded actuator
    # contract and preserves useful gradients throughout the interior.
    return torch.clamp(0.5 * (raw - mirrored), -1.0, 1.0).squeeze(-1)


def _manufacture_curriculum(
    *, config: LateralAthleteExpertConfig
) -> tuple[NDArray[np.float32], NDArray[np.float32], dict[str, Any]]:
    rng = np.random.default_rng(config.random_seed)
    half = config.samples_per_epoch // 2
    error = rng.uniform(0.0, 2.25, size=half)
    velocity = rng.uniform(-0.75, 0.75, size=half)
    remaining = rng.uniform(0.20, 10.0, size=half)
    pelvis = rng.uniform(0.70, 0.86, size=half)
    upright = rng.uniform(0.88, 1.0, size=half)
    angular = rng.uniform((-1.8, -1.1, -1.4), (1.8, 1.1, 1.4), size=(half, 3))
    previous = rng.uniform(-1.0, 1.0, size=half)
    sign = rng.choice(np.asarray((-1.0, 1.0)), size=half)
    error *= sign
    velocity *= sign
    angular[:, 0] *= sign
    angular[:, 2] *= sign
    previous *= sign
    features = lateral_athlete_features_numpy(
        lateral_error_m=error,
        lateral_velocity_mps=velocity,
        time_remaining_sec=remaining,
        pelvis_height_m=pelvis,
        upright_projection=upright,
        root_angular_velocity_rad_s=angular,
        previous_command=previous,
    )
    target = capture_point_teacher_numpy(
        lateral_error_m=error,
        lateral_velocity_mps=velocity,
        config=config,
    )
    mirrored = features.copy()
    mirrored[:, list(_MIRRORED_FEATURE_INDICES)] *= -1.0
    features = np.concatenate((features, mirrored), axis=0)
    target = np.concatenate((target, -target), axis=0)
    metadata = {
        "teacher": "SMOOTH_CAPTURE_POINT_ACCELERATE_BRAKE_WITH_IDENTIFIED_COMMAND_DEADZONE",
        "distance_curriculum_m": [0.0, 2.25],
        "velocity_curriculum_mps": [-0.75, 0.75],
        "bilateral_pairing": "EXACT_SAGITTAL_MIRROR",
        "successor_state_objective": "LOW_LATERAL_SPEED_AND_LOW_ROOT_ANGULAR_MOMENTUM",
    }
    return features, target, metadata


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)


def train_lateral_athlete_expert(
    *,
    locomotion_policy_path: Path,
    output_dir: Path,
    config: LateralAthleteExpertConfig | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Distill the command expert on one or many GPUs.

    ``torchrun`` shards exact mirrored examples across ranks.  Only rank zero
    writes the canonical checkpoint and report.
    """

    import torch
    from torch import nn

    active = config or LateralAthleteExpertConfig()
    locomotion_path = locomotion_policy_path.expanduser().resolve()
    if not locomotion_path.is_file():
        raise FileNotFoundError(f"qualified locomotion policy not found: {locomotion_path}")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("lateral athlete DDP requires CUDA")
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        active_device = torch.device(f"cuda:{local_rank}")
    else:
        active_device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(active.random_seed)
    if active_device.type == "cuda":
        torch.cuda.manual_seed_all(active.random_seed)
    features_np, target_np, curriculum = _manufacture_curriculum(config=active)
    shard: NDArray[np.int64] = np.arange(rank, active.samples_per_epoch, world_size, dtype=np.int64)
    features = torch.as_tensor(features_np[shard], device=active_device)
    target = torch.as_tensor(target_np[shard], device=active_device)
    raw_model = build_lateral_athlete_actor(torch, nn, hidden_size=active.hidden_size).to(
        active_device
    )
    model: Any = raw_model
    if distributed:
        model = nn.parallel.DistributedDataParallel(raw_model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=active.learning_rate, weight_decay=active.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=active.epochs, eta_min=active.learning_rate * 0.05
    )
    generator = torch.Generator(device=active_device)
    generator.manual_seed(active.random_seed + 10_007 * rank)
    history: list[dict[str, float]] = []
    for epoch in range(active.epochs):
        order = torch.randperm(features.shape[0], generator=generator, device=active_device)
        total_loss = torch.zeros((), device=active_device)
        batches = 0
        for start in range(0, features.shape[0], active.minibatch_size):
            ids = order[start : start + active.minibatch_size]
            predicted = _equivariant_command_torch(torch, model, features[ids])
            square_error = torch.square(predicted - target[ids])
            tail_count = max(1, square_error.numel() // 50)
            loss = torch.mean(square_error) + 0.20 * torch.mean(
                torch.topk(square_error, k=tail_count).values
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.detach()
            batches += 1
        scheduler.step()
        if distributed:
            torch.distributed.all_reduce(total_loss)
            total_loss /= world_size
        if rank == 0 and (epoch == 0 or (epoch + 1) % 20 == 0 or epoch + 1 == active.epochs):
            history.append({"epoch": float(epoch + 1), "mean_loss": float(total_loss / batches)})
    all_features = torch.as_tensor(features_np, device=active_device)
    all_target = torch.as_tensor(target_np, device=active_device)
    raw_model.eval()
    with torch.inference_mode():
        decoded = _equivariant_command_torch(torch, raw_model, all_features)
        error = decoded - all_target
        metrics_tensor = torch.stack(
            (torch.sqrt(torch.mean(torch.square(error))), torch.max(torch.abs(error)))
        )
        mirror_error = torch.max(
            torch.abs(
                _equivariant_command_torch(
                    torch, raw_model, _mirror_features_torch(torch, all_features)
                )
                + decoded
            )
        )
    if distributed:
        torch.distributed.broadcast(metrics_tensor, src=0)
        torch.distributed.broadcast(mirror_error, src=0)
    metrics = {
        "command_rmse": float(metrics_tensor[0]),
        "maximum_command_error": float(metrics_tensor[1]),
        "maximum_bilateral_symmetry_error": float(mirror_error),
    }
    fit_passed = bool(
        metrics["command_rmse"] <= active.maximum_fit_rmse
        and metrics["maximum_command_error"] <= active.maximum_fit_error
        and metrics["maximum_bilateral_symmetry_error"] <= 1.0e-6
    )
    destination = output_dir.expanduser().resolve()
    checkpoint_path = destination / "lateral-athlete-expert.pt"
    if rank == 0:
        destination.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "schema_version": "rosclaw_soccer.lateral_athlete_expert.v1",
            "model_state_dict": {
                key: value.detach().cpu() for key, value in raw_model.state_dict().items()
            },
            "input_size": LATERAL_ATHLETE_INPUT_SIZE,
            "output_size": LATERAL_ATHLETE_OUTPUT_SIZE,
            "hidden_size": active.hidden_size,
            "output_representation": "BOUNDED_LOCAL_LATERAL_VELOCITY_COMMAND",
            "symmetry_enforcement": "EXACT_ODD_SAGITTAL_EQUIVARIANCE_V1",
            "locomotion_policy_hash": hash_bytes(locomotion_path.read_bytes()),
            "training_config": asdict(active),
            "curriculum": curriculum,
            "activation_ceiling": "SIM_ONLY",
            "hardware_authorized": False,
            "commercial_use_allowed": False,
        }
        torch.save(checkpoint, checkpoint_path)
        checkpoint_hash = hash_bytes(checkpoint_path.read_bytes())
        report: dict[str, Any] = {
            "schema_version": "rosclaw_soccer.lateral_athlete_training_report.v1",
            "training_backend": "torch_ddp" if distributed else str(active_device),
            "world_size": world_size,
            "config": asdict(active),
            "config_hash": active.config_hash,
            "curriculum": curriculum,
            "metrics": metrics,
            "history": history,
            "checkpoint_hash": checkpoint_hash,
            "locomotion_policy_hash": checkpoint["locomotion_policy_hash"],
            "fit_gate_passed": fit_passed,
            "policy_integration_completed": False,
            "promotion_status": "CANDIDATE_REQUIRES_CPU_MUJOCO_EXAM",
            "activation_ceiling": "SIM_ONLY",
            "hardware_command_sent": False,
            "commercial_use_allowed": False,
        }
        report["report_hash"] = hash_json(report)
        _atomic_json(destination / "training-report.json", report)
    else:
        report = {"rank": rank, "status": "WORKER_COMPLETE"}
    if distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    return report


def load_lateral_athlete_expert(
    *, checkpoint_path: Path, locomotion_policy_path: Path, device: Any
) -> tuple[Any, dict[str, Any]]:
    """Safely load an expert bound to the exact frozen locomotion policy."""

    import torch
    from torch import nn

    checkpoint = torch.load(
        checkpoint_path.expanduser().resolve(), map_location=device, weights_only=True
    )
    locomotion_path = locomotion_policy_path.expanduser().resolve()
    if (
        checkpoint.get("activation_ceiling") != "SIM_ONLY"
        or checkpoint.get("hardware_authorized") is not False
        or checkpoint.get("commercial_use_allowed") is not False
        or int(checkpoint.get("input_size", -1)) != LATERAL_ATHLETE_INPUT_SIZE
        or int(checkpoint.get("output_size", -1)) != LATERAL_ATHLETE_OUTPUT_SIZE
        or checkpoint.get("output_representation") != "BOUNDED_LOCAL_LATERAL_VELOCITY_COMMAND"
        or checkpoint.get("symmetry_enforcement") != "EXACT_ODD_SAGITTAL_EQUIVARIANCE_V1"
        or checkpoint.get("locomotion_policy_hash") != hash_bytes(locomotion_path.read_bytes())
    ):
        raise ValueError("lateral athlete checkpoint boundary is invalid")
    model = build_lateral_athlete_actor(torch, nn, hidden_size=int(checkpoint["hidden_size"])).to(
        device
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def decode_lateral_athlete_command(*, torch: Any, model: Any, features: Any) -> Any:
    """Decode a finite, exactly bilateral command in ``[-1, 1]``."""

    if features.ndim != 2 or features.shape[1] != LATERAL_ATHLETE_INPUT_SIZE:
        raise ValueError("lateral athlete features have an invalid shape")
    if not torch.all(torch.isfinite(features)):
        raise ValueError("lateral athlete features must be finite")
    return _equivariant_command_torch(torch, model, features)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locomotion-policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--samples-per-epoch", type=int, default=65_536)
    args = parser.parse_args()
    report = train_lateral_athlete_expert(
        locomotion_policy_path=args.locomotion_policy,
        output_dir=args.output_dir,
        config=LateralAthleteExpertConfig(
            epochs=args.epochs,
            samples_per_epoch=args.samples_per_epoch,
        ),
    )
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["fit_gate_passed"] else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "LateralAthleteExpertConfig",
    "build_lateral_athlete_actor",
    "capture_point_teacher_numpy",
    "decode_lateral_athlete_command",
    "lateral_athlete_features_numpy",
    "lateral_athlete_features_torch",
    "load_lateral_athlete_expert",
    "train_lateral_athlete_expert",
]
