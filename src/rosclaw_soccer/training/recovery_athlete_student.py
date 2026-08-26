"""Four-GPU, trajectory-anchored recovery-command student for G1.

S104 proved that a successful save can walk off impact, re-enter a measured
ready state, accept another lateral command and become ready again.  This
module distils the successful post-contact recovery commands into a compact,
exactly bilateral actor.  The actor only proposes bounded high-level
locomotion commands; it cannot emit joint targets, torques, ROS messages or
hardware commands.  Integration into physics remains a separate downstream
exam.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.dynamic_corner_save import validate_dynamic_corner_evidence
from rosclaw_soccer.training.save_to_ready_successor import (
    validate_save_to_ready_successor_evidence,
)

RECOVERY_ATHLETE_INPUT_SIZE = 13
RECOVERY_ATHLETE_OUTPUT_SIZE = 3
_FEATURE_MIRROR_SIGN = np.asarray(
    (1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0),
    dtype=np.float32,
)
_OUTPUT_MIRROR_SIGN = np.asarray((1.0, -1.0, -1.0), dtype=np.float32)
_OUTPUT_SCALE = np.asarray((0.30, 0.12, 0.12), dtype=np.float32)
_HOLDOUT_LANE = "right-inner"


@dataclass(frozen=True)
class RecoveryAthleteStudentConfig:
    """Bounded training and authority contract for the recovery student."""

    hidden_size: int = 96
    epochs: int = 300
    minibatch_size: int = 512
    augmentation_factor: int = 16
    learning_rate: float = 1.5e-3
    weight_decay: float = 1.0e-5
    recovery_depth_gain: float = 1.20
    recovery_lateral_gain: float = 0.60
    recovery_yaw_gain: float = 0.40
    depth_speed_limit_mps: float = 0.30
    lateral_speed_limit_mps: float = 0.12
    yaw_rate_limit_rad_s: float = 0.12
    lateral_deadband_m: float = 0.15
    maximum_all_lane_command_mae: float = 0.012
    maximum_holdout_command_mae: float = 0.015
    maximum_normalized_rmse: float = 0.08
    maximum_normalized_error: float = 0.35
    random_seed: int = 105_104
    required_world_size: int = 4
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    commercial_use_allowed: bool = False
    schema_version: str = "rosclaw_soccer.recovery_athlete_student_config.v1"

    def __post_init__(self) -> None:
        if not 32 <= self.hidden_size <= 512:
            raise ValueError("recovery athlete hidden size is invalid")
        if not 20 <= self.epochs <= 5_000:
            raise ValueError("recovery athlete epoch count is invalid")
        if not 128 <= self.minibatch_size <= 8_192:
            raise ValueError("recovery athlete minibatch size is invalid")
        if not 1 <= self.augmentation_factor <= 128:
            raise ValueError("recovery athlete augmentation factor is invalid")
        if self.required_world_size != 4:
            raise ValueError("recovery athlete training requires exactly four GPUs")
        positive = (
            self.learning_rate,
            self.recovery_depth_gain,
            self.recovery_lateral_gain,
            self.recovery_yaw_gain,
            self.depth_speed_limit_mps,
            self.lateral_speed_limit_mps,
            self.yaw_rate_limit_rad_s,
            self.lateral_deadband_m,
            self.maximum_all_lane_command_mae,
            self.maximum_holdout_command_mae,
            self.maximum_normalized_rmse,
            self.maximum_normalized_error,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("recovery athlete settings must be finite and positive")
        if not math.isfinite(self.weight_decay) or not 0.0 <= self.weight_decay <= 0.01:
            raise ValueError("recovery athlete weight decay is invalid")
        if (
            self.depth_speed_limit_mps > 0.30
            or self.lateral_speed_limit_mps > 0.20
            or self.yaw_rate_limit_rad_s > 0.20
            or not 0.08 <= self.lateral_deadband_m <= 0.20
        ):
            raise ValueError("recovery athlete output authority is invalid")
        if (
            self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
            or self.commercial_use_allowed
        ):
            raise ValueError("recovery athlete student must remain non-commercial SIM_ONLY")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


def recovery_athlete_features_numpy(
    *,
    depth_error_m: NDArray[np.float64],
    lateral_position_m: NDArray[np.float64],
    yaw_error_rad: NDArray[np.float64],
    root_velocity: NDArray[np.float64],
    pelvis_height_m: NDArray[np.float64],
    upright_projection: NDArray[np.float64],
    foot_contact: NDArray[np.bool_],
    elapsed_since_contact_sec: NDArray[np.float64],
) -> NDArray[np.float32]:
    """Build a causal feature vector with an explicit sagittal mirror map."""

    depth = np.asarray(depth_error_m, dtype=np.float64)
    lateral = np.asarray(lateral_position_m, dtype=np.float64)
    yaw = np.asarray(yaw_error_rad, dtype=np.float64)
    velocity = np.asarray(root_velocity, dtype=np.float64)
    pelvis = np.asarray(pelvis_height_m, dtype=np.float64)
    upright = np.asarray(upright_projection, dtype=np.float64)
    support = np.asarray(foot_contact, dtype=np.bool_)
    elapsed = np.asarray(elapsed_since_contact_sec, dtype=np.float64)
    shape = depth.shape
    if (
        lateral.shape != shape
        or yaw.shape != shape
        or pelvis.shape != shape
        or upright.shape != shape
        or elapsed.shape != shape
        or velocity.shape != (*shape, 6)
        or support.shape != (*shape, 2)
    ):
        raise ValueError("recovery athlete feature arrays are not shape-aligned")
    values = (depth, lateral, yaw, velocity, pelvis, upright, elapsed)
    if any(not np.all(np.isfinite(value)) for value in values):
        raise ValueError("recovery athlete features must be finite")
    support_sum = support[..., 0].astype(np.float64) + support[..., 1].astype(np.float64)
    support_difference = support[..., 0].astype(np.float64) - support[..., 1].astype(np.float64)
    return np.asarray(
        np.stack(
            (
                np.clip(depth / 0.75, -1.5, 1.5),
                np.clip(lateral / 0.50, -1.5, 1.5),
                np.sin(yaw),
                np.cos(yaw),
                np.clip(velocity[..., 0] / 0.60, -1.5, 1.5),
                np.clip(velocity[..., 1] / 0.60, -1.5, 1.5),
                np.clip(velocity[..., 2] / 0.60, -1.5, 1.5),
                np.clip(velocity[..., 5] / 2.0, -1.5, 1.5),
                np.clip((pelvis - 0.77) / 0.20, -1.5, 1.5),
                np.clip(upright, -1.0, 1.0),
                support_sum / 2.0,
                support_difference,
                np.clip(elapsed / 10.0, 0.0, 1.5),
            ),
            axis=-1,
        ),
        dtype=np.float32,
    )


def recovery_teacher_numpy(
    *,
    depth_error_m: NDArray[np.float64],
    lateral_position_m: NDArray[np.float64],
    yaw_error_rad: NDArray[np.float64],
    config: RecoveryAthleteStudentConfig,
) -> NDArray[np.float32]:
    """Return the successful S104 world-frame recovery teacher command."""

    depth = np.asarray(depth_error_m, dtype=np.float64)
    lateral = np.asarray(lateral_position_m, dtype=np.float64)
    yaw = np.asarray(yaw_error_rad, dtype=np.float64)
    if (
        depth.shape != lateral.shape
        or depth.shape != yaw.shape
        or any(not np.all(np.isfinite(value)) for value in (depth, lateral, yaw))
    ):
        raise ValueError("recovery athlete teacher state is invalid")
    world_x = np.clip(
        config.recovery_depth_gain * depth,
        -config.depth_speed_limit_mps,
        config.depth_speed_limit_mps,
    )
    # S104's hard position deadband deliberately proved the recovery route,
    # but it jumps from zero to roughly 0.09 m/s at the boundary.  Distilling
    # that discontinuity would teach the exact command chatter the student is
    # meant to remove.  Measure error outside the qualified centre pocket so
    # the command grows continuously from zero while retaining the same cap.
    lateral_outside_deadband = np.sign(lateral) * np.maximum(
        np.abs(lateral) - config.lateral_deadband_m,
        0.0,
    )
    world_y = np.clip(
        -config.recovery_lateral_gain * lateral_outside_deadband,
        -config.lateral_speed_limit_mps,
        config.lateral_speed_limit_mps,
    )
    yaw_rate = np.clip(
        config.recovery_yaw_gain * yaw,
        -config.yaw_rate_limit_rad_s,
        config.yaw_rate_limit_rad_s,
    )
    return np.asarray(
        np.stack((world_x, world_y, yaw_rate), axis=-1) / _OUTPUT_SCALE,
        dtype=np.float32,
    )


def build_recovery_athlete_actor(torch: Any, nn: Any, *, hidden_size: int) -> Any:
    return nn.Sequential(
        nn.Linear(RECOVERY_ATHLETE_INPUT_SIZE, hidden_size),
        nn.SiLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.SiLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.SiLU(),
        nn.Linear(hidden_size, RECOVERY_ATHLETE_OUTPUT_SIZE),
    )


def _mirror_features_numpy(features: NDArray[np.float32]) -> NDArray[np.float32]:
    return np.asarray(features * _FEATURE_MIRROR_SIGN, dtype=np.float32)


def _equivariant_command_torch(torch: Any, model: Any, features: Any) -> Any:
    feature_sign = torch.as_tensor(
        _FEATURE_MIRROR_SIGN, dtype=features.dtype, device=features.device
    )
    output_sign = torch.as_tensor(_OUTPUT_MIRROR_SIGN, dtype=features.dtype, device=features.device)
    raw = model(features)
    mirrored = model(features * feature_sign)
    return torch.tanh(0.5 * (raw + mirrored * output_sign))


def decode_recovery_athlete_command(*, torch: Any, model: Any, features: Any) -> Any:
    """Decode an exactly bilateral, finite normalized command in [-1, 1]."""

    if features.ndim != 2 or features.shape[1] != RECOVERY_ATHLETE_INPUT_SIZE:
        raise ValueError("recovery athlete features have an invalid shape")
    if not torch.all(torch.isfinite(features)):
        raise ValueError("recovery athlete features must be finite")
    return _equivariant_command_torch(torch, model, features)


def _yaw_error(quaternion: NDArray[np.float64]) -> NDArray[np.float64]:
    norm = np.linalg.norm(quaternion, axis=1, keepdims=True)
    if np.any(norm <= 1.0e-8):
        raise ValueError("recovery athlete root quaternion is degenerate")
    w, x, y, z = (quaternion / norm).T
    current = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.asarray(np.arctan2(np.sin(math.pi - current), np.cos(math.pi - current)))


def _load_success_anchors(
    *,
    evidence_path: Path,
    parent_evidence_path: Path,
    config: RecoveryAthleteStudentConfig,
) -> tuple[dict[str, tuple[NDArray[np.float32], NDArray[np.float32]]], dict[str, Any]]:
    evidence_file = evidence_path.expanduser().resolve()
    parent_file = parent_evidence_path.expanduser().resolve()
    evidence = validate_save_to_ready_successor_evidence(evidence_file)
    parent = validate_dynamic_corner_evidence(parent_file)
    artifacts = cast(dict[str, Any], evidence["artifacts"])
    if hash_bytes(parent_file.read_bytes()) != artifacts.get(
        "parent_evidence_file_hash"
    ) or parent.get("report_hash") != artifacts.get("parent_report_hash"):
        raise ValueError("recovery athlete parent evidence binding changed")
    evidence_request = json.loads((evidence_file.parent / "request.json").read_text())
    parent_request = json.loads((parent_file.parent / "request.json").read_text())
    goal_specs = parent_request.get("lane_goal_specs")
    cases = evidence.get("cases")
    if (
        not isinstance(goal_specs, dict)
        or not isinstance(cases, dict)
        or set(goal_specs) != set(cases)
    ):
        raise ValueError("recovery athlete lane geometry is incomplete")
    anchors: dict[str, tuple[NDArray[np.float32], NDArray[np.float32]]] = {}
    trajectory_hashes: dict[str, str] = {}
    for lane_id, case in cases.items():
        trajectory_path = evidence_file.parent / str(case["trajectory_file"])
        with np.load(trajectory_path, allow_pickle=False) as archive:
            time = np.asarray(archive["time"], dtype=np.float64)
            pelvis = np.asarray(archive["goalkeeper_pelvis_pose"], dtype=np.float64)
            velocity = np.asarray(archive["goalkeeper_root_velocity"], dtype=np.float64)
            torso = np.asarray(archive["goalkeeper_torso_quaternion"], dtype=np.float64)
            support = np.asarray(archive["goalkeeper_foot_contact"], dtype=np.bool_)
        contact = float(case["result"]["goalkeeper_ball_contact_time_sec"])
        recovery_start = contact + float(evidence_request["config"]["recovery_delay_sec"])
        recovery_stop = contact + float(evidence_request["config"]["probe_delay_sec"]) - 1.0
        mask = (time >= recovery_start) & (time < recovery_stop)
        quaternion = pelvis[mask, 3:7]
        yaw = _yaw_error(quaternion)
        upright = 1.0 - 2.0 * (torso[mask, 1] ** 2 + torso[mask, 2] ** 2)
        desired_depth = float(goal_specs[lane_id]["plane_x_m"]) - 0.48
        depth = desired_depth - pelvis[mask, 0]
        lateral = pelvis[mask, 1]
        elapsed = time[mask] - contact
        features = recovery_athlete_features_numpy(
            depth_error_m=depth,
            lateral_position_m=lateral,
            yaw_error_rad=yaw,
            root_velocity=velocity[mask],
            pelvis_height_m=pelvis[mask, 2],
            upright_projection=upright,
            foot_contact=support[mask],
            elapsed_since_contact_sec=elapsed,
        )
        target = recovery_teacher_numpy(
            depth_error_m=depth,
            lateral_position_m=lateral,
            yaw_error_rad=yaw,
            config=config,
        )
        anchors[str(lane_id)] = (features, target)
        trajectory_hashes[str(lane_id)] = hash_bytes(trajectory_path.read_bytes())
    metadata = {
        "evidence_file_hash": hash_bytes(evidence_file.read_bytes()),
        "evidence_report_hash": evidence["report_hash"],
        "evidence_source_commit": evidence_request["source_commit"],
        "parent_evidence_file_hash": hash_bytes(parent_file.read_bytes()),
        "parent_report_hash": parent["report_hash"],
        "trajectory_hashes": trajectory_hashes,
        "holdout_lane": _HOLDOUT_LANE,
        "phase_window": "CONTACT_PLUS_RECOVERY_DELAY_TO_ONE_SECOND_BEFORE_PROBE",
        "teacher": "S104_TRAJECTORY_ANCHORED_CONTINUOUS_DEPTH_CENTER_YAW_RECOVERY",
    }
    return anchors, metadata


def _augmented_training_set(
    *,
    anchors: dict[str, tuple[NDArray[np.float32], NDArray[np.float32]]],
    config: RecoveryAthleteStudentConfig,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    rng = np.random.default_rng(config.random_seed)
    base = np.concatenate(
        [features for lane, (features, _) in anchors.items() if lane != _HOLDOUT_LANE], axis=0
    )
    rows = [base]
    for _ in range(config.augmentation_factor - 1):
        noisy = base.copy()
        noise_scale = np.asarray(
            (0.025, 0.025, 0.025, 0.01, 0.03, 0.03, 0.02, 0.03, 0.02, 0.005, 0.0, 0.0, 0.01),
            dtype=np.float32,
        )
        noisy += rng.normal(0.0, noise_scale, size=noisy.shape).astype(np.float32)
        rows.append(noisy)
    features = np.concatenate(rows, axis=0)
    depth = features[:, 0].astype(np.float64) * 0.75
    lateral = features[:, 1].astype(np.float64) * 0.50
    yaw = np.arctan2(features[:, 2], features[:, 3]).astype(np.float64)
    target = recovery_teacher_numpy(
        depth_error_m=depth,
        lateral_position_m=lateral,
        yaw_error_rad=yaw,
        config=config,
    )
    mirrored = _mirror_features_numpy(features)
    return (
        np.concatenate((features, mirrored), axis=0),
        np.concatenate((target, target * _OUTPUT_MIRROR_SIGN), axis=0),
    )


def _lane_metrics(
    *, torch: Any, model: Any, anchors: dict[str, tuple[NDArray[np.float32], NDArray[np.float32]]]
) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    device = next(model.parameters()).device
    with torch.inference_mode():
        for lane_id, (features, target) in anchors.items():
            tensor = torch.as_tensor(features, device=device)
            decoded = decode_recovery_athlete_command(torch=torch, model=model, features=tensor)
            predicted = decoded.detach().cpu().numpy() * _OUTPUT_SCALE
            physical_target = target * _OUTPUT_SCALE
            error = np.abs(predicted - physical_target)
            rows[lane_id] = {
                "command_mae": float(np.mean(error)),
                "maximum_command_error": float(np.max(error)),
                "sample_count": float(features.shape[0]),
            }
    return rows


def train_recovery_athlete_student(
    *,
    evidence_path: Path,
    parent_evidence_path: Path,
    locomotion_policy_path: Path,
    output_dir: Path,
    config: RecoveryAthleteStudentConfig | None = None,
) -> dict[str, Any]:
    """Train on four A6000 GPUs and write one safe, content-bound checkpoint."""

    import torch
    from torch import nn

    active = config or RecoveryAthleteStudentConfig()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != active.required_world_size or not torch.cuda.is_available():
        raise RuntimeError("recovery athlete training requires torchrun on exactly four CUDA GPUs")
    if torch.cuda.device_count() < active.required_world_size:
        raise RuntimeError("recovery athlete training cannot see four CUDA devices")
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(active.random_seed)
    torch.cuda.manual_seed_all(active.random_seed)
    locomotion = locomotion_policy_path.expanduser().resolve()
    if not locomotion.is_file():
        raise FileNotFoundError("qualified locomotion policy is missing")
    anchors, metadata = _load_success_anchors(
        evidence_path=evidence_path,
        parent_evidence_path=parent_evidence_path,
        config=active,
    )
    features_np, target_np = _augmented_training_set(anchors=anchors, config=active)
    shard = np.arange(rank, features_np.shape[0], world_size, dtype=np.int64)
    features = torch.as_tensor(features_np[shard], device=device)
    target = torch.as_tensor(target_np[shard], device=device)
    raw_model = build_recovery_athlete_actor(torch, nn, hidden_size=active.hidden_size).to(device)
    model = nn.parallel.DistributedDataParallel(raw_model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=active.learning_rate, weight_decay=active.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=active.epochs, eta_min=active.learning_rate * 0.05
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(active.random_seed + 10_007 * rank)
    history: list[dict[str, float]] = []
    for epoch in range(active.epochs):
        order = torch.randperm(features.shape[0], generator=generator, device=device)
        total_loss = torch.zeros((), device=device)
        batches = 0
        for start in range(0, features.shape[0], active.minibatch_size):
            ids = order[start : start + active.minibatch_size]
            predicted = _equivariant_command_torch(torch, model, features[ids])
            square_error = torch.square(predicted - target[ids])
            tail = max(1, square_error.numel() // 50)
            loss = torch.mean(square_error) + 0.20 * torch.mean(
                torch.topk(square_error.flatten(), k=tail).values
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.detach()
            batches += 1
        scheduler.step()
        torch.distributed.all_reduce(total_loss)
        total_loss /= world_size
        if rank == 0 and (epoch == 0 or (epoch + 1) % 25 == 0 or epoch + 1 == active.epochs):
            history.append({"epoch": float(epoch + 1), "mean_loss": float(total_loss / batches)})
    raw_model.eval()
    all_features = torch.as_tensor(features_np, device=device)
    all_target = torch.as_tensor(target_np, device=device)
    with torch.inference_mode():
        decoded = _equivariant_command_torch(torch, raw_model, all_features)
        error = decoded - all_target
        normalized_rmse = float(torch.sqrt(torch.mean(torch.square(error))))
        maximum_error = float(torch.max(torch.abs(error)))
        mirror_features = all_features * torch.as_tensor(
            _FEATURE_MIRROR_SIGN, dtype=all_features.dtype, device=device
        )
        output_sign = torch.as_tensor(_OUTPUT_MIRROR_SIGN, dtype=all_features.dtype, device=device)
        symmetry_error = float(
            torch.max(
                torch.abs(
                    _equivariant_command_torch(torch, raw_model, mirror_features)
                    - decoded * output_sign
                )
            )
        )
    lane_metrics = _lane_metrics(torch=torch, model=raw_model, anchors=anchors)
    all_lane_mae = max(row["command_mae"] for row in lane_metrics.values())
    holdout_mae = lane_metrics[_HOLDOUT_LANE]["command_mae"]
    fit_passed = bool(
        normalized_rmse <= active.maximum_normalized_rmse
        and maximum_error <= active.maximum_normalized_error
        and symmetry_error <= 1.0e-6
        and all_lane_mae <= active.maximum_all_lane_command_mae
        and holdout_mae <= active.maximum_holdout_command_mae
    )
    destination = output_dir.expanduser().resolve()
    if rank == 0:
        destination.mkdir(parents=True, exist_ok=True)
        checkpoint_path = destination / "recovery-athlete-student.pt"
        checkpoint = {
            "schema_version": "rosclaw_soccer.recovery_athlete_student.v1",
            "model_state_dict": {
                key: value.detach().cpu() for key, value in raw_model.state_dict().items()
            },
            "input_size": RECOVERY_ATHLETE_INPUT_SIZE,
            "output_size": RECOVERY_ATHLETE_OUTPUT_SIZE,
            "hidden_size": active.hidden_size,
            "feature_mirror_sign": _FEATURE_MIRROR_SIGN.tolist(),
            "output_mirror_sign": _OUTPUT_MIRROR_SIGN.tolist(),
            "output_scale": _OUTPUT_SCALE.tolist(),
            "output_representation": "BOUNDED_WORLD_DEPTH_LATERAL_YAW_LOCOMOTION_COMMAND",
            "symmetry_enforcement": "EXACT_SAGITTAL_EQUIVARIANCE_V1",
            "locomotion_policy_hash": hash_bytes(locomotion.read_bytes()),
            "source_evidence": metadata,
            "training_config": asdict(active),
            "activation_ceiling": "SIM_ONLY",
            "hardware_authorized": False,
            "commercial_use_allowed": False,
        }
        torch.save(checkpoint, checkpoint_path)
        report: dict[str, Any] = {
            "schema_version": "rosclaw_soccer.recovery_athlete_training_report.v1",
            "training_backend": "torch_ddp_nccl",
            "world_size": world_size,
            "visible_cuda_devices": torch.cuda.device_count(),
            "gpu_names": [torch.cuda.get_device_name(index) for index in range(world_size)],
            "config": asdict(active),
            "config_hash": active.config_hash,
            "source_evidence": metadata,
            "training_sample_count": int(features_np.shape[0]),
            "physical_anchor_count": int(sum(value[0].shape[0] for value in anchors.values())),
            "training_lanes": sorted(set(anchors) - {_HOLDOUT_LANE}),
            "holdout_lane": _HOLDOUT_LANE,
            "metrics": {
                "normalized_rmse": normalized_rmse,
                "maximum_normalized_error": maximum_error,
                "maximum_bilateral_symmetry_error": symmetry_error,
                "maximum_all_lane_command_mae": all_lane_mae,
                "holdout_command_mae": holdout_mae,
            },
            "lane_metrics": lane_metrics,
            "history": history,
            "checkpoint_hash": hash_bytes(checkpoint_path.read_bytes()),
            "locomotion_policy_hash": checkpoint["locomotion_policy_hash"],
            "fit_gate_passed": fit_passed,
            "policy_integration_completed": False,
            "promotion_status": "CANDIDATE_REQUIRES_CPU_MUJOCO_INTEGRATION_EXAM",
            "activation_ceiling": "SIM_ONLY",
            "hardware_command_sent": False,
            "commercial_use_allowed": False,
            "implementation_hash": hash_bytes(Path(__file__).read_bytes()),
        }
        report["report_hash"] = hash_json(report)
        _atomic_json(destination / "training-report.json", report)
    else:
        report = {"rank": rank, "status": "WORKER_COMPLETE"}
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
    return report


def load_recovery_athlete_student(
    *, checkpoint_path: Path, locomotion_policy_path: Path, device: Any
) -> tuple[Any, dict[str, Any]]:
    """Safely load a student bound to the exact frozen locomotion prior."""

    import torch
    from torch import nn

    checkpoint = torch.load(
        checkpoint_path.expanduser().resolve(), map_location=device, weights_only=True
    )
    locomotion = locomotion_policy_path.expanduser().resolve()
    if (
        checkpoint.get("schema_version") != "rosclaw_soccer.recovery_athlete_student.v1"
        or checkpoint.get("activation_ceiling") != "SIM_ONLY"
        or checkpoint.get("hardware_authorized") is not False
        or checkpoint.get("commercial_use_allowed") is not False
        or checkpoint.get("input_size") != RECOVERY_ATHLETE_INPUT_SIZE
        or checkpoint.get("output_size") != RECOVERY_ATHLETE_OUTPUT_SIZE
        or checkpoint.get("output_representation")
        != "BOUNDED_WORLD_DEPTH_LATERAL_YAW_LOCOMOTION_COMMAND"
        or checkpoint.get("symmetry_enforcement") != "EXACT_SAGITTAL_EQUIVARIANCE_V1"
        or checkpoint.get("feature_mirror_sign") != _FEATURE_MIRROR_SIGN.tolist()
        or checkpoint.get("output_mirror_sign") != _OUTPUT_MIRROR_SIGN.tolist()
        or checkpoint.get("output_scale") != _OUTPUT_SCALE.tolist()
        or checkpoint.get("locomotion_policy_hash") != hash_bytes(locomotion.read_bytes())
    ):
        raise ValueError("recovery athlete checkpoint boundary is invalid")
    model = build_recovery_athlete_actor(torch, nn, hidden_size=int(checkpoint["hidden_size"])).to(
        device
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, cast(dict[str, Any], checkpoint)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--parent-evidence", type=Path, required=True)
    parser.add_argument("--locomotion-policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--augmentation-factor", type=int, default=16)
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = train_recovery_athlete_student(
        evidence_path=args.evidence,
        parent_evidence_path=args.parent_evidence,
        locomotion_policy_path=args.locomotion_policy,
        output_dir=args.output_dir,
        config=RecoveryAthleteStudentConfig(
            epochs=args.epochs,
            augmentation_factor=args.augmentation_factor,
        ),
    )
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
        return 0 if report["fit_gate_passed"] else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RecoveryAthleteStudentConfig",
    "build_recovery_athlete_actor",
    "decode_recovery_athlete_command",
    "load_recovery_athlete_student",
    "recovery_athlete_features_numpy",
    "recovery_teacher_numpy",
    "train_recovery_athlete_student",
]
