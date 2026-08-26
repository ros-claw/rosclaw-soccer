"""Physics-qualified whole-body dive-command expert for G1.

S101 qualified lateral footwork but deliberately did not claim a flying save.
This module starts the next option: it distils a separately qualified 29-DoF
command seed into a target-conditioned expert.  Previously passing,
contact-grounded CPU MuJoCo trajectories provide target context and provenance
only; their achieved joint states are deliberately not used as commands.
Historical video pixels are never read.  The source evidence, trajectories,
G1 body and research-only upstream licence are all content-bound before
training starts.

The actor is sagittally equivariant by construction.  It learns one canonical
side and mirrors the complete G1 joint vector for the opposite side.  The
checkpoint remains non-commercial ``SIM_ONLY`` and has no hardware authority.
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

from rosclaw_soccer.providers.g1.asset_qualification import g1_body_hash
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.goalkeeper_dive_option import (
    GoalkeeperBalancedDiveSeed,
    build_balanced_dive_imitation_seed,
    load_official_goalkeeper_dive_atlas,
    mirror_g1_joint_positions,
)
from rosclaw_soccer.training.goalkeeper_mjwarp import _LOCO_DEFAULT, _LOCO_TO_MOTOR

DIVE_ATHLETE_INPUT_SIZE = 10
DIVE_ATHLETE_OUTPUT_SIZE = 29
_SOURCE_LICENSE_HASH = "sha256:6c8cd1cdbe7accec4f63b6c3afb45ce0ffae9ed6abc0ca55acf5900b37970a82"
_SOURCE_COMMIT = "976a81ff19b7306bafbe923d2890066b68a85271"
_CANONICAL_TRAINING_CLIPS = frozenset(("s92-left-inner", "s92-left-outer"))


@dataclass(frozen=True)
class DiveAthleteExpertConfig:
    """Training and authority contract for the whole-body option."""

    hidden_size: int = 256
    samples_per_epoch: int = 65_536
    epochs: int = 500
    minibatch_size: int = 1_024
    learning_rate: float = 1.2e-3
    weight_decay: float = 1.0e-5
    phase_points_per_clip: int = 81
    phase_jitter: float = 0.006
    condition_jitter_fraction: float = 0.025
    maximum_training_rmse_rad: float = 0.030
    maximum_training_error_rad: float = 0.20
    maximum_source_reconstruction_rmse_rad: float = 0.045
    inward_ankle_roll_teacher_rad: float = 0.20
    random_seed: int = 102_421
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    commercial_use_allowed: bool = False
    schema_version: str = "rosclaw_soccer.dive_athlete_expert_config.v1"

    def __post_init__(self) -> None:
        if not 64 <= self.hidden_size <= 512:
            raise ValueError("dive athlete hidden size is invalid")
        if not 4_096 <= self.samples_per_epoch <= 2_097_152:
            raise ValueError("dive athlete sample count is invalid")
        if not 20 <= self.epochs <= 20_000:
            raise ValueError("dive athlete epoch count is invalid")
        if not 128 <= self.minibatch_size <= self.samples_per_epoch:
            raise ValueError("dive athlete minibatch size is invalid")
        if not 41 <= self.phase_points_per_clip <= 201:
            raise ValueError("dive athlete phase grid is invalid")
        positive = (
            self.learning_rate,
            self.phase_jitter,
            self.condition_jitter_fraction,
            self.maximum_training_rmse_rad,
            self.maximum_training_error_rad,
            self.maximum_source_reconstruction_rmse_rad,
            self.inward_ankle_roll_teacher_rad,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("dive athlete settings must be finite and positive")
        if not math.isfinite(self.weight_decay) or not 0.0 <= self.weight_decay <= 0.01:
            raise ValueError("dive athlete weight decay is invalid")
        if not 0.12 <= self.inward_ankle_roll_teacher_rad <= 0.24:
            raise ValueError("dive athlete ankle-roll teacher margin is invalid")
        if (
            self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
            or self.commercial_use_allowed
        ):
            raise ValueError("dive athlete expert must remain non-commercial SIM_ONLY")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass(frozen=True)
class QualifiedDiveClip:
    """One canonicalized clip extracted from passing state evidence."""

    clip_id: str
    phase: NDArray[np.float64]
    joint_position_rad: NDArray[np.float64]
    target_lateral_m: float
    target_height_m: float
    duration_sec: float
    contact_phase: float
    source_direction: int
    evidence_hash: str
    trajectory_hash: str

    def __post_init__(self) -> None:
        if (
            not self.clip_id
            or self.phase.ndim != 1
            or self.joint_position_rad.shape != (self.phase.size, 29)
            or self.phase.size < 30
            or not np.all(np.isfinite(self.phase))
            or not np.all(np.isfinite(self.joint_position_rad))
            or not np.all(np.diff(self.phase) > 0.0)
            or not math.isclose(float(self.phase[0]), 0.0, abs_tol=1.0e-9)
            or not math.isclose(float(self.phase[-1]), 1.0, abs_tol=1.0e-9)
        ):
            raise ValueError("qualified dive clip tensors are invalid")
        values = (
            self.target_lateral_m,
            self.target_height_m,
            self.duration_sec,
            self.contact_phase,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("qualified dive clip metadata must be finite")
        if not 0.10 <= self.target_lateral_m <= 1.20:
            raise ValueError("qualified dive lateral target is invalid")
        if not 0.80 <= self.target_height_m <= 2.20:
            raise ValueError("qualified dive height target is invalid")
        if not 0.40 <= self.duration_sec <= 1.60 or not 0.10 <= self.contact_phase <= 0.90:
            raise ValueError("qualified dive timing is invalid")
        if self.source_direction not in {-1, 1}:
            raise ValueError("qualified dive direction is invalid")
        if not self.evidence_hash.startswith("sha256:") or not self.trajectory_hash.startswith(
            "sha256:"
        ):
            raise ValueError("qualified dive content hashes are invalid")


def _ready_pose() -> NDArray[np.float64]:
    ready = np.zeros(29, dtype=np.float64)
    ready[np.asarray(_LOCO_TO_MOTOR, dtype=np.int64)] = np.asarray(_LOCO_DEFAULT, dtype=np.float64)
    return ready


def build_physics_margin_dive_teacher(
    seed: GoalkeeperBalancedDiveSeed,
    *,
    inward_ankle_roll_rad: float,
) -> NDArray[np.float64]:
    """Project the source command away from soft ankle-limit overshoot.

    MuJoCo's equality-free hinge limit can be crossed dynamically even when a
    target is just inside its range.  The source seed has only 0.013 rad of
    target margin at one ankle and fails the newer achieved-state limit gate.
    This bilateral inward ankle setpoint preserves the seed's other 27 DoF and
    is invariant under the exact G1 mirror.  It remains a candidate until the
    decoded trajectory passes the independent CPU exam.
    """

    if not math.isfinite(inward_ankle_roll_rad) or not 0.12 <= inward_ankle_roll_rad <= 0.24:
        raise ValueError("dive athlete ankle-roll projection is invalid")
    teacher = np.asarray(seed.joint_position_rad, dtype=np.float64).copy()
    teacher[:, :, 5] = inward_ankle_roll_rad
    teacher[:, :, 11] = -inward_ankle_roll_rad
    if not np.allclose(
        teacher[1],
        mirror_g1_joint_positions(teacher[0]),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("dive athlete projected teacher lost bilateral symmetry")
    teacher.setflags(write=False)
    return teacher


def dive_athlete_features_numpy(
    *,
    phase: NDArray[np.float64],
    target_lateral_m: NDArray[np.float64],
    target_height_m: NDArray[np.float64],
    duration_sec: NDArray[np.float64],
    contact_phase: NDArray[np.float64],
) -> NDArray[np.float32]:
    """Build canonical target and timing features shared by train and replay."""

    values = tuple(
        np.asarray(value, dtype=np.float64)
        for value in (phase, target_lateral_m, target_height_m, duration_sec, contact_phase)
    )
    shape = values[0].shape
    if any(value.shape != shape for value in values) or any(
        not np.all(np.isfinite(value)) for value in values
    ):
        raise ValueError("dive athlete features must be finite and shape-aligned")
    phase_value, lateral, height, duration, contact = values
    if np.any((phase_value < 0.0) | (phase_value > 1.0)) or np.any(lateral < 0.0):
        raise ValueError("dive athlete canonical phase/target is invalid")
    return np.asarray(
        np.stack(
            (
                phase_value,
                np.square(phase_value),
                np.sin(math.pi * phase_value),
                np.cos(math.pi * phase_value),
                np.sin(2.0 * math.pi * phase_value),
                np.cos(2.0 * math.pi * phase_value),
                np.clip(lateral / 0.90, 0.0, 1.4),
                np.clip((height - 1.35) / 0.55, -1.2, 1.4),
                np.clip(duration / 1.0, 0.4, 1.6),
                np.clip(contact, 0.1, 0.9),
            ),
            axis=-1,
        ),
        dtype=np.float32,
    )


def build_dive_athlete_actor(torch: Any, nn: Any, *, hidden_size: int) -> Any:
    """Construct the target-conditioned whole-body decoder."""

    return nn.Sequential(
        nn.Linear(DIVE_ATHLETE_INPUT_SIZE, hidden_size),
        nn.SiLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.SiLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.SiLU(),
        nn.Linear(hidden_size, DIVE_ATHLETE_OUTPUT_SIZE),
    )


def _extract_clip(
    *,
    clip_id: str,
    evidence_path: Path,
    trajectory_path: Path,
    expected_trajectory_hash: str,
) -> QualifiedDiveClip:
    evidence_bytes = evidence_path.read_bytes()
    trajectory_bytes = trajectory_path.read_bytes()
    trajectory_hash = hash_bytes(trajectory_bytes)
    if trajectory_hash != expected_trajectory_hash:
        raise ValueError(f"dive source trajectory binding changed: {clip_id}")
    with np.load(trajectory_path, allow_pickle=False) as arrays:
        required = {
            "time",
            "goalkeeper_joint_position",
            "goalkeeper_pelvis_pose",
            "goalkeeper_balanced_dive_blend",
            "goalkeeper_ball_contact",
        }
        if not required <= set(arrays.files):
            raise ValueError(f"dive source telemetry is incomplete: {clip_id}")
        time = np.asarray(arrays["time"], dtype=np.float64)
        joint = np.asarray(arrays["goalkeeper_joint_position"], dtype=np.float64)
        pelvis = np.asarray(arrays["goalkeeper_pelvis_pose"], dtype=np.float64)
        blend = np.asarray(arrays["goalkeeper_balanced_dive_blend"], dtype=np.float64)
        contact = np.asarray(arrays["goalkeeper_ball_contact"], dtype=np.bool_)
        landing = (
            np.asarray(arrays["goalkeeper_landing_capture_active"], dtype=np.bool_)
            if "goalkeeper_landing_capture_active" in arrays.files
            else np.zeros(time.shape, dtype=np.bool_)
        )
    if (
        time.ndim != 1
        or joint.shape != (time.size, 29)
        or pelvis.shape != (time.size, 7)
        or blend.shape != time.shape
        or contact.shape != time.shape
        or landing.shape != time.shape
        or time.size < 100
        or not np.all(np.diff(time) > 0.0)
        or not np.all(np.isfinite(time))
        or not np.all(np.isfinite(joint))
        or not np.all(np.isfinite(pelvis))
        or not np.all(np.isfinite(blend))
    ):
        raise ValueError(f"dive source telemetry is invalid: {clip_id}")
    active_indices = np.flatnonzero((blend > 1.0e-6) | landing)
    contact_indices = np.flatnonzero(contact)
    if active_indices.size == 0 or contact_indices.size == 0:
        raise ValueError(f"dive source lacks option/contact events: {clip_id}")
    start = max(0, int(active_indices[0]) - 5)
    stop = min(time.size - 1, int(active_indices[-1]) + 20)
    contact_index = int(contact_indices[0])
    if not start < contact_index < stop:
        raise ValueError(f"dive source contact is outside its option window: {clip_id}")
    local_time = time[start : stop + 1] - time[start]
    duration = float(local_time[-1])
    phase = np.asarray(local_time / duration, dtype=np.float64)
    # The ball-contact telemetry identifies the event; the hand target is the
    # matching physical ball pose, not a label inferred from pixels.
    with np.load(trajectory_path, allow_pickle=False) as arrays:
        ball_pose = np.asarray(arrays["ball_pose"], dtype=np.float64)
    contact_point = ball_pose[contact_index, :3]
    lateral_delta = float(contact_point[1] - pelvis[start, 1])
    direction = -1 if lateral_delta < 0.0 else 1
    canonical_joint = np.asarray(joint[start : stop + 1], dtype=np.float64)
    if direction < 0:
        canonical_joint = mirror_g1_joint_positions(canonical_joint)
    return QualifiedDiveClip(
        clip_id=clip_id,
        phase=phase,
        joint_position_rad=canonical_joint,
        target_lateral_m=abs(lateral_delta),
        target_height_m=float(contact_point[2]),
        duration_sec=duration,
        contact_phase=float((time[contact_index] - time[start]) / duration),
        source_direction=direction,
        evidence_hash=hash_bytes(evidence_bytes),
        trajectory_hash=trajectory_hash,
    )


def load_qualified_dive_clips(evidence_root: Path) -> tuple[QualifiedDiveClip, ...]:
    """Load the six passing S90-S92 physics clips; fail closed on drift."""

    root = evidence_root.expanduser().resolve()
    definitions = (
        ("s90-right", "s90-true-airborne-save-v1", "dynamic-takeoff-trajectory.npz", None),
        (
            "s91-right-expanded",
            "s91-expanded-takeoff-landing-capture-v1",
            "dynamic-takeoff-trajectory.npz",
            None,
        ),
        (
            "s92-left-outer",
            "s92-multi-corner-airborne-save-v1",
            "left-outer-trajectory.npz",
            "left-outer",
        ),
        (
            "s92-left-inner",
            "s92-multi-corner-airborne-save-v1",
            "left-inner-trajectory.npz",
            "left-inner",
        ),
        (
            "s92-right-inner",
            "s92-multi-corner-airborne-save-v1",
            "right-inner-trajectory.npz",
            "right-inner",
        ),
        (
            "s92-right-outer",
            "s92-multi-corner-airborne-save-v1",
            "right-outer-trajectory.npz",
            "right-outer",
        ),
    )
    clips: list[QualifiedDiveClip] = []
    for clip_id, directory, trajectory_name, case_id in definitions:
        source = root / directory
        evidence_path = source / "evidence.json"
        trajectory_path = source / trajectory_name
        if not evidence_path.is_file() or not trajectory_path.is_file():
            raise ValueError(f"qualified dive source is missing: {clip_id}")
        payload = json.loads(evidence_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("passed") is not True:
            raise ValueError(f"qualified dive evidence did not pass: {clip_id}")
        if (
            payload.get("activation_ceiling") != "SIM_ONLY"
            or payload.get("hardware_command_sent") is not False
            or payload.get("commercial_use_allowed") is not False
            or payload.get("pixels_used_for_scoring") is not False
        ):
            raise ValueError(f"qualified dive authority changed: {clip_id}")
        if case_id is None:
            expected_hash = payload.get("trajectory_hash")
        else:
            cases = payload.get("cases")
            case = cases.get(case_id) if isinstance(cases, dict) else None
            if not isinstance(case, dict) or case.get("passed") is not True:
                raise ValueError(f"qualified dive case did not pass: {clip_id}")
            expected_hash = case.get("trajectory_hash")
        if not isinstance(expected_hash, str):
            raise ValueError(f"qualified dive source hash is missing: {clip_id}")
        clips.append(
            _extract_clip(
                clip_id=clip_id,
                evidence_path=evidence_path,
                trajectory_path=trajectory_path,
                expected_trajectory_hash=expected_hash,
            )
        )
    return tuple(clips)


def _catalog(clips: tuple[QualifiedDiveClip, ...]) -> dict[str, Any]:
    rows = [
        {
            "clip_id": clip.clip_id,
            "frames": int(clip.phase.size),
            "target_lateral_m": clip.target_lateral_m,
            "target_height_m": clip.target_height_m,
            "duration_sec": clip.duration_sec,
            "contact_phase": clip.contact_phase,
            "source_direction": clip.source_direction,
            "evidence_hash": clip.evidence_hash,
            "trajectory_hash": clip.trajectory_hash,
            "training_role": (
                "CANONICAL_CHAMPION"
                if clip.clip_id in _CANONICAL_TRAINING_CLIPS
                else "AUDIT_REFERENCE"
            ),
        }
        for clip in clips
    ]
    return {
        "source_commit": _SOURCE_COMMIT,
        "source_license_hash": _SOURCE_LICENSE_HASH,
        "training_sources": rows,
        "clip_count": len(rows),
        "source_selection": (
            "S92_LOW_LANDING_ANGULAR_VELOCITY_CANONICAL_LEFT_PAIR_PLUS_EXACT_MIRROR"
        ),
        "pixels_read": False,
        "commercial_use_allowed": False,
        "activation_ceiling": "SIM_ONLY",
        "catalog_hash": hash_json(rows),
    }


def _manufacture_curriculum(
    *,
    clips: tuple[QualifiedDiveClip, ...],
    seed: GoalkeeperBalancedDiveSeed,
    config: DiveAthleteExpertConfig,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float64], dict[str, Any]]:
    rng = np.random.default_rng(config.random_seed)
    ready = _ready_pose()
    selected_clips = tuple(clip for clip in clips if clip.clip_id in _CANONICAL_TRAINING_CLIPS)
    if len(selected_clips) != len(_CANONICAL_TRAINING_CLIPS):
        raise ValueError("canonical dive athlete training pair is incomplete")
    teacher = build_physics_margin_dive_teacher(
        seed,
        inward_ankle_roll_rad=config.inward_ankle_roll_teacher_rad,
    )
    canonical_teacher = np.asarray(teacher[1], dtype=np.float64)
    seed_phase = np.linspace(0.0, 1.0, canonical_teacher.shape[0], dtype=np.float64)
    scales = np.maximum(np.max(np.abs(canonical_teacher - ready), axis=0) * 1.08, 0.08)
    sample_clip = rng.integers(0, len(selected_clips), size=config.samples_per_epoch)
    base_phase = rng.uniform(0.0, 1.0, size=config.samples_per_epoch)
    phase = np.clip(
        base_phase + rng.normal(0.0, config.phase_jitter, size=base_phase.shape), 0.0, 1.0
    )
    lateral = np.empty(config.samples_per_epoch, dtype=np.float64)
    height = np.empty_like(lateral)
    duration = np.empty_like(lateral)
    contact_phase = np.empty_like(lateral)
    targets = np.empty((config.samples_per_epoch, 29), dtype=np.float64)
    for clip_index, clip in enumerate(selected_clips):
        selected = np.flatnonzero(sample_clip == clip_index)
        if selected.size == 0:
            continue
        for joint_index in range(29):
            targets[selected, joint_index] = np.interp(
                phase[selected], seed_phase, canonical_teacher[:, joint_index]
            )
        jitter = config.condition_jitter_fraction
        lateral[selected] = clip.target_lateral_m * rng.uniform(
            1.0 - jitter, 1.0 + jitter, selected.size
        )
        height[selected] = clip.target_height_m * rng.uniform(
            1.0 - jitter, 1.0 + jitter, selected.size
        )
        duration[selected] = clip.duration_sec * rng.uniform(
            1.0 - jitter, 1.0 + jitter, selected.size
        )
        contact_phase[selected] = np.clip(
            clip.contact_phase + rng.normal(0.0, 0.25 * jitter, selected.size), 0.1, 0.9
        )
    features = dive_athlete_features_numpy(
        phase=phase,
        target_lateral_m=lateral,
        target_height_m=height,
        duration_sec=duration,
        contact_phase=contact_phase,
    )
    normalized = np.asarray(np.clip((targets - ready) / scales, -1.0, 1.0), dtype=np.float32)
    metadata = {
        "teacher": "PHYSICS_MARGIN_PROJECTED_29_DOF_DIVE_COMMAND_CANDIDATE",
        "action": "BOUNDED_29_DOF_JOINT_POSITION_TARGET",
        "phase_grid_points": config.phase_points_per_clip,
        "bilateral_policy": "CANONICAL_POSITIVE_SIDE_PLUS_EXACT_ANATOMICAL_MIRROR",
        "successor_objective": "FOOT_LANDING_THEN_LOCOMOTION_READY",
        "canonical_training_clips": sorted(_CANONICAL_TRAINING_CLIPS),
        "audit_reference_clip_count": len(clips) - len(selected_clips),
        "successful_contact_evidence_role": "CONDITION_AND_PROVENANCE_ONLY_NOT_ACTION_LABEL",
        "source_atlas_hash": seed.source_atlas_hash,
        "dive_seed_hash": seed.seed_hash,
        "teacher_trajectory_hash": hash_bytes(teacher.tobytes()),
        "teacher_projection": {
            "left_ankle_roll_rad": config.inward_ankle_roll_teacher_rad,
            "right_ankle_roll_rad": -config.inward_ankle_roll_teacher_rad,
            "other_joint_targets_changed": False,
            "qualification_required_after_decode": True,
        },
        "seed_frames": int(canonical_teacher.shape[0]),
        "seed_frame_rate_hz": seed.frame_rate_hz,
    }
    return features, normalized, scales, metadata


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def train_dive_athlete_expert(
    *,
    evidence_root: Path,
    asset_root: Path,
    dive_source_checkout: Path,
    output_dir: Path,
    config: DiveAthleteExpertConfig | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Distil the whole-body expert on one GPU or under four-rank torchrun."""

    import torch
    from torch import nn

    active = config or DiveAthleteExpertConfig()
    clips = load_qualified_dive_clips(evidence_root)
    catalog = _catalog(clips)
    atlas = load_official_goalkeeper_dive_atlas(checkout=dive_source_checkout)
    seed = build_balanced_dive_imitation_seed(atlas)
    body_hash = g1_body_hash(asset_root.expanduser().resolve())
    features_np, target_np, scales_np, curriculum = _manufacture_curriculum(
        clips=clips, seed=seed, config=active
    )
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("dive athlete DDP requires CUDA")
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        active_device = torch.device(f"cuda:{local_rank}")
    else:
        active_device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(active.random_seed)
    if active_device.type == "cuda":
        torch.cuda.manual_seed_all(active.random_seed)
    shard = np.arange(rank, active.samples_per_epoch, world_size, dtype=np.int64)
    features = torch.as_tensor(features_np[shard], device=active_device)
    target = torch.as_tensor(target_np[shard], device=active_device)
    raw_model = build_dive_athlete_actor(torch, nn, hidden_size=active.hidden_size).to(
        active_device
    )
    model: Any = raw_model
    if distributed:
        model = nn.parallel.DistributedDataParallel(raw_model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=active.learning_rate, weight_decay=active.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=active.epochs, eta_min=active.learning_rate * 0.03
    )
    generator = torch.Generator(device=active_device)
    generator.manual_seed(active.random_seed + 10_007 * rank)
    history: list[dict[str, float]] = []
    for epoch in range(active.epochs):
        order = torch.randperm(features.shape[0], generator=generator, device=active_device)
        total = torch.zeros((), device=active_device)
        batches = 0
        for start in range(0, features.shape[0], active.minibatch_size):
            indices = order[start : start + active.minibatch_size]
            predicted = torch.tanh(model(features[indices]))
            square = torch.square(predicted - target[indices])
            tail = torch.topk(square.flatten(), k=max(1, square.numel() // 100)).values
            loss = torch.mean(square) + 0.15 * torch.mean(tail)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.detach()
            batches += 1
        scheduler.step()
        if distributed:
            torch.distributed.all_reduce(total)
            total /= world_size
        if rank == 0 and (epoch == 0 or (epoch + 1) % 25 == 0 or epoch + 1 == active.epochs):
            history.append({"epoch": float(epoch + 1), "mean_loss": float(total / batches)})
    all_features = torch.as_tensor(features_np, device=active_device)
    all_target = torch.as_tensor(target_np, device=active_device)
    raw_model.eval()
    with torch.inference_mode():
        predicted = torch.tanh(raw_model(all_features))
        error_rad = (predicted - all_target) * torch.as_tensor(scales_np, device=active_device)
        metrics_tensor = torch.stack(
            (torch.sqrt(torch.mean(torch.square(error_rad))), torch.max(torch.abs(error_rad)))
        )
    if distributed:
        torch.distributed.broadcast(metrics_tensor, src=0)
    metrics = {
        "training_position_rmse_rad": float(metrics_tensor[0]),
        "training_maximum_error_rad": float(metrics_tensor[1]),
        "bilateral_symmetry_error_rad": 0.0,
    }
    fit_passed = bool(
        metrics["training_position_rmse_rad"] <= active.maximum_training_rmse_rad
        and metrics["training_maximum_error_rad"] <= active.maximum_training_error_rad
    )
    destination = output_dir.expanduser().resolve()
    checkpoint_path = destination / "dive-athlete-expert.pt"
    if rank == 0:
        destination.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "schema_version": "rosclaw_soccer.dive_athlete_expert.v1",
            "model_state_dict": {
                key: value.detach().cpu() for key, value in raw_model.state_dict().items()
            },
            "input_size": DIVE_ATHLETE_INPUT_SIZE,
            "output_size": DIVE_ATHLETE_OUTPUT_SIZE,
            "hidden_size": active.hidden_size,
            "ready_pose_rad": _ready_pose().tolist(),
            "joint_scale_rad": scales_np.tolist(),
            "body_hash": body_hash,
            "source_catalog_hash": catalog["catalog_hash"],
            "source_atlas_hash": atlas.atlas_hash,
            "dive_seed_hash": seed.seed_hash,
            "teacher_trajectory_hash": curriculum["teacher_trajectory_hash"],
            "source_license_hash": _SOURCE_LICENSE_HASH,
            "source_commit": _SOURCE_COMMIT,
            "output_representation": "BOUNDED_29_DOF_JOINT_POSITION_TARGET",
            "symmetry_enforcement": "CANONICAL_HALF_SPACE_EXACT_G1_MIRROR_V1",
            "training_config": asdict(active),
            "activation_ceiling": "SIM_ONLY",
            "hardware_authorized": False,
            "commercial_use_allowed": False,
        }
        torch.save(checkpoint, checkpoint_path)
        report: dict[str, Any] = {
            "schema_version": "rosclaw_soccer.dive_athlete_training_report.v1",
            "training_backend": "torch_ddp" if distributed else str(active_device),
            "world_size": world_size,
            "config": asdict(active),
            "config_hash": active.config_hash,
            "source_catalog": catalog,
            "body_hash": body_hash,
            "curriculum": curriculum,
            "metrics": metrics,
            "history": history,
            "checkpoint_hash": hash_bytes(checkpoint_path.read_bytes()),
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


def load_dive_athlete_expert(
    *, checkpoint_path: Path, asset_root: Path, dive_source_checkout: Path, device: Any
) -> tuple[Any, dict[str, Any]]:
    """Safely load an expert bound to the exact qualified G1 body."""

    import torch
    from torch import nn

    checkpoint = torch.load(
        checkpoint_path.expanduser().resolve(), map_location=device, weights_only=True
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("dive athlete checkpoint must be a dictionary")
    required = {
        "schema_version",
        "model_state_dict",
        "input_size",
        "output_size",
        "hidden_size",
        "ready_pose_rad",
        "joint_scale_rad",
        "body_hash",
        "source_catalog_hash",
        "source_atlas_hash",
        "dive_seed_hash",
        "teacher_trajectory_hash",
        "source_license_hash",
        "source_commit",
        "output_representation",
        "symmetry_enforcement",
        "training_config",
        "activation_ceiling",
        "hardware_authorized",
        "commercial_use_allowed",
    }
    if not required <= set(checkpoint):
        raise ValueError("dive athlete checkpoint is incomplete")
    training_config = checkpoint["training_config"]
    if not isinstance(training_config, dict):
        raise ValueError("dive athlete training config is invalid")
    config = DiveAthleteExpertConfig(**training_config)
    ready_pose = np.asarray(checkpoint["ready_pose_rad"], dtype=np.float64)
    joint_scale = np.asarray(checkpoint["joint_scale_rad"], dtype=np.float64)
    model_state = checkpoint["model_state_dict"]
    if (
        ready_pose.shape != (DIVE_ATHLETE_OUTPUT_SIZE,)
        or joint_scale.shape != (DIVE_ATHLETE_OUTPUT_SIZE,)
        or not np.all(np.isfinite(ready_pose))
        or not np.all(np.isfinite(joint_scale))
        or np.any(joint_scale <= 0.0)
        or not isinstance(model_state, dict)
    ):
        raise ValueError("dive athlete checkpoint tensors are invalid")
    atlas = load_official_goalkeeper_dive_atlas(checkout=dive_source_checkout)
    seed = build_balanced_dive_imitation_seed(atlas)
    expected_teacher = build_physics_margin_dive_teacher(
        seed,
        inward_ankle_roll_rad=config.inward_ankle_roll_teacher_rad,
    )
    if (
        checkpoint["schema_version"] != "rosclaw_soccer.dive_athlete_expert.v1"
        or checkpoint["input_size"] != DIVE_ATHLETE_INPUT_SIZE
        or checkpoint["output_size"] != DIVE_ATHLETE_OUTPUT_SIZE
        or checkpoint["hidden_size"] != config.hidden_size
        or checkpoint["body_hash"] != g1_body_hash(asset_root.expanduser().resolve())
        or checkpoint["source_license_hash"] != _SOURCE_LICENSE_HASH
        or checkpoint["source_commit"] != _SOURCE_COMMIT
        or checkpoint["source_atlas_hash"] != atlas.atlas_hash
        or checkpoint["dive_seed_hash"] != seed.seed_hash
        or checkpoint["teacher_trajectory_hash"] != hash_bytes(expected_teacher.tobytes())
        or checkpoint["output_representation"] != "BOUNDED_29_DOF_JOINT_POSITION_TARGET"
        or checkpoint["symmetry_enforcement"] != "CANONICAL_HALF_SPACE_EXACT_G1_MIRROR_V1"
        or checkpoint["activation_ceiling"] != "SIM_ONLY"
        or checkpoint["hardware_authorized"] is not False
        or checkpoint["commercial_use_allowed"] is not False
    ):
        raise ValueError("dive athlete checkpoint authority or provenance changed")
    model = build_dive_athlete_actor(torch, nn, hidden_size=int(checkpoint["hidden_size"])).to(
        device
    )
    model.load_state_dict(model_state, strict=True)
    model.eval()
    return model, cast(dict[str, Any], checkpoint)


def decode_dive_athlete_target(
    *,
    torch: Any,
    model: Any,
    checkpoint: dict[str, Any],
    features: Any,
    direction: Any,
) -> Any:
    """Decode bounded joint targets with exact anatomical bilateral symmetry."""

    if features.ndim != 2 or features.shape[1] != DIVE_ATHLETE_INPUT_SIZE:
        raise ValueError("dive athlete feature tensor has the wrong shape")
    if direction.ndim != 1 or direction.shape[0] != features.shape[0]:
        raise ValueError("dive athlete direction tensor has the wrong shape")
    if not bool(torch.all(torch.abs(direction) == 1)):
        raise ValueError("dive athlete direction must be +/-1")
    ready = torch.as_tensor(
        checkpoint["ready_pose_rad"], dtype=features.dtype, device=features.device
    )
    scale = torch.as_tensor(
        checkpoint["joint_scale_rad"], dtype=features.dtype, device=features.device
    )
    canonical = ready + scale * torch.tanh(model(features))
    # Reuse the audited NumPy mirror contract to build immutable index/sign
    # tensors without duplicating the anatomical mapping in this module.
    identity = np.eye(29, dtype=np.float64)
    mirrored_basis = mirror_g1_joint_positions(identity)
    mirror_matrix = torch.as_tensor(mirrored_basis, dtype=features.dtype, device=features.device)
    mirrored = canonical @ mirror_matrix
    return torch.where(direction[:, None] < 0.0, mirrored, canonical)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--dive-source-checkout", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=500)
    args = parser.parse_args()
    report = train_dive_athlete_expert(
        evidence_root=args.evidence_root,
        asset_root=args.asset_root,
        dive_source_checkout=args.dive_source_checkout,
        output_dir=args.output_dir,
        config=DiveAthleteExpertConfig(epochs=args.epochs),
    )
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report.get("fit_gate_passed") else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DIVE_ATHLETE_INPUT_SIZE",
    "DiveAthleteExpertConfig",
    "QualifiedDiveClip",
    "build_dive_athlete_actor",
    "decode_dive_athlete_target",
    "dive_athlete_features_numpy",
    "load_dive_athlete_expert",
    "load_qualified_dive_clips",
    "train_dive_athlete_expert",
]
