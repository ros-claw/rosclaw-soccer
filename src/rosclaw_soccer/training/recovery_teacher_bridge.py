"""Failure-driven routing from post-skill states to recovery teachers.

This module is deliberately independent of OpenTrack, MuJoCo and Torch.  It
mines content-bound recovery entries from pickle-free 29-DoF trajectories and
turns development trials into a deterministic bridge schedule.  A schedule is
training evidence only: the reference phase and teacher identity are privileged
signals and therefore cannot be consumed by the deployable recovery gate.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.recovery_snapshot import (
    RecoverySnapshot,
    classify_recovery_posture,
)

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _valid_hash(value: str) -> bool:
    return value.startswith("sha256:") and len(value) == 71


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def body_gravity_vector(
    quaternion_wxyz: Sequence[float] | NDArray[np.floating[Any]],
) -> NDArray[np.float64]:
    """Return world gravity expressed in the body frame."""

    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("body quaternion must contain four finite values")
    norm = float(np.linalg.norm(quaternion))
    if not 0.95 <= norm <= 1.05:
        raise ValueError("body quaternion must be normalized")
    w, x, y, z = quaternion / norm
    return np.asarray(
        (
            -2.0 * (x * z - w * y),
            -2.0 * (y * z + w * x),
            -(w * w - x * x - y * y + z * z),
        ),
        dtype=np.float64,
    )


@dataclass(frozen=True)
class RecoveryReferenceMotion:
    """One safe-to-inspect, 50 Hz G1 recovery reference."""

    motion_id: str
    qpos: NDArray[np.float64]
    qvel: NDArray[np.float64]
    source_hash: str
    frequency_hz: float = 50.0
    schema_version: str = "rosclaw_soccer.recovery_reference_motion.v1"

    def __post_init__(self) -> None:
        qpos = np.asarray(self.qpos, dtype=np.float64)
        qvel = np.asarray(self.qvel, dtype=np.float64)
        if (
            not _IDENTIFIER.fullmatch(self.motion_id)
            or qpos.ndim != 2
            or qvel.ndim != 2
            or qpos.shape[1] != 36
            or qvel.shape[1] != 35
            or qpos.shape[0] != qvel.shape[0]
            or qpos.shape[0] < 100
            or not np.all(np.isfinite(qpos))
            or not np.all(np.isfinite(qvel))
            or not _valid_hash(self.source_hash)
            or not math.isfinite(self.frequency_hz)
            or not math.isclose(self.frequency_hz, 50.0, abs_tol=1e-9)
        ):
            raise ValueError("recovery reference motion contract is invalid")
        object.__setattr__(self, "qpos", qpos)
        object.__setattr__(self, "qvel", qvel)

    @classmethod
    def from_npz(cls, path: Path) -> RecoveryReferenceMotion:
        source = path.expanduser().resolve()
        if not source.is_file() or source.suffix != ".npz":
            raise FileNotFoundError("recovery reference must be an NPZ file")
        # Only numeric allow-listed arrays are accessed.  Object metadata in an
        # upstream archive is never deserialized.
        with np.load(source, allow_pickle=False) as archive:
            if not {"qpos", "qvel", "frequency"}.issubset(archive.files):
                raise ValueError("recovery reference archive is incomplete")
            qpos = np.asarray(archive["qpos"], dtype=np.float64)
            qvel = np.asarray(archive["qvel"], dtype=np.float64)
            frequency = float(archive["frequency"])
        return cls(
            motion_id=source.stem,
            qpos=qpos,
            qvel=qvel,
            source_hash=_file_hash(source),
            frequency_hz=frequency,
        )


@dataclass(frozen=True)
class RecoveryEntrySearchConfig:
    """Frozen geometry and successor-state criteria for entry mining."""

    candidate_stride_frames: int = 2
    maximum_entry_pelvis_height_m: float = 0.28
    maximum_entry_upright_projection: float = 0.50
    minimum_future_offset_frames: int = 37
    maximum_future_offset_frames: int = 900
    successor_pelvis_height_m: float = 0.62
    successor_upright_projection: float = 0.75
    successor_hold_frames: int = 75
    gravity_weight: float = 2.5
    joint_weight: float = 1.0
    height_weight: float = 0.5
    nonmaximum_spacing_frames: int = 120
    maximum_matches: int = 16
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.recovery_entry_search_config.v1"

    def __post_init__(self) -> None:
        finite = (
            self.maximum_entry_pelvis_height_m,
            self.maximum_entry_upright_projection,
            self.successor_pelvis_height_m,
            self.successor_upright_projection,
            self.gravity_weight,
            self.joint_weight,
            self.height_weight,
        )
        if (
            not all(math.isfinite(value) for value in finite)
            or not 1 <= self.candidate_stride_frames <= 20
            or not 0.10 <= self.maximum_entry_pelvis_height_m <= 0.50
            or not -0.5 <= self.maximum_entry_upright_projection <= 0.8
            or self.minimum_future_offset_frames < 1
            or not self.minimum_future_offset_frames < self.maximum_future_offset_frames <= 3000
            or not 0.50 <= self.successor_pelvis_height_m <= 0.90
            or not 0.50 <= self.successor_upright_projection <= 1.0
            or not 25 <= self.successor_hold_frames <= 250
            or min(self.gravity_weight, self.joint_weight, self.height_weight) <= 0.0
            or not 10 <= self.nonmaximum_spacing_frames <= 1000
            or not 1 <= self.maximum_matches <= 64
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("recovery entry search config is invalid")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass(frozen=True)
class RecoveryEntryMatch:
    """One state-to-reference match; it grants no deployment authority."""

    motion_id: str
    source_hash: str
    entry_frame: int
    successor_end_frame: int
    score: float
    joint_rmse_rad: float
    gravity_distance: float
    pelvis_height_error_m: float
    search_config_hash: str
    schema_version: str = "rosclaw_soccer.recovery_entry_match.v1"

    def __post_init__(self) -> None:
        values = (
            self.score,
            self.joint_rmse_rad,
            self.gravity_distance,
            self.pelvis_height_error_m,
        )
        if (
            not _IDENTIFIER.fullmatch(self.motion_id)
            or not _valid_hash(self.source_hash)
            or not _valid_hash(self.search_config_hash)
            or self.entry_frame < 0
            or self.successor_end_frame <= self.entry_frame
            or any(not math.isfinite(value) or value < 0.0 for value in values)
        ):
            raise ValueError("recovery entry match is invalid")

    @property
    def match_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass(frozen=True)
class RecoveryPerturbationConfig:
    """Frozen local robustness distribution for an unseen bridge exam."""

    samples_per_snapshot: int = 3
    joint_position_half_width_rad: float = 0.020
    joint_velocity_half_width_rad_s: float = 0.050
    root_tilt_half_width_rad: float = 0.015
    root_linear_velocity_half_width_mps: float = 0.030
    root_angular_velocity_half_width_rad_s: float = 0.050
    seed_namespace: str = "rosclaw-s51-recovery-holdout-v1"
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.recovery_perturbation_config.v1"

    def __post_init__(self) -> None:
        widths = (
            self.joint_position_half_width_rad,
            self.joint_velocity_half_width_rad_s,
            self.root_tilt_half_width_rad,
            self.root_linear_velocity_half_width_mps,
            self.root_angular_velocity_half_width_rad_s,
        )
        if (
            not 1 <= self.samples_per_snapshot <= 16
            or not all(math.isfinite(value) and value > 0.0 for value in widths)
            or self.joint_position_half_width_rad > 0.10
            or self.joint_velocity_half_width_rad_s > 0.50
            or self.root_tilt_half_width_rad > 0.10
            or self.root_linear_velocity_half_width_mps > 0.20
            or self.root_angular_velocity_half_width_rad_s > 0.30
            or not _IDENTIFIER.fullmatch(self.seed_namespace)
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("recovery perturbation config is invalid")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass(frozen=True)
class RecoveryPerturbation:
    """Content-bound provenance for one deterministic unseen state."""

    base_snapshot_hash: str
    perturbed_snapshot_hash: str
    sample_index: int
    random_seed: int
    config_hash: str
    joint_position_linf_rad: float
    joint_velocity_linf_rad_s: float
    root_tilt_angle_rad: float
    root_linear_velocity_linf_mps: float
    root_angular_velocity_linf_rad_s: float
    schema_version: str = "rosclaw_soccer.recovery_perturbation.v1"

    def __post_init__(self) -> None:
        values = (
            self.joint_position_linf_rad,
            self.joint_velocity_linf_rad_s,
            self.root_tilt_angle_rad,
            self.root_linear_velocity_linf_mps,
            self.root_angular_velocity_linf_rad_s,
        )
        if (
            not _valid_hash(self.base_snapshot_hash)
            or not _valid_hash(self.perturbed_snapshot_hash)
            or not _valid_hash(self.config_hash)
            or self.sample_index < 0
            or not 0 <= self.random_seed < 2**63
            or any(not math.isfinite(value) or value < 0.0 for value in values)
        ):
            raise ValueError("recovery perturbation is invalid")

    @property
    def perturbation_hash(self) -> str:
        return str(hash_json(asdict(self)))


def _quaternion_product_wxyz(
    left: NDArray[np.float64], right: NDArray[np.float64]
) -> NDArray[np.float64]:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.asarray(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dtype=np.float64,
    )


def build_recovery_perturbation_holdout(
    snapshots: Sequence[RecoverySnapshot],
    *,
    config: RecoveryPerturbationConfig | None = None,
) -> tuple[tuple[RecoverySnapshot, RecoveryPerturbation], ...]:
    """Create a deterministic local holdout without consulting trial outcomes."""

    if not snapshots or len({item.snapshot_hash for item in snapshots}) != len(snapshots):
        raise ValueError("recovery perturbation holdout requires unique snapshots")
    active = config or RecoveryPerturbationConfig()
    generated: list[tuple[RecoverySnapshot, RecoveryPerturbation]] = []
    for snapshot in snapshots:
        for sample_index in range(active.samples_per_snapshot):
            seed_material = (
                f"{active.seed_namespace}:{active.config_hash}:"
                f"{snapshot.snapshot_hash}:{sample_index}"
            ).encode()
            random_seed = int.from_bytes(
                hashlib.sha256(seed_material).digest()[:8], "big"
            ) % (2**63)
            generator = np.random.default_rng(random_seed)
            joint_position_noise = generator.uniform(
                -active.joint_position_half_width_rad,
                active.joint_position_half_width_rad,
                size=29,
            )
            joint_velocity_noise = generator.uniform(
                -active.joint_velocity_half_width_rad_s,
                active.joint_velocity_half_width_rad_s,
                size=29,
            )
            root_linear_noise = generator.uniform(
                -active.root_linear_velocity_half_width_mps,
                active.root_linear_velocity_half_width_mps,
                size=3,
            )
            root_angular_noise = generator.uniform(
                -active.root_angular_velocity_half_width_rad_s,
                active.root_angular_velocity_half_width_rad_s,
                size=3,
            )
            tilt_xy = generator.uniform(
                -active.root_tilt_half_width_rad,
                active.root_tilt_half_width_rad,
                size=2,
            )
            tilt_angle = float(np.linalg.norm(tilt_xy))
            if tilt_angle > 0.0:
                half_angle = tilt_angle / 2.0
                delta_quaternion = np.asarray(
                    (
                        math.cos(half_angle),
                        math.sin(half_angle) * tilt_xy[0] / tilt_angle,
                        math.sin(half_angle) * tilt_xy[1] / tilt_angle,
                        0.0,
                    ),
                    dtype=np.float64,
                )
            else:
                delta_quaternion = np.asarray((1.0, 0.0, 0.0, 0.0))
            qpos = snapshot.qpos.copy()
            qvel = snapshot.qvel.copy()
            qpos[3:7] = _quaternion_product_wxyz(
                delta_quaternion, np.asarray(qpos[3:7], dtype=np.float64)
            )
            qpos[3:7] /= np.linalg.norm(qpos[3:7])
            qpos[7:36] += joint_position_noise
            qvel[:3] += root_linear_noise
            qvel[3:6] += root_angular_noise
            qvel[6:35] += joint_velocity_noise
            posture = classify_recovery_posture(
                root_quaternion_wxyz=qpos[3:7],
                pelvis_height_m=float(qpos[2]),
                root_linear_speed_mps=float(np.linalg.norm(qvel[:3])),
                root_angular_speed_rad_s=float(np.linalg.norm(qvel[3:6])),
                left_foot_supported=snapshot.left_foot_supported,
                right_foot_supported=snapshot.right_foot_supported,
            )
            perturbed = replace(
                snapshot,
                posture_cluster=posture,
                qpos=qpos,
                qvel=qvel,
            )
            record = RecoveryPerturbation(
                base_snapshot_hash=snapshot.snapshot_hash,
                perturbed_snapshot_hash=perturbed.snapshot_hash,
                sample_index=sample_index,
                random_seed=random_seed,
                config_hash=active.config_hash,
                joint_position_linf_rad=float(np.max(np.abs(joint_position_noise))),
                joint_velocity_linf_rad_s=float(np.max(np.abs(joint_velocity_noise))),
                root_tilt_angle_rad=tilt_angle,
                root_linear_velocity_linf_mps=float(np.max(np.abs(root_linear_noise))),
                root_angular_velocity_linf_rad_s=float(np.max(np.abs(root_angular_noise))),
            )
            generated.append((perturbed, record))
    return tuple(generated)


class RecoveryEntryMatcher:
    """Find diverse fallen reference entries with a proven upright future."""

    def __init__(
        self,
        motions: Sequence[RecoveryReferenceMotion],
        *,
        config: RecoveryEntrySearchConfig | None = None,
    ) -> None:
        if not motions or len({item.motion_id for item in motions}) != len(motions):
            raise ValueError("recovery matcher requires unique reference motions")
        self.motions = tuple(motions)
        self.config = config or RecoveryEntrySearchConfig()
        self._by_id = {item.motion_id: item for item in self.motions}
        self._candidates = self._mine_candidates()
        if not self._candidates:
            raise ValueError("recovery references contain no valid successor entries")
        self.library_hash = hash_json(
            {
                "sources": [item.source_hash for item in self.motions],
                "search_config_hash": self.config.config_hash,
            }
        )

    @classmethod
    def from_paths(
        cls,
        paths: Sequence[Path],
        *,
        config: RecoveryEntrySearchConfig | None = None,
    ) -> RecoveryEntryMatcher:
        return cls(tuple(RecoveryReferenceMotion.from_npz(path) for path in paths), config=config)

    def _mine_candidates(self) -> tuple[tuple[str, int, int], ...]:
        found: list[tuple[str, int, int]] = []
        cfg = self.config
        for motion in self.motions:
            gravity = np.stack(
                tuple(body_gravity_vector(value) for value in motion.qpos[:, 3:7])
            )
            upright = -gravity[:, 2]
            stable = (motion.qpos[:, 2] >= cfg.successor_pelvis_height_m) & (
                upright >= cfg.successor_upright_projection
            )
            streak = 0
            successor_ends: list[int] = []
            for frame, value in enumerate(stable):
                streak = streak + 1 if bool(value) else 0
                if streak == cfg.successor_hold_frames:
                    successor_ends.append(frame)
            for frame in range(0, motion.qpos.shape[0] - 1, cfg.candidate_stride_frames):
                if (
                    motion.qpos[frame, 2] > cfg.maximum_entry_pelvis_height_m
                    or upright[frame] > cfg.maximum_entry_upright_projection
                ):
                    continue
                future = next(
                    (
                        end
                        for end in successor_ends
                        if cfg.minimum_future_offset_frames < end - frame
                        < cfg.maximum_future_offset_frames
                    ),
                    None,
                )
                if future is not None:
                    found.append((motion.motion_id, frame, future))
        return tuple(found)

    def match(
        self,
        snapshot: RecoverySnapshot,
        *,
        maximum_matches: int | None = None,
    ) -> tuple[RecoveryEntryMatch, ...]:
        limit = self.config.maximum_matches if maximum_matches is None else maximum_matches
        if not 1 <= limit <= self.config.maximum_matches:
            raise ValueError("recovery match result limit is invalid")
        snapshot_gravity = body_gravity_vector(snapshot.qpos[3:7])
        scored: list[RecoveryEntryMatch] = []
        for motion_id, frame, successor_end in self._candidates:
            motion = self._by_id[motion_id]
            reference = motion.qpos[frame]
            gravity_distance = float(
                np.linalg.norm(snapshot_gravity - body_gravity_vector(reference[3:7]))
            )
            joint_rmse = float(
                np.sqrt(np.mean(np.square(snapshot.qpos[7:36] - reference[7:36])))
            )
            height_error = abs(float(snapshot.qpos[2] - reference[2]))
            score = (
                self.config.gravity_weight * gravity_distance
                + self.config.joint_weight * joint_rmse
                + self.config.height_weight * height_error
            )
            scored.append(
                RecoveryEntryMatch(
                    motion_id=motion_id,
                    source_hash=motion.source_hash,
                    entry_frame=frame,
                    successor_end_frame=successor_end,
                    score=score,
                    joint_rmse_rad=joint_rmse,
                    gravity_distance=gravity_distance,
                    pelvis_height_error_m=height_error,
                    search_config_hash=self.config.config_hash,
                )
            )
        selected: list[RecoveryEntryMatch] = []
        for item in sorted(scored, key=lambda value: (value.score, value.match_hash)):
            if any(
                item.motion_id == chosen.motion_id
                and abs(item.entry_frame - chosen.entry_frame)
                < self.config.nonmaximum_spacing_frames
                for chosen in selected
            ):
                continue
            selected.append(item)
            if len(selected) == limit:
                break
        if not selected:
            raise ValueError("recovery matcher found no diverse candidates")
        return tuple(selected)


@dataclass(frozen=True)
class RecoveryBridgeTrial:
    """One physical development trial for a privileged teacher bridge."""

    snapshot_hash: str
    match: RecoveryEntryMatch
    teacher_policy_hash: str
    time_dilation: int
    succeeded: bool
    final_stable_sec: float
    executed_sec: float
    peak_root_angular_speed_rad_s: float
    final_pelvis_height_m: float
    finite_state: bool
    ready_handoff_triggered: bool
    physics_backend: str = "mujoco_cpu"
    activation_ceiling: str = "SIM_ONLY"
    hardware_command_sent: bool = False
    schema_version: str = "rosclaw_soccer.recovery_bridge_trial.v1"

    def __post_init__(self) -> None:
        values = (
            self.final_stable_sec,
            self.executed_sec,
            self.peak_root_angular_speed_rad_s,
            self.final_pelvis_height_m,
        )
        if (
            not _valid_hash(self.snapshot_hash)
            or not _valid_hash(self.teacher_policy_hash)
            or not 1 <= self.time_dilation <= 4
            or any(not math.isfinite(value) or value < 0.0 for value in values)
            or self.physics_backend != "mujoco_cpu"
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_command_sent
            or (self.succeeded and (not self.finite_state or not self.ready_handoff_triggered))
        ):
            raise ValueError("recovery bridge trial is invalid")

    @property
    def trial_hash(self) -> str:
        return str(hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["match"] = asdict(self.match)
        return payload


def select_recovery_bridge_trial(
    trials: Sequence[RecoveryBridgeTrial],
) -> RecoveryBridgeTrial:
    """Select a trial without letting match proximity outrank physics."""

    if not trials or len({item.snapshot_hash for item in trials}) != 1:
        raise ValueError("bridge selection requires trials for exactly one snapshot")
    # A successful successor state dominates.  Among successes prefer less
    # temporal intervention, lower peak momentum, and then shorter execution.
    return min(
        trials,
        key=lambda item: (
            not item.succeeded,
            item.time_dilation,
            item.peak_root_angular_speed_rad_s,
            item.executed_sec,
            item.match.score,
            item.trial_hash,
        ),
    )


def build_recovery_bridge_schedule(
    trials: Sequence[RecoveryBridgeTrial],
) -> dict[str, Any]:
    """Build a content-addressed development schedule from physical outcomes."""

    if not trials:
        raise ValueError("recovery bridge schedule requires physical trials")
    grouped: dict[str, list[RecoveryBridgeTrial]] = {}
    for item in trials:
        grouped.setdefault(item.snapshot_hash, []).append(item)
    selected = [select_recovery_bridge_trial(grouped[key]) for key in sorted(grouped)]
    payload: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.recovery_bridge_schedule.v1",
        "snapshot_count": len(selected),
        "passed_snapshot_count": sum(item.succeeded for item in selected),
        "development_pass_rate": sum(item.succeeded for item in selected) / len(selected),
        "selected_trials": [item.to_dict() | {"trial_hash": item.trial_hash} for item in selected],
        "selection_rule": (
            "SUCCESSOR_STATE_THEN_MIN_TIME_DILATION_THEN_MIN_PEAK_ANGULAR_SPEED"
        ),
        "claim_boundary": "PRIVILEGED_TEACHER_DEVELOPMENT_NOT_DEPLOYABLE_GATE",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    payload["schedule_hash"] = hash_json(payload)
    return payload


__all__ = [
    "RecoveryBridgeTrial",
    "RecoveryEntryMatch",
    "RecoveryEntryMatcher",
    "RecoveryEntrySearchConfig",
    "RecoveryPerturbation",
    "RecoveryPerturbationConfig",
    "RecoveryReferenceMotion",
    "body_gravity_vector",
    "build_recovery_perturbation_holdout",
    "build_recovery_bridge_schedule",
    "select_recovery_bridge_trial",
]
