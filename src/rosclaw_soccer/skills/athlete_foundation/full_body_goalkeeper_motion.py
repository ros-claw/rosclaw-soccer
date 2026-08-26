"""Lossless 21-to-29 DoF adapter for research goalkeeper motion priors.

The historical S11 adapter intentionally extracted only nine upper-body
joints.  That made dives look like arm gestures on frozen legs.  This module
keeps every source leg, waist-yaw, and arm trajectory, plus the root and link
kinematics needed by a future whole-body tracker.  The eight unavailable G1
joints remain explicitly masked and neutral rather than being fabricated.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.sim.contracts import hash_json

_REFERENCE_COMMIT = "976a81ff19b7306bafbe923d2890066b68a85271"
_REFERENCE_REPOSITORY = "https://github.com/InternRobotics/Humanoid-Goalkeeper"
_EXPECTED_LICENSE_HASH = "sha256:6c8cd1cdbe7accec4f63b6c3afb45ce0ffae9ed6abc0ca55acf5900b37970a82"
_MOTION_ROOT = Path("legged_gym/resources/datasets/goalkeeper")
_FAMILIES = (
    "lefthand",
    "righthand",
    "leftjump",
    "rightjump",
    "leftstep",
    "rightstep",
)
_SOURCE_TO_G1 = (
    (0, 0),
    (1, 1),
    (2, 2),
    (3, 3),
    (4, 4),
    (5, 5),
    (6, 6),
    (7, 7),
    (8, 8),
    (9, 9),
    (10, 10),
    (11, 11),
    (12, 12),
    (13, 15),
    (14, 16),
    (15, 17),
    (16, 18),
    (17, 22),
    (18, 23),
    (19, 24),
    (20, 25),
)
_SOURCE_FRAME_RATE_HZ = 30.0
_SIGNALS = (
    "time_sec",
    "qpos_29",
    "qvel_29",
    "root_position",
    "root_position_local",
    "root_quaternion_xyzw",
    "root_linear_velocity",
    "root_angular_velocity",
    "link_position",
    "link_quaternion_xyzw",
    "link_linear_velocity",
    "link_angular_velocity",
)


def _source_mask() -> tuple[bool, ...]:
    mask = [False] * len(G1_DDS_JOINT_NAMES)
    for _, target_index in _SOURCE_TO_G1:
        mask[target_index] = True
    return tuple(mask)


@dataclass(frozen=True)
class FullBodyGoalkeeperMotionManifest:
    archive_file: str
    archive_hash: str
    source_motion_hashes: tuple[tuple[str, str], ...]
    source_license_hash: str
    source_commit: str
    family_frame_counts: tuple[tuple[str, int], ...]
    joint_names: tuple[str, ...] = G1_DDS_JOINT_NAMES
    source_joint_mask: tuple[bool, ...] = _source_mask()
    neutral_unobserved_joint_names: tuple[str, ...] = (
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    )
    source_frame_rate_hz: float = _SOURCE_FRAME_RATE_HZ
    root_quaternion_order: str = "xyzw"
    attribution_required: bool = True
    commercial_use_allowed: bool = False
    share_alike_required: bool = True
    external_teacher_only: bool = True
    champion_eligible: bool = False
    activation_ceiling: str = "SIM_ONLY"
    repository: str = _REFERENCE_REPOSITORY
    schema_version: str = "rosclaw_soccer.full_body_goalkeeper_motion_manifest.v1"

    def __post_init__(self) -> None:
        if Path(self.archive_file).name != self.archive_file or not self.archive_file.endswith(
            ".npz"
        ):
            raise ValueError("full-body goalkeeper motion requires a sibling NPZ archive")
        if tuple(name for name, _ in self.source_motion_hashes) != _FAMILIES:
            raise ValueError("source goalkeeper motion family order changed")
        if tuple(name for name, _ in self.family_frame_counts) != _FAMILIES or any(
            count < 3 for _, count in self.family_frame_counts
        ):
            raise ValueError("goalkeeper motion frame counts are invalid")
        for value in (
            self.archive_hash,
            self.source_license_hash,
            *(value for _, value in self.source_motion_hashes),
        ):
            if not value.startswith("sha256:"):
                raise ValueError("goalkeeper motion provenance requires content hashes")
        if self.source_commit != _REFERENCE_COMMIT or self.repository != _REFERENCE_REPOSITORY:
            raise ValueError("goalkeeper motion source identity changed")
        if self.joint_names != G1_DDS_JOINT_NAMES or self.source_joint_mask != _source_mask():
            raise ValueError("goalkeeper motion G1 joint mapping changed")
        missing = tuple(
            name for name, available in zip(self.joint_names, self.source_joint_mask, strict=True)
            if not available
        )
        if missing != self.neutral_unobserved_joint_names:
            raise ValueError("unobserved goalkeeper joints are not explicitly neutral")
        if not math.isclose(self.source_frame_rate_hz, 30.0, abs_tol=1e-12):
            raise ValueError("goalkeeper source frame rate changed")
        if self.root_quaternion_order != "xyzw":
            raise ValueError("goalkeeper root quaternion contract changed")
        if (
            not self.attribution_required
            or self.commercial_use_allowed
            or not self.share_alike_required
            or not self.external_teacher_only
            or self.champion_eligible
            or self.activation_ceiling != "SIM_ONLY"
        ):
            raise ValueError("goalkeeper motion violated its research-only boundary")

    @property
    def manifest_hash(self) -> str:
        return str(hash_json(self.to_dict(include_hash=False)))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_motion_hashes"] = [list(item) for item in self.source_motion_hashes]
        payload["family_frame_counts"] = [list(item) for item in self.family_frame_counts]
        payload["joint_names"] = list(self.joint_names)
        payload["source_joint_mask"] = list(self.source_joint_mask)
        payload["neutral_unobserved_joint_names"] = list(
            self.neutral_unobserved_joint_names
        )
        if include_hash:
            payload["manifest_hash"] = self.manifest_hash
        return payload


@dataclass(frozen=True)
class FullBodyMotionFrame:
    family: str
    time_sec: float
    qpos_29: NDArray[np.float64]
    qvel_29: NDArray[np.float64]
    root_position_local: NDArray[np.float64]
    root_quaternion_xyzw: NDArray[np.float64]
    source_joint_mask: NDArray[np.bool_]


class FullBodyGoalkeeperMotionLibrary:
    """Interpolation-only view of a lossless external motion bundle."""

    def __init__(
        self,
        manifest: FullBodyGoalkeeperMotionManifest,
        arrays: dict[str, NDArray[np.float64]],
    ) -> None:
        self.manifest = manifest
        self._arrays = arrays

    @property
    def families(self) -> tuple[str, ...]:
        return _FAMILIES

    def sample(self, family: str, *, time_sec: float) -> FullBodyMotionFrame:
        if family not in _FAMILIES or not math.isfinite(time_sec):
            raise ValueError("full-body goalkeeper motion query is invalid")
        times = self._arrays[f"{family}__time_sec"]
        clipped = float(np.clip(time_sec, 0.0, times[-1]))
        upper = min(int(np.searchsorted(times, clipped, side="right")), len(times) - 1)
        lower = max(0, upper - 1)
        denominator = float(times[upper] - times[lower])
        alpha = 0.0 if denominator <= 0.0 else (clipped - float(times[lower])) / denominator

        def interpolate(signal: str) -> NDArray[np.float64]:
            values = self._arrays[f"{family}__{signal}"]
            return np.asarray((1.0 - alpha) * values[lower] + alpha * values[upper])

        quaternion = interpolate("root_quaternion_xyzw")
        norm = float(np.linalg.norm(quaternion))
        if norm <= 1e-9:
            raise ValueError("interpolated goalkeeper root quaternion is invalid")
        quaternion /= norm
        return FullBodyMotionFrame(
            family=family,
            time_sec=clipped,
            qpos_29=interpolate("qpos_29"),
            qvel_29=interpolate("qvel_29"),
            root_position_local=interpolate("root_position_local"),
            root_quaternion_xyzw=quaternion,
            source_joint_mask=np.asarray(self.manifest.source_joint_mask, dtype=np.bool_),
        )


def build_full_body_goalkeeper_motion_bundle(
    *,
    reference_checkout: Path,
    output_manifest_path: Path,
    source_checkout: Path,
) -> FullBodyGoalkeeperMotionManifest:
    """Convert the six upstream tensors without cropping or clipping any tracked joint."""

    root = reference_checkout.expanduser().resolve()
    output = output_manifest_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if output == checkout or checkout in output.parents:
        raise ValueError("external goalkeeper motion must remain outside source checkout")
    if output.suffix != ".json" or output.exists():
        raise ValueError("goalkeeper motion manifest requires a new JSON path")
    license_path = root / "LICENSE"
    if _file_hash(license_path) != _EXPECTED_LICENSE_HASH:
        raise ValueError("external goalkeeper motion license changed")
    if "Attribution-NonCommercial-ShareAlike 4.0" not in license_path.read_text(
        encoding="utf-8"
    ):
        raise ValueError("external goalkeeper motion license terms are unavailable")
    commit = subprocess.run(
        ("git", "-C", str(root), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != _REFERENCE_COMMIT:
        raise ValueError("external goalkeeper motion checkout is not pinned")

    import torch

    archive: dict[str, NDArray[np.generic]] = {}
    hashes: list[tuple[str, str]] = []
    frame_counts: list[tuple[str, int]] = []
    key_map = {
        "root_position": "base_position",
        "root_quaternion_xyzw": "base_pose",
        "root_linear_velocity": "base_velocity",
        "root_angular_velocity": "base_angular_velocity",
        "link_position": "link_position",
        "link_quaternion_xyzw": "link_orientation",
        "link_linear_velocity": "link_velocity",
        "link_angular_velocity": "link_angular_velocity",
    }
    for family in _FAMILIES:
        path = root / _MOTION_ROOT / f"{family}.pt"
        hashes.append((family, _file_hash(path)))
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError(f"external goalkeeper tensor changed: {family}")
        source_qpos = _tensor_array(payload.get("joint_position"), (None, 21), family)
        source_qvel = _tensor_array(payload.get("joint_velocity"), source_qpos.shape, family)
        frame_count = int(source_qpos.shape[0])
        frame_counts.append((family, frame_count))
        qpos = np.zeros((frame_count, 29), dtype=np.float32)
        qvel = np.zeros((frame_count, 29), dtype=np.float32)
        for source_index, target_index in _SOURCE_TO_G1:
            qpos[:, target_index] = source_qpos[:, source_index]
            qvel[:, target_index] = source_qvel[:, source_index]
        archive[f"{family}__time_sec"] = np.arange(frame_count, dtype=np.float32) / 30.0
        archive[f"{family}__qpos_29"] = qpos
        archive[f"{family}__qvel_29"] = qvel
        for target_name, source_name in key_map.items():
            archive[f"{family}__{target_name}"] = _tensor_array(
                payload.get(source_name), None, family
            )
        local = np.asarray(archive[f"{family}__root_position"], dtype=np.float32).copy()
        local[:, :2] -= local[:1, :2]
        archive[f"{family}__root_position_local"] = local
    archive_path = output.with_suffix(".npz")
    if archive_path.exists():
        raise ValueError("goalkeeper motion archive output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(archive_path, **archive)  # type: ignore[arg-type]
    manifest = FullBodyGoalkeeperMotionManifest(
        archive_file=archive_path.name,
        archive_hash=_file_hash(archive_path),
        source_motion_hashes=tuple(hashes),
        source_license_hash=_EXPECTED_LICENSE_HASH,
        source_commit=_REFERENCE_COMMIT,
        family_frame_counts=tuple(frame_counts),
    )
    output.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_full_body_goalkeeper_motion_library(
    manifest_path: Path,
) -> FullBodyGoalkeeperMotionLibrary:
    path = manifest_path.expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed_hash = str(payload.pop("manifest_hash", ""))
    try:
        payload["source_motion_hashes"] = tuple(
            (str(item[0]), str(item[1])) for item in payload["source_motion_hashes"]
        )
        payload["family_frame_counts"] = tuple(
            (str(item[0]), int(item[1])) for item in payload["family_frame_counts"]
        )
        for key in (
            "joint_names",
            "source_joint_mask",
            "neutral_unobserved_joint_names",
        ):
            payload[key] = tuple(payload[key])
        manifest = FullBodyGoalkeeperMotionManifest(**payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("full-body goalkeeper motion manifest is invalid") from exc
    if claimed_hash != manifest.manifest_hash:
        raise ValueError("full-body goalkeeper motion manifest hash mismatch")
    archive_path = path.parent / manifest.archive_file
    if _file_hash(archive_path) != manifest.archive_hash:
        raise ValueError("full-body goalkeeper motion archive hash mismatch")
    expected_keys = {f"{family}__{signal}" for family in _FAMILIES for signal in _SIGNALS}
    with np.load(archive_path, allow_pickle=False) as archive:
        if set(archive.files) != expected_keys:
            raise ValueError("full-body goalkeeper motion signal set changed")
        arrays = {name: np.asarray(archive[name], dtype=np.float64) for name in archive.files}
    missing_mask = np.logical_not(np.asarray(manifest.source_joint_mask, dtype=np.bool_))
    for family, frame_count in manifest.family_frame_counts:
        for signal in _SIGNALS:
            value = arrays[f"{family}__{signal}"]
            if value.shape[0] != frame_count or not np.all(np.isfinite(value)):
                raise ValueError(f"full-body goalkeeper signal is invalid: {family}/{signal}")
        if arrays[f"{family}__qpos_29"].shape != (frame_count, 29) or arrays[
            f"{family}__qvel_29"
        ].shape != (frame_count, 29):
            raise ValueError(f"full-body goalkeeper joint shape changed: {family}")
        if np.any(arrays[f"{family}__qpos_29"][:, missing_mask]) or np.any(
            arrays[f"{family}__qvel_29"][:, missing_mask]
        ):
            raise ValueError(f"unobserved goalkeeper joints are not neutral: {family}")
        times = arrays[f"{family}__time_sec"]
        if times.shape != (frame_count,) or not np.allclose(
            np.diff(times), 1.0 / manifest.source_frame_rate_hz, atol=1e-6
        ):
            raise ValueError(f"full-body goalkeeper timing changed: {family}")
        quaternions = arrays[f"{family}__root_quaternion_xyzw"]
        if quaternions.shape != (frame_count, 4) or not np.allclose(
            np.linalg.norm(quaternions, axis=1), 1.0, atol=2e-3
        ):
            raise ValueError(f"full-body goalkeeper root orientation is invalid: {family}")
    return FullBodyGoalkeeperMotionLibrary(manifest, arrays)


def _tensor_array(
    value: object,
    expected_shape: tuple[int | None, ...] | None,
    family: str,
) -> NDArray[np.float32]:
    if value is None or not hasattr(value, "detach"):
        raise ValueError(f"external goalkeeper tensor is missing: {family}")
    result = np.asarray(value.detach().cpu().numpy(), dtype=np.float32)
    if expected_shape is not None and (
        result.ndim != len(expected_shape)
        or any(
            expected is not None and actual != expected
            for actual, expected in zip(result.shape, expected_shape, strict=True)
        )
    ):
        raise ValueError(f"external goalkeeper tensor shape changed: {family}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"external goalkeeper tensor is non-finite: {family}")
    return result


def _file_hash(path: Path) -> str:
    if not path.is_file() or path.stat().st_size > 1024 * 1024 * 1024:
        raise ValueError(f"goalkeeper motion file is unavailable or oversized: {path}")
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "FullBodyGoalkeeperMotionLibrary",
    "FullBodyGoalkeeperMotionManifest",
    "FullBodyMotionFrame",
    "build_full_body_goalkeeper_motion_bundle",
    "load_full_body_goalkeeper_motion_library",
]
