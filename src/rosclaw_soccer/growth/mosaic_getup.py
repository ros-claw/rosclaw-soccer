"""Content-bound MOSAIC G1 get-up skill for simulation-only recovery.

The source NPZ already contains retargeted G1 joint and body trajectories.
This module never unpickles data and never commands a robot.  It distils one
complete fallen-to-standing sequence into a JSON skill bound to the qualified
G1 body, MuJoCo scene, MOSAIC tensor semantics, and GMT checkpoint.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.growth.mosaic_g1_contract import (
    MOSAIC_G1_ISAACLAB_BODY_NAMES,
    MOSAIC_G1_ISAACLAB_JOINT_NAMES,
    MOSAIC_G1_SEMANTIC_CONTRACT_HASH,
)
from rosclaw_soccer.growth.mosaic_gmt import (
    _quat_multiply_wxyz,
    load_mosaic_gmt_torch,
)
from rosclaw_soccer.providers.g1.asset_qualification import g1_body_hash
from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


@dataclass(frozen=True)
class G1MosaicGMTGetUpSkill:
    checkpoint_hash: str
    checkpoint_contract_hash: str
    body_hash: str
    physics_scene_hash: str
    dataset_readme_hash: str
    source_hash: str
    semantic_contract_hash: str
    source_relative_path: str
    source_fps: float
    source_start_frame: int
    source_standing_frame: int
    source_stop_frame: int
    relative_times_sec: tuple[float, ...]
    raw_joint_position_rad: tuple[tuple[float, ...], ...]
    raw_joint_velocity_rad_s: tuple[tuple[float, ...], ...]
    aligned_torso_quaternion_wxyz: tuple[tuple[float, ...], ...]
    initial_pelvis_height_m: float
    initial_upright_projection: float
    final_pelvis_height_m: float
    final_upright_projection: float
    dataset_license: str = "CDLA-Permissive-2.0"
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw.growth.g1_mosaic_gmt_getup_skill.v1"

    def __post_init__(self) -> None:
        count = len(self.relative_times_sec)
        position = np.asarray(self.raw_joint_position_rad, dtype=np.float64)
        velocity = np.asarray(self.raw_joint_velocity_rad_s, dtype=np.float64)
        quaternion = np.asarray(self.aligned_torso_quaternion_wxyz, dtype=np.float64)
        hashes = (
            self.checkpoint_hash,
            self.checkpoint_contract_hash,
            self.body_hash,
            self.physics_scene_hash,
            self.dataset_readme_hash,
            self.source_hash,
            self.semantic_contract_hash,
        )
        if (
            any(not value.startswith("sha256:") or len(value) != 71 for value in hashes)
            or count < 100
            or position.shape != (count, 29)
            or velocity.shape != position.shape
            or quaternion.shape != (count, 4)
            or not all(np.isfinite(value).all() for value in (position, velocity, quaternion))
            or tuple(sorted(self.relative_times_sec)) != self.relative_times_sec
            or self.relative_times_sec[0] != 0.0
            or not 2.5 <= self.relative_times_sec[-1] <= 5.0
            or not 30.0 <= self.source_fps <= 240.0
            or not 0
            <= self.source_start_frame
            < self.source_standing_frame
            < self.source_stop_frame
            or self.source_stop_frame - self.source_start_frame != count
            or np.max(np.abs(np.linalg.norm(quaternion, axis=1) - 1.0)) > 2.0e-4
            or self.initial_pelvis_height_m > 0.45
            or abs(self.initial_upright_projection) > 0.65
            or self.final_pelvis_height_m < 0.62
            or self.final_upright_projection < 0.75
            or self.semantic_contract_hash != MOSAIC_G1_SEMANTIC_CONTRACT_HASH
            or self.dataset_license != "CDLA-Permissive-2.0"
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("MOSAIC GMT get-up skill is invalid or unqualified")

    @property
    def duration_sec(self) -> float:
        return float(self.relative_times_sec[-1])

    @property
    def skill_hash(self) -> str:
        return str(hash_json(asdict(self)))


def build_g1_mosaic_gmt_getup_skill(
    *,
    mosaic_root: Path,
    source_path: Path,
    checkpoint_path: Path,
    asset_root: Path,
    output_path: Path,
    source_start_frame: int = 0,
    standing_hold_frames: int = 25,
    standing_tail_frames: int = 50,
    device: Any = "cpu",
) -> G1MosaicGMTGetUpSkill:
    """Distil one complete low-posture-to-standing G1 motion without pickle."""

    root = mosaic_root.expanduser().resolve()
    source = source_path.expanduser().resolve()
    assets = asset_root.expanduser().resolve()
    destination = output_path.expanduser().resolve()
    if root not in source.parents or not source.is_file() or source.suffix != ".npz":
        raise ValueError("MOSAIC get-up source must be one NPZ below the dataset root")
    if destination.exists():
        raise ValueError("MOSAIC get-up output already exists")
    readme = root / "README.md"
    if (
        not readme.is_file()
        or "license: cdla-permissive-2.0" not in readme.read_text(encoding="utf-8").lower()
    ):
        raise ValueError("MOSAIC get-up requires the CDLA-Permissive-2.0 dataset card")
    scene = assets / "g1_description" / "scene_with_ball.xml"
    if not scene.is_file():
        raise ValueError("MOSAIC get-up requires the qualified G1 MuJoCo scene")
    _, contract = load_mosaic_gmt_torch(checkpoint_path, device=device)
    with np.load(source, allow_pickle=False) as data:
        required = {"fps", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w"}
        if not required.issubset(data.files):
            raise ValueError("MOSAIC get-up source arrays are incomplete")
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        position = np.asarray(data["joint_pos"], dtype=np.float64)
        velocity = np.asarray(data["joint_vel"], dtype=np.float64)
        body_position = np.asarray(data["body_pos_w"], dtype=np.float64)
        body_quaternion = np.asarray(data["body_quat_w"], dtype=np.float64)
    if (
        position.ndim != 2
        or position.shape[1] != 29
        or velocity.shape != position.shape
        or body_position.shape != (position.shape[0], 30, 3)
        or body_quaternion.shape != (position.shape[0], 30, 4)
        or not all(
            np.isfinite(value).all()
            for value in (position, velocity, body_position, body_quaternion)
        )
        or not math.isfinite(fps)
        or not 30.0 <= fps <= 240.0
    ):
        raise ValueError("MOSAIC get-up source shapes or values are invalid")
    if not 0 <= source_start_frame < position.shape[0] - standing_hold_frames:
        raise ValueError("MOSAIC get-up start frame is invalid")
    pelvis_height = body_position[:, 0, 2]
    pelvis_quaternion = body_quaternion[:, 0]
    upright = 2.0 * (pelvis_quaternion[:, 0] ** 2 + pelvis_quaternion[:, 3] ** 2) - 1.0
    if pelvis_height[source_start_frame] > 0.45 or abs(upright[source_start_frame]) > 0.65:
        raise ValueError("MOSAIC get-up source does not begin in a recoverable down posture")
    standing = (pelvis_height >= 0.62) & (upright >= 0.75)
    standing_frame = None
    for frame in range(source_start_frame + 1, position.shape[0] - standing_hold_frames + 1):
        if bool(np.all(standing[frame : frame + standing_hold_frames])):
            standing_frame = frame
            break
    if standing_frame is None:
        raise ValueError("MOSAIC get-up source has no sustained standing completion")
    stop = min(position.shape[0], standing_frame + standing_tail_frames)
    if stop - standing_frame < standing_hold_frames:
        raise ValueError("MOSAIC get-up source has insufficient standing tail")
    torso = body_quaternion[source_start_frame:stop, 9]
    torso /= np.linalg.norm(torso, axis=1, keepdims=True)
    initial = torso[0]
    yaw = math.atan2(
        2.0 * (initial[0] * initial[3] + initial[1] * initial[2]),
        1.0 - 2.0 * (initial[2] ** 2 + initial[3] ** 2),
    )
    inverse_yaw = np.asarray((math.cos(yaw / 2.0), 0.0, 0.0, -math.sin(yaw / 2.0)))
    aligned = _quat_multiply_wxyz(np.broadcast_to(inverse_yaw, torso.shape), torso)
    aligned /= np.linalg.norm(aligned, axis=1, keepdims=True)
    times = np.arange(stop - source_start_frame, dtype=np.float64) / fps
    skill = G1MosaicGMTGetUpSkill(
        checkpoint_hash=contract.checkpoint_hash,
        checkpoint_contract_hash=contract.contract_hash,
        body_hash=g1_body_hash(assets),
        physics_scene_hash=hash_bytes(scene.read_bytes()),
        dataset_readme_hash=hash_bytes(readme.read_bytes()),
        source_hash=hash_bytes(source.read_bytes()),
        semantic_contract_hash=MOSAIC_G1_SEMANTIC_CONTRACT_HASH,
        source_relative_path=source.relative_to(root).as_posix(),
        source_fps=fps,
        source_start_frame=source_start_frame,
        source_standing_frame=standing_frame,
        source_stop_frame=stop,
        relative_times_sec=tuple(float(value) for value in times),
        raw_joint_position_rad=tuple(
            tuple(float(value) for value in row) for row in position[source_start_frame:stop]
        ),
        raw_joint_velocity_rad_s=tuple(
            tuple(float(value) for value in row) for row in velocity[source_start_frame:stop]
        ),
        aligned_torso_quaternion_wxyz=tuple(
            tuple(float(value) for value in row) for row in aligned
        ),
        initial_pelvis_height_m=float(pelvis_height[source_start_frame]),
        initial_upright_projection=float(upright[source_start_frame]),
        final_pelvis_height_m=float(pelvis_height[stop - 1]),
        final_upright_projection=float(upright[stop - 1]),
    )
    payload = asdict(skill)
    payload.update(
        joint_names=list(G1_DDS_JOINT_NAMES),
        raw_joint_names=list(MOSAIC_G1_ISAACLAB_JOINT_NAMES),
        raw_body_names=list(MOSAIC_G1_ISAACLAB_BODY_NAMES),
        skill_hash=skill.skill_hash,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return skill


def load_g1_mosaic_gmt_getup_skill(path: Path) -> G1MosaicGMTGetUpSkill:
    """Load and content-verify one no-pickle G1 get-up skill."""

    source = path.expanduser().resolve()
    if not source.is_file() or not 1 <= source.stat().st_size <= 32 * 1024 * 1024:
        raise ValueError("MOSAIC GMT get-up skill is missing, empty, or oversized")
    payload = json.loads(source.read_text(encoding="utf-8"))
    expected_hash = payload.pop("skill_hash", None)
    if tuple(payload.pop("joint_names", ())) != G1_DDS_JOINT_NAMES:
        raise ValueError("MOSAIC GMT get-up canonical joint names are invalid")
    if tuple(payload.pop("raw_joint_names", ())) != MOSAIC_G1_ISAACLAB_JOINT_NAMES:
        raise ValueError("MOSAIC GMT get-up raw joint names are invalid")
    if tuple(payload.pop("raw_body_names", ())) != MOSAIC_G1_ISAACLAB_BODY_NAMES:
        raise ValueError("MOSAIC GMT get-up raw body names are invalid")
    for name in (
        "relative_times_sec",
        "raw_joint_position_rad",
        "raw_joint_velocity_rad_s",
        "aligned_torso_quaternion_wxyz",
    ):
        payload[name] = tuple(
            tuple(float(item) for item in value) if isinstance(value, list) else float(value)
            for value in payload[name]
        )
    try:
        skill = G1MosaicGMTGetUpSkill(**payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("MOSAIC GMT get-up payload is invalid") from exc
    if expected_hash != skill.skill_hash:
        raise ValueError("MOSAIC GMT get-up content hash mismatch")
    return skill


__all__ = [
    "G1MosaicGMTGetUpSkill",
    "build_g1_mosaic_gmt_getup_skill",
    "load_g1_mosaic_gmt_getup_skill",
]
