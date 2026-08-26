from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.skills.athlete_foundation.full_body_goalkeeper_motion import (
    FullBodyGoalkeeperMotionManifest,
    load_full_body_goalkeeper_motion_library,
)

_HASH = "sha256:" + "a" * 64
_COMMIT = "976a81ff19b7306bafbe923d2890066b68a85271"
_FAMILIES = (
    "lefthand",
    "righthand",
    "leftjump",
    "rightjump",
    "leftstep",
    "rightstep",
)
_MASK = tuple(index not in {13, 14, 19, 20, 21, 26, 27, 28} for index in range(29))


def _manifest(archive_hash: str = _HASH) -> FullBodyGoalkeeperMotionManifest:
    return FullBodyGoalkeeperMotionManifest(
        archive_file="full-body.npz",
        archive_hash=archive_hash,
        source_motion_hashes=tuple((name, _HASH) for name in _FAMILIES),
        source_license_hash=_HASH,
        source_commit=_COMMIT,
        family_frame_counts=tuple((name, 3) for name in _FAMILIES),
    )


def _write_bundle(root: Path) -> Path:
    arrays: dict[str, np.ndarray] = {}
    for family_index, family in enumerate(_FAMILIES):
        arrays[f"{family}__time_sec"] = np.arange(3, dtype=np.float32) / 30.0
        qpos = np.zeros((3, 29), dtype=np.float32)
        qpos[:, 3] = np.linspace(0.0, 0.6 + 0.1 * family_index, 3)
        arrays[f"{family}__qpos_29"] = qpos
        arrays[f"{family}__qvel_29"] = np.zeros((3, 29), dtype=np.float32)
        root_position = np.zeros((3, 3), dtype=np.float32)
        root_position[:, 2] = 0.8
        arrays[f"{family}__root_position"] = root_position
        arrays[f"{family}__root_position_local"] = root_position.copy()
        quaternion = np.zeros((3, 4), dtype=np.float32)
        quaternion[:, 3] = 1.0
        arrays[f"{family}__root_quaternion_xyzw"] = quaternion
        arrays[f"{family}__root_linear_velocity"] = np.zeros((3, 3), dtype=np.float32)
        arrays[f"{family}__root_angular_velocity"] = np.zeros((3, 3), dtype=np.float32)
        arrays[f"{family}__link_position"] = np.zeros((3, 17, 3), dtype=np.float32)
        arrays[f"{family}__link_quaternion_xyzw"] = np.pad(
            np.ones((3, 17, 1), dtype=np.float32), ((0, 0), (0, 0), (3, 0))
        )
        arrays[f"{family}__link_linear_velocity"] = np.zeros((3, 17, 3), dtype=np.float32)
        arrays[f"{family}__link_angular_velocity"] = np.zeros((3, 17, 3), dtype=np.float32)
    archive = root / "full-body.npz"
    np.savez_compressed(archive, **arrays)
    manifest = _manifest("sha256:" + hashlib.sha256(archive.read_bytes()).hexdigest())
    path = root / "full-body.json"
    path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")
    return path


def test_full_body_library_preserves_leg_motion_and_marks_missing_joints(tmp_path: Path) -> None:
    library = load_full_body_goalkeeper_motion_library(_write_bundle(tmp_path))

    frame = library.sample("leftjump", time_sec=1.0 / 30.0)
    assert frame.qpos_29[3] == pytest.approx(0.4)
    assert tuple(frame.source_joint_mask) == _MASK
    assert np.all(frame.qpos_29[np.logical_not(frame.source_joint_mask)] == 0.0)
    assert library.manifest.joint_names == G1_DDS_JOINT_NAMES
    assert library.manifest.external_teacher_only
    assert not library.manifest.champion_eligible


def test_full_body_library_rejects_fabricated_unobserved_joint(tmp_path: Path) -> None:
    path = _write_bundle(tmp_path)
    archive_path = tmp_path / "full-body.npz"
    with np.load(archive_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    arrays["leftjump__qpos_29"][:, 13] = 0.1
    np.savez_compressed(archive_path, **arrays)
    manifest = _manifest("sha256:" + hashlib.sha256(archive_path.read_bytes()).hexdigest())
    path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")

    with pytest.raises(ValueError, match="not neutral"):
        load_full_body_goalkeeper_motion_library(path)


def test_manifest_rejects_commercial_or_champion_claim() -> None:
    with pytest.raises(ValueError, match="research-only"):
        FullBodyGoalkeeperMotionManifest(
            **{**_manifest().__dict__, "commercial_use_allowed": True}
        )
