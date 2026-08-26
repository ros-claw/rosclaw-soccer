from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.recovery_snapshot import (
    RecoverySnapshot,
    classify_recovery_posture,
    load_recovery_snapshot_corpus,
    write_recovery_snapshot_corpus,
)

_HASH = hash_json({"contract": "test"})


def _quaternion(*, roll: float = 0.0, pitch: float = 0.0) -> np.ndarray:
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    return np.asarray((cr * cp, sr * cp, cr * sp, -sr * sp), dtype=np.float64)


def _snapshot(*, environment_index: int = 0, stage: str = "RECOVERY_ENTRY") -> RecoverySnapshot:
    qpos = np.zeros(36, dtype=np.float64)
    qpos[2] = 0.30
    qpos[3:7] = _quaternion(roll=-math.pi / 2.0)
    return RecoverySnapshot(
        episode_seed=49_001,
        environment_index=environment_index,
        control_step=73,
        stage=stage,  # type: ignore[arg-type]
        save_kind="HAND",
        posture_cluster="LEFT_SIDE",
        qpos=qpos,
        qvel=np.zeros(35),
        applied_action=np.zeros(30),
        ball_position_m=np.asarray((4.4, -0.8, 0.7)),
        ball_velocity_mps=np.asarray((-0.3, 0.1, 0.2)),
        target_position_m=np.asarray((4.52, -0.9, 0.8)),
        left_foot_supported=False,
        right_foot_supported=True,
        failed=False,
        body_hash=_HASH,
        physics_scene_hash=_HASH,
        source_policy_hash=_HASH,
        source_config_hash=_HASH,
    )


def test_recovery_posture_classifier_routes_momentum_before_pose() -> None:
    common = {
        "pelvis_height_m": 0.28,
        "root_linear_speed_mps": 0.10,
        "root_angular_speed_rad_s": 0.20,
        "left_foot_supported": False,
        "right_foot_supported": True,
    }
    assert classify_recovery_posture(
        root_quaternion_wxyz=_quaternion(roll=-math.pi / 2.0), **common
    ) == "LEFT_SIDE"
    assert classify_recovery_posture(
        root_quaternion_wxyz=_quaternion(roll=math.pi / 2.0), **common
    ) == "RIGHT_SIDE"
    assert classify_recovery_posture(
        root_quaternion_wxyz=_quaternion(pitch=math.pi / 2.0), **common
    ) == "PRONE"
    assert classify_recovery_posture(
        root_quaternion_wxyz=_quaternion(pitch=-math.pi / 2.0), **common
    ) == "SUPINE"
    assert classify_recovery_posture(
        root_quaternion_wxyz=_quaternion(roll=-math.pi / 2.0),
        **{**common, "root_angular_speed_rad_s": 2.2},
    ) == "AIRBORNE_OR_HIGH_MOMENTUM"


def test_recovery_snapshot_corpus_round_trip_is_content_bound(tmp_path: Path) -> None:
    snapshots = (_snapshot(), _snapshot(environment_index=1, stage="FAILURE_TERMINAL"))
    manifest = write_recovery_snapshot_corpus(
        snapshots=snapshots,
        output_dir=tmp_path,
        corpus_name="post-save-hard-corners",
    )
    loaded = load_recovery_snapshot_corpus(tmp_path / "post-save-hard-corners.json")
    assert len(loaded) == 2
    assert loaded[0].snapshot_hash == snapshots[0].snapshot_hash
    assert manifest["cluster_counts"] == {"LEFT_SIDE": 2}
    assert manifest["stage_counts"] == {"FAILURE_TERMINAL": 1, "RECOVERY_ENTRY": 1}
    assert manifest["activation_ceiling"] == "SIM_ONLY"
    assert not manifest["hardware_authorized"]


def test_recovery_snapshot_corpus_rejects_tampered_archive(tmp_path: Path) -> None:
    write_recovery_snapshot_corpus(snapshots=(_snapshot(),), output_dir=tmp_path)
    archive = tmp_path / "recovery-snapshots.npz"
    archive.write_bytes(archive.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="archive hash mismatch"):
        load_recovery_snapshot_corpus(tmp_path / "recovery-snapshots.json")


def test_recovery_snapshot_fails_closed_on_nonfinite_or_mixed_contract(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="qvel"):
        replace(_snapshot(), qvel=np.full(35, np.nan))
    with pytest.raises(ValueError, match="mixes incompatible"):
        write_recovery_snapshot_corpus(
            snapshots=(
                _snapshot(),
                replace(_snapshot(environment_index=1), source_policy_hash=hash_json({"v": 2})),
            ),
            output_dir=tmp_path,
        )
