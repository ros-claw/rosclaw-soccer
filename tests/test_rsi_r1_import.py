import json
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.rsi.r1_import import import_r1_parent
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def _receipt(path: Path, payload: dict) -> None:
    path.write_text(json.dumps({**payload, "manifest_hash": hash_json(payload)}), encoding="utf-8")


def _fixture(root: Path) -> Path:
    root.mkdir()
    scene = "s200"
    stance = -0.17
    shot = {"in_frame": True, "keeper_touched": False, "launch_speed_mps": 9.6}
    _receipt(
        root / "protocol.json",
        {
            "training_authorized": False,
            "promoted": False,
            "bindings": {"source_hash": "sha256:" + "a" * 64, "kick_assets": {}},
            "scenes": [{"name": scene, "ball": [1.92, -0.8, 0.115]}],
        },
    )
    folder = root / scene / "lat--0.17"
    folder.mkdir(parents=True)
    evidence_hashes = []
    for label in ("sweep-0", "confirm-1", "confirm-2"):
        stem = folder / label
        trace = stem.with_suffix(".npz")
        np.savez_compressed(
            trace,
            ball_pose=np.array([[1.92, -0.8, 0.115]]),
            red_finisher_pelvis_pose=np.array([[0.0, 0.0, 0.8]]),
        )
        physics_path = folder / f"{label}-physics.json"
        _receipt(
            physics_path,
            {
                "result": {
                    "safe": True,
                    "scenario_hash": hash_json("s"),
                    "config_hash": hash_json("c"),
                },
                "rows": [1],
            },
        )
        file_hash = hash_bytes(trace.read_bytes())
        evidence_hashes.append(file_hash)
        _receipt(
            stem.with_suffix(".json"),
            {
                "file_hash": file_hash,
                "physics_file_hash": hash_bytes(physics_path.read_bytes()),
                "scene": scene,
                "stance_lateral_m": stance,
                "safe": True,
                "chain_clean": True,
                "shot": shot,
            },
        )
    _receipt(
        root / "bank.json",
        {
            "training_authorized": False,
            "promoted": False,
            "bank": [
                {
                    "scene": scene,
                    "stance_lateral_m": stance,
                    "goals": True,
                    "repeats_identical": True,
                    "shot": shot,
                    "evidence_hashes": evidence_hashes,
                }
            ],
        },
    )
    return root


def test_r1_import_deduplicates_teacher_executions(tmp_path):
    report = import_r1_parent(_fixture(tmp_path / "r1"))
    assert report["physical_execution_count"] == 3
    assert report["unique_physical_trajectory_count"] == 1
    assert report["teacher_active"]
    assert not report["training_authorized"]
    assert report["episodes"][0]["episode"]["partition"] == "CONSUMED_DEV"


def test_r1_import_rejects_tampered_trajectory(tmp_path):
    root = _fixture(tmp_path / "r1")
    trace = root / "s200" / "lat--0.17" / "confirm-1.npz"
    trace.write_bytes(trace.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="reauthentication"):
        import_r1_parent(root)


def test_r1_import_rejects_unsafe_scene_id_even_with_valid_manifest(tmp_path):
    root = _fixture(tmp_path / "r1")
    protocol_path = root / "protocol.json"
    payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    payload.pop("manifest_hash")
    payload["scenes"][0]["name"] = "../outside"
    _receipt(protocol_path, payload)
    with pytest.raises(ValueError, match="unsafe scene"):
        import_r1_parent(root)
