from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.skills.goalkeeper_v2.external_motion import (
    ExternalGoalkeeperMotionManifest,
    load_external_goalkeeper_motion_decoder,
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


def _manifest() -> ExternalGoalkeeperMotionManifest:
    return ExternalGoalkeeperMotionManifest(
        archive_file="reference.npz",
        archive_hash=_HASH,
        source_motion_hashes=tuple((name, _HASH) for name in _FAMILIES),
        source_license_hash=_HASH,
        source_commit=_COMMIT,
        family_names=_FAMILIES,
    )


def test_external_motion_is_research_only_and_never_champion() -> None:
    manifest = _manifest()

    assert manifest.external_teacher_only
    assert not manifest.commercial_use_allowed
    assert not manifest.champion_eligible
    assert manifest.activation_ceiling == "SIM_ONLY"
    with pytest.raises(ValueError, match="research boundary"):
        ExternalGoalkeeperMotionManifest(**{**manifest.__dict__, "commercial_use_allowed": True})


def test_external_motion_decoder_is_bounded_and_releases(tmp_path: Path) -> None:
    motions = {name: np.zeros((25, 29), dtype=np.float32) for name in _FAMILIES}
    motions["leftjump"][:, 15] = np.linspace(0.0, 0.50, 25)
    archive = tmp_path / "reference.npz"
    np.savez_compressed(archive, **motions)
    manifest = ExternalGoalkeeperMotionManifest(
        **{
            **_manifest().__dict__,
            "archive_hash": "sha256:" + hashlib.sha256(archive.read_bytes()).hexdigest(),
        }
    )
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")

    decoder = load_external_goalkeeper_motion_decoder(path)

    assert decoder.residual(region=2, elapsed_sec=0.80)[15] == pytest.approx(0.50)
    assert decoder.residual(region=2, elapsed_sec=1.20)[15] == pytest.approx(0.0)
    assert np.all(decoder.residual(region=2, elapsed_sec=0.40)[:12] == 0.0)


def test_external_motion_loader_rejects_manifest_tampering(tmp_path: Path) -> None:
    path = tmp_path / "reference.json"
    payload = _manifest().to_dict()
    payload["commercial_use_allowed"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid"):
        load_external_goalkeeper_motion_decoder(path)
