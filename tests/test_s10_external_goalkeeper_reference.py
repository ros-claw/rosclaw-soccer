from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.skills.goalkeeper_v2.external_reference import (
    HumanoidGoalkeeperReferenceManifest,
    load_humanoid_goalkeeper_reference_actor,
)

_HASH = "sha256:" + "a" * 64


def _manifest() -> HumanoidGoalkeeperReferenceManifest:
    return HumanoidGoalkeeperReferenceManifest(
        weights_file="reference.npz",
        weights_file_hash=_HASH,
        source_checkpoint_hash=_HASH,
        source_license_hash=_HASH,
        source_commit="976a81ff19b7306bafbe923d2890066b68a85271",
        joint_names=G1_DDS_JOINT_NAMES,
        default_joint_position_rad=(0.0,) * 29,
    )


def test_external_reference_manifest_is_fail_closed_and_never_champion() -> None:
    manifest = _manifest()

    assert manifest.external_teacher_only
    assert not manifest.commercial_use_allowed
    assert not manifest.champion_eligible
    assert manifest.activation_ceiling == "SIM_ONLY"
    with pytest.raises(ValueError, match="research boundary"):
        HumanoidGoalkeeperReferenceManifest(**{**manifest.__dict__, "commercial_use_allowed": True})


def test_external_reference_loader_rejects_tampering(tmp_path: Path) -> None:
    manifest = _manifest()
    path = tmp_path / "reference.json"
    payload = manifest.to_dict()
    payload["commercial_use_allowed"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid"):
        load_humanoid_goalkeeper_reference_actor(path)


def test_reference_action_requires_finite_g1_state() -> None:
    # The full reference is intentionally tested only when its separately
    # licensed external bundle exists; this unit test locks the manifest side.
    manifest = _manifest()
    assert len(manifest.joint_names) == 29
    assert np.isfinite(manifest.action_scale_rad)
