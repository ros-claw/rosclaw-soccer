from __future__ import annotations

from dataclasses import replace

import pytest
from rosclaw.continual.learner_backend import LearnerBackendContract, LearnerCapability

from rosclaw_soccer.skills.athlete_foundation.backend_catalog import (
    AthleteBackendCatalog,
    BackendArtifact,
    BackendReadiness,
    BackendReadinessStage,
)

_HASH = "sha256:" + "a" * 64


def _backend(index: int) -> LearnerBackendContract:
    return LearnerBackendContract(
        backend_id=f"backend-{index}",
        backend_version="1.0",
        source_url=f"https://example.org/backend-{index}",
        source_commit=f"{index + 1:x}" * 40,
        license_id="Apache-2.0",
        capabilities=(LearnerCapability.MOTION_TRACKING,),
        supported_body_ids=("unitree.g1.29dof",),
        training_available=True,
        inference_available=True,
    )


def _readiness(index: int, *, inference: bool) -> BackendReadiness:
    return BackendReadiness(
        backend=_backend(index),
        source_checkout_hash=_HASH,
        artifacts=(
            BackendArtifact(
                role="inference" if inference else "training_config",
                relative_path=f"backend-{index}/artifact.bin",
                artifact_hash=_HASH if inference else None,
                size_bytes=2048 if inference else 0,
                available=inference,
                license_id="Apache-2.0",
            ),
        ),
        training_entrypoint_available=True,
    )


def test_backend_catalog_exposes_honest_readiness_stages() -> None:
    source_only = _readiness(0, inference=False)
    artifact_ready = _readiness(1, inference=True)
    catalog = AthleteBackendCatalog(
        backends=(
            source_only,
            artifact_ready,
            _readiness(2, inference=True),
            _readiness(3, inference=False),
        )
    )

    assert source_only.stage is BackendReadinessStage.SOURCE_ONLY
    assert artifact_ready.stage is BackendReadinessStage.ARTIFACT_READY_ENVIRONMENT_PENDING
    assert catalog.catalog_hash.startswith("sha256:")


def test_backend_cannot_claim_physics_before_environment_and_atlas() -> None:
    with pytest.raises(ValueError, match="cannot pass physics"):
        replace(_readiness(0, inference=True), physical_exam_passed=True)
    with pytest.raises(ValueError, match="four unique"):
        AthleteBackendCatalog(backends=(_readiness(0, inference=True),) * 4)
