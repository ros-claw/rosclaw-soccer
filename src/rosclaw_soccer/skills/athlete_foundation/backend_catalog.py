"""Local readiness audit for interchangeable athlete learner backends."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from rosclaw.continual.learner_backend import LearnerBackendContract, LearnerCapability

from rosclaw_soccer.sim.contracts import hash_json


class BackendReadinessStage(StrEnum):
    SOURCE_ONLY = "source_only"
    ARTIFACT_READY_ENVIRONMENT_PENDING = "artifact_ready_environment_pending"
    ENVIRONMENT_READY_ATLAS_PENDING = "environment_ready_atlas_pending"
    ATLAS_ADAPTED_PHYSICS_PENDING = "atlas_adapted_physics_pending"
    PHYSICS_QUALIFIED = "physics_qualified"


@dataclass(frozen=True)
class BackendArtifact:
    role: str
    relative_path: str
    artifact_hash: str | None
    size_bytes: int
    available: bool
    license_id: str
    schema_version: str = "rosclaw_soccer.backend_artifact.v1"

    def __post_init__(self) -> None:
        path = Path(self.relative_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("backend artifact path must be relative")
        if self.available != (self.artifact_hash is not None and self.size_bytes > 0):
            raise ValueError("backend artifact availability is inconsistent")


@dataclass(frozen=True)
class BackendReadiness:
    backend: LearnerBackendContract
    source_checkout_hash: str
    artifacts: tuple[BackendArtifact, ...]
    training_entrypoint_available: bool
    environment_smoke_passed: bool = False
    common_atlas_adapter_available: bool = False
    physical_exam_passed: bool = False
    activation_ceiling: str = "SIM_ONLY"
    hardware_command_sent: bool = False
    schema_version: str = "rosclaw_soccer.backend_readiness.v1"

    def __post_init__(self) -> None:
        if not self.source_checkout_hash.startswith("sha256:"):
            raise ValueError("backend source checkout requires a content hash")
        if not self.artifacts or len({item.role for item in self.artifacts}) != len(
            self.artifacts
        ):
            raise ValueError("backend artifacts must be non-empty and role-unique")
        if self.physical_exam_passed and not (
            self.environment_smoke_passed and self.common_atlas_adapter_available
        ):
            raise ValueError("backend cannot pass physics before environment and atlas adaptation")
        if self.activation_ceiling != "SIM_ONLY" or self.hardware_command_sent:
            raise ValueError("learner backend readiness must remain SIM_ONLY")

    @property
    def inference_artifact_available(self) -> bool:
        return any(item.role == "inference" and item.available for item in self.artifacts)

    @property
    def stage(self) -> BackendReadinessStage:
        if self.physical_exam_passed:
            return BackendReadinessStage.PHYSICS_QUALIFIED
        if self.common_atlas_adapter_available:
            return BackendReadinessStage.ATLAS_ADAPTED_PHYSICS_PENDING
        if self.environment_smoke_passed:
            return BackendReadinessStage.ENVIRONMENT_READY_ATLAS_PENDING
        if self.inference_artifact_available:
            return BackendReadinessStage.ARTIFACT_READY_ENVIRONMENT_PENDING
        return BackendReadinessStage.SOURCE_ONLY

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "backend": self.backend.to_dict(),
            "backend_contract_hash": self.backend.contract_hash,
            "source_checkout_hash": self.source_checkout_hash,
            "artifacts": [asdict(item) for item in self.artifacts],
            "training_entrypoint_available": self.training_entrypoint_available,
            "inference_artifact_available": self.inference_artifact_available,
            "environment_smoke_passed": self.environment_smoke_passed,
            "common_atlas_adapter_available": self.common_atlas_adapter_available,
            "physical_exam_passed": self.physical_exam_passed,
            "stage": self.stage.value,
            "activation_ceiling": self.activation_ceiling,
            "hardware_command_sent": self.hardware_command_sent,
        }


@dataclass(frozen=True)
class AthleteBackendCatalog:
    backends: tuple[BackendReadiness, ...]
    activation_ceiling: str = "SIM_ONLY"
    hardware_command_sent: bool = False
    schema_version: str = "rosclaw_soccer.athlete_backend_catalog.v1"

    def __post_init__(self) -> None:
        identifiers = tuple(item.backend.backend_id for item in self.backends)
        if len(self.backends) != 4 or len(set(identifiers)) != 4:
            raise ValueError("Foundation Shootout requires four unique backends")
        if self.activation_ceiling != "SIM_ONLY" or self.hardware_command_sent:
            raise ValueError("backend catalog must remain SIM_ONLY")

    @property
    def catalog_hash(self) -> str:
        return str(hash_json(self.to_dict(include_hash=False)))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": self.schema_version,
            "backends": [item.to_dict() for item in self.backends],
            "activation_ceiling": self.activation_ceiling,
            "hardware_command_sent": self.hardware_command_sent,
        }
        if include_hash:
            value["catalog_hash"] = self.catalog_hash
        return value


def build_local_backend_catalog(*, repos_root: Path, datasets_root: Path) -> AthleteBackendCatalog:
    repos = repos_root.expanduser().resolve()
    datasets = datasets_root.expanduser().resolve()
    specifications = (
        (
            LearnerBackendContract(
                backend_id="sonic",
                backend_version="1.1",
                source_url="https://github.com/NVlabs/GR00T-WholeBodyControl",
                source_commit="32c8260e54118b1f92b1fdeb9395d70d828e51a5",
                license_id="Apache-2.0",
                capabilities=(
                    LearnerCapability.MOTION_TRACKING,
                    LearnerCapability.SPECIALIST_TRAINING,
                    LearnerCapability.GENERALIST_DISTILLATION,
                    LearnerCapability.MULTI_GPU,
                    LearnerCapability.SIM2SIM,
                ),
                supported_body_ids=("unitree.g1.29dof",),
                training_available=True,
                inference_available=True,
            ),
            repos / "GR00T-WholeBodyControl",
            repos / "GR00T-WholeBodyControl" / "gear_sonic" / "train_agent_trl.py",
            (
                (
                    "inference",
                    datasets / "GEAR-SONIC" / "sonic_v1_1" / "model_decoder.onnx",
                    "NVIDIA-Open-Model",
                ),
                (
                    "encoder",
                    datasets / "GEAR-SONIC" / "sonic_v1_1" / "model_encoder.onnx",
                    "NVIDIA-Open-Model",
                ),
            ),
        ),
        (
            LearnerBackendContract(
                backend_id="humanoid-gpt",
                backend_version="2026.07",
                source_url="https://github.com/GalaxyGeneralRobotics/Humanoid-GPT",
                source_commit="d5a8a5f4809760cafed6a75b97494ecf4b650408",
                license_id="Apache-2.0",
                capabilities=(LearnerCapability.MOTION_TRACKING, LearnerCapability.SIM2SIM),
                supported_body_ids=("unitree.g1.29dof",),
                training_available=False,
                inference_available=True,
            ),
            repos / "Humanoid-GPT",
            None,
            (
                (
                    "inference",
                    repos / "Humanoid-GPT" / "storage" / "ckpts" / "pns_wo_priv216.onnx",
                    "Apache-2.0",
                ),
                (
                    "sample_motion",
                    repos
                    / "Humanoid-GPT"
                    / "storage"
                    / "test"
                    / "human_walking_50Hz_29dof.npz",
                    "Apache-2.0",
                ),
            ),
        ),
        (
            LearnerBackendContract(
                backend_id="opentrack",
                backend_version="2026.05",
                source_url="https://github.com/GalaxyGeneralRobotics/OpenTrack",
                source_commit="cb9b751993a2483e5d1805a2565ddbfe950c04c9",
                license_id="Apache-2.0",
                capabilities=(
                    LearnerCapability.MOTION_TRACKING,
                    LearnerCapability.SPECIALIST_TRAINING,
                    LearnerCapability.GENERALIST_DISTILLATION,
                    LearnerCapability.PERSONAL_ADAPTER,
                    LearnerCapability.MULTI_GPU,
                ),
                supported_body_ids=("unitree.g1.29dof",),
                training_available=True,
                inference_available=True,
            ),
            repos / "OpenTrack",
            repos / "OpenTrack" / "track_mj" / "learning" / "train" / "train_ppo_track.py",
            (
                (
                    "training_config",
                    repos
                    / "OpenTrack"
                    / "storage"
                    / "training_configs"
                    / "dagger"
                    / "demo_v2.json",
                    "Apache-2.0",
                ),
            ),
        ),
        (
            LearnerBackendContract(
                backend_id="protomotions3",
                backend_version="3.0",
                source_url="https://github.com/NVlabs/ProtoMotions",
                source_commit="a6df301d312dc58ac40a4d994f4f1064728d854c",
                license_id="Apache-2.0",
                capabilities=(
                    LearnerCapability.MOTION_TRACKING,
                    LearnerCapability.ADVERSARIAL_MOTION_PRIOR,
                    LearnerCapability.SPECIALIST_TRAINING,
                    LearnerCapability.GENERALIST_DISTILLATION,
                    LearnerCapability.MULTI_GPU,
                    LearnerCapability.SIM2SIM,
                ),
                supported_body_ids=("unitree.g1.29dof",),
                training_available=True,
                inference_available=True,
            ),
            repos / "ProtoMotions",
            repos / "ProtoMotions" / "protomotions" / "train_agent.py",
            (
                (
                    "inference",
                    repos
                    / "ProtoMotions"
                    / "data"
                    / "pretrained_models"
                    / "motion_tracker"
                    / "g1-bones-deploy"
                    / "compiled_models"
                    / "unified_pipeline.onnx",
                    "Apache-2.0",
                ),
                (
                    "sample_motion",
                    repos
                    / "ProtoMotions"
                    / "data"
                    / "motion_for_trackers"
                    / "g1_random_subset_tiny.pt",
                    "Apache-2.0",
                ),
            ),
        ),
    )
    readiness: list[BackendReadiness] = []
    for backend, checkout, training_entrypoint, artifacts in specifications:
        if _git_commit(checkout) != backend.source_commit:
            raise ValueError(f"backend checkout is not pinned: {backend.backend_id}")
        artifact_records = tuple(
            _artifact(role=role, path=path, checkout=checkout, license_id=license_id)
            for role, path, license_id in artifacts
        )
        readiness.append(
            BackendReadiness(
                backend=backend,
                source_checkout_hash=str(
                    hash_json(
                        {
                            "source_url": backend.source_url,
                            "source_commit": backend.source_commit,
                        }
                    )
                ),
                artifacts=artifact_records,
                training_entrypoint_available=(
                    training_entrypoint is not None and training_entrypoint.is_file()
                ),
            )
        )
    return AthleteBackendCatalog(backends=tuple(readiness))


def write_backend_catalog(catalog: AthleteBackendCatalog, output_path: Path) -> None:
    target = output_path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(catalog.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, target)


def _artifact(*, role: str, path: Path, checkout: Path, license_id: str) -> BackendArtifact:
    available = path.is_file() and path.stat().st_size > 1024
    try:
        relative_path = str(path.resolve().relative_to(checkout.parent.resolve()))
    except ValueError:
        relative_path = f"external-dataset/{path.name}"
    return BackendArtifact(
        role=role,
        relative_path=relative_path,
        artifact_hash=(
            "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() if available else None
        ),
        size_bytes=path.stat().st_size if available else 0,
        available=available,
        license_id=license_id,
    )


def _git_commit(checkout: Path) -> str:
    if not checkout.is_dir():
        raise ValueError(f"backend checkout is unavailable: {checkout.name}")
    return subprocess.run(
        ("git", "-C", str(checkout), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


__all__ = [
    "AthleteBackendCatalog",
    "BackendArtifact",
    "BackendReadiness",
    "BackendReadinessStage",
    "build_local_backend_catalog",
    "write_backend_catalog",
]
