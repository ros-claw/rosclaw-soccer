"""Whole-body motion foundations used by Soccer training only."""

from rosclaw_soccer.skills.athlete_foundation.backend_adapters import (
    export_humanoid_gpt_reference,
    export_opentrack_reference,
)
from rosclaw_soccer.skills.athlete_foundation.backend_catalog import (
    AthleteBackendCatalog,
    BackendArtifact,
    BackendReadiness,
    BackendReadinessStage,
    build_local_backend_catalog,
    write_backend_catalog,
)
from rosclaw_soccer.skills.athlete_foundation.foundation_shootout import (
    FoundationEvaluation,
    FoundationMetrics,
    FoundationResultStatus,
    FoundationShootout,
    FoundationThresholds,
    write_foundation_shootout,
)
from rosclaw_soccer.skills.athlete_foundation.full_body_goalkeeper_motion import (
    FullBodyGoalkeeperMotionLibrary,
    FullBodyGoalkeeperMotionManifest,
    FullBodyMotionFrame,
    build_full_body_goalkeeper_motion_bundle,
    load_full_body_goalkeeper_motion_library,
)
from rosclaw_soccer.skills.athlete_foundation.humanoid_gpt_evidence import (
    HumanoidGPTTrackingMetrics,
    HumanoidGPTTrackingResult,
    seal_humanoid_gpt_tracking_evidence,
)
from rosclaw_soccer.skills.athlete_foundation.motion_atlas import (
    MotionAtlasCategory,
    MotionAtlasManifest,
    MotionAtlasRecord,
    build_s33_motion_atlas,
    write_s33_motion_atlas,
)

__all__ = [
    "AthleteBackendCatalog",
    "BackendArtifact",
    "BackendReadiness",
    "BackendReadinessStage",
    "FullBodyGoalkeeperMotionLibrary",
    "FullBodyGoalkeeperMotionManifest",
    "FullBodyMotionFrame",
    "FoundationEvaluation",
    "FoundationMetrics",
    "FoundationResultStatus",
    "FoundationShootout",
    "FoundationThresholds",
    "HumanoidGPTTrackingMetrics",
    "HumanoidGPTTrackingResult",
    "MotionAtlasCategory",
    "MotionAtlasManifest",
    "MotionAtlasRecord",
    "build_full_body_goalkeeper_motion_bundle",
    "build_local_backend_catalog",
    "build_s33_motion_atlas",
    "export_humanoid_gpt_reference",
    "export_opentrack_reference",
    "load_full_body_goalkeeper_motion_library",
    "seal_humanoid_gpt_tracking_evidence",
    "write_s33_motion_atlas",
    "write_backend_catalog",
    "write_foundation_shootout",
]
