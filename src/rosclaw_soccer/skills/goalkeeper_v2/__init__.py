"""Causal observation and evaluation contracts for Goalkeeper V2."""

from rosclaw_soccer.skills.goalkeeper_v2.coverage_time import (
    GoalkeeperCoveragePoint,
    GoalkeeperCoverageTimeReport,
    GoalkeeperCoverageTrial,
    aggregate_coverage_time,
)
from rosclaw_soccer.skills.goalkeeper_v2.external_motion import (
    ExternalGoalkeeperMotionDecoder,
    ExternalGoalkeeperMotionManifest,
    build_external_goalkeeper_motion_bundle,
    load_external_goalkeeper_motion_decoder,
)
from rosclaw_soccer.skills.goalkeeper_v2.external_reference import (
    HumanoidGoalkeeperReferenceAction,
    HumanoidGoalkeeperReferenceManifest,
    NumpyHumanoidGoalkeeperReferenceActor,
    build_humanoid_goalkeeper_reference_bundle,
    load_humanoid_goalkeeper_reference_actor,
)
from rosclaw_soccer.skills.goalkeeper_v2.motion_library import (
    GoalkeeperMotionClip,
    GoalkeeperMotionFamily,
    GoalkeeperMotionLibrary,
    build_motiondecode_goalkeeper_library,
    load_goalkeeper_motion_library,
    load_motion_clip_frames,
)
from rosclaw_soccer.skills.goalkeeper_v2.observations import (
    GoalkeeperActorObservation,
    GoalkeeperActorObserver,
    GoalkeeperObservationSpec,
)
from rosclaw_soccer.skills.goalkeeper_v2.policy import (
    GoalkeeperActorAction,
    GoalkeeperActorArtifact,
    GoalkeeperDenseLayer,
    NumpyGoalkeeperActor,
    load_goalkeeper_actor_artifact,
    save_goalkeeper_actor_artifact,
)
from rosclaw_soccer.skills.goalkeeper_v2.promotion import (
    GoalkeeperGateMetric,
    GoalkeeperPromotionDecision,
    GoalkeeperPromotionThresholds,
    evaluate_goalkeeper_promotion,
)

__all__ = [
    "GoalkeeperActorObservation",
    "GoalkeeperActorObserver",
    "GoalkeeperActorAction",
    "GoalkeeperActorArtifact",
    "GoalkeeperDenseLayer",
    "GoalkeeperObservationSpec",
    "GoalkeeperCoveragePoint",
    "GoalkeeperCoverageTimeReport",
    "GoalkeeperCoverageTrial",
    "GoalkeeperMotionClip",
    "GoalkeeperMotionFamily",
    "GoalkeeperMotionLibrary",
    "ExternalGoalkeeperMotionDecoder",
    "ExternalGoalkeeperMotionManifest",
    "HumanoidGoalkeeperReferenceAction",
    "HumanoidGoalkeeperReferenceManifest",
    "GoalkeeperGateMetric",
    "GoalkeeperPromotionDecision",
    "GoalkeeperPromotionThresholds",
    "NumpyGoalkeeperActor",
    "NumpyHumanoidGoalkeeperReferenceActor",
    "aggregate_coverage_time",
    "build_motiondecode_goalkeeper_library",
    "build_external_goalkeeper_motion_bundle",
    "build_humanoid_goalkeeper_reference_bundle",
    "evaluate_goalkeeper_promotion",
    "load_goalkeeper_motion_library",
    "load_goalkeeper_actor_artifact",
    "load_motion_clip_frames",
    "load_external_goalkeeper_motion_decoder",
    "load_humanoid_goalkeeper_reference_actor",
    "save_goalkeeper_actor_artifact",
]
