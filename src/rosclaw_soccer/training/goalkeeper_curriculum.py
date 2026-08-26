"""Measured curriculum scheduler for large-scale goalkeeper learning."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from rosclaw_soccer.sim.contracts import hash_json


@dataclass(frozen=True)
class GoalkeeperCurriculumStage:
    """One immutable stage of the physics curriculum."""

    name: str
    minimum_first_save_rate: float
    minimum_recovery_rate: float
    minimum_second_save_rate: float
    minimum_updates: int
    maximum_updates: int
    shot_deadline_range_sec: tuple[float, float]
    target_lateral_range_m: tuple[float, float]
    target_height_range_m: tuple[float, float]
    second_shot_probability: float
    motion_prior_weight: float
    dynamics_randomization_scale: float
    schema_version: str = "rosclaw_soccer.goalkeeper_curriculum_stage.v1"

    def __post_init__(self) -> None:
        if not self.name.strip() or not 1 <= self.minimum_updates <= self.maximum_updates:
            raise ValueError("goalkeeper curriculum stage identity or update range is invalid")
        rates = (
            self.minimum_first_save_rate,
            self.minimum_recovery_rate,
            self.minimum_second_save_rate,
            self.second_shot_probability,
            self.motion_prior_weight,
            self.dynamics_randomization_scale,
        )
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in rates):
            raise ValueError("goalkeeper curriculum rates must be in [0, 1]")
        for lower, upper in (
            self.shot_deadline_range_sec,
            self.target_lateral_range_m,
            self.target_height_range_m,
        ):
            if not all(math.isfinite(value) for value in (lower, upper)) or lower >= upper:
                raise ValueError("goalkeeper curriculum ranges must be finite and increasing")
        if self.shot_deadline_range_sec[0] < 0.35:
            raise ValueError("goalkeeper curriculum deadline violates the safety floor")

    @property
    def stage_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass(frozen=True)
class GoalkeeperCurriculumEvidence:
    """Held-out evidence used to decide whether a stage can advance."""

    completed_updates: int
    first_save_rate: float
    recovery_rate: float
    second_save_rate: float
    unsafe_rate: float
    strict_replay: bool

    def __post_init__(self) -> None:
        if self.completed_updates < 0:
            raise ValueError("goalkeeper completed updates cannot be negative")
        for value in (
            self.first_save_rate,
            self.recovery_rate,
            self.second_save_rate,
            self.unsafe_rate,
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("goalkeeper curriculum evidence rates must be in [0, 1]")


@dataclass(frozen=True)
class GoalkeeperCurriculumDecision:
    stage_name: str
    stage_hash: str
    action: str
    reasons: tuple[str, ...]
    schema_version: str = "rosclaw_soccer.goalkeeper_curriculum_decision.v1"

    def __post_init__(self) -> None:
        if self.action not in {"HOLD", "ADVANCE", "REJECT_UNSAFE", "EXHAUSTED"}:
            raise ValueError("goalkeeper curriculum decision action is invalid")
        if not self.stage_hash.startswith("sha256:") or not self.reasons:
            raise ValueError("goalkeeper curriculum decision lacks evidence identity")


def default_goalkeeper_curriculum() -> tuple[GoalkeeperCurriculumStage, ...]:
    """Return an easy-to-hard curriculum that ends with sequential saves."""

    return (
        GoalkeeperCurriculumStage(
            name="stand_reach",
            minimum_first_save_rate=0.70,
            minimum_recovery_rate=0.98,
            minimum_second_save_rate=0.0,
            minimum_updates=500,
            maximum_updates=4_000,
            shot_deadline_range_sec=(0.90, 1.20),
            target_lateral_range_m=(-0.45, 0.45),
            target_height_range_m=(0.45, 1.35),
            second_shot_probability=0.0,
            motion_prior_weight=0.40,
            dynamics_randomization_scale=0.05,
        ),
        GoalkeeperCurriculumStage(
            name="shuffle_save",
            minimum_first_save_rate=0.65,
            minimum_recovery_rate=0.95,
            minimum_second_save_rate=0.0,
            minimum_updates=1_500,
            maximum_updates=8_000,
            shot_deadline_range_sec=(0.65, 1.00),
            target_lateral_range_m=(-0.95, 0.95),
            target_height_range_m=(0.30, 1.60),
            second_shot_probability=0.0,
            motion_prior_weight=0.30,
            dynamics_randomization_scale=0.15,
        ),
        GoalkeeperCurriculumStage(
            name="dive_land_recover",
            minimum_first_save_rate=0.58,
            minimum_recovery_rate=0.85,
            minimum_second_save_rate=0.0,
            minimum_updates=3_000,
            maximum_updates=12_000,
            shot_deadline_range_sec=(0.50, 0.85),
            target_lateral_range_m=(-1.25, 1.25),
            target_height_range_m=(0.20, 1.75),
            second_shot_probability=0.0,
            motion_prior_weight=0.25,
            dynamics_randomization_scale=0.30,
        ),
        GoalkeeperCurriculumStage(
            name="sequential_saves",
            minimum_first_save_rate=0.52,
            minimum_recovery_rate=0.80,
            minimum_second_save_rate=0.35,
            minimum_updates=5_000,
            maximum_updates=20_000,
            shot_deadline_range_sec=(0.40, 0.80),
            target_lateral_range_m=(-1.45, 1.45),
            target_height_range_m=(0.15, 1.85),
            second_shot_probability=0.70,
            motion_prior_weight=0.20,
            dynamics_randomization_scale=0.50,
        ),
    )


def decide_curriculum_stage(
    stage: GoalkeeperCurriculumStage,
    evidence: GoalkeeperCurriculumEvidence,
) -> GoalkeeperCurriculumDecision:
    """Advance only from held-out, replayable, zero-unsafe evidence."""

    if evidence.unsafe_rate > 0.0:
        return GoalkeeperCurriculumDecision(
            stage_name=stage.name,
            stage_hash=stage.stage_hash,
            action="REJECT_UNSAFE",
            reasons=("nonzero_unsafe_rate",),
        )
    unmet: list[str] = []
    if evidence.completed_updates < stage.minimum_updates:
        unmet.append("minimum_updates_not_reached")
    if evidence.first_save_rate < stage.minimum_first_save_rate:
        unmet.append("first_save_rate_below_stage_floor")
    if evidence.recovery_rate < stage.minimum_recovery_rate:
        unmet.append("recovery_rate_below_stage_floor")
    if evidence.second_save_rate < stage.minimum_second_save_rate:
        unmet.append("second_save_rate_below_stage_floor")
    if not evidence.strict_replay:
        unmet.append("strict_replay_missing")
    if not unmet:
        return GoalkeeperCurriculumDecision(
            stage_name=stage.name,
            stage_hash=stage.stage_hash,
            action="ADVANCE",
            reasons=("all_stage_thresholds_met",),
        )
    action = "EXHAUSTED" if evidence.completed_updates >= stage.maximum_updates else "HOLD"
    return GoalkeeperCurriculumDecision(
        stage_name=stage.name,
        stage_hash=stage.stage_hash,
        action=action,
        reasons=tuple(unmet),
    )


def curriculum_manifest() -> dict[str, Any]:
    stages = default_goalkeeper_curriculum()
    return {
        "schema_version": "rosclaw_soccer.goalkeeper_curriculum_manifest.v1",
        "stages": [asdict(stage) | {"stage_hash": stage.stage_hash} for stage in stages],
        "curriculum_hash": hash_json([stage.stage_hash for stage in stages]),
        "promotion_authority": False,
        "activation_ceiling": "SIM_ONLY",
    }


__all__ = [
    "GoalkeeperCurriculumDecision",
    "GoalkeeperCurriculumEvidence",
    "GoalkeeperCurriculumStage",
    "curriculum_manifest",
    "decide_curriculum_stage",
    "default_goalkeeper_curriculum",
]
