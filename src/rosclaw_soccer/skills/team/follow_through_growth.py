"""Visible, stable follow-through growth from a semantic motion teacher."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rosclaw_soccer.growth.football_motion_prior import load_g1_football_motion_prior
from rosclaw_soccer.growth.mosaic_agility_prior import load_g1_mosaic_agility_prior
from rosclaw_soccer.providers.g1.asset_qualification import trajectory_digest
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.skills.team.agility_growth import (
    G1AgilityCandidate,
    G1FollowThroughAgilityMetrics,
    measure_g1_follow_through_agility,
)
from rosclaw_soccer.skills.team.imitation_learning import (
    G1MotionNaturalnessMetrics,
    measure_g1_motion_naturalness,
)
from rosclaw_soccer.skills.team.shared_world import G1SharedWorldResult, simulate_shared_world

_ARM_ONLY_SCALES = (0.0,) * 15 + (1.0,) * 14


@dataclass(frozen=True)
class G1FollowThroughCandidate:
    position_blend: float
    center_policy_frame: int
    schema_version: str = "rosclaw_soccer.g1_follow_through_candidate.v1"

    def __post_init__(self) -> None:
        if not math.isfinite(self.position_blend) or not 0.0 < self.position_blend <= 0.50:
            raise ValueError("follow-through position blend must be in (0, 0.50]")
        if not 270 <= self.center_policy_frame <= 290:
            raise ValueError("follow-through center frame must be in [270, 290]")

    @property
    def candidate_hash(self) -> str:
        return str(hash_json(asdict(self)))

    def simulation_overrides(self, *, mosaic_prior_path: Path) -> dict[str, Any]:
        return {
            "shooter_agility_prior_path": mosaic_prior_path,
            "shooter_agility_prior_position_blend": self.position_blend,
            "shooter_agility_prior_velocity_blend": 0.0,
            "shooter_agility_prior_contact_policy_frame": self.center_policy_frame,
            "shooter_agility_prior_joint_scales": _ARM_ONLY_SCALES,
        }


@dataclass(frozen=True)
class G1FollowThroughTrial:
    candidate: G1FollowThroughCandidate
    candidate_hash: str
    trajectory_digest: str
    parent_agility: G1FollowThroughAgilityMetrics
    agility: G1FollowThroughAgilityMetrics
    naturalness: G1MotionNaturalnessMetrics
    eligible: bool
    score: float
    reasons: tuple[str, ...]
    schema_version: str = "rosclaw_soccer.g1_follow_through_trial.v1"


@dataclass(frozen=True)
class G1FollowThroughSearchResult:
    parent_trajectory_digest: str
    parent_result: G1SharedWorldResult
    parent_naturalness: G1MotionNaturalnessMetrics
    trials: tuple[G1FollowThroughTrial, ...]
    selected_trial: G1FollowThroughTrial | None
    selected_candidate: G1FollowThroughCandidate | None
    neighborhood_eligible_fraction: float
    passed: bool
    reasons: tuple[str, ...]
    activation_ceiling: str = "SIM_ONLY"
    hardware_command_sent: bool = False
    schema_version: str = "rosclaw_soccer.g1_follow_through_search_result.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "parent_result": self.parent_result.to_dict(),
            "parent_naturalness": asdict(self.parent_naturalness),
            "trials": [asdict(trial) for trial in self.trials],
            "selected_trial": (
                None if self.selected_trial is None else asdict(self.selected_trial)
            ),
            "selected_candidate": (
                None if self.selected_candidate is None else asdict(self.selected_candidate)
            ),
        }


def default_g1_follow_through_candidates() -> tuple[G1FollowThroughCandidate, ...]:
    return tuple(
        G1FollowThroughCandidate(blend, frame)
        for frame in (288, 289, 290)
        for blend in (0.30, 0.40, 0.50)
    )


def search_g1_follow_through_candidate(
    *,
    asset_root: Path,
    motion_prior_path: Path,
    contact_prior_path: Path,
    mosaic_prior_path: Path,
    simulation_kwargs: dict[str, Any],
    candidates: tuple[G1FollowThroughCandidate, ...] | None = None,
) -> G1FollowThroughSearchResult:
    """Search one nearby arm-only basin against a retained S7 parent."""

    motion = load_g1_football_motion_prior(motion_prior_path)
    contact = load_g1_football_motion_prior(contact_prior_path)
    mosaic = load_g1_mosaic_agility_prior(mosaic_prior_path)
    if contact.source_dataset != "OmniContact":
        raise ValueError("follow-through growth requires the qualified contact prior")
    if mosaic.teacher_skill_id != "SE63":
        raise ValueError("follow-through growth requires the preselected SE63 soccer teacher")
    active = candidates or default_g1_follow_through_candidates()
    if not 3 <= len(active) <= 25 or len({item.candidate_hash for item in active}) != len(active):
        raise ValueError("follow-through candidates must contain 3-25 unique values")
    base = G1AgilityCandidate(1.14, 1.15)
    base_kwargs = {
        **simulation_kwargs,
        **base.simulation_overrides(
            motion_prior_path=motion_prior_path,
            contact_prior_path=contact_prior_path,
        ),
    }
    parent_result, parent_trajectory = simulate_shared_world(asset_root, **base_kwargs)
    parent_naturalness = measure_g1_motion_naturalness(
        trajectory=parent_trajectory,
        result=parent_result,
        prior=motion,
        contact_policy_frame=253,
    )
    trials: list[G1FollowThroughTrial] = []
    for candidate in active:
        result, trajectory = simulate_shared_world(
            asset_root,
            **{
                **base_kwargs,
                **candidate.simulation_overrides(mosaic_prior_path=mosaic_prior_path),
            },
        )
        parent_agility = measure_g1_follow_through_agility(
            parent_trajectory,
            center_policy_frame=candidate.center_policy_frame,
        )
        agility = measure_g1_follow_through_agility(
            trajectory,
            center_policy_frame=candidate.center_policy_frame,
        )
        naturalness = measure_g1_motion_naturalness(
            trajectory=trajectory,
            result=result,
            prior=motion,
            contact_policy_frame=253,
        )
        trials.append(
            evaluate_g1_follow_through_trial(
                candidate=candidate,
                result=result,
                trajectory=trajectory,
                parent_naturalness=parent_naturalness,
                parent_agility=parent_agility,
                naturalness=naturalness,
                agility=agility,
            )
        )
    eligible = [trial for trial in trials if trial.eligible]
    selected = max(eligible, key=lambda trial: trial.score) if eligible else None
    fraction = len(eligible) / len(trials)
    passed = bool(selected is not None and fraction >= 0.80)
    return G1FollowThroughSearchResult(
        parent_trajectory_digest=trajectory_digest(parent_trajectory),
        parent_result=parent_result,
        parent_naturalness=parent_naturalness,
        trials=tuple(trials),
        selected_trial=selected,
        selected_candidate=None if selected is None else selected.candidate,
        neighborhood_eligible_fraction=fraction,
        passed=passed,
        reasons=() if passed else ("no_visible_safe_follow_through_basin",),
    )


def evaluate_g1_follow_through_trial(
    *,
    candidate: G1FollowThroughCandidate,
    result: G1SharedWorldResult,
    trajectory: dict[str, Any],
    parent_naturalness: G1MotionNaturalnessMetrics,
    parent_agility: G1FollowThroughAgilityMetrics,
    naturalness: G1MotionNaturalnessMetrics,
    agility: G1FollowThroughAgilityMetrics,
) -> G1FollowThroughTrial:
    """Require visible plasticity and retention in the same gate."""

    gates = {
        "physical_rollout_passed": result.passed,
        "accuracy_3cm": naturalness.target_error_m <= 0.03,
        "support_slip_6cm": naturalness.post_contact_support_slip_m <= 0.06,
        "backward_speed_1cm_s": naturalness.post_contact_peak_backward_velocity_mps <= 0.01,
        "roll_bounded": naturalness.torso_roll_peak_rad <= 0.24,
        "contact_acceleration_not_forgotten": (
            naturalness.contact_joint_acceleration_rms_rad_s2
            <= 1.02 * parent_naturalness.contact_joint_acceleration_rms_rad_s2
        ),
        "post_acceleration_not_forgotten": (
            naturalness.post_contact_joint_acceleration_rms_rad_s2
            <= 1.02 * parent_naturalness.post_contact_joint_acceleration_rms_rad_s2
        ),
        "root_acceleration_not_forgotten": (
            naturalness.post_contact_root_acceleration_rms_m_s2
            <= 1.02 * parent_naturalness.post_contact_root_acceleration_rms_m_s2
        ),
        "visible_arm_excursion_gain": (
            agility.arm_excursion_rms_rad >= 1.08 * parent_agility.arm_excursion_rms_rad
        ),
        "visible_motion_energy_gain": (
            agility.upper_body_motion_energy >= 1.10 * parent_agility.upper_body_motion_energy
        ),
        "pose_teacher_reached_pd": agility.teacher_position_l1_rad > 0.0,
    }
    reasons = tuple(name for name, value in gates.items() if not value)
    excursion_gain = _gain(parent_agility.arm_excursion_rms_rad, agility.arm_excursion_rms_rad)
    energy_gain = _gain(parent_agility.upper_body_motion_energy, agility.upper_body_motion_energy)
    score = (
        2.0 * excursion_gain
        + energy_gain
        + _gain(
            parent_naturalness.post_contact_peak_backward_velocity_mps,
            naturalness.post_contact_peak_backward_velocity_mps,
        )
        - _regression(
            parent_naturalness.post_contact_support_slip_m,
            naturalness.post_contact_support_slip_m,
        )
        - _regression(
            parent_naturalness.post_contact_joint_acceleration_rms_rad_s2,
            naturalness.post_contact_joint_acceleration_rms_rad_s2,
        )
    )
    return G1FollowThroughTrial(
        candidate=candidate,
        candidate_hash=candidate.candidate_hash,
        trajectory_digest=trajectory_digest(trajectory),
        parent_agility=parent_agility,
        agility=agility,
        naturalness=naturalness,
        eligible=not reasons,
        score=score,
        reasons=reasons,
    )


def _gain(parent: float, candidate: float) -> float:
    return (candidate - parent) / max(parent, 1e-9)


def _regression(parent: float, candidate: float) -> float:
    return max(0.0, candidate - parent) / max(parent, 1e-9)


__all__ = [
    "G1FollowThroughCandidate",
    "G1FollowThroughSearchResult",
    "G1FollowThroughTrial",
    "default_g1_follow_through_candidates",
    "evaluate_g1_follow_through_trial",
    "search_g1_follow_through_candidate",
]
