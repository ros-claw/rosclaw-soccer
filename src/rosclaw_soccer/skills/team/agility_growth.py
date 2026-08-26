"""Stability-plasticity gates for joint-group imitation agility.

The football world is only the current proving ground.  The reusable ability
implemented here is a bounded, per-joint plasticity budget for position and
velocity teachers plus a counterfactual gate against a retained parent.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.growth.football_motion_prior import load_g1_football_motion_prior
from rosclaw_soccer.providers.g1.asset_qualification import trajectory_digest
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.skills.team.imitation_learning import (
    G1MotionNaturalnessMetrics,
    measure_g1_motion_naturalness,
)
from rosclaw_soccer.skills.team.shared_world import (
    G1SharedWorldResult,
    simulate_shared_world,
)

_CONTROL_DT_SEC = 0.02
_LEG = slice(0, 12)
_WAIST = slice(12, 15)
_ARMS = slice(15, 29)


@dataclass(frozen=True)
class G1AgilityCandidate:
    """Independent teacher authority for stable legs and expressive upper body."""

    waist_velocity_scale: float
    arm_velocity_scale: float
    foot_yaw_offset_rad: float = 0.09
    foot_pitch_offset_rad: float = 0.01
    contact_position_blend: float = 0.0025
    post_policy_forward_velocity_mps: float = 0.06
    schema_version: str = "rosclaw_soccer.g1_agility_candidate.v1"

    def __post_init__(self) -> None:
        values = (
            self.waist_velocity_scale,
            self.arm_velocity_scale,
            self.foot_yaw_offset_rad,
            self.foot_pitch_offset_rad,
            self.contact_position_blend,
            self.post_policy_forward_velocity_mps,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("agility candidate values must be finite")
        if not 0.0 <= self.waist_velocity_scale <= 2.0:
            raise ValueError("agility waist velocity scale must be in [0, 2]")
        if not 0.0 <= self.arm_velocity_scale <= 2.0:
            raise ValueError("agility arm velocity scale must be in [0, 2]")
        if not 0.0 < self.contact_position_blend <= 0.05:
            raise ValueError("agility contact blend must be in (0, 0.05]")
        if not 0.0 <= self.post_policy_forward_velocity_mps <= 0.15:
            raise ValueError("agility follow-through must be in [0, 0.15] m/s")

    @property
    def velocity_joint_scales(self) -> tuple[float, ...]:
        return (1.0,) * 12 + (self.waist_velocity_scale,) * 3 + (self.arm_velocity_scale,) * 14

    @property
    def candidate_hash(self) -> str:
        return str(hash_json(asdict(self)))

    def simulation_overrides(
        self,
        *,
        motion_prior_path: Path,
        contact_prior_path: Path,
    ) -> dict[str, Any]:
        return {
            "shooter_parameter_overrides": {
                "foot_yaw_offset": self.foot_yaw_offset_rad,
                "foot_pitch_offset": self.foot_pitch_offset_rad,
            },
            "shooter_motion_prior_path": motion_prior_path,
            "shooter_motion_prior_position_blend": 0.02,
            "shooter_motion_prior_velocity_blend": 0.02,
            "shooter_motion_prior_strike_leg_scale": 1.0,
            "shooter_motion_prior_joint_scales": (1.0,) * 29,
            "shooter_motion_prior_velocity_joint_scales": self.velocity_joint_scales,
            "shooter_motion_prior_contact_policy_frame": 253,
            "shooter_contact_prior_path": contact_prior_path,
            "shooter_contact_prior_position_blend": self.contact_position_blend,
            "shooter_contact_prior_contact_policy_frame": 253,
            "shooter_contact_prior_joint_scales": (1.0,) * 6,
            "shooter_post_policy_forward_velocity_mps": (self.post_policy_forward_velocity_mps),
        }


@dataclass(frozen=True)
class G1AgilityMetrics:
    waist_velocity_rms_rad_s: float
    arm_velocity_rms_rad_s: float
    waist_excursion_rms_rad: float
    arm_excursion_rms_rad: float
    upper_body_motion_energy: float
    schema_version: str = "rosclaw_soccer.g1_agility_metrics.v1"

    def __post_init__(self) -> None:
        values = tuple(value for name, value in asdict(self).items() if name != "schema_version")
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("agility metrics must be finite and non-negative")


@dataclass(frozen=True)
class G1FollowThroughAgilityMetrics:
    active_frame_count: int
    waist_velocity_rms_rad_s: float
    arm_velocity_rms_rad_s: float
    waist_excursion_rms_rad: float
    arm_excursion_rms_rad: float
    upper_body_motion_energy: float
    teacher_position_l1_rad: float
    teacher_velocity_l1_rad_s: float
    schema_version: str = "rosclaw_soccer.g1_follow_through_agility_metrics.v1"

    def __post_init__(self) -> None:
        if self.active_frame_count < 0:
            raise ValueError("follow-through active-frame count must be non-negative")
        values = tuple(value for name, value in asdict(self).items() if name != "schema_version")
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("follow-through agility metrics must be finite and non-negative")


@dataclass(frozen=True)
class G1AgilityTrial:
    candidate: G1AgilityCandidate
    candidate_hash: str
    trajectory_digest: str
    naturalness: G1MotionNaturalnessMetrics
    agility: G1AgilityMetrics
    safety_cost: float
    eligible: bool
    score: float
    reasons: tuple[str, ...]
    schema_version: str = "rosclaw_soccer.g1_agility_trial.v1"


@dataclass(frozen=True)
class G1AgilityNeighborhood:
    center_candidate_hash: str
    trials: tuple[G1AgilityTrial, ...]
    eligible_fraction: float
    passed: bool
    reasons: tuple[str, ...]
    schema_version: str = "rosclaw_soccer.g1_agility_neighborhood.v1"

    def __post_init__(self) -> None:
        if not self.center_candidate_hash.startswith("sha256:"):
            raise ValueError("agility neighborhood requires a candidate hash")
        if not 3 <= len(self.trials) <= 25:
            raise ValueError("agility neighborhood requires 3-25 trials")
        if not 0.0 <= self.eligible_fraction <= 1.0:
            raise ValueError("agility neighborhood fraction must be in [0, 1]")


@dataclass(frozen=True)
class G1AgilitySearchResult:
    parent_trajectory_digest: str
    parent_result: G1SharedWorldResult
    parent_naturalness: G1MotionNaturalnessMetrics
    parent_agility: G1AgilityMetrics
    trials: tuple[G1AgilityTrial, ...]
    selected_trial: G1AgilityTrial | None
    selected_candidate: G1AgilityCandidate | None
    neighborhood: G1AgilityNeighborhood | None
    passed: bool
    reasons: tuple[str, ...]
    activation_ceiling: str = "SIM_ONLY"
    hardware_command_sent: bool = False
    schema_version: str = "rosclaw_soccer.g1_agility_search_result.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "parent_result": self.parent_result.to_dict(),
            "parent_naturalness": asdict(self.parent_naturalness),
            "parent_agility": asdict(self.parent_agility),
            "trials": [asdict(trial) for trial in self.trials],
            "selected_trial": (
                None if self.selected_trial is None else asdict(self.selected_trial)
            ),
            "selected_candidate": (
                None if self.selected_candidate is None else asdict(self.selected_candidate)
            ),
            "neighborhood": (None if self.neighborhood is None else asdict(self.neighborhood)),
        }


def default_g1_agility_candidates() -> tuple[G1AgilityCandidate, ...]:
    """Return the bounded local basin discovered by coarse-to-fine search."""

    return (
        G1AgilityCandidate(1.13, 1.13),
        G1AgilityCandidate(1.14, 1.14),
        G1AgilityCandidate(1.14, 1.15),
        G1AgilityCandidate(1.15, 1.14),
        G1AgilityCandidate(1.15, 1.15),
        G1AgilityCandidate(1.16, 1.16),
    )


def search_g1_agility_candidate(
    *,
    asset_root: Path,
    motion_prior_path: Path,
    contact_prior_path: Path,
    simulation_kwargs: dict[str, Any],
    candidates: tuple[G1AgilityCandidate, ...] | None = None,
) -> G1AgilitySearchResult:
    """Search upper-body plasticity while retaining the qualified S6 parent."""

    motion_prior = load_g1_football_motion_prior(motion_prior_path)
    contact_prior = load_g1_football_motion_prior(contact_prior_path)
    if not motion_prior.whole_body_velocity_reference_rad_s:
        raise ValueError("agility search requires a velocity-aware whole-body prior")
    if contact_prior.source_dataset != "OmniContact":
        raise ValueError("agility search requires an OmniContact contact prior")
    active_candidates = candidates or default_g1_agility_candidates()
    if not 3 <= len(active_candidates) <= 25:
        raise ValueError("agility search requires 3-25 candidates")
    if len({candidate.candidate_hash for candidate in active_candidates}) != len(active_candidates):
        raise ValueError("agility candidates must be unique")

    parent = G1AgilityCandidate(1.0, 1.0)
    parent_kwargs = {
        **simulation_kwargs,
        **parent.simulation_overrides(
            motion_prior_path=motion_prior_path,
            contact_prior_path=contact_prior_path,
        ),
    }
    parent_result, parent_trajectory = simulate_shared_world(asset_root, **parent_kwargs)
    parent_naturalness = measure_g1_motion_naturalness(
        trajectory=parent_trajectory,
        result=parent_result,
        prior=motion_prior,
        contact_policy_frame=253,
    )
    parent_agility = measure_g1_agility(parent_trajectory)
    trials: list[G1AgilityTrial] = []
    for candidate in active_candidates:
        kwargs = {
            **simulation_kwargs,
            **candidate.simulation_overrides(
                motion_prior_path=motion_prior_path,
                contact_prior_path=contact_prior_path,
            ),
        }
        result, trajectory = simulate_shared_world(asset_root, **kwargs)
        naturalness = measure_g1_motion_naturalness(
            trajectory=trajectory,
            result=result,
            prior=motion_prior,
            contact_policy_frame=253,
        )
        agility = measure_g1_agility(trajectory)
        trials.append(
            evaluate_g1_agility_trial(
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
    selected_candidate = None if selected is None else selected.candidate
    neighborhood = (
        None
        if selected_candidate is None
        else evaluate_g1_agility_neighborhood(
            center=selected_candidate,
            trials=tuple(trials),
            required_fraction=0.60,
        )
    )
    passed = bool(selected is not None and neighborhood is not None and neighborhood.passed)
    reasons = () if passed else ("no_robust_safe_agility_candidate",)
    return G1AgilitySearchResult(
        parent_trajectory_digest=trajectory_digest(parent_trajectory),
        parent_result=parent_result,
        parent_naturalness=parent_naturalness,
        parent_agility=parent_agility,
        trials=tuple(trials),
        selected_trial=selected,
        selected_candidate=selected_candidate,
        neighborhood=neighborhood,
        passed=passed,
        reasons=reasons,
    )


def evaluate_g1_agility_neighborhood(
    *,
    center: G1AgilityCandidate,
    trials: tuple[G1AgilityTrial, ...],
    required_fraction: float = 0.60,
) -> G1AgilityNeighborhood:
    """Require a local basin of safe behavior, not one contact-lucky rollout."""

    if not 0.5 <= required_fraction <= 1.0 or not math.isfinite(required_fraction):
        raise ValueError("agility neighborhood required fraction must be in [0.5, 1]")
    if len({trial.candidate_hash for trial in trials}) != len(trials):
        raise ValueError("agility neighborhood trials must be unique")
    fraction = sum(trial.eligible for trial in trials) / len(trials)
    center_trial = next(
        (trial for trial in trials if trial.candidate_hash == center.candidate_hash),
        None,
    )
    reasons: list[str] = []
    if center_trial is None:
        reasons.append("center_candidate_missing")
    elif not center_trial.eligible:
        reasons.append("center_candidate_ineligible")
    if fraction < required_fraction:
        reasons.append("local_basin_too_narrow")
    return G1AgilityNeighborhood(
        center_candidate_hash=center.candidate_hash,
        trials=trials,
        eligible_fraction=fraction,
        passed=not reasons,
        reasons=tuple(reasons),
    )


def measure_g1_agility(
    trajectory: dict[str, NDArray[np.generic]], *, contact_policy_frame: int = 253
) -> G1AgilityMetrics:
    """Measure coordinated joint motion in a fixed, contact-centred window."""

    policy_frame = np.asarray(trajectory["shooter_policy_frame"], dtype=np.int64)
    position = np.asarray(trajectory["shooter_joint_position"], dtype=np.float64)
    velocity = np.asarray(trajectory["shooter_joint_velocity"], dtype=np.float64)
    if (
        position.shape != (len(policy_frame), 29)
        or velocity.shape != position.shape
        or not np.isfinite(position).all()
        or not np.isfinite(velocity).all()
    ):
        raise ValueError("agility trajectory is malformed or non-finite")
    mask = (policy_frame >= contact_policy_frame - 9) & (policy_frame <= contact_policy_frame + 9)
    if int(np.sum(mask)) < 10:
        raise ValueError("agility contact window is too short")
    excursion = np.ptp(position[mask], axis=0)
    waist_velocity = _rms(velocity[mask, _WAIST])
    arm_velocity = _rms(velocity[mask, _ARMS])
    return G1AgilityMetrics(
        waist_velocity_rms_rad_s=waist_velocity,
        arm_velocity_rms_rad_s=arm_velocity,
        waist_excursion_rms_rad=_rms(excursion[_WAIST]),
        arm_excursion_rms_rad=_rms(excursion[_ARMS]),
        upper_body_motion_energy=(3.0 * waist_velocity**2 + 14.0 * arm_velocity**2)
        * _CONTROL_DT_SEC,
    )


def measure_g1_follow_through_agility(
    trajectory: dict[str, NDArray[np.generic]],
    *,
    center_policy_frame: int | None = None,
) -> G1FollowThroughAgilityMetrics:
    """Measure actual motion in a teacher-active or fixed counterfactual window."""

    position = np.asarray(trajectory["shooter_joint_position"], dtype=np.float64)
    velocity = np.asarray(trajectory["shooter_joint_velocity"], dtype=np.float64)
    teacher = np.asarray(
        trajectory["shooter_agility_prior_velocity_delta"],
        dtype=np.float64,
    )
    teacher_active = np.asarray(
        trajectory["shooter_agility_prior_active"],
        dtype=np.bool_,
    )
    if center_policy_frame is None:
        active = teacher_active
    else:
        policy_frame = np.asarray(trajectory["shooter_policy_frame"], dtype=np.int64)
        if policy_frame.shape != (position.shape[0],):
            raise ValueError("follow-through policy-frame trace is malformed")
        active = (policy_frame >= center_policy_frame - 10) & (
            policy_frame <= center_policy_frame + 10
        )
    if (
        position.ndim != 2
        or position.shape[1] != 29
        or velocity.shape != position.shape
        or teacher.shape != position.shape
        or teacher_active.shape != (position.shape[0],)
        or not all(np.isfinite(value).all() for value in (position, velocity, teacher))
    ):
        raise ValueError("follow-through agility trajectory is malformed or non-finite")
    count = int(np.sum(active))
    if count == 0:
        return G1FollowThroughAgilityMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    excursion = np.ptp(position[active], axis=0)
    waist_velocity = _rms(velocity[active, _WAIST])
    arm_velocity = _rms(velocity[active, _ARMS])
    target_teacher = np.asarray(
        trajectory.get("shooter_agility_prior_target_delta", np.zeros_like(position)),
        dtype=np.float64,
    )
    if target_teacher.shape != position.shape or not np.isfinite(target_teacher).all():
        raise ValueError("follow-through pose-teacher trace is malformed or non-finite")
    return G1FollowThroughAgilityMetrics(
        active_frame_count=count,
        waist_velocity_rms_rad_s=waist_velocity,
        arm_velocity_rms_rad_s=arm_velocity,
        waist_excursion_rms_rad=_rms(excursion[_WAIST]),
        arm_excursion_rms_rad=_rms(excursion[_ARMS]),
        upper_body_motion_energy=(3.0 * waist_velocity**2 + 14.0 * arm_velocity**2)
        * _CONTROL_DT_SEC,
        teacher_position_l1_rad=float(np.sum(np.abs(target_teacher[active]))),
        teacher_velocity_l1_rad_s=float(np.sum(np.abs(teacher[active]))),
    )


def evaluate_g1_agility_trial(
    *,
    candidate: G1AgilityCandidate,
    result: G1SharedWorldResult,
    trajectory: dict[str, NDArray[np.generic]],
    parent_naturalness: G1MotionNaturalnessMetrics,
    parent_agility: G1AgilityMetrics,
    naturalness: G1MotionNaturalnessMetrics,
    agility: G1AgilityMetrics,
) -> G1AgilityTrial:
    """Reject apparent flexibility that forgets accuracy, balance, or support."""

    hard_failures = (
        not result.finite_state,
        not result.passed,
        result.joint_limit_violation,
        result.torque_limit_violation,
        result.actuator_saturation,
        result.shooter_post_kick_fall,
        result.shooter_min_pelvis_height_m < 0.65,
    )
    safety_cost = float(sum(hard_failures) / len(hard_failures))
    gates = {
        "accuracy_3cm": naturalness.target_error_m <= 0.03,
        "backward_speed_1cm_s": (naturalness.post_contact_peak_backward_velocity_mps <= 0.01),
        "support_slip_6cm": naturalness.post_contact_support_slip_m <= 0.06,
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
        "upper_body_velocity_retained": (
            agility.upper_body_motion_energy >= 0.995 * parent_agility.upper_body_motion_energy
        ),
    }
    reasons = tuple(name for name, passed in gates.items() if not passed)
    eligible = bool(safety_cost == 0.0 and not reasons)
    score = (
        2.0 * _relative_gain(parent_naturalness.target_error_m, naturalness.target_error_m)
        + 1.5
        * _relative_gain(
            parent_naturalness.post_contact_peak_backward_velocity_mps,
            naturalness.post_contact_peak_backward_velocity_mps,
        )
        + _relative_gain(
            parent_naturalness.contact_joint_acceleration_rms_rad_s2,
            naturalness.contact_joint_acceleration_rms_rad_s2,
        )
        + _relative_gain(
            parent_naturalness.post_contact_root_acceleration_rms_m_s2,
            naturalness.post_contact_root_acceleration_rms_m_s2,
        )
        + _relative_gain(
            parent_agility.upper_body_motion_energy,
            agility.upper_body_motion_energy,
        )
        - _relative_regression(
            parent_naturalness.post_contact_support_slip_m,
            naturalness.post_contact_support_slip_m,
        )
        - 10.0 * safety_cost
    )
    return G1AgilityTrial(
        candidate=candidate,
        candidate_hash=candidate.candidate_hash,
        trajectory_digest=trajectory_digest(trajectory),
        naturalness=naturalness,
        agility=agility,
        safety_cost=safety_cost,
        eligible=eligible,
        score=score,
        reasons=reasons,
    )


def _rms(values: NDArray[np.generic]) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _relative_gain(parent: float, candidate: float) -> float:
    return (parent - candidate) / max(parent, 1e-9)


def _relative_regression(parent: float, candidate: float) -> float:
    return max(0.0, candidate - parent) / max(parent, 1e-9)


__all__ = [
    "G1AgilityCandidate",
    "G1FollowThroughAgilityMetrics",
    "G1AgilityMetrics",
    "G1AgilityNeighborhood",
    "G1AgilityTrial",
    "G1AgilitySearchResult",
    "default_g1_agility_candidates",
    "evaluate_g1_agility_trial",
    "evaluate_g1_agility_neighborhood",
    "measure_g1_agility",
    "measure_g1_follow_through_agility",
    "search_g1_agility_candidate",
]
