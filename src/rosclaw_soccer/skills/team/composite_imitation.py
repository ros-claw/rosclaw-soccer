"""Stability-aware MotionDecode + OmniContact composite imitation search."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.growth.football_motion_prior import (
    G1FootballMotionPrior,
    load_g1_football_motion_prior,
)
from rosclaw_soccer.providers.g1.asset_qualification import trajectory_digest
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.skills.team.imitation_learning import (
    G1ImitationCandidate,
    G1MotionNaturalnessMetrics,
    measure_g1_motion_naturalness,
)
from rosclaw_soccer.skills.team.shared_world import G1SharedWorldResult, simulate_shared_world

_CONTROL_DT_SEC = 0.02
_RIGHT_LEG = slice(6, 12)


@dataclass(frozen=True)
class G1CompositeImitationCandidate:
    contact_position_blend: float
    contact_policy_frame: int
    foot_yaw_offset_rad: float
    post_policy_forward_velocity_mps: float
    contact_joint_scales: tuple[float, ...] = (1.0,) * 6
    foot_pitch_offset_rad: float = 0.01
    schema_version: str = "rosclaw_soccer.g1_composite_imitation_candidate.v1"

    def __post_init__(self) -> None:
        values = (
            self.contact_position_blend,
            self.foot_yaw_offset_rad,
            self.foot_pitch_offset_rad,
            self.post_policy_forward_velocity_mps,
            *self.contact_joint_scales,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("composite imitation candidate values must be finite")
        if not 0.0 < self.contact_position_blend <= 0.05:
            raise ValueError("contact imitation blend must be in (0, 0.05]")
        if not 220 <= self.contact_policy_frame <= 290:
            raise ValueError("contact imitation frame must be in [220, 290]")
        if not 0.0 <= self.post_policy_forward_velocity_mps <= 0.15:
            raise ValueError("composite follow-through speed must be in [0, 0.15] m/s")
        if len(self.contact_joint_scales) != 6 or not all(
            0.0 <= value <= 1.0 for value in self.contact_joint_scales
        ):
            raise ValueError("contact joint scales must contain six values in [0, 1]")
        if not any(value > 0.0 for value in self.contact_joint_scales):
            raise ValueError("composite contact teacher must retain at least one joint")

    @property
    def candidate_hash(self) -> str:
        return hash_json(asdict(self))

    def simulation_overrides(
        self,
        *,
        motion_prior_path: Path,
        contact_prior_path: Path,
    ) -> dict[str, Any]:
        motion = G1ImitationCandidate(
            position_blend=0.02,
            velocity_blend=0.02,
            contact_policy_frame=253,
            foot_yaw_offset_rad=self.foot_yaw_offset_rad,
            foot_pitch_offset_rad=self.foot_pitch_offset_rad,
            post_policy_forward_velocity_mps=self.post_policy_forward_velocity_mps,
        ).simulation_overrides(motion_prior_path)
        motion.update(
            shooter_contact_prior_path=contact_prior_path,
            shooter_contact_prior_position_blend=self.contact_position_blend,
            shooter_contact_prior_contact_policy_frame=self.contact_policy_frame,
            shooter_contact_prior_joint_scales=self.contact_joint_scales,
        )
        return motion


@dataclass(frozen=True)
class G1ContactImitationMetrics:
    teacher_displacement_error_rms_rad: float
    peak_contact_target_delta_rad: float
    active_fraction: float
    schema_version: str = "rosclaw_soccer.g1_contact_imitation_metrics.v1"


@dataclass(frozen=True)
class G1CompositeImitationTrial:
    candidate: G1CompositeImitationCandidate
    candidate_hash: str
    trajectory_digest: str
    result_passed: bool
    finite_state: bool
    safety_cost: float
    motiondecode_naturalness: G1MotionNaturalnessMetrics
    omnicontact_tracking: G1ContactImitationMetrics
    eligible: bool
    score: float
    reasons: tuple[str, ...]
    schema_version: str = "rosclaw_soccer.g1_composite_imitation_trial.v1"


@dataclass(frozen=True)
class G1CompositeImitationSearchResult:
    parent_trajectory_digest: str
    parent_result: G1SharedWorldResult
    parent_motiondecode_naturalness: G1MotionNaturalnessMetrics
    parent_omnicontact_tracking: G1ContactImitationMetrics
    trials: tuple[G1CompositeImitationTrial, ...]
    selected_trial: G1CompositeImitationTrial | None
    selected_candidate: G1CompositeImitationCandidate | None
    passed: bool
    reasons: tuple[str, ...]
    activation_ceiling: str = "SIM_ONLY"
    hardware_command_sent: bool = False
    schema_version: str = "rosclaw_soccer.g1_composite_imitation_search_result.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "parent_result": self.parent_result.to_dict(),
            "parent_motiondecode_naturalness": asdict(self.parent_motiondecode_naturalness),
            "parent_omnicontact_tracking": asdict(self.parent_omnicontact_tracking),
            "trials": [asdict(trial) for trial in self.trials],
            "selected_trial": (
                None if self.selected_trial is None else asdict(self.selected_trial)
            ),
            "selected_candidate": (
                None if self.selected_candidate is None else asdict(self.selected_candidate)
            ),
        }


def default_composite_imitation_candidates() -> tuple[G1CompositeImitationCandidate, ...]:
    """Return a tiny curriculum around the qualified S5 parent skill."""

    return (
        G1CompositeImitationCandidate(0.0010, 253, 0.0900, 0.060),
        G1CompositeImitationCandidate(0.0015, 253, 0.0900, 0.060),
        G1CompositeImitationCandidate(0.0025, 253, 0.0900, 0.060),
        G1CompositeImitationCandidate(0.0025, 256, 0.0900, 0.060),
    )


def search_g1_composite_imitation_candidate(
    *,
    asset_root: Path,
    motion_prior_path: Path,
    contact_prior_path: Path,
    simulation_kwargs: dict[str, Any],
    candidates: tuple[G1CompositeImitationCandidate, ...] | None = None,
) -> G1CompositeImitationSearchResult:
    """Search real-contact plasticity while retaining the qualified S5 memory."""

    motion_prior = load_g1_football_motion_prior(motion_prior_path)
    contact_prior = load_g1_football_motion_prior(contact_prior_path)
    if not motion_prior.whole_body_velocity_reference_rad_s:
        raise ValueError("composite imitation requires a velocity-aware MotionDecode prior")
    if (
        contact_prior.schema_version != "rosclaw.growth.g1_football_motion_prior.v1"
        or contact_prior.source_dataset != "OmniContact"
    ):
        raise ValueError("composite imitation requires the train-only OmniContact prior")
    active_candidates = candidates or default_composite_imitation_candidates()
    if not 2 <= len(active_candidates) <= 16:
        raise ValueError("composite imitation search requires 2-16 candidates")
    if len({candidate.candidate_hash for candidate in active_candidates}) != len(active_candidates):
        raise ValueError("composite imitation candidates must be unique")

    parent_kwargs = {
        **simulation_kwargs,
        **G1ImitationCandidate(0.02, 0.02, 253, 0.09, 0.01, 0.06).simulation_overrides(
            motion_prior_path
        ),
    }
    parent_result, parent_trajectory = simulate_shared_world(asset_root, **parent_kwargs)
    parent_naturalness = measure_g1_motion_naturalness(
        trajectory=parent_trajectory,
        result=parent_result,
        prior=motion_prior,
        contact_policy_frame=253,
    )
    parent_contact = measure_g1_contact_imitation(
        trajectory=parent_trajectory,
        result=parent_result,
        prior=contact_prior,
        contact_policy_frame=253,
        joint_scales=(1.0,) * 6,
    )
    trials: list[G1CompositeImitationTrial] = []
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
        contact = measure_g1_contact_imitation(
            trajectory=trajectory,
            result=result,
            prior=contact_prior,
            contact_policy_frame=candidate.contact_policy_frame,
            joint_scales=candidate.contact_joint_scales,
        )
        trials.append(
            evaluate_g1_composite_imitation_trial(
                candidate=candidate,
                result=result,
                trajectory=trajectory,
                parent_result=parent_result,
                parent=parent_naturalness,
                parent_contact=parent_contact,
                naturalness=naturalness,
                contact=contact,
            )
        )
    eligible = [trial for trial in trials if trial.eligible]
    selected = max(eligible, key=lambda trial: trial.score) if eligible else None
    return G1CompositeImitationSearchResult(
        parent_trajectory_digest=trajectory_digest(parent_trajectory),
        parent_result=parent_result,
        parent_motiondecode_naturalness=parent_naturalness,
        parent_omnicontact_tracking=parent_contact,
        trials=tuple(trials),
        selected_trial=selected,
        selected_candidate=None if selected is None else selected.candidate,
        passed=selected is not None,
        reasons=() if selected is not None else ("no_safe_composite_imitation_candidate",),
    )


def measure_g1_contact_imitation(
    *,
    trajectory: dict[str, np.ndarray],
    result: G1SharedWorldResult,
    prior: G1FootballMotionPrior,
    contact_policy_frame: int,
    joint_scales: tuple[float, ...],
) -> G1ContactImitationMetrics:
    """Compare expert-relative right-leg displacement to train-only contact data."""

    policy_frame = np.asarray(trajectory["shooter_policy_frame"], dtype=np.int64)
    position = np.asarray(trajectory["shooter_joint_position"], dtype=np.float64)[:, _RIGHT_LEG]
    if position.shape != (len(policy_frame), 6) or not np.isfinite(position).all():
        raise ValueError("contact imitation trajectory is malformed or non-finite")
    times = np.asarray(prior.reference_times_sec, dtype=np.float64)
    relative_time = (policy_frame - contact_policy_frame) * _CONTROL_DT_SEC
    active = (relative_time >= times[0]) & (relative_time <= times[-1])
    if int(np.sum(active)) < 5:
        raise ValueError("contact imitation window is too short")
    reference = np.asarray(prior.right_leg_reference_rad, dtype=np.float64)
    teacher = np.stack(
        [np.interp(relative_time[active], times, reference[:, joint]) for joint in range(6)],
        axis=1,
    )
    teacher -= reference[0]
    observed = position[active] - position[active][0]
    scales = np.asarray(joint_scales, dtype=np.float64)
    selected = scales > 0.0
    teacher = teacher[:, selected] * scales[selected]
    observed = observed[:, selected] * scales[selected]
    runtime_active = np.asarray(
        trajectory.get("shooter_contact_prior_active", np.zeros(len(policy_frame))),
        dtype=np.float64,
    )
    runtime_delta = np.asarray(
        trajectory.get("shooter_contact_prior_target_delta", np.zeros((len(policy_frame), 29))),
        dtype=np.float64,
    )
    peak = float(np.max(np.abs(runtime_delta))) if runtime_delta.size else 0.0
    return G1ContactImitationMetrics(
        teacher_displacement_error_rms_rad=_rms(observed - teacher),
        peak_contact_target_delta_rad=peak,
        active_fraction=float(np.mean(runtime_active)) if runtime_active.size else 0.0,
    )


def evaluate_g1_composite_imitation_trial(
    *,
    candidate: G1CompositeImitationCandidate,
    result: G1SharedWorldResult,
    trajectory: dict[str, np.ndarray],
    parent_result: G1SharedWorldResult,
    parent: G1MotionNaturalnessMetrics,
    parent_contact: G1ContactImitationMetrics,
    naturalness: G1MotionNaturalnessMetrics,
    contact: G1ContactImitationMetrics,
) -> G1CompositeImitationTrial:
    """Apply an explicit stability/plasticity gate against the qualified S5 parent."""

    if parent_result.target_error_m != parent.target_error_m:
        raise ValueError("composite parent metrics are not bound to the parent result")
    if result.target_error_m != naturalness.target_error_m:
        raise ValueError("composite candidate metrics are not bound to the candidate result")
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
        "accuracy_bounded": naturalness.target_error_m <= max(0.03, parent.target_error_m + 0.005),
        "omnicontact_tracking_improved": (
            contact.teacher_displacement_error_rms_rad
            < parent_contact.teacher_displacement_error_rms_rad
        ),
        "motiondecode_position_retained": (
            naturalness.teacher_position_error_rms_rad
            <= 1.01 * parent.teacher_position_error_rms_rad
        ),
        "motiondecode_velocity_retained": (
            naturalness.teacher_velocity_error_rms_rad_s
            <= 1.01 * parent.teacher_velocity_error_rms_rad_s
        ),
        "contact_smoothness_retained": (
            naturalness.contact_joint_acceleration_rms_rad_s2
            <= 1.01 * parent.contact_joint_acceleration_rms_rad_s2
        ),
        "post_contact_smoother": (
            naturalness.post_contact_joint_acceleration_rms_rad_s2
            <= parent.post_contact_joint_acceleration_rms_rad_s2
        ),
        "root_acceleration_bounded": (
            naturalness.post_contact_root_acceleration_rms_m_s2
            <= 1.02 * parent.post_contact_root_acceleration_rms_m_s2
        ),
        "backward_motion_bounded": (naturalness.post_contact_peak_backward_velocity_mps <= 0.01),
        "support_slip_reduced": (
            naturalness.post_contact_support_slip_m < parent.post_contact_support_slip_m
        ),
        "roll_reduced": naturalness.torso_roll_peak_rad < parent.torso_roll_peak_rad,
        "tail_wobble_bounded": naturalness.tail_wobble_index <= 1.10 * parent.tail_wobble_index,
        "contact_teacher_executed": contact.active_fraction > 0.0,
    }
    reasons = tuple(name for name, passed in gates.items() if not passed)
    eligible = bool(safety_cost == 0.0 and not reasons)
    score = (
        2.0
        * _improvement(
            parent_contact.teacher_displacement_error_rms_rad,
            contact.teacher_displacement_error_rms_rad,
        )
        + _improvement(
            parent.post_contact_support_slip_m,
            naturalness.post_contact_support_slip_m,
        )
        + _improvement(parent.torso_roll_peak_rad, naturalness.torso_roll_peak_rad)
        + _improvement(
            parent.contact_joint_acceleration_rms_rad_s2,
            naturalness.contact_joint_acceleration_rms_rad_s2,
        )
        + _improvement(
            parent.post_contact_joint_acceleration_rms_rad_s2,
            naturalness.post_contact_joint_acceleration_rms_rad_s2,
        )
        - max(0.0, naturalness.target_error_m - parent.target_error_m) / 0.01
        - 10.0 * safety_cost
    )
    return G1CompositeImitationTrial(
        candidate=candidate,
        candidate_hash=candidate.candidate_hash,
        trajectory_digest=trajectory_digest(trajectory),
        result_passed=result.passed,
        finite_state=result.finite_state,
        safety_cost=safety_cost,
        motiondecode_naturalness=naturalness,
        omnicontact_tracking=contact,
        eligible=eligible,
        score=score,
        reasons=reasons,
    )


def _rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(value))))


def _improvement(parent: float, candidate: float) -> float:
    return (parent - candidate) / max(abs(parent), 1e-9)


__all__ = [
    "G1CompositeImitationCandidate",
    "G1CompositeImitationSearchResult",
    "G1CompositeImitationTrial",
    "G1ContactImitationMetrics",
    "default_composite_imitation_candidates",
    "evaluate_g1_composite_imitation_trial",
    "measure_g1_contact_imitation",
    "search_g1_composite_imitation_candidate",
]
