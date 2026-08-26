"""Data-driven, stability-aware imitation search in the shared G1 world."""

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
from rosclaw_soccer.skills.team.shared_world import (
    G1SharedWorldResult,
    simulate_shared_world,
)

_CONTROL_DT_SEC = 0.02


@dataclass(frozen=True)
class G1ImitationCandidate:
    position_blend: float
    velocity_blend: float
    contact_policy_frame: int
    foot_yaw_offset_rad: float
    foot_pitch_offset_rad: float
    post_policy_forward_velocity_mps: float
    strike_leg_scale: float = 1.0
    schema_version: str = "rosclaw_soccer.g1_imitation_candidate.v1"

    def __post_init__(self) -> None:
        values = (
            self.position_blend,
            self.velocity_blend,
            self.foot_yaw_offset_rad,
            self.foot_pitch_offset_rad,
            self.post_policy_forward_velocity_mps,
            self.strike_leg_scale,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("imitation candidate values must be finite")
        if not 0.0 < self.position_blend <= 0.10:
            raise ValueError("imitation position blend must be in (0, 0.10]")
        if not 0.0 < self.velocity_blend <= 0.10:
            raise ValueError("imitation velocity blend must be in (0, 0.10]")
        if not 220 <= self.contact_policy_frame <= 290:
            raise ValueError("imitation contact frame must be in [220, 290]")
        if not 0.0 <= self.post_policy_forward_velocity_mps <= 0.15:
            raise ValueError("imitation follow-through speed must be in [0, 0.15] m/s")
        if not 0.0 <= self.strike_leg_scale <= 1.0:
            raise ValueError("imitation strike-leg scale must be in [0, 1]")

    @property
    def candidate_hash(self) -> str:
        return hash_json(asdict(self))

    def simulation_overrides(self, motion_prior_path: Path) -> dict[str, Any]:
        return {
            "shooter_parameter_overrides": {
                "foot_yaw_offset": self.foot_yaw_offset_rad,
                "foot_pitch_offset": self.foot_pitch_offset_rad,
            },
            "shooter_motion_prior_path": motion_prior_path,
            "shooter_motion_prior_position_blend": self.position_blend,
            "shooter_motion_prior_velocity_blend": self.velocity_blend,
            "shooter_motion_prior_strike_leg_scale": self.strike_leg_scale,
            "shooter_motion_prior_contact_policy_frame": self.contact_policy_frame,
            "shooter_post_policy_forward_velocity_mps": (self.post_policy_forward_velocity_mps),
        }


@dataclass(frozen=True)
class G1MotionNaturalnessMetrics:
    contact_joint_acceleration_rms_rad_s2: float
    teacher_position_error_rms_rad: float
    teacher_velocity_error_rms_rad_s: float
    post_contact_joint_acceleration_rms_rad_s2: float
    post_contact_root_acceleration_rms_m_s2: float
    post_contact_peak_backward_velocity_mps: float
    post_contact_support_slip_m: float
    torso_roll_peak_rad: float
    tail_wobble_index: float
    target_error_m: float
    schema_version: str = "rosclaw_soccer.g1_motion_naturalness_metrics.v1"

    def __post_init__(self) -> None:
        values = tuple(value for name, value in asdict(self).items() if name != "schema_version")
        if not all(
            isinstance(value, int | float) and math.isfinite(value) and value >= 0.0
            for value in values
        ):
            raise ValueError("motion naturalness metrics must be finite and non-negative")


@dataclass(frozen=True)
class G1ImitationTrial:
    candidate: G1ImitationCandidate
    candidate_hash: str
    trajectory_digest: str
    result_passed: bool
    finite_state: bool
    safety_cost: float
    naturalness: G1MotionNaturalnessMetrics
    eligible: bool
    score: float
    reasons: tuple[str, ...]
    schema_version: str = "rosclaw_soccer.g1_imitation_trial.v1"


@dataclass(frozen=True)
class G1ImitationSearchResult:
    parent_trajectory_digest: str
    parent_result: G1SharedWorldResult
    parent_naturalness: G1MotionNaturalnessMetrics
    trials: tuple[G1ImitationTrial, ...]
    selected_trial: G1ImitationTrial | None
    selected_candidate: G1ImitationCandidate | None
    passed: bool
    reasons: tuple[str, ...]
    activation_ceiling: str = "SIM_ONLY"
    hardware_command_sent: bool = False
    schema_version: str = "rosclaw_soccer.g1_imitation_search_result.v1"

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


def default_imitation_candidates() -> tuple[G1ImitationCandidate, ...]:
    """Return the small bounded curriculum retained after coarse ablation."""

    return (
        G1ImitationCandidate(0.01, 0.01, 253, 0.085, 0.010, 0.04),
        G1ImitationCandidate(0.02, 0.02, 253, 0.090, 0.010, 0.02),
        G1ImitationCandidate(0.02, 0.02, 253, 0.090, 0.010, 0.04),
        G1ImitationCandidate(0.02, 0.02, 253, 0.090, 0.010, 0.06),
        G1ImitationCandidate(0.02, 0.02, 253, 0.090, 0.010, 0.08),
    )


def search_g1_imitation_candidate(
    *,
    asset_root: Path,
    motion_prior_path: Path,
    simulation_kwargs: dict[str, Any],
    candidates: tuple[G1ImitationCandidate, ...] | None = None,
) -> G1ImitationSearchResult:
    """Search a teacher residual without forgetting safety, accuracy or recovery."""

    prior = load_g1_football_motion_prior(motion_prior_path)
    if not prior.whole_body_velocity_reference_rad_s:
        raise ValueError("imitation search requires a whole-body velocity-aware prior")
    active_candidates = candidates or default_imitation_candidates()
    if not 2 <= len(active_candidates) <= 16:
        raise ValueError("imitation search requires 2-16 candidates")
    if len({candidate.candidate_hash for candidate in active_candidates}) != len(active_candidates):
        raise ValueError("imitation candidates must be unique")

    parent_result, parent_trajectory = simulate_shared_world(asset_root, **simulation_kwargs)
    parent_metrics = measure_g1_motion_naturalness(
        trajectory=parent_trajectory,
        result=parent_result,
        prior=prior,
        contact_policy_frame=253,
    )
    trials: list[G1ImitationTrial] = []
    for candidate in active_candidates:
        kwargs = {**simulation_kwargs, **candidate.simulation_overrides(motion_prior_path)}
        result, trajectory = simulate_shared_world(asset_root, **kwargs)
        naturalness = measure_g1_motion_naturalness(
            trajectory=trajectory,
            result=result,
            prior=prior,
            contact_policy_frame=candidate.contact_policy_frame,
        )
        trials.append(
            evaluate_g1_imitation_trial(
                candidate=candidate,
                result=result,
                trajectory=trajectory,
                parent_result=parent_result,
                parent=parent_metrics,
                naturalness=naturalness,
            )
        )
    eligible = [trial for trial in trials if trial.eligible]
    selected = max(eligible, key=lambda trial: trial.score) if eligible else None
    selected_candidate = None if selected is None else selected.candidate
    return G1ImitationSearchResult(
        parent_trajectory_digest=trajectory_digest(parent_trajectory),
        parent_result=parent_result,
        parent_naturalness=parent_metrics,
        trials=tuple(trials),
        selected_trial=selected,
        selected_candidate=selected_candidate,
        passed=selected is not None,
        reasons=() if selected is not None else ("no_safe_natural_imitation_candidate",),
    )


def measure_g1_motion_naturalness(
    *,
    trajectory: dict[str, np.ndarray],
    result: G1SharedWorldResult,
    prior: G1FootballMotionPrior,
    contact_policy_frame: int,
) -> G1MotionNaturalnessMetrics:
    """Measure teacher tracking, impact smoothness and post-kick composure."""

    if result.shot_contact_time_sec is None or result.target_error_m is None:
        raise ValueError("motion naturalness requires measured shot contact and target error")
    time = np.asarray(trajectory["time"], dtype=np.float64)
    policy_frame = np.asarray(trajectory["shooter_policy_frame"], dtype=np.int64)
    joint_position = np.asarray(trajectory["shooter_joint_position"], dtype=np.float64)
    joint_velocity = np.asarray(trajectory["shooter_joint_velocity"], dtype=np.float64)
    pelvis = np.asarray(trajectory["shooter_pelvis_pose"], dtype=np.float64)[:, :3]
    required_shapes = (
        policy_frame.shape == time.shape,
        joint_position.shape == (len(time), 29),
        joint_velocity.shape == (len(time), 29),
        pelvis.shape == (len(time), 3),
    )
    if not all(required_shapes) or not all(
        np.all(np.isfinite(value))
        for value in (time, policy_frame, joint_position, joint_velocity, pelvis)
    ):
        raise ValueError("motion naturalness trajectory is malformed or non-finite")

    prior_times = np.asarray(prior.reference_times_sec, dtype=np.float64)
    start_frame = contact_policy_frame + int(math.ceil(prior_times[0] / _CONTROL_DT_SEC))
    end_frame = contact_policy_frame + int(math.floor(prior_times[-1] / _CONTROL_DT_SEC))
    contact_mask = (policy_frame >= start_frame) & (policy_frame <= end_frame)
    if int(np.sum(contact_mask)) < 5:
        raise ValueError("motion naturalness contact window is too short")
    relative_time = (policy_frame[contact_mask] - contact_policy_frame) * _CONTROL_DT_SEC
    position_reference = np.asarray(prior.whole_body_reference_rad, dtype=np.float64)
    velocity_reference = np.asarray(
        prior.whole_body_velocity_reference_rad_s,
        dtype=np.float64,
    )
    teacher_position = np.stack(
        [
            np.interp(relative_time, prior_times, position_reference[:, joint])
            for joint in range(29)
        ],
        axis=1,
    )
    teacher_velocity = np.stack(
        [
            np.interp(relative_time, prior_times, velocity_reference[:, joint])
            for joint in range(29)
        ],
        axis=1,
    )
    contact_velocity = joint_velocity[contact_mask]
    contact_acceleration = np.diff(contact_velocity, axis=0) / _CONTROL_DT_SEC

    post_start = int(np.searchsorted(time, result.shot_contact_time_sec, side="left"))
    post_end = int(np.searchsorted(time, result.shot_contact_time_sec + 3.0, side="right"))
    if post_end - post_start < 5:
        raise ValueError("motion naturalness post-contact window is too short")
    post_velocity = joint_velocity[post_start:post_end]
    post_acceleration = np.diff(post_velocity, axis=0) / _CONTROL_DT_SEC
    root_velocity = (
        np.diff(pelvis[post_start:post_end], axis=0) / np.diff(time[post_start:post_end])[:, None]
    )
    root_acceleration = np.diff(root_velocity, axis=0) / _CONTROL_DT_SEC
    return G1MotionNaturalnessMetrics(
        contact_joint_acceleration_rms_rad_s2=_rms(contact_acceleration),
        teacher_position_error_rms_rad=_rms(joint_position[contact_mask] - teacher_position),
        teacher_velocity_error_rms_rad_s=_rms(contact_velocity - teacher_velocity),
        post_contact_joint_acceleration_rms_rad_s2=_rms(post_acceleration),
        post_contact_root_acceleration_rms_m_s2=_rms(root_acceleration),
        post_contact_peak_backward_velocity_mps=float(max(0.0, -np.min(root_velocity[:, 0]))),
        post_contact_support_slip_m=result.shooter_post_contact_support_foot_slip_m,
        torso_roll_peak_rad=result.shooter_roll_peak_rad,
        tail_wobble_index=result.shooter_tail_wobble_index,
        target_error_m=result.target_error_m,
    )


def evaluate_g1_imitation_trial(
    *,
    candidate: G1ImitationCandidate,
    result: G1SharedWorldResult,
    trajectory: dict[str, np.ndarray],
    parent_result: G1SharedWorldResult,
    parent: G1MotionNaturalnessMetrics,
    naturalness: G1MotionNaturalnessMetrics,
) -> G1ImitationTrial:
    """Apply a conservative stability-plasticity gate to one rollout."""

    if parent_result.target_error_m is None or not math.isclose(
        parent_result.target_error_m,
        parent.target_error_m,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("parent naturalness is not bound to the parent result")
    if result.target_error_m is None or not math.isclose(
        result.target_error_m,
        naturalness.target_error_m,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("candidate naturalness is not bound to the candidate result")
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
        "accuracy_not_forgotten": naturalness.target_error_m <= parent.target_error_m,
        "teacher_position_improved": (
            naturalness.teacher_position_error_rms_rad <= parent.teacher_position_error_rms_rad
        ),
        "teacher_velocity_improved": (
            naturalness.teacher_velocity_error_rms_rad_s <= parent.teacher_velocity_error_rms_rad_s
        ),
        "contact_smoother": (
            naturalness.contact_joint_acceleration_rms_rad_s2
            <= parent.contact_joint_acceleration_rms_rad_s2
        ),
        "post_contact_smoother": (
            naturalness.post_contact_joint_acceleration_rms_rad_s2
            <= parent.post_contact_joint_acceleration_rms_rad_s2
        ),
        "root_smoother": (
            naturalness.post_contact_root_acceleration_rms_m_s2
            <= parent.post_contact_root_acceleration_rms_m_s2
        ),
        "backward_motion_reduced": (
            naturalness.post_contact_peak_backward_velocity_mps
            <= parent.post_contact_peak_backward_velocity_mps
        ),
        "support_slip_reduced": (
            naturalness.post_contact_support_slip_m <= parent.post_contact_support_slip_m
        ),
        "roll_reduced": naturalness.torso_roll_peak_rad <= parent.torso_roll_peak_rad,
        "tail_wobble_bounded": naturalness.tail_wobble_index <= 1.10 * parent.tail_wobble_index,
    }
    reasons = tuple(name for name, passed in gates.items() if not passed)
    eligible = bool(safety_cost == 0.0 and not reasons)
    score = (
        2.0 * _improvement(parent.target_error_m, naturalness.target_error_m)
        + _improvement(
            parent.contact_joint_acceleration_rms_rad_s2,
            naturalness.contact_joint_acceleration_rms_rad_s2,
        )
        + _improvement(
            parent.post_contact_joint_acceleration_rms_rad_s2,
            naturalness.post_contact_joint_acceleration_rms_rad_s2,
        )
        + _improvement(
            parent.post_contact_root_acceleration_rms_m_s2,
            naturalness.post_contact_root_acceleration_rms_m_s2,
        )
        + 1.5
        * _improvement(
            parent.post_contact_peak_backward_velocity_mps,
            naturalness.post_contact_peak_backward_velocity_mps,
        )
        + _improvement(parent.torso_roll_peak_rad, naturalness.torso_roll_peak_rad)
        + _improvement(
            parent.post_contact_support_slip_m,
            naturalness.post_contact_support_slip_m,
        )
        - 10.0 * safety_cost
    )
    return G1ImitationTrial(
        candidate=candidate,
        candidate_hash=candidate.candidate_hash,
        trajectory_digest=trajectory_digest(trajectory),
        result_passed=result.passed,
        finite_state=result.finite_state,
        safety_cost=safety_cost,
        naturalness=naturalness,
        eligible=eligible,
        score=score,
        reasons=reasons,
    )


def _rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(value))))


def _improvement(parent: float, candidate: float) -> float:
    return (parent - candidate) / max(abs(parent), 1e-9)


__all__ = [
    "G1ImitationCandidate",
    "G1ImitationSearchResult",
    "G1ImitationTrial",
    "G1MotionNaturalnessMetrics",
    "default_imitation_candidates",
    "evaluate_g1_imitation_trial",
    "measure_g1_motion_naturalness",
    "search_g1_imitation_candidate",
]
