"""Bounded, evidence-driven goalkeeper block discovery in CPU MuJoCo."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.asset_qualification import trajectory_digest
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.skills.team.shared_world import (
    G1GoalkeeperConfig,
    G1SharedWorldResult,
    simulate_shared_world,
)


@dataclass(frozen=True)
class GoalkeeperBlockSearchConfig:
    """Small SIM_ONLY curriculum around the first safe contact boundary."""

    hip_pitch_candidates_rad: tuple[float, ...] = (
        0.252,
        0.255,
        0.258,
        0.260,
        0.262,
        0.265,
        0.268,
        0.270,
    )
    minimum_pelvis_height_m: float = 0.65
    maximum_post_contact_speed_ratio: float = 0.80
    schema_version: str = "rosclaw_soccer.goalkeeper_block_search_config.v1"

    def __post_init__(self) -> None:
        if not 3 <= len(self.hip_pitch_candidates_rad) <= 32:
            raise ValueError("goalkeeper block search requires 3-32 candidates")
        if len(set(self.hip_pitch_candidates_rad)) != len(self.hip_pitch_candidates_rad):
            raise ValueError("goalkeeper block candidates must be unique")
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 0.50
            for value in self.hip_pitch_candidates_rad
        ):
            raise ValueError("goalkeeper block candidate is outside [0, 0.50] rad")
        if not 0.55 <= self.minimum_pelvis_height_m <= 0.80:
            raise ValueError("goalkeeper block pelvis gate must be in [0.55, 0.80] m")
        if not 0.25 <= self.maximum_post_contact_speed_ratio <= 1.0:
            raise ValueError("goalkeeper post-contact speed ratio must be in [0.25, 1]")


@dataclass(frozen=True)
class GoalkeeperBlockTrial:
    hip_pitch_rad: float
    policy_hash: str
    trajectory_digest: str
    goalkeeper_contact_observed: bool
    goalkeeper_save_observed: bool
    goal_plane_crossed: bool
    goal_crossed: bool
    goalkeeper_min_pelvis_height_m: float
    pre_contact_ball_speed_mps: float
    post_contact_peak_ball_speed_mps: float
    post_contact_speed_ratio: float
    safety_cost: float
    eligible: bool
    score: float
    schema_version: str = "rosclaw_soccer.goalkeeper_block_trial.v1"


@dataclass(frozen=True)
class GoalkeeperBlockSearchResult:
    trials: tuple[GoalkeeperBlockTrial, ...]
    selected_trial: GoalkeeperBlockTrial | None
    selected_config: G1GoalkeeperConfig | None
    passed: bool
    reasons: tuple[str, ...]
    activation_ceiling: str = "SIM_ONLY"
    physics_authority: str = "CPU_MUJOCO"
    hardware_command_sent: bool = False
    schema_version: str = "rosclaw_soccer.goalkeeper_block_search_result.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "selected_config": (
                None if self.selected_config is None else asdict(self.selected_config)
            ),
        }


def goalkeeper_block_parent_config(parent: G1GoalkeeperConfig) -> G1GoalkeeperConfig:
    """Preserve the locomotion expert and expose only a causal block residual."""

    return replace(
        parent,
        reaction_delay_sec=0.04,
        anticipation_enabled=True,
        anticipation_start_policy_frame=180,
        anticipation_maximum_foot_ball_distance_m=4.0,
        anticipation_target_blend=1.0,
        anticipation_velocity_scale=1.0,
        block_action_enabled=False,
        block_action_start_policy_frame=250,
        block_action_blend_frames=12,
        block_action_hold_frames=40,
    )


def search_goalkeeper_block_candidate(
    *,
    asset_root: Path,
    simulation_kwargs: dict[str, Any],
    config: GoalkeeperBlockSearchConfig | None = None,
) -> GoalkeeperBlockSearchResult:
    """Select the smallest safe save from measured shared-world rollouts."""

    active = config or GoalkeeperBlockSearchConfig()
    parent = simulation_kwargs.get("goalkeeper_config")
    if not isinstance(parent, G1GoalkeeperConfig):
        raise ValueError("goalkeeper block search requires a goalkeeper parent")
    parent = goalkeeper_block_parent_config(parent)
    trials: list[GoalkeeperBlockTrial] = []
    configs: list[G1GoalkeeperConfig] = []
    for hip_pitch in active.hip_pitch_candidates_rad:
        candidate = replace(
            parent,
            block_action_enabled=True,
            block_action_hip_pitch_rad=hip_pitch,
        )
        kwargs = dict(simulation_kwargs)
        kwargs["goalkeeper_config"] = candidate
        result, trajectory = simulate_shared_world(asset_root, **kwargs)
        trial = _block_trial(
            candidate=candidate,
            result=result,
            trajectory=trajectory,
            search_config=active,
        )
        trials.append(trial)
        configs.append(candidate)
    eligible_indices = [index for index, trial in enumerate(trials) if trial.eligible]
    if not eligible_indices:
        return GoalkeeperBlockSearchResult(
            trials=tuple(trials),
            selected_trial=None,
            selected_config=None,
            passed=False,
            reasons=("no_safe_goalkeeper_save",),
        )
    selected_index = max(eligible_indices, key=lambda index: trials[index].score)
    return GoalkeeperBlockSearchResult(
        trials=tuple(trials),
        selected_trial=trials[selected_index],
        selected_config=configs[selected_index],
        passed=True,
        reasons=(),
    )


def _block_trial(
    *,
    candidate: G1GoalkeeperConfig,
    result: G1SharedWorldResult,
    trajectory: dict[str, np.ndarray],
    search_config: GoalkeeperBlockSearchConfig,
) -> GoalkeeperBlockTrial:
    time = np.asarray(trajectory["time"], dtype=np.float64)
    velocity = np.asarray(trajectory["ball_velocity"], dtype=np.float64)
    contact_time = result.goalkeeper_ball_contact_time_sec
    contact_index = len(time) if contact_time is None else int(np.searchsorted(time, contact_time))
    pre_index = min(max(0, contact_index - 2), len(velocity) - 1)
    pre_speed = float(np.linalg.norm(velocity[pre_index, :3]))
    post_window = velocity[contact_index : min(len(velocity), contact_index + 8), :3]
    post_speed = 0.0 if not len(post_window) else float(np.max(np.linalg.norm(post_window, axis=1)))
    ratio = post_speed / pre_speed if pre_speed > 1e-9 else math.inf
    minimum_height = float(result.goalkeeper_min_pelvis_height_m or 0.0)
    hard_failures = (
        not result.finite_state,
        result.joint_limit_violation,
        result.torque_limit_violation,
        result.actuator_saturation,
        result.goalkeeper_joint_limit_violation,
        minimum_height < search_config.minimum_pelvis_height_m,
    )
    safety_cost = float(sum(hard_failures) / len(hard_failures))
    eligible = bool(
        result.goalkeeper_ball_contact_observed
        and result.goalkeeper_save_observed
        and safety_cost == 0.0
        and ratio <= search_config.maximum_post_contact_speed_ratio
    )
    # Prefer an actual stop before the plane, then the smallest policy residual;
    # speed dissipation is a tie-breaker rather than an incentive for hard hits.
    score = (
        5.0 * float(result.goalkeeper_save_observed)
        + 1.0 * float(not result.goal_plane_crossed)
        - candidate.block_action_hip_pitch_rad
        - 0.10 * min(ratio, 10.0)
        - 10.0 * safety_cost
    )
    return GoalkeeperBlockTrial(
        hip_pitch_rad=candidate.block_action_hip_pitch_rad,
        policy_hash=hash_json(asdict(candidate)),
        trajectory_digest=trajectory_digest(trajectory),
        goalkeeper_contact_observed=result.goalkeeper_ball_contact_observed,
        goalkeeper_save_observed=result.goalkeeper_save_observed,
        goal_plane_crossed=result.goal_plane_crossed,
        goal_crossed=result.goal_crossed,
        goalkeeper_min_pelvis_height_m=minimum_height,
        pre_contact_ball_speed_mps=pre_speed,
        post_contact_peak_ball_speed_mps=post_speed,
        post_contact_speed_ratio=ratio,
        safety_cost=safety_cost,
        eligible=eligible,
        score=score,
    )


__all__ = [
    "GoalkeeperBlockSearchConfig",
    "GoalkeeperBlockSearchResult",
    "GoalkeeperBlockTrial",
    "goalkeeper_block_parent_config",
    "search_goalkeeper_block_candidate",
]
