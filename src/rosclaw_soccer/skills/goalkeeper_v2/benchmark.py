"""Measured CPU MuJoCo launcher benchmark for the Goalkeeper V2 parent."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from rosclaw.continual.reproducibility import NumericalRuntimeContract

from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.skills.goalkeeper_v2.coverage_time import (
    GoalkeeperCoverageTimeReport,
    GoalkeeperCoverageTrial,
    aggregate_coverage_time,
)
from rosclaw_soccer.skills.goalkeeper_v2.observations import GoalkeeperObservationSpec
from rosclaw_soccer.skills.team.agility_profiler import profile_temporal_agility
from rosclaw_soccer.skills.team.development_evidence import three_role_development_kwargs
from rosclaw_soccer.skills.team.shared_world import G1GoalkeeperConfig, simulate_shared_world

_DEADLINES_SEC = (1.0, 0.8, 0.6, 0.5, 0.4)
_TARGETS = (
    ("upper_left", 0.90, 1.60),
    ("upper_right", -0.90, 1.60),
    ("lower_left", 0.90, 0.35),
    ("lower_right", -0.90, 0.35),
    ("center", 0.0, 1.0),
)


def run_parent_coverage_time_baseline(
    *,
    asset_root: Path,
    numerical_contract: NumericalRuntimeContract,
    actor_artifact_path: Path | None = None,
) -> tuple[GoalkeeperCoverageTimeReport, tuple[GoalkeeperCoverageTrial, ...]]:
    """Measure the untrained causal parent; this function never claims promotion."""

    environment = numerical_contract.verify_environment()
    if not environment.passed:
        raise RuntimeError(
            "goalkeeper coverage benchmark numerical environment mismatches: "
            + ",".join(environment.mismatches)
        )
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    base = three_role_development_kwargs()
    parent = base.get("goalkeeper_config")
    if not isinstance(parent, G1GoalkeeperConfig):
        raise RuntimeError("coverage benchmark requires a goalkeeper parent")
    parent = replace(
        parent,
        actor_observation_mode="visible_ball_history_v3",
        anticipation_enabled=False,
        actor_artifact_path=None,
    )
    observation_spec = GoalkeeperObservationSpec()
    evaluated_actor_hash = hash_json(asdict(parent))
    if actor_artifact_path is not None:
        from rosclaw_soccer.skills.goalkeeper_v2.policy import (
            load_goalkeeper_actor_artifact,
        )

        artifact = load_goalkeeper_actor_artifact(actor_artifact_path)
        if artifact.body_hash != qualification.body_hash:
            raise ValueError("coverage benchmark actor Body hash mismatch")
        if artifact.actor_observation_contract_hash != observation_spec.actor_contract_hash:
            raise ValueError("coverage benchmark actor observation contract changed")
        if artifact.parent_policy_hash != evaluated_actor_hash:
            raise ValueError("coverage benchmark actor was not derived from the frozen parent")
        evaluated_actor_hash = artifact.policy_hash
        parent = replace(parent, actor_artifact_path=actor_artifact_path)
    launcher_x = float(base["goal_spec"].plane_x_m - 5.0)
    trials: list[GoalkeeperCoverageTrial] = []
    for deadline in _DEADLINES_SEC:
        for seed, (region, target_y, target_z) in enumerate(_TARGETS):
            start = (launcher_x, 0.0, 0.6)
            target = (float(base["goal_spec"].plane_x_m), target_y, target_z)
            velocity = (
                (target[0] - start[0]) / deadline,
                (target[1] - start[1]) / deadline,
                (target[2] - start[2] + 4.905 * deadline * deadline) / deadline,
            )
            kwargs = dict(base)
            kwargs.update(
                {
                    "goalkeeper_config": parent,
                    "ball_launcher_position_m": start,
                    "ball_launcher_velocity_mps": velocity,
                    "simulation_duration_sec": 3.0,
                }
            )
            result, trajectory = simulate_shared_world(asset_root, **kwargs)
            flight = np.asarray(trajectory["goalkeeper_observed_flight_active"], dtype=bool)
            observed = (
                None
                if not np.any(flight)
                else float(trajectory["time"][int(np.flatnonzero(flight)[0])])
            )
            profile = profile_temporal_agility(
                trajectory,
                role="goalkeeper",
                observation_event_sec=observed,
                window_start_sec=observed,
                window_end_sec=min(3.0, deadline + 1.0),
            )
            intercept_error = _lateral_intercept_prediction_error(
                trajectory,
                contact_time_sec=result.goalkeeper_ball_contact_time_sec,
            )
            safety_failures = tuple(
                code
                for code, failed in (
                    ("NON_FINITE_STATE", not result.finite_state),
                    ("GLOBAL_JOINT_LIMIT", result.joint_limit_violation),
                    ("TORQUE_LIMIT", result.torque_limit_violation),
                    ("ACTUATOR_SATURATION", result.actuator_saturation),
                    ("GOALKEEPER_JOINT_LIMIT", result.goalkeeper_joint_limit_violation),
                    (
                        "GOALKEEPER_PELVIS_BELOW_0_60_M",
                        float(result.goalkeeper_min_pelvis_height_m or 0.0) < 0.60,
                    ),
                )
                if failed
            )
            safety_cost = float(bool(safety_failures))
            # "Reaction" means the first target-directed whole-body command,
            # not any non-zero neural residual.  The latter let a wrong-way
            # twitch masquerade as zero-latency intelligence in S9.
            reaction = np.asarray(trajectory["goalkeeper_useful_reaction_active"], dtype=bool)
            first_action = (
                None
                if not np.any(reaction)
                else float(trajectory["time"][int(np.flatnonzero(reaction)[0])])
            )
            trials.append(
                GoalkeeperCoverageTrial(
                    scenario_hash=hash_json(
                        {
                            "deadline_sec": deadline,
                            "region": region,
                            "target_m": list(target),
                            "launcher_position_m": list(start),
                            "launcher_velocity_mps": list(velocity),
                            "seed": seed,
                        }
                    ),
                    frozen_shooter_policy_hash=hash_json(
                        {
                            "type": "deterministic_ball_launcher",
                            "source_policy": qualification.kick_prior_hash,
                        }
                    ),
                    numerical_contract_hash=numerical_contract.contract_hash,
                    seed=seed,
                    target_region=region,
                    target_y_m=target_y,
                    target_z_m=target_z,
                    deadline_sec=deadline,
                    observed_flight_start_sec=observed,
                    first_action_sec=first_action,
                    ball_contact=result.goalkeeper_ball_contact_observed,
                    true_save=result.goalkeeper_save_observed,
                    intercept_error_m=intercept_error,
                    recovery_time_sec=None,
                    second_save_success=False,
                    idle_ratio=profile.idle_ratio,
                    human_motion_score=None,
                    safety_cost=safety_cost,
                    actor_observation_contract_hash=observation_spec.actor_contract_hash,
                    safety_failure_codes=safety_failures,
                    minimum_pelvis_height_m=result.goalkeeper_min_pelvis_height_m,
                    evaluated_actor_policy_hash=evaluated_actor_hash,
                )
            )
    report = aggregate_coverage_time(
        tuple(trials),
        strict_replay=False,
        sealed_holdout=False,
    )
    return report, tuple(trials)


def _lateral_intercept_prediction_error(
    trajectory: dict[str, np.ndarray],
    *,
    contact_time_sec: float | None,
) -> float:
    """Measure causal lateral prediction error at the keeper's intercept plane."""

    time = np.asarray(trajectory["time"], dtype=np.float64)
    ball = np.asarray(trajectory["ball_pose"], dtype=np.float64)
    keeper = np.asarray(trajectory["goalkeeper_pelvis_pose"], dtype=np.float64)
    predicted = np.asarray(trajectory["goalkeeper_predicted_target_y_m"], dtype=np.float64)
    if contact_time_sec is not None:
        index = min(len(time) - 1, int(np.searchsorted(time, contact_time_sec, side="left")))
    else:
        keeper_plane_x = float(keeper[0, 0])
        crossing = np.flatnonzero(ball[:, 0] >= keeper_plane_x)
        index = (
            int(crossing[0])
            if crossing.size
            else int(np.argmin(np.abs(ball[:, 0] - keeper_plane_x)))
        )
    return abs(float(predicted[index]) - float(ball[index, 1]))


__all__ = ["run_parent_coverage_time_baseline"]
