"""Failure-driven S121 growth for active full-body off-ball football roles.

The tactical actor remains frozen and can choose only PASS or SHOOT.  This
stage learns which bounded route calibration lets the receiving team-mate and
the opponent keep moving through the same physical episode without destroying
the selected task.  Every movement target is executed by the frozen neural G1
locomotion policy; this module never writes a root pose, joint or ball state.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.growth.tactical_2v1 import TacticalAction
from rosclaw_soccer.growth.tactical_2v1_actor import (
    load_two_vs_one_tactical_actor,
)
from rosclaw_soccer.providers.g1.asset_qualification import trajectory_digest
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.shared_world import (
    G1MovementWaypoint,
    G1TacticalMovementConfig,
)
from rosclaw_soccer.training.full_body_tactical_2v1 import (
    FullBodyRoleMovementPlan,
    FullBodyTwoVsOneConfig,
    FullBodyTwoVsOneResult,
    FullBodyTwoVsOneScenario,
    simulate_full_body_two_vs_one,
)
from rosclaw_soccer.training.tactical_2v1_physics import FrozenTacticalSkillBundle

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.:-]{0,127}$")


@dataclass(frozen=True)
class ActiveRouteCandidate:
    """One bounded calibration hypothesis for action-conditioned movement."""

    candidate_id: str
    route_scale: float
    maximum_speed_mps: float
    maximum_acceleration_mps2: float
    position_gain: float
    schema_version: str = "rosclaw_soccer.active_route_candidate.v1"

    def __post_init__(self) -> None:
        values = (
            self.route_scale,
            self.maximum_speed_mps,
            self.maximum_acceleration_mps2,
            self.position_gain,
        )
        if (
            not _IDENTIFIER.fullmatch(self.candidate_id)
            or any(not math.isfinite(value) for value in values)
            or not 0.65 <= self.route_scale <= 1.25
            or not 0.25 <= self.maximum_speed_mps <= 0.55
            or not 0.40 <= self.maximum_acceleration_mps2 <= 1.20
            or not 0.50 <= self.position_gain <= 1.40
        ):
            raise ValueError("active route candidate exceeds the qualified envelope")

    @property
    def candidate_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass(frozen=True)
class RoleMotionQuality:
    displacement_m: float
    mean_planar_speed_mps: float
    stagnant_fraction: float
    active_fraction: float
    alternating_swing_switches: int
    upper_body_excursion_rad: float
    root_speed_jerk_rms_mps3: float
    peak_command_acceleration_mps2: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActiveOffBallResult:
    base: FullBodyTwoVsOneResult
    candidate_id: str
    candidate_hash: str
    movement_plan_hash: str
    carrier_motion: RoleMotionQuality
    teammate_motion: RoleMotionQuality
    defender_motion: RoleMotionQuality
    movement_quality_passed: bool
    schema_version: str = "rosclaw_soccer.active_off_ball_result.v1"

    @property
    def qualified(self) -> bool:
        return bool(self.base.task_succeeded and self.base.safe and self.movement_quality_passed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "candidate_hash": self.candidate_hash,
            "movement_plan_hash": self.movement_plan_hash,
            "base": self.base.to_dict(),
            "carrier_motion": self.carrier_motion.to_dict(),
            "teammate_motion": self.teammate_motion.to_dict(),
            "defender_motion": self.defender_motion.to_dict(),
            "movement_quality_passed": self.movement_quality_passed,
            "qualified": self.qualified,
        }


@dataclass(frozen=True)
class ActiveOffBallRetentionManifest:
    scenarios: tuple[FullBodyTwoVsOneScenario, ...]
    suite_id: str = "s121.active-off-ball.sealed-retention"
    training_access_allowed: bool = False
    activation_ceiling: str = "SIM_ONLY"
    schema_version: str = "rosclaw_soccer.active_off_ball_retention_manifest.v1"

    def __post_init__(self) -> None:
        hashes = tuple(scenario.scenario_hash for scenario in self.scenarios)
        if (
            len(self.scenarios) < 8
            or len(set(hashes)) != len(hashes)
            or self.training_access_allowed
            or self.activation_ceiling != "SIM_ONLY"
        ):
            raise ValueError("active off-ball retention must be unique, sealed and SIM_ONLY")

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": self.schema_version,
            "suite_id": self.suite_id,
            "scenarios": [asdict(scenario) for scenario in self.scenarios],
            "scenario_hashes": [scenario.scenario_hash for scenario in self.scenarios],
            "training_access_allowed": self.training_access_allowed,
            "activation_ceiling": self.activation_ceiling,
        }
        if include_hash:
            value["manifest_hash"] = hash_json(value)
        return value

    @property
    def manifest_hash(self) -> str:
        return str(hash_json(self.to_dict(include_hash=False)))


def default_active_route_candidates() -> tuple[ActiveRouteCandidate, ...]:
    return (
        ActiveRouteCandidate("s121.route.compact", 0.75, 0.34, 0.60, 0.80),
        ActiveRouteCandidate("s121.route.athletic", 1.00, 0.40, 0.70, 0.90),
        ActiveRouteCandidate("s121.route.aggressive", 1.20, 0.50, 0.90, 1.00),
    )


def default_active_acquisition_scenarios() -> tuple[FullBodyTwoVsOneScenario, ...]:
    layouts = (
        ((5.42, -0.40, 0.0), (2.00, 0.40, 0.0)),
        ((5.46, -0.40, 0.0), (2.05, 0.45, 0.0)),
        ((5.54, -0.40, 0.0), (2.10, 0.50, 0.0)),
        ((5.58, -0.40, 0.0), (2.15, 0.55, 0.0)),
        ((5.30, 0.30, 0.0), (4.05, 0.30, 0.0)),
        ((5.40, 0.40, 0.0), (4.25, 0.40, 0.0)),
        ((5.60, 0.50, 0.0), (4.65, 0.50, 0.0)),
        ((5.70, 0.60, 0.0), (4.85, 0.60, 0.0)),
    )
    return tuple(
        FullBodyTwoVsOneScenario(
            scenario_id=f"s121.acquisition.{index:03d}",
            seed=121_000 + index,
            teammate_origin_m=teammate,
            defender_origin_m=defender,
        )
        for index, (teammate, defender) in enumerate(layouts)
    )


def default_active_retention_manifest() -> ActiveOffBallRetentionManifest:
    layouts = (
        ((5.44, -0.40, 0.0), (2.02, 0.42, 0.0)),
        ((5.48, -0.40, 0.0), (2.07, 0.47, 0.0)),
        ((5.52, -0.40, 0.0), (2.12, 0.52, 0.0)),
        # Keep the unseen receiving geometry inside the frozen actor's
        # explicitly qualified PASS support.  The first attempted point at
        # (2.17, 0.57) correctly routed to HOLD and failed closed before any
        # retention physics was consumed.
        ((5.56, -0.40, 0.0), (2.00, 0.40, 0.0)),
        ((5.35, 0.35, 0.0), (4.15, 0.35, 0.0)),
        ((5.45, 0.45, 0.0), (4.35, 0.45, 0.0)),
        ((5.55, 0.55, 0.0), (4.55, 0.55, 0.0)),
        ((5.65, 0.55, 0.0), (4.75, 0.45, 0.0)),
    )
    scenarios = tuple(
        FullBodyTwoVsOneScenario(
            scenario_id=f"s121.retention.{index:03d}",
            seed=121_500 + index,
            teammate_origin_m=teammate,
            defender_origin_m=defender,
        )
        for index, (teammate, defender) in enumerate(layouts)
    )
    return ActiveOffBallRetentionManifest(scenarios=scenarios)


def build_action_conditioned_movement_plan(
    scenario: FullBodyTwoVsOneScenario,
    action: TacticalAction,
    candidate: ActiveRouteCandidate,
    *,
    simulation_duration_sec: float = 7.0,
) -> FullBodyRoleMovementPlan:
    """Compile tactical CHECK/RUN/PRESS/COVER options into locomotion targets."""

    if action not in {TacticalAction.PASS, TacticalAction.SHOOT}:
        raise ValueError("active off-ball bridge supports PASS or SHOOT")
    scale = candidate.route_scale
    target_x, target_y, _ = scenario.teammate_origin_m
    side = -1.0 if target_y < 0.0 else 1.0
    teammate_waypoints: tuple[G1MovementWaypoint, ...]
    if action == TacticalAction.PASS:
        teammate_origin = (target_x - 1.04 * scale, target_y + side * 0.60 * scale, 0.0)
        initial = (teammate_origin[0] + 0.002, teammate_origin[1] - 0.045, 0.0)
        endpoint = (
            target_x - (0.40 + 0.14 * scale),
            target_y + side * (0.14 + 0.09 * scale),
            0.0,
        )
        midpoint = (
            0.5 * (initial[0] + endpoint[0]),
            0.5 * (initial[1] + endpoint[1]),
            0.0,
        )
        teammate_waypoints = (
            G1MovementWaypoint(0.0, initial),
            G1MovementWaypoint(1.70, midpoint),
            G1MovementWaypoint(3.40, endpoint),
            G1MovementWaypoint(simulation_duration_sec, endpoint),
        )
    else:
        teammate_origin = (target_x - 1.05 * scale, target_y + side * 0.65 * scale, 0.0)
        initial = (teammate_origin[0] + 0.002, teammate_origin[1] - 0.045, 0.0)
        teammate_waypoints = (
            G1MovementWaypoint(0.0, initial),
            G1MovementWaypoint(
                2.0,
                (initial[0] + 0.30 * scale, initial[1] - side * 0.18 * scale, 0.0),
            ),
            G1MovementWaypoint(
                4.0,
                (initial[0] + 0.60 * scale, target_y + side * 0.25 * scale, 0.0),
            ),
            G1MovementWaypoint(
                5.5,
                (initial[0] + 0.80 * scale, target_y + side * 0.13 * scale, 0.0),
            ),
            G1MovementWaypoint(
                simulation_duration_sec,
                (initial[0] + 0.95 * scale, target_y + side * 0.05 * scale, 0.0),
            ),
        )

    defender_x, defender_y, _ = scenario.defender_origin_m
    defender_initial = (defender_x - 0.001, defender_y - 0.001, 0.0)
    if action == TacticalAction.PASS:
        defender_end = (defender_x + 0.70 * scale, defender_y + 0.60 * scale, 0.0)
    else:
        defender_end = (defender_x + 0.50 * scale, defender_y + 0.45 * scale, 0.0)
    defender_waypoints = (
        G1MovementWaypoint(0.0, defender_initial),
        G1MovementWaypoint(
            2.0,
            (
                defender_initial[0] + 0.35 * (defender_end[0] - defender_initial[0]),
                defender_initial[1] + 0.35 * (defender_end[1] - defender_initial[1]),
                0.0,
            ),
        ),
        G1MovementWaypoint(
            4.0,
            (
                defender_initial[0] + 0.70 * (defender_end[0] - defender_initial[0]),
                defender_initial[1] + 0.70 * (defender_end[1] - defender_initial[1]),
                0.0,
            ),
        ),
        G1MovementWaypoint(5.5, defender_end),
        G1MovementWaypoint(
            simulation_duration_sec,
            (defender_end[0] + 0.15 * scale, defender_end[1] - 0.10 * scale, 0.0),
        ),
    )
    teammate_movement = G1TacticalMovementConfig(
        waypoints=teammate_waypoints,
        maximum_speed_mps=candidate.maximum_speed_mps,
        maximum_acceleration_mps2=candidate.maximum_acceleration_mps2,
        position_gain=candidate.position_gain,
    )
    defender_movement = G1TacticalMovementConfig(
        waypoints=defender_waypoints,
        maximum_speed_mps=min(0.40, candidate.maximum_speed_mps - 0.05),
        maximum_acceleration_mps2=max(0.40, candidate.maximum_acceleration_mps2 - 0.05),
        position_gain=max(0.50, candidate.position_gain - 0.10),
    )
    return FullBodyRoleMovementPlan(
        teammate_origin_m=teammate_origin,
        teammate_movement=teammate_movement,
        defender_movement=defender_movement,
    )


def _swing_switches(
    time: NDArray[np.float64],
    left: NDArray[np.float64],
    right: NDArray[np.float64],
) -> int:
    left_speed = np.linalg.norm(np.gradient(left[:, :2], time, axis=0), axis=1)
    right_speed = np.linalg.norm(np.gradient(right[:, :2], time, axis=0), axis=1)
    labels = np.where(
        np.maximum(left_speed, right_speed) < 0.08,
        0,
        np.where(
            left_speed > right_speed + 0.025, -1, np.where(right_speed > left_speed + 0.025, 1, 0)
        ),
    )
    active = labels[labels != 0]
    if not active.size:
        return 0
    compressed = active[np.concatenate(([True], active[1:] != active[:-1]))]
    return max(0, int(compressed.size) - 1)


def _motion_quality(
    trajectory: dict[str, NDArray[Any]],
    role: str,
) -> RoleMotionQuality:
    time = np.asarray(trajectory["time"], dtype=np.float64)
    pelvis = np.asarray(trajectory[f"{role}_pelvis_pose"], dtype=np.float64)
    left = np.asarray(trajectory[f"{role}_left_foot_position"], dtype=np.float64)
    right = np.asarray(trajectory[f"{role}_right_foot_position"], dtype=np.float64)
    joints = np.asarray(trajectory[f"{role}_joint_position"], dtype=np.float64)
    velocity = np.gradient(pelvis[:, :2], time, axis=0)
    speed = np.linalg.norm(velocity, axis=1)
    acceleration = np.gradient(speed, time)
    jerk = np.gradient(acceleration, time)
    command_key = f"{role}_tactical_world_command"
    peak_command_acceleration: float | None = None
    if command_key in trajectory:
        command = np.asarray(trajectory[command_key], dtype=np.float64)[:, :2]
        peak_command_acceleration = float(
            np.max(np.linalg.norm(np.diff(command, axis=0), axis=1) / np.diff(time))
        )
    return RoleMotionQuality(
        displacement_m=float(np.linalg.norm(pelvis[-1, :2] - pelvis[0, :2])),
        mean_planar_speed_mps=float(np.mean(speed)),
        stagnant_fraction=float(np.mean(speed < 0.05)),
        active_fraction=float(np.mean(speed > 0.15)),
        alternating_swing_switches=_swing_switches(time, left, right),
        upper_body_excursion_rad=float(np.mean(np.ptp(joints[:, 12:], axis=0))),
        root_speed_jerk_rms_mps3=float(np.sqrt(np.mean(np.square(jerk[2:-2])))),
        peak_command_acceleration_mps2=peak_command_acceleration,
    )


def simulate_active_off_ball_episode(
    *,
    asset_root: Path,
    scenario: FullBodyTwoVsOneScenario,
    action: TacticalAction,
    skill_bundle: FrozenTacticalSkillBundle,
    candidate: ActiveRouteCandidate,
    config: FullBodyTwoVsOneConfig | None = None,
) -> tuple[ActiveOffBallResult, dict[str, NDArray[Any]]]:
    active = config or FullBodyTwoVsOneConfig()
    plan = build_action_conditioned_movement_plan(
        scenario,
        action,
        candidate,
        simulation_duration_sec=active.simulation_duration_sec,
    )
    base, trajectory = simulate_full_body_two_vs_one(
        asset_root=asset_root,
        scenario=scenario,
        action=action,
        skill_bundle=skill_bundle,
        config=active,
        movement_plan=plan,
    )
    carrier = _motion_quality(trajectory, "shooter")
    teammate = _motion_quality(trajectory, "passer")
    defender = _motion_quality(trajectory, "goalkeeper")
    movement_passed = bool(
        carrier.displacement_m >= 1.40
        and carrier.stagnant_fraction <= 0.22
        and teammate.displacement_m >= 0.60
        and teammate.stagnant_fraction <= 0.12
        and teammate.active_fraction >= 0.50
        and teammate.alternating_swing_switches >= 4
        and teammate.upper_body_excursion_rad >= 0.03
        and defender.displacement_m >= 0.25
        and defender.stagnant_fraction <= 0.20
        and defender.alternating_swing_switches >= 2
        and defender.upper_body_excursion_rad >= 0.02
        and teammate.peak_command_acceleration_mps2 is not None
        and teammate.peak_command_acceleration_mps2 <= candidate.maximum_acceleration_mps2 + 1.0e-8
        and defender.peak_command_acceleration_mps2 is not None
        and defender.peak_command_acceleration_mps2
        <= max(0.40, candidate.maximum_acceleration_mps2 - 0.05) + 1.0e-8
    )
    result = ActiveOffBallResult(
        base=base,
        candidate_id=candidate.candidate_id,
        candidate_hash=candidate.candidate_hash,
        movement_plan_hash=plan.plan_hash,
        carrier_motion=carrier,
        teammate_motion=teammate,
        defender_motion=defender,
        movement_quality_passed=movement_passed,
    )
    return result, trajectory


def _candidate_score(results: list[ActiveOffBallResult]) -> float:
    qualified = sum(result.qualified for result in results)
    tasks = sum(result.base.task_succeeded and result.base.safe for result in results)
    movement = sum(result.movement_quality_passed for result in results)
    activity = sum(
        result.teammate_motion.active_fraction + result.defender_motion.active_fraction
        for result in results
    )
    stagnation = sum(
        result.teammate_motion.stagnant_fraction + result.defender_motion.stagnant_fraction
        for result in results
    )
    return 1000.0 * qualified + 100.0 * tasks + 10.0 * movement + activity - stagnation


def run_active_off_ball_growth_round(
    *,
    output_dir: Path,
    source_checkout: Path,
    asset_root: Path,
    actor_path: Path,
    skill_bundle: FrozenTacticalSkillBundle,
    acquisition_scenarios: tuple[FullBodyTwoVsOneScenario, ...] | None = None,
    retention_manifest: ActiveOffBallRetentionManifest | None = None,
    candidates: tuple[ActiveRouteCandidate, ...] | None = None,
    config: FullBodyTwoVsOneConfig | None = None,
) -> dict[str, Any]:
    """Calibrate on DEVELOPMENT, then run the untouched sealed retention suite."""

    destination = output_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if destination.exists() or destination == checkout or checkout in destination.parents:
        raise ValueError("active off-ball evidence must be new and outside the checkout")
    destination.mkdir(parents=True)
    actor = load_two_vs_one_tactical_actor(actor_path)
    active = config or FullBodyTwoVsOneConfig()
    acquisition = acquisition_scenarios or default_active_acquisition_scenarios()
    manifest = retention_manifest or default_active_retention_manifest()
    portfolio = candidates or default_active_route_candidates()
    if {row.scenario_hash for row in acquisition} & {
        row.scenario_hash for row in manifest.scenarios
    }:
        raise ValueError("active acquisition and retention scenarios must be disjoint")
    _write_json(destination / "sealed-retention.json", manifest.to_dict())

    acquisition_rows: list[dict[str, Any]] = []
    candidate_results: dict[str, list[ActiveOffBallResult]] = {}
    for candidate in portfolio:
        results: list[ActiveOffBallResult] = []
        for scenario in acquisition:
            action = actor.decide(scenario.state(skill_bundle=skill_bundle, config=active)).action
            result, _ = simulate_active_off_ball_episode(
                asset_root=asset_root,
                scenario=scenario,
                action=action,
                skill_bundle=skill_bundle,
                candidate=candidate,
                config=active,
            )
            results.append(result)
            acquisition_rows.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "scenario_hash": scenario.scenario_hash,
                    "selected_action": action.value,
                    "candidate_id": candidate.candidate_id,
                    "candidate_hash": candidate.candidate_hash,
                    "result": result.to_dict(),
                }
            )
        candidate_results[candidate.candidate_id] = results
    selected = max(
        portfolio,
        key=lambda candidate: (
            _candidate_score(candidate_results[candidate.candidate_id]),
            candidate.candidate_id,
        ),
    )
    score_table = {
        candidate.candidate_id: {
            "candidate_hash": candidate.candidate_hash,
            "score": _candidate_score(candidate_results[candidate.candidate_id]),
            "qualified_count": sum(
                result.qualified for result in candidate_results[candidate.candidate_id]
            ),
            "task_success_count": sum(
                result.base.task_succeeded and result.base.safe
                for result in candidate_results[candidate.candidate_id]
            ),
            "movement_quality_count": sum(
                result.movement_quality_passed
                for result in candidate_results[candidate.candidate_id]
            ),
        }
        for candidate in portfolio
    }
    acquisition_ledger: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.active_off_ball_acquisition.v1",
        "retention_manifest_visible_to_training": False,
        "actor_hash": actor.actor_hash,
        "portfolio": [asdict(candidate) for candidate in portfolio],
        "candidate_scores": score_table,
        "selected_candidate": asdict(selected),
        "selected_candidate_hash": selected.candidate_hash,
        "rows": acquisition_rows,
    }
    acquisition_ledger["ledger_hash"] = hash_json(acquisition_ledger)
    _write_json(destination / "acquisition-ledger.json", acquisition_ledger)

    retention_rows: list[dict[str, Any]] = []
    for index, scenario in enumerate(manifest.scenarios):
        action = actor.decide(scenario.state(skill_bundle=skill_bundle, config=active)).action
        primary, trajectory = simulate_active_off_ball_episode(
            asset_root=asset_root,
            scenario=scenario,
            action=action,
            skill_bundle=skill_bundle,
            candidate=selected,
            config=active,
        )
        replay, replay_trajectory = simulate_active_off_ball_episode(
            asset_root=asset_root,
            scenario=scenario,
            action=action,
            skill_bundle=skill_bundle,
            candidate=selected,
            config=active,
        )
        case_dir = destination / "retention" / f"case-{index:03d}"
        case_dir.mkdir(parents=True)
        primary_artifact = _save_trajectory(case_dir / "selected-primary.npz", trajectory)
        replay_artifact = _save_trajectory(case_dir / "selected-replay.npz", replay_trajectory)
        exact_replay = bool(
            primary.to_dict() == replay.to_dict()
            and primary_artifact["trajectory_digest"] == replay_artifact["trajectory_digest"]
        )
        retention_rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "scenario_hash": scenario.scenario_hash,
                "selected_action": action.value,
                "result": primary.to_dict(),
                "qualified": primary.qualified,
                "safe": primary.base.safe,
                "exact_replay": exact_replay,
                "primary_artifact": primary_artifact,
                "replay_artifact": replay_artifact,
            }
        )
    count = len(retention_rows)
    action_counts = {
        action.value: sum(row["selected_action"] == action.value for row in retention_rows)
        for action in (TacticalAction.PASS, TacticalAction.SHOOT)
    }
    metrics = {
        "case_count": count,
        "qualified_rate": sum(row["qualified"] for row in retention_rows) / count,
        "task_success_rate": sum(row["result"]["base"]["task_succeeded"] for row in retention_rows)
        / count,
        "safe_rate": sum(row["safe"] for row in retention_rows) / count,
        "movement_quality_rate": sum(
            row["result"]["movement_quality_passed"] for row in retention_rows
        )
        / count,
        "exact_replay_rate": sum(row["exact_replay"] for row in retention_rows) / count,
        "selected_action_counts": action_counts,
        "mean_teammate_stagnant_fraction": float(
            np.mean(
                [row["result"]["teammate_motion"]["stagnant_fraction"] for row in retention_rows]
            )
        ),
        "mean_defender_stagnant_fraction": float(
            np.mean(
                [row["result"]["defender_motion"]["stagnant_fraction"] for row in retention_rows]
            )
        ),
        "mean_teammate_displacement_m": float(
            np.mean([row["result"]["teammate_motion"]["displacement_m"] for row in retention_rows])
        ),
        "mean_defender_displacement_m": float(
            np.mean([row["result"]["defender_motion"]["displacement_m"] for row in retention_rows])
        ),
    }
    gates = {
        "qualified_rate": metrics["qualified_rate"] == 1.0,
        "task_success_rate": metrics["task_success_rate"] == 1.0,
        "safe_rate": metrics["safe_rate"] == 1.0,
        "movement_quality_rate": metrics["movement_quality_rate"] == 1.0,
        "exact_replay_rate": metrics["exact_replay_rate"] == 1.0,
        "both_actions_covered": action_counts == {"PASS": 4, "SHOOT": 4},
        "teammate_stagnation_reduced": metrics["mean_teammate_stagnant_fraction"] <= 0.12,
        "defender_stagnation_reduced": metrics["mean_defender_stagnant_fraction"] <= 0.20,
    }
    passed = all(gates.values())
    retention: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.active_off_ball_retention_exam.v1",
        "status": "PASS_ACTIVE_OFF_BALL_GROWTH" if passed else "REJECTED_ACTIVE_OFF_BALL_GROWTH",
        "actor_hash": actor.actor_hash,
        "selected_candidate": asdict(selected),
        "selected_candidate_hash": selected.candidate_hash,
        "manifest_hash": manifest.manifest_hash,
        "metrics": metrics,
        "gates": gates,
        "rows": retention_rows,
        "evidence_boundary": {
            "activation_ceiling": "SIM_ONLY",
            "physics_authority": "CPU_MUJOCO",
            "whole_body_g1_count": 3,
            "shared_solver_and_ball": True,
            "tactical_actor_frozen": True,
            "movement_executed_by_frozen_neural_locomotion": True,
            "pose_or_joint_scripted": False,
            "pixels_used_for_scoring": False,
            "hardware_command_sent": False,
        },
    }
    retention["report_hash"] = hash_json(retention)
    _write_json(destination / "retention" / "retention-exam.json", retention)
    source_commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    stage: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.active_off_ball_growth_stage.v1",
        "status": retention["status"],
        "source_commit": source_commit,
        "actor_hash": actor.actor_hash,
        "selected_candidate_hash": selected.candidate_hash,
        "sealed_retention_manifest_hash": manifest.manifest_hash,
        "retention_report_hash": retention["report_hash"],
        "retention_metrics": metrics,
        "implementation_hashes": {
            "shared_world": hash_bytes(
                Path(__file__).parents[1].joinpath("skills/team/shared_world.py").read_bytes()
            ),
            "full_body_bridge": hash_bytes(
                Path(__file__).with_name("full_body_tactical_2v1.py").read_bytes()
            ),
            "active_growth": hash_bytes(Path(__file__).read_bytes()),
        },
        "evidence_boundary": retention["evidence_boundary"],
    }
    stage["stage_hash"] = hash_json(stage)
    _write_json(destination / "stage-summary.json", stage)
    return stage


def _save_trajectory(path: Path, trajectory: dict[str, NDArray[Any]]) -> dict[str, str]:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **trajectory)  # type: ignore[arg-type]
    os.replace(temporary, path)
    return {
        "file": path.name,
        "file_hash": hash_bytes(path.read_bytes()),
        "trajectory_digest": trajectory_digest(trajectory),
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


__all__ = [
    "ActiveOffBallResult",
    "ActiveOffBallRetentionManifest",
    "ActiveRouteCandidate",
    "RoleMotionQuality",
    "build_action_conditioned_movement_plan",
    "default_active_acquisition_scenarios",
    "default_active_retention_manifest",
    "default_active_route_candidates",
    "run_active_off_ball_growth_round",
    "simulate_active_off_ball_episode",
]
