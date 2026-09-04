"""Failure-driven growth for causal red/blue role autonomy in full-body 2v1.

This stage keeps the released tactical and route actors frozen.  It searches
only a bounded high-level target-correction envelope, then evaluates the
selected candidate on new CPU-MuJoCo episodes.  Every candidate is SIM-only;
none can write poses, joints, torques, football state, ROS, or hardware.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.growth.tactical_2v1 import TacticalAction
from rosclaw_soccer.growth.temporal_route_actor import RouteActor
from rosclaw_soccer.providers.g1.asset_qualification import trajectory_digest
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.shared_world import G1RoleAutonomyConfig
from rosclaw_soccer.training.active_off_ball_growth import RoleMotionQuality, _motion_quality
from rosclaw_soccer.training.full_body_tactical_2v1 import (
    FullBodyAutonomousRoleMovementPlan,
    FullBodyTwoVsOneConfig,
    FullBodyTwoVsOneResult,
    FullBodyTwoVsOneScenario,
    simulate_full_body_two_vs_one,
)
from rosclaw_soccer.training.reactive_route_growth import build_reactive_movement_plan
from rosclaw_soccer.training.tactical_2v1_physics import FrozenTacticalSkillBundle


@dataclass(frozen=True)
class AutonomousRoleCandidate:
    candidate_id: str
    maximum_target_shift_m: float
    ball_prediction_horizon_sec: float
    lane_width_m: float
    moving_ball_threshold_mps: float = 0.24
    decision_period_sec: float = 0.10
    intent_hysteresis_sec: float = 0.20
    schema_version: str = "rosclaw_soccer.autonomous_role_candidate.v1"

    def __post_init__(self) -> None:
        if not self.candidate_id.startswith("s198.role."):
            raise ValueError("autonomous role candidate identity is invalid")
        # Reuse the runtime contract as the single source of envelope truth.
        G1RoleAutonomyConfig(
            role="teammate",
            team_id="red",
            maximum_target_shift_m=self.maximum_target_shift_m,
            ball_prediction_horizon_sec=self.ball_prediction_horizon_sec,
            lane_width_m=self.lane_width_m,
            moving_ball_threshold_mps=self.moving_ball_threshold_mps,
            decision_period_sec=self.decision_period_sec,
            intent_hysteresis_sec=self.intent_hysteresis_sec,
        )

    @property
    def candidate_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass(frozen=True)
class AutonomousRoleCase:
    case_id: str
    scenario: FullBodyTwoVsOneScenario
    action: TacticalAction
    schema_version: str = "rosclaw_soccer.autonomous_role_case.v1"

    def __post_init__(self) -> None:
        if not self.case_id.startswith("s198.") or self.action not in {
            TacticalAction.PASS,
            TacticalAction.SHOOT,
        }:
            raise ValueError("autonomous role case is invalid")

    @property
    def case_hash(self) -> str:
        value = asdict(self)
        value["action"] = self.action.value
        return str(hash_json(value))


@dataclass(frozen=True)
class RoleIntentQuality:
    distinct_intent_count: int
    switch_count: int
    dynamic_target_fraction: float
    peak_target_shift_m: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.distinct_intent_count, bool)
            or isinstance(self.switch_count, bool)
            or self.distinct_intent_count < 0
            or self.switch_count < 0
            or not math.isfinite(self.dynamic_target_fraction)
            or not 0.0 <= self.dynamic_target_fraction <= 1.0
            or not math.isfinite(self.peak_target_shift_m)
            or self.peak_target_shift_m < 0.0
        ):
            raise ValueError("role intent quality is invalid")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class AutonomousRoleEpisodeResult:
    base: FullBodyTwoVsOneResult
    candidate_id: str
    candidate_hash: str
    movement_plan_hash: str
    carrier_motion: RoleMotionQuality
    teammate_motion: RoleMotionQuality
    defender_motion: RoleMotionQuality
    teammate_intent: RoleIntentQuality
    defender_intent: RoleIntentQuality
    route_actor_accepted: bool
    movement_quality_passed: bool
    intent_quality_passed: bool
    schema_version: str = "rosclaw_soccer.autonomous_role_episode_result.v1"

    @property
    def qualified(self) -> bool:
        return bool(
            self.base.task_succeeded
            and self.base.safe
            and self.route_actor_accepted
            and self.movement_quality_passed
            and self.intent_quality_passed
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "base": self.base.to_dict(),
            "candidate_id": self.candidate_id,
            "candidate_hash": self.candidate_hash,
            "movement_plan_hash": self.movement_plan_hash,
            "carrier_motion": self.carrier_motion.to_dict(),
            "teammate_motion": self.teammate_motion.to_dict(),
            "defender_motion": self.defender_motion.to_dict(),
            "teammate_intent": self.teammate_intent.to_dict(),
            "defender_intent": self.defender_intent.to_dict(),
            "route_actor_accepted": self.route_actor_accepted,
            "movement_quality_passed": self.movement_quality_passed,
            "intent_quality_passed": self.intent_quality_passed,
            "qualified": self.qualified,
        }


def default_autonomous_role_candidates() -> tuple[AutonomousRoleCandidate, ...]:
    """Ordered stability/plasticity candidates, including one hard failure."""

    return (
        AutonomousRoleCandidate("s198.role.stable", 0.08, 0.08, 0.16),
        AutonomousRoleCandidate("s198.role.balanced", 0.12, 0.08, 0.16),
        AutonomousRoleCandidate("s198.role.plastic", 0.22, 0.18, 0.24),
    )


def _scenario(
    case_id: str,
    *,
    seed: int,
    teammate: tuple[float, float, float],
    defender: tuple[float, float, float],
) -> FullBodyTwoVsOneScenario:
    return FullBodyTwoVsOneScenario(
        scenario_id=case_id,
        seed=seed,
        teammate_origin_m=teammate,
        defender_origin_m=defender,
    )


def default_autonomous_development_cases() -> tuple[AutonomousRoleCase, ...]:
    return (
        AutonomousRoleCase(
            "s198.development.000",
            _scenario(
                "s198.development.scene.000",
                seed=198_000,
                teammate=(5.45, -0.40, 0.0),
                defender=(2.04, 0.44, 0.0),
            ),
            TacticalAction.PASS,
        ),
        AutonomousRoleCase(
            "s198.development.001",
            _scenario(
                "s198.development.scene.001",
                seed=198_001,
                teammate=(5.53, -0.40, 0.0),
                defender=(2.12, 0.52, 0.0),
            ),
            TacticalAction.PASS,
        ),
        AutonomousRoleCase(
            "s198.development.002",
            _scenario(
                "s198.development.scene.002",
                seed=198_002,
                teammate=(5.36, 0.36, 0.0),
                defender=(4.16, 0.36, 0.0),
            ),
            TacticalAction.SHOOT,
        ),
        AutonomousRoleCase(
            "s198.development.003",
            _scenario(
                "s198.development.scene.003",
                seed=198_003,
                teammate=(5.56, 0.56, 0.0),
                defender=(4.56, 0.46, 0.0),
            ),
            TacticalAction.SHOOT,
        ),
    )


def default_autonomous_retention_cases() -> tuple[AutonomousRoleCase, ...]:
    """New cells kept out of candidate selection and development retries."""

    return (
        AutonomousRoleCase(
            "s198.retention.000",
            _scenario(
                "s198.retention.scene.000",
                seed=198_500,
                teammate=(5.48, -0.40, 0.0),
                defender=(2.07, 0.47, 0.0),
            ),
            TacticalAction.PASS,
        ),
        AutonomousRoleCase(
            "s198.retention.001",
            _scenario(
                "s198.retention.scene.001",
                seed=198_501,
                teammate=(5.56, -0.40, 0.0),
                defender=(2.00, 0.40, 0.0),
            ),
            TacticalAction.PASS,
        ),
        AutonomousRoleCase(
            "s198.retention.002",
            _scenario(
                "s198.retention.scene.002",
                seed=198_502,
                teammate=(5.46, 0.46, 0.0),
                defender=(4.36, 0.46, 0.0),
            ),
            TacticalAction.SHOOT,
        ),
        AutonomousRoleCase(
            "s198.retention.003",
            _scenario(
                "s198.retention.scene.003",
                seed=198_503,
                teammate=(5.64, 0.54, 0.0),
                defender=(4.74, 0.44, 0.0),
            ),
            TacticalAction.SHOOT,
        ),
    )


def build_autonomous_role_plan(
    *,
    case: AutonomousRoleCase,
    candidate: AutonomousRoleCandidate,
    actor_path: Path,
    actor: RouteActor,
    simulation_duration_sec: float = 7.0,
) -> FullBodyAutonomousRoleMovementPlan:
    reactive = build_reactive_movement_plan(
        scenario=case.scenario,
        action=case.action,
        actor_path=actor_path,
        actor=actor,
        simulation_duration_sec=simulation_duration_sec,
    )

    def autonomy(role: str, team_id: str) -> G1RoleAutonomyConfig:
        return G1RoleAutonomyConfig(
            role=role,
            team_id=team_id,
            decision_period_sec=candidate.decision_period_sec,
            intent_hysteresis_sec=candidate.intent_hysteresis_sec,
            moving_ball_threshold_mps=candidate.moving_ball_threshold_mps,
            maximum_target_shift_m=candidate.maximum_target_shift_m,
            ball_prediction_horizon_sec=candidate.ball_prediction_horizon_sec,
            lane_width_m=candidate.lane_width_m,
        )

    return FullBodyAutonomousRoleMovementPlan(
        teammate_origin_m=reactive.teammate_origin_m,
        defender_origin_m=reactive.defender_origin_m,
        teammate_movement=reactive.teammate_movement,
        defender_movement=reactive.defender_movement,
        teammate_autonomy=autonomy("teammate", "red"),
        defender_autonomy=autonomy("defender", "blue"),
    )


def _intent_quality(trajectory: dict[str, NDArray[Any]], role: str) -> RoleIntentQuality:
    codes = np.asarray(trajectory[f"{role}_role_intent_code"], dtype=np.int64)
    shifts = np.asarray(trajectory[f"{role}_role_intent_target_shift"], dtype=np.float64)
    switches = np.asarray(trajectory[f"{role}_role_intent_switch_count"], dtype=np.int64)
    norms = np.linalg.norm(shifts, axis=1)
    return RoleIntentQuality(
        distinct_intent_count=len(set(int(value) for value in codes if value > 0)),
        switch_count=int(switches[-1]),
        dynamic_target_fraction=float(np.mean(norms > 1.0e-4)),
        peak_target_shift_m=float(np.max(norms)),
    )


def simulate_autonomous_role_episode(
    *,
    asset_root: Path,
    case: AutonomousRoleCase,
    candidate: AutonomousRoleCandidate,
    actor_path: Path,
    actor: RouteActor,
    skill_bundle: FrozenTacticalSkillBundle,
    config: FullBodyTwoVsOneConfig | None = None,
) -> tuple[AutonomousRoleEpisodeResult, dict[str, NDArray[Any]]]:
    active = config or FullBodyTwoVsOneConfig()
    plan = build_autonomous_role_plan(
        case=case,
        candidate=candidate,
        actor_path=actor_path,
        actor=actor,
        simulation_duration_sec=active.simulation_duration_sec,
    )
    base, trajectory = simulate_full_body_two_vs_one(
        asset_root=asset_root,
        scenario=case.scenario,
        action=case.action,
        skill_bundle=skill_bundle,
        config=active,
        autonomous_movement_plan=plan,
    )
    carrier = _motion_quality(trajectory, "shooter")
    teammate = _motion_quality(trajectory, "passer")
    defender = _motion_quality(trajectory, "goalkeeper")
    teammate_intent = _intent_quality(trajectory, "passer")
    defender_intent = _intent_quality(trajectory, "goalkeeper")
    route_accepted = bool(
        np.all(np.asarray(trajectory["passer_reactive_route_accepted"], dtype=np.bool_))
        and np.all(np.asarray(trajectory["goalkeeper_reactive_route_accepted"], dtype=np.bool_))
    )
    movement_passed = bool(
        carrier.active_fraction >= 0.40
        and teammate.active_fraction >= 0.75
        and defender.active_fraction >= 0.35
        and teammate.alternating_swing_switches >= 4
        and defender.alternating_swing_switches >= 2
        and teammate.stagnant_fraction <= 0.18
        and defender.stagnant_fraction <= 0.28
    )
    intent_passed = bool(
        teammate_intent.distinct_intent_count >= 2
        and teammate_intent.switch_count >= 2
        and teammate_intent.dynamic_target_fraction >= 0.90
        and defender_intent.distinct_intent_count >= 2
        and defender_intent.switch_count >= 2
        and defender_intent.dynamic_target_fraction >= 0.90
        and teammate_intent.peak_target_shift_m <= candidate.maximum_target_shift_m + 1.0e-9
        and defender_intent.peak_target_shift_m <= candidate.maximum_target_shift_m + 1.0e-9
    )
    return (
        AutonomousRoleEpisodeResult(
            base=base,
            candidate_id=candidate.candidate_id,
            candidate_hash=candidate.candidate_hash,
            movement_plan_hash=plan.plan_hash,
            carrier_motion=carrier,
            teammate_motion=teammate,
            defender_motion=defender,
            teammate_intent=teammate_intent,
            defender_intent=defender_intent,
            route_actor_accepted=route_accepted,
            movement_quality_passed=movement_passed,
            intent_quality_passed=intent_passed,
        ),
        trajectory,
    )


def _candidate_score(rows: list[AutonomousRoleEpisodeResult]) -> tuple[float, ...]:
    qualified = sum(row.qualified for row in rows)
    task = sum(row.base.task_succeeded for row in rows)
    safe = sum(row.base.safe for row in rows)
    progress = float(np.mean([row.base.possession_progress for row in rows]))
    activity = float(
        np.mean(
            [
                0.5 * (row.teammate_motion.active_fraction + row.defender_motion.active_fraction)
                for row in rows
            ]
        )
    )
    pass_errors = [
        row.base.teammate_foot_reception_distance_m
        for row in rows
        if row.base.action is TacticalAction.PASS
        and row.base.teammate_foot_reception_distance_m is not None
    ]
    precision = -float(np.mean(pass_errors)) if pass_errors else -1.0
    # Lexicographic selection: football and safety cannot be bought by motion.
    return float(qualified), float(task), float(safe), progress, precision, activity


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


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


def _git_head(path: Path) -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def run_autonomous_role_growth(
    *,
    asset_root: Path,
    actor_path: Path,
    actor: RouteActor,
    skill_bundle: FrozenTacticalSkillBundle,
    source_stage_path: Path,
    output_dir: Path,
    source_checkout: Path,
    candidates: tuple[AutonomousRoleCandidate, ...] | None = None,
) -> dict[str, Any]:
    """Select on development, then consume each sealed retention cell twice."""

    destination = output_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if destination.exists() or destination == checkout or checkout in destination.parents:
        raise ValueError("autonomous role evidence must use a new external directory")
    stage_path = source_stage_path.expanduser().resolve()
    source_stage = json.loads(stage_path.read_text(encoding="utf-8"))
    source_expected = source_stage.pop("stage_hash", None)
    if (
        source_stage.get("status") != "PASS_REACTIVE_MULTI_AGENT_GROWTH"
        or source_expected != hash_json(source_stage)
        or source_stage.get("actor_hash") != actor.actor_hash
    ):
        raise ValueError("released reactive-route parent is invalid")
    source_stage["stage_hash"] = source_expected
    selected_candidates = candidates or default_autonomous_role_candidates()
    if len(selected_candidates) < 3 or len(
        {item.candidate_hash for item in selected_candidates}
    ) != len(selected_candidates):
        raise ValueError("autonomous role selection needs three unique candidates")
    retention_cases = default_autonomous_retention_cases()
    destination.mkdir(parents=True)
    manifest: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.autonomous_role_retention_manifest.v1",
        "suite_id": "s198.autonomous-role.unseen-retention",
        "cases": [
            {**asdict(case), "action": case.action.value, "case_hash": case.case_hash}
            for case in retention_cases
        ],
        "training_access_allowed": False,
        "activation_ceiling": "SIM_ONLY",
    }
    manifest["manifest_hash"] = hash_json(manifest)
    _atomic_json(destination / "sealed-retention.json", manifest)
    request: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.autonomous_role_growth_request.v1",
        "source_commit": _git_head(checkout),
        "source_stage_hash": source_expected,
        "source_stage_file_hash": hash_bytes(stage_path.read_bytes()),
        "route_actor_hash": actor.actor_hash,
        "route_actor_file_hash": hash_bytes(actor_path.expanduser().resolve().read_bytes()),
        "skill_bundle": asdict(skill_bundle),
        "skill_bundle_hash": skill_bundle.bundle_hash,
        "candidate_hashes": [item.candidate_hash for item in selected_candidates],
        "retention_manifest_hash": manifest["manifest_hash"],
        "physics_authority": "CPU_MUJOCO",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "pose_joint_torque_or_ball_scripted_by_role_policy": False,
    }
    request["request_hash"] = hash_json(request)
    _atomic_json(destination / "request.json", request)

    development_rows: dict[str, list[AutonomousRoleEpisodeResult]] = {}
    development_report: dict[str, Any] = {}
    development_cases = default_autonomous_development_cases()
    for candidate in selected_candidates:
        rows: list[AutonomousRoleEpisodeResult] = []
        cases_payload: list[dict[str, Any]] = []
        for index, case in enumerate(development_cases):
            result, trajectory = simulate_autonomous_role_episode(
                asset_root=asset_root,
                case=case,
                candidate=candidate,
                actor_path=actor_path,
                actor=actor,
                skill_bundle=skill_bundle,
            )
            rows.append(result)
            artifact = _save_trajectory(
                destination
                / f"development-{candidate.candidate_id.rsplit('.', 1)[-1]}-{index:03d}.npz",
                trajectory,
            )
            cases_payload.append(
                {
                    "case_id": case.case_id,
                    "case_hash": case.case_hash,
                    "action": case.action.value,
                    "result": result.to_dict(),
                    "artifact": artifact,
                }
            )
        development_rows[candidate.candidate_id] = rows
        development_report[candidate.candidate_id] = {
            "candidate": asdict(candidate),
            "candidate_hash": candidate.candidate_hash,
            "selection_score": list(_candidate_score(rows)),
            "qualified_count": sum(row.qualified for row in rows),
            "task_success_count": sum(row.base.task_succeeded for row in rows),
            "safe_count": sum(row.base.safe for row in rows),
            "cases": cases_payload,
        }
    selected = max(
        selected_candidates,
        key=lambda candidate: _candidate_score(development_rows[candidate.candidate_id]),
    )
    acquisition: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.autonomous_role_acquisition.v1",
        "development_only": True,
        "retention_visible_to_selection": False,
        "selected_candidate_id": selected.candidate_id,
        "selected_candidate_hash": selected.candidate_hash,
        "candidates": development_report,
    }
    acquisition["report_hash"] = hash_json(acquisition)
    _atomic_json(destination / "acquisition.json", acquisition)

    retention_root = destination / "retention"
    retention_root.mkdir()
    retention_rows: list[dict[str, Any]] = []
    for index, case in enumerate(retention_cases):
        primary, trajectory = simulate_autonomous_role_episode(
            asset_root=asset_root,
            case=case,
            candidate=selected,
            actor_path=actor_path,
            actor=actor,
            skill_bundle=skill_bundle,
        )
        replay, replay_trajectory = simulate_autonomous_role_episode(
            asset_root=asset_root,
            case=case,
            candidate=selected,
            actor_path=actor_path,
            actor=actor,
            skill_bundle=skill_bundle,
        )
        case_dir = retention_root / f"case-{index:03d}"
        case_dir.mkdir()
        primary_artifact = _save_trajectory(case_dir / "primary.npz", trajectory)
        replay_artifact = _save_trajectory(case_dir / "replay.npz", replay_trajectory)
        exact_replay = bool(
            primary.to_dict() == replay.to_dict()
            and primary_artifact["trajectory_digest"] == replay_artifact["trajectory_digest"]
        )
        retention_rows.append(
            {
                "case_id": case.case_id,
                "case_hash": case.case_hash,
                "action": case.action.value,
                "qualified": primary.qualified,
                "task_succeeded": primary.base.task_succeeded,
                "safe": primary.base.safe,
                "exact_replay": exact_replay,
                "result": primary.to_dict(),
                "primary_artifact": primary_artifact,
                "replay_artifact": replay_artifact,
            }
        )
    count = len(retention_rows)
    metrics = {
        "case_count": count,
        "qualified_rate": sum(bool(row["qualified"]) for row in retention_rows) / count,
        "task_success_rate": sum(bool(row["task_succeeded"]) for row in retention_rows) / count,
        "safe_rate": sum(bool(row["safe"]) for row in retention_rows) / count,
        "exact_replay_rate": sum(bool(row["exact_replay"]) for row in retention_rows) / count,
        "pass_count": sum(row["action"] == TacticalAction.PASS.value for row in retention_rows),
        "shoot_count": sum(row["action"] == TacticalAction.SHOOT.value for row in retention_rows),
        "mean_teammate_active_fraction": float(
            np.mean([row["result"]["teammate_motion"]["active_fraction"] for row in retention_rows])
        ),
        "mean_defender_active_fraction": float(
            np.mean([row["result"]["defender_motion"]["active_fraction"] for row in retention_rows])
        ),
        "mean_teammate_intent_count": float(
            np.mean(
                [
                    row["result"]["teammate_intent"]["distinct_intent_count"]
                    for row in retention_rows
                ]
            )
        ),
        "mean_defender_intent_count": float(
            np.mean(
                [
                    row["result"]["defender_intent"]["distinct_intent_count"]
                    for row in retention_rows
                ]
            )
        ),
    }
    parent_metrics = source_stage.get("retention_metrics", {})
    gates = {
        "qualified_all": metrics["qualified_rate"] == 1.0,
        "task_success_parent_retained": metrics["task_success_rate"]
        >= float(parent_metrics.get("task_success_rate", 1.0)),
        "safe_parent_retained": metrics["safe_rate"] >= float(parent_metrics.get("safe_rate", 1.0)),
        "exact_replay_all": metrics["exact_replay_rate"] == 1.0,
        "balanced_actions": metrics["pass_count"] == 2 and metrics["shoot_count"] == 2,
        "all_roles_active": metrics["mean_teammate_active_fraction"] >= 0.75
        and metrics["mean_defender_active_fraction"] >= 0.35,
        "all_roles_autonomous": metrics["mean_teammate_intent_count"] >= 2.0
        and metrics["mean_defender_intent_count"] >= 2.0,
        "oversized_plastic_candidate_rejected": development_report["s198.role.plastic"][
            "qualified_count"
        ]
        < count,
    }
    passed = all(gates.values())
    evidence: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.autonomous_role_growth_evidence.v1",
        "status": "PASS_AUTONOMOUS_ROLE_GROWTH" if passed else "REJECTED_AUTONOMOUS_ROLE_GROWTH",
        "passed": passed,
        "selected_candidate_id": selected.candidate_id,
        "selected_candidate_hash": selected.candidate_hash,
        "source_stage_hash": source_expected,
        "route_actor_hash": actor.actor_hash,
        "skill_bundle_hash": skill_bundle.bundle_hash,
        "request_hash": request["request_hash"],
        "manifest_hash": manifest["manifest_hash"],
        "acquisition_report_hash": acquisition["report_hash"],
        "metrics": metrics,
        "gates": gates,
        "rows": retention_rows,
        "evidence_boundary": {
            "physics_authority": "CPU_MUJOCO",
            "activation_ceiling": "SIM_ONLY",
            "whole_body_g1_count": 3,
            "red_agents": 2,
            "blue_agents": 1,
            "shared_solver_and_ball": True,
            "role_decision_hz": 1.0 / selected.decision_period_sec,
            "movement_executed_by_frozen_neural_locomotion": True,
            "pose_joint_torque_or_ball_scripted_by_role_policy": False,
            "pixels_used_for_scoring": False,
            "hardware_command_sent": False,
        },
        "source_commit": request["source_commit"],
        "implementation_hash": hash_json(
            {
                "growth": hash_bytes(Path(__file__).read_bytes()),
                "bridge": hash_bytes(
                    (Path(__file__).parent / "full_body_tactical_2v1.py").read_bytes()
                ),
                "shared_world": hash_bytes(
                    (Path(__file__).parents[1] / "skills/team/shared_world.py").read_bytes()
                ),
            }
        ),
    }
    evidence["report_hash"] = hash_json(evidence)
    _atomic_json(retention_root / "retention-exam.json", evidence)
    return validate_autonomous_role_growth(retention_root / "retention-exam.json")


def validate_autonomous_role_growth(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("autonomous role evidence must be an object")
    expected = payload.pop("report_hash", None)
    try:
        rows = payload.get("rows")
        if not isinstance(rows, list) or len(rows) != 4:
            raise ValueError("autonomous role retention rows are incomplete")
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError("autonomous role row is malformed")
            for key in ("primary_artifact", "replay_artifact"):
                artifact = row.get(key)
                if not isinstance(artifact, dict):
                    raise ValueError("autonomous role trajectory binding is absent")
                trajectory = resolved.parent / f"case-{index:03d}" / str(artifact.get("file"))
                if not trajectory.is_file() or hash_bytes(trajectory.read_bytes()) != artifact.get(
                    "file_hash"
                ):
                    raise ValueError("autonomous role trajectory binding changed")
        passed = bool(all(payload.get("gates", {}).values()))
        if (
            expected != hash_json(payload)
            or payload.get("schema_version") != "rosclaw_soccer.autonomous_role_growth_evidence.v1"
            or payload.get("passed") is not passed
            or payload.get("status")
            != ("PASS_AUTONOMOUS_ROLE_GROWTH" if passed else "REJECTED_AUTONOMOUS_ROLE_GROWTH")
            or payload.get("evidence_boundary", {}).get("physics_authority") != "CPU_MUJOCO"
            or payload.get("evidence_boundary", {}).get("activation_ceiling") != "SIM_ONLY"
            or payload.get("evidence_boundary", {}).get("hardware_command_sent") is not False
        ):
            raise ValueError("autonomous role integrity or authority contract is invalid")
    finally:
        if expected is not None:
            payload["report_hash"] = expected
    return cast(dict[str, Any], payload)


__all__ = [
    "AutonomousRoleCandidate",
    "AutonomousRoleCase",
    "AutonomousRoleEpisodeResult",
    "RoleIntentQuality",
    "build_autonomous_role_plan",
    "default_autonomous_development_cases",
    "default_autonomous_retention_cases",
    "default_autonomous_role_candidates",
    "run_autonomous_role_growth",
    "simulate_autonomous_role_episode",
    "validate_autonomous_role_growth",
]
