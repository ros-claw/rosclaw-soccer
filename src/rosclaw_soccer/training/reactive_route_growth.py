"""S122 learning and physical evaluation for reactive multi-agent routes."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.growth.reactive_route_actor import (
    G1ReactiveRouteActor,
    ReactiveRouteSample,
    fit_reactive_route_actor,
    reactive_route_features,
    save_reactive_route_actor,
)
from rosclaw_soccer.growth.tactical_2v1 import TacticalAction
from rosclaw_soccer.growth.tactical_2v1_actor import load_two_vs_one_tactical_actor
from rosclaw_soccer.growth.temporal_route_actor import (
    RouteActor,
)
from rosclaw_soccer.growth.temporal_route_actor import (
    load_route_actor as load_route_actor_artifact,
)
from rosclaw_soccer.providers.g1.asset_qualification import trajectory_digest
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.shared_world import G1ReactiveMovementConfig
from rosclaw_soccer.training.active_off_ball_growth import (
    ActiveOffBallResult,
    ActiveRouteCandidate,
    _motion_quality,
    build_action_conditioned_movement_plan,
    default_active_route_candidates,
)
from rosclaw_soccer.training.full_body_tactical_2v1 import (
    FullBodyReactiveRoleMovementPlan,
    FullBodyTwoVsOneConfig,
    FullBodyTwoVsOneScenario,
    simulate_full_body_two_vs_one,
)
from rosclaw_soccer.training.tactical_2v1_physics import FrozenTacticalSkillBundle

_FEATURE_COUNT = 14


@dataclass(frozen=True)
class ReactiveRouteSourceSnapshot:
    source_stage_hash: str
    source_report_hash: str
    source_actor_hash: str
    episode_ids: tuple[str, ...]
    samples: tuple[ReactiveRouteSample, ...]
    schema_version: str = "rosclaw_soccer.reactive_route_source_snapshot.v1"

    @property
    def snapshot_hash(self) -> str:
        return str(
            hash_json(
                {
                    "schema_version": self.schema_version,
                    "source_stage_hash": self.source_stage_hash,
                    "source_report_hash": self.source_report_hash,
                    "source_actor_hash": self.source_actor_hash,
                    "episode_ids": self.episode_ids,
                    "sample_count": len(self.samples),
                    "samples_hash": hash_json([asdict(row) for row in self.samples]),
                }
            )
        )


@dataclass(frozen=True)
class ReactiveRouteCase:
    scenario: FullBodyTwoVsOneScenario
    teammate_origin_offset_m: tuple[float, float]
    defender_origin_offset_m: tuple[float, float]
    schema_version: str = "rosclaw_soccer.reactive_route_case.v1"

    def __post_init__(self) -> None:
        values = (*self.teammate_origin_offset_m, *self.defender_origin_offset_m)
        if any(not math.isfinite(value) or abs(value) > 0.18 for value in values):
            raise ValueError("reactive route initial disturbance exceeds 0.18 m")

    @property
    def case_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass(frozen=True)
class ReactiveRouteRetentionManifest:
    cases: tuple[ReactiveRouteCase, ...]
    suite_id: str = "s122.reactive-route.sealed-retention"
    training_access_allowed: bool = False
    activation_ceiling: str = "SIM_ONLY"
    schema_version: str = "rosclaw_soccer.reactive_route_retention_manifest.v1"

    def __post_init__(self) -> None:
        hashes = tuple(case.case_hash for case in self.cases)
        if (
            len(self.cases) < 8
            or len(set(hashes)) != len(hashes)
            or self.training_access_allowed
            or self.activation_ceiling != "SIM_ONLY"
        ):
            raise ValueError("reactive route retention must be unique, sealed and SIM_ONLY")

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": self.schema_version,
            "suite_id": self.suite_id,
            "cases": [asdict(case) for case in self.cases],
            "case_hashes": [case.case_hash for case in self.cases],
            "training_access_allowed": self.training_access_allowed,
            "activation_ceiling": self.activation_ceiling,
        }
        if include_hash:
            value["manifest_hash"] = hash_json(value)
        return value

    @property
    def manifest_hash(self) -> str:
        return str(hash_json(self.to_dict(include_hash=False)))


def load_released_route_snapshot(source_stage_dir: Path) -> ReactiveRouteSourceSnapshot:
    """Verify and extract the already released S121 physical trajectories."""

    root = source_stage_dir.expanduser().resolve()
    stage = json.loads((root / "stage-summary.json").read_text(encoding="utf-8"))
    claimed_stage_hash = stage.pop("stage_hash", None)
    if (
        claimed_stage_hash != hash_json(stage)
        or stage.get("status") != "PASS_ACTIVE_OFF_BALL_GROWTH"
    ):
        raise ValueError("released route source stage is not a valid passing stage")
    report_path = root / "retention" / "retention-exam.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    claimed_report_hash = report.pop("report_hash", None)
    if (
        claimed_report_hash != hash_json(report)
        or claimed_report_hash != stage.get("retention_report_hash")
        or report.get("status") != "PASS_ACTIVE_OFF_BALL_GROWTH"
    ):
        raise ValueError("released route source retention report failed integrity checks")

    samples: list[ReactiveRouteSample] = []
    episode_ids: list[str] = []
    for index, row in enumerate(report["rows"]):
        episode_id = str(row["scenario_id"])
        episode_ids.append(episode_id)
        path = root / "retention" / f"case-{index:03d}" / "selected-primary.npz"
        artifact = row["primary_artifact"]
        if hash_bytes(path.read_bytes()) != artifact["file_hash"]:
            raise ValueError(f"released route trajectory {episode_id} changed on disk")
        with np.load(path, allow_pickle=False) as archive:
            trajectory = {key: np.asarray(archive[key]) for key in archive.files}
        if trajectory_digest(trajectory) != artifact["trajectory_digest"]:
            raise ValueError(f"released route trajectory {episode_id} digest does not match")
        action = str(row["selected_action"])
        time = np.asarray(trajectory["time"], dtype=np.float64)
        for trace_role, actor_role, other_role in (
            ("passer", "teammate", "goalkeeper"),
            ("goalkeeper", "defender", "passer"),
        ):
            position = np.asarray(trajectory[f"{trace_role}_pelvis_pose"], dtype=np.float64)[:, :2]
            velocity = np.gradient(position, time, axis=0)
            target = np.asarray(
                trajectory[f"{trace_role}_tactical_world_target"], dtype=np.float64
            )[-1, :2]
            ball = np.asarray(trajectory["ball_pose"], dtype=np.float64)[:, :2]
            carrier = np.asarray(trajectory["shooter_pelvis_pose"], dtype=np.float64)[:, :2]
            other = np.asarray(trajectory[f"{other_role}_pelvis_pose"], dtype=np.float64)[:, :2]
            command = np.asarray(
                trajectory[f"{trace_role}_tactical_world_command"], dtype=np.float64
            )[:, :2]
            for frame in range(len(time)):
                features = reactive_route_features(
                    target_xy_m=target,
                    self_position_xy_m=position[frame],
                    self_velocity_xy_mps=velocity[frame],
                    ball_position_xy_m=ball[frame],
                    carrier_position_xy_m=carrier[frame],
                    other_role_position_xy_m=other[frame],
                    action=action,
                    role=actor_role,
                )
                samples.append(
                    ReactiveRouteSample(
                        episode_id=episode_id,
                        features=tuple(float(value) for value in features),
                        teacher_world_command_xy_mps=(
                            float(command[frame, 0]),
                            float(command[frame, 1]),
                        ),
                    )
                )
    return ReactiveRouteSourceSnapshot(
        source_stage_hash=str(claimed_stage_hash),
        source_report_hash=str(claimed_report_hash),
        source_actor_hash=str(report["actor_hash"]),
        episode_ids=tuple(episode_ids),
        samples=tuple(samples),
    )


def _ridge_prediction(
    training: tuple[ReactiveRouteSample, ...],
    evaluation: tuple[ReactiveRouteSample, ...],
    columns: tuple[int, ...],
    *,
    ridge_l2: float,
) -> tuple[float, float]:
    train_x = np.asarray([row.features for row in training], dtype=np.float64)[:, columns]
    train_y = np.asarray([row.teacher_world_command_xy_mps for row in training], dtype=np.float64)
    test_x = np.asarray([row.features for row in evaluation], dtype=np.float64)[:, columns]
    test_y = np.asarray([row.teacher_world_command_xy_mps for row in evaluation], dtype=np.float64)
    center = np.mean(train_x, axis=0)
    scale = np.std(train_x, axis=0)
    scale = np.where(scale < 1.0e-6, 1.0, scale)
    design = np.column_stack(((train_x - center) / scale, np.ones(len(train_x))))
    test_design = np.column_stack(((test_x - center) / scale, np.ones(len(test_x))))
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge_l2
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ train_y)
    prediction = test_design @ weights
    rmse = float(np.sqrt(np.mean((prediction - test_y) ** 2)))
    correlations = [
        float(np.corrcoef(prediction[:, axis], test_y[:, axis])[0, 1]) for axis in range(2)
    ]
    return rmse, float(np.mean(correlations))


def grouped_route_cross_validation(
    snapshot: ReactiveRouteSourceSnapshot,
    *,
    ridge_l2: float = 1.0,
) -> dict[str, Any]:
    """Leave one physical match out and audit the value of team context."""

    feature_sets = {
        "target_only": (0, 1),
        "target_self_action_role": (0, 1, 2, 3, 10, 11, 12, 13),
        "full_team_context": tuple(range(_FEATURE_COUNT)),
    }
    report: dict[str, Any] = {}
    for name, columns in feature_sets.items():
        folds = []
        for episode_id in snapshot.episode_ids:
            training = tuple(row for row in snapshot.samples if row.episode_id != episode_id)
            evaluation = tuple(row for row in snapshot.samples if row.episode_id == episode_id)
            rmse, correlation = _ridge_prediction(training, evaluation, columns, ridge_l2=ridge_l2)
            folds.append({"episode_id": episode_id, "rmse_mps": rmse, "correlation": correlation})
        report[name] = {
            "feature_indices": columns,
            "mean_rmse_mps": float(np.mean([row["rmse_mps"] for row in folds])),
            "maximum_rmse_mps": float(np.max([row["rmse_mps"] for row in folds])),
            "mean_correlation": float(np.mean([row["correlation"] for row in folds])),
            "folds": folds,
        }
    full = report["full_team_context"]
    target = report["target_only"]
    report["gates"] = {
        "full_mean_rmse_mps": full["mean_rmse_mps"] <= 0.07,
        "full_maximum_rmse_mps": full["maximum_rmse_mps"] <= 0.08,
        "full_mean_correlation": full["mean_correlation"] >= 0.95,
        "context_reduces_rmse": full["mean_rmse_mps"] <= 0.70 * target["mean_rmse_mps"],
    }
    report["passed"] = all(report["gates"].values())
    report["report_hash"] = hash_json(report)
    return report


def train_reactive_route_actor(
    *,
    source_stage_dir: Path,
    output_path: Path,
    ridge_l2: float = 1.0,
) -> tuple[G1ReactiveRouteActor, ReactiveRouteSourceSnapshot, dict[str, Any]]:
    snapshot = load_released_route_snapshot(source_stage_dir)
    cross_validation = grouped_route_cross_validation(snapshot, ridge_l2=ridge_l2)
    if not cross_validation["passed"]:
        raise RuntimeError("reactive route actor failed grouped cross-validation")
    actor = fit_reactive_route_actor(
        snapshot.samples,
        source_stage_hash=snapshot.source_stage_hash,
        ridge_l2=ridge_l2,
    )
    save_reactive_route_actor(actor, output_path)
    return actor, snapshot, cross_validation


def label_reactive_route_failure(
    trajectory: dict[str, NDArray[Any]],
    *,
    episode_id: str,
    teammate_lateral_bias_m: float = 0.0,
) -> tuple[ReactiveRouteSample, ...]:
    """Relabel one on-policy failure with a bounded goal-recovery expert."""

    if not math.isfinite(teammate_lateral_bias_m) or not -0.12 <= teammate_lateral_bias_m <= 0.12:
        raise ValueError("failure relabel lateral bias must be in [-0.12, 0.12] m")
    rows: list[ReactiveRouteSample] = []
    for trace_role, maximum_speed, gain, damping in (
        ("passer", 0.45, 1.35, 0.12),
        ("goalkeeper", 0.38, 1.05, 0.15),
    ):
        features = np.asarray(trajectory[f"{trace_role}_reactive_route_features"], dtype=np.float64)
        error = features[:, :2].copy()
        if trace_role == "passer":
            error[:, 1] += teammate_lateral_bias_m
        command = gain * error - damping * features[:, 2:4]
        speed = np.linalg.norm(command, axis=1)
        command *= np.minimum(1.0, maximum_speed / np.maximum(speed, 1.0e-9))[:, None]
        for observation, target in zip(features, command, strict=True):
            rows.append(
                ReactiveRouteSample(
                    episode_id=f"{episode_id}.{trace_role}",
                    features=tuple(float(value) for value in observation),
                    teacher_world_command_xy_mps=(float(target[0]), float(target[1])),
                )
            )
    return tuple(rows)


def build_reactive_movement_plan(
    *,
    scenario: FullBodyTwoVsOneScenario,
    action: TacticalAction,
    actor_path: Path,
    actor: RouteActor,
    teammate_origin_offset_m: tuple[float, float] = (0.0, 0.0),
    defender_origin_offset_m: tuple[float, float] = (0.0, 0.0),
    candidate: ActiveRouteCandidate | None = None,
    simulation_duration_sec: float = 7.0,
) -> FullBodyReactiveRoleMovementPlan:
    selected = candidate or default_active_route_candidates()[1]
    teacher = build_action_conditioned_movement_plan(
        scenario, action, selected, simulation_duration_sec=simulation_duration_sec
    )
    teammate_origin: tuple[float, float, float] = (
        float(teacher.teammate_origin_m[0] + teammate_origin_offset_m[0]),
        float(teacher.teammate_origin_m[1] + teammate_origin_offset_m[1]),
        0.0,
    )
    defender_origin: tuple[float, float, float] = (
        float(scenario.defender_origin_m[0] + defender_origin_offset_m[0]),
        float(scenario.defender_origin_m[1] + defender_origin_offset_m[1]),
        0.0,
    )
    actor_artifact_path = str(actor_path.expanduser().resolve())
    return FullBodyReactiveRoleMovementPlan(
        teammate_origin_m=teammate_origin,
        defender_origin_m=defender_origin,
        teammate_movement=G1ReactiveMovementConfig(
            actor_artifact_path=actor_artifact_path,
            actor_hash=actor.actor_hash,
            action=action.value,
            role="teammate",
            target_position_m=teacher.teammate_movement.waypoints[-1].position_m,
            maximum_speed_mps=selected.maximum_speed_mps,
            maximum_acceleration_mps2=selected.maximum_acceleration_mps2,
            arrival_radius_m=0.02,
        ),
        defender_movement=G1ReactiveMovementConfig(
            actor_artifact_path=actor_artifact_path,
            actor_hash=actor.actor_hash,
            action=action.value,
            role="defender",
            target_position_m=teacher.defender_movement.waypoints[-1].position_m,
            maximum_speed_mps=min(0.40, selected.maximum_speed_mps - 0.05),
            maximum_acceleration_mps2=max(0.40, selected.maximum_acceleration_mps2 - 0.05),
            arrival_radius_m=0.02,
        ),
    )


def simulate_reactive_route_episode(
    *,
    asset_root: Path,
    case: ReactiveRouteCase,
    action: TacticalAction,
    actor_path: Path,
    actor: RouteActor,
    skill_bundle: FrozenTacticalSkillBundle,
    config: FullBodyTwoVsOneConfig | None = None,
) -> tuple[ActiveOffBallResult, dict[str, NDArray[Any]]]:
    active = config or FullBodyTwoVsOneConfig()
    candidate = default_active_route_candidates()[1]
    plan = build_reactive_movement_plan(
        scenario=case.scenario,
        action=action,
        actor_path=actor_path,
        actor=actor,
        teammate_origin_offset_m=case.teammate_origin_offset_m,
        defender_origin_offset_m=case.defender_origin_offset_m,
        candidate=candidate,
        simulation_duration_sec=active.simulation_duration_sec,
    )
    base, trajectory = simulate_full_body_two_vs_one(
        asset_root=asset_root,
        scenario=case.scenario,
        action=action,
        skill_bundle=skill_bundle,
        config=active,
        reactive_movement_plan=plan,
    )
    carrier = _motion_quality(trajectory, "shooter")
    teammate = _motion_quality(trajectory, "passer")
    defender = _motion_quality(trajectory, "goalkeeper")
    accepted = bool(
        np.all(np.asarray(trajectory["passer_reactive_route_accepted"], dtype=np.bool_))
        and np.all(np.asarray(trajectory["goalkeeper_reactive_route_accepted"], dtype=np.bool_))
    )
    movement_passed = bool(
        accepted
        and carrier.displacement_m >= 1.40
        and teammate.displacement_m >= 0.55
        and teammate.stagnant_fraction <= 0.16
        and teammate.alternating_swing_switches >= 4
        and defender.displacement_m >= 0.22
        and defender.stagnant_fraction <= 0.24
        and defender.alternating_swing_switches >= 2
        and teammate.peak_command_acceleration_mps2 is not None
        and teammate.peak_command_acceleration_mps2 <= candidate.maximum_acceleration_mps2 + 1.0e-8
        and defender.peak_command_acceleration_mps2 is not None
        and defender.peak_command_acceleration_mps2
        <= max(0.40, candidate.maximum_acceleration_mps2 - 0.05) + 1.0e-8
    )
    return (
        ActiveOffBallResult(
            base=base,
            candidate_id="s122.route.reactive-observation",
            candidate_hash=actor.actor_hash,
            movement_plan_hash=plan.plan_hash,
            carrier_motion=carrier,
            teammate_motion=teammate,
            defender_motion=defender,
            movement_quality_passed=movement_passed,
        ),
        trajectory,
    )


def default_reactive_development_cases() -> tuple[ReactiveRouteCase, ...]:
    layouts = (
        ((5.43, -0.40, 0.0), (2.01, 0.41, 0.0), (0.04, -0.03), (-0.03, 0.04)),
        ((5.49, -0.40, 0.0), (2.08, 0.48, 0.0), (-0.05, 0.02), (0.04, -0.03)),
        ((5.53, -0.40, 0.0), (2.13, 0.53, 0.0), (0.06, 0.04), (-0.05, -0.02)),
        ((5.57, -0.40, 0.0), (2.01, 0.41, 0.0), (-0.04, -0.05), (0.03, 0.05)),
        ((5.34, 0.34, 0.0), (4.14, 0.34, 0.0), (0.05, -0.04), (-0.04, 0.03)),
        ((5.44, 0.44, 0.0), (4.34, 0.44, 0.0), (-0.05, 0.03), (0.05, -0.04)),
        ((5.54, 0.54, 0.0), (4.54, 0.54, 0.0), (0.07, 0.02), (-0.06, -0.03)),
        ((5.64, 0.54, 0.0), (4.74, 0.44, 0.0), (-0.06, -0.04), (0.04, 0.06)),
    )
    return tuple(
        ReactiveRouteCase(
            scenario=FullBodyTwoVsOneScenario(
                scenario_id=f"s122.development.{index:03d}",
                seed=122_000 + index,
                teammate_origin_m=teammate,
                defender_origin_m=defender,
            ),
            teammate_origin_offset_m=teammate_offset,
            defender_origin_offset_m=defender_offset,
        )
        for index, (teammate, defender, teammate_offset, defender_offset) in enumerate(layouts)
    )


def default_reactive_retention_manifest() -> ReactiveRouteRetentionManifest:
    layouts = (
        ((5.45, -0.40, 0.0), (2.03, 0.43, 0.0), (0.09, -0.07), (-0.08, 0.06)),
        ((5.50, -0.40, 0.0), (2.09, 0.49, 0.0), (-0.10, 0.06), (0.08, -0.07)),
        ((5.55, -0.40, 0.0), (2.14, 0.54, 0.0), (0.12, 0.05), (-0.09, -0.06)),
        ((5.59, -0.40, 0.0), (2.02, 0.42, 0.0), (-0.08, -0.10), (0.07, 0.09)),
        ((5.36, 0.36, 0.0), (4.16, 0.36, 0.0), (0.09, -0.08), (-0.08, 0.07)),
        ((5.46, 0.46, 0.0), (4.36, 0.46, 0.0), (-0.11, 0.07), (0.10, -0.08)),
        ((5.56, 0.56, 0.0), (4.56, 0.56, 0.0), (0.13, 0.05), (-0.11, -0.07)),
        ((5.66, 0.56, 0.0), (4.76, 0.46, 0.0), (-0.10, -0.09), (0.08, 0.12)),
    )
    cases = tuple(
        ReactiveRouteCase(
            scenario=FullBodyTwoVsOneScenario(
                scenario_id=f"s122.retention.{index:03d}",
                seed=122_500 + index,
                teammate_origin_m=teammate,
                defender_origin_m=defender,
            ),
            teammate_origin_offset_m=teammate_offset,
            defender_origin_offset_m=defender_offset,
        )
        for index, (teammate, defender, teammate_offset, defender_offset) in enumerate(layouts)
    )
    return ReactiveRouteRetentionManifest(cases=cases)


def default_reactive_retention_manifest_v2() -> ReactiveRouteRetentionManifest:
    """Fresh sealed suite used only after the first champion was rejected."""

    layouts = (
        ((5.47, -0.40, 0.0), (2.04, 0.44, 0.0), (0.11, -0.09), (-0.10, 0.08)),
        ((5.51, -0.40, 0.0), (2.10, 0.50, 0.0), (-0.13, 0.09), (0.11, -0.09)),
        ((5.56, -0.40, 0.0), (2.15, 0.55, 0.0), (0.14, 0.07), (-0.12, -0.09)),
        ((5.60, -0.40, 0.0), (2.03, 0.43, 0.0), (-0.11, -0.13), (0.10, 0.11)),
        ((5.38, 0.38, 0.0), (4.18, 0.38, 0.0), (0.11, -0.10), (-0.10, 0.09)),
        ((5.48, 0.48, 0.0), (4.38, 0.48, 0.0), (-0.13, 0.09), (0.12, -0.10)),
        ((5.58, 0.58, 0.0), (4.58, 0.58, 0.0), (0.14, 0.07), (-0.13, -0.09)),
        ((5.68, 0.58, 0.0), (4.78, 0.48, 0.0), (-0.12, -0.11), (0.10, 0.14)),
    )
    cases = tuple(
        ReactiveRouteCase(
            scenario=FullBodyTwoVsOneScenario(
                scenario_id=f"s122.retention-v2.{index:03d}",
                seed=122_700 + index,
                teammate_origin_m=teammate,
                defender_origin_m=defender,
            ),
            teammate_origin_offset_m=teammate_offset,
            defender_origin_offset_m=defender_offset,
        )
        for index, (teammate, defender, teammate_offset, defender_offset) in enumerate(layouts)
    )
    return ReactiveRouteRetentionManifest(
        cases=cases,
        suite_id="s122.reactive-route.sealed-retention-v2",
    )


def run_reactive_route_retention_exam(
    *,
    output_dir: Path,
    asset_root: Path,
    route_actor_path: Path,
    tactical_actor_path: Path,
    skill_bundle: FrozenTacticalSkillBundle,
    manifest: ReactiveRouteRetentionManifest,
    config: FullBodyTwoVsOneConfig | None = None,
) -> dict[str, Any]:
    """Run a frozen manifest twice without exposing it to any training hook."""

    destination = output_dir.expanduser().resolve()
    if destination.exists():
        raise ValueError("reactive route retention output must be new")
    destination.mkdir(parents=True)
    route_actor = load_route_actor(route_actor_path)
    tactical_actor = load_two_vs_one_tactical_actor(tactical_actor_path)
    active = config or FullBodyTwoVsOneConfig()
    _write_json(destination / "sealed-retention.json", manifest.to_dict())
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(manifest.cases):
        decision = tactical_actor.decide(
            case.scenario.state(skill_bundle=skill_bundle, config=active)
        )
        if not decision.accepted or decision.action not in {
            TacticalAction.PASS,
            TacticalAction.SHOOT,
        }:
            raise RuntimeError("frozen tactical actor rejected a sealed reactive route case")
        runs = []
        for label in ("primary", "replay"):
            result, trajectory = simulate_reactive_route_episode(
                asset_root=asset_root,
                case=case,
                action=decision.action,
                actor_path=route_actor_path,
                actor=route_actor,
                skill_bundle=skill_bundle,
                config=active,
            )
            case_dir = destination / "retention" / f"case-{index:03d}"
            case_dir.mkdir(parents=True, exist_ok=True)
            artifact = _save_trajectory(case_dir / f"{label}.npz", trajectory)
            runs.append((result, trajectory, artifact))
        primary, primary_trajectory, primary_artifact = runs[0]
        replay, _, replay_artifact = runs[1]
        rows.append(
            {
                "case_id": case.scenario.scenario_id,
                "case_hash": case.case_hash,
                "action": decision.action.value,
                "result": primary.to_dict(),
                "qualified": primary.qualified,
                "safe": primary.base.safe,
                "movement_quality_passed": primary.movement_quality_passed,
                "reactive_actor_accepted": bool(
                    np.all(primary_trajectory["passer_reactive_route_accepted"])
                    and np.all(primary_trajectory["goalkeeper_reactive_route_accepted"])
                ),
                "maximum_teammate_support_distance": float(
                    np.max(primary_trajectory["passer_reactive_route_support_distance"])
                ),
                "maximum_defender_support_distance": float(
                    np.max(primary_trajectory["goalkeeper_reactive_route_support_distance"])
                ),
                "exact_replay": bool(
                    primary.to_dict() == replay.to_dict()
                    and primary_artifact["trajectory_digest"]
                    == replay_artifact["trajectory_digest"]
                ),
                "primary_artifact": primary_artifact,
                "replay_artifact": replay_artifact,
            }
        )
    count = len(rows)
    metrics = {
        "case_count": count,
        "qualified_rate": sum(row["qualified"] for row in rows) / count,
        "task_success_rate": sum(row["result"]["base"]["task_succeeded"] for row in rows) / count,
        "safe_rate": sum(row["safe"] for row in rows) / count,
        "movement_quality_rate": sum(row["movement_quality_passed"] for row in rows) / count,
        "reactive_actor_acceptance_rate": sum(row["reactive_actor_accepted"] for row in rows)
        / count,
        "exact_replay_rate": sum(row["exact_replay"] for row in rows) / count,
        "selected_action_counts": {
            action: sum(row["action"] == action for row in rows) for action in ("pass", "shoot")
        },
        "mean_teammate_displacement_m": float(
            np.mean([row["result"]["teammate_motion"]["displacement_m"] for row in rows])
        ),
        "mean_defender_displacement_m": float(
            np.mean([row["result"]["defender_motion"]["displacement_m"] for row in rows])
        ),
        "mean_teammate_stagnant_fraction": float(
            np.mean([row["result"]["teammate_motion"]["stagnant_fraction"] for row in rows])
        ),
        "mean_defender_stagnant_fraction": float(
            np.mean([row["result"]["defender_motion"]["stagnant_fraction"] for row in rows])
        ),
    }
    gates = {
        "qualified_rate": metrics["qualified_rate"] == 1.0,
        "task_success_rate": metrics["task_success_rate"] == 1.0,
        "safe_rate": metrics["safe_rate"] == 1.0,
        "movement_quality_rate": metrics["movement_quality_rate"] == 1.0,
        "reactive_actor_acceptance_rate": metrics["reactive_actor_acceptance_rate"] == 1.0,
        "exact_replay_rate": metrics["exact_replay_rate"] == 1.0,
        "both_actions_covered": metrics["selected_action_counts"] == {"pass": 4, "shoot": 4},
    }
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.reactive_route_retention_exam.v2",
        "status": (
            "PASS_REACTIVE_MULTI_AGENT_GROWTH"
            if all(gates.values())
            else "REJECTED_REACTIVE_MULTI_AGENT_GROWTH"
        ),
        "actor_hash": route_actor.actor_hash,
        "actor_file_hash": hash_bytes(route_actor_path.read_bytes()),
        "manifest_hash": manifest.manifest_hash,
        "metrics": metrics,
        "gates": gates,
        "rows": rows,
        "evidence_boundary": {
            "activation_ceiling": "SIM_ONLY",
            "physics_authority": "CPU_MUJOCO",
            "whole_body_g1_count": 3,
            "shared_solver_and_ball": True,
            "tactical_actor_frozen": True,
            "route_actor_observation_closed_loop": True,
            "movement_executed_by_frozen_neural_locomotion": True,
            "pose_joint_torque_or_ball_scripted_by_route_actor": False,
            "pixels_used_for_scoring": False,
            "hardware_command_sent": False,
        },
    }
    report["report_hash"] = hash_json(report)
    _write_json(destination / "retention" / "retention-exam.json", report)
    return report


def load_route_actor(path: Path) -> RouteActor:
    """Late import seam kept explicit for retention-runner test doubles."""

    return load_route_actor_artifact(path)


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
    "ReactiveRouteCase",
    "ReactiveRouteRetentionManifest",
    "ReactiveRouteSourceSnapshot",
    "build_reactive_movement_plan",
    "default_reactive_development_cases",
    "default_reactive_retention_manifest",
    "default_reactive_retention_manifest_v2",
    "grouped_route_cross_validation",
    "load_released_route_snapshot",
    "load_route_actor",
    "label_reactive_route_failure",
    "simulate_reactive_route_episode",
    "train_reactive_route_actor",
    "run_reactive_route_retention_exam",
]
