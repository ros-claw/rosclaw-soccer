"""S114 role-local teacher rehearsal and target-conditioned contact distillation."""

from __future__ import annotations

import argparse
import inspect
import itertools
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from rosclaw.feedback import contracts as growth_core_contracts
from rosclaw.simforge.reproducibility import (
    ReproducibilityClosure,
    build_reproducibility_closure,
)

from rosclaw_soccer.growth.ballistic_contact_impulse_actor import (
    G1BallisticContactImpulseActor,
    load_g1_ballistic_contact_impulse_actor,
)
from rosclaw_soccer.providers.g1.asset_qualification import trajectory_digest
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.shoot.loft_teacher import G1LoftTeacherConfig
from rosclaw_soccer.skills.team.shared_world import (
    G1PhysicalSecondStrikerConfig,
    simulate_shared_world,
)
from rosclaw_soccer.training.continuous_second_striker_save_exam import (
    ContinuousSecondStrikerSaveExamConfig,
    evaluate_continuous_second_striker_save,
    physical_second_striker_kwargs,
)
from rosclaw_soccer.training.dynamic_corner_save import expanded_dynamic_corner_lanes

_CLAIM = "ROLE_ISOLATED_FULL_CHAIN_CONTACT_TEACHER_DISTILLATION"
_V2_ACTOR_SCHEMA = "rosclaw.growth.g1_ballistic_contact_impulse_actor.v2"
_G1_ARTIFACTS = (
    ("g1-policy", Path("policy/robonaldo/model/policy-obs-aic.onnx")),
    ("g1-motion", Path("policy/robonaldo/model/freekick_motion.npz")),
    ("g1-scene", Path("g1_description/scene_with_ball.xml")),
    ("g1-model", Path("g1_description/g1_liao.xml")),
    ("g1-free-kick", Path("policy/robonaldo/FreeKick.py")),
)


@dataclass(frozen=True)
class RoleIsolatedContactTeacherProbe:
    label: str
    target_lateral_speed_mps: float
    target_vertical_speed_mps: float
    maximum_lateral_force_n: float
    maximum_vertical_force_n: float

    def teacher_config(self) -> G1LoftTeacherConfig:
        return G1LoftTeacherConfig(
            target_lateral_speed_mps=self.target_lateral_speed_mps,
            target_vertical_speed_mps=self.target_vertical_speed_mps,
            lateral_velocity_gain_n_per_mps=30.0,
            velocity_gain_n_per_mps=30.0,
            maximum_lateral_force_n=self.maximum_lateral_force_n,
            maximum_vertical_force_n=self.maximum_vertical_force_n,
            start_policy_frame=230,
            end_policy_frame=335,
            maximum_foot_ball_distance_m=0.25,
        )


_PROBES = (
    RoleIsolatedContactTeacherProbe("lat2-level", 2.0, 0.0, 60.0, 60.0),
    RoleIsolatedContactTeacherProbe("lat2-down", 2.0, -4.0, 180.0, 250.0),
    RoleIsolatedContactTeacherProbe("lat3-down", 3.0, -4.0, 250.0, 250.0),
    RoleIsolatedContactTeacherProbe("lat5-down", 5.0, -4.0, 250.0, 250.0),
    RoleIsolatedContactTeacherProbe("lat7-up7", 7.0, 7.0, 250.0, 180.0),
    RoleIsolatedContactTeacherProbe("lat10-up5", 10.0, 5.0, 250.0, 120.0),
    RoleIsolatedContactTeacherProbe("lat8-up5", 8.0, 5.0, 250.0, 120.0),
    RoleIsolatedContactTeacherProbe("lat8-up6", 8.0, 6.0, 250.0, 150.0),
)


@dataclass(frozen=True)
class RoleIsolatedContactActorGrowthConfig:
    lane_id: str = "left-inner"
    simulation_duration_sec: float = 23.0
    local_probe_count: int = 5
    probe_replay_count: int = 2
    ridge_regularization: float = 0.05
    # The first complete-chain failure showed that a 0.25 inverse-model step
    # moved a successful teacher launch across the goalkeeper's contact
    # boundary.  Keep the learner plastic, but contract the update around the
    # nearest successful rehearsal and use a deliberately slower muscle-speed
    # feedback loop.  The failure-update evidence binds this revision.
    local_plasticity_gain: float = 0.05
    proprioceptive_feedback_gain_n_per_mps: float = 6.0
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.role_isolated_contact_actor_growth_config.v1"

    def __post_init__(self) -> None:
        if self.lane_id not in {"left-inner", "left-outer", "right-inner", "right-outer"}:
            raise ValueError("role-isolated contact-growth lane is unknown")
        if not 23.0 <= self.simulation_duration_sec <= 25.0:
            raise ValueError("role-isolated contact-growth duration is invalid")
        if not 4 <= self.local_probe_count <= len(_PROBES):
            raise ValueError("role-isolated local probe count is invalid")
        if self.probe_replay_count != 2:
            raise ValueError("role-isolated contact growth requires two exact probe replays")
        if not 0.0 < self.ridge_regularization <= 10.0:
            raise ValueError("role-isolated ridge regularization is invalid")
        if not 0.05 <= self.local_plasticity_gain <= 0.50:
            raise ValueError("role-isolated plasticity gain is invalid")
        if not 5.0 <= self.proprioceptive_feedback_gain_n_per_mps <= 20.0:
            raise ValueError("role-isolated proprioceptive feedback gain is invalid")
        if self.activation_ceiling != "SIM_ONLY" or self.hardware_authorized:
            raise ValueError("role-isolated contact growth must remain SIM_ONLY")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_trajectory(path: Path, trajectory: dict[str, NDArray[Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **trajectory)  # type: ignore[arg-type]
    os.replace(temporary, path)


def _closure_inputs(
    *, asset_root: Path, assets: dict[str, Path]
) -> tuple[dict[str, Path], dict[str, Path]]:
    source_trees = {
        "dive-source": assets["dive_source"],
        "rosclaw-core-reproducibility": Path(inspect.getfile(ReproducibilityClosure))
        .resolve()
        .parent,
        "rosclaw-core-runtime": Path(inspect.getfile(growth_core_contracts)).resolve().parents[2],
        "soccer": Path(__file__).resolve().parents[1],
    }
    artifacts = {
        key.replace("_", "-"): value for key, value in assets.items() if key != "dive_source"
    }
    artifacts.update({label: asset_root / relative for label, relative in _G1_ARTIFACTS})
    return source_trees, artifacts


def _desired_launch_yz(
    *, striker: G1PhysicalSecondStrikerConfig, reference_forward_speed_mps: float, goal_x_m: float
) -> NDArray[np.float64]:
    target_x = goal_x_m - striker.ballistic_target_depth_before_goal_m
    flight_time = (target_x - striker.ball_origin_m[0]) / reference_forward_speed_mps
    if not math.isfinite(flight_time) or flight_time <= 0.0:
        raise ValueError("role-isolated target flight time is invalid")
    return np.asarray(
        (
            (striker.ballistic_target_y_m - striker.ball_origin_m[1]) / flight_time,
            (striker.ballistic_target_z_m - striker.ball_origin_m[2] + 0.5 * 9.81 * flight_time**2)
            / flight_time,
        ),
        dtype=np.float64,
    )


def _inside_convex_cloud(points: NDArray[np.float64], target: NDArray[np.float64]) -> bool:
    if points.ndim != 2 or points.shape[1] != 2 or target.shape != (2,):
        raise ValueError("role-isolated convex support input is invalid")
    for indices in itertools.combinations(range(points.shape[0]), 3):
        triangle = points[np.asarray(indices, dtype=np.int64)]
        matrix = np.vstack((triangle.T, np.ones(3, dtype=np.float64)))
        if abs(float(np.linalg.det(matrix))) < 1.0e-10:
            continue
        weights = np.linalg.solve(matrix, np.append(target, 1.0))
        if np.all(weights >= -1.0e-9) and np.all(weights <= 1.0 + 1.0e-9):
            return True
    return False


def _probe_row(
    *,
    probe: RoleIsolatedContactTeacherProbe,
    trajectory: dict[str, NDArray[Any]],
    result: dict[str, Any],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    time = np.asarray(trajectory["time"], dtype=np.float64)
    velocity = np.asarray(trajectory["second_ball_velocity"], dtype=np.float64)[:, :3]
    teacher_active = np.asarray(trajectory["second_striker_loft_teacher_active"], dtype=np.bool_)
    teacher_force = np.asarray(
        trajectory["second_striker_loft_teacher_force_yz_n"], dtype=np.float64
    )
    teacher_foot_velocity = np.asarray(
        trajectory["second_striker_loft_teacher_foot_velocity_yz_mps"], dtype=np.float64
    )
    parent_active = np.asarray(trajectory["second_striker_ballistic_actor_active"], dtype=np.bool_)
    contact_value = result.get("second_striker_contact_time_sec")
    if (
        time.ndim != 1
        or velocity.shape != (time.size, 3)
        or teacher_active.shape != time.shape
        or teacher_force.shape != (time.size, 2)
        or teacher_foot_velocity.shape != (time.size, 2)
        or parent_active.shape != time.shape
        or not isinstance(contact_value, int | float)
        or isinstance(contact_value, bool)
        or not math.isfinite(float(contact_value))
        or not np.any(teacher_active)
        or np.any(parent_active)
    ):
        raise ValueError("role-isolated teacher trajectory is invalid")
    contact_index = int(np.clip(np.searchsorted(time, float(contact_value)), 1, time.size - 1))
    active_force = teacher_force[teacher_active]
    peak_index = int(np.argmax(np.linalg.norm(active_force, axis=1)))
    peak_force = active_force[peak_index]
    peak_foot_velocity = teacher_foot_velocity[teacher_active][peak_index]
    gates = cast(dict[str, bool], evaluation.get("gates", {}))
    teacher_only_gates = {
        key: value
        for key, value in gates.items()
        if key
        not in {
            "learned_multi_role_contact_stack_active",
            "final_goalkeeper_ready",
        }
    }
    hard_safe = bool(
        evaluation.get("first_takeoff_exam", {}).get("passed") is True
        and gates.get("whole_world_safety") is True
        and result.get("finite_state") is True
        and result.get("second_striker_contact_observed") is True
    )
    return {
        "label": probe.label,
        "probe": asdict(probe),
        "hard_safe": hard_safe,
        "teacher_success": bool(hard_safe and all(teacher_only_gates.values())),
        "teacher_success_gates": teacher_only_gates,
        "launch_velocity_xyz_mps": velocity[contact_index].tolist(),
        "teacher_peak_force_yz_n": peak_force.tolist(),
        "teacher_peak_foot_velocity_yz_mps": peak_foot_velocity.tolist(),
        "result": result,
        "evaluation": evaluation,
    }


def _fit_candidate(
    *,
    rows: list[dict[str, Any]],
    parent_actor: G1BallisticContactImpulseActor,
    striker: G1PhysicalSecondStrikerConfig,
    goal_x_m: float,
    context_hash: str,
    config: RoleIsolatedContactActorGrowthConfig,
) -> tuple[G1BallisticContactImpulseActor, dict[str, Any]]:
    safe = [row for row in rows if row.get("hard_safe") is True]
    rejected = [row for row in rows if row.get("teacher_success") is not True]
    qualified = [row for row in rows if row.get("teacher_success") is True]
    if len(safe) < config.local_probe_count or len(rejected) < 2 or not qualified:
        raise ValueError("role-isolated probes lack safe, successful and rejected support")
    safe_launch = np.asarray(
        [cast(list[float], row["launch_velocity_xyz_mps"]) for row in safe], dtype=np.float64
    )
    reference = float(np.median(safe_launch[:, 0]))
    desired = _desired_launch_yz(
        striker=striker, reference_forward_speed_mps=reference, goal_x_m=goal_x_m
    )
    local: list[dict[str, Any]] = []
    for _ in range(2):
        launch_yz = safe_launch[:, 1:3]
        scale = np.maximum(np.ptp(launch_yz, axis=0), 0.05)
        order = np.argsort(np.linalg.norm((launch_yz - desired) / scale, axis=1))
        local = [safe[int(index)] for index in order[: config.local_probe_count]]
        reference = float(
            np.median([cast(list[float], row["launch_velocity_xyz_mps"])[0] for row in local])
        )
        desired = _desired_launch_yz(
            striker=striker, reference_forward_speed_mps=reference, goal_x_m=goal_x_m
        )
    launch = np.asarray(
        [cast(list[float], row["launch_velocity_xyz_mps"])[1:3] for row in local],
        dtype=np.float64,
    )
    force = np.asarray(
        [cast(list[float], row["teacher_peak_force_yz_n"]) for row in local],
        dtype=np.float64,
    )
    if not _inside_convex_cloud(launch, desired):
        raise ValueError("role-isolated target is outside the measured convex launch support")
    mean = np.mean(force, axis=0)
    scale = np.std(force, axis=0)
    if np.any(scale < 1.0e-4):
        raise ValueError("role-isolated local force distribution is degenerate")
    standardized = (force - mean) / scale
    design = np.column_stack((np.ones(force.shape[0]), standardized))
    penalty = np.diag((0.0, config.ridge_regularization, config.ridge_regularization))
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ launch)
    slopes = coefficients[1:, :] / scale[:, None]
    intercept = coefficients[0, :] - mean @ slopes
    predictions = np.column_stack((np.ones(force.shape[0]), force)) @ np.vstack((intercept, slopes))
    fit_rmse = float(np.sqrt(np.mean(np.square(predictions - launch))))
    if not math.isfinite(float(np.linalg.cond(slopes))) or np.linalg.cond(slopes) > 1.0e4:
        raise ValueError("role-isolated local force response is ill-conditioned")
    inverse = np.linalg.pinv(slopes, rcond=1.0e-4)
    successful_local = [row for row in local if row.get("teacher_success") is True]
    if not successful_local:
        raise ValueError("role-isolated local support has no successful teacher")
    selected = min(
        successful_local,
        key=lambda row: float(
            np.linalg.norm(
                np.asarray(cast(list[float], row["launch_velocity_xyz_mps"])[1:3]) - desired
            )
        ),
    )
    selected_launch = np.asarray(
        cast(list[float], selected["launch_velocity_xyz_mps"])[1:3], dtype=np.float64
    )
    selected_force = np.asarray(
        cast(list[float], selected["teacher_peak_force_yz_n"]), dtype=np.float64
    )
    selected_foot_velocity = np.asarray(
        cast(list[float], selected["teacher_peak_foot_velocity_yz_mps"]),
        dtype=np.float64,
    )
    feedback_gain = config.proprioceptive_feedback_gain_n_per_mps
    feedback_intercept = selected_force + feedback_gain * selected_foot_velocity
    local_inverse = config.local_plasticity_gain * inverse
    actor_intercept = feedback_intercept - selected_launch @ local_inverse
    margin = np.maximum(0.05, 0.10 * np.ptp(launch, axis=0))
    source_hashes = tuple(str(row["trajectory_hash"]) for row in rows)
    actor = G1BallisticContactImpulseActor(
        body_hash=parent_actor.body_hash,
        implementation_hash=str(hash_json({"module": hash_bytes(Path(__file__).read_bytes())})),
        experiment_context_hash=context_hash,
        source_evidence_hashes=source_hashes,
        selected_evidence_hash=str(selected["trajectory_hash"]),
        selected_goal_plane_target_error_m=float(np.linalg.norm(selected_launch - desired)),
        precision_success_count=len(qualified),
        rejected_probe_count=len(rejected),
        task_space_actor_weight_matrix=(
            (
                float(actor_intercept[0]),
                float(local_inverse[0, 0]),
                float(local_inverse[1, 0]),
                -feedback_gain,
                0.0,
            ),
            (
                float(actor_intercept[1]),
                float(local_inverse[0, 1]),
                float(local_inverse[1, 1]),
                0.0,
                -feedback_gain,
            ),
        ),
        maximum_lateral_force_n=float(np.max(force[:, 0])),
        maximum_vertical_force_n=float(np.max(force[:, 1])),
        minimum_lateral_force_n=float(np.min(force[:, 0])),
        minimum_vertical_force_n=float(np.min(force[:, 1])),
        maximum_foot_ball_distance_m=striker.ballistic_actor_proximity_m,
        start_policy_frame=230,
        end_policy_frame=335,
        foot_strike_point_offset_m=(0.13, 0.0, -0.025),
        qualified_error_max_m=0.20,
        reference_forward_ball_speed_mps=reference,
        minimum_supported_lateral_launch_speed_mps=float(np.min(launch[:, 0]) - margin[0]),
        maximum_supported_lateral_launch_speed_mps=float(np.max(launch[:, 0]) + margin[0]),
        minimum_supported_vertical_launch_speed_mps=float(np.min(launch[:, 1]) - margin[1]),
        maximum_supported_vertical_launch_speed_mps=float(np.max(launch[:, 1]) + margin[1]),
        forward_dynamics_fit_rmse_mps=fit_rmse,
        ridge_regularization=config.ridge_regularization,
        safe_probe_count=len(local),
        training_target_count=1,
        schema_version=_V2_ACTOR_SCHEMA,
    )
    diagnostics = {
        "desired_launch_velocity_yz_mps": desired.tolist(),
        "reference_forward_ball_speed_mps": reference,
        "local_probe_labels": [str(row["label"]) for row in local],
        "selected_probe_label": selected["label"],
        "selected_probe_launch_velocity_yz_mps": selected_launch.tolist(),
        "selected_probe_force_yz_n": selected_force.tolist(),
        "selected_probe_foot_velocity_yz_mps": selected_foot_velocity.tolist(),
        "proprioceptive_feedback_gain_n_per_mps": feedback_gain,
        "proprioceptive_feedback_intercept_yz_n": feedback_intercept.tolist(),
        "forward_fit_rmse_mps": fit_rmse,
        "forward_response_condition_number": float(np.linalg.cond(slopes)),
        "target_inside_measured_convex_support": True,
        "local_plasticity_gain": config.local_plasticity_gain,
    }
    return actor, diagnostics


def run_role_isolated_contact_actor_growth(
    *,
    asset_root: Path,
    striker_actor_path: Path,
    goalkeeper_actor_path: Path,
    gmt_model_path: Path,
    gmt_skill_path: Path,
    dive_source_checkout: Path,
    dive_athlete_checkpoint_path: Path,
    dive_athlete_exam_path: Path,
    recovery_athlete_checkpoint_path: Path,
    recovery_athlete_exam_path: Path,
    output_dir: Path,
    config: RoleIsolatedContactActorGrowthConfig | None = None,
) -> dict[str, Any]:
    active = config or RoleIsolatedContactActorGrowthConfig()
    root = asset_root.expanduser().resolve()
    destination = output_dir.expanduser().resolve()
    checkout = Path(__file__).resolve().parents[3]
    if destination.exists() or destination == checkout or checkout in destination.parents:
        raise ValueError("role-isolated contact growth requires a new external directory")
    assets = {
        "striker_actor": striker_actor_path.expanduser().resolve(),
        "goalkeeper_actor": goalkeeper_actor_path.expanduser().resolve(),
        "gmt_model": gmt_model_path.expanduser().resolve(),
        "gmt_skill": gmt_skill_path.expanduser().resolve(),
        "dive_source": dive_source_checkout.expanduser().resolve(),
        "dive_athlete_checkpoint": dive_athlete_checkpoint_path.expanduser().resolve(),
        "dive_athlete_exam": dive_athlete_exam_path.expanduser().resolve(),
        "recovery_athlete_checkpoint": recovery_athlete_checkpoint_path.expanduser().resolve(),
        "recovery_athlete_exam": recovery_athlete_exam_path.expanduser().resolve(),
    }
    if not root.is_dir() or not assets["dive_source"].is_dir():
        raise FileNotFoundError("role-isolated contact-growth root is missing")
    if any(not path.is_file() for key, path in assets.items() if key != "dive_source"):
        raise FileNotFoundError("role-isolated contact-growth artifact is missing")
    source_trees, artifacts = _closure_inputs(asset_root=root, assets=assets)
    closure = build_reproducibility_closure(
        source_trees=source_trees,
        dependency_packages=("mujoco", "numpy", "onnxruntime"),
        artifacts=artifacts,
        expected_replays=active.probe_replay_count,
    )
    request = {
        "schema_version": "rosclaw_soccer.role_isolated_contact_actor_growth_request.v1",
        "config": asdict(active),
        "config_hash": active.config_hash,
        "probes": [asdict(probe) for probe in _PROBES],
        "source_tree_locators": {key: str(value) for key, value in source_trees.items()},
        "artifact_locators": {key: str(value) for key, value in artifacts.items()},
        "reproducibility_closure": closure.to_dict(),
        "reproducibility_closure_hash": closure.closure_hash,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    destination.mkdir(parents=True)
    request_path = destination / "request.json"
    _atomic_json(request_path, request)
    lane = next(item for item in expanded_dynamic_corner_lanes() if item.lane_id == active.lane_id)
    exam = ContinuousSecondStrikerSaveExamConfig(
        lane_ids=(active.lane_id,), simulation_duration_sec=active.simulation_duration_sec
    )
    kwargs, goalkeeper, goal = physical_second_striker_kwargs(
        lane=lane,
        assets=assets,
        recovery_checkpoint=assets["recovery_athlete_checkpoint"],
        recovery_exam=assets["recovery_athlete_exam"],
        config=exam,
    )
    rows: list[dict[str, Any]] = []
    for probe in _PROBES:
        replays: list[dict[str, Any]] = []
        for replay_index in range(active.probe_replay_count):
            probe_kwargs = dict(kwargs)
            probe_kwargs["second_striker_loft_teacher_config"] = probe.teacher_config()
            result, trajectory = simulate_shared_world(root, **probe_kwargs)
            evaluation = evaluate_continuous_second_striker_save(
                result=result,
                trajectory=trajectory,
                lane=lane,
                goal=goal,
                goalkeeper=goalkeeper,
                config=exam,
            )
            trajectory_path = destination / f"{probe.label}-replay-{replay_index}.npz"
            _atomic_trajectory(trajectory_path, trajectory)
            replay = _probe_row(
                probe=probe,
                trajectory=trajectory,
                result=result.to_dict(),
                evaluation=evaluation,
            )
            replay.update(
                trajectory_file=trajectory_path.name,
                trajectory_hash=hash_bytes(trajectory_path.read_bytes()),
                trajectory_digest=trajectory_digest(trajectory),
            )
            replays.append(replay)
        semantic_keys = (
            "hard_safe",
            "teacher_success",
            "teacher_success_gates",
            "launch_velocity_xyz_mps",
            "teacher_peak_force_yz_n",
            "teacher_peak_foot_velocity_yz_mps",
            "result",
            "evaluation",
            "trajectory_hash",
            "trajectory_digest",
        )
        strict_replay = all(
            {key: replay.get(key) for key in semantic_keys}
            == {key: replays[0].get(key) for key in semantic_keys}
            for replay in replays[1:]
        )
        if not strict_replay:
            raise ValueError(f"role-isolated probe {probe.label} is not an exact replay")
        row = dict(replays[0])
        row["strict_replay"] = True
        row["replays"] = replays
        _atomic_json(destination / f"{probe.label}.json", row)
        rows.append(row)
    parent = load_g1_ballistic_contact_impulse_actor(assets["striker_actor"])
    context_hash = str(
        hash_json(
            {
                "body_hash": parent.body_hash,
                "config": asdict(active),
                "goal": asdict(goal),
                "striker": asdict(exam.striker),
                "probes": [asdict(probe) for probe in _PROBES],
            }
        )
    )
    actor, fit = _fit_candidate(
        rows=rows,
        parent_actor=parent,
        striker=exam.striker,
        goal_x_m=goal.plane_x_m,
        context_hash=context_hash,
        config=active,
    )
    actor_path = destination / "role-isolated-contact-actor.json"
    _atomic_json(actor_path, actor.to_dict())
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.role_isolated_contact_actor_growth_evidence.v1",
        "claim": _CLAIM,
        "candidate_derived": True,
        "candidate_promoted": False,
        "candidate_status": "UNQUALIFIED_SIM_ONLY_CANDIDATE",
        "actor_file": actor_path.name,
        "actor_hash": actor.actor_hash,
        "actor_file_hash": hash_bytes(actor_path.read_bytes()),
        "fit": fit,
        "probes": rows,
        "request_hash": hash_bytes(request_path.read_bytes()),
        "reproducibility_closure_hash": closure.closure_hash,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "commercial_use_allowed": False,
        "pixels_used_for_scoring": False,
    }
    report["report_hash"] = hash_json(report)
    evidence_path = destination / "evidence.json"
    _atomic_json(evidence_path, report)
    return validate_role_isolated_contact_actor_growth(evidence_path)


def validate_role_isolated_contact_actor_growth(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("role-isolated contact-growth evidence must be an object")
    expected_hash = payload.pop("report_hash", None)
    try:
        request_path = resolved.parent / "request.json"
        request = json.loads(request_path.read_text(encoding="utf-8"))
        config_value = request.get("config")
        source_locators = request.get("source_tree_locators")
        artifact_locators = request.get("artifact_locators")
        closure_value = request.get("reproducibility_closure")
        if not all(
            isinstance(value, dict)
            for value in (config_value, source_locators, artifact_locators, closure_value)
        ):
            raise ValueError("role-isolated contact-growth request is invalid")
        active = RoleIsolatedContactActorGrowthConfig(**cast(dict[str, Any], config_value))
        recorded_closure = ReproducibilityClosure.from_dict(cast(dict[str, Any], closure_value))
        closure = build_reproducibility_closure(
            source_trees={
                key: Path(value) for key, value in cast(dict[str, str], source_locators).items()
            },
            dependency_packages=("mujoco", "numpy", "onnxruntime"),
            artifacts={
                key: Path(value) for key, value in cast(dict[str, str], artifact_locators).items()
            },
            expected_replays=active.probe_replay_count,
        )
        rows = payload.get("probes")
        if not isinstance(rows, list) or len(rows) != len(_PROBES):
            raise ValueError("role-isolated contact-growth probe set is incomplete")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("role-isolated contact-growth probe is invalid")
            replays = row.get("replays")
            if (
                row.get("strict_replay") is not True
                or not isinstance(replays, list)
                or len(replays) != active.probe_replay_count
            ):
                raise ValueError("role-isolated contact-growth replay set is incomplete")
            for replay in replays:
                if not isinstance(replay, dict):
                    raise ValueError("role-isolated contact-growth replay is invalid")
                trajectory_path = resolved.parent / str(replay.get("trajectory_file", ""))
                if not trajectory_path.is_file() or replay.get("trajectory_hash") != hash_bytes(
                    trajectory_path.read_bytes()
                ):
                    raise ValueError("role-isolated contact-growth trajectory binding changed")
                with np.load(trajectory_path, allow_pickle=False) as archive:
                    trajectory = {key: np.asarray(archive[key]) for key in archive.files}
                if replay.get("trajectory_digest") != trajectory_digest(trajectory):
                    raise ValueError("role-isolated contact-growth trajectory semantics changed")
            semantic_keys = (
                "hard_safe",
                "teacher_success",
                "teacher_success_gates",
                "launch_velocity_xyz_mps",
                "teacher_peak_force_yz_n",
                "teacher_peak_foot_velocity_yz_mps",
                "result",
                "evaluation",
                "trajectory_hash",
                "trajectory_digest",
            )
            if any(
                {key: replay.get(key) for key in semantic_keys}
                != {key: replays[0].get(key) for key in semantic_keys}
                for replay in replays[1:]
            ):
                raise ValueError("role-isolated contact-growth replay semantics diverged")
        actor_path = resolved.parent / str(payload.get("actor_file", ""))
        actor = load_g1_ballistic_contact_impulse_actor(actor_path)
        if (
            request.get("schema_version")
            != "rosclaw_soccer.role_isolated_contact_actor_growth_request.v1"
            or request.get("config_hash") != active.config_hash
            or request.get("probes") != [asdict(probe) for probe in _PROBES]
            or request.get("activation_ceiling") != "SIM_ONLY"
            or request.get("hardware_command_sent") is not False
            or recorded_closure != closure
            or request.get("reproducibility_closure_hash") != closure.closure_hash
            or payload.get("reproducibility_closure_hash") != closure.closure_hash
            or payload.get("request_hash") != hash_bytes(request_path.read_bytes())
            or payload.get("schema_version")
            != "rosclaw_soccer.role_isolated_contact_actor_growth_evidence.v1"
            or payload.get("claim") != _CLAIM
            or payload.get("candidate_derived") is not True
            or payload.get("candidate_promoted") is not False
            or payload.get("candidate_status") != "UNQUALIFIED_SIM_ONLY_CANDIDATE"
            or payload.get("actor_hash") != actor.actor_hash
            or payload.get("actor_file_hash") != hash_bytes(actor_path.read_bytes())
            or payload.get("activation_ceiling") != "SIM_ONLY"
            or payload.get("hardware_command_sent") is not False
            or payload.get("commercial_use_allowed") is not False
            or payload.get("pixels_used_for_scoring") is not False
            or expected_hash != hash_json(payload)
        ):
            raise ValueError("role-isolated contact-growth authority contract is invalid")
    finally:
        if expected_hash is not None:
            payload["report_hash"] = expected_hash
    return cast(dict[str, Any], payload)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--striker-actor", type=Path, required=True)
    parser.add_argument("--goalkeeper-actor", type=Path, required=True)
    parser.add_argument("--gmt-model", type=Path, required=True)
    parser.add_argument("--gmt-skill", type=Path, required=True)
    parser.add_argument("--dive-source", type=Path, required=True)
    parser.add_argument("--dive-athlete-checkpoint", type=Path, required=True)
    parser.add_argument("--dive-athlete-exam", type=Path, required=True)
    parser.add_argument("--recovery-athlete-checkpoint", type=Path, required=True)
    parser.add_argument("--recovery-athlete-exam", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = run_role_isolated_contact_actor_growth(
        asset_root=args.asset_root,
        striker_actor_path=args.striker_actor,
        goalkeeper_actor_path=args.goalkeeper_actor,
        gmt_model_path=args.gmt_model,
        gmt_skill_path=args.gmt_skill,
        dive_source_checkout=args.dive_source,
        dive_athlete_checkpoint_path=args.dive_athlete_checkpoint,
        dive_athlete_exam_path=args.dive_athlete_exam,
        recovery_athlete_checkpoint_path=args.recovery_athlete_checkpoint,
        recovery_athlete_exam_path=args.recovery_athlete_exam,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RoleIsolatedContactActorGrowthConfig",
    "RoleIsolatedContactTeacherProbe",
    "run_role_isolated_contact_actor_growth",
    "validate_role_isolated_contact_actor_growth",
]
