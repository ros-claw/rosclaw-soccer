"""Cross-process requalification of the current continuous-team champion.

S110 proved that two replays inside one process are not a sufficient evidence
boundary: its selected implementation hashes omitted transitive modules, and
an otherwise identical later process produced a different physical outcome.
This exam reconstructs the frozen right-control cell, launches fresh Python
interpreters, and promotes only byte-identical, all-gate full-chain replays.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from rosclaw.feedback import contracts as growth_core_contracts

from rosclaw_soccer.providers.g1.asset_qualification import (
    qualify_g1_assets,
    trajectory_digest,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.shared_world import simulate_shared_world
from rosclaw_soccer.training.bilateral_continuous_team_population import (
    BilateralContinuousTeamPopulationConfig,
    validate_bilateral_continuous_team_population,
)
from rosclaw_soccer.training.continuous_second_striker_save_exam import (
    evaluate_continuous_second_striker_save,
    physical_second_striker_kwargs,
)
from rosclaw_soccer.training.dynamic_corner_save import (
    DynamicCornerSaveLane,
    expanded_dynamic_corner_lanes,
)

_CLAIM = "TRUE_AIRBORNE_SAVE_WITH_FOOT_CONTACT_GROUNDED_LANDING"
_REQUALIFICATION_CLAIM = "CROSS_PROCESS_CONTINUOUS_FOUR_G1_CURRENT_RUNTIME_CHAMPION"
_STATUS = "PROMOTED_SIM_ONLY_CURRENT_RUNTIME_FULL_CHAIN"
_THREAD_ENVIRONMENT_KEYS = (
    "PYTHONHASHSEED",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "ORT_NUM_THREADS",
)


@dataclass(frozen=True)
class CurrentRuntimePrefixRequalificationConfig:
    """Exact S110 right-control cell with a stronger process boundary."""

    lane_id: str = "left-inner"
    case_id: str = "right-control"
    ball_mass_kg: float = 0.41
    ball_ground_friction: float = 0.10
    simulation_duration_sec: float = 25.0
    cross_process_replays: int = 3
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    commercial_use_allowed: bool = False
    schema_version: str = "rosclaw_soccer.current_runtime_prefix_requalification_config.v2"

    def __post_init__(self) -> None:
        values = (
            self.ball_mass_kg,
            self.ball_ground_friction,
            self.simulation_duration_sec,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("current-runtime requalification values must be finite")
        if (
            self.lane_id != "left-inner"
            or self.case_id != "right-control"
            or abs(self.ball_mass_kg - 0.41) > 1.0e-12
            or abs(self.ball_ground_friction - 0.10) > 1.0e-12
            or not 23.0 <= self.simulation_duration_sec <= 25.0
            or not 2 <= self.cross_process_replays <= 4
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
            or self.commercial_use_allowed
        ):
            raise ValueError("current-runtime requalification contract is invalid")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


def current_runtime_prefix_lane(lane_id: str = "left-inner") -> DynamicCornerSaveLane:
    """Return the inherited lane without changing control or scoring values."""

    lanes = {lane.lane_id: lane for lane in expanded_dynamic_corner_lanes()}
    try:
        return lanes[lane_id]
    except KeyError as error:
        raise ValueError("current-runtime prefix lane is unknown") from error


def prefix_gate_contract(lane: DynamicCornerSaveLane) -> dict[str, Any]:
    """Extract the inherited contact, take-off, landing and recovery gates."""

    takeoff = lane.takeoff_config
    lunge = takeoff.lunge_config
    aerial = lunge.aerial_config
    return {
        "takeoff": {
            name: getattr(takeoff, name)
            for name in (
                "minimum_airborne_duration_sec",
                "minimum_takeoff_vertical_speed_mps",
                "minimum_flight_pelvis_rise_m",
                "maximum_landing_vertical_speed_mps",
                "maximum_landing_angular_speed_rad_s",
                "contact_time_tolerance_sec",
            )
        },
        "lunge": {
            name: getattr(lunge, name)
            for name in (
                "minimum_lunge_span_m",
                "minimum_peak_lateral_speed_mps",
                "minimum_controlled_pelvis_height_m",
                "minimum_recovery_pelvis_height_m",
                "minimum_recovery_upright_projection",
                "maximum_recovery_linear_speed_mps",
            )
        },
        "aerial": {
            name: getattr(aerial, name)
            for name in (
                "minimum_pass_precision_m",
                "minimum_aerial_apex_m",
                "minimum_glove_contact_height_m",
                "minimum_hand_height_m",
                "minimum_passer_pelvis_height_m",
                "minimum_shooter_pelvis_height_m",
                "minimum_goalkeeper_pelvis_height_m",
                "minimum_incoming_speed_mps",
                "minimum_outward_speed_mps",
                "maximum_post_contact_speed_mps",
                "maximum_glove_surface_separation_m",
                "maximum_glove_surface_penetration_m",
            )
        },
    }


def _right_control_contract(
    config: CurrentRuntimePrefixRequalificationConfig,
) -> tuple[DynamicCornerSaveLane, Any, Any]:
    population = BilateralContinuousTeamPopulationConfig()
    cases = {case.case_id: case for case in population.cases}
    case = cases[config.case_id]
    lane = current_runtime_prefix_lane(config.lane_id)
    exam = replace(
        population.exam,
        lane_ids=(lane.lane_id,),
        simulation_duration_sec=config.simulation_duration_sec,
        striker=case.striker,
    )
    return lane, case, exam


def _assemble_rollout(
    *, config: CurrentRuntimePrefixRequalificationConfig, assets: dict[str, Path]
) -> tuple[DynamicCornerSaveLane, Any, dict[str, Any], Any, Any]:
    lane, _, exam = _right_control_contract(config)
    kwargs, goalkeeper, base_goal = physical_second_striker_kwargs(
        lane=lane,
        assets=assets,
        recovery_checkpoint=assets["recovery_athlete_checkpoint"],
        recovery_exam=assets["recovery_athlete_exam"],
        config=exam,
    )
    goal = replace(base_goal, ball_mass_kg=config.ball_mass_kg)
    kwargs["goal_spec"] = goal
    kwargs["ball_ground_friction"] = config.ball_ground_friction
    return lane, exam, kwargs, goalkeeper, goal


def _worker_rollout(
    *,
    asset_root: Path,
    assets: dict[str, Path],
    config: CurrentRuntimePrefixRequalificationConfig,
    output_prefix: Path,
) -> dict[str, Any]:
    lane, exam, kwargs, goalkeeper, goal = _assemble_rollout(config=config, assets=assets)
    result, trajectory = simulate_shared_world(asset_root, **kwargs)
    evaluation = evaluate_continuous_second_striker_save(
        result=result,
        trajectory=trajectory,
        lane=lane,
        goal=goal,
        goalkeeper=goalkeeper,
        config=exam,
    )
    trajectory_path = output_prefix.with_suffix(".npz")
    _atomic_trajectory(trajectory_path, trajectory)
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.current_runtime_process_replay.v1",
        "process_id": os.getpid(),
        "passed": evaluation.get("passed") is True,
        "evaluation": evaluation,
        "trajectory_file": trajectory_path.name,
        "trajectory_hash": hash_bytes(trajectory_path.read_bytes()),
        "trajectory_digest": trajectory_digest(trajectory),
        "process_contract": _process_contract(),
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    _atomic_json(output_prefix.with_suffix(".json"), report)
    return report


def run_current_runtime_prefix_requalification(
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
    predecessor_evidence_path: Path,
    output_dir: Path,
    source_checkout: Path,
    config: CurrentRuntimePrefixRequalificationConfig | None = None,
) -> dict[str, Any]:
    """Run the current right-control champion in fresh Python processes."""

    active = config or CurrentRuntimePrefixRequalificationConfig()
    output = output_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    predecessor_path = predecessor_evidence_path.expanduser().resolve()
    if output.exists() or output == checkout or checkout in output.parents:
        raise ValueError("current-runtime evidence must use a new external directory")
    assets = _resolve_assets(
        striker_actor_path=striker_actor_path,
        goalkeeper_actor_path=goalkeeper_actor_path,
        gmt_model_path=gmt_model_path,
        gmt_skill_path=gmt_skill_path,
        dive_source_checkout=dive_source_checkout,
        dive_athlete_checkpoint_path=dive_athlete_checkpoint_path,
        dive_athlete_exam_path=dive_athlete_exam_path,
        recovery_athlete_checkpoint_path=recovery_athlete_checkpoint_path,
        recovery_athlete_exam_path=recovery_athlete_exam_path,
    )
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    predecessor = validate_bilateral_continuous_team_population(predecessor_path)
    predecessor_case = predecessor.get("cases", {}).get(active.case_id, {})
    predecessor_rejected = bool(
        predecessor.get("passed") is False
        and predecessor.get("promotion_status") == "REJECTED_BILATERAL_POPULATION"
        and isinstance(predecessor_case, dict)
        and predecessor_case.get("passed") is False
    )
    lane, case, exam = _right_control_contract(active)
    _, _, _, _, goal = _assemble_rollout(config=active, assets=assets)
    request: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.current_runtime_requalification_request.v2",
        "config": asdict(active),
        "config_hash": active.config_hash,
        "continuous_exam": asdict(exam),
        "case": asdict(case),
        "prefix_gate_contract": prefix_gate_contract(lane),
        "goal_spec": asdict(goal),
        "body_hash": qualification.body_hash,
        "kick_prior_hash": qualification.kick_prior_hash,
        "source_commit": _git_head(checkout),
        "soccer_source_tree_hash": _python_tree_hash(Path(__file__).resolve().parents[1]),
        "growth_core_source_tree_hash": _python_tree_hash(
            Path(inspect.getfile(growth_core_contracts)).resolve().parents[2]
        ),
        "runtime_dependencies": _runtime_dependency_contract(),
        "process_contract": _process_contract(),
        "artifacts": {
            key: (_git_head(value) if key == "dive_source" else hash_bytes(value.read_bytes()))
            for key, value in assets.items()
        },
        "predecessor_evidence_hash": hash_bytes(predecessor_path.read_bytes()),
        "predecessor_report_hash": predecessor.get("report_hash"),
        "physics_authority": "CPU_MUJOCO",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "reset_or_teleport_used": False,
        "ball_cannon_used": False,
    }
    output.mkdir(parents=True)
    request_path = output / "request.json"
    _atomic_json(request_path, request)
    worker_reports: list[dict[str, Any]] = []
    for index in range(active.cross_process_replays):
        prefix = output / f"process-replay-{index}"
        command = _worker_command(
            asset_root=asset_root,
            assets=assets,
            config=active,
            output_prefix=prefix,
        )
        completed = subprocess.run(command, cwd=checkout, check=False)
        report_path = prefix.with_suffix(".json")
        if completed.returncode or not report_path.is_file():
            raise RuntimeError(f"current-runtime worker {index} failed")
        worker_reports.append(
            _validate_worker_report(report_path, expected_prefix=prefix, require_passed=True)
        )
    first = worker_reports[0]
    reference_payload = {
        "evaluation": first["evaluation"],
        "trajectory_digest": first["trajectory_digest"],
    }
    process_contracts = [report["process_contract"] for report in worker_reports]
    process_ids = {int(report["process_id"]) for report in worker_reports}
    exact_replays = all(
        {
            "evaluation": report["evaluation"],
            "trajectory_digest": report["trajectory_digest"],
        }
        == reference_payload
        for report in worker_reports
    )
    continuous = cast(dict[str, Any], first["evaluation"])
    prefix_exam = cast(dict[str, Any], continuous.get("first_takeoff_exam", {}))
    qualification_gates = {
        "predecessor_rejection_bound": predecessor_rejected,
        "fresh_process_count": len(process_ids) == active.cross_process_replays,
        "process_contract_identical": all(
            value == process_contracts[0] for value in process_contracts
        ),
        "cross_process_exact_replay": exact_replays,
        "all_prefix_physics_gates": prefix_exam.get("passed") is True,
        "all_continuous_team_gates": continuous.get("passed") is True,
        "all_workers_sim_only_safe": all(report.get("passed") is True for report in worker_reports),
        "full_source_trees_bound": True,
        "current_runtime_bound": True,
        "sim_only_authority": True,
    }
    passed = bool(all(qualification_gates.values()))
    first_trajectory = output / str(first["trajectory_file"])
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.current_runtime_requalification_evidence.v2",
        "claim": _CLAIM,
        "requalification_claim": _REQUALIFICATION_CLAIM,
        "passed": passed,
        "promotion_status": "FROZEN_RESEARCH_DEMO" if passed else "REJECTED_DEVELOPMENT",
        "requalification_status": _STATUS if passed else "REJECTED_CURRENT_RUNTIME_FULL_CHAIN",
        "qualification_gates": qualification_gates,
        "strict_replay": exact_replays,
        "cross_process_replay_count": len(worker_reports),
        "process_ids": sorted(process_ids),
        "physics_authority": "CPU_MUJOCO",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "commercial_use_allowed": False,
        "reset_or_teleport_used": False,
        "ball_cannon_used": False,
        "pixels_used_for_scoring": False,
        "request_hash": hash_bytes(request_path.read_bytes()),
        "predecessor": {
            "path": str(predecessor_path),
            "evidence_hash": request["predecessor_evidence_hash"],
            "report_hash": predecessor.get("report_hash"),
            "passed": predecessor.get("passed"),
            "right_control_passed": predecessor_case.get("passed"),
        },
        "worker_reports": [
            {
                "report_file": f"process-replay-{index}.json",
                "report_hash": worker["report_hash"],
                "trajectory_file": worker["trajectory_file"],
                "trajectory_hash": worker["trajectory_hash"],
                "trajectory_digest": worker["trajectory_digest"],
                "process_id": worker["process_id"],
            }
            for index, worker in enumerate(worker_reports)
        ],
        "trajectory_file": first_trajectory.name,
        "trajectory_hash": hash_bytes(first_trajectory.read_bytes()),
        "implementation_hash": _implementation_hash(),
        "continuous": continuous,
        "first": prefix_exam,
        "replay": prefix_exam,
    }
    report["report_hash"] = hash_json(report)
    evidence_path = output / "evidence.json"
    _atomic_json(evidence_path, report)
    return validate_current_runtime_prefix_requalification(evidence_path)


def validate_current_runtime_prefix_requalification(path: Path) -> dict[str, Any]:
    """Validate all fresh-process reports, trajectories and authority fields."""

    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("current-runtime evidence must be a JSON object")
    expected = payload.pop("report_hash", None)
    try:
        request_path = resolved.parent / "request.json"
        if not request_path.is_file() or payload.get("request_hash") != hash_bytes(
            request_path.read_bytes()
        ):
            raise ValueError("current-runtime request binding changed")
        request = json.loads(request_path.read_text(encoding="utf-8"))
        soccer_root = Path(__file__).resolve().parents[1]
        growth_root = Path(inspect.getfile(growth_core_contracts)).resolve().parents[2]
        if (
            not isinstance(request, dict)
            or request.get("soccer_source_tree_hash") != _python_tree_hash(soccer_root)
            or request.get("growth_core_source_tree_hash") != _python_tree_hash(growth_root)
            or request.get("runtime_dependencies") != _runtime_dependency_contract()
            or request.get("process_contract") != _process_contract()
            or payload.get("implementation_hash") != _implementation_hash()
        ):
            raise ValueError("current-runtime source or process closure changed")
        request_config = request.get("config")
        workers = payload.get("worker_reports")
        if not isinstance(workers, list) or len(workers) < 2:
            raise ValueError("current-runtime process replay set is incomplete")
        if not isinstance(request_config, dict):
            raise ValueError("current-runtime request config is absent")
        expected_replays = request_config.get("cross_process_replays")
        if expected_replays != len(workers):
            raise ValueError("current-runtime process replay count changed")
        validated_workers: list[dict[str, Any]] = []
        for worker in workers:
            if not isinstance(worker, dict):
                raise ValueError("current-runtime worker binding is invalid")
            report_file = (resolved.parent / str(worker.get("report_file", ""))).resolve()
            trajectory_file = (resolved.parent / str(worker.get("trajectory_file", ""))).resolve()
            if report_file.parent != resolved.parent or trajectory_file.parent != resolved.parent:
                raise ValueError("current-runtime worker path escaped evidence directory")
            validated = _validate_worker_report(
                report_file,
                expected_prefix=report_file.with_suffix(""),
                require_passed=True,
            )
            if (
                validated.get("report_hash") != worker.get("report_hash")
                or validated.get("trajectory_hash") != worker.get("trajectory_hash")
                or not trajectory_file.is_file()
                or hash_bytes(trajectory_file.read_bytes()) != worker.get("trajectory_hash")
                or validated.get("trajectory_digest") != worker.get("trajectory_digest")
                or validated.get("process_id") != worker.get("process_id")
            ):
                raise ValueError("current-runtime worker trajectory binding changed")
            validated_workers.append(validated)
        predecessor = payload.get("predecessor")
        if not isinstance(predecessor, dict):
            raise ValueError("current-runtime predecessor binding is absent")
        predecessor_path = Path(str(predecessor.get("path", ""))).expanduser().resolve()
        if not predecessor_path.is_file() or hash_bytes(
            predecessor_path.read_bytes()
        ) != predecessor.get("evidence_hash"):
            raise ValueError("current-runtime predecessor binding changed")
        process_contract = request.get("process_contract")
        raw_process_ids = [worker.get("process_id") for worker in validated_workers]
        if not all(
            isinstance(process_id, int) and process_id > 0 for process_id in raw_process_ids
        ):
            raise ValueError("current-runtime worker process identity is invalid")
        process_ids = {cast(int, process_id) for process_id in raw_process_ids}
        reference = {
            "evaluation": validated_workers[0].get("evaluation"),
            "trajectory_digest": validated_workers[0].get("trajectory_digest"),
        }
        exact_replay = all(
            {
                "evaluation": worker.get("evaluation"),
                "trajectory_digest": worker.get("trajectory_digest"),
            }
            == reference
            for worker in validated_workers
        )
        continuous = validated_workers[0].get("evaluation")
        prefix = continuous.get("first_takeoff_exam") if isinstance(continuous, dict) else None
        derived_gates = {
            "predecessor_rejection_bound": predecessor.get("passed") is False
            and predecessor.get("right_control_passed") is False,
            "fresh_process_count": len(process_ids) == expected_replays,
            "process_contract_identical": all(
                worker.get("process_contract") == process_contract for worker in validated_workers
            ),
            "cross_process_exact_replay": exact_replay,
            "all_prefix_physics_gates": isinstance(prefix, dict) and prefix.get("passed") is True,
            "all_continuous_team_gates": isinstance(continuous, dict)
            and continuous.get("passed") is True,
            "all_workers_sim_only_safe": all(
                worker.get("passed") is True
                and worker.get("activation_ceiling") == "SIM_ONLY"
                and worker.get("hardware_command_sent") is False
                for worker in validated_workers
            ),
            "full_source_trees_bound": True,
            "current_runtime_bound": True,
            "sim_only_authority": True,
        }
        gates = payload.get("qualification_gates")
        passed = all(derived_gates.values())
        first_worker = workers[0]
        if (
            payload.get("schema_version")
            != "rosclaw_soccer.current_runtime_requalification_evidence.v2"
            or payload.get("claim") != _CLAIM
            or payload.get("requalification_claim") != _REQUALIFICATION_CLAIM
            or gates != derived_gates
            or payload.get("passed") is not passed
            or payload.get("strict_replay") is not exact_replay
            or payload.get("cross_process_replay_count") != len(workers)
            or payload.get("process_ids") != sorted(process_ids)
            or payload.get("continuous") != continuous
            or payload.get("first") != prefix
            or payload.get("replay") != prefix
            or payload.get("trajectory_file") != first_worker.get("trajectory_file")
            or payload.get("trajectory_hash") != first_worker.get("trajectory_hash")
            or payload.get("promotion_status")
            != ("FROZEN_RESEARCH_DEMO" if passed else "REJECTED_DEVELOPMENT")
            or payload.get("requalification_status")
            != (_STATUS if passed else "REJECTED_CURRENT_RUNTIME_FULL_CHAIN")
            or payload.get("physics_authority") != "CPU_MUJOCO"
            or payload.get("activation_ceiling") != "SIM_ONLY"
            or payload.get("hardware_command_sent") is not False
            or payload.get("commercial_use_allowed") is not False
            or payload.get("reset_or_teleport_used") is not False
            or payload.get("ball_cannon_used") is not False
            or payload.get("pixels_used_for_scoring") is not False
            or expected != hash_json(payload)
        ):
            raise ValueError("current-runtime authority or integrity contract is invalid")
    finally:
        if expected is not None:
            payload["report_hash"] = expected
    return cast(dict[str, Any], payload)


def _validate_worker_report(
    path: Path, *, expected_prefix: Path, require_passed: bool
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("current-runtime worker report must be an object")
    expected = payload.pop("report_hash", None)
    try:
        trajectory = expected_prefix.with_suffix(".npz")
        trajectory_hash = hash_bytes(trajectory.read_bytes()) if trajectory.is_file() else None
        trajectory_value: dict[str, NDArray[Any]] = {}
        if trajectory.is_file():
            try:
                with np.load(trajectory, allow_pickle=False) as archive:
                    trajectory_value = {name: archive[name] for name in archive.files}
            except (OSError, ValueError) as error:
                raise ValueError("current-runtime worker trajectory is unreadable") from error
        evaluation = payload.get("evaluation")
        if (
            payload.get("schema_version") != "rosclaw_soccer.current_runtime_process_replay.v1"
            or payload.get("passed") is not require_passed
            or not isinstance(evaluation, dict)
            or evaluation.get("passed") is not require_passed
            or payload.get("activation_ceiling") != "SIM_ONLY"
            or payload.get("hardware_command_sent") is not False
            or payload.get("trajectory_file") != trajectory.name
            or not trajectory.is_file()
            or not trajectory_value
            or payload.get("trajectory_hash") != trajectory_hash
            or payload.get("trajectory_digest") != trajectory_digest(trajectory_value)
            or expected != hash_json(payload)
        ):
            raise ValueError("current-runtime worker report integrity changed")
    finally:
        if expected is not None:
            payload["report_hash"] = expected
    return cast(dict[str, Any], payload)


def _resolve_assets(**values: Path) -> dict[str, Path]:
    names = {
        "striker_actor_path": "striker_actor",
        "goalkeeper_actor_path": "goalkeeper_actor",
        "gmt_model_path": "gmt_model",
        "gmt_skill_path": "gmt_skill",
        "dive_source_checkout": "dive_source",
        "dive_athlete_checkpoint_path": "dive_athlete_checkpoint",
        "dive_athlete_exam_path": "dive_athlete_exam",
        "recovery_athlete_checkpoint_path": "recovery_athlete_checkpoint",
        "recovery_athlete_exam_path": "recovery_athlete_exam",
    }
    assets = {names[key]: value.expanduser().resolve() for key, value in values.items()}
    files = tuple(value for key, value in assets.items() if key != "dive_source")
    if not all(path.is_file() for path in files) or not (assets["dive_source"] / ".git").exists():
        raise FileNotFoundError("current-runtime input artifact is missing")
    return assets


def _worker_command(
    *,
    asset_root: Path,
    assets: dict[str, Path],
    config: CurrentRuntimePrefixRequalificationConfig,
    output_prefix: Path,
) -> tuple[str, ...]:
    return (
        sys.executable,
        "-m",
        "rosclaw_soccer.training.current_runtime_prefix_requalification",
        "--worker-output-prefix",
        str(output_prefix),
        "--asset-root",
        str(asset_root),
        "--striker-actor",
        str(assets["striker_actor"]),
        "--goalkeeper-actor",
        str(assets["goalkeeper_actor"]),
        "--gmt-model",
        str(assets["gmt_model"]),
        "--gmt-skill",
        str(assets["gmt_skill"]),
        "--dive-source",
        str(assets["dive_source"]),
        "--dive-athlete-checkpoint",
        str(assets["dive_athlete_checkpoint"]),
        "--dive-athlete-exam",
        str(assets["dive_athlete_exam"]),
        "--recovery-athlete-checkpoint",
        str(assets["recovery_athlete_checkpoint"]),
        "--recovery-athlete-exam",
        str(assets["recovery_athlete_exam"]),
        "--cross-process-replays",
        str(config.cross_process_replays),
    )


def _python_tree_hash(root: Path) -> str:
    files = sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)
    if not files:
        raise ValueError("current-runtime source tree is empty")
    return str(
        hash_json({str(path.relative_to(root)): hash_bytes(path.read_bytes()) for path in files})
    )


def _runtime_dependency_contract() -> dict[str, str]:
    return {
        package: importlib.metadata.version(package)
        for package in ("mujoco", "numpy", "onnxruntime")
    }


def _process_contract() -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "libc": list(platform.libc_ver()),
        "cpu_count": os.cpu_count(),
        "hash_randomization": int(sys.flags.hash_randomization),
        "thread_environment": {key: os.environ.get(key) for key in _THREAD_ENVIRONMENT_KEYS},
    }


def _implementation_hash() -> str:
    root = Path(__file__).resolve().parents[1]
    core = Path(inspect.getfile(growth_core_contracts)).resolve().parents[2]
    return str(
        hash_json(
            {
                "soccer_source_tree": _python_tree_hash(root),
                "growth_core_source_tree": _python_tree_hash(core),
            }
        )
    )


def _git_head(checkout: Path) -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


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
    parser.add_argument("--predecessor-evidence", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--source-checkout", type=Path)
    parser.add_argument("--worker-output-prefix", type=Path)
    parser.add_argument("--cross-process-replays", type=int, default=3)
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = CurrentRuntimePrefixRequalificationConfig(
        cross_process_replays=args.cross_process_replays
    )
    assets = _resolve_assets(
        striker_actor_path=cast(Path, args.striker_actor),
        goalkeeper_actor_path=cast(Path, args.goalkeeper_actor),
        gmt_model_path=cast(Path, args.gmt_model),
        gmt_skill_path=cast(Path, args.gmt_skill),
        dive_source_checkout=cast(Path, args.dive_source),
        dive_athlete_checkpoint_path=cast(Path, args.dive_athlete_checkpoint),
        dive_athlete_exam_path=cast(Path, args.dive_athlete_exam),
        recovery_athlete_checkpoint_path=cast(Path, args.recovery_athlete_checkpoint),
        recovery_athlete_exam_path=cast(Path, args.recovery_athlete_exam),
    )
    worker_prefix = cast(Path | None, args.worker_output_prefix)
    if worker_prefix is not None:
        report = _worker_rollout(
            asset_root=cast(Path, args.asset_root),
            assets=assets,
            config=config,
            output_prefix=worker_prefix,
        )
        return 0 if report.get("passed") is True else 1
    if args.predecessor_evidence is None or args.output_dir is None or args.source_checkout is None:
        raise SystemExit("parent mode requires predecessor evidence, output dir and checkout")
    report = run_current_runtime_prefix_requalification(
        asset_root=cast(Path, args.asset_root),
        striker_actor_path=assets["striker_actor"],
        goalkeeper_actor_path=assets["goalkeeper_actor"],
        gmt_model_path=assets["gmt_model"],
        gmt_skill_path=assets["gmt_skill"],
        dive_source_checkout=assets["dive_source"],
        dive_athlete_checkpoint_path=assets["dive_athlete_checkpoint"],
        dive_athlete_exam_path=assets["dive_athlete_exam"],
        recovery_athlete_checkpoint_path=assets["recovery_athlete_checkpoint"],
        recovery_athlete_exam_path=assets["recovery_athlete_exam"],
        predecessor_evidence_path=cast(Path, args.predecessor_evidence),
        output_dir=cast(Path, args.output_dir),
        source_checkout=cast(Path, args.source_checkout),
        config=config,
    )
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if report.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CurrentRuntimePrefixRequalificationConfig",
    "current_runtime_prefix_lane",
    "prefix_gate_contract",
    "run_current_runtime_prefix_requalification",
    "validate_current_runtime_prefix_requalification",
]
