"""Paired CPU-MuJoCo promotion exam for the S105 recovery athlete.

The neural candidate is evaluated only after a real airborne glove save.  It
must preserve the complete pass/shot/save prefix, recover more smoothly than
the qualified S104 controller, accept a new lateral command, and return to a
continuous measured ready state.  The actor remains a bounded, high-level
locomotion-command proposal and this exam grants no ROS or hardware authority.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.providers.g1.asset_qualification import (
    qualify_g1_assets,
    trajectory_digest,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.shared_world import (
    G1GoalkeeperConfig,
    G1SharedWorldResult,
    simulate_shared_world,
)
from rosclaw_soccer.training.dynamic_corner_save import (
    DynamicCornerSaveLane,
    dynamic_corner_lane_kwargs,
    expanded_dynamic_corner_lanes,
)
from rosclaw_soccer.training.dynamic_takeoff_exam import evaluate_dynamic_takeoff_save
from rosclaw_soccer.training.recovery_athlete_cpu_exam import (
    validate_recovery_athlete_cpu_exam,
)
from rosclaw_soccer.training.save_to_ready_successor import (
    SaveToReadySuccessorConfig,
    _ready_window,
    evaluate_save_to_ready_successor,
)
from rosclaw_soccer.world.field import G1TrainingGoalSpec

_AUTHORITY_BY_LANE = {
    "left-inner": 0.05,
    "left-outer": 0.75,
    "right-inner": 0.50,
    "right-outer": 0.75,
}
_PREFIX_KEYS = (
    "time",
    "ball_pose",
    "ball_velocity",
    "goalkeeper_pelvis_pose",
    "goalkeeper_root_velocity",
    "goalkeeper_joint_position",
    "goalkeeper_foot_contact",
)


@dataclass(frozen=True)
class RecoveryAthleteIntegrationConfig:
    """Fail-closed paired-physics gates for recovery-actor promotion."""

    successor: SaveToReadySuccessorConfig = SaveToReadySuccessorConfig()
    candidate_blend: float = 1.0
    maximum_portfolio_command_variation_ratio: float = 0.90
    minimum_improved_lane_count: int = 2
    maximum_ready_latency_regression_sec: float = 0.50
    minimum_actor_active_fraction: float = 0.30
    readiness_scan_stride_frames: int = 5
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    commercial_use_allowed: bool = False
    schema_version: str = "rosclaw_soccer.recovery_athlete_integration_config.v1"

    def __post_init__(self) -> None:
        values = (
            self.candidate_blend,
            self.maximum_portfolio_command_variation_ratio,
            self.maximum_ready_latency_regression_sec,
            self.minimum_actor_active_fraction,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("recovery athlete integration settings must be finite")
        if not math.isclose(self.candidate_blend, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("recovery athlete integration requires full candidate authority")
        if not 0.50 <= self.maximum_portfolio_command_variation_ratio < 1.0:
            raise ValueError("recovery athlete variation ratio is invalid")
        if not 1 <= self.minimum_improved_lane_count <= 4:
            raise ValueError("recovery athlete improved lane count is invalid")
        if not 0.0 <= self.maximum_ready_latency_regression_sec <= 1.0:
            raise ValueError("recovery athlete ready latency allowance is invalid")
        if not 0.10 <= self.minimum_actor_active_fraction <= 0.80:
            raise ValueError("recovery athlete activation floor is invalid")
        if not 1 <= self.readiness_scan_stride_frames <= 10:
            raise ValueError("recovery athlete readiness scan stride is invalid")
        if (
            self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
            or self.commercial_use_allowed
        ):
            raise ValueError("recovery athlete integration must remain non-commercial SIM_ONLY")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


def _git_head(checkout: Path) -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_trajectory(path: Path, trajectory: dict[str, NDArray[Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **trajectory)  # type: ignore[arg-type]
    os.replace(temporary, path)


def _prefix_digest(trajectory: dict[str, NDArray[Any]], *, contact_time_sec: float) -> str:
    time = np.asarray(trajectory["time"], dtype=np.float64)
    mask = time <= contact_time_sec + 1.0e-9
    payload: dict[str, Any] = {}
    for key in _PREFIX_KEYS:
        values = np.ascontiguousarray(np.asarray(trajectory[key])[mask])
        payload[key] = {
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "hash": hash_bytes(values.tobytes()),
        }
    return str(hash_json(payload))


def _recovery_command_metrics(
    trajectory: dict[str, NDArray[Any]],
    *,
    contact_time_sec: float,
    config: SaveToReadySuccessorConfig,
) -> dict[str, float | int]:
    time = np.asarray(trajectory["time"], dtype=np.float64)
    command = np.asarray(trajectory["goalkeeper_command_mps"], dtype=np.float64)
    probe_start = contact_time_sec + config.probe_delay_sec
    probe_stop = probe_start + config.probe_duration_sec
    segment_masks = (
        # Exclude the bounded mode-switch transient; this metric isolates
        # command chatter while the recovery route is already active.
        (time >= contact_time_sec + config.recovery_delay_sec + 0.25) & (time < probe_start - 0.02),
        (time >= probe_stop + 0.10) & (time <= time[-1]),
    )
    differences = [np.diff(command[mask]) for mask in segment_masks if np.count_nonzero(mask) > 1]
    merged = np.concatenate(differences) if differences else np.zeros(1, dtype=np.float64)
    active = np.asarray(
        trajectory.get("goalkeeper_recovery_athlete_active", np.zeros(time.shape)),
        dtype=np.bool_,
    )
    learned = np.asarray(
        trajectory.get(
            "goalkeeper_recovery_athlete_world_command",
            np.zeros((time.size, 3), dtype=np.float64),
        ),
        dtype=np.float64,
    )
    learned_delta = (
        np.diff(learned[active], axis=0) if np.count_nonzero(active) > 1 else np.zeros((1, 3))
    )
    return {
        "lateral_command_total_variation_mps": float(np.sum(np.abs(merged))),
        "lateral_command_peak_step_mps": float(np.max(np.abs(merged))),
        "lateral_command_rms_step_mps": float(np.sqrt(np.mean(np.square(merged)))),
        "recovery_actor_active_frame_count": int(np.count_nonzero(active)),
        "recovery_actor_active_fraction": float(np.mean(active)),
        "learned_world_command_total_variation": float(np.sum(np.abs(learned_delta))),
        "learned_world_command_peak_step": float(np.max(np.abs(learned_delta))),
    }


def _earliest_ready_latency(
    *,
    result: G1SharedWorldResult,
    trajectory: dict[str, NDArray[Any]],
    goal: G1TrainingGoalSpec,
    goalkeeper_config: G1GoalkeeperConfig,
    config: RecoveryAthleteIntegrationConfig,
) -> float | None:
    contact = result.goalkeeper_ball_contact_time_sec
    if contact is None:
        return None
    time = np.asarray(trajectory["time"], dtype=np.float64)
    start = contact + config.successor.recovery_delay_sec
    stop = contact + config.successor.probe_delay_sec
    candidates = np.flatnonzero((time >= start) & (time + config.successor.ready_hold_sec <= stop))
    for index in candidates[:: config.readiness_scan_stride_frames]:
        window_start = float(time[index])
        mask = (time >= window_start) & (
            time < window_start + config.successor.ready_hold_sec - 1.0e-9
        )
        if np.count_nonzero(mask) < 30:
            continue
        ready = _ready_window(
            trajectory=trajectory,
            mask=mask,
            goal=goal,
            depth_from_goal_line_m=goalkeeper_config.depth_from_goal_line_m,
            config=config.successor,
        )
        if ready["passed"]:
            return float(window_start - contact)
    return None


def _lane_kwargs(
    *,
    lane: DynamicCornerSaveLane,
    asset_paths: dict[str, Path],
    recovery_checkpoint_path: Path | None,
    recovery_exam_path: Path | None,
    config: RecoveryAthleteIntegrationConfig,
) -> tuple[dict[str, Any], G1GoalkeeperConfig, G1TrainingGoalSpec, float]:
    expected_probe = config.successor.probe_speed_mps * (
        1.0 if lane.lane_id.startswith("left") else -1.0
    )
    kwargs = dynamic_corner_lane_kwargs(
        lane=lane,
        striker_actor_path=asset_paths["striker_actor"],
        goalkeeper_actor_path=asset_paths["goalkeeper_actor"],
        gmt_model_path=asset_paths["gmt_model"],
        gmt_skill_path=asset_paths["gmt_skill"],
        dive_source_checkout=asset_paths["dive_source"],
        dive_athlete_checkpoint_path=asset_paths["dive_athlete_checkpoint"],
        dive_athlete_exam_path=asset_paths["dive_athlete_exam"],
        dive_athlete_blend=_AUTHORITY_BY_LANE[lane.lane_id],
    )
    goalkeeper = cast(G1GoalkeeperConfig, kwargs["goalkeeper_config"])
    goal = cast(G1TrainingGoalSpec, kwargs["goal_spec"])
    kwargs["simulation_duration_sec"] = config.successor.simulation_duration_sec
    kwargs["goalkeeper_config"] = replace(
        goalkeeper,
        maximum_depth_correction_mps=config.successor.recovery_depth_speed_mps,
        post_contact_ready_recovery_enabled=True,
        post_contact_ready_recovery_delay_sec=config.successor.recovery_delay_sec,
        post_contact_ready_lateral_deadband_m=config.successor.recovery_lateral_deadband_m,
        successor_lateral_probe_enabled=True,
        successor_lateral_probe_delay_sec=config.successor.probe_delay_sec,
        successor_lateral_probe_duration_sec=config.successor.probe_duration_sec,
        successor_lateral_probe_command_mps=expected_probe,
        recovery_athlete_checkpoint_path=recovery_checkpoint_path,
        recovery_athlete_exam_path=recovery_exam_path,
        recovery_athlete_blend=(
            0.0 if recovery_checkpoint_path is None else config.candidate_blend
        ),
    )
    return kwargs, goalkeeper, goal, expected_probe


def _evaluate_rollout(
    *,
    result: G1SharedWorldResult,
    trajectory: dict[str, NDArray[Any]],
    lane: DynamicCornerSaveLane,
    goal: G1TrainingGoalSpec,
    goalkeeper: G1GoalkeeperConfig,
    expected_probe: float,
    config: RecoveryAthleteIntegrationConfig,
) -> dict[str, Any]:
    contact = result.goalkeeper_ball_contact_time_sec
    if contact is None:
        return {"passed": False, "reason": "goalkeeper contact is absent"}
    takeoff = evaluate_dynamic_takeoff_save(
        result=result,
        trajectory=trajectory,
        config=lane.takeoff_config,
    )
    successor = evaluate_save_to_ready_successor(
        result=result,
        trajectory=trajectory,
        goal=goal,
        depth_from_goal_line_m=goalkeeper.depth_from_goal_line_m,
        expected_probe_command_mps=expected_probe,
        config=config.successor,
    )
    return {
        "passed": bool(takeoff.get("passed") and successor.get("passed")),
        "takeoff": takeoff,
        "successor": successor,
        "ready_latency_sec": _earliest_ready_latency(
            result=result,
            trajectory=trajectory,
            goal=goal,
            goalkeeper_config=goalkeeper,
            config=config,
        ),
        "command_metrics": _recovery_command_metrics(
            trajectory,
            contact_time_sec=contact,
            config=config.successor,
        ),
        "prefix_digest": _prefix_digest(trajectory, contact_time_sec=contact),
        "trajectory_digest": trajectory_digest(trajectory),
        "result": result.to_dict(),
    }


def run_recovery_athlete_integration_exam(
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
    source_checkout: Path,
    config: RecoveryAthleteIntegrationConfig | None = None,
) -> dict[str, Any]:
    """Run paired parent/candidate physics and a strict candidate replay."""

    active = config or RecoveryAthleteIntegrationConfig()
    paths = {
        "striker_actor": striker_actor_path.expanduser().resolve(),
        "goalkeeper_actor": goalkeeper_actor_path.expanduser().resolve(),
        "gmt_model": gmt_model_path.expanduser().resolve(),
        "gmt_skill": gmt_skill_path.expanduser().resolve(),
        "dive_source": dive_source_checkout.expanduser().resolve(),
        "dive_athlete_checkpoint": dive_athlete_checkpoint_path.expanduser().resolve(),
        "dive_athlete_exam": dive_athlete_exam_path.expanduser().resolve(),
    }
    recovery_checkpoint = recovery_athlete_checkpoint_path.expanduser().resolve()
    recovery_exam_path = recovery_athlete_exam_path.expanduser().resolve()
    files = tuple(value for key, value in paths.items() if key != "dive_source") + (
        recovery_checkpoint,
        recovery_exam_path,
    )
    if not all(path.is_file() for path in files):
        raise FileNotFoundError("recovery athlete integration input artifact is missing")
    if not (paths["dive_source"] / ".git").exists():
        raise ValueError("recovery athlete integration dive source is incomplete")
    cpu_exam = validate_recovery_athlete_cpu_exam(recovery_exam_path)
    checkpoint_hash = hash_bytes(recovery_checkpoint.read_bytes())
    if cpu_exam.get("checkpoint_hash") != checkpoint_hash:
        raise ValueError("recovery athlete integration checkpoint binding changed")
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    destination = output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    checkout = source_checkout.expanduser().resolve()
    request = {
        "schema_version": "rosclaw_soccer.recovery_athlete_integration_request.v1",
        "config": asdict(active),
        "config_hash": active.config_hash,
        "source_commit": _git_head(checkout),
        "body_hash": qualification.body_hash,
        "kick_prior_hash": qualification.kick_prior_hash,
        "recovery_checkpoint_hash": checkpoint_hash,
        "recovery_cpu_exam_hash": cpu_exam["report_hash"],
        "artifacts": {
            key: (_git_head(value) if key == "dive_source" else hash_bytes(value.read_bytes()))
            for key, value in paths.items()
        },
        "physics_backend": "mujoco_cpu",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    request["request_hash"] = hash_json(request)
    _atomic_json(destination / "request.json", request)

    cases: dict[str, Any] = {}
    parent_variation = 0.0
    candidate_variation = 0.0
    improved_lane_count = 0
    maximum_latency_regression = -math.inf
    for lane in expanded_dynamic_corner_lanes():
        parent_kwargs, goalkeeper, goal, expected_probe = _lane_kwargs(
            lane=lane,
            asset_paths=paths,
            recovery_checkpoint_path=None,
            recovery_exam_path=None,
            config=active,
        )
        candidate_kwargs, _, _, _ = _lane_kwargs(
            lane=lane,
            asset_paths=paths,
            recovery_checkpoint_path=recovery_checkpoint,
            recovery_exam_path=recovery_exam_path,
            config=active,
        )
        parent_result, parent_trajectory = simulate_shared_world(asset_root, **parent_kwargs)
        candidate_result, candidate_trajectory = simulate_shared_world(
            asset_root, **candidate_kwargs
        )
        replay_result, replay_trajectory = simulate_shared_world(asset_root, **candidate_kwargs)
        parent = _evaluate_rollout(
            result=parent_result,
            trajectory=parent_trajectory,
            lane=lane,
            goal=goal,
            goalkeeper=goalkeeper,
            expected_probe=expected_probe,
            config=active,
        )
        candidate = _evaluate_rollout(
            result=candidate_result,
            trajectory=candidate_trajectory,
            lane=lane,
            goal=goal,
            goalkeeper=goalkeeper,
            expected_probe=expected_probe,
            config=active,
        )
        replay = _evaluate_rollout(
            result=replay_result,
            trajectory=replay_trajectory,
            lane=lane,
            goal=goal,
            goalkeeper=goalkeeper,
            expected_probe=expected_probe,
            config=active,
        )
        strict_replay = bool(
            candidate_result.to_dict() == replay_result.to_dict()
            and trajectory_digest(candidate_trajectory) == trajectory_digest(replay_trajectory)
        )
        prefix_unchanged = bool(parent.get("prefix_digest") == candidate.get("prefix_digest"))
        parent_tv = float(
            cast(dict[str, Any], parent.get("command_metrics", {})).get(
                "lateral_command_total_variation_mps", math.inf
            )
        )
        candidate_tv = float(
            cast(dict[str, Any], candidate.get("command_metrics", {})).get(
                "lateral_command_total_variation_mps", math.inf
            )
        )
        parent_latency = parent.get("ready_latency_sec")
        candidate_latency = candidate.get("ready_latency_sec")
        latency_regression = (
            math.inf
            if not isinstance(parent_latency, int | float)
            or not isinstance(candidate_latency, int | float)
            else float(candidate_latency - parent_latency)
        )
        actor_fraction = float(candidate_result.goalkeeper_recovery_athlete_active_fraction)
        binding_passed = bool(
            candidate_result.goalkeeper_recovery_athlete_checkpoint_hash == checkpoint_hash
            and math.isclose(
                candidate_result.goalkeeper_recovery_athlete_blend,
                active.candidate_blend,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        )
        lane_gates = {
            "qualified_parent": parent.get("passed") is True,
            "candidate_full_chain": candidate.get("passed") is True,
            "candidate_strict_replay": strict_replay and replay.get("passed") is True,
            "pre_contact_prefix_unchanged": prefix_unchanged,
            "checkpoint_and_authority_bound": binding_passed,
            "actor_route_active": actor_fraction >= active.minimum_actor_active_fraction,
            "ready_latency_not_regressed": latency_regression
            <= active.maximum_ready_latency_regression_sec,
        }
        parent_variation += parent_tv
        candidate_variation += candidate_tv
        improved_lane_count += int(candidate_tv < parent_tv - 1.0e-9)
        maximum_latency_regression = max(maximum_latency_regression, latency_regression)
        parent_path = destination / f"{lane.lane_id}-parent-trajectory.npz"
        candidate_path = destination / f"{lane.lane_id}-candidate-trajectory.npz"
        _atomic_trajectory(parent_path, parent_trajectory)
        _atomic_trajectory(candidate_path, candidate_trajectory)
        cases[lane.lane_id] = {
            "passed": bool(all(lane_gates.values())),
            "gates": lane_gates,
            "parent": parent,
            "candidate": candidate,
            "strict_replay": strict_replay,
            "latency_regression_sec": latency_regression,
            "parent_trajectory_file": parent_path.name,
            "parent_trajectory_hash": hash_bytes(parent_path.read_bytes()),
            "candidate_trajectory_file": candidate_path.name,
            "candidate_trajectory_hash": hash_bytes(candidate_path.read_bytes()),
        }

    variation_ratio = candidate_variation / max(parent_variation, 1.0e-12)
    portfolio_gates = {
        "all_four_lanes_pass": len(cases) == 4 and all(row["passed"] for row in cases.values()),
        "portfolio_command_smoother": variation_ratio
        <= active.maximum_portfolio_command_variation_ratio,
        "multiple_lanes_improved": improved_lane_count >= active.minimum_improved_lane_count,
        "heldout_right_inner_passed": cases.get("right-inner", {}).get("passed") is True,
        "ready_latency_not_regressed": maximum_latency_regression
        <= active.maximum_ready_latency_regression_sec,
    }
    passed = bool(all(portfolio_gates.values()))
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.recovery_athlete_integration_exam.v1",
        "passed": passed,
        "promotion_status": (
            "PROMOTED_SIM_ONLY_RECOVERY_ATHLETE" if passed else "REJECTED_DEVELOPMENT"
        ),
        "claim": "PAIRED_PHYSICS_SAVE_TO_SMOOTHER_READY_NEURAL_RECOVERY",
        "request_hash": request["request_hash"],
        "source_commit": request["source_commit"],
        "checkpoint_hash": checkpoint_hash,
        "cpu_exam_hash": cpu_exam["report_hash"],
        "portfolio_gates": portfolio_gates,
        "portfolio_metrics": {
            "parent_lateral_command_total_variation_mps": parent_variation,
            "candidate_lateral_command_total_variation_mps": candidate_variation,
            "candidate_to_parent_variation_ratio": variation_ratio,
            "improved_lane_count": improved_lane_count,
            "maximum_ready_latency_regression_sec": maximum_latency_regression,
        },
        "cases": cases,
        "physics_backend": "mujoco_cpu",
        "pixels_used_for_scoring": False,
        "reset_or_teleport_used": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "commercial_use_allowed": False,
        "implementation_hash": _implementation_hash(),
    }
    report["report_hash"] = hash_json(report)
    _atomic_json(destination / "evidence.json", report)
    return report


def validate_recovery_athlete_integration_exam(path: Path) -> dict[str, Any]:
    """Validate integrity, physics gates and the current implementation."""

    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("recovery athlete integration evidence must be an object")
    claimed = payload.get("report_hash")
    unhashed = dict(payload)
    unhashed.pop("report_hash", None)
    gates = payload.get("portfolio_gates")
    cases = payload.get("cases")
    if not (
        claimed == hash_json(unhashed)
        and payload.get("schema_version") == "rosclaw_soccer.recovery_athlete_integration_exam.v1"
        and payload.get("passed") is True
        and payload.get("promotion_status") == "PROMOTED_SIM_ONLY_RECOVERY_ATHLETE"
        and payload.get("physics_backend") == "mujoco_cpu"
        and payload.get("pixels_used_for_scoring") is False
        and payload.get("reset_or_teleport_used") is False
        and payload.get("activation_ceiling") == "SIM_ONLY"
        and payload.get("hardware_command_sent") is False
        and payload.get("commercial_use_allowed") is False
        and payload.get("implementation_hash") == _implementation_hash()
        and isinstance(gates, dict)
        and all(gates.values())
        and isinstance(cases, dict)
        and set(cases) == set(_AUTHORITY_BY_LANE)
        and all(
            isinstance(case, dict)
            and case.get("passed") is True
            and case.get("strict_replay") is True
            for case in cases.values()
        )
    ):
        raise ValueError("recovery athlete integration evidence failed closed")
    return cast(dict[str, Any], payload)


def _implementation_hash() -> str:
    root = Path(__file__).parents[1]
    files = (
        Path(__file__),
        root / "skills" / "team" / "shared_world.py",
        Path(__file__).parent / "recovery_athlete_student.py",
        Path(__file__).parent / "recovery_athlete_cpu_exam.py",
        Path(__file__).parent / "save_to_ready_successor.py",
    )
    return str(hash_json({path.name: hash_bytes(path.read_bytes()) for path in files}))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--striker-actor", type=Path, required=True)
    parser.add_argument("--goalkeeper-actor", type=Path, required=True)
    parser.add_argument("--gmt-model", type=Path, required=True)
    parser.add_argument("--gmt-skill", type=Path, required=True)
    parser.add_argument("--dive-source", type=Path, required=True)
    parser.add_argument("--dive-checkpoint", type=Path, required=True)
    parser.add_argument("--dive-exam", type=Path, required=True)
    parser.add_argument("--recovery-checkpoint", type=Path, required=True)
    parser.add_argument("--recovery-exam", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-checkout", type=Path, default=Path.cwd())
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = run_recovery_athlete_integration_exam(
        asset_root=args.asset_root,
        striker_actor_path=args.striker_actor,
        goalkeeper_actor_path=args.goalkeeper_actor,
        gmt_model_path=args.gmt_model,
        gmt_skill_path=args.gmt_skill,
        dive_source_checkout=args.dive_source,
        dive_athlete_checkpoint_path=args.dive_checkpoint,
        dive_athlete_exam_path=args.dive_exam,
        recovery_athlete_checkpoint_path=args.recovery_checkpoint,
        recovery_athlete_exam_path=args.recovery_exam,
        output_dir=args.output_dir,
        source_checkout=args.source_checkout,
    )
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RecoveryAthleteIntegrationConfig",
    "run_recovery_athlete_integration_exam",
    "validate_recovery_athlete_integration_exam",
]
