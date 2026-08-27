"""Bilateral, perturbed population exam for the continuous four-G1 chain.

The population deliberately contains both a frozen S109 control and hard
counterexamples.  Every case runs from time zero in CPU MuJoCo and is strictly
replayed.  Failed phases are retained as a content-bound Growth curriculum;
they never receive policy-promotion authority merely because an anatomical
foot contact occurred.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import os
import subprocess
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
from rosclaw_soccer.world.field import G1TrainingGoalSpec

_CLAIM = "BILATERAL_PERTURBED_CONTINUOUS_TEAM_POPULATION"
_PHASE_GATES: tuple[tuple[str, tuple[str, ...], str], ...] = (
    (
        "first_save",
        ("qualified_first_airborne_save",),
        "goalkeeper",
    ),
    (
        "rearm_and_handoff",
        (
            "four_g1_two_ball_from_time_zero",
            "measured_ready_rearm_before_foot_contact",
            "stationary_second_ball_until_contact",
        ),
        "team_coordinator",
    ),
    (
        "second_striker_contact",
        (
            "anatomical_second_striker_contact",
            "learned_multi_role_contact_stack_active",
            "bounded_forward_high_launch",
        ),
        "second_striker",
    ),
    (
        "second_save",
        (
            "new_causal_goalkeeper_flight_epoch",
            "collision_faithful_high_glove_contact",
            "outward_physical_save",
        ),
        "goalkeeper",
    ),
    (
        "successor_state",
        (
            "second_striker_remains_stable",
            "whole_world_safety",
            "continuous_clock",
            "final_goalkeeper_ready",
        ),
        "whole_team",
    ),
)


@dataclass(frozen=True)
class BilateralContinuousTeamCase:
    """One immutable physics cell in the bilateral population."""

    case_id: str
    lane_id: str
    striker: G1PhysicalSecondStrikerConfig
    ball_mass_kg: float = 0.41
    ball_ground_friction: float = 0.10
    schema_version: str = "rosclaw_soccer.bilateral_continuous_team_case.v1"

    def __post_init__(self) -> None:
        if (
            not self.case_id
            or not self.case_id.replace("-", "").isalnum()
            or self.lane_id not in {"left-inner", "left-outer", "right-inner", "right-outer"}
            or not 0.40 <= self.ball_mass_kg <= 0.46
            or not 0.03 <= self.ball_ground_friction <= 0.80
        ):
            raise ValueError("bilateral continuous-team case is invalid")


def _default_cases() -> tuple[BilateralContinuousTeamCase, ...]:
    right = G1PhysicalSecondStrikerConfig()
    left = replace(
        right,
        kick_foot="left",
        ball_origin_m=(1.785, 0.380, 0.115),
        policy_target_m=(7.50, -0.70, 1.20),
        ballistic_actor_proximity_m=0.30,
        foot_yaw_offset=0.12,
    )
    return (
        BilateralContinuousTeamCase("right-control", "left-inner", right),
        BilateralContinuousTeamCase(
            "right-heavy-grip",
            "left-inner",
            right,
            ball_mass_kg=0.46,
            ball_ground_friction=0.16,
        ),
        BilateralContinuousTeamCase(
            "left-foot-frontier",
            "left-inner",
            left,
            ball_mass_kg=0.43,
            ball_ground_friction=0.08,
        ),
        BilateralContinuousTeamCase(
            "right-mirrored-lane",
            "right-inner",
            right,
            ball_mass_kg=0.40,
            ball_ground_friction=0.05,
        ),
    )


@dataclass(frozen=True)
class BilateralContinuousTeamPopulationConfig:
    """Fail-closed population and its shared continuous-world exam."""

    cases: tuple[BilateralContinuousTeamCase, ...] = _default_cases()
    exam: ContinuousSecondStrikerSaveExamConfig = ContinuousSecondStrikerSaveExamConfig()
    minimum_case_count: int = 4
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    commercial_use_allowed: bool = False
    schema_version: str = "rosclaw_soccer.bilateral_continuous_team_population_config.v1"

    def __post_init__(self) -> None:
        identifiers = tuple(case.case_id for case in self.cases)
        feet = {case.striker.kick_foot for case in self.cases}
        lanes = {case.lane_id for case in self.cases}
        masses = {case.ball_mass_kg for case in self.cases}
        frictions = {case.ball_ground_friction for case in self.cases}
        if (
            len(self.cases) < self.minimum_case_count
            or len(set(identifiers)) != len(identifiers)
            or feet != {"left", "right"}
            or len(lanes) < 2
            or len(masses) < 2
            or len(frictions) < 2
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
            or self.commercial_use_allowed
        ):
            raise ValueError("bilateral continuous-team population is incomplete")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


def attribute_population_failure(evaluation: dict[str, Any]) -> dict[str, Any]:
    """Assign causal learning ownership without collapsing to match outcome."""

    gates = evaluation.get("gates")
    if not isinstance(gates, dict):
        return {
            "passed": False,
            "first_failed_phase": "telemetry_contract",
            "learning_owner": "simforge",
            "failed_gates": ["evaluation_gates_missing"],
            "phases": [],
        }
    phases: list[dict[str, Any]] = []
    first_failed_phase: str | None = None
    first_owner: str | None = None
    failed_gates: list[str] = []
    for phase, names, owner in _PHASE_GATES:
        missing = [name for name in names if gates.get(name) is not True]
        phase_passed = not missing
        phases.append(
            {
                "phase": phase,
                "passed": phase_passed,
                "learning_owner": owner,
                "failed_gates": missing,
            }
        )
        failed_gates.extend(missing)
        if missing and first_failed_phase is None:
            first_failed_phase = phase
            first_owner = owner
    return {
        "passed": not failed_gates and evaluation.get("passed") is True,
        "first_failed_phase": first_failed_phase,
        "learning_owner": first_owner,
        "failed_gates": failed_gates,
        "phases": phases,
    }


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


def _implementation_hash() -> str:
    root = Path(__file__).resolve().parents[1]
    growth_core_path = Path(inspect.getfile(growth_core_contracts)).resolve()
    return str(
        hash_json(
            {
                str(path.relative_to(root)): hash_bytes(path.read_bytes())
                for path in (
                    Path(__file__),
                    root / "training/continuous_second_striker_save_exam.py",
                    root / "skills/team/shared_world.py",
                    root / "growth/ballistic_contact_impulse_actor.py",
                    root / "growth/ballistic_contact_residual.py",
                    root / "growth/ballistic_contact_torque_residual.py",
                )
            }
            | {"external/rosclaw/feedback/contracts.py": hash_bytes(growth_core_path.read_bytes())}
        )
    )


def _runtime_dependency_contract() -> dict[str, str]:
    """Bind the external numerical stack that can change contact outcomes."""

    return {
        package: importlib.metadata.version(package)
        for package in ("mujoco", "numpy", "onnxruntime")
    }


def _perturbed_goal(
    goal: G1TrainingGoalSpec,
    case: BilateralContinuousTeamCase,
) -> G1TrainingGoalSpec:
    return replace(
        goal,
        ball_mass_kg=case.ball_mass_kg,
    )


def run_bilateral_continuous_team_population(
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
    config: BilateralContinuousTeamPopulationConfig | None = None,
) -> dict[str, Any]:
    """Run each physical cell twice and freeze failures as Growth evidence."""

    active = config or BilateralContinuousTeamPopulationConfig()
    checkout = source_checkout.expanduser().resolve()
    destination = output_dir.expanduser().resolve()
    if destination.exists() or destination == checkout or checkout in destination.parents:
        raise ValueError("bilateral population evidence must use a new external directory")
    assets = {
        "striker_actor": striker_actor_path.expanduser().resolve(),
        "goalkeeper_actor": goalkeeper_actor_path.expanduser().resolve(),
        "gmt_model": gmt_model_path.expanduser().resolve(),
        "gmt_skill": gmt_skill_path.expanduser().resolve(),
        "dive_source": dive_source_checkout.expanduser().resolve(),
        "dive_athlete_checkpoint": dive_athlete_checkpoint_path.expanduser().resolve(),
        "dive_athlete_exam": dive_athlete_exam_path.expanduser().resolve(),
    }
    recovery_checkpoint = recovery_athlete_checkpoint_path.expanduser().resolve()
    recovery_exam = recovery_athlete_exam_path.expanduser().resolve()
    files = tuple(value for key, value in assets.items() if key != "dive_source") + (
        recovery_checkpoint,
        recovery_exam,
    )
    if not all(path.is_file() for path in files) or not (assets["dive_source"] / ".git").exists():
        raise FileNotFoundError("bilateral population input artifact is missing")
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    destination.mkdir(parents=True)
    request: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.bilateral_continuous_team_population_request.v1",
        "config": asdict(active),
        "config_hash": active.config_hash,
        "source_commit": _git_head(checkout),
        "body_hash": qualification.body_hash,
        "kick_prior_hash": qualification.kick_prior_hash,
        "artifacts": {
            key: (_git_head(value) if key == "dive_source" else hash_bytes(value.read_bytes()))
            for key, value in assets.items()
        },
        "recovery_checkpoint_hash": hash_bytes(recovery_checkpoint.read_bytes()),
        "recovery_exam_hash": hash_bytes(recovery_exam.read_bytes()),
        "runtime_dependencies": _runtime_dependency_contract(),
        "growth_core_contract_hash": hash_bytes(
            Path(inspect.getfile(growth_core_contracts)).resolve().read_bytes()
        ),
        "physics_backend": "mujoco_cpu",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "reset_or_teleport_used": False,
        "ball_cannon_used": False,
    }
    request["request_hash"] = hash_json(request)
    _atomic_json(destination / "request.json", request)
    lanes = {lane.lane_id: lane for lane in expanded_dynamic_corner_lanes()}
    case_reports: dict[str, Any] = {}
    for case in active.cases:
        lane = lanes[case.lane_id]
        exam = replace(active.exam, lane_ids=(case.lane_id,), striker=case.striker)
        kwargs, goalkeeper, base_goal = physical_second_striker_kwargs(
            lane=lane,
            assets=assets,
            recovery_checkpoint=recovery_checkpoint,
            recovery_exam=recovery_exam,
            config=exam,
        )
        goal = _perturbed_goal(base_goal, case)
        kwargs["goal_spec"] = goal
        kwargs["ball_ground_friction"] = case.ball_ground_friction
        result, trajectory = simulate_shared_world(asset_root, **kwargs)
        replay_result, replay_trajectory = simulate_shared_world(asset_root, **kwargs)
        evaluation = evaluate_continuous_second_striker_save(
            result=result,
            trajectory=trajectory,
            lane=lane,
            goal=goal,
            goalkeeper=goalkeeper,
            config=exam,
        )
        replay_evaluation = evaluate_continuous_second_striker_save(
            result=replay_result,
            trajectory=replay_trajectory,
            lane=lane,
            goal=goal,
            goalkeeper=goalkeeper,
            config=exam,
        )
        digest = trajectory_digest(trajectory)
        replay_digest = trajectory_digest(replay_trajectory)
        strict_replay = bool(
            result.to_dict() == replay_result.to_dict()
            and evaluation == replay_evaluation
            and digest == replay_digest
        )
        trajectory_file = f"{case.case_id}-trajectory.npz"
        trajectory_path = destination / trajectory_file
        _atomic_trajectory(trajectory_path, trajectory)
        attribution = attribute_population_failure(evaluation)
        case_reports[case.case_id] = {
            "case": asdict(case),
            "passed": bool(evaluation.get("passed") is True and strict_replay),
            "strict_replay": strict_replay,
            "evaluation": evaluation,
            "failure_attribution": attribution,
            "trajectory_file": trajectory_file,
            "trajectory_hash": hash_bytes(trajectory_path.read_bytes()),
            "trajectory_digest": digest,
            "replay_trajectory_digest": replay_digest,
        }
    passed_cases = [name for name, case in case_reports.items() if case["passed"]]
    rejected_cases = [name for name, case in case_reports.items() if not case["passed"]]
    observed_feet = {
        case["evaluation"].get("result", {}).get("second_striker_contact_foot")
        for case in case_reports.values()
    } - {None}
    population_gates = {
        "strict_replay_all_cases": all(case["strict_replay"] for case in case_reports.values()),
        "right_control_retained": "right-control" in passed_cases,
        "both_anatomical_feet_observed": observed_feet == {"left", "right"},
        "multi_lane_coverage": len({case.lane_id for case in active.cases}) >= 2,
        "mass_and_friction_coverage": bool(
            len({case.ball_mass_kg for case in active.cases}) >= 2
            and len({case.ball_ground_friction for case in active.cases}) >= 2
        ),
        "all_cases_qualified": not rejected_cases,
    }
    passed = all(population_gates.values())
    evidence: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.bilateral_continuous_team_population_evidence.v1",
        "claim": _CLAIM,
        "passed": passed,
        "promotion_status": (
            "PROMOTED_SIM_ONLY_BILATERAL_POPULATION" if passed else "REJECTED_BILATERAL_POPULATION"
        ),
        "growth_status": (
            "POPULATION_QUALIFIED" if passed else "COUNTEREXAMPLES_RETAINED_FOR_GROWTH"
        ),
        "population_gates": population_gates,
        "passed_cases": passed_cases,
        "rejected_cases": rejected_cases,
        "observed_contact_feet": sorted(cast(set[str], observed_feet)),
        "cases": case_reports,
        "request_hash": request["request_hash"],
        "source_commit": request["source_commit"],
        "implementation_hash": _implementation_hash(),
        "physics_backend": "mujoco_cpu",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "commercial_use_allowed": False,
        "reset_or_teleport_used": False,
        "ball_cannon_used": False,
        "pixels_used_for_scoring": False,
    }
    evidence["report_hash"] = hash_json(evidence)
    path = destination / "evidence.json"
    _atomic_json(path, evidence)
    return validate_bilateral_continuous_team_population(path)


def validate_bilateral_continuous_team_population(path: Path) -> dict[str, Any]:
    """Validate immutable trajectories and fail-closed authority semantics."""

    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("bilateral population evidence must be a JSON object")
    expected = payload.pop("report_hash", None)
    try:
        cases = payload.get("cases")
        if not isinstance(cases, dict) or len(cases) < 4:
            raise ValueError("bilateral population case set is incomplete")
        for case in cases.values():
            if not isinstance(case, dict) or case.get("strict_replay") is not True:
                raise ValueError("bilateral population strict replay is absent")
            trajectory = resolved.parent / str(case.get("trajectory_file", ""))
            if not trajectory.is_file() or case.get("trajectory_hash") != hash_bytes(
                trajectory.read_bytes()
            ):
                raise ValueError("bilateral population trajectory binding changed")
        expected_passed = bool(
            all(payload.get("population_gates", {}).values()) and not payload.get("rejected_cases")
        )
        expected_status = (
            "PROMOTED_SIM_ONLY_BILATERAL_POPULATION"
            if expected_passed
            else "REJECTED_BILATERAL_POPULATION"
        )
        if (
            payload.get("schema_version")
            != "rosclaw_soccer.bilateral_continuous_team_population_evidence.v1"
            or payload.get("claim") != _CLAIM
            or payload.get("passed") is not expected_passed
            or payload.get("promotion_status") != expected_status
            or payload.get("physics_backend") != "mujoco_cpu"
            or payload.get("activation_ceiling") != "SIM_ONLY"
            or payload.get("hardware_command_sent") is not False
            or payload.get("commercial_use_allowed") is not False
            or payload.get("reset_or_teleport_used") is not False
            or payload.get("ball_cannon_used") is not False
            or payload.get("pixels_used_for_scoring") is not False
            or expected != hash_json(payload)
        ):
            raise ValueError("bilateral population authority or integrity contract is invalid")
    finally:
        if expected is not None:
            payload["report_hash"] = expected
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
    parser.add_argument("--source-checkout", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = run_bilateral_continuous_team_population(
        asset_root=cast(Path, args.asset_root),
        striker_actor_path=cast(Path, args.striker_actor),
        goalkeeper_actor_path=cast(Path, args.goalkeeper_actor),
        gmt_model_path=cast(Path, args.gmt_model),
        gmt_skill_path=cast(Path, args.gmt_skill),
        dive_source_checkout=cast(Path, args.dive_source),
        dive_athlete_checkpoint_path=cast(Path, args.dive_athlete_checkpoint),
        dive_athlete_exam_path=cast(Path, args.dive_athlete_exam),
        recovery_athlete_checkpoint_path=cast(Path, args.recovery_athlete_checkpoint),
        recovery_athlete_exam_path=cast(Path, args.recovery_athlete_exam),
        output_dir=cast(Path, args.output_dir),
        source_checkout=cast(Path, args.source_checkout),
    )
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if report.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BilateralContinuousTeamCase",
    "BilateralContinuousTeamPopulationConfig",
    "attribute_population_failure",
    "run_bilateral_continuous_team_population",
    "validate_bilateral_continuous_team_population",
]
