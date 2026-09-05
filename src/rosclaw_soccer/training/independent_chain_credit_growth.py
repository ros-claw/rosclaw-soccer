"""S201: turn a physical pass miss into role-local continual-learning credit."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, cast

import numpy as np

from rosclaw_soccer.growth.continuous_option_chain import (
    ChainRoleBinding,
    ContinuousChainEvent,
    ContinuousChainFailure,
    ContinuousChainPhase,
    ContinuousOptionChainRequest,
    assess_continuous_option_chain,
    build_chain_repair_lease,
)
from rosclaw_soccer.growth.pass_aim_calibration import (
    PassAimCalibrationSample,
    train_pass_aim_residual_actor,
)
from rosclaw_soccer.growth.role_self_model import MatchRole
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.independent_option_growth import (
    validate_independent_option_growth,
)
from rosclaw_soccer.training.independent_team_growth import (
    build_independent_three_vs_three_fixture,
)


def run_independent_chain_credit_growth(
    *,
    evidence_dir: Path,
    source_report_path: Path,
    asset_root: Path,
) -> dict[str, Any]:
    """Attribute the S200 miss and fit a non-deployable residual seed actor."""

    output = evidence_dir.expanduser().resolve()
    source_path = source_report_path.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("S201 evidence directory must be empty")
    source = validate_independent_option_growth(source_path)
    rows = source.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("S201 source report has no physical pass row")
    pass_row = cast(dict[str, Any], rows[0])
    result = cast(dict[str, Any], pass_row.get("result"))
    request_value = cast(dict[str, Any], result.get("request"))
    outcome = cast(dict[str, Any], result.get("outcome"))
    artifact = cast(dict[str, Any], pass_row.get("primary_artifact"))
    case_path = source_path.parent / "case-000" / str(artifact.get("file"))
    if (
        request_value.get("option") != "pass"
        or not case_path.is_file()
        or hash_bytes(case_path.read_bytes()) != artifact.get("file_hash")
        or pass_row.get("exact_replay") is not True
        or outcome.get("safe") is not True
        or outcome.get("contact_observed") is not True
    ):
        raise ValueError("S201 requires the strict, safe S200 physical pass source")

    with np.load(case_path, allow_pickle=False) as archive:
        ball_pose = np.asarray(archive["ball_pose"], dtype=np.float64)
        ball_velocity = np.asarray(archive["ball_velocity"], dtype=np.float64)
        time = np.asarray(archive["time"], dtype=np.float64)
    if (
        ball_pose.ndim != 2
        or ball_pose.shape[1] != 7
        or ball_velocity.shape != (len(ball_pose), 6)
        or time.shape != (len(ball_pose),)
        or not np.all(np.isfinite(ball_pose))
        or not np.all(np.isfinite(ball_velocity))
        or not np.all(np.isfinite(time))
    ):
        raise ValueError("S201 source trajectory contract changed")
    aim = np.asarray(request_value.get("target_position_m"), dtype=np.float64)
    nearest_index = int(np.argmin(np.linalg.norm(ball_pose[:, :3] - aim, axis=1)))
    observed = ball_pose[nearest_index, :3]

    sample = PassAimCalibrationSample(
        outcome_hash=str(pass_row["result_hash"]),
        physical_policy_hash=str(request_value["option_policy_hash"]),
        context_hash=str(pass_row["scenario_hash"]),
        trajectory_digest=str(artifact["trajectory_digest"]),
        requested_aim_m=(float(aim[0]), float(aim[1]), float(aim[2])),
        observed_delivery_m=(
            float(observed[0]),
            float(observed[1]),
            float(observed[2]),
        ),
        contact_observed=True,
        safe=True,
        exact_replay=True,
    )
    actor = train_pass_aim_residual_actor((sample,))
    fixture = build_independent_three_vs_three_fixture(asset_root)
    cell_by_id = {cell.agent_id: cell for cell in fixture.cells}
    selected = (
        ("red.playmaker", MatchRole.PLAYMAKER),
        ("red.finisher", MatchRole.FINISHER),
        ("blue.goalkeeper", MatchRole.GOALKEEPER),
    )
    ball_initial_hash = _ball_state_hash(ball_pose[0], ball_velocity[0])
    chain_request = ContinuousOptionChainRequest(
        chain_id="s201.chain.red-attack",
        roster_hash=fixture.roster.roster_hash,
        ball_id="match.ball.0001",
        initial_ball_state_hash=ball_initial_hash,
        role_bindings=tuple(
            ChainRoleBinding(
                agent_id=agent_id,
                role=role,
                cell_hash=cell_by_id[agent_id].cell_hash,
                champion_policy_hash=cell_by_id[agent_id].growth_scope.champion_policy.version_hash,
                parent_policy_hash=cell_by_id[agent_id].growth_scope.parent_policy.version_hash,
            )
            for agent_id, role in selected
        ),
        goal_target_m=(
            fixture.goal.plane_x_m,
            fixture.goal.target_y_m,
            fixture.goal.target_z_m,
        ),
    )
    event = ContinuousChainEvent(
        phase=ContinuousChainPhase.PASS,
        agent_id="red.playmaker",
        ball_id=chain_request.ball_id,
        input_ball_state_hash=ball_initial_hash,
        output_ball_state_hash=_ball_state_hash(
            ball_pose[nearest_index], ball_velocity[nearest_index]
        ),
        decision_hash=str(request_value["decision_hash"]),
        option_request_hash=str(outcome["request_hash"]),
        started_at_sec=0.0,
        ended_at_sec=float(time[nearest_index]),
        contact_observed=True,
        safe=True,
        phase_ready=True,
        target_error_m=float(outcome["target_delivery_distance_m"]),
        post_contact_ball_speed_mps=float(outcome["post_contact_peak_ball_speed_mps"]),
        exact_replay=True,
        root_pose_write_after_start=bool(outcome["root_pose_write_after_start"]),
        ball_state_write_after_start=bool(outcome["ball_state_write_after_start"]),
        pixels_used_for_scoring=bool(outcome["pixels_used_for_scoring"]),
        hardware_command_sent=bool(outcome["hardware_command_sent"]),
    )
    assessment = assess_continuous_option_chain(chain_request, (event,))
    if assessment.earliest_failure is not ContinuousChainFailure.PASS_INACCURATE:
        raise RuntimeError("S201 source no longer exposes the expected first causal failure")
    lease = build_chain_repair_lease(
        cells=fixture.cells,
        assessment=assessment,
        dataset_manifest_hash=actor.dataset_manifest_hash,
        scenario_contract_hash=chain_request.request_hash,
        maximum_optimizer_steps=1_000,
    )
    reconstruction = actor.propose_aim(sample.observed_delivery_m)
    reconstruction_error = float(np.linalg.norm(np.asarray(reconstruction) - aim))
    output.mkdir(parents=True, exist_ok=True)
    actor_path = output / "playmaker-pass-aim-residual-seed.json"
    _atomic_json(actor_path, actor.to_dict())
    gates = {
        "strict_physical_source": pass_row.get("exact_replay") is True,
        "same_ball_source_bound": artifact.get("trajectory_digest")
        == pass_row.get("replay_artifact", {}).get("trajectory_digest"),
        "earliest_failure_attributed": assessment.earliest_failure
        is ContinuousChainFailure.PASS_INACCURATE,
        "only_playmaker_plastic": lease.focal_agent_id == "red.playmaker"
        and sum(binding.mode.value == "PLASTIC" for binding in lease.bindings) == 1,
        "all_six_cells_in_lease": len(lease.bindings) == 6,
        "seed_actor_reconstructs_training_sample": reconstruction_error <= 1.0e-12,
        "seed_actor_not_deployable": not actor.deployment_ready,
        "sim_only": lease.activation_ceiling == "SIM_ONLY" and not lease.hardware_authorized,
    }
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.independent_chain_credit_growth_evidence.v1",
        "status": "PASS_ROLE_LOCAL_FAILURE_CREDIT"
        if all(gates.values())
        else "REJECTED_ROLE_LOCAL_FAILURE_CREDIT",
        "passed": all(gates.values()),
        "source": {
            "report_hash": source["report_hash"],
            "result_hash": pass_row["result_hash"],
            "trajectory_file_hash": artifact["file_hash"],
            "trajectory_digest": artifact["trajectory_digest"],
        },
        "chain_request": chain_request.to_dict(),
        "chain_request_hash": chain_request.request_hash,
        "pass_event": event.to_dict(),
        "pass_event_hash": event.event_hash,
        "assessment": assessment.to_dict(),
        "assessment_hash": assessment.assessment_hash,
        "sample": sample.to_dict(),
        "sample_hash": sample.sample_hash,
        "actor": actor.to_dict(),
        "actor_hash": actor.actor_hash,
        "actor_artifact": {
            "file": actor_path.name,
            "file_hash": hash_bytes(actor_path.read_bytes()),
        },
        "training_reconstruction_error_m": reconstruction_error,
        "plasticity_lease": lease.to_dict(),
        "plasticity_lease_hash": lease.lease_hash,
        "parent_retention": {
            "status": "NOT_RUN_NO_CANDIDATE",
            "candidate_update_applied": False,
            "frozen_parent_policy_hashes": {
                cell.agent_id: cell.growth_scope.champion_policy.version_hash
                for cell in fixture.cells
            },
        },
        "gates": gates,
        "evidence_boundary": {
            "physics_authority": "CPU_MUJOCO",
            "activation_ceiling": "SIM_ONLY",
            "source_whole_body_g1_count": 6,
            "same_physical_ball": True,
            "continuous_pass_receive_shoot_save_complete": False,
            "independent_role_credit_assignment": True,
            "candidate_update_applied": False,
            "actor_deployment_ready": False,
            "pixels_used_for_scoring": False,
            "hardware_command_sent": False,
        },
        "implementation_hash": hash_json(
            {
                "growth": hash_bytes(Path(__file__).read_bytes()),
                "chain": hash_bytes(
                    (Path(__file__).parents[1] / "growth/continuous_option_chain.py").read_bytes()
                ),
                "calibration": hash_bytes(
                    (Path(__file__).parents[1] / "growth/pass_aim_calibration.py").read_bytes()
                ),
            }
        ),
    }
    report["report_hash"] = hash_json(report)
    _atomic_json(output / "chain-credit-exam.json", report)
    return validate_independent_chain_credit_growth(output / "chain-credit-exam.json")


def validate_independent_chain_credit_growth(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    value = json.loads(resolved.read_text(encoding="utf-8"))
    expected = value.pop("report_hash", None) if isinstance(value, dict) else None
    try:
        if not isinstance(value, dict):
            raise ValueError("S201 evidence must be an object")
        artifact = value.get("actor_artifact")
        actor = value.get("actor")
        assessment = value.get("assessment")
        lease = value.get("plasticity_lease")
        gates = value.get("gates")
        boundary = value.get("evidence_boundary")
        if not isinstance(artifact, dict) or not isinstance(artifact.get("file"), str):
            raise ValueError("S201 actor artifact binding is absent")
        actor_path = (resolved.parent / artifact["file"]).resolve()
        if resolved.parent != actor_path.parent or not actor_path.is_file():
            raise ValueError("S201 actor artifact escaped its evidence directory")
        loaded_actor = json.loads(actor_path.read_text(encoding="utf-8"))
        if (
            expected != hash_json(value)
            or hash_bytes(actor_path.read_bytes()) != artifact.get("file_hash")
            or loaded_actor != actor
            or not isinstance(gates, dict)
            or not all(item is True for item in gates.values())
            or not isinstance(assessment, dict)
            or assessment.get("earliest_failure") != "pass_inaccurate"
            or assessment.get("focal_agent_id") != "red.playmaker"
            or not isinstance(lease, dict)
            or sum(
                item.get("mode") == "PLASTIC"
                for item in cast(list[dict[str, Any]], lease.get("bindings", []))
            )
            != 1
            or not isinstance(boundary, dict)
            or boundary.get("continuous_pass_receive_shoot_save_complete") is not False
            or boundary.get("actor_deployment_ready") is not False
            or boundary.get("hardware_command_sent") is not False
            or value.get("passed") is not True
            or value.get("status") != "PASS_ROLE_LOCAL_FAILURE_CREDIT"
        ):
            raise ValueError("S201 evidence integrity or authority contract is invalid")
    finally:
        if isinstance(value, dict) and expected is not None:
            value["report_hash"] = expected
    return cast(dict[str, Any], value)


def _ball_state_hash(pose: np.ndarray[Any, Any], velocity: np.ndarray[Any, Any]) -> str:
    return str(hash_json({"pose": pose.tolist(), "velocity": velocity.tolist()}))


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".json", delete=False
    ) as descriptor:
        temporary = Path(descriptor.name)
        json.dump(value, descriptor, indent=2, sort_keys=True, allow_nan=False)
        descriptor.write("\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args()
    report = (
        validate_independent_chain_credit_growth(args.validate)
        if args.validate is not None
        else run_independent_chain_credit_growth(
            evidence_dir=args.evidence_dir,
            source_report_path=args.source_report,
            asset_root=args.asset_root,
        )
    )
    print(json.dumps({"status": report["status"], "report_hash": report["report_hash"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "run_independent_chain_credit_growth",
    "validate_independent_chain_credit_growth",
]
