"""S202 evidence bridge from learned role backends to fresh team physics.

Historical training evidence is useful only when it still selects the intended
role-qualified backend and reproduces in the current simulator.  This module
validates the frozen dynamic lead-pass artifact, routes it through an
independent playmaker cell, and executes a fresh deterministic PASS -> SHOOT
prefix with one shared physical ball.

The bridge remains high-level and SIM-only.  It does not expose a joint,
torque, ROS, DDS, vendor, or hardware execution path.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
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
from rosclaw_soccer.growth.dynamic_lead_pass import DynamicLeadPassPolicy
from rosclaw_soccer.growth.physical_option_router import PhysicalSoccerOption
from rosclaw_soccer.growth.role_option_backend import (
    RoleOptionBackend,
    RoleOptionBackendCandidate,
    select_role_option_backend,
)
from rosclaw_soccer.growth.role_self_model import MatchRole
from rosclaw_soccer.providers.g1.asset_qualification import (
    qualify_g1_assets,
    trajectory_digest,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.development_evidence import (
    three_role_development_kwargs,
)
from rosclaw_soccer.skills.team.shared_world import (
    G1SharedWorldResult,
    simulate_shared_world,
)
from rosclaw_soccer.training.dynamic_lead_pass_evidence import (
    validate_dynamic_lead_pass_evidence,
)
from rosclaw_soccer.training.independent_team_growth import (
    build_independent_three_vs_three_fixture,
)


def build_dynamic_lead_pass_candidate(
    source: dict[str, Any],
) -> RoleOptionBackendCandidate:
    """Adapt a validated S95 report into the stricter role-backend contract."""

    discovery = source.get("discovery")
    holdouts = source.get("holdouts")
    if not isinstance(discovery, dict) or not isinstance(holdouts, dict):
        raise ValueError("dynamic lead-pass source is missing discovery or holdouts")
    context_hashes: set[str] = set()
    trajectory_hashes: set[str] = set()
    for record in discovery.values():
        if not isinstance(record, dict) or not isinstance(record.get("sample"), dict):
            raise ValueError("dynamic lead-pass discovery record is invalid")
        sample = cast(dict[str, Any], record["sample"])
        context_hashes.add(
            str(
                hash_json(
                    {
                        "receiver_phase_start_sec": sample.get("receiver_phase_start_sec"),
                        "passer_yaw_delta_rad": sample.get("passer_yaw_delta_rad"),
                    }
                )
            )
        )
        digest = record.get("trajectory_digest")
        if not isinstance(digest, str):
            raise ValueError("dynamic lead-pass discovery trajectory is unbound")
        trajectory_hashes.add(digest)
    strict_replay = bool(holdouts) and all(
        isinstance(item, dict) and item.get("strict_replay") is True for item in holdouts.values()
    )
    holdout_passed = bool(holdouts) and all(
        isinstance(item, dict) and item.get("passed") is True for item in holdouts.values()
    )
    parent_retention_passed = bool(holdouts) and all(
        isinstance(item, dict)
        and isinstance(item.get("gates"), dict)
        and item["gates"].get("beats_fixed_parent") is True
        for item in holdouts.values()
    )
    return RoleOptionBackendCandidate(
        backend=RoleOptionBackend.DYNAMIC_LEAD_PASS,
        option=PhysicalSoccerOption.PASS,
        artifact_hash=str(source.get("policy_hash")),
        evidence_hash=str(source.get("evidence_hash")),
        distinct_context_count=len(context_hashes),
        distinct_trajectory_count=len(trajectory_hashes),
        strict_replay=strict_replay,
        holdout_passed=holdout_passed,
        parent_retention_passed=parent_retention_passed,
    )


def run_role_backend_continuity_evidence(
    *,
    asset_root: Path,
    output_dir: Path,
    source_checkout: Path,
    lead_pass_evidence_path: Path,
    lead_pass_policy_path: Path,
) -> dict[str, Any]:
    """Validate, route, and freshly replay a learned playmaker backend."""

    output = output_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("S202 evidence directory must be empty")
    if output == checkout or checkout in output.parents:
        raise ValueError("S202 evidence must be external to the source checkout")
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    source = validate_dynamic_lead_pass_evidence(lead_pass_evidence_path)
    policy_path = lead_pass_policy_path.expanduser().resolve()
    policy_value = json.loads(policy_path.read_text(encoding="utf-8"))
    if (
        not isinstance(policy_value, dict)
        or policy_value != source.get("policy")
        or hash_json(policy_value) != source.get("policy_hash")
    ):
        raise ValueError("dynamic lead-pass policy artifact does not match its evidence")
    policy_payload = dict(policy_value)
    policy_payload["discovery_sample_hashes"] = tuple(policy_payload["discovery_sample_hashes"])
    policy = DynamicLeadPassPolicy(**policy_payload)
    candidate = build_dynamic_lead_pass_candidate(source)
    fixture = build_independent_three_vs_three_fixture(asset_root)
    playmaker = next(cell for cell in fixture.cells if cell.agent_id == "red.playmaker")
    generic = RoleOptionBackendCandidate(
        backend=RoleOptionBackend.GENERIC_FREEKICK,
        option=PhysicalSoccerOption.PASS,
        artifact_hash=hash_json({"backend": "generic_freekick"}),
        evidence_hash=hash_json({"source": "s200-local-sweep"}),
        distinct_context_count=100,
        distinct_trajectory_count=100,
        strict_replay=True,
        holdout_passed=True,
        parent_retention_passed=True,
    )
    route = select_role_option_backend(
        cell=playmaker,
        option=PhysicalSoccerOption.PASS,
        candidates=(generic, candidate),
    )
    if route.selected_backend is not RoleOptionBackend.DYNAMIC_LEAD_PASS:
        raise RuntimeError("S202 did not select the learned role-qualified backend")

    # Re-certify the learned action on one of its sealed source holdouts.  The
    # default team phase (2.19 s) lies outside the policy's discovery envelope
    # and a seemingly more precise extrapolation can push the receiver into a
    # joint-limit violation.  Evidence-qualified routing must not extrapolate.
    holdouts = cast(dict[str, Any], source["holdouts"])
    holdout_id, selected_holdout = max(
        (
            (str(case_id), cast(dict[str, Any], item))
            for case_id, item in holdouts.items()
            if isinstance(item, dict)
            and isinstance(item.get("case"), dict)
            and item.get("passed") is True
        ),
        key=lambda item: (
            float(cast(dict[str, Any], item[1]["case"])["receiver_lateral_lane_m"]),
            item[0],
        ),
    )
    holdout_case = cast(dict[str, Any], selected_holdout["case"])
    receiver_phase = float(holdout_case["receiver_phase_start_sec"])
    receiver_lane = float(holdout_case["receiver_lateral_lane_m"])
    reception_target = policy.reception_target(
        receiver_phase_start_sec=receiver_phase,
        receiver_lateral_lane_m=receiver_lane,
    )
    passer_yaw = policy.passer_world_yaw(target_lateral_m=receiver_lane)
    kwargs = three_role_development_kwargs()
    kwargs.update(
        shooter_start_sec=receiver_phase,
        shooter_origin=(0.0, receiver_lane, 0.0),
        pass_reception_target_m=reception_target,
        passer_yaw_rad=passer_yaw,
    )
    request = {
        "schema_version": "rosclaw_soccer.role_backend_continuity_request.v1",
        "source_commit": _git_head(checkout),
        "body_hash": qualification.body_hash,
        "kick_prior_hash": qualification.kick_prior_hash,
        "fixture_hash": fixture.fixture_hash,
        "playmaker_cell_hash": playmaker.cell_hash,
        "candidate_hash": candidate.candidate_hash,
        "route_hash": route.route_hash,
        "source_sealed_holdout_id": holdout_id,
        "receiver_phase_start_sec": receiver_phase,
        "receiver_lateral_lane_m": receiver_lane,
        "pass_reception_target_m": list(reception_target),
        "executed_passer_yaw_rad": passer_yaw,
        "runtime": _runtime_manifest(),
        "physics_authority": "CPU_MUJOCO",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "request.json", request)
    result, trajectory = simulate_shared_world(asset_root, **kwargs)
    replay_result, replay_trajectory = simulate_shared_world(asset_root, **kwargs)
    digest = trajectory_digest(trajectory)
    replay_digest = trajectory_digest(replay_trajectory)
    strict_replay = bool(result.to_dict() == replay_result.to_dict() and digest == replay_digest)
    safe = _safe_team_result(result)
    chain_request, chain_events = _physical_chain_prefix(
        fixture=fixture,
        result=result,
        trajectory=trajectory,
        route_hash=route.route_hash,
        goal_target=cast(tuple[float, float, float], kwargs["shooter_target"]),
        safe=safe,
        exact_replay=strict_replay,
    )
    assessment = assess_continuous_option_chain(chain_request, chain_events)
    lease = build_chain_repair_lease(
        cells=fixture.cells,
        assessment=assessment,
        dataset_manifest_hash=digest,
        scenario_contract_hash=chain_request.request_hash,
        maximum_optimizer_steps=2_000,
    )
    gates = {
        "source_evidence_valid": source.get("passed") is True,
        "candidate_evidence_ready": candidate.evidence_ready,
        "generic_backend_rejected": not generic.evidence_ready,
        "role_backend_selected": route.accepted
        and route.selected_backend is RoleOptionBackend.DYNAMIC_LEAD_PASS,
        "action_conditioned": abs(passer_yaw - np.pi) > 1.0e-6,
        "fresh_strict_replay": strict_replay,
        "same_ball_pass_then_shot": bool(
            result.pass_contact_time_sec is not None
            and result.shot_contact_time_sec is not None
            and result.pass_contact_time_sec < result.shot_contact_time_sec
        ),
        "fresh_delivery_precision": bool(
            result.pass_delivery_error_m is not None and result.pass_delivery_error_m <= 0.05
        ),
        "receiver_kept_moving": result.receiver_phase_hold_frames == 0,
        "effective_pass": result.pass_peak_ball_speed_mps >= 1.0,
        "effective_shot": result.shot_peak_ball_speed_mps >= 4.0,
        "fresh_team_safe": safe,
        "next_failure_attributed_to_finisher": bool(
            assessment.earliest_failure is ContinuousChainFailure.SHOT_INACCURATE
            and assessment.focal_agent_id == "red.finisher"
            and assessment.completed_phase_count == 2
            and assessment.downstream_credit_blocked
        ),
        "only_finisher_plastic": bool(
            lease.focal_agent_id == "red.finisher"
            and len(lease.bindings) == 6
            and sum(binding.mode.value == "PLASTIC" for binding in lease.bindings) == 1
        ),
    }
    trajectory_path = output / "routed-continuity-trajectory.npz"
    np.savez_compressed(trajectory_path, **trajectory)  # type: ignore[arg-type]
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.role_backend_continuity_evidence.v1",
        "status": "PASS_ROLE_BACKEND_AND_CAUSAL_HANDOFF"
        if all(gates.values())
        else "REJECTED_ROLE_BACKEND_AND_CAUSAL_HANDOFF",
        "passed": all(gates.values()),
        "source": {
            "evidence_hash": source["evidence_hash"],
            "evidence_file_hash": hash_bytes(
                lead_pass_evidence_path.expanduser().resolve().read_bytes()
            ),
            "policy_hash": source["policy_hash"],
            "policy_file_hash": hash_bytes(policy_path.read_bytes()),
        },
        "candidate": candidate.to_dict(),
        "candidate_hash": candidate.candidate_hash,
        "rejected_generic_candidate": generic.to_dict(),
        "route": route.to_dict(),
        "route_hash": route.route_hash,
        "request": request,
        "request_hash": hash_bytes((output / "request.json").read_bytes()),
        "result": result.to_dict(),
        "result_hash": hash_json(result.to_dict()),
        "chain_request": chain_request.to_dict(),
        "chain_request_hash": chain_request.request_hash,
        "chain_events": [event.to_dict() for event in chain_events],
        "chain_assessment": assessment.to_dict(),
        "chain_assessment_hash": assessment.assessment_hash,
        "next_growth_lease": lease.to_dict(),
        "next_growth_lease_hash": lease.lease_hash,
        "trajectory": {
            "file": trajectory_path.name,
            "file_hash": hash_bytes(trajectory_path.read_bytes()),
            "trajectory_digest": digest,
            "replay_trajectory_digest": replay_digest,
        },
        "gates": gates,
        "evidence_boundary": {
            "physics_authority": "CPU_MUJOCO",
            "activation_ceiling": "SIM_ONLY",
            "fresh_whole_body_g1_count": 3,
            "same_physical_ball": True,
            "physical_pass_receive_shoot_contacts_complete": True,
            "continuous_chain_passed": assessment.passed,
            "continuous_save_complete": result.goalkeeper_ball_contact_observed,
            "next_plastic_agent_id": assessment.focal_agent_id,
            "candidate_promoted_to_hardware": False,
            "pixels_used_for_scoring": False,
            "hardware_command_sent": False,
        },
        "implementation_hash": _implementation_hash(),
    }
    report["report_hash"] = hash_json(report)
    _write_json(output / "role-backend-continuity.json", report)
    return validate_role_backend_continuity_evidence(output / "role-backend-continuity.json")


def validate_role_backend_continuity_evidence(path: Path) -> dict[str, Any]:
    """Fail closed if report, trajectory, or role authority was edited."""

    resolved = path.expanduser().resolve()
    value = json.loads(resolved.read_text(encoding="utf-8"))
    expected = value.pop("report_hash", None) if isinstance(value, dict) else None
    try:
        if not isinstance(value, dict):
            raise ValueError("S202 evidence must be an object")
        trajectory = value.get("trajectory")
        gates = value.get("gates")
        route = value.get("route")
        candidate = value.get("candidate")
        generic = value.get("rejected_generic_candidate")
        request = value.get("request")
        result = value.get("result")
        chain_request = value.get("chain_request")
        chain_events = value.get("chain_events")
        assessment = value.get("chain_assessment")
        lease = value.get("next_growth_lease")
        boundary = value.get("evidence_boundary")
        bindings = lease.get("bindings") if isinstance(lease, dict) else None
        if (
            expected != hash_json(value)
            or value.get("passed") is not True
            or value.get("status") != "PASS_ROLE_BACKEND_AND_CAUSAL_HANDOFF"
            or not isinstance(gates, dict)
            or not gates
            or not all(item is True for item in gates.values())
            or not isinstance(route, dict)
            or route.get("selected_backend") != "dynamic_lead_pass"
            or route.get("accepted") is not True
            or not isinstance(candidate, dict)
            or candidate.get("backend") != "dynamic_lead_pass"
            or candidate.get("option") != "pass"
            or candidate.get("evidence_ready") is not True
            or value.get("candidate_hash") != hash_json(candidate)
            or route.get("candidate_hash") != value.get("candidate_hash")
            or value.get("route_hash") != hash_json(route)
            or not isinstance(generic, dict)
            or generic.get("backend") != "generic_freekick"
            or generic.get("evidence_ready") is not False
            or not isinstance(request, dict)
            or not isinstance(result, dict)
            or value.get("result_hash") != hash_json(result)
            or not isinstance(chain_request, dict)
            or value.get("chain_request_hash") != hash_json(chain_request)
            or not isinstance(chain_events, list)
            or len(chain_events) != 3
            or not all(isinstance(item, dict) for item in chain_events)
            or not isinstance(assessment, dict)
            or value.get("chain_assessment_hash") != hash_json(assessment)
            or assessment.get("event_hashes")
            != [hash_json(cast(dict[str, Any], item)) for item in chain_events]
            or assessment.get("request_hash") != value.get("chain_request_hash")
            or assessment.get("earliest_failure") != "shot_inaccurate"
            or assessment.get("focal_agent_id") != "red.finisher"
            or assessment.get("completed_phase_count") != 2
            or assessment.get("downstream_credit_blocked") is not True
            or assessment.get("ball_lineage_verified") is not True
            or assessment.get("safe") is not True
            or assessment.get("exact_replay") is not True
            or assessment.get("passed") is not False
            or not isinstance(lease, dict)
            or value.get("next_growth_lease_hash") != hash_json(lease)
            or not isinstance(bindings, list)
            or len(bindings) != 6
            or sum(isinstance(item, dict) and item.get("mode") == "PLASTIC" for item in bindings)
            != 1
            or not any(
                isinstance(item, dict)
                and item.get("agent_id") == "red.finisher"
                and item.get("mode") == "PLASTIC"
                for item in bindings
            )
            or not isinstance(boundary, dict)
            or boundary.get("activation_ceiling") != "SIM_ONLY"
            or boundary.get("hardware_command_sent") is not False
            or boundary.get("pixels_used_for_scoring") is not False
            or boundary.get("continuous_chain_passed") is not False
            or boundary.get("next_plastic_agent_id") != "red.finisher"
            or not isinstance(trajectory, dict)
            or not isinstance(trajectory.get("file"), str)
        ):
            raise ValueError("S202 evidence authority contract is invalid")
        trajectory_path = (resolved.parent / trajectory["file"]).resolve()
        request_path = resolved.parent / "request.json"
        if (
            trajectory_path.parent != resolved.parent
            or not trajectory_path.is_file()
            or hash_bytes(trajectory_path.read_bytes()) != trajectory.get("file_hash")
            or not request_path.is_file()
            or hash_bytes(request_path.read_bytes()) != value.get("request_hash")
            or json.loads(request_path.read_text(encoding="utf-8")) != request
        ):
            raise ValueError("S202 trajectory binding changed")
        with np.load(trajectory_path, allow_pickle=False) as archive:
            loaded_trajectory = {name: archive[name] for name in archive.files}
        if trajectory_digest(loaded_trajectory) != trajectory.get("trajectory_digest"):
            raise ValueError("S202 trajectory semantic digest changed")
    finally:
        if isinstance(value, dict) and expected is not None:
            value["report_hash"] = expected
    return value


def _physical_chain_prefix(
    *,
    fixture: Any,
    result: G1SharedWorldResult,
    trajectory: dict[str, np.ndarray],
    route_hash: str,
    goal_target: tuple[float, float, float],
    safe: bool,
    exact_replay: bool,
) -> tuple[ContinuousOptionChainRequest, tuple[ContinuousChainEvent, ...]]:
    if result.pass_contact_time_sec is None or result.shot_contact_time_sec is None:
        raise ValueError("S202 chain prefix requires physical pass and shot contacts")
    if result.pass_delivery_error_m is None or result.target_error_m is None:
        raise ValueError("S202 chain prefix requires physical target errors")
    time = np.asarray(trajectory["time"], dtype=np.float64)
    ball_pose = np.asarray(trajectory["ball_pose"], dtype=np.float64)
    ball_velocity = np.asarray(trajectory["ball_velocity"], dtype=np.float64)
    if (
        time.ndim != 1
        or ball_pose.shape != (len(time), 7)
        or ball_velocity.shape != (len(time), 6)
        or len(time) < 4
        or not np.all(np.isfinite(time))
        or not np.all(np.isfinite(ball_pose))
        or not np.all(np.isfinite(ball_velocity))
    ):
        raise ValueError("S202 physical trajectory contract changed")
    pass_index = int(np.searchsorted(time, result.pass_contact_time_sec, side="left"))
    shot_index = int(np.searchsorted(time, result.shot_contact_time_sec, side="left"))
    if not 0 <= pass_index < shot_index - 2 < len(time):
        raise ValueError("S202 contacts do not leave a receive interval")
    pass_end_index = shot_index - 2
    receive_end_index = shot_index - 1
    shot_end_index = min(shot_index + 150, len(time) - 1)
    by_id = {cell.agent_id: cell for cell in fixture.cells}
    selected = (
        ("red.playmaker", "playmaker"),
        ("red.finisher", "finisher"),
        ("blue.goalkeeper", "goalkeeper"),
    )
    request = ContinuousOptionChainRequest(
        chain_id="s202.chain.role-backend-continuity",
        roster_hash=fixture.roster.roster_hash,
        ball_id="match.ball.0001",
        initial_ball_state_hash=_ball_state_hash(ball_pose[0], ball_velocity[0]),
        role_bindings=tuple(
            ChainRoleBinding(
                agent_id=agent_id,
                role=MatchRole(role),
                cell_hash=by_id[agent_id].cell_hash,
                champion_policy_hash=by_id[agent_id].growth_scope.champion_policy.version_hash,
                parent_policy_hash=by_id[agent_id].growth_scope.parent_policy.version_hash,
            )
            for agent_id, role in selected
        ),
        goal_target_m=goal_target,
        maximum_pass_error_m=0.05,
        maximum_receive_error_m=0.05,
        maximum_shot_error_m=0.10,
    )
    pass_output = _ball_state_hash(ball_pose[pass_end_index], ball_velocity[pass_end_index])
    receive_output = _ball_state_hash(
        ball_pose[receive_end_index], ball_velocity[receive_end_index]
    )
    shot_output = _ball_state_hash(ball_pose[shot_end_index], ball_velocity[shot_end_index])
    receive_speed = float(np.linalg.norm(ball_velocity[receive_end_index, :3]))
    events = (
        ContinuousChainEvent(
            phase=ContinuousChainPhase.PASS,
            agent_id="red.playmaker",
            ball_id=request.ball_id,
            input_ball_state_hash=request.initial_ball_state_hash,
            output_ball_state_hash=pass_output,
            decision_hash=route_hash,
            option_request_hash=hash_json({"route_hash": route_hash, "phase": "pass"}),
            started_at_sec=0.0,
            ended_at_sec=float(time[pass_end_index]),
            contact_observed=True,
            target_error_m=float(result.pass_delivery_error_m),
            post_contact_ball_speed_mps=float(result.pass_peak_ball_speed_mps),
            safe=safe,
            phase_ready=True,
            exact_replay=exact_replay,
        ),
        ContinuousChainEvent(
            phase=ContinuousChainPhase.RECEIVE,
            agent_id="red.finisher",
            ball_id=request.ball_id,
            input_ball_state_hash=pass_output,
            output_ball_state_hash=receive_output,
            decision_hash=hash_json({"controller": "shared_world_receiver", "phase": "receive"}),
            option_request_hash=hash_json(
                {"trajectory_digest": trajectory_digest(trajectory), "phase": "receive"}
            ),
            started_at_sec=float(time[pass_end_index]),
            ended_at_sec=float(time[receive_end_index]),
            contact_observed=False,
            target_error_m=float(result.pass_delivery_error_m),
            post_contact_ball_speed_mps=receive_speed,
            safe=safe,
            phase_ready=True,
            exact_replay=exact_replay,
        ),
        ContinuousChainEvent(
            phase=ContinuousChainPhase.SHOOT,
            agent_id="red.finisher",
            ball_id=request.ball_id,
            input_ball_state_hash=receive_output,
            output_ball_state_hash=shot_output,
            decision_hash=hash_json({"controller": "shared_world_finisher", "phase": "shoot"}),
            option_request_hash=hash_json(
                {"trajectory_digest": trajectory_digest(trajectory), "phase": "shoot"}
            ),
            started_at_sec=float(time[receive_end_index]),
            ended_at_sec=float(time[shot_end_index]),
            contact_observed=True,
            target_error_m=float(result.target_error_m),
            post_contact_ball_speed_mps=float(result.shot_peak_ball_speed_mps),
            safe=safe,
            phase_ready=True,
            exact_replay=exact_replay,
        ),
    )
    return request, events


def _ball_state_hash(pose: np.ndarray, velocity: np.ndarray) -> str:
    return str(
        hash_json(
            {
                "pose": [float(value) for value in pose],
                "velocity": [float(value) for value in velocity],
            }
        )
    )


def _safe_team_result(result: G1SharedWorldResult) -> bool:
    return bool(
        result.finite_state
        and not result.joint_limit_violation
        and not result.torque_limit_violation
        and not result.actuator_saturation
        and result.robot_robot_contact_count == 0
        and result.passer_min_pelvis_height_m >= 0.60
        and result.shooter_min_pelvis_height_m >= 0.60
        and result.goalkeeper_min_pelvis_height_m is not None
        and result.goalkeeper_min_pelvis_height_m >= 0.70
    )


def _runtime_manifest() -> dict[str, str]:
    import mujoco
    import torch

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "mujoco": mujoco.__version__,
        "torch": torch.__version__,
        "torch_cuda": str(torch.version.cuda),
    }


def _git_head(checkout: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()


def _implementation_hash() -> str:
    return str(
        hash_json(
            {
                path.name: hash_bytes(path.read_bytes())
                for path in (
                    Path(__file__),
                    Path(__file__).parents[1] / "growth/role_option_backend.py",
                    Path(__file__).parents[1] / "growth/dynamic_lead_pass.py",
                )
            }
        )
    )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("--lead-pass-evidence", type=Path, required=True)
    parser.add_argument("--lead-pass-policy", type=Path, required=True)
    args = parser.parse_args()
    report = run_role_backend_continuity_evidence(
        asset_root=args.asset_root,
        output_dir=args.output_dir,
        source_checkout=args.source_checkout,
        lead_pass_evidence_path=args.lead_pass_evidence,
        lead_pass_policy_path=args.lead_pass_policy,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "build_dynamic_lead_pass_candidate",
    "run_role_backend_continuity_evidence",
    "validate_role_backend_continuity_evidence",
]
