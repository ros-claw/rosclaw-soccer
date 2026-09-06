"""S203 role-local target calibration around a frozen whole-body kick prior."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from rosclaw_soccer.growth.contextual_finish_target import (
    FinishTargetCalibrationSample,
    fit_contextual_finish_target_actor,
    load_contextual_finish_target_actor,
    save_contextual_finish_target_actor,
)
from rosclaw_soccer.growth.dynamic_lead_pass import DynamicLeadPassPolicy
from rosclaw_soccer.growth.physical_option_router import PhysicalSoccerOption
from rosclaw_soccer.growth.role_option_backend import (
    RoleOptionBackend,
    RoleOptionBackendCandidate,
)
from rosclaw_soccer.growth.runtime_finish_plan_actor import prepared_finish_plan_features
from rosclaw_soccer.providers.g1.asset_qualification import (
    qualify_g1_assets,
    trajectory_digest,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.development_evidence import three_role_development_kwargs
from rosclaw_soccer.skills.team.shared_world import (
    G1JointGuardConfig,
    G1SharedWorldResult,
    simulate_shared_world,
)
from rosclaw_soccer.training.independent_team_growth import (
    build_independent_three_vs_three_fixture,
)
from rosclaw_soccer.training.role_backend_continuity_evidence import (
    validate_role_backend_continuity_evidence,
)


@dataclass(frozen=True)
class ContextualFinishTargetGrowthConfig:
    policy_target_y_candidates_m: tuple[float, ...] = (
        0.25,
        0.255,
        0.26,
        0.265,
        0.27,
        0.275,
        0.28,
        0.285,
    )
    foot_yaw_offset_candidates_rad: tuple[float, ...] = (0.04, 0.06, 0.08, 0.10)
    policy_target_z_m: float = 0.50
    shooter_precontact_joint_guard_enabled: bool = True
    shooter_joint_guard_margin_rad: float = 0.06
    shooter_joint_guard_prediction_horizon_sec: float = 0.16
    shooter_joint_guard_boundary_kp: float = 120.0
    shooter_joint_guard_boundary_kd: float = 10.0
    shooter_post_policy_frame: int = 258
    shooter_post_policy_blend_frames: int = 2
    simulation_duration_sec: float = 10.0
    maximum_target_error_m: float = 0.10
    maximum_pelvis_regression_m: float = 0.03
    maximum_support_slip_regression_m: float = 0.05
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False

    def __post_init__(self) -> None:
        candidates = np.asarray(self.policy_target_y_candidates_m, dtype=np.float64)
        yaw_candidates = np.asarray(self.foot_yaw_offset_candidates_rad, dtype=np.float64)
        if (
            candidates.ndim != 1
            or not 4 <= candidates.size <= 32
            or len(set(self.policy_target_y_candidates_m)) != candidates.size
            or not np.all(np.isfinite(candidates))
            or np.any(np.abs(candidates) > 2.0)
            or yaw_candidates.ndim != 1
            or not 1 <= yaw_candidates.size <= 16
            or len(set(self.foot_yaw_offset_candidates_rad)) != yaw_candidates.size
            or not np.all(np.isfinite(yaw_candidates))
            or np.any(np.abs(yaw_candidates) > 0.12)
            or candidates.size * yaw_candidates.size > 32
            or not math.isfinite(self.policy_target_z_m)
            or not 0.05 <= self.policy_target_z_m <= 2.50
            or self.shooter_precontact_joint_guard_enabled is not True
            or not 0.01 <= self.shooter_joint_guard_margin_rad <= 0.10
            or not 0.02 <= self.shooter_joint_guard_prediction_horizon_sec <= 0.20
            or not 20.0 <= self.shooter_joint_guard_boundary_kp <= 200.0
            or not 1.0 <= self.shooter_joint_guard_boundary_kd <= 20.0
            or not 250 <= self.shooter_post_policy_frame <= 290
            or not 0 <= self.shooter_post_policy_blend_frames <= 40
            or not 8.0 <= self.simulation_duration_sec <= 15.0
            or not 0.05 <= self.maximum_target_error_m <= 0.20
            or not 0.0 <= self.maximum_pelvis_regression_m <= 0.10
            or not 0.0 <= self.maximum_support_slip_regression_m <= 0.15
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("contextual finish target growth config is invalid")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


def run_contextual_finish_target_growth(
    *,
    asset_root: Path,
    source_s202_path: Path,
    source_lead_pass_dir: Path,
    source_checkout: Path,
    output_dir: Path,
    config: ContextualFinishTargetGrowthConfig | None = None,
    workers: int = 4,
) -> dict[str, Any]:
    """Search target intent only, then strictly replay the selected outcome."""

    if not 1 <= workers <= 4:
        raise ValueError("contextual finish target workers must be in [1, 4]")
    active = config or ContextualFinishTargetGrowthConfig()
    source_path = source_s202_path.expanduser().resolve()
    source = validate_role_backend_continuity_evidence(source_path)
    lease_bindings = source.get("next_growth_lease", {}).get("bindings", ())
    plastic_agents = {
        str(binding.get("agent_id"))
        for binding in lease_bindings
        if isinstance(binding, dict) and binding.get("mode") == "PLASTIC"
    }
    if (
        source.get("status") != "PASS_ROLE_BACKEND_AND_CAUSAL_HANDOFF"
        or source.get("chain_assessment", {}).get("focal_agent_id") != "red.finisher"
        or plastic_agents != {"red.finisher"}
    ):
        raise ValueError("S203 requires a passing S202 finisher handoff")
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    if source["request"]["body_hash"] != qualification.body_hash:
        raise ValueError("S203 source body changed")
    policy, policy_source = _load_lead_pass(source_lead_pass_dir)
    if (
        source["source"]["policy_hash"] != policy.artifact_hash
        or source["source"]["evidence_hash"] != policy_source["evidence_hash"]
    ):
        raise ValueError("S203 lead-pass lineage changed")
    fixture = build_independent_three_vs_three_fixture(asset_root)
    finisher = next(cell for cell in fixture.cells if cell.agent_id == "red.finisher")
    output = _new_external_output(output_dir, source_checkout)
    request_values = cast(dict[str, Any], source["request"])
    raw_physical_target = source["chain_request"]["goal_target_m"]
    physical_target = (
        float(raw_physical_target[0]),
        float(raw_physical_target[1]),
        float(raw_physical_target[2]),
    )
    base_kwargs = three_role_development_kwargs()
    parent_policy_target = cast(tuple[float, float, float], base_kwargs["shooter_policy_target"])
    request: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.contextual_finish_target_growth_request.v1",
        "source_commit": _git_head(source_checkout),
        "source_s202_hash": source["report_hash"],
        "source_s202_file_hash": hash_bytes(source_path.read_bytes()),
        "source_lease_hash": source["next_growth_lease_hash"],
        "source_lead_pass_evidence_hash": policy_source["evidence_hash"],
        "source_lead_pass_policy_hash": policy.artifact_hash,
        "body_hash": qualification.body_hash,
        "kick_prior_hash": qualification.kick_prior_hash,
        "fixture_hash": fixture.fixture_hash,
        "roster_hash": fixture.roster.roster_hash,
        "finisher_cell_hash": finisher.cell_hash,
        "finisher_self_model_hash": finisher.self_model.self_model_hash,
        "physical_target_m": list(physical_target),
        "parent_policy_target_m": list(parent_policy_target),
        "parent_foot_yaw_offset_rad": base_kwargs["shooter_parameter_overrides"]["foot_yaw_offset"],
        "receiver_phase_start_sec": request_values["receiver_phase_start_sec"],
        "receiver_lateral_lane_m": request_values["receiver_lateral_lane_m"],
        "pass_reception_target_m": request_values["pass_reception_target_m"],
        "passer_yaw_rad": request_values["executed_passer_yaw_rad"],
        "passer_ball_local_xy_m": list(base_kwargs["passer_ball_local_xy"]),
        "passer_stance_offset_xy_m": [0.0, 0.0],
        "passer_swing_speed_scale": 0.80,
        "ball_ground_friction": base_kwargs["ball_ground_friction"],
        "config": asdict(active),
        "config_hash": active.config_hash,
        "control_envelope_hash": _control_envelope_hash(active),
        "implementation_hash": _implementation_hash(),
        "runtime": _runtime_manifest(),
        "physics_authority": "CPU_MUJOCO",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    _write_json(output / "request.json", request)
    jobs = [
        (
            asset_root.expanduser().resolve(),
            _simulation_kwargs(
                request=request,
                policy_target=(physical_target[0], value, active.policy_target_z_m),
                foot_yaw_offset_rad=foot_yaw,
                duration=active.simulation_duration_sec,
                config=active,
            ),
        )
        for value in active.policy_target_y_candidates_m
        for foot_yaw in active.foot_yaw_offset_candidates_rad
    ]
    if workers == 1:
        outcomes = [simulate_shared_world(asset, **kwargs) for asset, kwargs in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(simulate_shared_world, asset, **kwargs) for asset, kwargs in jobs
            ]
            outcomes = [future.result() for future in futures]
    rows = []
    for index, ((_, kwargs), (result, trajectory)) in enumerate(zip(jobs, outcomes, strict=True)):
        artifact = _save_trajectory(output / f"candidate-{index:03d}.npz", trajectory)
        rows.append(
            _candidate_row(
                index=index,
                policy_target=cast(tuple[float, float, float], kwargs["shooter_policy_target"]),
                foot_yaw_offset_rad=float(kwargs["shooter_parameter_overrides"]["foot_yaw_offset"]),
                result=result,
                trajectory=artifact,
            )
        )
    parent_result, parent_trajectory = simulate_shared_world(
        asset_root,
        **_simulation_kwargs(
            request=request,
            policy_target=parent_policy_target,
            foot_yaw_offset_rad=float(request["parent_foot_yaw_offset_rad"]),
            duration=active.simulation_duration_sec,
            config=active,
        ),
    )
    parent_row = _candidate_row(
        index=-1,
        policy_target=parent_policy_target,
        foot_yaw_offset_rad=float(request["parent_foot_yaw_offset_rad"]),
        result=parent_result,
        trajectory=_save_trajectory(output / "parent.npz", parent_trajectory),
    )
    selected = min(rows, key=_selection_key)
    selected_kwargs = _simulation_kwargs(
        request=request,
        policy_target=cast(tuple[float, float, float], tuple(selected["policy_target_m"])),
        foot_yaw_offset_rad=float(selected["foot_yaw_offset_rad"]),
        duration=active.simulation_duration_sec,
        config=active,
    )
    replay_result, replay_trajectory = simulate_shared_world(asset_root, **selected_kwargs)
    replay_artifact = _save_trajectory(output / "selected-replay.npz", replay_trajectory)
    exact_replay = bool(
        selected["result"] == replay_result.to_dict()
        and selected["trajectory"]["trajectory_digest"] == replay_artifact["trajectory_digest"]
    )
    metrics, gates = _derive_metrics_and_gates(
        rows=rows,
        selected=selected,
        parent=parent_row,
        exact_replay=exact_replay,
        config=active,
    )
    features = prepared_finish_plan_features(
        receiver_lane_m=float(request["receiver_lateral_lane_m"]),
        reception_target_x_m=float(request["pass_reception_target_m"][0]),
        passer_ball_local_xy_m=request["passer_ball_local_xy_m"],
        ball_ground_friction=float(request["ball_ground_friction"]),
        passer_yaw_rad=float(request["passer_yaw_rad"]),
        passer_stance_offset_xy_m=request["passer_stance_offset_xy_m"],
        passer_swing_speed_scale=float(request["passer_swing_speed_scale"]),
    )
    actor_record: dict[str, Any] | None = None
    backend_candidate: RoleOptionBackendCandidate | None = None
    if gates["selected_precise_safe"] and exact_replay:
        selected_result = cast(dict[str, Any], selected["result"])
        crossing = (
            physical_target[0],
            float(selected_result["goal_crossing_y_m"]),
            float(selected_result["goal_crossing_z_m"]),
        )
        sample = FinishTargetCalibrationSample(
            context_hash=str(hash_json({"features": features})),
            trajectory_hash=str(selected["trajectory"]["trajectory_digest"]),
            control_envelope_hash=str(request["control_envelope_hash"]),
            features=features,
            requested_physical_target_m=physical_target,
            executed_policy_target_m=cast(
                tuple[float, float, float], tuple(selected["policy_target_m"])
            ),
            executed_foot_yaw_offset_rad=float(selected["foot_yaw_offset_rad"]),
            observed_crossing_m=crossing,
            target_error_m=float(selected_result["target_error_m"]),
            safe=True,
            exact_replay=True,
        )
        actor = fit_contextual_finish_target_actor(
            body_hash=qualification.body_hash,
            kick_prior_hash=qualification.kick_prior_hash,
            roster_hash=fixture.roster.roster_hash,
            finisher_self_model_hash=finisher.self_model.self_model_hash,
            control_envelope_hash=str(request["control_envelope_hash"]),
            source_evidence_hashes=(str(source["report_hash"]),),
            samples=(sample,),
        )
        actor_path = output / "contextual-finish-target-actor.json"
        save_contextual_finish_target_actor(actor, actor_path)
        actor_record = {
            "file": actor_path.name,
            "file_hash": hash_bytes(actor_path.read_bytes()),
            "actor_hash": actor.actor_hash,
            "evidence_ready": actor.evidence_ready,
            "distinct_context_count": actor.distinct_context_count,
            "distinct_trajectory_count": actor.distinct_trajectory_count,
        }
        backend_candidate = RoleOptionBackendCandidate(
            backend=RoleOptionBackend.CONTEXTUAL_FINISH_TARGET,
            option=PhysicalSoccerOption.SHOOT,
            artifact_hash=actor.actor_hash,
            evidence_hash=str(source["report_hash"]),
            distinct_context_count=actor.distinct_context_count,
            distinct_trajectory_count=actor.distinct_trajectory_count,
            strict_replay=exact_replay,
            holdout_passed=False,
            parent_retention_passed=gates["selected_stability_retained"],
        )
        gates["single_context_seed_not_deployable"] = not backend_candidate.evidence_ready
    else:
        gates["single_context_seed_not_deployable"] = True
    passed = all(gates.values())
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.contextual_finish_target_growth.v1",
        "status": (
            "PASS_SINGLE_CONTEXT_FINISH_TARGET_SEED"
            if passed
            else "REJECTED_CONTEXTUAL_FINISH_TARGET_GROWTH"
        ),
        "passed": passed,
        "promotion_eligible": False,
        "request_hash": hash_bytes((output / "request.json").read_bytes()),
        "source_s202_hash": source["report_hash"],
        "source_lease_hash": source["next_growth_lease_hash"],
        "finisher_cell_hash": finisher.cell_hash,
        "metrics": metrics,
        "gates": gates,
        "selected": selected,
        "selected_replay": {
            "result": replay_result.to_dict(),
            "trajectory": replay_artifact,
            "exact_replay": exact_replay,
        },
        "parent": parent_row,
        "rows": rows,
        "actor": actor_record,
        "backend_candidate": (None if backend_candidate is None else backend_candidate.to_dict()),
        "implementation_hash": _implementation_hash(),
        "evidence_boundary": {
            "partition": "DEVELOPMENT",
            "plastic_agent_id": "red.finisher",
            "frozen_role_count": 5,
            "joint_torque_owner": "FROZEN_WHOLE_BODY_KICK_PRIOR",
            "pixels_used_for_scoring": False,
            "activation_ceiling": "SIM_ONLY",
            "hardware_command_sent": False,
        },
    }
    report["report_hash"] = hash_json(report)
    _write_json(output / "contextual-finish-target-growth.json", report)
    return report


def validate_contextual_finish_target_growth(path: Path) -> dict[str, Any]:
    report_path = path.expanduser().resolve()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("S203 report must be an object")
    claimed_hash = payload.pop("report_hash", None)
    try:
        request_path = report_path.parent / "request.json"
        request = (
            json.loads(request_path.read_text(encoding="utf-8")) if request_path.is_file() else None
        )
        rows = payload.get("rows")
        replay = payload.get("selected_replay")
        if (
            claimed_hash != hash_json(payload)
            or payload.get("implementation_hash") != _implementation_hash()
            or not request_path.is_file()
            or hash_bytes(request_path.read_bytes()) != payload.get("request_hash")
            or not isinstance(request, dict)
            or request.get("source_s202_hash") != payload.get("source_s202_hash")
            or request.get("source_lease_hash") != payload.get("source_lease_hash")
            or not isinstance(rows, list)
            or not isinstance(replay, dict)
        ):
            raise ValueError("S203 report integrity changed")
        for row in (*rows, payload.get("parent"), replay):
            if not isinstance(row, dict) or not isinstance(row.get("trajectory"), dict):
                raise ValueError("S203 trajectory record is invalid")
            _validate_trajectory(report_path.parent, row["trajectory"])
        actor_record = payload.get("actor")
        loaded_actor = None
        if actor_record is not None:
            if not isinstance(actor_record, dict):
                raise ValueError("S203 actor record is invalid")
            actor_path = report_path.parent / str(actor_record.get("file", ""))
            loaded_actor = load_contextual_finish_target_actor(actor_path)
            if (
                not actor_path.is_file()
                or hash_bytes(actor_path.read_bytes()) != actor_record.get("file_hash")
                or loaded_actor.actor_hash != actor_record.get("actor_hash")
                or loaded_actor.body_hash != request.get("body_hash")
                or loaded_actor.kick_prior_hash != request.get("kick_prior_hash")
                or loaded_actor.roster_hash != request.get("roster_hash")
                or loaded_actor.finisher_self_model_hash != request.get("finisher_self_model_hash")
                or loaded_actor.control_envelope_hash != request.get("control_envelope_hash")
                or loaded_actor.source_evidence_hashes != (str(payload.get("source_s202_hash")),)
                or loaded_actor.evidence_ready != actor_record.get("evidence_ready")
                or loaded_actor.distinct_context_count != actor_record.get("distinct_context_count")
                or loaded_actor.distinct_trajectory_count
                != actor_record.get("distinct_trajectory_count")
            ):
                raise ValueError("S203 actor integrity changed")
        raw_config = cast(dict[str, Any], request["config"])
        config = ContextualFinishTargetGrowthConfig(
            **{
                **raw_config,
                "policy_target_y_candidates_m": tuple(raw_config["policy_target_y_candidates_m"]),
                "foot_yaw_offset_candidates_rad": tuple(
                    raw_config["foot_yaw_offset_candidates_rad"]
                ),
            }
        )
        if request.get("config_hash") != config.config_hash or request.get(
            "control_envelope_hash"
        ) != _control_envelope_hash(config):
            raise ValueError("S203 control envelope binding changed")
        selected = payload.get("selected")
        parent = payload.get("parent")
        exact_replay = bool(
            isinstance(selected, dict)
            and selected.get("result") == replay.get("result")
            and selected.get("trajectory", {}).get("trajectory_digest")
            == replay.get("trajectory", {}).get("trajectory_digest")
        )
        if not isinstance(selected, dict) or not isinstance(parent, dict):
            raise ValueError("S203 selected or parent outcome is invalid")
        metrics, gates = _derive_metrics_and_gates(
            rows=cast(list[dict[str, Any]], rows),
            selected=selected,
            parent=parent,
            exact_replay=exact_replay,
            config=config,
        )
        candidate = payload.get("backend_candidate")
        if loaded_actor is not None:
            sample = loaded_actor.samples[0]
            if (
                not isinstance(candidate, dict)
                or sample.trajectory_hash != selected.get("trajectory", {}).get("trajectory_digest")
                or list(sample.requested_physical_target_m) != request.get("physical_target_m")
                or list(sample.executed_policy_target_m) != selected.get("policy_target_m")
                or sample.executed_foot_yaw_offset_rad != selected.get("foot_yaw_offset_rad")
                or candidate.get("backend") != RoleOptionBackend.CONTEXTUAL_FINISH_TARGET.value
                or candidate.get("option") != PhysicalSoccerOption.SHOOT.value
                or candidate.get("artifact_hash") != loaded_actor.actor_hash
                or candidate.get("evidence_hash") != payload.get("source_s202_hash")
                or candidate.get("strict_replay") is not True
                or candidate.get("holdout_passed") is not False
                or candidate.get("parent_retention_passed")
                is not gates["selected_stability_retained"]
                or candidate.get("evidence_ready") is not False
            ):
                raise ValueError("S203 backend candidate binding changed")
        gates["single_context_seed_not_deployable"] = bool(
            candidate is None
            or (isinstance(candidate, dict) and candidate.get("evidence_ready") is False)
        )
        expected_passed = all(gates.values())
        if (
            metrics != payload.get("metrics")
            or gates != payload.get("gates")
            or payload.get("passed") is not expected_passed
            or payload.get("status")
            != (
                "PASS_SINGLE_CONTEXT_FINISH_TARGET_SEED"
                if expected_passed
                else "REJECTED_CONTEXTUAL_FINISH_TARGET_GROWTH"
            )
        ):
            raise ValueError("S203 derived metrics or gates changed")
        boundary = payload.get("evidence_boundary")
        if (
            not isinstance(boundary, dict)
            or boundary.get("partition") != "DEVELOPMENT"
            or boundary.get("plastic_agent_id") != "red.finisher"
            or boundary.get("frozen_role_count") != 5
            or boundary.get("pixels_used_for_scoring") is not False
            or boundary.get("activation_ceiling") != "SIM_ONLY"
            or boundary.get("hardware_command_sent") is not False
            or payload.get("promotion_eligible") is not False
        ):
            raise ValueError("S203 authority boundary changed")
    finally:
        if claimed_hash is not None:
            payload["report_hash"] = claimed_hash
    return payload


def _simulation_kwargs(
    *,
    request: dict[str, Any],
    policy_target: tuple[float, float, float],
    foot_yaw_offset_rad: float,
    duration: float,
    config: ContextualFinishTargetGrowthConfig | None = None,
) -> dict[str, Any]:
    kwargs = three_role_development_kwargs()
    active = config
    if active is None:
        raw = request.get("config")
        if not isinstance(raw, dict):
            raise ValueError("S203 simulation request has no growth config")
        active = ContextualFinishTargetGrowthConfig(
            **{
                **raw,
                "policy_target_y_candidates_m": tuple(raw["policy_target_y_candidates_m"]),
                "foot_yaw_offset_candidates_rad": tuple(raw["foot_yaw_offset_candidates_rad"]),
            }
        )
    lane = float(request["receiver_lateral_lane_m"])
    kwargs.update(
        shooter_start_sec=float(request["receiver_phase_start_sec"]),
        shooter_origin=(0.0, lane, 0.0),
        pass_reception_target_m=tuple(request["pass_reception_target_m"]),
        passer_yaw_rad=float(request["passer_yaw_rad"]),
        shooter_policy_target=policy_target,
        shooter_parameter_overrides={
            "foot_yaw_offset": foot_yaw_offset_rad,
            "foot_pitch_offset": 0.01,
        },
        shooter_precontact_joint_guard_enabled=(active.shooter_precontact_joint_guard_enabled),
        shooter_joint_guard_config=G1JointGuardConfig(
            margin_rad=active.shooter_joint_guard_margin_rad,
            prediction_horizon_sec=(active.shooter_joint_guard_prediction_horizon_sec),
            boundary_kp=active.shooter_joint_guard_boundary_kp,
            boundary_kd=active.shooter_joint_guard_boundary_kd,
        ),
        shooter_post_policy_frame=active.shooter_post_policy_frame,
        shooter_post_policy_blend_frames=active.shooter_post_policy_blend_frames,
        simulation_duration_sec=duration,
    )
    return kwargs


def _candidate_row(
    *,
    index: int,
    policy_target: tuple[float, float, float],
    foot_yaw_offset_rad: float,
    result: G1SharedWorldResult,
    trajectory: dict[str, str],
) -> dict[str, Any]:
    return {
        "candidate_index": index,
        "policy_target_m": list(policy_target),
        "foot_yaw_offset_rad": foot_yaw_offset_rad,
        "result": result.to_dict(),
        "safe": _safe(result),
        "trajectory": trajectory,
    }


def _selection_key(row: dict[str, Any]) -> tuple[Any, ...]:
    result = cast(dict[str, Any], row["result"])
    error = result.get("target_error_m")
    finite_error = (
        float(error)
        if isinstance(error, int | float)
        and not isinstance(error, bool)
        and math.isfinite(float(error))
        else math.inf
    )
    ordered = bool(
        result.get("pass_contact_time_sec") is not None
        and result.get("shot_contact_time_sec") is not None
        and result["pass_contact_time_sec"] < result["shot_contact_time_sec"]
    )
    return (
        not row["safe"],
        not ordered,
        finite_error,
        -float(result["shooter_min_pelvis_height_m"]),
        float(result["shooter_post_contact_support_foot_slip_m"]),
        tuple(row["policy_target_m"]),
    )


def _derive_metrics_and_gates(
    *,
    rows: list[dict[str, Any]],
    selected: dict[str, Any],
    parent: dict[str, Any],
    exact_replay: bool,
    config: ContextualFinishTargetGrowthConfig,
) -> tuple[dict[str, Any], dict[str, bool]]:
    result = cast(dict[str, Any], selected["result"])
    parent_result = cast(dict[str, Any], parent["result"])
    error = result.get("target_error_m")
    precise = bool(
        selected["safe"]
        and result.get("goal_crossed") is True
        and isinstance(error, int | float)
        and not isinstance(error, bool)
        and math.isfinite(float(error))
        and float(error) <= config.maximum_target_error_m
    )
    stability_retained = bool(
        selected["safe"]
        and float(result["shooter_min_pelvis_height_m"]) + config.maximum_pelvis_regression_m
        >= float(parent_result["shooter_min_pelvis_height_m"])
        and float(result["shooter_post_contact_support_foot_slip_m"])
        <= float(parent_result["shooter_post_contact_support_foot_slip_m"])
        + config.maximum_support_slip_regression_m
    )
    metrics = {
        "candidate_count": len(rows),
        "safe_candidate_count": sum(bool(row["safe"]) for row in rows),
        "goal_candidate_count": sum(bool(row["result"]["goal_crossed"]) for row in rows),
        "precise_safe_candidate_count": sum(
            bool(
                row["safe"]
                and row["result"]["goal_crossed"]
                and isinstance(row["result"].get("target_error_m"), int | float)
                and float(row["result"]["target_error_m"]) <= config.maximum_target_error_m
            )
            for row in rows
        ),
        "selected_policy_target_m": selected["policy_target_m"],
        "selected_foot_yaw_offset_rad": selected["foot_yaw_offset_rad"],
        "selected_target_error_m": error,
        "selected_shot_speed_mps": result["shot_peak_ball_speed_mps"],
        "selected_min_pelvis_height_m": result["shooter_min_pelvis_height_m"],
        "selected_support_slip_m": result["shooter_post_contact_support_foot_slip_m"],
        "selected_joint_guard_fraction": result["shooter_joint_guard_fraction"],
        "parent_target_error_m": parent_result.get("target_error_m"),
        "parent_min_pelvis_height_m": parent_result["shooter_min_pelvis_height_m"],
        "parent_support_slip_m": parent_result["shooter_post_contact_support_foot_slip_m"],
    }
    gates = {
        "source_finisher_lease_bound": True,
        "all_candidate_trajectories_bound": True,
        "selected_precise_safe": precise,
        "selected_stability_retained": stability_retained,
        "selected_exact_replay": exact_replay,
        "frozen_kick_prior_retained": True,
        "predictive_joint_envelope_active": bool(
            float(result["shooter_joint_guard_fraction"]) > 0.0
        ),
    }
    return metrics, gates


def _safe(result: G1SharedWorldResult) -> bool:
    return bool(
        result.finite_state
        and not result.actuator_saturation
        and not result.torque_limit_violation
        and not result.joint_limit_violation
        and not result.passer_joint_limit_violation
        and not result.shooter_joint_limit_violation
        and not result.goalkeeper_joint_limit_violation
        and not result.passer_post_kick_fall
        and not result.shooter_post_kick_fall
        and result.robot_robot_contact_count == 0
        and result.passer_min_pelvis_height_m >= 0.60
        and result.shooter_min_pelvis_height_m >= 0.60
        and result.goalkeeper_min_pelvis_height_m is not None
        and result.goalkeeper_min_pelvis_height_m >= 0.60
    )


def _load_lead_pass(root: Path) -> tuple[DynamicLeadPassPolicy, dict[str, Any]]:
    evidence = json.loads((root / "evidence.json").read_text(encoding="utf-8"))
    policy_value = json.loads((root / "dynamic-lead-pass-policy.json").read_text(encoding="utf-8"))
    claimed = evidence.pop("evidence_hash", None)
    try:
        if claimed != hash_json(evidence):
            raise ValueError("lead-pass evidence integrity changed")
    finally:
        if claimed is not None:
            evidence["evidence_hash"] = claimed
    if evidence.get("promotion_status") != "FROZEN_RESEARCH_DEMO":
        raise ValueError("lead-pass source is not frozen evidence")
    policy_value["discovery_sample_hashes"] = tuple(policy_value["discovery_sample_hashes"])
    policy = DynamicLeadPassPolicy(**policy_value)
    if policy.artifact_hash != evidence.get("policy_hash"):
        raise ValueError("lead-pass policy binding changed")
    return policy, evidence


def _save_trajectory(path: Path, trajectory: dict[str, np.ndarray]) -> dict[str, str]:
    np.savez_compressed(path, **trajectory)  # type: ignore[arg-type]
    return {
        "file": path.name,
        "file_hash": hash_bytes(path.read_bytes()),
        "trajectory_digest": trajectory_digest(trajectory),
    }


def _validate_trajectory(root: Path, record: dict[str, Any]) -> None:
    path = root / str(record.get("file", ""))
    if not path.is_file() or hash_bytes(path.read_bytes()) != record.get("file_hash"):
        raise ValueError("S203 trajectory file integrity changed")
    with np.load(path, allow_pickle=False) as archive:
        trajectory = {name: archive[name] for name in archive.files}
    if trajectory_digest(trajectory) != record.get("trajectory_digest"):
        raise ValueError("S203 semantic trajectory integrity changed")


def _runtime_manifest() -> dict[str, str]:
    import mujoco

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "mujoco": mujoco.__version__,
    }


def _control_envelope_hash(config: ContextualFinishTargetGrowthConfig) -> str:
    return str(
        hash_json(
            {
                "shooter_precontact_joint_guard_enabled": (
                    config.shooter_precontact_joint_guard_enabled
                ),
                "shooter_joint_guard_margin_rad": config.shooter_joint_guard_margin_rad,
                "shooter_joint_guard_prediction_horizon_sec": (
                    config.shooter_joint_guard_prediction_horizon_sec
                ),
                "shooter_joint_guard_boundary_kp": config.shooter_joint_guard_boundary_kp,
                "shooter_joint_guard_boundary_kd": config.shooter_joint_guard_boundary_kd,
                "shooter_post_policy_frame": config.shooter_post_policy_frame,
                "shooter_post_policy_blend_frames": (config.shooter_post_policy_blend_frames),
            }
        )
    )


def _git_head(checkout: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()


def _implementation_hash() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(__file__).parents[1] / "growth" / "contextual_finish_target.py",
        Path(__file__).parents[1] / "growth" / "role_option_backend.py",
        Path(__file__).parents[1] / "skills" / "team" / "shared_world.py",
    ):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def _new_external_output(path: Path, checkout: Path) -> Path:
    output = path.expanduser().resolve()
    source = checkout.expanduser().resolve()
    if output.exists() or output == source or source in output.parents:
        raise ValueError("S203 evidence output must be new and external")
    output.mkdir(parents=True)
    return output


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--source-s202", type=Path, required=True)
    parser.add_argument("--source-lead-pass", type=Path, required=True)
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    report = run_contextual_finish_target_growth(
        asset_root=args.asset_root,
        source_s202_path=args.source_s202,
        source_lead_pass_dir=args.source_lead_pass,
        source_checkout=args.source_checkout,
        output_dir=args.output,
        workers=args.workers,
    )
    print(json.dumps({"status": report["status"], "report_hash": report["report_hash"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ContextualFinishTargetGrowthConfig",
    "run_contextual_finish_target_growth",
    "validate_contextual_finish_target_growth",
]
