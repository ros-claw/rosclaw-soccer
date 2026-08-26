from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.mjlab_getup_probe import route_recovery_entry_torch
from rosclaw_soccer.training.opentrack_recovery_mjx_failure_exam import (
    RecoveryMJXFailureStateExamConfig,
)
from rosclaw_soccer.training.recovery_capture_router import (
    calibrate_recovery_capture_router,
    confirm_recovery_capture_router_closed_loop,
)
from rosclaw_soccer.training.recovery_reachability import (
    ReachabilityMainline,
    RecoveryReachabilityBankConfig,
    build_recovery_reachability_bank,
    decide_recovery_mainline,
    write_recovery_moe_reachability_aggregate,
    write_recovery_phase_alignment_ab_decision,
)


def _write_source(root: Path, *, seed: int, count: int = 160) -> Path:
    root.mkdir()
    rng = np.random.default_rng(seed)
    qpos = rng.normal(0.0, 0.01, (count, 36)).astype(np.float32)
    qpos[:, 2] = np.linspace(0.15, 0.75, count, dtype=np.float32)
    qpos[:, 3:7] = np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32)
    qvel = rng.normal(0.0, 1.0, (count, 35)).astype(np.float32)
    steps = np.resize(np.asarray((200, 300, 399, 400, 500, 599), dtype=np.int32), count)
    archive = root / "failure-window-states.npz"
    np.savez_compressed(
        archive,
        qpos=qpos,
        qvel=qvel,
        control_step=steps,
        environment_index=np.arange(count, dtype=np.int32),
        handoff_frozen=np.zeros(count, dtype=np.bool_),
        trajectory_step=np.arange(count, dtype=np.int32) + 6230,
        trajectory_initial_step=np.full(count, 6230, dtype=np.int32),
        root_body_backward_speed_mps=np.abs(qvel[:, 0]),
        root_body_lateral_speed_mps=np.abs(qvel[:, 1]),
        pelvis_yaw_speed_rad_s=np.abs(qvel[:, 5]),
        last_motor_targets=np.zeros((count, 29), dtype=np.float32),
        last_teacher_action=np.zeros((count, 29), dtype=np.float32),
        last_residual=np.zeros((count, 29), dtype=np.float32),
        proprioception_history=np.zeros((count, 4, 96), dtype=np.float32),
        phase_repeat=np.zeros(count, dtype=np.int32),
    )
    digest = "sha256:" + "a" * 64
    manifest = {
        "schema_version": "rosclaw_soccer.recovery_mjx_failure_state_manifest.v2",
        "config": {"num_environments": count},
        "source_failure_window_plan_hash": hash_json([digest, seed]),
        "source_failure_window_plan_file_hash": hash_json([digest, seed, "file"]),
        "source_training_report_hash": digest,
        "source_actor_checkpoint_hash": digest,
        "source_actor_config_hash": digest,
        "source_route_manifest_hash": digest,
        "source_route_group_hash": digest,
        "teacher_checkpoint_hash": digest,
        "motion_archive_hash": digest,
        "snapshot_manifest_hash": digest,
        "compiled_model_contract": {"model": "test"},
        "rollout_backend": "MUJOCO_MJX",
        "physics_truth_backend": "CPU_MUJOCO_REQUIRED_FOR_PROMOTION",
        "deterministic_actor": True,
        "full_route_reset": True,
        "requested_collection_steps": sorted(set(steps.tolist())),
        "collected_state_count": count,
        "state_archive": archive.name,
        "state_archive_hash": hash_bytes(archive.read_bytes()),
        "qpos_shape": [count, 36],
        "qvel_shape": [count, 35],
        "proprioception_history_shape": [count, 4, 96],
        "context_features_collected": [
            "qpos",
            "qvel",
            "trajectory_step",
            "trajectory_initial_step",
            "handoff_frozen",
            "last_motor_targets",
            "last_teacher_action",
            "last_residual",
            "proprioception_history",
            "phase_repeat",
        ],
        "curriculum_use_only": True,
        "deployment_candidate": False,
        "promotion_eligible": False,
        "promotion_authority": "NONE",
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "hardware_command_sent": False,
    }
    manifest["report_hash"] = hash_json(manifest)
    path = root / "failure-state-manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_build_reachability_bank_selects_256_exact_states(tmp_path: Path) -> None:
    first = _write_source(tmp_path / "first", seed=1)
    second = _write_source(tmp_path / "second", seed=2)
    output = tmp_path / "external" / "bank"
    result = build_recovery_reachability_bank(
        source_manifest_paths=(first, second),
        output_dir=output,
        source_checkout_path=tmp_path / "checkout",
        config=RecoveryReachabilityBankConfig(state_count=256, random_seed=76),
    )
    assert result["collected_state_count"] == 256
    assert result["stratification_complete"] is False
    assert result["contact_topology_coverage"].startswith("MISSING")
    assert len(result["selection_rows"]) == 256
    assert {row["source_manifest_hash"] for row in result["selection_rows"]} == {
        json.loads(first.read_text())["report_hash"],
        json.loads(second.read_text())["report_hash"],
    }
    with np.load(output / "failure-window-states.npz", allow_pickle=False) as archive:
        assert archive["qpos"].shape == (256, 36)
        assert np.array_equal(archive["environment_index"], np.arange(256))


@pytest.mark.parametrize(
    ("bounded", "expanded", "oracle", "expected"),
    (
        (0.10, 0.65, None, ReachabilityMainline.EXPAND_RESIDUAL_AUTHORITY),
        (0.10, 0.15, None, ReachabilityMainline.TRAIN_PARENT_FREE_EXPERT_ORACLE),
        (0.10, 0.15, 0.75, ReachabilityMainline.TRAIN_PARENT_FREE_EXPERT_ORACLE),
        (0.10, 0.15, 0.55, ReachabilityMainline.FIX_ENVIRONMENT_REWARD_OR_ACTUATOR),
        (0.70, 0.72, None, ReachabilityMainline.RESIDUAL_REMAINS_PLAUSIBLE),
    ),
)
def test_reachability_decision_routes_learner(
    bounded: float,
    expanded: float,
    oracle: float | None,
    expected: ReachabilityMainline,
) -> None:
    assert (
        decide_recovery_mainline(
            bounded_residual_success_rate=bounded,
            expanded_residual_success_rate=expanded,
            privileged_oracle_success_rate=oracle,
        )
        is expected
    )


def test_reachability_failure_exam_gain_is_bounded() -> None:
    config = RecoveryMJXFailureStateExamConfig(candidate_adapter_gain=4.0)
    assert config.candidate_adapter_gain == 4.0
    with pytest.raises(ValueError, match="invalid"):
        RecoveryMJXFailureStateExamConfig(candidate_adapter_gain=4.01)


def test_recovery_entry_router_is_mutually_exclusive_and_preserves_athlete() -> None:
    torch = pytest.importorskip("torch")
    athlete, capture, getup = route_recovery_entry_torch(
        torch=torch,
        pelvis_height_m=torch.tensor((0.72, 0.70, 0.65)),
        upright_projection=torch.tensor((0.98, 0.98, 0.70)),
        root_angular_speed_rad_s=torch.tensor((0.80, 1.20, 1.20)),
        route_upright_to_locomotion=True,
        capture_router_enabled=True,
        capture_maximum_pelvis_height_m=0.705,
        capture_minimum_root_angular_speed_rad_s=1.10,
    )
    assert athlete.tolist() == [True, False, False]
    assert capture.tolist() == [False, True, False]
    assert getup.tolist() == [False, False, True]


def test_recovery_moe_aggregate_does_not_hide_failed_getup_route(tmp_path: Path) -> None:
    first = _write_source(tmp_path / "first", seed=1)
    second = _write_source(tmp_path / "second", seed=2)
    bank_dir = tmp_path / "external" / "bank"
    manifest = build_recovery_reachability_bank(
        source_manifest_paths=(first, second),
        output_dir=bank_dir,
        source_checkout_path=tmp_path / "checkout",
        config=RecoveryReachabilityBankConfig(state_count=256, random_seed=76),
    )
    manifest_path = bank_dir / "failure-state-manifest.json"
    shared = {
        "schema_version": "rosclaw_soccer.mjlab_recovery_moe_reachability_probe.v1",
        "contract_hash": "contract",
        "checkpoint_hash": "checkpoint",
        "source_hash": "source",
        "body_hash": "body",
        "physics_scene_hash": "physics",
        "failure_state_manifest_hash": manifest["report_hash"],
        "failure_state_manifest_file_hash": hash_bytes(manifest_path.read_bytes()),
        "source_compiled_model_contract_hash": "compiled",
        "routing": "UPRIGHT_TO_ATHLETE_OTHERWISE_MJLAB_GETUP",
        "termination_semantics": "BILATERAL_LOW_MOMENTUM_HOLD",
        "getup_reference_phase_alignment": False,
        "getup_reference_phase_estimator_weights": None,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "promotion_eligible": False,
    }
    paths = []
    for gpu in range(4):
        start = gpu * 64
        rows = []
        for index in range(start, start + 64):
            route = "ATHLETE" if index < 224 else "GET_UP"
            stable = route == "ATHLETE" and index >= 8
            rows.append(
                {
                    "failure_state_index": index,
                    "state_identity": manifest["selection_rows"][index]["state_identity"],
                    "route": route,
                    "initial_reference_frame": 0,
                    "final_stable_recovery": stable,
                    "handoff_completed": route == "ATHLETE",
                }
            )
        payload = {
            **shared,
            "physics_device": f"cuda:{gpu}",
            "random_seed": 7_600 + gpu,
            "environment_count": 64,
            "failure_state_start_index": start,
            "failure_state_stop_index": start + 64,
            "state_results": rows,
            "athlete_route_count": sum(row["route"] == "ATHLETE" for row in rows),
            "getup_route_count": sum(row["route"] == "GET_UP" for row in rows),
            "final_stable_recovery_count": sum(row["final_stable_recovery"] for row in rows),
            "handoff_completed_count": sum(row["handoff_completed"] for row in rows),
        }
        payload["report_hash"] = hash_json(payload)
        path = tmp_path / f"gpu{gpu}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(path)
    output = tmp_path / "aggregate.json"
    result = write_recovery_moe_reachability_aggregate(
        shard_report_paths=paths,
        failure_state_manifest_path=manifest_path,
        output_path=output,
    )
    assert result["overall_final_stable_recovery_rate"] == pytest.approx(216 / 256)
    assert result["route_metrics"]["GET_UP"]["final_stable_recovery_rate"] == 0.0
    assert result["weakest_populated_route_recovery_rate"] == 0.0
    assert (
        result["architecture_decision"]
        == "KEEP_ATHLETE_FOUNDATION_TRAIN_PREDECESSOR_CONDITIONED_GETUP"
    )
    aligned_paths = []
    for gpu, path in enumerate(paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["getup_reference_phase_alignment"] = True
        payload["getup_reference_phase_estimator_weights"] = {"joint_position_mse": 1.0}
        for row in payload["state_results"]:
            if row["route"] == "GET_UP":
                row["initial_reference_frame"] = 436
                row["final_stable_recovery"] = True
                row["handoff_completed"] = True
        payload["final_stable_recovery_count"] = sum(
            row["final_stable_recovery"] for row in payload["state_results"]
        )
        payload["handoff_completed_count"] = sum(
            row["handoff_completed"] for row in payload["state_results"]
        )
        payload.pop("report_hash")
        payload["report_hash"] = hash_json(payload)
        aligned_path = tmp_path / f"aligned-gpu{gpu}.json"
        aligned_path.write_text(json.dumps(payload), encoding="utf-8")
        aligned_paths.append(aligned_path)
    aligned_output = tmp_path / "aligned-aggregate.json"
    aligned = write_recovery_moe_reachability_aggregate(
        shard_report_paths=aligned_paths,
        failure_state_manifest_path=manifest_path,
        output_path=aligned_output,
    )
    assert aligned["route_metrics"]["GET_UP"]["final_stable_recovery_rate"] == 1.0
    decision = write_recovery_phase_alignment_ab_decision(
        baseline_aggregate_path=output,
        aligned_aggregate_path=aligned_output,
        output_path=tmp_path / "ab.json",
    )
    assert decision["decision"] == "PHASE_ENTRY_ADAPTER_BREAKTHROUGH_CONFIRMED"
    assert decision["state_improvement_count"] == 32
    assert decision["state_regression_count"] == 0

    oracle_paths = []
    for gpu, path in enumerate(aligned_paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["routing"] = "MJLAB_GETUP_ONLY"
        for row in payload["state_results"]:
            row["route"] = "GET_UP"
            row["initial_reference_frame"] = 436
            row["final_stable_recovery"] = True
            row["handoff_completed"] = True
        payload["athlete_route_count"] = 0
        payload["getup_route_count"] = 64
        payload["final_stable_recovery_count"] = 64
        payload["handoff_completed_count"] = 64
        payload.pop("report_hash")
        payload["report_hash"] = hash_json(payload)
        oracle_path = tmp_path / f"oracle-gpu{gpu}.json"
        oracle_path.write_text(json.dumps(payload), encoding="utf-8")
        oracle_paths.append(oracle_path)
    oracle_output = tmp_path / "oracle-aggregate.json"
    oracle = write_recovery_moe_reachability_aggregate(
        shard_report_paths=oracle_paths,
        failure_state_manifest_path=manifest_path,
        output_path=oracle_output,
    )
    assert oracle["overall_final_stable_recovery_rate"] == 1.0

    contact_rows = []
    for index, selection in enumerate(manifest["selection_rows"]):
        failure = index < 8
        contact_rows.append(
            {
                "state_index": index,
                "state_identity": selection["state_identity"],
                "pelvis_height_m": 0.70 if failure else 0.72,
                "root_angular_speed_rad_s": 1.2 if failure else 0.8,
                "ground_contact_classes": ["FOOT"],
                "grounded_foot_sides": ["LEFT"],
                "support_aabb": {"com_aabb_signed_margin_m": -0.01},
                "contacts": [],
            }
        )
    contact_report = {
        "schema_version": "rosclaw_soccer.recovery_contact_enrichment.v1",
        "failure_state_manifest_hash": manifest["report_hash"],
        "state_count": 256,
        "state_rows": contact_rows,
        "contact_enrichment_complete": True,
        "contact_truth_backend": "CPU_MUJOCO_FORWARD",
        "deployment_candidate": False,
        "promotion_eligible": False,
        "promotion_authority": "NONE",
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "hardware_command_sent": False,
    }
    contact_report["report_hash"] = hash_json(contact_report)
    contact_path = tmp_path / "contacts.json"
    contact_path.write_text(json.dumps(contact_report), encoding="utf-8")
    calibration = calibrate_recovery_capture_router(
        baseline_aggregate_path=aligned_output,
        all_capture_oracle_aggregate_path=oracle_output,
        contact_enrichment_path=contact_path,
        output_path=tmp_path / "calibration.json",
    )
    assert calibration["selected_router"]["capture_state_count"] == 8
    assert calibration["counterfactual_success_rate"] == 1.0
    expected_routes = {
        row["state_index"]: row["route"] for row in calibration["composed_state_rows"]
    }
    mixed_paths = []
    for gpu, path in enumerate(oracle_paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["routing"] = "RISK_ROUTED_ATHLETE_CAPTURE_GETUP"
        payload["capture_router_config"] = {
            "maximum_pelvis_height_m": calibration["selected_router"]["maximum_pelvis_height_m"],
            "minimum_root_angular_speed_rad_s": calibration["selected_router"][
                "minimum_root_angular_speed_rad_s"
            ],
            "feature_contract": "ENTRY_PROPRIOCEPTION_ONLY",
        }
        for row in payload["state_results"]:
            row["route"] = expected_routes[row["failure_state_index"]]
        payload["athlete_route_count"] = sum(
            row["route"] == "ATHLETE" for row in payload["state_results"]
        )
        payload["capture_route_count"] = sum(
            row["route"] == "CAPTURE" for row in payload["state_results"]
        )
        payload["getup_route_count"] = sum(
            row["route"] == "GET_UP" for row in payload["state_results"]
        )
        payload.pop("report_hash")
        payload["report_hash"] = hash_json(payload)
        mixed_path = tmp_path / f"mixed-gpu{gpu}.json"
        mixed_path.write_text(json.dumps(payload), encoding="utf-8")
        mixed_paths.append(mixed_path)
    mixed_output = tmp_path / "mixed-aggregate.json"
    mixed = write_recovery_moe_reachability_aggregate(
        shard_report_paths=mixed_paths,
        failure_state_manifest_path=manifest_path,
        output_path=mixed_output,
    )
    assert mixed["route_metrics"]["CAPTURE"]["final_stable_recovery_rate"] == 1.0
    confirmation = confirm_recovery_capture_router_closed_loop(
        calibration_report_path=tmp_path / "calibration.json",
        mixed_aggregate_path=mixed_output,
        output_path=tmp_path / "confirmation.json",
    )
    assert confirmation["decision"] == "CAPTURE_ROUTER_CLOSED_LOOP_CONFIRMED"
