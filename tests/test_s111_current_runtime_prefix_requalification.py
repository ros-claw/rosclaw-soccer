from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from rosclaw.feedback import contracts as growth_core_contracts

from rosclaw_soccer.media.current_runtime_requalification_video import (
    _implementation_hash as video_implementation_hash,
)
from rosclaw_soccer.media.current_runtime_requalification_video import (
    validate_current_runtime_requalification_video,
)
from rosclaw_soccer.providers.g1.asset_qualification import trajectory_digest
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.current_runtime_prefix_requalification import (
    CurrentRuntimePrefixRequalificationConfig,
    _implementation_hash,
    _process_contract,
    _python_tree_hash,
    _runtime_dependency_contract,
    current_runtime_prefix_lane,
    prefix_gate_contract,
    validate_current_runtime_prefix_requalification,
)


def test_current_runtime_contract_preserves_frozen_right_control() -> None:
    config = CurrentRuntimePrefixRequalificationConfig()
    lane = current_runtime_prefix_lane(config.lane_id)
    gates = prefix_gate_contract(lane)

    assert config.case_id == "right-control"
    assert config.ball_mass_kg == 0.41
    assert config.ball_ground_friction == 0.10
    assert config.cross_process_replays == 3
    assert gates["takeoff"]["minimum_airborne_duration_sec"] == 0.15
    assert gates["takeoff"]["maximum_landing_angular_speed_rad_s"] == 3.5
    assert config.config_hash.startswith("sha256:")


def test_config_requires_multiple_fresh_processes_and_sim_only() -> None:
    with pytest.raises(ValueError, match="contract is invalid"):
        CurrentRuntimePrefixRequalificationConfig(cross_process_replays=1)
    with pytest.raises(ValueError, match="contract is invalid"):
        CurrentRuntimePrefixRequalificationConfig(ball_mass_kg=0.42)
    with pytest.raises(ValueError, match="contract is invalid"):
        CurrentRuntimePrefixRequalificationConfig(hardware_authorized=True)


def _worker(directory: Path, index: int) -> dict[str, object]:
    trajectory = directory / f"process-replay-{index}.npz"
    trajectory_value = {"time": np.asarray([0.0, 0.01], dtype=np.float64)}
    np.savez_compressed(trajectory, **trajectory_value)
    payload: dict[str, object] = {
        "schema_version": "rosclaw_soccer.current_runtime_process_replay.v1",
        "process_id": 1000 + index,
        "passed": True,
        "evaluation": {"passed": True, "first_takeoff_exam": {"passed": True}},
        "trajectory_file": trajectory.name,
        "trajectory_hash": hash_bytes(trajectory.read_bytes()),
        "trajectory_digest": trajectory_digest(trajectory_value),
        "process_contract": _process_contract(),
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    payload["report_hash"] = hash_json(payload)
    report = directory / f"process-replay-{index}.json"
    report.write_text(json.dumps(payload), encoding="utf-8")
    return {
        "report_file": report.name,
        "report_hash": payload["report_hash"],
        "trajectory_file": trajectory.name,
        "trajectory_hash": payload["trajectory_hash"],
        "trajectory_digest": payload["trajectory_digest"],
        "process_id": payload["process_id"],
    }


def _passing_evidence(directory: Path) -> Path:
    request = directory / "request.json"
    predecessor = directory / "predecessor.json"
    from rosclaw_soccer.training import current_runtime_prefix_requalification as module

    request.write_text(
        json.dumps(
            {
                "soccer_source_tree_hash": _python_tree_hash(
                    Path(module.__file__).resolve().parents[1]
                ),
                "growth_core_source_tree_hash": _python_tree_hash(
                    Path(inspect.getfile(growth_core_contracts)).resolve().parents[2]
                ),
                "runtime_dependencies": _runtime_dependency_contract(),
                "process_contract": _process_contract(),
                "config": {"cross_process_replays": 2},
            }
        ),
        encoding="utf-8",
    )
    predecessor.write_bytes(b"predecessor")
    workers = [_worker(directory, index) for index in range(2)]
    payload = {
        "schema_version": "rosclaw_soccer.current_runtime_requalification_evidence.v2",
        "claim": "TRUE_AIRBORNE_SAVE_WITH_FOOT_CONTACT_GROUNDED_LANDING",
        "requalification_claim": "CROSS_PROCESS_CONTINUOUS_FOUR_G1_CURRENT_RUNTIME_CHAMPION",
        "passed": True,
        "promotion_status": "FROZEN_RESEARCH_DEMO",
        "requalification_status": "PROMOTED_SIM_ONLY_CURRENT_RUNTIME_FULL_CHAIN",
        "qualification_gates": {
            "predecessor_rejection_bound": True,
            "fresh_process_count": True,
            "process_contract_identical": True,
            "cross_process_exact_replay": True,
            "all_prefix_physics_gates": True,
            "all_continuous_team_gates": True,
            "all_workers_sim_only_safe": True,
            "full_source_trees_bound": True,
            "current_runtime_bound": True,
            "sim_only_authority": True,
        },
        "strict_replay": True,
        "cross_process_replay_count": 2,
        "process_ids": [1000, 1001],
        "physics_authority": "CPU_MUJOCO",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "commercial_use_allowed": False,
        "reset_or_teleport_used": False,
        "ball_cannon_used": False,
        "pixels_used_for_scoring": False,
        "request_hash": hash_bytes(request.read_bytes()),
        "predecessor": {
            "path": str(predecessor),
            "evidence_hash": hash_bytes(predecessor.read_bytes()),
            "report_hash": "sha256:predecessor",
            "passed": False,
            "right_control_passed": False,
        },
        "worker_reports": workers,
        "trajectory_file": workers[0]["trajectory_file"],
        "trajectory_hash": workers[0]["trajectory_hash"],
        "implementation_hash": _implementation_hash(),
        "continuous": {"passed": True, "first_takeoff_exam": {"passed": True}},
        "first": {"passed": True},
        "replay": {"passed": True},
    }
    payload["report_hash"] = hash_json(payload)
    evidence = directory / "evidence.json"
    evidence.write_text(json.dumps(payload), encoding="utf-8")
    return evidence


def test_evidence_binds_fresh_process_reports_and_predecessor(tmp_path: Path) -> None:
    evidence = _passing_evidence(tmp_path)

    report = validate_current_runtime_prefix_requalification(evidence)

    assert report["passed"] is True
    assert report["cross_process_replay_count"] == 2
    assert report["requalification_status"] == "PROMOTED_SIM_ONLY_CURRENT_RUNTIME_FULL_CHAIN"


def test_evidence_rejects_worker_trajectory_mutation(tmp_path: Path) -> None:
    evidence = _passing_evidence(tmp_path)
    np.savez_compressed(
        tmp_path / "process-replay-1.npz",
        time=np.asarray([0.0, 0.02], dtype=np.float64),
    )

    with pytest.raises(ValueError, match="worker report integrity changed"):
        validate_current_runtime_prefix_requalification(evidence)


def test_evidence_rejects_rehashed_duplicate_process_identity(tmp_path: Path) -> None:
    evidence = _passing_evidence(tmp_path)
    worker_path = tmp_path / "process-replay-1.json"
    worker = json.loads(worker_path.read_text(encoding="utf-8"))
    worker["process_id"] = 1000
    worker["report_hash"] = hash_json(
        {key: value for key, value in worker.items() if key != "report_hash"}
    )
    worker_path.write_text(json.dumps(worker), encoding="utf-8")

    payload = json.loads(evidence.read_text(encoding="utf-8"))
    payload["worker_reports"][1]["process_id"] = 1000
    payload["worker_reports"][1]["report_hash"] = worker["report_hash"]
    payload["process_ids"] = [1000]
    payload["report_hash"] = hash_json(
        {key: value for key, value in payload.items() if key != "report_hash"}
    )
    evidence.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="authority or integrity"):
        validate_current_runtime_prefix_requalification(evidence)


def test_evidence_rejects_authority_mutation_even_when_rehashed(tmp_path: Path) -> None:
    evidence = _passing_evidence(tmp_path)
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    payload["hardware_command_sent"] = True
    payload["report_hash"] = hash_json(
        {key: value for key, value in payload.items() if key != "report_hash"}
    )
    evidence.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="authority or integrity"):
        validate_current_runtime_prefix_requalification(evidence)


def test_video_manifest_is_source_bound_and_never_promotion_truth(tmp_path: Path) -> None:
    video = tmp_path / "s111.mp4"
    source = tmp_path / "evidence.json"
    video.write_bytes(b"video")
    source.write_bytes(b"evidence")
    payload = {
        "schema_version": "rosclaw_soccer.current_runtime_requalification_video.v1",
        "video_path": str(video),
        "video_hash": hash_bytes(video.read_bytes()),
        "source_files": {str(source): hash_bytes(source.read_bytes())},
        "claim": "CROSS_PROCESS_CONTINUOUS_FOUR_G1_CURRENT_RUNTIME_CHAMPION_VIDEO",
        "evidence_report_hash": "sha256:evidence",
        "evidence_passed": True,
        "strict_cross_process_replay": True,
        "cross_process_replay_count": 3,
        "trajectory_digest": "sha256:trajectory",
        "four_g1_visible": True,
        "two_physical_balls_visible": True,
        "two_physical_saves": True,
        "first_glove_contact_height_m": 1.417,
        "second_glove_contact_height_m": 1.441,
        "second_striker_contact_force_peak_n": 791.0,
        "fps": 30,
        "width": 1920,
        "height": 1080,
        "frame_count": 1200,
        "duration_sec": 40.0,
        "clips": [],
        "visualization_only": True,
        "pixels_used_for_scoring": False,
        "promotion_eligible": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "commercial_use_allowed": False,
        "implementation_hash": video_implementation_hash(),
    }
    payload["manifest_hash"] = hash_json(payload)
    manifest = tmp_path / "s111.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    assert validate_current_runtime_requalification_video(manifest)["evidence_passed"] is True

    payload["promotion_eligible"] = True
    payload["manifest_hash"] = hash_json(
        {key: value for key, value in payload.items() if key != "manifest_hash"}
    )
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="authority or integrity"):
        validate_current_runtime_requalification_video(manifest)
