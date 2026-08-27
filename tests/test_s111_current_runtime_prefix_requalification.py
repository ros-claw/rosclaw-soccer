from __future__ import annotations

import inspect
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from rosclaw.feedback import contracts as growth_core_contracts
from rosclaw.simforge.reproducibility import evaluate_cross_process_replays

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
    _closure_source_manifest,
    _git_head,
    _implementation_hash,
    _process_contract,
    _python_tree_hash,
    _runtime_dependency_contract,
    build_current_runtime_reproducibility_closure,
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
        "schema_version": "rosclaw_soccer.current_runtime_requalification_video.v2",
        "video_path": str(video),
        "video_hash": hash_bytes(video.read_bytes()),
        "source_files": {str(source): hash_bytes(source.read_bytes())},
        "claim": "CROSS_PROCESS_CONTINUOUS_FOUR_G1_CORE_CLOSURE_CHAMPION_VIDEO",
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


def _core_closure_evidence(directory: Path) -> tuple[Path, Path]:
    dive_source = directory / "dive-source"
    dive_source.mkdir()
    (dive_source / "controller.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(("git", "init", "-q"), cwd=dive_source, check=True)
    subprocess.run(
        ("git", "config", "user.email", "test@example.invalid"), cwd=dive_source, check=True
    )
    subprocess.run(("git", "config", "user.name", "ROSClaw Test"), cwd=dive_source, check=True)
    subprocess.run(("git", "add", "controller.py"), cwd=dive_source, check=True)
    subprocess.run(("git", "commit", "-qm", "fixture"), cwd=dive_source, check=True)

    asset_root = directory / "g1-assets"
    g1_paths = (
        "policy/robonaldo/model/policy-obs-aic.onnx",
        "policy/robonaldo/model/freekick_motion.npz",
        "g1_description/scene_with_ball.xml",
        "g1_description/g1_liao.xml",
        "policy/robonaldo/FreeKick.py",
    )
    for relative in g1_paths:
        path = asset_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
    assets: dict[str, Path] = {"dive_source": dive_source}
    for key in (
        "striker_actor",
        "goalkeeper_actor",
        "gmt_model",
        "gmt_skill",
        "dive_athlete_checkpoint",
        "dive_athlete_exam",
        "recovery_athlete_checkpoint",
        "recovery_athlete_exam",
    ):
        path = directory / "artifacts" / key
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(key.encode())
        assets[key] = path
    predecessor = directory / "predecessor.json"
    predecessor.write_bytes(b"predecessor")
    closure = build_current_runtime_reproducibility_closure(
        asset_root=asset_root,
        assets=assets,
        predecessor_evidence_path=predecessor,
        expected_replays=2,
    )
    config = CurrentRuntimePrefixRequalificationConfig(cross_process_replays=2)
    request_payload = {
        "schema_version": "rosclaw_soccer.current_runtime_requalification_request.v3",
        "config": asdict(config),
        "config_hash": config.config_hash,
        "reproducibility_closure": closure.to_dict(),
        "reproducibility_closure_hash": closure.closure_hash,
        "closure_sources": _closure_source_manifest(
            asset_root=asset_root,
            assets=assets,
            predecessor_evidence_path=predecessor,
        ),
        "dive_source_commit": _git_head(dive_source),
        "launcher_process_id": 1999,
        "predecessor_evidence_hash": hash_bytes(predecessor.read_bytes()),
    }
    request = directory / "request.json"
    request.write_text(json.dumps(request_payload), encoding="utf-8")
    trajectory_value = {"time": np.asarray([0.0, 0.01], dtype=np.float64)}
    first_trajectory = directory / "process-replay-0.npz"
    np.savez_compressed(first_trajectory, **trajectory_value)
    second_trajectory = directory / "process-replay-1.npz"
    second_trajectory.write_bytes(first_trajectory.read_bytes())
    evaluation = {"passed": True, "first_takeoff_exam": {"passed": True}}
    validated_workers: list[dict[str, object]] = []
    bindings: list[dict[str, object]] = []
    for index, trajectory in enumerate((first_trajectory, second_trajectory)):
        worker: dict[str, object] = {
            "schema_version": "rosclaw_soccer.current_runtime_process_replay.v2",
            "process_id": 2000 + index,
            "passed": True,
            "evaluation": evaluation,
            "trajectory_file": trajectory.name,
            "trajectory_hash": hash_bytes(trajectory.read_bytes()),
            "trajectory_digest": trajectory_digest(trajectory_value),
            "process_contract": closure.process_contract.to_dict(),
            "closure_hash": closure.closure_hash,
            "activation_ceiling": "SIM_ONLY",
            "hardware_authorized": False,
            "hardware_command_sent": False,
        }
        worker["report_hash"] = hash_json(worker)
        worker_path = directory / f"process-replay-{index}.json"
        worker_path.write_text(json.dumps(worker), encoding="utf-8")
        validated_workers.append(worker)
        bindings.append(
            {
                "report_file": worker_path.name,
                "report_hash": worker["report_hash"],
                "trajectory_file": trajectory.name,
                "trajectory_hash": worker["trajectory_hash"],
                "trajectory_digest": worker["trajectory_digest"],
                "process_id": worker["process_id"],
                "closure_hash": worker["closure_hash"],
            }
        )
    verdict = evaluate_cross_process_replays(
        closure,
        validated_workers,
        exact_fields=("evaluation", "trajectory_digest", "trajectory_hash"),
    )
    gates = {
        "predecessor_rejection_bound": True,
        **{f"core_{name}": value for name, value in verdict.gates},
        "all_prefix_physics_gates": True,
        "all_continuous_team_gates": True,
        "full_source_trees_bound": True,
        "current_runtime_bound": True,
        "sim_only_authority": True,
    }
    payload: dict[str, object] = {
        "schema_version": "rosclaw_soccer.current_runtime_requalification_evidence.v3",
        "claim": "TRUE_AIRBORNE_SAVE_WITH_FOOT_CONTACT_GROUNDED_LANDING",
        "requalification_claim": "CROSS_PROCESS_CONTINUOUS_FOUR_G1_CURRENT_RUNTIME_CHAMPION",
        "passed": True,
        "promotion_status": "FROZEN_RESEARCH_DEMO",
        "requalification_status": "PROMOTED_SIM_ONLY_CURRENT_RUNTIME_FULL_CHAIN",
        "qualification_gates": gates,
        "strict_replay": True,
        "cross_process_replay_count": 2,
        "launcher_process_id": 1999,
        "process_ids": [2000, 2001],
        "physics_authority": "CPU_MUJOCO",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "commercial_use_allowed": False,
        "reset_or_teleport_used": False,
        "ball_cannon_used": False,
        "pixels_used_for_scoring": False,
        "request_hash": hash_bytes(request.read_bytes()),
        "reproducibility_closure_hash": closure.closure_hash,
        "reproducibility_verdict": verdict.to_dict(),
        "reproducibility_verdict_hash": verdict.verdict_hash,
        "predecessor": {
            "path": str(predecessor),
            "evidence_hash": hash_bytes(predecessor.read_bytes()),
            "report_hash": "sha256:predecessor",
            "passed": False,
            "right_control_passed": False,
        },
        "worker_reports": bindings,
        "trajectory_file": first_trajectory.name,
        "trajectory_hash": hash_bytes(first_trajectory.read_bytes()),
        "implementation_hash": _implementation_hash(),
        "continuous": evaluation,
        "first": evaluation["first_takeoff_exam"],
        "replay": evaluation["first_takeoff_exam"],
    }
    payload["report_hash"] = hash_json(payload)
    evidence = directory / "evidence.json"
    evidence.write_text(json.dumps(payload), encoding="utf-8")
    return evidence, assets["gmt_model"]


def test_core_closure_evidence_recomputes_generic_cross_process_gates(tmp_path: Path) -> None:
    evidence, _ = _core_closure_evidence(tmp_path)

    report = validate_current_runtime_prefix_requalification(evidence)

    assert report["passed"] is True
    assert report["qualification_gates"]["core_closure_bound"] is True
    assert report["qualification_gates"]["core_cross_process_exact_replay"] is True


def test_core_closure_evidence_rejects_artifact_drift(tmp_path: Path) -> None:
    evidence, artifact = _core_closure_evidence(tmp_path)
    artifact.write_bytes(b"changed")

    with pytest.raises(ValueError, match="source, artifact or process closure changed"):
        validate_current_runtime_prefix_requalification(evidence)
