"""Fail-closed CPU MuJoCo exam for the S102 neural dive-command expert.

This exam deliberately proves only bounded bilateral command replay.  It does
not infer a save from video, does not place a ball in the scene, and cannot
promote the expert into a match policy.  Ball interception, controlled fall,
landing and successor routing remain separate downstream gates.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.dive_athlete_expert import (
    DiveAthleteExpertConfig,
    build_physics_margin_dive_teacher,
    decode_dive_athlete_target,
    dive_athlete_features_numpy,
    load_dive_athlete_expert,
    load_qualified_dive_clips,
)
from rosclaw_soccer.training.goalkeeper_dive_option import (
    build_balanced_dive_imitation_seed,
    load_official_goalkeeper_dive_atlas,
    mirror_g1_joint_positions,
    qualify_balanced_dive_seed_cpu_mujoco,
)

_EXAM_CLIP_IDS = ("s92-left-inner", "s92-left-outer")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _verified_training_report(path: Path, checkpoint_path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("dive athlete training report is invalid")
    report_hash = payload.get("report_hash")
    unhashed = dict(payload)
    unhashed.pop("report_hash", None)
    if (
        not isinstance(report_hash, str)
        or report_hash != hash_json(unhashed)
        or payload.get("fit_gate_passed") is not True
        or payload.get("checkpoint_hash") != hash_bytes(checkpoint_path.read_bytes())
        or payload.get("activation_ceiling") != "SIM_ONLY"
        or payload.get("hardware_command_sent") is not False
        or payload.get("commercial_use_allowed") is not False
    ):
        raise ValueError("dive athlete training evidence failed closed")
    return cast(dict[str, Any], payload)


def validate_dive_athlete_cpu_exam_report(path: Path) -> dict[str, Any]:
    """Validate integrity, authority and all declared physics case gates."""

    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("dive athlete CPU exam must be an object")
    report_hash = payload.get("report_hash")
    unhashed = dict(payload)
    unhashed.pop("report_hash", None)
    cases = payload.get("case_reports")
    if (
        not isinstance(report_hash, str)
        or report_hash != hash_json(unhashed)
        or payload.get("schema_version") != "rosclaw_soccer.dive_athlete_cpu_exam.v1"
        or payload.get("physics_backend") != "mujoco_cpu"
        or payload.get("passed") is not True
        or payload.get("fit_gate_passed") is not True
        or payload.get("cpu_mujoco_gate_passed") is not True
        or not isinstance(payload.get("teacher_cpu_mujoco_gate_passed"), bool)
        or payload.get("candidate_repaired_rejected_teacher")
        is not (payload.get("teacher_cpu_mujoco_gate_passed") is False)
        or not isinstance(cases, dict)
        or set(cases) != set(_EXAM_CLIP_IDS)
        or any(
            not isinstance(case, dict)
            or case.get("physics_passed") is not True
            or not str(case.get("state_trajectory_hash", "")).startswith("sha256:")
            for case in cases.values()
        )
        or payload.get("ball_contact_exam_completed") is not False
        or payload.get("landing_recovery_exam_completed") is not False
        or payload.get("policy_integration_completed") is not False
        or payload.get("pixels_used_for_scoring") is not False
        or payload.get("activation_ceiling") != "SIM_ONLY"
        or payload.get("hardware_command_sent") is not False
        or payload.get("commercial_use_allowed") is not False
    ):
        raise ValueError("dive athlete CPU exam failed closed")
    return cast(dict[str, Any], payload)


def _decode_case(
    *,
    torch: Any,
    model: Any,
    checkpoint: dict[str, Any],
    phase: NDArray[np.float64],
    lateral_m: float,
    height_m: float,
    duration_sec: float,
    contact_phase: float,
) -> NDArray[np.float64]:
    features = dive_athlete_features_numpy(
        phase=phase,
        target_lateral_m=np.full_like(phase, lateral_m),
        target_height_m=np.full_like(phase, height_m),
        duration_sec=np.full_like(phase, duration_sec),
        contact_phase=np.full_like(phase, contact_phase),
    )
    feature_tensor = torch.as_tensor(features, dtype=torch.float32)
    decoded: list[NDArray[np.float64]] = []
    with torch.inference_mode():
        for direction in (-1.0, 1.0):
            direction_tensor = torch.full((phase.size,), direction, dtype=torch.float32)
            value = decode_dive_athlete_target(
                torch=torch,
                model=model,
                checkpoint=checkpoint,
                features=feature_tensor,
                direction=direction_tensor,
            )
            decoded.append(np.asarray(value.cpu(), dtype=np.float64))
    return np.asarray(decoded, dtype=np.float64)


def run_dive_athlete_cpu_exam(
    *,
    checkpoint_path: Path,
    training_report_path: Path,
    evidence_root: Path,
    asset_root: Path,
    dive_source_checkout: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Decode two target contexts and require both sides to pass real physics."""

    import torch

    checkpoint_file = checkpoint_path.expanduser().resolve()
    training_file = training_report_path.expanduser().resolve()
    destination = output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    training = _verified_training_report(training_file, checkpoint_file)
    model, checkpoint = load_dive_athlete_expert(
        checkpoint_path=checkpoint_file,
        asset_root=asset_root,
        dive_source_checkout=dive_source_checkout,
        device=torch.device("cpu"),
    )
    config_payload = checkpoint["training_config"]
    if not isinstance(config_payload, dict):
        raise ValueError("dive athlete checkpoint config is invalid")
    config = DiveAthleteExpertConfig(**config_payload)
    atlas = load_official_goalkeeper_dive_atlas(checkout=dive_source_checkout)
    seed = build_balanced_dive_imitation_seed(atlas)
    teacher = build_physics_margin_dive_teacher(
        seed,
        inward_ankle_roll_rad=config.inward_ankle_roll_teacher_rad,
    )
    if (
        checkpoint["dive_seed_hash"] != seed.seed_hash
        or checkpoint["source_atlas_hash"] != atlas.atlas_hash
        or checkpoint["teacher_trajectory_hash"] != hash_bytes(teacher.tobytes())
    ):
        raise ValueError("dive athlete exam provenance changed")
    phase = np.linspace(0.0, 1.0, teacher.shape[1], dtype=np.float64)
    clips_by_id = {clip.clip_id: clip for clip in load_qualified_dive_clips(evidence_root)}
    if not set(_EXAM_CLIP_IDS) <= set(clips_by_id):
        raise ValueError("dive athlete CPU exam contexts are missing")

    teacher_exam = qualify_balanced_dive_seed_cpu_mujoco(
        asset_root=asset_root,
        source_checkout=dive_source_checkout,
        output_path=destination / "projected-teacher-cpu-exam.json",
        joint_position_rad=teacher,
        trajectory_kind="s102_physics_margin_projected_teacher",
    )
    case_reports: dict[str, Any] = {}
    trajectories: list[NDArray[np.float64]] = []
    reconstruction_errors: list[NDArray[np.float64]] = []
    symmetry_errors: list[NDArray[np.float64]] = []
    for clip_id in _EXAM_CLIP_IDS:
        clip = clips_by_id[clip_id]
        trajectory = _decode_case(
            torch=torch,
            model=model,
            checkpoint=checkpoint,
            phase=phase,
            lateral_m=clip.target_lateral_m,
            height_m=clip.target_height_m,
            duration_sec=clip.duration_sec,
            contact_phase=clip.contact_phase,
        )
        reconstruction_error = trajectory - teacher
        symmetry_error = trajectory[1] - mirror_g1_joint_positions(trajectory[0])
        physics = qualify_balanced_dive_seed_cpu_mujoco(
            asset_root=asset_root,
            source_checkout=dive_source_checkout,
            output_path=destination / f"{clip_id}-cpu-exam.json",
            joint_position_rad=trajectory,
            trajectory_kind=f"s102_dive_athlete_{clip_id}",
            state_trajectory_output_path=destination / f"{clip_id}-state.npz",
        )
        trajectories.append(trajectory)
        reconstruction_errors.append(reconstruction_error)
        symmetry_errors.append(symmetry_error)
        case_reports[clip_id] = {
            "target_lateral_m": clip.target_lateral_m,
            "target_height_m": clip.target_height_m,
            "duration_sec": clip.duration_sec,
            "contact_phase": clip.contact_phase,
            "source_evidence_hash": clip.evidence_hash,
            "source_trajectory_hash": clip.trajectory_hash,
            "decoded_trajectory_hash": hash_bytes(trajectory.tobytes()),
            "physics_report_hash": physics["report_hash"],
            "state_trajectory_hash": physics["state_trajectory_hash"],
            "physics_passed": physics["passed"],
            "outcomes": physics["outcomes"],
        }
    trajectory_array = np.asarray(trajectories, dtype=np.float64)
    error_array = np.asarray(reconstruction_errors, dtype=np.float64)
    symmetry_array = np.asarray(symmetry_errors, dtype=np.float64)
    archive_path = destination / "decoded-command-trajectories.npz"
    temporary_archive = destination / ".decoded-command-trajectories.npz.tmp"
    with temporary_archive.open("wb") as stream:
        np.savez_compressed(
            stream,
            case_id=np.asarray(_EXAM_CLIP_IDS),
            direction=np.asarray((-1, 1), dtype=np.int64),
            phase=phase,
            joint_position_rad=trajectory_array,
            teacher_joint_position_rad=teacher,
        )
    os.replace(temporary_archive, archive_path)
    metrics = {
        "source_reconstruction_rmse_rad": float(np.sqrt(np.mean(np.square(error_array)))),
        "maximum_source_reconstruction_error_rad": float(np.max(np.abs(error_array))),
        "bilateral_symmetry_error_rad": float(np.max(np.abs(symmetry_array))),
        "minimum_candidate_lateral_displacement_m": min(
            abs(float(outcome["final_lateral_displacement_m"]))
            for case in case_reports.values()
            for outcome in case["outcomes"]
        ),
        "minimum_candidate_pelvis_height_m": min(
            float(outcome["minimum_pelvis_height_m"])
            for case in case_reports.values()
            for outcome in case["outcomes"]
        ),
        "maximum_candidate_root_angular_speed_rad_s": max(
            float(outcome["maximum_root_angular_speed_rad_s"])
            for case in case_reports.values()
            for outcome in case["outcomes"]
        ),
    }
    fit_passed = bool(
        metrics["source_reconstruction_rmse_rad"] <= config.maximum_source_reconstruction_rmse_rad
        and metrics["maximum_source_reconstruction_error_rad"] <= config.maximum_training_error_rad
        and metrics["bilateral_symmetry_error_rad"] <= 1.0e-6
    )
    physics_passed = all(bool(case["physics_passed"]) for case in case_reports.values())
    teacher_repaired = bool(not teacher_exam["passed"] and physics_passed)
    passed = bool(fit_passed and physics_passed)
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.dive_athlete_cpu_exam.v1",
        "physics_backend": "mujoco_cpu",
        "checkpoint_hash": hash_bytes(checkpoint_file.read_bytes()),
        "training_report_hash": training["report_hash"],
        "source_atlas_hash": atlas.atlas_hash,
        "dive_seed_hash": seed.seed_hash,
        "teacher_trajectory_hash": hash_bytes(teacher.tobytes()),
        "teacher_physics_report_hash": teacher_exam["report_hash"],
        "teacher_cpu_mujoco_gate_passed": teacher_exam["passed"],
        "candidate_repaired_rejected_teacher": teacher_repaired,
        "trajectory_archive_hash": hash_bytes(archive_path.read_bytes()),
        "case_reports": case_reports,
        "metrics": metrics,
        "fit_gate_passed": fit_passed,
        "cpu_mujoco_gate_passed": physics_passed,
        "passed": passed,
        "status": (
            "QUALIFIED_DIVE_COMMAND_EXPERT_REPAIRED_TEACHER_PENDING_BALL_ROUTER"
            if passed
            else "REJECTED_DIVE_COMMAND_EXPERT"
        ),
        "ball_contact_exam_completed": False,
        "landing_recovery_exam_completed": False,
        "policy_integration_completed": False,
        "pixels_used_for_scoring": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "commercial_use_allowed": False,
    }
    report["report_hash"] = hash_json(report)
    _atomic_json(destination / "evidence.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--dive-source-checkout", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run_dive_athlete_cpu_exam(
        checkpoint_path=args.checkpoint,
        training_report_path=args.training_report,
        evidence_root=args.evidence_root,
        asset_root=args.asset_root,
        dive_source_checkout=args.dive_source_checkout,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_dive_athlete_cpu_exam", "validate_dive_athlete_cpu_exam_report"]
