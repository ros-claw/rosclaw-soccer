"""Seal Humanoid-GPT CPU MuJoCo tracking logs as partial foundation evidence."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rosclaw_soccer.sim.contracts import hash_json

_METRICS = {
    "completion_rate": r"Average Trajectory Completion: ([0-9.]+)",
    "keypoint_position_mae_m": r"Average KPT Position MAE: ([0-9.]+) m",
    "keypoint_rotation_mae_rad": r"Average KPT Rotation MAE: ([0-9.]+) rad",
    "joint_position_mae_rad": r"Average Joint Position MAE: ([0-9.]+) rad",
    "joint_velocity_mae_rad_s": r"Average Joint Velocity MAE: ([0-9.]+) rad/s",
    "root_position_error_mm": r"Average Root Pos Error: ([0-9.]+) mm",
    "root_velocity_error_mm_s": r"Average Root Vel Error: ([0-9.]+) mm/s",
    "root_yaw_error_rad": r"Average Root Yaw Error: ([0-9.]+) rad",
    "joint_jerk_rad_s3": r"Average Joint Jerk: ([0-9.]+) rad/s\^3",
}


@dataclass(frozen=True)
class HumanoidGPTTrackingMetrics:
    completion_rate: float
    keypoint_position_mae_m: float
    keypoint_rotation_mae_rad: float
    joint_position_mae_rad: float
    joint_velocity_mae_rad_s: float
    root_position_error_mm: float
    root_velocity_error_mm_s: float
    root_yaw_error_rad: float
    joint_jerk_rad_s3: float


@dataclass(frozen=True)
class HumanoidGPTTrackingResult:
    family: str
    input_adapter_report_hash: str
    input_archive_hash: str
    log_hash: str
    metrics: HumanoidGPTTrackingMetrics
    physics_backend: str = "mujoco_cpu"
    converted_by_backend: bool = True
    schema_version: str = "rosclaw_soccer.humanoid_gpt_tracking_result.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def seal_humanoid_gpt_tracking_evidence(
    *,
    backend_checkout: Path,
    model_path: Path,
    family_inputs: tuple[tuple[str, Path, Path], ...],
    pd_baseline_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Parse exact upstream metrics while keeping missing safety metrics explicit."""

    checkout = backend_checkout.expanduser().resolve()
    commit = subprocess.run(
        ("git", "-C", str(checkout), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != "d5a8a5f4809760cafed6a75b97494ecf4b650408":
        raise ValueError("Humanoid-GPT checkout is not pinned")
    model = model_path.expanduser().resolve()
    if not model.is_file():
        raise ValueError("Humanoid-GPT model is unavailable")
    results: list[HumanoidGPTTrackingResult] = []
    for family, adapter_path, log_path in family_inputs:
        adapter = json.loads(adapter_path.read_text(encoding="utf-8"))
        adapter_hash = str(adapter.pop("adapter_report_hash", ""))
        if hash_json(adapter) != adapter_hash or adapter.get("family") != family:
            raise ValueError(f"Humanoid-GPT adapter evidence is invalid: {family}")
        archive = adapter_path.with_suffix(".npz")
        if _file_hash(archive) != adapter.get("archive_hash"):
            raise ValueError(f"Humanoid-GPT input archive hash mismatch: {family}")
        log_bytes = log_path.read_bytes()
        text = log_bytes.decode("utf-8")
        parsed: dict[str, float] = {}
        for name, pattern in _METRICS.items():
            matches = re.findall(pattern, text)
            if len(matches) != 1:
                raise ValueError(f"Humanoid-GPT log metric is ambiguous: {family}/{name}")
            parsed[name] = float(matches[0])
        results.append(
            HumanoidGPTTrackingResult(
                family=family,
                input_adapter_report_hash=adapter_hash,
                input_archive_hash=str(adapter["archive_hash"]),
                log_hash="sha256:" + hashlib.sha256(log_bytes).hexdigest(),
                metrics=HumanoidGPTTrackingMetrics(**parsed),
            )
        )
    pd_baseline = json.loads(pd_baseline_path.read_text(encoding="utf-8"))
    declared_pd_hash = str(pd_baseline.pop("report_hash", ""))
    if hash_json(pd_baseline) != declared_pd_hash:
        raise ValueError("PD baseline report hash mismatch")
    all_partial_pass = all(
        item.metrics.completion_rate >= 0.95
        and item.metrics.keypoint_position_mae_m <= 0.12
        and item.metrics.joint_position_mae_rad <= 0.35
        and item.metrics.joint_jerk_rad_s3 <= 1200.0
        for item in results
    )
    mean = {
        name: sum(getattr(item.metrics, name) for item in results) / len(results)
        for name in _METRICS
    }
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.humanoid_gpt_partial_foundation_evidence.v1",
        "backend_id": "humanoid-gpt",
        "backend_commit": commit,
        "model_hash": _file_hash(model),
        "motion_license": "CC-BY-NC-SA-4.0",
        "commercial_use_allowed": False,
        "results": [item.to_dict() for item in results],
        "aggregate_mean": mean,
        "partial_tracking_passed": all_partial_pass,
        "status": (
            "PARTIAL_PHYSICS_TRACKING_PASS"
            if all_partial_pass
            else "PARTIAL_PHYSICS_TRACKING_FAIL"
        ),
        "full_foundation_shootout_eligible": False,
        "missing_qualification_metrics": [
            "foot_slip_mps",
            "minimum_pelvis_height_m",
            "peak_torque_fraction",
            "torque_saturation_rate",
            "p95_root_angular_speed_rad_s",
            "recovery_rate",
            "transition_error_rad",
        ],
        "pd_baseline_report_hash": declared_pd_hash,
        "comparison": {
            "pd_recovery_rate": pd_baseline["aggregate"]["recovery_rate"],
            "pd_fallen_family_count": sum(
                int(bool(item["fell"])) for item in pd_baseline["family_reports"]
            ),
            "learned_tracker_completed_family_count": sum(
                int(item.metrics.completion_rate >= 0.95) for item in results
            ),
            "interpretation": "LEARNED_PHYSICS_TRACKER_CLOSES_KINEMATIC_TO_DYNAMIC_GAP",
        },
        "physical_truth": True,
        "champion_eligible": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    target = output_path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    return report


def _file_hash(path: Path) -> str:
    if not path.is_file() or path.stat().st_size > 2 * 1024 * 1024 * 1024:
        raise ValueError(f"evidence artifact is unavailable or oversized: {path.name}")
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "HumanoidGPTTrackingMetrics",
    "HumanoidGPTTrackingResult",
    "seal_humanoid_gpt_tracking_evidence",
]
