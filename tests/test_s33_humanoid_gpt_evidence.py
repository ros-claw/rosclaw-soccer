from __future__ import annotations

from rosclaw_soccer.skills.athlete_foundation.humanoid_gpt_evidence import (
    HumanoidGPTTrackingMetrics,
    HumanoidGPTTrackingResult,
)

_HASH = "sha256:" + "a" * 64


def test_humanoid_gpt_result_is_content_serializable() -> None:
    metrics = HumanoidGPTTrackingMetrics(
        completion_rate=1.0,
        keypoint_position_mae_m=0.05,
        keypoint_rotation_mae_rad=0.15,
        joint_position_mae_rad=0.09,
        joint_velocity_mae_rad_s=0.30,
        root_position_error_mm=120.0,
        root_velocity_error_mm_s=150.0,
        root_yaw_error_rad=0.04,
        joint_jerk_rad_s3=400.0,
    )
    result = HumanoidGPTTrackingResult(
        family="leftjump",
        input_adapter_report_hash=_HASH,
        input_archive_hash=_HASH,
        log_hash=_HASH,
        metrics=metrics,
    )

    assert result.to_dict()["metrics"]["completion_rate"] == 1.0
    assert result.physics_backend == "mujoco_cpu"
