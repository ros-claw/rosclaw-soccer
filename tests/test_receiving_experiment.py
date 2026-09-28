from pathlib import Path

import pytest

from rosclaw_soccer.training.contact_teacher_ablation import ContactTeacherSuppression
from rosclaw_soccer.training.receiving_experiment import simulate_r0_receiving_course
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule
from rosclaw_soccer.training.role_receiving_courses import ReceivingCourse


@pytest.mark.parametrize(
    "kwargs",
    [
        {"checkpoint_frame": True},
        {"checkpoint_frame": 300},
        {"capture_support": 1},
        {"capture_oracle_authority": 1},
        {"capture_oracle_authority": True},
        {"capture_locomotion_memory": 1},
        {"capture_locomotion_memory": True},
        {"sonic_start_frame": 30},
        {"sonic_velocity_scale": 0.5},
        {"sonic_command_replanning": True},
        {"sonic_command_replanning": 1},
        {"sonic_command_replanning": True, "sonic_latent_schedule": object()},
        {"sonic_command_replanning": True, "sonic_pose_reference": object()},
        {"capture_sonic_targets": True},
        {"capture_sonic_targets": 1},
        {"sonic_ball_follow_gain": 0.5},
        {"sonic_ball_follow_fast_replan": True},
        {"sonic_ball_follow_fast_replan": 1},
        {"sonic_ball_follow_post_touch_chase": True},
        {"sonic_ball_follow_post_touch_chase": 1},
        {"capture_ball_follow_targets": True},
        {"capture_ball_follow_targets": 1},
        {"sonic_ball_follow_brake_distance_m": 0.6},
        {
            "sonic_model_root": Path("missing"),
            "sonic_ball_follow_gain": 0.75,
            "sonic_ball_follow_brake_distance_m": 0.6,
        },
        {
            "sonic_model_root": Path("missing"),
            "sonic_ball_follow_gain": 0.75,
            "sonic_ball_follow_fast_replan": True,
            "sonic_ball_follow_brake_distance_m": 1.0,
        },
        {
            "sonic_model_root": Path("missing"),
            "sonic_ball_follow_gain": 0.75,
            "sonic_ball_follow_fast_replan": True,
            "capture_ball_follow_targets": True,
        },
        {
            "sonic_model_root": Path("missing"),
            "sonic_ball_follow_gain": 0.75,
            "sonic_ball_follow_post_touch_chase": True,
        },
        {"sonic_model_root": Path("missing"), "sonic_ball_follow_gain": 0.0},
        {"sonic_model_root": Path("missing"), "sonic_ball_follow_gain": 1.1},
        {"sonic_model_root": Path("missing"), "sonic_ball_follow_gain": True},
        {
            "sonic_model_root": Path("missing"),
            "sonic_ball_follow_gain": 0.5,
            "capture_sonic_targets": True,
        },
        {
            "sonic_model_root": Path("missing"),
            "sonic_ball_follow_gain": 0.5,
            "sonic_start_frame": 1,
        },
        {
            "sonic_model_root": Path("missing"),
            "capture_sonic_targets": True,
            "sonic_start_frame": 1,
        },
        {"suppression": ContactTeacherSuppression("red.playmaker", 0)},
        {"suppression": ContactTeacherSuppression("blue.playmaker", 300)},
        {"sonic_model_root": Path("missing"), "sonic_start_frame": 300},
        {
            "oracle": ReceivingOracleSchedule(
                "blue.playmaker", "A3_sonic_residual", 0, 10, ((0.0,) * 29,)
            )
        },
        {"oracle": ReceivingOracleSchedule("red.playmaker", "A0_leg12", 0, 10, ((0.0,) * 12,))},
    ],
)
def test_invalid_contract_rejected_before_asset_loading(kwargs):
    with pytest.raises(ValueError):
        simulate_r0_receiving_course(
            asset_root=Path("must-not-load-assets"),
            reference_policy_path=Path("must-not-load-policy"),
            course=ReceivingCourse("blue.playmaker", 1, 0.75, 0.0),
            scenario_id="s199.test.receiving",
            **kwargs,
        )
