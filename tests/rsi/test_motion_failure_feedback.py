import pytest

from rosclaw_soccer.rsi.motion_failure_feedback import first_touch_failure_causes


def score():
    return dict(
        qualified=True,
        common_prefix_physical_safe=True,
        complete_post60_observable=True,
        first_foot_contact_global_frame=239,
        minimum_pelvis_z_m=0.65,
        minimum_upright=0.9,
        maximum_nonfoot_force_n=1.0,
        forward_to_task_end_m=0.5,
        lateral_to_task_end_m=0.3,
        maximum_ball_vx_mps=0.5,
    )


def test_original_thresholds_not_widened_for_almost_success():
    s = score()
    assert first_touch_failure_causes(s) == []
    s["lateral_to_task_end_m"] = -0.300001
    s["qualified"] = False
    assert first_touch_failure_causes(s) == ["lateral_miss"]


def test_multiple_failure_causes_are_not_hidden_or_summed_as_episodes():
    s = score()
    s.update(qualified=False, maximum_ball_vx_mps=0.35, minimum_pelvis_z_m=0.64)
    assert first_touch_failure_causes(s) == ["body_height", "insufficient_forward_speed"]


def test_terminal_observation_or_qualification_mismatch_is_rejected():
    s = score()
    s["first_foot_contact_global_frame"] = 240
    with pytest.raises(ValueError, match="post60"):
        first_touch_failure_causes(s)
    s = score()
    s["qualified"] = False
    with pytest.raises(ValueError, match="qualification"):
        first_touch_failure_causes(s)
