import numpy as np
import pytest

from rosclaw_soccer.rsi.first_touch_ball_path_diagnostics import diagnose


def trace(first=10):
    force = np.zeros((300, 1, 6))
    if first is not None:
        force[first, 0, 0] = 2
    position = np.zeros((300, 1, 3))
    position[:, 0, 0] = np.arange(300) * 0.02
    velocity = np.zeros((300, 1, 3))
    velocity[:, 0, 0] = 1
    return dict(
        force_n=force,
        pelvis_z_per_substep_m=np.full((300, 1, 10), 0.75),
        ball_position_after_step_m=position,
        ball_position_before_step_m=position.copy(),
        ball_linear_velocity_before_step_m_s=velocity,
    )


def test_actual_later_contact_and_sixty_frame_ratio_preserved():
    value = trace()
    value["force_n"][20, 0, 1] = 3
    value["ball_position_after_step_m"][11:, 0, 1] = 0.5
    result = diagnose(value)
    assert result["later_foot_contact_frames"] == [20]
    assert result["first_contact_left_foot"]
    final = result["windows"][-1]
    assert final["frames_after_first_contact"] == 60
    assert final["later_right_foot_frames"] == 1
    assert final["lateral_displacement_m"] == result["outcome"]["lateral_60_m"] == 0.5
    assert result["net_velocity_change_is_not_isolated_foot_impulse"]


@pytest.mark.parametrize("first", [None, 239, 240, 260, 299])
def test_no_contact_and_censoring_never_fabricate_full_window(first):
    result = diagnose(trace(first))
    assert bool(any(w["frames_after_first_contact"] == 60 for w in result["windows"])) == (
        first == 239
    )
    if first == 299:
        assert result["first_contact_frame_has_no_following_velocity_sample"]


def test_nonfinite_recorded_velocity_rejected():
    value = trace()
    value["ball_linear_velocity_before_step_m_s"][2, 0, 1] = np.nan
    with pytest.raises(ValueError):
        diagnose(value)
