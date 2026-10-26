import numpy as np
import pytest

from rosclaw_soccer.rsi.stage_tracking_diagnostic import signed_stage_tracking


def test_signed_tracking_reports_reverse_motion_and_return():
    p = np.array([[0, 0], [0, -0.02], [0, -0.01], [0, 0.01]], dtype=np.float64)
    v = np.array([[0, 0], [0, -1], [0, 0.5], [0, 1]], dtype=np.float64)
    result = signed_stage_tracking(
        p, v, intended_displacement=np.array([0, 0.1], dtype=np.float64), native_timestep_s=0.02
    )
    assert result["native_steps"] == 3
    assert result["maximum_opposite_displacement_m"] == pytest.approx(0.02)
    assert result["measured_signed_displacement_m"] == pytest.approx(0.01)
    assert result["first_positive_progress_s"] == pytest.approx(0.06)
    assert result["physical_safety_or_task_success_inferred"] is False


def test_negative_direction_and_censored_trace_are_not_padded():
    p = np.array([[0, 0], [0, -0.02]], dtype=np.float64)
    v = np.array([[0, 0], [0, -0.4]], dtype=np.float64)
    result = signed_stage_tracking(
        p, v, intended_displacement=np.array([0, -0.1], dtype=np.float64), native_timestep_s=0.02
    )
    assert result["observed_duration_s"] == pytest.approx(0.02)
    assert result["measured_signed_displacement_m"] == pytest.approx(0.02)
    assert result["final_signed_velocity_mps"] == pytest.approx(0.4)


@pytest.mark.parametrize("bad", [np.nan, np.inf, 0.0, -0.1])
def test_bad_clock_is_rejected(bad):
    p = np.zeros((2, 2), dtype=np.float64)
    with pytest.raises(ValueError):
        signed_stage_tracking(
            p, p, intended_displacement=np.array([0, 0.1], dtype=np.float64), native_timestep_s=bad
        )


def test_nonfinite_or_misaligned_state_rejected():
    p = np.zeros((2, 2), dtype=np.float64)
    p[1, 1] = np.nan
    with pytest.raises(ValueError):
        signed_stage_tracking(
            p, p, intended_displacement=np.array([0, 0.1], dtype=np.float64), native_timestep_s=0.02
        )
