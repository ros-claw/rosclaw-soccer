import numpy as np
import pytest

from rosclaw_soccer.rsi.measured_motion_quality import measured_motion_quality


def arrays():
    return [
        np.zeros((40, 3)),
        np.zeros((40, 3)),
        np.zeros((40, 29)),
        np.zeros((4, 29)),
        np.array([0.0, 1.0]),
    ]


def test_retreat_is_relative_to_explicit_heading_not_arbitrary_world_x():
    p, w, v, t, h = arrays()
    p[:, 0] = np.linspace(0.0, 4.0, 40)
    p[:, 1] = np.linspace(0.0, -0.2, 40)
    p[:, 2] = 0.8
    w[:, 1] = 2.0
    v[:] = np.arange(40)[:, None] * 0.002
    t[:] = (np.arange(4)[:, None] * 0.02) ** 3 / 6
    r = measured_motion_quality(p, w, v, t, h)
    assert r["maximum_retreat_from_window_start_along_initial_body_heading_m"] == 0.2
    assert r["net_forward_from_window_start_along_initial_body_heading_m"] == -0.2
    assert r["root_angular_speed_rms_rad_s"] == r["root_angular_speed_peak_rad_s"] == 2.0
    assert r["joint_acceleration_rms_rad_s2"] == pytest.approx(1.0)
    assert r["control_target_jerk_rms_rad_s3"] == pytest.approx(1.0)


def test_short_observed_failure_has_missing_derivatives_not_zero_or_padding():
    p, w, v, t, h = arrays()
    r = measured_motion_quality(p[:1], w[:1], v[:1], t[:1], h)
    assert r["joint_acceleration_rms_rad_s2"] is None
    assert r["control_target_jerk_rms_rad_s3"] is None
    assert r["observed_native_samples"] == 1


@pytest.mark.parametrize("fault", ["nan", "heading", "dtype", "empty", "shape"])
def test_invalid_observation_never_sanitized_into_good_motion(fault):
    a = arrays()
    if fault == "nan":
        a[1][0, 0] = np.nan
    elif fault == "heading":
        a[-1] *= 2
    elif fault == "dtype":
        a[2] = a[2].astype(int)
    elif fault == "empty":
        a[3] = a[3][:0]
    else:
        a[0] = a[0][:, :2]
    with pytest.raises(ValueError):
        measured_motion_quality(*a)
