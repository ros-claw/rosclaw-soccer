import numpy as np
import pytest

from rosclaw_soccer.physics.root_motion_window import summarize_root_motion_window


def body(n=60):
    q = np.zeros((n, 7))
    q[:, 3] = 1
    v = np.zeros((n, 6))
    return q, v


def summary(q, v, **kwargs):
    return summarize_root_motion_window(
        q, v, sample_dt_s=0.02, forward_axis_world=np.array([1.0, 0.0, 0.0]), **kwargs
    )


def test_net_forward_motion_can_contain_retreat():
    q, v = body(4)
    q[:, 0] = [0.0, 0.2, 0.1, 0.4]
    v[:, 0] = [1.0, -1.0, 1.0, 0.0]
    s = summary(q, v)
    assert s["net_root_translation_world_m"][0] == pytest.approx(0.4)
    assert s["sampled_forward_root_travel_m"] == pytest.approx(0.5)
    assert s["sampled_backward_root_travel_m"] == pytest.approx(0.1)
    assert s["negative_root_velocity_samples"] == 1
    assert s["position_endpoint_span_s"] == pytest.approx(0.06)


def test_upright_yaw_is_not_tilt_or_automatic_jitter():
    q, v = body()
    yaw = np.linspace(0.0, 0.7, len(q))
    q[:, 3], q[:, 6] = np.cos(yaw / 2), np.sin(yaw / 2)
    v[:, 5] = 1.2
    s = summary(q, v)
    assert s["root_tilt_maximum_rad"] == 0
    assert s["root_heading_change_rad"] == pytest.approx(0.7)
    assert s["root_local_angular_energy_fractions"] == [0.0, 0.0, 1.0]
    assert s["angular_speed_is_not_jitter_classification"] is True
    assert s["balance_or_next_skill_success_qualified"] is False


def test_heading_unwrap_and_quaternion_double_cover():
    q, v = body(3)
    yaw = np.array([3.0, 3.2, 3.4])
    q[:, 3], q[:, 6] = np.cos(yaw / 2), np.sin(yaw / 2)
    a = summary(q, v)
    q[1, 3:7] *= -1
    b = summary(q, v)
    assert a == b
    assert a["root_heading_change_rad"] == pytest.approx(0.4)


def test_vertical_body_does_not_fabricate_heading():
    q, v = body()
    q[:, 3] = q[:, 5] = np.sqrt(0.5)
    s = summary(q, v)
    assert s["root_tilt_maximum_rad"] == pytest.approx(np.pi / 2)
    assert s["root_heading_change_rad"] is None


def test_invalid_quaternion_not_silently_normalized():
    q, v = body()
    q[2, 3] = 2
    with pytest.raises(ValueError):
        summary(q, v)


def test_inputs_not_mutated_and_zero_angular_fraction_unknown():
    q, v = body()
    saved = (q.copy(), v.copy())
    assert summary(q, v)["root_local_angular_energy_fractions"] is None
    np.testing.assert_array_equal(q, saved[0])
    np.testing.assert_array_equal(v, saved[1])
