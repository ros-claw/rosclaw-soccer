import numpy as np
import pytest

from rosclaw_soccer.rsi.measured_joint_bridge import sampled_hermite_joint_bridge


def test_bridge_matches_actual_initial_velocity_and_terminal_stance():
    q = np.linspace(-0.1, 0.1, 29)
    v = np.linspace(-0.5, 0.5, 29)
    terminal = np.zeros(29)
    bounds = np.tile([-1.0, 1.0], (29, 1))
    result = sampled_hermite_joint_bridge(
        q,
        v,
        terminal,
        bounds,
        controls=20,
        control_interval_s=0.02,
        maximum_sampled_speed_rad_s=2.0,
    )
    np.testing.assert_array_equal(result["position"][0], q)
    np.testing.assert_allclose(result["velocity"][0], v, atol=1e-15)
    np.testing.assert_allclose(result["position"][-1], terminal, atol=1e-15)
    np.testing.assert_allclose(result["velocity"][-1], 0.0, atol=1e-15)
    assert result["contact_or_dynamics_not_qualified"]
    assert result["continuous_joint_bound_not_proven"]


def test_bridge_rejects_overshoot_and_excess_speed_without_clipping():
    q, v, terminal = np.zeros(29), np.zeros(29), np.zeros(29)
    bounds = np.tile([-0.1, 0.1], (29, 1))
    v[0] = 2.0
    with pytest.raises(ValueError):
        sampled_hermite_joint_bridge(
            q,
            v,
            terminal,
            bounds,
            controls=20,
            control_interval_s=0.02,
            maximum_sampled_speed_rad_s=1.0,
        )
    with pytest.raises(ValueError):
        sampled_hermite_joint_bridge(
            q,
            v,
            terminal,
            bounds,
            controls=20,
            control_interval_s=0.02,
            maximum_sampled_speed_rad_s=3.0,
        )
