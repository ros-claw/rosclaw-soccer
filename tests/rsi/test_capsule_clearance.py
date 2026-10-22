import numpy as np
import pytest

from rosclaw_soccer.rsi.capsule_clearance import sphere_capsule_clearance


def test_side_contact_surface_margin_and_away_gradient():
    ball, center, axis = np.array([0.2, 0.0, 0.0]), np.zeros(3), np.array([0.0, 0.0, 1.0])
    margin, point, away = sphere_capsule_clearance(ball, 0.11, center, axis, 0.125, 0.04)
    assert margin == pytest.approx(0.05)
    np.testing.assert_array_equal(point, center)
    np.testing.assert_array_equal(away, [-1, 0, 0])
    shifted = sphere_capsule_clearance(ball, 0.11, center + 0.01 * away, axis, 0.125, 0.04)[0]
    assert shifted > margin


def test_endcap_and_degenerate_centerline_reject():
    _, point, _ = sphere_capsule_clearance(
        np.array([0.2, 0.0, 0.5]), 0.11, np.zeros(3), np.array([0.0, 0.0, 1.0]), 0.125, 0.04
    )
    np.testing.assert_array_equal(point, [0, 0, 0.125])
    with pytest.raises(ValueError):
        sphere_capsule_clearance(
            np.zeros(3), 0.11, np.zeros(3), np.array([0.0, 0.0, 1.0]), 0.125, 0.04
        )
