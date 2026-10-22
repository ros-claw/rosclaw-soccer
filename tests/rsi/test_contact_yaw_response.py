import numpy as np
import pytest

from rosclaw_soccer.rsi.contact_yaw_response import contact_yaw_proposals


def test_only_yaw_changes_and_control_probe_included():
    base = np.full((2, 6), 0.1)
    p, r = contact_yaw_proposals(base, center=104, contact_control=104, radial_heading_degrees=12.0)
    assert p.shape == (11, 2, 6)
    assert np.array_equal(p[0], base)
    assert np.array_equal(p[:, :, :5], np.broadcast_to(base[:, :5], (11, 2, 5)))
    assert (p[8, :, 5] < base[:, 5]).all()
    assert (p[-1, :, 5] > base[:, 5]).all()
    assert np.abs(p).max() <= 1
    assert r["physical_success_predicted"] is False


def test_rightward_and_leftward_normals_mirror():
    base = np.zeros((2, 6))
    a, _ = contact_yaw_proposals(base, center=100, contact_control=101, radial_heading_degrees=10.0)
    b, _ = contact_yaw_proposals(
        base, center=100, contact_control=101, radial_heading_degrees=-10.0
    )
    assert np.array_equal(a, -b)


@pytest.mark.parametrize("angle", [float("nan"), 90.0, float("inf")])
def test_unqualified_heading_rejected(angle):
    with pytest.raises(ValueError):
        contact_yaw_proposals(
            np.zeros((2, 6)), center=100, contact_control=100, radial_heading_degrees=angle
        )


def test_outside_authority_window_rejected():
    with pytest.raises(ValueError):
        contact_yaw_proposals(
            np.zeros((2, 6)), center=100, contact_control=200, radial_heading_degrees=10.0
        )
