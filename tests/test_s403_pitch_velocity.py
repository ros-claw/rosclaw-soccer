import numpy as np
import pytest

from rosclaw_soccer.world.pitch_velocity import PitchRootBounds
from rosclaw_soccer.world.player_clearance import propose_clearance_velocity


def project(position, nominal, neighbors=()):
    return propose_clearance_velocity(
        np.asarray(nominal, dtype=float),
        np.asarray(neighbors, dtype=float).reshape(-1, 2),
        maximum_speed_mps=0.7,
        additional_halfplanes=PitchRootBounds().velocity_halfplanes(
            np.asarray(position, dtype=float)
        ),
    )


def test_center_preserves_running_and_boundary_slows_only_outward_component():
    center = project((3, 0), (0.3, 0.4))
    assert center.velocity_mps == (0.3, 0.4) and not center.constrained
    edge = project((3, 2.5), (0.3, 0.4))
    assert edge.feasible and edge.velocity_mps == pytest.approx((0.3, 0.06))


def test_already_outside_root_margin_requests_inward_motion():
    result = project((3, 2.8), (0.0, 0.3))
    assert result.feasible and result.velocity_mps[1] == pytest.approx(-0.12)


def test_joint_projection_retains_neighbor_halfplane():
    result = project((3, 2.5), (0.4, 0.4), ((0.6, 0),))
    assert result.feasible
    assert result.velocity_mps[0] <= -0.12 + 1e-10
    assert result.velocity_mps[1] <= 0.06 + 1e-10


def test_conflicting_player_and_pitch_constraints_report_infeasible():
    result = project((3, 2.8), (0.0, 0.3), ((0, -0.6),))
    assert not result.feasible and result.velocity_mps == (0.0, 0.0)


def test_half_turn_symmetry():
    a = project((2.0, 2.5), (0.3, 0.4), ((0.7, -0.2),))
    b = project((4.0, -2.5), (-0.3, -0.4), ((-0.7, 0.2),))
    assert a.feasible == b.feasible
    assert a.velocity_mps == pytest.approx(-np.asarray(b.velocity_mps))


@pytest.mark.parametrize(
    "planes",
    [
        np.array([[2.0, 0.0, 0.0]]),
        np.array([[1.0, 0.0, float("nan")]]),
        np.ones((9, 3)),
        np.ones((3, 2)),
        np.array([[1, 0, 0]]),
    ],
)
def test_bad_halfplanes_rejected(planes):
    with pytest.raises(ValueError):
        propose_clearance_velocity(
            np.zeros(2),
            np.zeros((0, 2)),
            maximum_speed_mps=0.7,
            additional_halfplanes=planes,
        )


@pytest.mark.parametrize("value", [True, float("nan"), -0.1, 1.0])
def test_invalid_root_margin_rejected(value):
    with pytest.raises(ValueError):
        PitchRootBounds(root_margin_m=value)
