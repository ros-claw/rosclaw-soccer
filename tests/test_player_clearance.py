import numpy as np
import pytest

from rosclaw_soccer.world.player_clearance import propose_clearance_velocity


def test_open_lane_preserves_run() -> None:
    result = propose_clearance_velocity(
        np.array([0.6, 0.2]), np.array([[2.0, 0.0]]), maximum_speed_mps=0.7
    )
    assert result.velocity_mps == (0.6, 0.2) and result.feasible and not result.constrained


def test_close_opponent_removes_closing_but_keeps_tangent() -> None:
    result = propose_clearance_velocity(
        np.array([0.6, 0.3]), np.array([[0.8, 0.0]]), maximum_speed_mps=0.7
    )
    np.testing.assert_allclose(result.velocity_mps, [0, 0.3], atol=1e-10)
    assert result.feasible and result.constrained


def test_overlapping_sandwich_is_not_called_safe() -> None:
    result = propose_clearance_velocity(
        np.array([0.6, 0.0]), np.array([[0.5, 0.0], [-0.5, 0.0]]), maximum_speed_mps=0.7
    )
    assert not result.feasible


def test_neighbor_order_and_half_turn_symmetry() -> None:
    offsets = np.array([[0.6, 0.3], [0.2, 0.85], [-0.7, 0.6]])
    nominal = np.array([0.55, 0.4])
    result = propose_clearance_velocity(nominal, offsets, maximum_speed_mps=0.7)
    reverse = propose_clearance_velocity(nominal, offsets[::-1], maximum_speed_mps=0.7)
    mirror = propose_clearance_velocity(-nominal, -offsets, maximum_speed_mps=0.7)
    np.testing.assert_allclose(result.velocity_mps, reverse.velocity_mps, atol=1e-10)
    np.testing.assert_allclose(result.velocity_mps, -np.asarray(mirror.velocity_mps), atol=1e-10)
    assert np.linalg.norm(result.velocity_mps) <= 0.700000001


def test_nonfinite_input_rejected() -> None:
    with pytest.raises(ValueError):
        propose_clearance_velocity(
            np.array([float("nan"), 0]), np.empty((0, 2)), maximum_speed_mps=0.7
        )
