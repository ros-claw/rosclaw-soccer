from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi.contact_exit_torque_mpc import pressure_side, torque_candidates
from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS


def history():
    h = np.zeros((3, 6))
    h[:, 0] = 20
    return h


def test_three_bins_and_current_solver_contact_and_near_surface_required():
    h = history()
    assert (
        pressure_side(
            h, np.array([-0.001, 1.0]), np.array([10.0, 0.0]), prior_nonfoot_contact=False
        )
        == 0
    )
    # A short strike can span three peak bins but be over at the last substep.
    assert (
        pressure_side(h, np.array([-0.001, 1.0]), np.zeros(2), prior_nonfoot_contact=False) is None
    )
    assert (
        pressure_side(h, np.array([0.02, 1.0]), np.array([10.0, 0.0]), prior_nonfoot_contact=False)
        is None
    )
    assert pressure_side(h[:2], np.zeros(2), np.ones(2) * 10, prior_nonfoot_contact=False) is None


def test_prior_or_recent_nonfoot_contact_cannot_activate_repair_as_if_safe():
    h = history()
    assert pressure_side(h, np.zeros(2), np.ones(2) * 10, prior_nonfoot_contact=True) is None
    h[0, 4] = 10
    assert pressure_side(h, np.zeros(2), np.ones(2) * 10, prior_nonfoot_contact=False) is None


def test_both_feet_choose_larger_current_solved_force_not_future_label():
    h = np.zeros((3, 6))
    h[:, :2] = 20
    assert pressure_side(h, np.zeros(2), np.array([10.0, 11.0]), prior_nonfoot_contact=False) == 1


@pytest.mark.parametrize("kind", ["nan", "negative", "shape", "dtype"])
def test_invalid_contact_history_rejected(kind):
    h = history()
    if kind == "nan":
        h[0, 0] = np.nan
    elif kind == "negative":
        h[0, 0] = -1
    elif kind == "shape":
        h = np.zeros((3, 5))
    else:
        h = h.astype(np.float32)
    with pytest.raises(ValueError):
        pressure_side(h, np.zeros(2), np.ones(2), prior_nonfoot_contact=False)


def sample():
    jac = np.zeros((2, 3, 29))
    jac[0, 2, :6] = [-0.3, 0.1, 0, 0.15, -0.02, 0.05]
    jac[0, 0, :6] = [0.3, 0.02, 0, -0.2, -0.1, 0.1]
    com = np.zeros((3, 29))
    com[1, 12:] = 0.01
    return SimpleNamespace(
        foot_material_point_jacobian_m_per_rad=jac,
        foot_origin_world_m=np.array([[0.0, 0.1, 0.0], [0.0, -0.1, 0.0]]),
        robot_subtree_com_world_m=np.array([0.0, 0.0, 0.7]),
        com_jacobian_m_per_rad=com,
    )


def test_candidate_torques_use_actuated_only_jacobians_and_intersect_caps_and_slew():
    p = np.zeros(29)
    choices = torque_candidates(sample(), 0, p)
    assert 1 <= len(choices) <= 6
    assert choices[0][0] == "decay" and np.all(choices[0][1] == 0)
    caps = np.minimum(12, 0.2 * np.asarray(G1_HARD_TORQUE_LIMITS))
    assert len({u.tobytes() for _, u in choices}) == len(choices)
    assert any(np.any(u != 0) for _, u in choices)
    for _, u in choices:
        assert u.shape == (29,) and u.dtype == np.float64
        assert np.all(abs(u) <= caps) and np.all(abs(u - p) <= 2)
    assert np.all(p == 0)


def test_decay_to_zero_is_rate_limited_not_an_abrupt_reset():
    p = np.minimum(12, 0.2 * np.asarray(G1_HARD_TORQUE_LIMITS))
    choices = torque_candidates(sample(), 0, p)
    np.testing.assert_array_equal(choices[0][1], np.maximum(p - 2, 0))
    assert any(np.any(u != 0) for _, u in choices)


@pytest.mark.parametrize("side", [-1, 2, True])
def test_invalid_selected_foot_rejected(side):
    with pytest.raises(ValueError):
        torque_candidates(sample(), side, np.zeros(29))
