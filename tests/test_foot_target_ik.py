import numpy as np
import pytest

from rosclaw_soccer.sim.foot_target_ik import bounded_foot_target_delta


def test_bounded_damped_foot_target() -> None:
    jacobian = np.array([[0.5, 0.2, 0.1], [0.1, -0.3, -0.2]])
    delta = bounded_foot_target_delta(jacobian, np.array([0.05, -0.01]))
    assert delta.shape == (3,)
    assert np.max(np.abs(delta)) <= 0.12
    assert (jacobian @ delta)[0] > 0.0


def test_rejects_unbounded_or_nonfinite_ik_request() -> None:
    with pytest.raises(ValueError):
        bounded_foot_target_delta(np.zeros((2, 3)), np.array([0.3, 0.0]))
    with pytest.raises(ValueError):
        bounded_foot_target_delta(np.full((2, 3), np.nan), np.zeros(2))


def test_knee_clearance_changes_joint_solution() -> None:
    foot = np.array([[0.5, 0.2, 0.1], [0.1, -0.3, -0.2]])
    knee = np.array([0.4, 0.3, 0.0])
    desired = np.array([0.04, 0.0])
    plain = bounded_foot_target_delta(foot, desired)
    clear = bounded_foot_target_delta(
        foot, desired, knee_jacobian_x_m_per_rad=knee, knee_retreat_m=0.03
    )
    assert (knee @ clear) < (knee @ plain)
