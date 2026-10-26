import numpy as np
import pytest

from rosclaw_soccer.rsi.contact_reach_teacher import bounded_contact_reach


def test_bounded_reach_preserves_direction_and_caps_joint_delta():
    proposal = bounded_contact_reach(
        np.eye(3), np.array([1.0, -0.5, 0.25]), damping=0.02, max_joint_delta=0.18
    )
    assert np.max(np.abs(proposal.delta)) == pytest.approx(0.18)
    np.testing.assert_allclose(proposal.delta, proposal.raw_delta * proposal.scale)
    np.testing.assert_allclose(proposal.predicted_displacement, proposal.delta)


def test_singular_jacobian_and_zero_goal_are_safe():
    proposal = bounded_contact_reach(
        np.zeros((3, 6)), np.zeros(3), damping=0.02, max_joint_delta=0.18
    )
    np.testing.assert_array_equal(proposal.delta, np.zeros(6))
    assert proposal.scale == 1.0


@pytest.mark.parametrize(
    ("jacobian", "displacement", "damping", "bound"),
    [
        (np.eye(3), [float("nan"), 0, 0], 0.02, 0.18),
        (np.eye(2), [0, 0, 0], 0.02, 0.18),
        (np.eye(3), [0, 0, 0], 0, 0.18),
        (np.eye(3), [0, 0, 0], 0.02, float("inf")),
    ],
)
def test_invalid_evidence_is_rejected(jacobian, displacement, damping, bound):
    with pytest.raises(ValueError):
        bounded_contact_reach(jacobian, displacement, damping=damping, max_joint_delta=bound)
