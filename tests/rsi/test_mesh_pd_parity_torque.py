import numpy as np
import pytest

from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS
from scripts.rsi_mesh_contact_pd_parity import feedback_torque


def test_repository_tuple_limits_are_used_in_actual_feedback_arithmetic():
    assert isinstance(G1_HARD_TORQUE_LIMITS, tuple)
    limits = np.asarray(G1_HARD_TORQUE_LIMITS)
    zeros, ones = np.zeros(29), np.ones(29)
    assert np.array_equal(feedback_torque(ones, zeros, zeros, 1000 * ones, ones), limits)
    assert np.array_equal(feedback_torque(-ones, zeros, zeros, 1000 * ones, ones), -limits)
    assert np.array_equal(feedback_torque(zeros, zeros, ones, ones, 2 * ones), -2 * ones)


@pytest.mark.parametrize(
    "value", [np.zeros(28), np.full(29, np.nan), np.full(29, np.inf), ["1"] * 29]
)
def test_invalid_feedback_rejects_before_step(value):
    with pytest.raises(ValueError, match="canonical 29-joint"):
        feedback_torque(value, np.zeros(29), np.zeros(29), np.ones(29), np.ones(29))
