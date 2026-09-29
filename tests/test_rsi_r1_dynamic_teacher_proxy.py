"""Teacher and frame-zero proxy actions fail closed before physics access."""

import numpy as np
import pytest
from rsi_r1_dynamic_teacher_proxy_v141 import teacher_torque
from rsi_r1_early_dynamic_proxy_v142 import replay


def test_teacher_rejects_unbounded_discrete_input():
    row = np.zeros(17, dtype=np.float64)
    row[0] = 2.0
    with pytest.raises(ValueError, match="finite recorded same-substep teacher"):
        teacher_torque(None, None, row, None, np.zeros(29))


@pytest.mark.parametrize("weights", [np.ones(5), np.full(6, float("nan")), np.full(6, 1.01)])
def test_early_proxy_rejects_invalid_action(weights):
    with pytest.raises(ValueError, match="bounded six-joint earlier"):
        replay(None, {}, weights)
