import numpy as np
import pytest

from rosclaw_soccer.rsi.receiving_legacy_proximity import legacy_proximity


def test_proximity_uses_only_precontact_and_marks_hypothetical_phase():
    q = np.zeros((2, 43))
    v = np.zeros((2, 41))
    q[:, 2] = 0.7
    q[:, 38] = 0.7
    flags = np.array([[False, False], [True, False]], dtype=np.bool_)
    anchors = np.zeros((3, 10))
    anchors[:, 8] = 0.08
    result = legacy_proximity(q, v, flags, anchors)
    assert result["precontact_states"] == 1 and result["nearest_control"] == 0
    assert result["any_relaxed_gate_match"] is True
    assert result["physical_transfer_proven"] is False
    flags[:] = [[True, False], [False, False]]
    with pytest.raises(ValueError, match="persistent"):
        legacy_proximity(q, v, flags, anchors)
