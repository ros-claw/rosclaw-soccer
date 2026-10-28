import numpy as np
import pytest

from rosclaw_soccer.rsi.passer_residual_blend import PasserContactExitGate, blended_passer_action


def test_contact_exit_gate_is_causal_and_permanent():
    gate = PasserContactExitGate()
    none = np.zeros((10, 3))
    hit = none.copy()
    hit[8, 0] = 11
    gate.observe(none)
    assert gate.active and not gate.contact_seen
    gate.observe(hit)
    assert gate.active and gate.contact_seen
    gate.observe(hit)
    assert gate.active
    gate.observe(none)
    assert not gate.active
    gate.observe(hit)
    assert not gate.active
    with pytest.raises(ValueError):
        gate.observe(np.full((10, 3), np.nan))
    with pytest.raises(ValueError):
        gate.observe(np.zeros((9, 3)))


def test_residual_blend_preserves_codec_and_zero_branch():
    action = np.full(29, 0.2, np.float32)
    reference = np.zeros(29, np.float32)
    scale = np.full(29, 0.5, np.float32)
    order = np.arange(28, -1, -1)
    assert np.array_equal(
        blended_passer_action(action, reference, scale, order, fraction=1.0), action
    )
    blended = blended_passer_action(action, reference, scale, order, fraction=0.5)
    assert np.allclose(blended, 0.1)
    assert np.array_equal(action, np.full(29, 0.2, np.float32))
    with pytest.raises(ValueError, match="declared bounded"):
        blended_passer_action(action, reference, scale, order, fraction=0.4)
    with pytest.raises(ValueError, match="authority"):
        blended_passer_action(action * 3, reference, scale, order, fraction=0.5)
