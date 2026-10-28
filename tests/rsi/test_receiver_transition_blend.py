import numpy as np
import pytest

from rosclaw_soccer.rsi.receiver_transition_blend import receiver_transition_action


def test_transition_projects_to_reference_envelope_and_uses_action_order():
    prev = np.full(29, 3, np.float32)
    proposal = np.full(29, 0.1, np.float32)
    reference = np.zeros(29, np.float32)
    scale = np.linspace(0.2, 1.0, 29, dtype=np.float32)
    order = np.arange(28, -1, -1)
    result = receiver_transition_action(
        prev, proposal, reference, scale, order, control_index=0, duration=10
    )
    assert result.dtype == np.float32
    assert np.max(np.abs((result - reference) * scale[np.argsort(order)])) <= 0.2
    assert np.allclose(result[order] * scale, 0.19999, atol=1e-6)


def test_final_blend_is_proposed_action_when_inside_envelope():
    prev = np.zeros(29, np.float32)
    proposal = np.full(29, 0.1, np.float32)
    result = receiver_transition_action(
        prev, proposal, prev, np.ones(29), np.arange(29), control_index=9, duration=10
    )
    np.testing.assert_allclose(result, proposal, atol=1e-7)


def test_nonfinite_and_out_of_authority_fail_closed():
    x = np.zeros(29, np.float32)
    bad = x.copy()
    bad[0] = np.nan
    with pytest.raises(ValueError):
        receiver_transition_action(
            bad, x, x, np.ones(29), np.arange(29), control_index=0, duration=10
        )
    with pytest.raises(ValueError):
        receiver_transition_action(
            x, np.ones(29, np.float32), x, np.ones(29), np.arange(29), control_index=0, duration=10
        )
