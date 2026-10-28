import numpy as np
import pytest

from rosclaw_soccer.rsi.reference_window_diagnostics import cosine_envelope


def test_compact_symmetric_envelope_preserves_reference_endpoints():
    weights = cosine_envelope(np.arange(20, 81), center=50, half_width=20)
    assert weights[0] == weights[-1] == 0
    assert weights[30] == 1
    np.testing.assert_allclose(weights, weights[::-1])
    assert np.all((weights >= 0) & (weights <= 1))


@pytest.mark.parametrize("width", [0, -1, 1.2, True])
def test_envelope_rejects_ambiguous_window(width):
    with pytest.raises(ValueError):
        cosine_envelope(np.arange(5), center=2, half_width=width)
