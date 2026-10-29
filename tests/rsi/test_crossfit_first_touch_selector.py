import numpy as np
import pytest

from scripts.rsi_crossfit_first_touch_selector import OPTIONS, choose, make_model


def test_selector_uses_success_minus_unsafe_and_stable_order():
    values = np.full((5, 2), 0.5)
    assert choose(values) == 0
    values[3] = (0.9, 0.1)
    assert choose(values) == 3
    assert len(OPTIONS) == 5
    assert make_model()[0].in_features == 3
    assert make_model()[-1].out_features == 10


def test_selector_rejects_nonfinite_and_wrong_shapes():
    values = np.full((5, 2), 0.5)
    values[0, 0] = np.nan
    with pytest.raises(ValueError):
        choose(values)
    with pytest.raises(ValueError):
        choose(np.zeros((4, 2)))
