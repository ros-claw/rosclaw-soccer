import numpy as np
import pytest

from rosclaw_soccer.rsi.quantized_action_drift import first_physical_divergence, token_drift


def test_suffix_drift_does_not_count_unused_prefix_actions():
    old = np.zeros((300, 64), np.float32)
    new = old.copy()
    new[3] = 0.0625
    new[21, 7] = -0.125
    result = token_drift(old, new)
    assert result["changed_frames"] == result["changed_dimensions"] == 1
    assert result["first_changed_control_frame"] == 21
    assert result["maximum_token_delta"] == 0.125
    assert not result["task_improvement_inferred"]
    assert token_drift(old, old)["first_changed_control_frame"] is None


@pytest.mark.parametrize("bad", ["shape", "nan", "off_grid", "range"])
def test_non_original_symbols_rejected(bad):
    old = np.zeros((300, 64), np.float32)
    new = old.copy()
    if bad == "shape":
        new = new[:299]
    elif bad == "nan":
        new[0, 0] = np.nan
    elif bad == "off_grid":
        new[0, 0] = 0.01
    else:
        new[0, 0] = 1.0
    with pytest.raises(ValueError):
        token_drift(old, new)


def test_native_divergence_checks_all_substeps_without_tolerance():
    old = np.zeros((300, 10, 43))
    new = old.copy()
    assert first_physical_divergence(old, new) is None
    new[20, 9, 42] = 1e-15
    assert first_physical_divergence(old, new) == 20
    new[21, 0, 0] = np.inf
    with pytest.raises(ValueError):
        first_physical_divergence(old, new)
