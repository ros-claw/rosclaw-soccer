import numpy as np
import pytest

from rosclaw_soccer.rsi.receiver_target_limit_projection import (
    projected_action_memory,
    receiver_target_limit_projection,
)


def test_explicit_leg_projection_leaves_upper_body_and_input_untouched():
    nominal = np.zeros(29)
    nominal[[0, 7, 20]] = [2.0, -3.0, 4.0]
    saved = nominal.copy()
    result = receiver_target_limit_projection(
        nominal, np.tile([-1.0, 1.0], (29, 1)), np.ones(29, dtype=bool), enabled=True
    )
    assert result["target"][[0, 7, 20]].tolist() == [1.0, -1.0, 4.0]
    assert np.flatnonzero(result["changed_axes"]).tolist() == [0, 7]
    np.testing.assert_array_equal(nominal, saved)


def test_disabled_is_exact_original_even_outside_joint_limits():
    nominal = np.full(29, 3.0)
    result = receiver_target_limit_projection(
        nominal, np.tile([-1.0, 1.0], (29, 1)), np.ones(29, dtype=bool), enabled=False
    )
    np.testing.assert_array_equal(result["target"], nominal)
    assert not result["changed_axes"].any()


@pytest.mark.parametrize("fault", ["nan", "missing_leg_limit", "reversed", "invalid_flag"])
def test_invalid_inputs_fail_closed(fault):
    nominal = np.zeros(29)
    limits = np.tile([-1.0, 1.0], (29, 1))
    limited = np.ones(29, dtype=bool)
    enabled = True
    if fault == "nan":
        nominal[3] = np.nan
    elif fault == "missing_leg_limit":
        limited[3] = False
    elif fault == "reversed":
        limits[0] = [1, -1]
    else:
        enabled = 1
    with pytest.raises(ValueError):
        receiver_target_limit_projection(nominal, limits, limited, enabled=enabled)


def test_projected_action_memory_reflects_only_changed_leg_axes():
    raw = np.arange(29, dtype=np.float32)
    before = raw.copy()
    mask = np.zeros(29, dtype=bool)
    mask[[2, 8]] = True
    target = np.full(29, 0.5, dtype=np.float64)
    offset = np.full(29, 0.1, dtype=np.float64)
    scale = np.full(29, 0.2, dtype=np.float64)
    memory = projected_action_memory(raw, target, offset, scale, mask)
    np.testing.assert_array_equal(raw, before)
    np.testing.assert_array_equal(memory[~mask], raw[~mask])
    np.testing.assert_allclose(memory[mask], 2.0)
    np.testing.assert_allclose(memory[mask] * scale[mask] + offset[mask], target[mask])


def test_projected_memory_rejects_unowned_axis_or_invalid_scale():
    mask = np.zeros(29, dtype=bool)
    mask[12] = True
    with pytest.raises(ValueError):
        projected_action_memory(
            np.zeros(29, np.float32), np.zeros(29), np.zeros(29), np.ones(29), mask
        )
    mask[12] = False
    with pytest.raises(ValueError):
        projected_action_memory(
            np.zeros(29, np.float32), np.zeros(29), np.zeros(29), np.zeros(29), mask
        )
