import numpy as np
import pytest
from test_contact_reference_ik import fixture

from rosclaw_soccer.rsi.temporal_contact_fit import fit_temporal_contact


def test_temporal_fit_leaves_source_untouched_and_is_not_dynamics():
    model, data, layout = fixture(0.12)
    source = np.repeat(data.qpos[None, :], 5, axis=0)
    before = source.copy(), data.qpos.copy(), data.qvel.copy(), data.time
    result = fit_temporal_contact(model, layout, source, foot="right", event=2, max_evaluations=2)
    np.testing.assert_array_equal(source, before[0])
    np.testing.assert_array_equal(data.qpos, before[1])
    np.testing.assert_array_equal(data.qvel, before[2])
    assert data.time == before[3]
    assert result["physics_steps"] == 0
    assert not result["physically_qualified"] and not result["promotion_authorized"]
    offsets = np.asarray(result["active_leg_offsets_rad"])
    np.testing.assert_array_equal(offsets[[0, -1]], 0)
    assert np.max(np.abs(offsets)) <= 0.35 + 1e-12


@pytest.mark.parametrize("event", [-1, 0, 4, True])
def test_temporal_fit_rejects_invalid_event(event):
    model, data, layout = fixture(0.12)
    source = np.repeat(data.qpos[None, :], 5, axis=0)
    with pytest.raises(ValueError):
        fit_temporal_contact(model, layout, source, foot="right", event=event)
