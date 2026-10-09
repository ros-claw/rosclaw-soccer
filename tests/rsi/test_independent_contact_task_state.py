import numpy as np
import pytest

from rosclaw_soccer.rsi.causal_contact_task_state import before_action_task_state
from rosclaw_soccer.rsi.independent_contact_task_state import reconstruct_before_action_state


def trace():
    return {
        "force_n": np.zeros((300, 1, 6)),
        "pelvis_z_per_substep_m": np.full((300, 1, 10), 0.75),
        "ball_position_after_step_m": np.zeros((300, 1, 3)),
    }


@pytest.mark.parametrize("seed", range(6))
def test_vectorized_reference_matches_independent_production_loop(seed):
    data = trace()
    rng = np.random.default_rng(seed)
    data["force_n"][:] = rng.choice([0.0, 1.0, 1.1], size=(300, 1, 6), p=[0.97, 0.02, 0.01])
    data["pelvis_z_per_substep_m"][:] = rng.choice(
        [0.65, 0.649, 0.75], size=(300, 1, 10), p=[0.09, 0.01, 0.90]
    )
    data["ball_position_after_step_m"][:, 0, 1] = rng.choice(
        [0.0, -4.0, 4.0, -4.01, 4.01], size=300
    )
    expected = before_action_task_state(data)
    result = reconstruct_before_action_state(data)
    np.testing.assert_array_equal(result, expected)
    assert not result.flags.writeable


@pytest.mark.parametrize("frame", [0, 29, 30, 69, 298, 299])
def test_current_and_future_events_cannot_enter_pre_action_state(frame):
    original = trace()
    changed = {key: value.copy() for key, value in original.items()}
    changed["force_n"][frame:, 0, 4] = 2.0
    changed["ball_position_after_step_m"][frame:, 0, 1] = 5.0
    changed["pelvis_z_per_substep_m"][frame:, 0, 2] = 0.64
    old = reconstruct_before_action_state(original)
    new = reconstruct_before_action_state(changed)
    np.testing.assert_array_equal(old[: frame + 1], new[: frame + 1])
    if frame < 299:
        assert new[frame + 1, 1] == 1
        assert new[frame + 1, 2] == 1 / 300
        np.testing.assert_array_equal(new[frame + 1, 3:], np.ones(3))


def test_exact_thresholds_no_contact_and_inputs_unchanged():
    data = trace()
    data["force_n"][:] = 1.0
    data["pelvis_z_per_substep_m"][:] = 0.65
    data["ball_position_after_step_m"][:, 0, 1] = -4.0
    snapshots = {key: value.copy() for key, value in data.items()}
    result = reconstruct_before_action_state(data)
    np.testing.assert_array_equal(result[:, 1:], np.zeros((300, 5)))
    for key in data:
        np.testing.assert_array_equal(data[key], snapshots[key])


@pytest.mark.parametrize("key", list(trace()))
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_primitives_fail_closed(key, value):
    data = trace()
    data[key].flat[0] = value
    with pytest.raises(ValueError, match="finite float64"):
        reconstruct_before_action_state(data)


def test_missing_wrong_shape_dtype_and_negative_force_rejected():
    for bad in (None, {}, {"force_n": np.zeros((300, 1, 6))}):
        with pytest.raises(ValueError):
            reconstruct_before_action_state(bad)
    for key in trace():
        for bad in (np.zeros((1,)), trace()[key].astype(np.float32)):
            data = trace()
            data[key] = bad
            with pytest.raises(ValueError):
                reconstruct_before_action_state(data)
    data = trace()
    data["force_n"][0, 0, 0] = -0.1
    with pytest.raises(ValueError):
        reconstruct_before_action_state(data)
