import numpy as np
import pytest

from rosclaw_soccer.rsi.causal_contact_task_state import before_action_task_state


def trace():
    return {
        "force_n": np.zeros((300, 1, 6)),
        "pelvis_z_per_substep_m": np.ones((300, 1, 10)),
        "ball_position_after_step_m": np.zeros((300, 1, 3)),
    }


def test_previous_completed_measurements_only_and_latches_do_not_clear():
    data = trace()
    data["force_n"][17, 0, 0] = 2
    data["force_n"][23, 0, 2] = 2
    data["ball_position_after_step_m"][31, 0, 1] = 4.1
    data["pelvis_z_per_substep_m"][43, 0, 7] = 0.64
    state = before_action_task_state(data)
    assert state.shape == (300, 6) and state.dtype == np.float64
    assert not state.flags.writeable
    for last_unknown, column in [(17, 1), (23, 3), (31, 4), (43, 5)]:
        assert np.all(state[: last_unknown + 1, column] == 0)
        assert np.all(state[last_unknown + 1 :, column] == 1)
    assert state[18, 2] == 1 / 300 and state[299, 2] == 282 / 300
    np.testing.assert_array_equal(state[:, 0], (300 - np.arange(300)) / 300)


@pytest.mark.parametrize("frame", [0, 30, 79, 200, 299])
def test_current_and_future_measurements_cannot_change_current_features(frame):
    data = trace()
    before = before_action_task_state(data)
    data["force_n"][frame:] = 100
    data["pelvis_z_per_substep_m"][frame:] = 0
    data["ball_position_after_step_m"][frame:] = 100
    after = before_action_task_state(data)
    np.testing.assert_array_equal(after[: frame + 1], before[: frame + 1])


def test_exact_event_thresholds_and_no_contact_state():
    data = trace()
    data["force_n"][:] = 1
    data["pelvis_z_per_substep_m"][:] = 0.65
    data["ball_position_after_step_m"][:, :, 1] = -4
    state = before_action_task_state(data)
    assert np.all(state[:, 1:] == 0)


@pytest.mark.parametrize("fault", ["nan", "inf", "negative", "shape", "integer"])
def test_invalid_full_trace_rejected(fault):
    data = trace()
    if fault == "nan":
        data["force_n"][299, 0, 0] = np.nan
    elif fault == "inf":
        data["ball_position_after_step_m"][0, 0, 0] = np.inf
    elif fault == "negative":
        data["force_n"][0, 0, 0] = -1
    elif fault == "shape":
        data["force_n"] = data["force_n"][:299]
    else:
        data["force_n"] = data["force_n"].astype(np.int64)
    with pytest.raises(ValueError, match="complete finite"):
        before_action_task_state(data)


def test_input_arrays_are_never_modified():
    data = trace()
    originals = {k: v.copy() for k, v in data.items()}
    before_action_task_state(data)
    for key in data:
        np.testing.assert_array_equal(data[key], originals[key])


@pytest.mark.parametrize("data", [None, {}, [], np.zeros(3)])
def test_missing_trace_mapping_rejected(data):
    with pytest.raises(ValueError, match="trace mapping"):
        before_action_task_state(data)
