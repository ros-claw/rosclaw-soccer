import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.physical_transition_targets import recorded_transition_targets


def trace():
    before = np.arange(900, dtype=np.float64).reshape(300, 1, 3) * 0.001
    after = np.concatenate((before[1:], before[-1:] + 0.003))
    return dict(
        ball_position_before_step_m=before,
        ball_position_after_step_m=after,
        force_n=np.zeros((300, 1, 6)),
        pelvis_z_per_substep_m=np.full((300, 1, 10), 0.7),
        motor_delta_rad=np.zeros((300, 1, 12)),
    )


def test_current_action_and_all_future_labels_aligned_without_terminal_drop():
    data = trace()
    data["force_n"][30, 0, 0] = 2
    data["force_n"][31, 0, 2] = 2
    data["pelvis_z_per_substep_m"][299, 0, 9] = 0.64
    data["motor_delta_rad"][30, 0, 4] = 0.1
    original = copy.deepcopy(data)
    targets = recorded_transition_targets(data)
    assert targets["recorded_ball_displacement_m"].shape == (270, 3)
    np.testing.assert_allclose(targets["recorded_ball_displacement_m"], 0.003)
    assert targets["foot_contact"][0] and not targets["nonfoot_contact"][0]
    assert targets["nonfoot_contact"][1]
    assert targets["pelvis_below_original_floor"][-1]
    assert targets["actual_executed_motor_delta_rad"][0, 4] == 0.1
    for k in data:
        np.testing.assert_array_equal(data[k], original[k])
    targets["actual_executed_motor_delta_rad"][:] = 0
    assert data["motor_delta_rad"][30, 0, 4] == 0.1


@pytest.mark.parametrize(
    "field,value",
    [
        ("force_n", np.nan),
        ("force_n", -1),
        ("pelvis_z_per_substep_m", -1),
        ("motor_delta_rad", 0.161),
        ("ball_position_before_step_m", np.inf),
    ],
)
def test_bad_physical_arrays_rejected(field, value):
    data = trace()
    data[field][50, 0, 0] = value
    with pytest.raises(ValueError):
        recorded_transition_targets(data)


def test_sampling_mismatch_and_missing_array_rejected():
    data = trace()
    data["ball_position_after_step_m"][40, 0, 0] += 0.01
    with pytest.raises(ValueError, match="aligned"):
        recorded_transition_targets(data)
    data = trace()
    del data["force_n"]
    with pytest.raises(ValueError, match="complete"):
        recorded_transition_targets(data)


def test_original_thresholds_strict_and_frame29_not_leaked():
    data = trace()
    data["force_n"][29, 0, 2] = 100
    data["force_n"][30, 0, 2] = 1
    data["pelvis_z_per_substep_m"][30] = 0.65
    targets = recorded_transition_targets(data)
    assert not targets["nonfoot_contact"].any()
    assert not targets["pelvis_below_original_floor"].any()
