"""Actual toy MuJoCo integration, not qualified G1 trajectories or teachers."""

import copy

import numpy as np
import pytest

from rosclaw_soccer.physics.local_impact_arm_replay import verify_local_recorded_arm
from tests.physics.test_recorded_early_episode import recorded  # noqa: F401


@pytest.mark.parametrize("recorded", [True], indirect=True)
def test_original_control_reintegrates_with_explicit_derived_zeros_not_fake_capture(recorded):  # noqa: F811
    replay, trace = recorded
    source = {k: v for k, v in trace.items() if not k.startswith("early_impact_feedback_")}
    original = copy.deepcopy(source)
    review = verify_local_recorded_arm(replay, source, arm="original_control")
    assert review["new_physical_steps"] == 3000
    assert review["zero_action_arrays_derived_not_recorded"] is True
    assert max(review["physical_review"]["maximum_absolute_errors"].values()) == 0
    assert set(source) == set(original)
    assert all(source[k].tobytes() == v.tobytes() for k, v in original.items())
    assert review["teacher_qualified"] is review["hardware_authorized"] is False
    with pytest.raises(ValueError, match="once-only"):
        verify_local_recorded_arm(replay, source, arm="original_control")


def test_correction_reintegrates_actual_recorded_action_without_zero_derivation(recorded):  # noqa: F811
    replay, trace = recorded
    review = verify_local_recorded_arm(replay, trace, arm="prospective_early_impact")
    assert review["new_physical_steps"] == 3000
    assert review["zero_action_arrays_derived_not_recorded"] is False
    assert review["zero_action_derivation"] is None


def test_cannot_remove_action_fields_and_mislabel_nonzero_control_as_original(recorded):  # noqa: F811
    replay, trace = recorded
    source = {k: v for k, v in trace.items() if not k.startswith("early_impact_feedback_")}
    with pytest.raises(ValueError, match="nonzero physical attribution"):
        verify_local_recorded_arm(replay, source, arm="original_control")
    assert replay.attempted_steps == 0


@pytest.mark.parametrize("key", ["pd_requested_torque_nm", "torque_nm", "actual_actuator_force_nm"])
@pytest.mark.parametrize("bad", ["short", "dtype", "nan"])
@pytest.mark.parametrize("recorded", [True], indirect=True)
def test_invalid_zero_derivation_rejected_before_integration(recorded, key, bad):  # noqa: F811
    replay, trace = recorded
    trace = {k: v for k, v in trace.items() if not k.startswith("early_impact_feedback_")}
    if bad == "short":
        trace[key] = trace[key][:-1]
    elif bad == "dtype":
        trace[key] = trace[key].astype(np.float32)
    else:
        trace[key][0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="typed measured"):
        verify_local_recorded_arm(replay, trace, arm="original_control")
    assert replay.attempted_steps == 0


@pytest.mark.parametrize("recorded", [True], indirect=True)
def test_original_arm_with_fake_zero_recording_is_not_accepted(recorded):  # noqa: F811
    replay, trace = recorded
    with pytest.raises(ValueError, match="must not contain"):
        verify_local_recorded_arm(replay, trace, arm="original_control")
    assert replay.attempted_steps == 0
