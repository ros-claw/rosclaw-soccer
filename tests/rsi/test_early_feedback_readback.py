"""Independent synthetic80ms array checks, not physical task performance."""

import numpy as np
import pytest

from scripts.rsi_read_early_feedback_arrays import verify_early_branch
from tests.rsi.test_zero_feedback_readback import fixture as old_fixture


def fixture():
    old, trace, dense = old_fixture()
    arrays = {"end_integration_state": old["end_integration_state"].copy()}
    for key, value in old.items():
        if key not in ("end_integration_state", "first_control_peak_force_n") and not (
            key.startswith("next_") or key.startswith("neural_next_")
        ):
            arrays[key] = np.concatenate((value, value))
    arrays.update(
        requested_offset_nm=np.zeros((40, 29)),
        completed_contact_peak_force_n=np.zeros((40, 6)),
        foot_contact_peak_force_n_after_substep=np.zeros((40, 2)),
        nonfoot_robot_contact_over1n_after_substep=np.zeros(40, dtype=np.bool_),
        completed_foot_ground_normal_n=np.zeros((40, 2)),
    )
    for j in range(1, 4):
        for key, value in old.items():
            if key.startswith("next_"):
                arrays[f"next_{j}_" + key.removeprefix("next_")] = value.copy()
            if key.startswith("neural_next_"):
                arrays[f"neural_next_{j}_" + key.removeprefix("neural_next_")] = value.copy()
    return arrays, trace, dense


def test_whole80ms_zero_branch():
    verify_early_branch(*fixture(), 30)


@pytest.mark.parametrize("j", [1, 2, 3])
@pytest.mark.parametrize("field", ["joint_target_rad", "motor_delta_rad", "canonical_qpos"])
def test_all_three_targets_are_independently_checked(j, field):
    a, t, d = fixture()
    a[f"next_{j}_" + field][0] = 0.01
    with pytest.raises(ValueError):
        verify_early_branch(a, t, d, 30)


@pytest.mark.parametrize("j", [1, 2, 3])
def test_every_intermediate_nn_signed_zero_is_checked(j):
    a, t, d = fixture()
    a[f"neural_next_{j}_encoder_features"][0, 0] = -0.0
    with pytest.raises(ValueError, match="bits"):
        verify_early_branch(a, t, d, 30)


@pytest.mark.parametrize("substep", [0, 9, 10, 19, 20, 29, 30, 39])
def test_does_not_skip_last_two_feedback_periods(substep):
    a, t, d = fixture()
    d["canonical_qpos_before_substep"][301 + substep, 0] = 0.01
    with pytest.raises(ValueError):
        verify_early_branch(a, t, d, 30)


@pytest.mark.parametrize("frame", range(4))
def test_all_four_completed_force_bins_checked(frame):
    a, t, d = fixture()
    a["completed_contact_peak_force_n"][frame * 10 + 9, 0] = 2
    with pytest.raises(ValueError):
        verify_early_branch(a, t, d, 30)


@pytest.mark.parametrize(
    "bad", ["short", "nan", "offset", "typed_flag", "negative_force", "refresh"]
)
def test_raw_schema_and_action_are_not_trusted(bad):
    a, t, d = fixture()
    frame = 30
    if bad == "short":
        a["canonical_qvel_after_substep"] = a["canonical_qvel_after_substep"][:39]
    elif bad == "nan":
        a["end_integration_state"][0] = np.nan
    elif bad == "offset":
        a["requested_offset_nm"][39, 0] = 0.01
    elif bad == "typed_flag":
        a["nonfoot_robot_contact_over1n_after_substep"] = np.zeros(40)
    elif bad == "negative_force":
        a["completed_foot_ground_normal_n"][0, 0] = -1
    else:
        frame = 37
    with pytest.raises(ValueError):
        verify_early_branch(a, t, d, frame)
