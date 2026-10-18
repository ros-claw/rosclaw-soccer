import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.pre_parent_reference_features import (
    FEATURE_NAMES,
    actor_features,
    trace_observations,
)
from tests.rsi.test_contact_reference_labels import trace as reference_trace


def measured_trace():
    result = reference_trace()
    for key, shape in (
        ("joint_position_rad", (300, 1, 29)),
        ("joint_velocity_rad_s", (300, 1, 29)),
        ("root_pose_xyzw_m", (300, 1, 7)),
        ("root_velocity_world", (300, 1, 6)),
        ("ball_position_before_step_m", (300, 1, 3)),
        ("ball_linear_velocity_before_step_m_s", (300, 1, 3)),
        ("foot_geometry_position_before_step_m", (300, 1, 4, 3)),
        ("force_n", (300, 1, 6)),
    ):
        result[key] = np.zeros(shape, dtype=np.float64)
    result["root_pose_xyzw_m"][:, :, 6] = 1
    return result


def test_same_state_reference_actor_features_include_deadline_not_own_current_label():
    trace = measured_trace()
    result = trace_observations(trace)
    assert len(FEATURE_NAMES) == 135
    assert result["observation"].shape == (300, 135)
    np.testing.assert_array_equal(result["observation"][:, -1], (300 - np.arange(300)) / 300)
    np.testing.assert_array_equal(
        result["same_state_reference_target_rad"], trace["taskspace_joint_increment_rad"][:, 0, :12]
    )
    other = copy.deepcopy(trace)
    other["pre_motor_joint_target_rad"][40, 0, 0] += 0.01
    other["taskspace_joint_increment_rad"] = (
        other["pre_motor_joint_target_rad"] - other["foundation_navigation_joint_target_rad"]
    )
    changed = trace_observations(other)
    np.testing.assert_array_equal(result["observation"][:41], changed["observation"][:41])
    assert not np.array_equal(result["observation"][41], changed["observation"][41])
    assert not np.array_equal(
        result["same_state_reference_target_rad"][40],
        changed["same_state_reference_target_rad"][40],
    )


def test_current_force_future_state_teacher_id_and_outcomes_cannot_change_current_input():
    trace = measured_trace()
    before = trace_observations(trace)["observation"]
    trace["force_n"][40, 0, 0] = 12
    trace["ball_position_before_step_m"][41:, 0, 0] = 4
    trace["terminal_return"] = 999
    trace["teacher_family"] = "oracle"
    trace["future_contact_frame"] = 100
    after = trace_observations(trace)["observation"]
    np.testing.assert_array_equal(before[:41], after[:41])
    assert not np.array_equal(before[41], after[41])


def test_past_reference_is_not_parent_motor_action_and_first_force_is_unavailable():
    trace = measured_trace()
    baseline = trace_observations(trace)["observation"]
    trace["motor_delta_rad"] = np.ones((300, 1, 12)) * 0.1
    np.testing.assert_array_equal(baseline, trace_observations(trace)["observation"])
    first = actor_features(
        trace,
        frame=0,
        navigation_target=trace["foundation_navigation_joint_target_rad"][0, 0],
        previous_applied_reference=np.zeros(12),
    )
    force_ids = [
        i for i, name in enumerate(FEATURE_NAMES) if name.startswith("previous_ball_body_force:")
    ]
    assert not first[force_ids].any()


@pytest.mark.parametrize("tamper", ["shape", "dtype", "nan", "orientation", "force"])
def test_no_repairs_or_coercion_of_original_measured_state(tamper):
    trace = measured_trace()
    if tamper == "shape":
        trace["force_n"] = trace["force_n"][:299]
    elif tamper == "dtype":
        trace["root_velocity_world"] = trace["root_velocity_world"].astype(np.float32)
    elif tamper == "nan":
        trace["joint_position_rad"][100, 0, 0] = np.nan
    elif tamper == "orientation":
        trace["root_pose_xyzw_m"][100, 0, 6] = 0
    else:
        trace["force_n"][50, 0, 0] = -1
    with pytest.raises(ValueError):
        trace_observations(trace)
