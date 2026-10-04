import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.foundation_body_transitions import body_transitions
from rosclaw_soccer.rsi.foundation_observation_capture import capture_contract
from tests.test_s234_sonic_vector import setup


def trace_fixture():
    tracker, _, q, v = setup(capture=True)
    tracker.reset(q, v)
    rows = {}
    for frame in range(4):
        if frame:
            v = v.copy()
            v[:, 6:35] += 0.002 * frame
            tracker.observe(q, v)
        tracker.update(frame, q, v)
        values = {
            "canonical_qpos": q.copy(),
            "canonical_qvel": v.copy(),
            "joint_target_rad": np.full((2, 29), frame + 0.5),
            "actual_actuator_force_nm": np.full((2, 10, 29), frame + 0.1),
            "root_pose_xyzw_m": q[:, :7].copy(),
            "root_velocity_world": v[:, :6].copy(),
            "ball_position_before_step_m": np.full((2, 3), frame + 1.0),
            "ball_linear_velocity_before_step_m_s": np.full((2, 3), frame + 0.3),
        }
        values.update(
            {
                "foundation_neural_" + k: x.numpy().copy()
                for k, x in tracker.neural_observation().items()
            }
        )
        for key, value in values.items():
            rows.setdefault(key, []).append(value)
    return {key: np.asarray(value) for key, value in rows.items()}


def test_actual_controls_next_state_and_no_terminal_fabrication():
    trace = trace_fixture()
    before = copy.deepcopy(trace)
    result = body_transitions(capture_contract(), trace)
    arrays = result["arrays"]
    assert result["transition_pairs"] == 6
    assert result["terminal_body_transitions_omitted"] == 2
    assert not result["policy_gradient_ready"]
    assert not result["source_receipts_replayed_by_this_function"]
    assert arrays["observation"].shape == (3, 2, 1000)
    np.testing.assert_array_equal(
        arrays["current_native_body_qvel"], trace["canonical_qvel"][:-1, :, :35]
    )
    np.testing.assert_array_equal(
        arrays["next_native_body_qvel"], trace["canonical_qvel"][1:, :, :35]
    )
    np.testing.assert_array_equal(
        arrays["applied_virtual_joint_target_rad"], trace["joint_target_rad"][:-1]
    )
    assert not np.array_equal(
        arrays["applied_virtual_joint_target_rad"],
        trace["foundation_neural_joint_target_mujoco_rad"][:-1],
    )
    for key, value in trace.items():
        np.testing.assert_array_equal(value, before[key])
    assert all(not value.flags.writeable for value in arrays.values())


def test_omitted_terminal_ball_sample_does_not_enter_actor_observations():
    trace = trace_fixture()
    baseline = body_transitions(capture_contract(), trace)["arrays"]["observation"]
    trace["ball_position_before_step_m"][-1] += 999
    actual = body_transitions(capture_contract(), trace)["arrays"]["observation"]
    np.testing.assert_array_equal(actual, baseline)


@pytest.mark.parametrize(
    "field", ["joint_target_rad", "actual_actuator_force_nm", "ball_position_before_step_m"]
)
def test_bad_actual_fields_rejected(field):
    trace = trace_fixture()
    trace[field].flat[0] = float("nan")
    with pytest.raises(ValueError, match="finite actual"):
        body_transitions(capture_contract(), trace)
