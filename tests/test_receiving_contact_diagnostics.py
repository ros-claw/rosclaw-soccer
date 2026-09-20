import numpy as np
import pytest

from rosclaw_soccer.training.receiving_contact_diagnostics import receiving_contact_diagnostics


def trace():
    n = 101
    ball = np.zeros((n, 7))
    ball[:, 2:4] = [0.115, 1]
    body = ball.copy()
    body[:, 2] = 0.75
    return {
        "time": np.arange(n) * 0.02,
        "ball_pose": ball,
        "ball_velocity": np.zeros((n, 6)),
        "blue_player_left_foot_position": ball[:, :3] + [0.15, 0, 0],
        "blue_player_right_foot_position": ball[:, :3] + [0.2, 0, 0],
        "ball_contact_agent_code": np.zeros(n),
        "ball_contact_effector_code": np.zeros(n),
        "ball_contact_force_n": np.zeros(n),
        "ball_nonfoot_contact_agent_code": np.zeros(n),
        "ball_nonfoot_contact_force_n": np.zeros(n),
        "blue_player_pelvis_pose": body,
        "blue_player_joint_safety_margin_rad": np.full((n, 29), 0.1),
        "robot_robot_contact_count": np.zeros(n),
    }


def run(value):
    return receiving_contact_diagnostics(value, agent_ids=("blue.player",), agent_id="blue.player")


def touch(value, frame):
    value["ball_contact_agent_code"][frame] = 1
    value["ball_contact_effector_code"][frame] = 1
    value["ball_contact_force_n"][frame] = 3


def test_proximity_cannot_invent_control_or_readiness():
    result = run(trace())
    assert np.all(result["phase_hint"] == "PRE_CONTACT")
    assert not result["continuous_control_sec"].any()
    assert result["successor_readiness"] is None
    assert result["support_state"] is None and result["center_of_mass"] is None
    assert not result["velocity_observed"][0]


def test_touch_and_actual_continuous_duration():
    value = trace()
    touch(value, 20)
    before = {key: array.copy() for key, array in value.items()}
    result = run(value)
    assert result["phase_hint"][20] == "CONTACT"
    assert result["phase_hint"][45] == "CONTROL"
    assert result["continuous_control_sec"][20] == 0
    assert result["continuous_control_sec"][45] == pytest.approx(0.5)
    assert not result["phase_is_success_certificate"]
    assert all(np.array_equal(before[key], array) for key, array in value.items())


@pytest.mark.parametrize("failure", ["foreign", "unsafe", "fast"])
def test_control_duration_breaks_on_failure(failure):
    value = trace()
    touch(value, 20)
    if failure == "foreign":
        value["ball_nonfoot_contact_agent_code"][55] = 1
        value["ball_nonfoot_contact_force_n"][55] = 2
    elif failure == "unsafe":
        value["blue_player_pelvis_pose"][55, 2] = 0.3
    else:
        value["ball_velocity"][55, 0] = 1
    result = run(value)
    assert result["continuous_control_sec"][55] == 0
    assert result["phase_hint"][55] != "CONTROL"
    if failure != "fast":
        assert not result["clean_contact_history"][56:].any()


def test_persistent_contact_does_not_reset_clean_contact_age():
    value = trace()
    for i in range(20, 51):
        touch(value, i)
    result = run(value)
    assert result["phase_hint"][51] == "CONTROL"
    assert result["contact_age_sec"][51] == pytest.approx(0.62)


def test_velocity_uses_past_only():
    value = trace()
    value["blue_player_left_foot_position"][50:, 0] += 0.1
    result = run(value)
    np.testing.assert_array_equal(result["foot_velocity_mps"][:50], 0)
    assert result["velocity_observed"][1:].all()


def test_reject_nonfinite_evidence():
    value = trace()
    value["ball_velocity"][70, 0] = np.nan
    with pytest.raises(ValueError):
        run(value)


def test_reject_overflow_in_derived_measurement():
    value = trace()
    value["ball_velocity"][70, 0] = 1e308
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="derived"):
        run(value)
