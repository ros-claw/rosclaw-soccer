import numpy as np
import pytest

from rosclaw_soccer.training.receiving_context import ReceivingContext, receiving_context


def trace():
    pose = np.zeros((5, 7))
    pose[:, 2:4] = [0.75, 1]
    ball = pose.copy()
    ball[:, :3] = [1, 0.2, 0.115]
    velocity = np.zeros((5, 6))
    velocity[:, 0] = -0.5
    return {
        "time": np.arange(1, 6) * 0.02,
        "red_playmaker_pelvis_pose": pose,
        "ball_pose": ball,
        "ball_velocity": velocity,
        "ball_contact_agent_code": np.zeros(5),
        "ball_contact_force_n": np.zeros(5),
        "ball_nonfoot_contact_agent_code": np.zeros(5),
        "ball_nonfoot_contact_force_n": np.zeros(5),
    }


def test_measured_relative_velocity_and_no_future_dependency():
    data = trace()
    data["red_playmaker_pelvis_pose"][1, 0] = 0.01
    result = receiving_context(data, agent_id="red.playmaker", frame=1)
    assert result.ball_forward_m == pytest.approx(0.99)
    assert result.player_speed_mps == pytest.approx(0.5)
    assert result.relative_forward_velocity_mps == pytest.approx(-1)
    assert result.ball_speed_mps == 0.5
    for value in data.values():
        value[2:] = np.nan
    assert receiving_context(data, agent_id="red.playmaker", frame=1) == result


def test_yaw_local_coordinates():
    data = trace()
    data["red_playmaker_pelvis_pose"][:, 3:] = [2**-0.5, 0, 0, 2**-0.5]
    result = receiving_context(data, agent_id="red.playmaker", frame=1)
    assert result.ball_forward_m == pytest.approx(0.2)
    assert result.ball_lateral_m == pytest.approx(-1)
    assert result.relative_lateral_velocity_mps == pytest.approx(0.5)


@pytest.mark.parametrize("fault", ["contact", "nonfoot", "clock", "pose", "nan", "force"])
def test_invalid_or_postcontact_input_rejected(fault):
    data = trace()
    if fault in {"contact", "nonfoot"}:
        key = "ball_contact" if fault == "contact" else "ball_nonfoot_contact"
        data[key + "_agent_code"][0] = 1
        data[key + "_force_n"][0] = 2
    elif fault == "clock":
        data["time"][1] = 0.05
    elif fault == "pose":
        data["red_playmaker_pelvis_pose"][1, 3] = 2
    elif fault == "nan":
        data["ball_velocity"][0, 0] = np.nan
    else:
        data["ball_contact_force_n"][0] = -1
    with pytest.raises(ValueError):
        receiving_context(data, agent_id="red.playmaker", frame=1)


@pytest.mark.parametrize("frame", [0, -1, True, 1.0, 8])
def test_explicit_observation_frame_required(frame):
    with pytest.raises(ValueError):
        receiving_context(trace(), agent_id="red.playmaker", frame=frame)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), True, -1])
def test_context_rejects_invalid_speed(invalid):
    values = receiving_context(trace(), agent_id="red.playmaker", frame=1).to_dict()
    values["ball_speed_mps"] = invalid
    with pytest.raises(ValueError):
        ReceivingContext(**values)
