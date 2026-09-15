import numpy as np
import pytest
from test_receiving_rollout import contact, trace

from rosclaw_soccer.training.receiving_transition_credit import receiving_transition_window


def evaluate(value, target=(10.0, 0.0)):
    return receiving_transition_window(
        value,
        agent_ids=("blue.other", "red.receiver"),
        agent_id="red.receiver",
        start=1,
        frames=100,
        required_frames=100,
        next_target_xy=target,
    )


def test_transition_credit_keeps_capture_gate_and_never_moves_the_world():
    value = trace()
    contact(value)
    value["red_receiver_pelvis_pose"][:, :2] = value["ball_pose"][:, :2] - (0.4, 0.0)
    before = {k: v.copy() for k, v in value.items()}
    good, outcome = evaluate(value)
    poor = {k: v.copy() for k, v in value.items()}
    poor["red_receiver_pelvis_pose"][:, 0] += 0.7
    bad, bad_outcome = evaluate(poor)
    assert good.sum() > bad.sum()
    assert outcome["controlled_reception"] == bad_outcome["controlled_reception"]
    assert not outcome["promotion_eligible"] and not outcome["transition_success_claim"]
    assert all(np.array_equal(value[k], v) for k, v in before.items())


@pytest.mark.parametrize("target", [(float("nan"), 0), (0,), (float("inf"), 2)])
def test_invalid_target_is_rejected(target):
    with pytest.raises(ValueError):
        evaluate(trace(), target)


def test_rotated_geometry_has_same_credit():
    value = trace()
    contact(value)
    rotated = {k: v.copy() for k, v in value.items()}
    for key in (
        "ball_pose",
        "ball_velocity",
        "red_receiver_pelvis_pose",
        "red_receiver_left_foot_position",
        "red_receiver_right_foot_position",
    ):
        rotated[key][:, :2] *= -1
    rotated["red_receiver_pelvis_pose"][:, 3:7] = (0, 0, 0, 1)
    np.testing.assert_array_equal(evaluate(value)[0], evaluate(rotated, (-10.0, 0.0))[0])
