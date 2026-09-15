import numpy as np
from test_receiving_rollout import contact, trace

from rosclaw_soccer.training.receiving_capture_credit import capture_retention_window
from rosclaw_soccer.training.receiving_rollout import receiving_window


def evaluate(value, function=capture_retention_window):
    return function(
        value,
        agent_ids=("blue.other", "red.receiver"),
        agent_id="red.receiver",
        start=1,
        frames=100,
    )


def test_kicking_ball_away_is_worse_than_retaining_it_and_gate_is_unchanged():
    good = trace()
    contact(good)
    kicked = {k: v.copy() for k, v in good.items()}
    kicked["ball_velocity"][21:, 0] = 2.2
    far = {k: v.copy() for k, v in good.items()}
    far["red_receiver_left_foot_position"][21:, 0] += 1
    far["red_receiver_right_foot_position"][21:, 0] += 1
    assert evaluate(good)[0].sum() > evaluate(kicked)[0].sum()
    assert evaluate(good)[0].sum() > evaluate(far)[0].sum()
    for value in (good, kicked, far, trace()):
        before = {k: v.copy() for k, v in value.items()}
        _, old = evaluate(value, receiving_window)
        reward, new = evaluate(value)
        assert new["controlled_reception"] == old["controlled_reception"]
        assert new["promotion_eligible"] is False
        assert new["shaped_return"] == float(reward.sum())
        assert new["dense_distance_speed_cost"] > 0
        assert all(np.array_equal(value[k], v) for k, v in before.items())
