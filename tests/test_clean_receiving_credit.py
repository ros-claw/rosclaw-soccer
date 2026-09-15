import numpy as np
from test_receiving_rollout import contact, trace

from rosclaw_soccer.training.clean_receiving_credit import clean_receiving_transition_window
from rosclaw_soccer.training.receiving_transition_credit import receiving_transition_window


def evaluate(value, function):
    return function(
        value,
        agent_ids=("blue.other", "red.receiver"),
        agent_id="red.receiver",
        start=1,
        frames=100,
        required_frames=100,
        next_target_xy=(10.0, 0.0),
    )


def test_body_contact_before_first_foot_touch_cannot_earn_clean_capture_bonus():
    value = trace()
    contact(value)
    value["ball_nonfoot_contact_agent_code"][10] = 2
    value["ball_nonfoot_contact_force_n"][10] = 3.0
    before = {k: v.copy() for k, v in value.items()}
    legacy, old = evaluate(value, receiving_transition_window)
    reward, outcome = evaluate(value, clean_receiving_transition_window)
    assert old["controlled_reception"] and outcome["controlled_reception"]
    assert not outcome["clean_controlled_reception"]
    assert outcome["window_nonfoot_contact_frames"] == 1
    assert outcome["withheld_capture_bonus"] == 10
    np.testing.assert_array_equal(reward[:-1], legacy[:-1])
    assert reward[-1] == legacy[-1] - 10
    assert not outcome["promotion_eligible"]
    assert all(np.array_equal(value[k], v) for k, v in before.items())


def test_clean_contact_keeps_reward_and_missing_foot_does_not_get_success():
    value = trace()
    contact(value)
    legacy, old = evaluate(value, receiving_transition_window)
    reward, outcome = evaluate(value, clean_receiving_transition_window)
    np.testing.assert_array_equal(reward, legacy)
    assert old["controlled_reception"] and outcome["clean_controlled_reception"]
    assert outcome["withheld_capture_bonus"] == 0
    _, missing = evaluate(trace(), clean_receiving_transition_window)
    assert not missing["clean_controlled_reception"]
