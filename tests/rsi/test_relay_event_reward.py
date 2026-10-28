import numpy as np
import pytest

from rosclaw_soccer.rsi.relay_event_reward import strict_relay_event_rewards


def test_events_preserve_strict_return_at_observed_controls():
    forces = np.zeros((8, 10, 2, 3), np.float64)
    failures = np.zeros((8, 2), np.bool_)
    forces[1, 2, 0, 0] = 11
    forces[2, 1, 1, 0] = 11
    forces[3, 4, 1, 2] = 5
    forces[4, 5, 0, 2] = 5
    failures[5, 1] = True
    rewards, evidence = strict_relay_event_rewards(forces, failures, handoff=True, controls=7)
    np.testing.assert_array_equal(rewards, [0, 0, 0, -1, -1, -1, 0])
    assert evidence["terminal_score"] == -3
    assert evidence["placements"]["receiver_nonfoot_substep"]["observed_control"] == 3
    assert evidence["placements"]["either_body_failure"]["assigned_control"] == 5


def test_post_horizon_failures_are_explicitly_attributed_to_last_sampled_action():
    forces = np.zeros((8, 10, 2, 3), np.float64)
    failures = np.zeros((8, 2), np.bool_)
    forces[6, 9, 1, 2] = 2
    rewards, evidence = strict_relay_event_rewards(forces, failures, handoff=False, controls=3)
    np.testing.assert_array_equal(rewards, [0, 0, -1])
    assert evidence["placements"]["receiver_nonfoot_substep"] == {
        "observed_control": 6,
        "assigned_control": 2,
        "after_sampling_horizon": True,
    }


def test_clean_success_remains_at_last_action_not_first_touch():
    forces = np.zeros((8, 10, 2, 3), np.float64)
    failures = np.zeros((8, 2), np.bool_)
    forces[1, 0, 0, 0] = 11
    forces[2, 0, 1, 1] = 11
    rewards, evidence = strict_relay_event_rewards(forces, failures, handoff=True, controls=5)
    np.testing.assert_array_equal(rewards, [0, 0, 0, 0, 2])
    assert evidence["outcome"]["strict_both_clean_handoff"] is True


@pytest.mark.parametrize("controls", [0, 9, True])
def test_invalid_sample_horizon_fails_closed(controls):
    forces = np.zeros((8, 10, 2, 3), np.float64)
    failures = np.zeros((8, 2), np.bool_)
    with pytest.raises(ValueError):
        strict_relay_event_rewards(forces, failures, handoff=False, controls=controls)
