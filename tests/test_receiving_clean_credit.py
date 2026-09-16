import numpy as np
import pytest
from test_receiving_capture_credit import evaluate
from test_receiving_rollout import contact, trace

from rosclaw_soccer.training.receiving_capture_credit import capture_retention_window
from rosclaw_soccer.training.receiving_clean_credit import clean_capture_credit_window


def assess(value):
    return evaluate(value, clean_capture_credit_window)


def test_clean_success_preserves_reward_and_physical_success_exactly():
    value = trace()
    contact(value)
    old, old_outcome = evaluate(value)
    new, outcome = assess(value)
    np.testing.assert_array_equal(old, new)
    assert outcome["controlled_reception"] is old_outcome["controlled_reception"] is True
    assert outcome["terminal_criterion_cost"] == 0
    assert outcome["removed_ineligible_hold_bonus_frames"] == 0


@pytest.mark.parametrize("failure", ["nonfoot", "other", "joint", "collision", "height"])
def test_irreversible_failure_never_resumes_hold_bonus(failure):
    value = trace()
    contact(value)
    if failure == "nonfoot":
        value["ball_nonfoot_contact_agent_code"][25] = 2
        value["ball_nonfoot_contact_force_n"][25] = 1
    elif failure == "other":
        value["ball_contact_agent_code"][25] = 1
        value["ball_contact_force_n"][25] = 1
    elif failure == "joint":
        value["red_receiver_joint_safety_margin_rad"][25, 0] = -0.01
    elif failure == "collision":
        value["robot_robot_contact_count"][25] = 1
    else:
        value["red_receiver_pelvis_pose"][25, 2] = 0.4
    original = {k: v.copy() for k, v in value.items()}
    old, old_outcome = evaluate(value)
    new, outcome = assess(value)
    assert outcome["controlled_reception"] is old_outcome["controlled_reception"] is False
    assert outcome["irreversible_failure_cost"] == 8
    assert outcome["removed_ineligible_hold_bonus_frames"] == 75
    np.testing.assert_allclose(old[25:-1] - new[25:-1], 0.04, atol=1e-7)
    assert float(old.sum() - new.sum()) == pytest.approx(8 + 75 * 0.04, abs=1e-5)
    for key in value:
        np.testing.assert_array_equal(value[key], original[key])


def test_nonfoot_before_first_foot_does_not_invent_new_failure():
    value = trace()
    contact(value)
    value["ball_nonfoot_contact_agent_code"][10] = 2
    value["ball_nonfoot_contact_force_n"][10] = 1
    old, previous = evaluate(value)
    new, outcome = assess(value)
    assert outcome["controlled_reception"] is previous["controlled_reception"] is True
    np.testing.assert_array_equal(old, new)


def test_missing_contact_never_becomes_success_or_stale_bonus():
    value = trace()
    old, _ = evaluate(value)
    new, outcome = assess(value)
    assert not outcome["controlled_reception"]
    assert outcome["missing_contact_cost"] == 8
    assert outcome["removed_ineligible_hold_bonus_frames"] == 0
    assert old[-1] - new[-1] == pytest.approx(8)


@pytest.mark.parametrize("criterion", ["distance", "speed", "late"])
def test_terminal_cost_matches_failed_criterion_without_changing_success(criterion):
    value = trace()
    contact(value, 99 if criterion == "late" else 20)
    if criterion == "distance":
        for suffix in ("left", "right"):
            value[f"red_receiver_{suffix}_foot_position"][-1, 0] = 0.7
    elif criterion == "speed":
        value["ball_velocity"][-1, 0] = 0.7
    _, old = evaluate(value)
    reward, outcome = assess(value)
    assert outcome["controlled_reception"] is old["controlled_reception"] is False
    field = {
        "distance": "tail_distance_excess_cost",
        "speed": "tail_speed_excess_cost",
        "late": "post_contact_observation_cost",
    }[criterion]
    assert outcome[field] > 0
    assert outcome["shaped_return"] == float(reward.sum())
    assert outcome["promotion_eligible"] is False


def test_bad_physical_input_still_rejected():
    value = trace()
    value["ball_velocity"][5, 0] = np.nan
    with pytest.raises(ValueError):
        assess(value)


def test_original_objective_is_not_mutated():
    value = trace()
    before = evaluate(value, capture_retention_window)
    assess(value)
    after = evaluate(value, capture_retention_window)
    np.testing.assert_array_equal(before[0], after[0])
    assert before[1] == after[1]
