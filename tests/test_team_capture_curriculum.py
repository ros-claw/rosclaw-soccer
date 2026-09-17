import numpy as np
import pytest
from test_receiving_rollout import contact, trace

from rosclaw_soccer.growth.role_self_model import TacticalIntent
from rosclaw_soccer.training.team_capture_curriculum import team_capture_trials

IDS = ("blue.other", "red.receiver")


def fixture():
    value = trace()
    for suffix in ("_left_foot_position", "_right_foot_position"):
        value["blue_other" + suffix] = value["red_receiver" + suffix] + 10
    for agent in IDS:
        value[agent.replace(".", "_") + "_intent_code"] = np.full(
            101, list(TacticalIntent).index(TacticalIntent.RECEIVE), dtype=int
        )
    return value


def evaluate(value):
    return team_capture_trials(value, agent_ids=IDS)


def test_admission_does_not_require_success_or_touch():
    value = fixture()
    reward, trials = evaluate(value)
    assert len(trials) == 1
    assert not trials[0]["controlled_reception"]
    assert trials[0]["first_foot_contact_sec"] is None
    assert reward[:, 1].sum() < 0
    contact(value)
    good, outcomes = evaluate(value)
    assert outcomes[0]["controlled_reception"]
    assert good[:, 1].sum() > reward[:, 1].sum()


def test_task_switch_does_not_erase_capture_failure():
    value = fixture()
    contact(value)
    value["red_receiver_intent_code"][21:] = list(TacticalIntent).index(TacticalIntent.PASS)
    value["ball_velocity"][21:, 0] = 2
    _, outcomes = evaluate(value)
    assert len(outcomes) == 1 and not outcomes[0]["controlled_reception"]
    assert outcomes[0]["frames"] == 100


def test_incomplete_window_is_censored_not_promoted():
    value = fixture()
    contact(value)
    value = {key: array[:70] for key, array in value.items()}
    _, outcomes = evaluate(value)
    assert outcomes[0]["censored"]
    assert not outcomes[0]["controlled_reception"]
    assert not outcomes[0]["promotion_eligible"]


def test_nonoverlap_and_no_input_mutation():
    value = fixture()
    value = {key: np.concatenate([array, array[1:]]) for key, array in value.items()}
    value["time"] = np.arange(1, 202) * 0.02
    contact(value)
    before = {k: v.copy() for k, v in value.items()}
    _, outcomes = evaluate(value)
    assert [x["start_frame"] for x in outcomes] == [1, 101]
    assert all(np.array_equal(value[k], v) for k, v in before.items())


@pytest.mark.parametrize("fault", ["reset", "time", "intent", "feet"])
def test_malformed_or_reset_crossing_rejected(fault):
    value = fixture()
    if fault == "reset":
        value["training_return_event_code"] = np.zeros(101, int)
        value["training_return_event_code"][[0, 50]] = [3, 2]
    if fault == "time":
        value["time"][4] += 0.005
    if fault == "intent":
        value["red_receiver_intent_code"][4] = 999
    if fault == "feet":
        value["red_receiver_left_foot_position"][4, 0] = np.nan
    with pytest.raises(ValueError):
        evaluate(value)


def test_ppo_composes_capture_credit_without_removing_body_penalties():
    from test_s215_near_ball_residual import policy, samples

    from rosclaw_soccer.training.near_ball_residual_ppo import physical_rewards

    parent = policy()
    value = samples(parent)
    n = len(value["time"])
    value["ball_nonfoot_contact_force_n"] = np.zeros(n)
    value["robot_robot_contact_count"] = np.zeros(n)
    value["ball_pose"][:, 3] = 1
    for agent in parent.agent_ids:
        key = agent.replace(".", "_")
        value[key + "_intent_code"] = np.full(
            n, list(TacticalIntent).index(TacticalIntent.RECEIVE), dtype=int
        )
        value[key + "_pelvis_pose"][:, 2] = 0.75
        value[key + "_pelvis_pose"][:, 3:] = 0
        value[key + "_pelvis_pose"][:, 3] = 1
        value[key + "_joint_safety_margin_rad"] = np.full((n, 29), 0.1)
    capture, _ = team_capture_trials(value, agent_ids=parent.agent_ids)
    old = physical_rewards(value, parent.agent_ids, reward_shaping="contact_safety_v1")
    new = physical_rewards(value, parent.agent_ids, reward_shaping="team_capture_v1")
    np.testing.assert_allclose(new, old + capture)
    value["agent_0_pelvis_pose"][:, 2] = 0.4
    unsafe = physical_rewards(value, parent.agent_ids, reward_shaping="team_capture_v1")
    assert unsafe[:, 0].sum() < new[:, 0].sum()
    np.testing.assert_array_equal(unsafe[:, 1:], new[:, 1:])
