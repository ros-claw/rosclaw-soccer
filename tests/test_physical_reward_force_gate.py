import numpy as np
import pytest
from test_s215_near_ball_residual import policy, samples

from rosclaw_soccer.training.near_ball_residual_ppo import physical_rewards


def reward_trace():
    actor = policy()
    value = samples(actor)
    value["ball_velocity"][:, 0] = 1.0
    for identity in actor.agent_ids:
        value[identity.replace(".", "_") + "_target_position"][:, 0] = 3.0
    value["ball_contact_agent_code"][0] = 1
    value["ball_contact_effector_code"][0] = 1
    value["ball_contact_force_n"][0] = 0.0
    return actor, value


def test_zero_force_foot_record_cannot_earn_directed_ball_reward():
    actor, trace = reward_trace()
    zero_force = physical_rewards(trace, actor.agent_ids)
    trace["ball_contact_agent_code"][0] = 0
    trace["ball_contact_effector_code"][0] = 0
    absent = physical_rewards(trace, actor.agent_ids)
    np.testing.assert_allclose(zero_force, absent)
    trace["ball_contact_agent_code"][0] = 1
    trace["ball_contact_effector_code"][0] = 1
    trace["ball_contact_force_n"][0] = 2.0
    forceful = physical_rewards(trace, actor.agent_ids)
    assert forceful[0, 0] > absent[0, 0]


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -1.0])
def test_invalid_contact_force_is_rejected(invalid):
    actor, trace = reward_trace()
    trace["ball_contact_force_n"][0] = invalid
    with pytest.raises(ValueError, match="contact force"):
        physical_rewards(trace, actor.agent_ids)


@pytest.mark.parametrize("weak_frame", [20, 45])
def test_weak_launch_or_reception_cannot_earn_pass_credit(weak_frame):
    from test_s224_launch_credit import delayed_pass_trace

    ids, trace = delayed_pass_trace()
    trace["pass_feedback_launch_relative"] = np.ones(len(trace["time"]), dtype=bool)
    trace["ball_contact_force_n"][weak_frame] = 1.0
    weak = physical_rewards(trace, ids)
    trace["pass_source_agent_code"][:] = 0
    trace["pass_target_agent_code"][:] = 0
    no_commitment = physical_rewards(trace, ids)
    np.testing.assert_allclose(weak, no_commitment)


def test_ppo_update_hash_commits_to_reward_contract(monkeypatch):
    pytest.importorskip("torch")
    import rosclaw_soccer.training.near_ball_residual_ppo as training

    actor = policy()
    observed = []
    original = training.hash_json

    def capture(value):
        if isinstance(value, dict) and "dataset" in value:
            observed.append(value)
        return original(value)

    monkeypatch.setattr(training, "hash_json", capture)
    training.update_private_actors(actor, [samples(actor)])
    assert observed[0]["physical_reward_contract"] == training.PHYSICAL_REWARD_CONTRACT
