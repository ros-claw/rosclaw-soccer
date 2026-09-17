import numpy as np
import pytest

from rosclaw_soccer.training.receiving_contact_reward import receiving_contact_adjustment


def measurements():
    n = 20
    contact = np.zeros(n, bool)
    contact[0] = True
    return dict(
        time=np.arange(n) * 0.02,
        receiving=np.ones(n, bool),
        foot_contact=contact,
        owns_ball=np.ones(n, bool),
        speed_before=np.ones(n),
        speed_after=np.full(n, 0.2),
        foot_distance=np.full(n, 0.2),
        directed_reward=np.zeros(n),
    )


def test_receiving_slows_ball_and_retention_credit_expires():
    values = measurements()
    values["directed_reward"][0] = -0.02
    adjustment = receiving_contact_adjustment(**values)
    assert adjustment[0] == pytest.approx(0.02 + 0.032 + 0.01)
    assert adjustment[10] == pytest.approx(0.01)
    assert adjustment[-1] == 0


def test_possession_without_physical_foot_contact_earns_nothing():
    values = measurements()
    values["foot_contact"][:] = False
    assert not receiving_contact_adjustment(**values).any()


def test_pass_shoot_or_support_are_not_rewritten():
    values = measurements()
    values["receiving"][:] = False
    assert not receiving_contact_adjustment(**values).any()


@pytest.mark.parametrize(
    "key,value", [("speed_after", float("nan")), ("foot_distance", -1.0), ("time", 0.0)]
)
def test_bad_physical_measurements_rejected(key, value):
    values = measurements()
    values[key][:] = value
    with pytest.raises(ValueError):
        receiving_contact_adjustment(**values)


def test_no_control_credit_for_fast_or_distant_ball():
    for key, value in [("speed_after", 2.0), ("foot_distance", 1.0)]:
        values = measurements()
        values[key][:] = value
        assert receiving_contact_adjustment(**values)[1] == 0


def test_ppo_role_mode_preserves_safety_terms_and_corrects_receive_credit():
    from test_s215_near_ball_residual import policy, samples

    from rosclaw_soccer.training.near_ball_residual_ppo import physical_rewards

    parent = policy()
    trace = samples(parent)
    n = len(trace["time"])
    trace["training_return_after_ball_velocity"] = np.zeros((n, 6))
    trace["training_return_after_ball_velocity"][:, 0] = 1.0
    trace["ball_velocity"][:, 0] = 0.2
    trace["possession_agent_code"] = np.ones(n, int)
    trace["ball_contact_agent_code"][0] = 1
    trace["ball_contact_effector_code"][0] = 1
    trace["ball_contact_force_n"][0] = 2.0
    for i, a in enumerate(parent.agent_ids):
        key = a.replace(".", "_")
        trace[key + "_intent_code"] = np.full(n, 1 if i == 0 else 3, dtype=int)
        trace[key + "_joint_safety_margin_rad"] = np.full((n, 29), 0.5)
        for suffix in ("_left_foot_position", "_right_foot_position"):
            trace[key + suffix][:] = (0.2, 0.0, 0.0)
        trace[key + "_target_position"][:] = (-0.2, 0.0, 0.0)
    old = physical_rewards(trace, parent.agent_ids, reward_shaping="contact_safety_v1")
    new = physical_rewards(trace, parent.agent_ids, reward_shaping="role_receiving_v1")
    assert new[0, 0] - old[0, 0] == pytest.approx(0.008 + 0.032 + 0.01)
    np.testing.assert_array_equal(new[:, 1:], old[:, 1:])
    del trace["training_return_after_ball_velocity"]
    with pytest.raises(ValueError, match="pre-control"):
        physical_rewards(trace, parent.agent_ids, reward_shaping="role_receiving_v1")
