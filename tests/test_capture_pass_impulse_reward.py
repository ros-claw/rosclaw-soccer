import numpy as np
import pytest
from test_nonfoot_impact_reward import fixture

from rosclaw_soccer.growth.near_ball_residual import NearBallResidualPolicy
from rosclaw_soccer.growth.role_self_model import TacticalIntent
from rosclaw_soccer.training import near_ball_residual_ppo as trainer


def passing_trace():
    ids, trace = fixture()
    i = 3
    trace["blue_0_intent_code"][i] = list(TacticalIntent).index(TacticalIntent.PASS)
    trace["blue_0_target_position"][i, :2] = (2.0, 0.0)
    trace["ball_pose"][i, :2] = (0.0, 0.0)
    trace["ball_contact_agent_code"][i] = 1
    trace["ball_contact_effector_code"][i] = 1
    trace["ball_contact_force_n"][i] = 5.0
    trace["pass_source_agent_code"] = trace["pass_source_agent_code"].astype(int)
    trace["pass_target_agent_code"] = trace["pass_target_agent_code"].astype(int)
    trace["pass_source_agent_code"][i] = 1
    trace["pass_target_agent_code"][i] = 2
    return ids, trace, i


def difference(ids, trace):
    return trainer.physical_rewards(
        trace, ids, reward_shaping="team_capture_in_play_pass_impulse_v1"
    ) - trainer.physical_rewards(trace, ids, reward_shaping="team_capture_in_play_impact_v1")


def test_preexisting_velocity_does_not_earn_pass_credit():
    ids, trace, i = passing_trace()
    trace["ball_control_entry_velocity"][i, 0] = 1.0
    trace["ball_velocity"][i, 0] = 1.0
    result = difference(ids, trace)
    assert result[i, 0] == pytest.approx(-0.04)
    assert np.count_nonzero(np.abs(result) > 1e-12) == 1


def test_actual_forward_and_wrong_direction_velocity_changes():
    ids, trace, i = passing_trace()
    trace["ball_velocity"][i, 0] = 1.0
    assert difference(ids, trace)[i, 0] == pytest.approx(0.0)
    trace["ball_control_entry_velocity"][i, 0] = 2.0
    assert difference(ids, trace)[i, 0] == pytest.approx(-0.08)


@pytest.mark.parametrize("fault", ["missing_receiver", "nonfoot", "wrong_actor"])
def test_unattributed_contact_has_no_impulse_credit(fault):
    ids, trace, i = passing_trace()
    trace["ball_velocity"][i, 0] = 1.0
    if fault == "missing_receiver":
        trace["pass_target_agent_code"][i] = 0
    elif fault == "nonfoot":
        trace["ball_nonfoot_contact_agent_code"][i] = 2
        trace["ball_nonfoot_contact_force_n"][i] = 2.0
    else:
        trace["pass_source_agent_code"][i] = 3
    assert difference(ids, trace)[i, 0] == pytest.approx(-0.04)


def test_loaded_source_mismatch_fails_closed(monkeypatch):
    from rosclaw_soccer.training import pass_impulse_reward

    monkeypatch.setattr(pass_impulse_reward, "LOADED_SOURCE_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="differs from disk"):
        trainer._pass_impulse_source_hash()


def test_new_combined_reward_updates_actor_and_binds_source(monkeypatch):
    pytest.importorskip("torch")
    ids, trace, i = passing_trace()
    trace["ball_velocity"][i, 0] = -1.0
    parent = NearBallResidualPolicy.initialize(ids, "sha256:" + "a" * 64)
    hashes = []
    for source in ("a", "b"):
        monkeypatch.setattr(
            trainer, "_pass_impulse_source_hash", lambda source=source: "sha256:" + source * 64
        )
        child, rows = trainer.update_private_actors(
            parent, [trace], reward_shaping="team_capture_in_play_pass_impulse_v1"
        )
        assert child.parent_hash == parent.policy_hash
        assert rows[0]["updated"]
        hashes.append(rows[0]["core_plasticity"]["lease"]["scenario_contract_hash"])
    assert hashes[0] != hashes[1]
