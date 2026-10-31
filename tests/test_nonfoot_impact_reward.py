import numpy as np
import pytest
from test_capture_in_play_reward import fixture as base_fixture

from rosclaw_soccer.growth.near_ball_residual import NearBallResidualPolicy
from rosclaw_soccer.growth.role_self_model import TacticalIntent
from rosclaw_soccer.training.near_ball_residual_ppo import physical_rewards, update_private_actors
from rosclaw_soccer.training.nonfoot_impact_reward import nonfoot_impact_adjustment


def fixture():
    ids, trace = base_fixture()
    trace["ball_nonfoot_contact_agent_code"] = trace["ball_nonfoot_contact_agent_code"].astype(int)
    return ids, trace


def test_high_force_and_speed_change_are_bounded_and_attributed():
    ids, trace = fixture()
    trace["blue_1_intent_code"][3] = list(TacticalIntent).index(TacticalIntent.SHOOT)
    trace["ball_nonfoot_contact_agent_code"][3] = 2
    trace["ball_nonfoot_contact_force_n"][3] = 251.7
    trace["ball_velocity"][3, 1] = -3.3
    before = {key: value.copy() for key, value in trace.items()}
    result = nonfoot_impact_adjustment(trace, ids)
    assert result[3, 1] == pytest.approx(-0.4)
    assert np.count_nonzero(result) == 1
    base = physical_rewards(trace, ids, reward_shaping="team_capture_dissipation_in_play_v1")
    actual = physical_rewards(trace, ids, reward_shaping="team_capture_in_play_impact_v1")
    np.testing.assert_allclose(actual, base + result)
    for key in trace:
        np.testing.assert_array_equal(trace[key], before[key])


def test_ambiguous_foot_contact_removes_speed_component():
    ids, trace = fixture()
    trace["ball_nonfoot_contact_agent_code"][3] = 2
    trace["ball_nonfoot_contact_force_n"][3] = 50
    trace["ball_velocity"][3, 1] = -3
    trace["ball_contact_force_n"][3] = 2
    assert nonfoot_impact_adjustment(trace, ids)[3, 1] == pytest.approx(-0.075)


def test_new_objective_drives_a_real_on_policy_update():
    pytest.importorskip("torch")
    ids, trace = fixture()
    parent = NearBallResidualPolicy.initialize(ids, "sha256:" + "a" * 64)
    trace["ball_nonfoot_contact_agent_code"][3] = 1
    trace["ball_nonfoot_contact_force_n"][3] = 50
    trace["ball_velocity"][3, 1] = -1.0
    child, rows = update_private_actors(
        parent, [trace], reward_shaping="team_capture_in_play_impact_v1"
    )
    assert child.parent_hash == parent.policy_hash
    assert rows[0]["updated"]
    assert not any(row["updated"] for row in rows[1:])


def test_impact_source_hash_is_bound_to_optimizer_lease(monkeypatch):
    pytest.importorskip("torch")
    from rosclaw_soccer.training import near_ball_residual_ppo as trainer

    ids, trace = fixture()
    parent = NearBallResidualPolicy.initialize(ids, "sha256:" + "a" * 64)
    hashes = []
    weights = []
    for value in ("a", "b"):
        monkeypatch.setattr(
            trainer, "_impact_reward_source_hash", lambda value=value: "sha256:" + value * 64
        )
        child, rows = trainer.update_private_actors(
            parent, [trace], reward_shaping="team_capture_in_play_impact_v1"
        )
        hashes.append(rows[0]["core_plasticity"]["lease"]["scenario_contract_hash"])
        weights.append(child.weights)
    assert hashes[0] != hashes[1]
    for key in weights[0]:
        np.testing.assert_array_equal(weights[0][key], weights[1][key])


def test_stale_loaded_impact_source_fails_closed(monkeypatch):
    from rosclaw_soccer.training import near_ball_residual_ppo, nonfoot_impact_reward

    monkeypatch.setattr(nonfoot_impact_reward, "LOADED_SOURCE_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="differs from disk"):
        near_ball_residual_ppo._impact_reward_source_hash()


def test_goalkeeper_and_unlisted_tasks_are_not_additionally_penalized():
    ids, trace = fixture()
    keeper_ids = tuple(a.replace("blue.3", "blue.goalkeeper") for a in ids)
    trace["blue_goalkeeper_intent_code"] = trace.pop("blue_3_intent_code")
    trace["ball_nonfoot_contact_agent_code"][3] = 4
    trace["ball_nonfoot_contact_force_n"][3] = 200
    trace["ball_velocity"][3, 1] = -3
    assert not nonfoot_impact_adjustment(trace, keeper_ids).any()
    trace["ball_nonfoot_contact_agent_code"][3] = 2
    trace["blue_1_intent_code"][3] = list(TacticalIntent).index(TacticalIntent.SAVE)
    assert not nonfoot_impact_adjustment(trace, keeper_ids).any()


@pytest.mark.parametrize("fault", ["missing_actor", "bad_code", "nan", "negative", "bad_intent"])
def test_incomplete_or_malformed_evidence_rejected(fault):
    ids, trace = fixture()
    if fault == "missing_actor":
        trace["ball_nonfoot_contact_force_n"][0] = 2
    elif fault == "bad_code":
        trace["ball_nonfoot_contact_agent_code"][0] = 9
    elif fault == "nan":
        trace["ball_control_entry_velocity"][0, 0] = np.nan
    elif fault == "negative":
        trace["ball_nonfoot_contact_force_n"][0] = -1
    else:
        trace["blue_1_intent_code"][0] = 1000
    with pytest.raises(ValueError):
        nonfoot_impact_adjustment(trace, ids)
