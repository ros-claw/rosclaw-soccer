import numpy as np
import pytest
from test_capture_pass_impulse_reward import passing_trace

from rosclaw_soccer.growth.near_ball_residual import NearBallResidualPolicy
from rosclaw_soccer.training.near_ball_residual_ppo import physical_rewards, update_private_actors
from rosclaw_soccer.training.normal_pass_impulse_reward import pass_normal_adjustment


def fixture():
    ids, trace, i = passing_trace()
    n = len(trace["time"])
    trace["contact_normal_impulse_ns"] = np.zeros((n, 8, 3))
    trace["contact_normal_substeps"] = np.zeros((n, 8), dtype=int)
    trace["contact_normal_ambiguous"] = np.zeros((n, 8), dtype=bool)
    trace["contact_normal_world_safe"] = np.ones(n, dtype=bool)
    trace["ball_velocity"][i, 0] = 1.0
    return ids, trace, i


def test_preexisting_ball_speed_does_not_earn_pass_credit():
    ids, trace, i = fixture()
    correction = pass_normal_adjustment(trace, ids)
    assert correction[i, 0] == pytest.approx(-0.04)
    assert np.count_nonzero(correction) == 1


def test_direction_and_lateral_penalty_from_measured_impulse():
    ids, trace, i = fixture()
    trace["contact_normal_substeps"][i, 0] = 2
    trace["contact_normal_impulse_ns"][i, 0, :2] = [0.43, 0.215]
    assert pass_normal_adjustment(trace, ids)[i, 0] == pytest.approx(-0.02)
    trace["contact_normal_impulse_ns"][i, 0, :2] = [-0.43, 0]
    assert pass_normal_adjustment(trace, ids)[i, 0] == pytest.approx(-0.08)


@pytest.mark.parametrize("fault", ["ambiguous", "unsafe", "wrong_sender", "no_receiver"])
def test_unattributed_or_unsafe_impulses_receive_no_credit(fault):
    ids, trace, i = fixture()
    trace["contact_normal_substeps"][i, 0] = 2
    trace["contact_normal_impulse_ns"][i, 0, 0] = 0.43
    if fault == "ambiguous":
        trace["contact_normal_ambiguous"][i, 0] = True
    elif fault == "unsafe":
        trace["contact_normal_world_safe"][i] = False
    elif fault == "wrong_sender":
        trace["pass_source_agent_code"][i] = 3
    else:
        trace["pass_target_agent_code"][i] = 0
    assert pass_normal_adjustment(trace, ids)[i, 0] == pytest.approx(-0.04)


@pytest.mark.parametrize("fault", ["missing", "nonfinite", "count", "orphan_impulse"])
def test_malformed_physical_evidence_rejected(fault):
    ids, trace, i = fixture()
    if fault == "missing":
        del trace["contact_normal_ambiguous"]
    elif fault == "nonfinite":
        trace["contact_normal_impulse_ns"][i, 0, 0] = np.nan
    elif fault == "count":
        trace["contact_normal_substeps"][i, 0] = 11
    else:
        trace["contact_normal_impulse_ns"][i, 0, 0] = 0.1
    with pytest.raises((ValueError, KeyError)):
        pass_normal_adjustment(trace, ids)


def test_normal_objective_preserves_base_and_updates_private_actor():
    pytest.importorskip("torch")
    ids, trace, i = fixture()
    trace["contact_normal_substeps"][i, 0] = 1
    trace["contact_normal_impulse_ns"][i, 0, 0] = -0.43
    actual = physical_rewards(trace, ids, reward_shaping="team_capture_in_play_normal_impulse_v1")
    base = physical_rewards(trace, ids, reward_shaping="team_capture_in_play_impact_v1")
    np.testing.assert_allclose(actual, base + pass_normal_adjustment(trace, ids))
    parent = NearBallResidualPolicy.initialize(ids, "sha256:" + "a" * 64)
    child, rows = update_private_actors(
        parent, [trace], reward_shaping="team_capture_in_play_normal_impulse_v1"
    )
    assert child.parent_hash == parent.policy_hash
    assert rows[0]["updated"]
    assert not any(row["updated"] for row in rows[1:])


def test_normal_source_hashes_bound_to_update_lease(monkeypatch):
    pytest.importorskip("torch")
    from rosclaw_soccer.training import near_ball_residual_ppo as trainer

    ids, trace, _ = fixture()
    parent = NearBallResidualPolicy.initialize(ids, "sha256:" + "a" * 64)
    contexts = []
    for digit in ("a", "b"):
        monkeypatch.setattr(
            trainer,
            "_normal_impulse_source_hashes",
            lambda digit=digit: {
                "normal_contact_recorder_source_hash": "sha256:" + digit * 64,
                "normal_pass_reward_source_hash": "sha256:" + "c" * 64,
            },
        )
        _, rows = trainer.update_private_actors(
            parent, [trace], reward_shaping="team_capture_in_play_normal_impulse_v1"
        )
        contexts.append(rows[0]["core_plasticity"]["lease"]["scenario_contract_hash"])
    assert contexts[0] != contexts[1]


def test_normal_source_drift_fails_closed(monkeypatch):
    from rosclaw_soccer.training import contact_normal_impulse, near_ball_residual_ppo

    monkeypatch.setattr(contact_normal_impulse, "LOADED_SOURCE_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="differs from disk"):
        near_ball_residual_ppo._normal_impulse_source_hashes()
