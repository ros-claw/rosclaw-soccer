from pathlib import Path

import pytest
from test_s215_near_ball_residual import policy

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training import nonfoot_impact_reward
from rosclaw_soccer.training.near_ball_learning_audit import _training_context_hash


def test_legacy_context_is_not_retroactively_rewritten():
    parent = policy()
    payload = {
        "body": parent.body_hash,
        "parent": parent.policy_hash,
        "roster": parent.agent_ids,
        "dataset": "sha256:" + "b" * 64,
        "reward_shaping": "team_capture_v1",
        "gamma": 0.997,
    }
    manifest = {"reward_shaping": "team_capture_v1", "credit": {"gamma": 0.997}}
    assert _training_context_hash(manifest, parent, payload["dataset"], None) == hash_json(payload)


def test_current_impact_context_binds_exact_reward_source():
    parent = policy()
    digest = str(hash_bytes(Path(nonfoot_impact_reward.__file__).read_bytes()))
    manifest = {
        "physical_reward_contract": "rosclaw_soccer.physical_reward.force_gate.v2",
        "reward_shaping": "team_capture_in_play_impact_v1",
        "nonfoot_impact_reward_source_hash": digest,
        "credit": {"gamma": 0.997},
    }
    dataset = "sha256:" + "b" * 64
    payload = {
        "body": parent.body_hash,
        "parent": parent.policy_hash,
        "roster": parent.agent_ids,
        "dataset": dataset,
        "physical_reward_contract": manifest["physical_reward_contract"],
        "reward_shaping": manifest["reward_shaping"],
        "gamma": 0.997,
        "nonfoot_impact_reward_source_hash": digest,
    }
    assert _training_context_hash(manifest, parent, dataset, None) == hash_json(payload)
    manifest["nonfoot_impact_reward_source_hash"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="unbound"):
        _training_context_hash(manifest, parent, dataset, None)
    del manifest["nonfoot_impact_reward_source_hash"]
    with pytest.raises(ValueError, match="unbound"):
        _training_context_hash(manifest, parent, dataset, None)


def test_unknown_physical_reward_contract_is_not_accepted():
    with pytest.raises(ValueError, match="unknown physical reward"):
        _training_context_hash({"physical_reward_contract": "unrecognized"}, policy(), "x", None)
