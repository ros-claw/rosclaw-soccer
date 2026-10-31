import numpy as np
import pytest
from test_s215_near_ball_residual import policy, samples

from rosclaw_soccer.growth.role_self_model import TacticalIntent
from rosclaw_soccer.training.near_ball_residual_ppo import update_private_actors
from rosclaw_soccer.training.receiving_learning_scope import receiving_learning_scope
from rosclaw_soccer.training.role_sample_coverage import role_sample_coverage


def fixture():
    parent = policy()
    trace = samples(parent)
    n = len(trace["time"])
    for agent in parent.agent_ids:
        trace[agent.replace(".", "_") + "_intent_code"] = np.full(
            n, list(TacticalIntent).index(TacticalIntent.SHOOT), dtype=int
        )
    trace["agent_0_intent_code"][:32] = list(TacticalIntent).index(TacticalIntent.RECEIVE)
    return parent, trace


def test_mask_is_causal_and_coverage_intersects_physical_activation():
    parent, trace = fixture()
    copied = {k: v.copy() for k, v in trace.items()}
    scoped = receiving_learning_scope(trace, agent_ids=parent.agent_ids)
    assert "residual_learning_mask" not in trace
    assert scoped["residual_learning_mask"][:, 0].sum() == 32
    assert not scoped["residual_learning_mask"][:, 1:].any()
    assert role_sample_coverage([scoped], agent_ids=parent.agent_ids, required=("agent.0",)) == {
        "agent.0": 32
    }
    for key in trace:
        np.testing.assert_array_equal(trace[key], copied[key])


def test_masked_ppo_updates_only_receiving_actor():
    pytest.importorskip("torch")
    parent, trace = fixture()
    trace["residual_active"][:, 1] = True
    scoped = receiving_learning_scope(trace, agent_ids=parent.agent_ids)
    child, rows = update_private_actors(parent, [scoped])
    assert child.parent_hash == parent.policy_hash
    assert rows[0]["updated"] and rows[0]["learning_samples"] == 32
    assert not any(row["updated"] for row in rows[1:])


@pytest.mark.parametrize("fault", ["existing", "invalid", "float", "missing"])
def test_malformed_or_replaced_mask_is_rejected(fault):
    parent, trace = fixture()
    if fault == "existing":
        trace["residual_learning_mask"] = np.zeros((40, 8), bool)
    elif fault == "invalid":
        trace["agent_0_intent_code"][0] = 999
    elif fault == "float":
        trace["agent_0_intent_code"] = trace["agent_0_intent_code"].astype(float)
    else:
        del trace["agent_0_intent_code"]
    with pytest.raises((ValueError, KeyError)):
        receiving_learning_scope(trace, agent_ids=parent.agent_ids)
