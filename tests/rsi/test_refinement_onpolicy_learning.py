import numpy as np
import pytest
from test_joint_delivery_teacher import row

from rosclaw_soccer.rsi.refinement_onpolicy_learning import (
    KIND,
    PEER_KIND,
    config_for_kind,
    joint_delivery_reward,
    kind_for_config,
    learning_config,
    observed_terminal_returns,
)


def test_joint_reward_does_not_hide_precision_or_speed_behind_receive_bonus():
    good = joint_delivery_reward(row(0.05, 2.7)["outcome"])
    fast = joint_delivery_reward(row(0.03, 3.6)["outcome"])
    near_slow = joint_delivery_reward(row(0.08, 2.9, received=False)["outcome"])
    slow_miss = joint_delivery_reward(row(0.8, 2.7)["outcome"])
    assert good > near_slow > fast > slow_miss
    assert 0 < slow_miss < 3.25


def test_body_failure_and_missing_recovery_are_actual_negative_learning_targets():
    x = row(0.05, 2.7)["outcome"]
    x["whole_body_healthy"] = False
    assert joint_delivery_reward(x) == -3
    x["whole_body_healthy"] = True
    x["passer_full100NNrecovery"] = False
    assert joint_delivery_reward(x) == -3


def test_actual_neural_tail_delays_reward_without_holding_or_padding_motion():
    returns = observed_terminal_returns(289, 100, 1.0)
    assert returns.shape == (289,) and returns.dtype == np.float32
    assert returns[-1] == pytest.approx(0.995**100)
    assert returns[0] == pytest.approx(0.995**388)
    failure = observed_terminal_returns(7, 0, -3.0)
    assert failure[-1] == -3 and len(failure) == 7


@pytest.mark.parametrize(
    "n,tail,reward",
    [(0, 0, 1), (301, 0, 1), (12, 101, 1), (True, 0, 1), (12, 0, float("nan")), (12, 0, 3.26)],
)
def test_invalid_or_invented_terminal_clock_rejected(n, tail, reward):
    with pytest.raises(ValueError):
        observed_terminal_returns(n, tail, reward)


def test_fixed_small_onpolicy_generation_and_no_success_claim():
    c = learning_config(25)
    assert c["epochs"] == 4 and c["source_batch_generations"] == 1
    assert c["max_conditional_kl"] == 0.015 and c["source_batch_worlds"] == 24
    assert c["physical_retention_proven"] is c["continuous_learning_proven"] is False


def test_peer_variant_changes_only_credit_assignment_and_has_distinct_kind():
    old, new = learning_config(25), learning_config(25, variant="peer-loo-v2")
    assert {k for k in old if old[k] != new[k]} == {"protocol", "advantage_rule"}
    assert kind_for_config(old) == KIND and kind_for_config(new) == PEER_KIND
    assert config_for_kind(PEER_KIND, 25) == new
    assert config_for_kind(KIND, 25) == old
    with pytest.raises(ValueError):
        kind_for_config(new | dict(epochs=8))
    with pytest.raises(ValueError):
        learning_config(25, variant="unreviewed")
    with pytest.raises(ValueError):
        config_for_kind("consolidated-refinement-v1", 25)
