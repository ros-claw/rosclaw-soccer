import copy

import pytest

from scripts.rsi_review_parallel_learning_equivalence import equal_policy_except_bank_binding


def pair():
    a = dict(
        model_hash="serial",
        generation=1,
        residual_layers=[dict(weight=[[0.1]], bias=[0.2])],
        critic_readout=[[0.3]],
        learning_receipt=dict(
            physical_batch_hash="serial-bank",
            learner_parent_hash="same-parent",
            full_batch_loss_history=[0.1, 0.0],
            exact_mean_latent_kl=0.001,
        ),
    )
    b = copy.deepcopy(a)
    b["model_hash"] = "parallel"
    b["learning_receipt"]["physical_batch_hash"] = "parallel-bank"
    return a, b


def test_only_manifest_and_derived_model_hash_may_differ():
    a, b = pair()
    original = copy.deepcopy((a, b))
    assert equal_policy_except_bank_binding(a, b)
    assert (a, b) == original


@pytest.mark.parametrize("field", ["weights", "critic", "loss", "kl", "parent", "generation"])
def test_parallel_numeric_or_lineage_changes_cannot_be_hidden(field):
    a, b = pair()
    if field == "weights":
        b["residual_layers"][0]["weight"][0][0] += 1e-9
    elif field == "critic":
        b["critic_readout"][0][0] += 1e-9
    elif field == "loss":
        b["learning_receipt"]["full_batch_loss_history"][0] += 1e-9
    elif field == "kl":
        b["learning_receipt"]["exact_mean_latent_kl"] += 1e-9
    elif field == "parent":
        b["learning_receipt"]["learner_parent_hash"] = "stale-parent"
    else:
        b["generation"] = 2
    assert not equal_policy_except_bank_binding(a, b)
