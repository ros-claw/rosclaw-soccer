"""Synthetic label partitioning; no physical or model-performance evidence."""

import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.postcontact_learning_observation import observation_contract
from rosclaw_soccer.rsi.postcontact_outcome_dataset import masked_brier, measured_dataset


def synthetic_readback():
    rows = []
    for i in range(18):
        observed = i < 8 or i >= 12
        rows.append(
            dict(
                context_index=i,
                partition="CONSUMED_LOCAL_DEVELOPMENT" if i < 12 else "CONSUMED_ORIGINAL_RETENTION",
                source_cluster=i // 4 if i < 12 else (i - 12) // 2,
                training_authorized=False,
                observation_available=observed,
                current_features=[i / 18.0] * 121 if observed else None,
                observed_trigger_frame=150 if observed else None,
                candidate_available=[observed] * 13,
                observed_action_task_pass=[False] * 13,
                observed_action_safe=[True] * 13,
                observed_action_quality_measured=[i % 2 == 0] * 13,
            )
        )
    return dict(
        schema="soccer.rsi.postcontact_full_bank_readback.v1",
        observation_contract=observation_contract(),
        training_authorized=False,
        teacher_qualified=False,
        candidate_supervision_not_admitted=rows,
    )


def test_local_retention_and_cluster_folds():
    d = measured_dataset(synthetic_readback())
    assert d.features.shape == (14, 121)
    assert d.context_indices == (*range(8), *range(12, 18))
    assert d.excluded_context_indices == (8, 9, 10, 11)
    assert np.array_equal(d.local_indices(), np.arange(8))
    assert len(d.group_folds()) == 2
    for train, test in d.group_folds():
        assert not set(train) & set(test) and set(train) | set(test) == set(range(8))
        assert not {d.source_clusters[i] for i in train} & {d.source_clusters[i] for i in test}
    assert not d.mask[1, :, 0].any() and d.mask[1, :, 1].all()


@pytest.mark.parametrize(
    "change",
    ["partition", "boolean", "false_observed", "unsafe_pass", "censored_pass", "one_cluster"],
)
def test_invalid_or_insufficient_data_rejected(change):
    rb = copy.deepcopy(synthetic_readback())
    rows = rb["candidate_supervision_not_admitted"]
    if change == "partition":
        rows[12]["partition"] = "CONSUMED_LOCAL_DEVELOPMENT"
    elif change == "boolean":
        rows[0]["observed_action_safe"][0] = 1
    elif change == "false_observed":
        rows[8]["candidate_available"][0] = True
    elif change == "unsafe_pass":
        rows[0]["observed_action_task_pass"][0] = True
        rows[0]["observed_action_safe"][0] = False
    elif change == "censored_pass":
        rows[1]["observed_action_task_pass"][0] = True
    else:
        for row in rows[:12]:
            row["source_cluster"] = 0
    with pytest.raises(ValueError):
        measured_dataset(rb)


def test_brier_measured_mask_and_empty_head():
    p = np.full((2, 13, 2), 0.5, dtype=np.float32)
    y = np.zeros_like(p, dtype=bool)
    mask = np.ones_like(y)
    assert masked_brier(p, y, mask) == [0.25, 0.25]
    mask[:, :, 0] = False
    assert masked_brier(p, y, mask) == [None, 0.25]
    p[0, 0, 1] = np.nan
    with pytest.raises(ValueError):
        masked_brier(p, y, mask)
