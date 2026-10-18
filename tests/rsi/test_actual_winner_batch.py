"""Synthetic arithmetic contracts, not physical or data-admission evidence."""

import numpy as np
import pytest

from rosclaw_soccer.rsi.actual_winner_batch import prepare_actual_winner_batch
from rosclaw_soccer.rsi.actual_winner_supervision import select_actual_winner_paths
from rosclaw_soccer.sim.contracts import hash_json


def bank():
    groups = (12, 13, 28, 29, 76, 77, 92, 93)

    def rows(role, passed):
        return [
            dict(
                group=g,
                role=role,
                receipt_hash=f"{role}-{g}",
                outcome=dict(task_pass=g in passed, safe=True),
            )
            for g in groups
        ]

    return dict(
        parent_model_hash="sha256:" + "a" * 64,
        zeros=rows("zero", (12, 13, 92, 93)),
        teachers=rows("teacher", (12, 13, 76, 93)),
        zero_nonfoot={g: False for g in groups},
        teacher_nonfoot={g: False for g in groups},
    )


def inputs():
    rng = np.random.default_rng(714)
    zx, tx = [rng.normal(size=(8, 300, 135)) for _ in range(2)]
    for x in (zx, tx):
        x[:, :, -1] = (300 - np.arange(300)) / 300
    zy = np.zeros((8, 300, 12))
    ty = np.full_like(zy, 0.12)
    ty[:, :30] = 0
    return dict(
        zero_x=zx,
        zero_y=zy,
        teacher_x=tx,
        teacher_y=ty,
        selection=select_actual_winner_paths(**bank()),
    )


def test_zero_anchors_receive_positive_weight_and_own_inputs_remain_exact():
    kw = inputs()
    batch, report = prepare_actual_winner_batch(**kw)
    assert (report["train_paths"], report["validation_paths"]) == (3, 2)
    assert report["train_positive_weight_frames"] == 810
    assert batch["original_observations"][::2].tobytes() == kw["zero_x"].tobytes()
    assert batch["original_observations"][1::2].tobytes() == kw["teacher_x"].tobytes()
    assert np.all(batch["weights"][[0, 2], 30:] == 1)
    assert np.all(batch["targets"][[0, 2]] == 0)
    assert np.all(batch["weights"][9, 30:] == 5)
    assert np.all(batch["weights"][[1, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15]] == 0)
    assert not report["neural_training_authorized"]


def test_validation_and_failed_inputs_do_not_change_normalization():
    kw = inputs()
    before, _ = prepare_actual_winner_batch(**kw)
    kw["zero_x"][6:, :, :-1] += 100
    kw["teacher_x"][:4, :, :-1] -= 100
    after, _ = prepare_actual_winner_batch(**kw)
    for k in ("mean", "scale", "weights", "train"):
        np.testing.assert_array_equal(before[k], after[k])


@pytest.mark.parametrize("tamper", ["nonzero_nav", "partial", "nan", "seal", "role", "query"])
def test_rejects_incomplete_or_counterfactual_data(tamper):
    kw = inputs()
    if tamper == "nonzero_nav":
        kw["zero_y"][0, 31, 0] = 0.001
    elif tamper == "partial":
        kw["teacher_x"] = kw["teacher_x"][:-1]
    elif tamper == "nan":
        kw["zero_x"][0, 0, 0] = np.nan
    elif tamper == "seal":
        kw["selection"]["rows"][0]["partition"] = "validation"
    else:
        kw["selection"]["rows"][0]["role" if tamper == "role" else "counterfactual_labels"] = (
            "teacher" if tamper == "role" else True
        )
        s = kw["selection"]
        s["report_hash"] = hash_json({k: v for k, v in s.items() if k != "report_hash"})
    with pytest.raises(ValueError):
        prepare_actual_winner_batch(**kw)
