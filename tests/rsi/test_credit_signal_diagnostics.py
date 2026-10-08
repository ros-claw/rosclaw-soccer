import numpy as np
import pytest

from rosclaw_soccer.rsi.credit_signal_diagnostics import diagnose_credit


def data():
    ids = np.array([0, 0, 0, 1, 2, 3, 4], dtype=np.int64)
    y = np.broadcast_to(np.arange(270, dtype=np.float64), (7, 270)).copy()
    y[:3] += 2
    return dict(
        context_ids=ids,
        targets=y,
        predictions=np.zeros_like(y),
        advantages=np.zeros_like(y),
        high_quality=np.array([True, False, False, False, False, False, False]),
        safety_passed=np.ones(7, dtype=bool),
    )


def test_grain_and_time_only_baseline_are_explicit():
    result = diagnose_credit(**data())
    assert (result["episodes"], result["independent_consumed_contexts"], result["frame_rows"]) == (
        7,
        5,
        1890,
    )
    assert result["contexts"][0]["episodes"] == 3
    assert result["frame_weighted_mse"] != result["equally_weighted_context_mse"]
    assert result["frame_weighted_mse"]["excluded_time_only_mean"] < 4
    assert result["outcome_groups"][-1]["mean_advantage"] is None
    assert result["promotion_authorized"] is False


def test_baseline_never_uses_same_context_targets():
    original = data()
    first = diagnose_credit(**original)
    original["targets"][:3] += 100
    changed = diagnose_credit(**original)
    # Fold0's train contexts must remain independent of the changed test0.
    assert first["folds"][0] == changed["folds"][0]
    for fold in first["folds"]:
        assert not set(fold["train_context_ids"]) & set(fold["held_out_context_ids"])
    # Explicit analytic error for context0: its excluded time baseline is t.
    assert first["contexts"][0]["mse"]["excluded_time_only_mean"] == 4
    assert changed["contexts"][0]["mse"]["excluded_time_only_mean"] == 102**2


@pytest.mark.parametrize("field", ["targets", "predictions", "advantages"])
def test_nonfinite_credit_rejected(field):
    values = data()
    values[field][0, 0] = np.nan
    with pytest.raises(ValueError):
        diagnose_credit(**values)


def test_missing_fold_and_wrong_label_grain_rejected():
    values = data()
    values["context_ids"][-1] = 3
    with pytest.raises(ValueError):
        diagnose_credit(**values)
    values = data()
    values["safety_passed"] = np.ones((7, 270), dtype=bool)
    with pytest.raises(ValueError):
        diagnose_credit(**values)
