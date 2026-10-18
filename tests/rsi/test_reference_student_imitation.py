"""Small synthetic fits; no physical teacher, real robot or skills claim."""

import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi import reference_student_imitation as imitation
from rosclaw_soccer.rsi.reference_student_execution import own_model


def arrays():
    x = np.random.default_rng(1).normal(0, 0.1, (4, 300, 135)).astype(np.float64)
    x[:, :, -1] = (300 - np.arange(300)) / 300
    y = np.zeros((4, 300, 12), dtype=np.float64)
    y[:, 30:, 4] = 0.08
    return x, y


def batch(x=None, y=None):
    if x is None:
        x, y = arrays()
    return imitation.prepare_numeric_batch(
        x,
        y,
        partitions=["train", "train", "validation", "train"],
        source_seeds=[11, 12, 13, 14],
        eligible=[True, True, True, False],
    )


@pytest.mark.parametrize("changed", [None, "nan", "dtype", "time", "cap", "prefix", "count"])
def test_whole_numeric_validation_before_fit(changed):
    x, y = arrays()
    if changed == "nan":
        x[3, 0, 0] = np.nan  # ineligible failures still require finite complete rows
    elif changed == "dtype":
        x = x.astype(np.float32)
    elif changed == "time":
        x[2, 4, -1] = 0
    elif changed == "cap":
        y[0, 100, 0] = 0.36
    elif changed == "prefix":
        y[0, 29, 0] = 1e-15
    elif changed == "count":
        x = x[:, :299]
    if changed:
        with pytest.raises(ValueError, match="complete finite"):
            batch(x, y)
    else:
        b = batch(x, y)
        assert np.count_nonzero(b["weights"]) == 540
        assert np.all(b["weights"][2:] == 0)


@pytest.mark.parametrize("changed", ["source_split", "one_train", "no_validation"])
def test_no_source_leakage_or_single_source_supervision(changed):
    x, y = arrays()
    seeds, eligible = [11, 12, 13, 14], [True, True, True, False]
    if changed == "source_split":
        seeds[2] = 11
    elif changed == "one_train":
        eligible[1] = False
    else:
        eligible[2] = False
    with pytest.raises(ValueError, match="source"):
        imitation.prepare_numeric_batch(
            x,
            y,
            partitions=["train", "train", "validation", "train"],
            source_seeds=seeds,
            eligible=eligible,
        )


def test_validation_and_failed_rows_cannot_change_normalization():
    x, y = arrays()
    before = batch(x, y)
    x[2:, :, :-1] = 1e5
    after = batch(x, y)
    np.testing.assert_array_equal(before["mean"], after["mean"])
    np.testing.assert_array_equal(before["scale"], after["scale"])
    assert after["mean"][-1] == 0 and after["scale"][-1] == 1
    assert np.all(after["scale"] >= 0.05)


@pytest.mark.parametrize("changed", ["mask", "normalization", "missing_original"])
def test_mutable_batch_cannot_bypass_rederivation(changed):
    b = batch()
    if changed == "mask":
        b["train"][2] = True
    elif changed == "normalization":
        b["mean"][0] += 1
    else:
        b.pop("original_observations")
    with pytest.raises(ValueError, match="original|unchanged"):
        imitation.fit_numeric_reference_student(
            b,
            world_hash="sha256:" + "a" * 64,
            parent_model_hash="sha256:" + "b" * 64,
            seed=10,
            steps=1,
        )


def test_two_actual_synthetic_updates_reuse_core_and_hard_exclude_validation(monkeypatch):
    b, captured = batch(), []
    original = imitation.fit_recurrent_residual_imitation

    def spy(**kw):
        captured.append(copy.deepcopy({k: v for k, v in kw.items() if k != "parameters"}))
        return original(**kw)

    monkeypatch.setattr(imitation, "fit_recurrent_residual_imitation", spy)
    result = imitation.fit_numeric_reference_student(
        b, world_hash="sha256:" + "a" * 64, parent_model_hash="sha256:" + "b" * 64, seed=10, steps=2
    )
    assert len(captured) == 1
    np.testing.assert_array_equal(captured[0]["context"], b["normalized"][:2])
    np.testing.assert_array_equal(captured[0]["targets"], b["targets"][:2] * (0.2 / 0.35))
    assert captured[0]["config"].residual_cap == 0.2
    fit = result["learning_receipt"]
    assert fit["completed_optimizer_steps"] == 2 and fit["original_episode_count"] == 2
    assert fit["training_loss_improved"]
    assert result["train_source_seeds"] == [11, 12]
    assert result["validation_source_seeds"] == [13]
    assert result["input_episode_count"] == 4 and result["original_ineligible_episode_count"] == 1
    assert (
        not result["own_body_student_evaluated"] and not result["training_provenance_authenticated"]
    )
    assert own_model(result["numeric_model"]) == result["numeric_model"]
    assert len(result["final_same_state_diagnostics"]) == 4
